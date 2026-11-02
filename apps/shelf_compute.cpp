// shelf-compute — エンジンの実体（JSONL ドライバ）（`engine_contract.md` §1/§2 の実体）。
//
// 役割: Server（Python 側・パス A）とエンジンの**唯一の接点**。stdin から 1 行 1
// オブジェクトの操作を読み、stdout へイベントを 1 行ずつ書く。プロトコルの型は
// `src/api/protocol.hpp` が正本で、ここは「型 → 実行」の対応だけを持つ。
//
// 設計（いまの範囲）:
//   * 1 リクエスト直列（`max_sequences = 1` に一致）。多重化は Server 側の順番待ち
//   * `generate` は prefill（チャンク）→ decode（1 トークンずつ）→ `token` イベント →
//     `done`。抽選は `runtime/generate`（`api` 経由で再輸出）に従う
//   * `cancel` は **decode の各ステップの間**で受け付ける（stdin を非ブロッキングで覗く）。
//     生成前に届いた cancel は `done(cancelled, 0 トークン)`
//   * エラーは `error` イベント（`Status::Code` → `code` の写像は `api` が持つ）。
//     1 リクエストの失敗でプロセスは落とさない
//   * `--model` 省略時はモデルを読まない（`ping` だけ応答する。`generate` は
//     `model_not_found`）。プロトコルだけの検証に使える
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <poll.h>
#include <unistd.h>

#include <shelf/api/protocol.hpp>
#include <shelf/api/sampling.hpp>
#include <shelf/backend/common/capabilities.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/model/qwen35/step_graph.hpp>
#include <shelf/runtime/generate/batch_scheduler.hpp>
#include <shelf/runtime/generate/stop_tokens.hpp>
#include <shelf/runtime/state/prefix_cache.hpp>
#include <shelf/runtime/state/prefix_key.hpp>

#include "common/app_support.hpp"
#include "common/boot.hpp"
#include "common/pipeline.hpp"

namespace {

using namespace shelf;

constexpr const char* kVersion = "shelf-compute/0.1";

// 1 行読む（stdin）。EOF は nullopt。改行は落とす。上限超過は空行扱いで読み捨て。
bool read_line(std::string* out) {
    out->clear();
    for (;;) {
        char c = 0;
        const ssize_t n = ::read(STDIN_FILENO, &c, 1);
        if (n <= 0) return false;
        if (c == '\n') return true;
        out->push_back(c);
        if (out->size() > (4u << 20)) {   // 壊れた入力で無限に伸ばさない
            char sink = 0;
            while (::read(STDIN_FILENO, &sink, 1) == 1 && sink != '\n') {
            }
            out->clear();
            return true;
        }
    }
}

void write_event(const api::Event& e) {
    const std::string line = api::encode_event(e) + "\n";
    std::size_t off = 0;
    while (off < line.size()) {
        const ssize_t n = ::write(STDOUT_FILENO, line.data() + off, line.size() - off);
        if (n <= 0) return;
        off += static_cast<std::size_t>(n);
    }
}

// 未読があるか（ブロックしない）。cancel の検出に使う。
bool stdin_pending() {
    pollfd p{STDIN_FILENO, POLLIN, 0};
    return ::poll(&p, 1, 0) > 0;
}

struct Server {
    std::string model_dir;
    int gpu = 0;
    std::size_t pp = 1;
    std::size_t prefill_rows = 64;
    std::size_t max_context = 0;
    bool has_model = false;

    model::qwen35::TextConfig cfg;
    app::Fp8ModelFormat fmt;
    app::Pipeline pipeline;
    bool stdin_eof = false;              // 入力が閉じた（生成は最後まで続ける）
    bool last_prefill_step_ = false;     // N06: prefill/decode の交互実行
    bool interleave = true;              // N06: 交互実行（--no-interleave で無効）
    std::uint64_t request_counter = 0;
    runtime::generate::StopTokens stops;          // 解決済みの停止 ID（T01）
    std::vector<std::int32_t> eos_override;       // --eos-token（config より優先）
    bool has_eos_override = false;
    bool ignore_eos = false;                      // --no-eos（max_new_tokens だけで止める）
    // T14: 重みの置き場（pinned host）と VRAM 予算
    long long offload_mib = 0;
    long long vram_budget_mib = 0;
    bool allow_host_weights = false;
    bool kv_in_host = false;       // T14: KV プールを pinned host に置く
    bool allow_host_kv = false;    // 予算不足時に KV を host へ
    std::size_t online_kv_pages = 0;   // T14: 直近 N ページを VRAM にミラー
    std::size_t max_batch = 4;     // T15: 連続バッチングの最大スロット数
    // N01: この行数以下の decode は行ごとに decode1 で回す（0 = 常にバッチ経路）。
    std::uint32_t decode1_max_rows = 4;
    // T16: prefix cache（0 = 無効）
    long long prefix_cache_blocks = 0;
    std::size_t prefix_snapshot_tokens = 64;
    std::size_t prefix_snapshot_limit = 8;

    // ---- T15: 連続バッチングの状態 ----
    std::unique_ptr<runtime::generate::BatchScheduler> sched;
    std::unordered_map<std::string, std::uint64_t> seed_of;
    std::vector<std::uint32_t> row_slots;
    std::vector<std::uint32_t> row_positions;
    std::vector<std::int32_t> row_tokens;
    std::vector<std::int32_t> chunk_tokens;

    // ---- T16: Prefix Cache ----
    std::unique_ptr<runtime::state::PrefixCache> prefix;
    std::size_t prefix_bs = 16;              // KV ブロックのトークン数
    std::size_t kv_block_count = 0;          // 層あたりのグローバル block 数
    std::uint64_t prefix_clock = 0;
    struct PrefixSnapshot {
        std::size_t tokens = 0;
        std::uint64_t tick = 0;
        std::vector<std::vector<float>> stages;   // PP ステージごとの GDN 状態
    };
    std::map<runtime::state::BlockKey, PrefixSnapshot> snapshots;
    struct SlotPrefix {
        bool prepared = false;
        std::size_t matched = 0;              // 共有した prefix トークン数
        std::size_t pages_assigned = 0;
        std::vector<std::uint32_t> blocks;    // 論理 page 順（共有 + 新規）
        std::vector<char> registered;         // prefix cache に登録済みか
    };
    std::unordered_map<std::string, SlotPrefix> slot_prefix;

    api::PongEvent capabilities() const {
        api::PongEvent p;
        p.backend = pipeline.empty() ? "none" : "hip";
        p.model = has_model ? model_dir : std::string("(none)");
        p.max_context = max_context;
        const auto fs = backend::build_features(pipeline.empty() ? false : true);
        for (std::size_t i = 0; i < static_cast<std::size_t>(backend::Feature::count); ++i)
            if (fs.has(static_cast<backend::Feature>(i)))
                p.features.push_back(backend::feature_name(static_cast<backend::Feature>(i)));
        return p;
    }

    // ---- T15: 連続バッチングのエンジンループ ---------------------------------
    bool idle() const { return sched == nullptr || (sched->active() == 0 && sched->queued() == 0); }

    void emit_finished() {
        if (!sched) return;
        runtime::generate::Finished f;
        while (sched->take_finished(&f)) {
            release_prefix(f.request_id);
            api::DoneEvent d;
            d.request_id = f.request_id;
            d.generated_ids = f.generated;
            d.prompt_tokens = f.prompt_tokens;
            d.generated_tokens = f.generated.size();
            const auto it = seed_of.find(f.request_id);
            d.seed = it == seed_of.end() ? 0u : it->second;
            switch (f.end) {
            case runtime::generate::SlotEnd::stop:
                d.finish_reason = api::FinishReason::stop;
                break;
            case runtime::generate::SlotEnd::length:
                d.finish_reason = api::FinishReason::length;
                break;
            case runtime::generate::SlotEnd::cancelled:
                d.finish_reason = api::FinishReason::cancelled;
                break;
            default:
                d.finish_reason = api::FinishReason::error;
                break;
            }
            write_event(api::Event{d});
            seed_of.erase(f.request_id);
        }
    }

    void submit_generate(const api::GenerateRequest& g) {
        if (!has_model) {
            api::EngineError e;
            e.request_id = g.request_id;
            e.code = "model_not_found";
            e.message = "--model が指定されていない（このプロセスは ping のみ応答する）";
            write_event(api::Event{e});
            return;
        }
        if (g.input_ids.empty()) {
            api::EngineError e;
            e.request_id = g.request_id;
            e.code = "invalid_argument";
            e.message = "input_ids が空";
            e.param = "input_ids";
            write_event(api::Event{e});
            return;
        }
        if (max_context != 0 && g.input_ids.size() + g.max_new_tokens > max_context) {
            const Status st = Status::out_of_range(
                "context: prompt " + std::to_string(g.input_ids.size()) + " + max_new " +
                    std::to_string(g.max_new_tokens) + " > capacity " +
                    std::to_string(max_context),
                __FILE__, __LINE__);
            write_event(api::Event{api::error_from_status(
                st, api::InvalidStateKind::state_error, g.request_id)});
            return;
        }
        std::string param;
        if (const Status st = api::validate_sampling(g.sampling, &param); !st.ok()) {
            write_event(api::Event{api::error_from_status(
                st, api::InvalidStateKind::state_error, g.request_id, param)});
            return;
        }
        runtime::generate::BatchRequest br;
        br.request_id = g.request_id;
        br.prompt_ids = g.input_ids;
        br.max_new_tokens = g.max_new_tokens;
        br.stop = stops;
        br.sampling = g.sampling;
        br.sampling.seed = api::assign_seed(g.sampling.seed, ++request_counter);
        seed_of[br.request_id] = *br.sampling.seed;
        (void)sched->submit(std::move(br));
    }

    bool process_line(const std::string& line) {
        if (line.empty()) return true;
        const auto req = api::parse_request(line);
        if (!req.ok()) {
            write_event(api::Event{api::error_from_status(req.status())});
            return true;
        }
        if (std::holds_alternative<api::PingRequest>(req.value())) {
            write_event(api::Event{capabilities()});
            return true;
        }
        if (const auto* c = std::get_if<api::CancelRequest>(&req.value())) {
            // request_id を照合して、そのスロット（または queued）だけを止める。
            (void)sched->cancel(c->request_id);
            emit_finished();
            return true;
        }
        submit_generate(std::get<api::GenerateRequest>(req.value()));
        emit_finished();   // 受付拒否（error イベント）だけの場合は何も出ない
        return true;
    }

    // 生成中に届いた制御行を読み切る（ブロックしない。1 回の上限つき）。
    void drain_input() {
        int budget = 64;
        while (budget-- > 0 && stdin_pending()) {
            std::string line;
            if (!read_line(&line)) {   // EOF: 実行中の要求は最後まで続ける
                stdin_eof = true;
                break;
            }
            process_line(line);
        }
    }

    void fail_request(std::size_t slot, const Status& st) {
        const std::string rid = sched->request(slot).request_id;
        write_event(api::Event{
            api::error_from_status(st, api::InvalidStateKind::state_error, rid)});
        sched->fail_slot(slot);
        emit_finished();
    }

    // 1 行分の logits から 1 トークンを抽選し、token イベント / 終了へつなぐ。
    void sample_row(std::size_t slot, const float* logits, std::uint32_t position,
                    bool from_prefill = false) {
        const auto& req = sched->request(slot);
        const bool dbg = std::getenv("SHELF_DEBUG_BATCH") != nullptr;
        const std::size_t vocab = cfg.vocab_size;
        auto draw = api::sample_range(logits, vocab, req.sampling, position);
        if (!draw.ok()) {
            write_event(api::Event{api::EngineError{req.request_id, "state_error",
                                                    "sampling: " + draw.status().message(), ""}});
            sched->fail_slot(slot);
            emit_finished();
            return;
        }
        const std::int32_t tok = static_cast<std::int32_t>(draw.value().index);
        if (dbg)
            std::fprintf(stderr, "[batch] sample slot=%zu rid=%s pos=%u tok=%d\n", slot,
                         req.request_id.c_str(), position, tok);
        const bool stop_hit = req.stop.contains(tok);
        const auto end = sched->decode_result(slot, tok, stop_hit, from_prefill);
        if (end == runtime::generate::SlotEnd::error) {
            write_event(api::Event{api::EngineError{req.request_id, "state_error",
                                                    "decode: スロット状態が不正", ""}});
            emit_finished();
            return;
        }
        // EOS は本文に載せない（`engine_contract.md` §2）。
        if (!stop_hit) write_event(api::Event{api::TokenEvent{req.request_id, tok}});
        if (end != runtime::generate::SlotEnd::none) emit_finished();
    }

    // ---- T16: Prefix Cache の実処理 ----------------------------------------
    // 論理 page にグローバル block を 1 つ割り当てる（必要なら）。
    Status ensure_page(std::size_t slot, SlotPrefix& sp, std::size_t page) {
        if (std::getenv("SHELF_DEBUG_PAGES") != nullptr)
            std::fprintf(stderr, "[pages] slot=%zu want=%zu assigned=%zu\n", slot, page,
                         sp.pages_assigned);
        while (sp.pages_assigned <= page) {
            auto b = prefix->allocate();
            if (!b.ok()) return b.status();
            const std::uint32_t blk = b.value();
            if (auto st = app::pipeline_assign_blocks(pipeline, slot, &blk, 1,
                                                      sp.pages_assigned);
                !st.ok()) {
                prefix->release(blk);
                return st;
            }
            if (std::getenv("SHELF_DEBUG_PAGES") != nullptr)
                std::fprintf(stderr, "[pages] slot=%zu page=%zu -> block %u\n", slot,
                             sp.pages_assigned, blk);
            sp.blocks.push_back(blk);
            sp.registered.push_back(0);
            ++sp.pages_assigned;
        }
        return Status::make_ok();
    }

    // 最初の prefill の前に、スロットを初期化し prefix の block 表 / GDN 状態を入れる。
    Status prepare_prefix(std::size_t slot, const std::string& rid) {
        if (!prefix) return Status::make_ok();
        auto& sp = slot_prefix[rid];
        if (sp.prepared) return Status::make_ok();
        const auto info = sched->slot(slot);
        if (info.state != runtime::generate::SlotState::prefill)
            return Status::make_ok();
        sp.prepared = true;
        const auto& prompt = sched->request(slot).prompt_ids;
        if (auto st = app::pipeline_reset_slot(pipeline, slot); !st.ok()) return st;
        if (prompt.size() < prefix_bs) return Status::make_ok();

        const std::size_t B = prefix_bs;
        auto acq = prefix->acquire(prompt);
        if (!acq.ok()) return acq.status();
        const auto hit = acq.value();
        std::vector<runtime::state::BlockKey> keys;
        runtime::state::BlockKey parent = runtime::state::kRootBlockKey;
        for (std::size_t b = 0; b < hit.blocks.size(); ++b) {
            parent = runtime::state::chain_block_key(parent, prompt.data() + b * B, B);
            keys.push_back(parent);
        }
        std::size_t chosen = 0;
        for (std::size_t k = hit.blocks.size(); k >= 1; --k) {
            const std::size_t L = k * B;
            if (L >= prompt.size()) continue;
            const auto it = snapshots.find(keys[k - 1]);
            if (it == snapshots.end()) continue;
            chosen = k;
            it->second.tick = ++prefix_clock;
            if (auto st = app::pipeline_import_gdn(pipeline, slot, it->second.stages);
                !st.ok())
                return st;
            if (std::getenv("SHELF_DEBUG_PREFIX") != nullptr)
                std::fprintf(stderr, "[prefix] import tokens=%zu hash=%016llx\n",
                             it->second.tokens,
                             static_cast<unsigned long long>(state_hash(it->second.stages)));
            break;
        }
        if (chosen == 0) {
            for (const auto blk : hit.blocks) prefix->release(blk);
            return Status::make_ok();
        }
        sp.blocks.assign(hit.blocks.begin(),
                         hit.blocks.begin() + static_cast<std::ptrdiff_t>(chosen));
        for (std::size_t i = chosen; i < hit.blocks.size(); ++i)
            prefix->release(hit.blocks[i]);
        sp.pages_assigned = chosen;
        sp.registered.assign(chosen, 0);
        sp.matched = chosen * B;
        debug_kv("hit", slot, chosen);
        if (auto st = app::pipeline_assign_blocks(pipeline, slot, sp.blocks.data(), chosen, 0);
            !st.ok())
            return st;
        if (!sched->prime_prefill(slot, sp.matched))
            return Status::invalid_state("prefix: prime_prefill に失敗", __FILE__, __LINE__);
        std::fprintf(stderr, "[prefix] hit rid=%s tokens=%zu prompt=%zu\n", rid.c_str(),
                     sp.matched, prompt.size());
        return Status::make_ok();
    }

    void debug_kv(const char* tag, std::size_t slot, std::size_t pages) const {
        if (std::getenv("SHELF_DEBUG_PREFIX") == nullptr) return;
        std::uint64_t h = 1469598103934665603ull;
        for (auto& st : pipeline.stages) {
            const auto r = st.exec->debug_kv_hash(slot, pages);
            if (!r.ok()) return;
            h = (h ^ r.value()) * 1099511628211ull;
        }
        std::fprintf(stderr, "[prefix] kv %s slot=%zu pages=%zu hash=%016llx\n", tag, slot,
                     pages, static_cast<unsigned long long>(h));
    }

    static std::uint64_t state_hash(const std::vector<std::vector<float>>& parts) {
        std::uint64_t h = 1469598103934665603ull;
        for (const auto& part : parts)
            for (const float v : part) {
                std::uint32_t b = 0;
                std::memcpy(&b, &v, sizeof(b));
                h = (h ^ b) * 1099511628211ull;
            }
        return h;
    }

    // prefill が snapshot_tokens の倍数に達したら GDN 状態を退避する。
    void maybe_store_snapshot(std::size_t slot) {
        if (!prefix || prefix_snapshot_tokens == 0) return;
        const auto info = sched->slot(slot);
        const auto& prompt = sched->request(slot).prompt_ids;
        const std::size_t L = info.prefilled;
        const std::size_t B = prefix_bs;
        if (L == 0 || L >= prompt.size()) return;
        if (L % prefix_snapshot_tokens != 0 || L % B != 0) return;
        runtime::state::BlockKey parent = runtime::state::kRootBlockKey;
        for (std::size_t b = 0; b < L / B; ++b)
            parent = runtime::state::chain_block_key(parent, prompt.data() + b * B, B);
        auto exp = app::pipeline_export_gdn(pipeline, slot);
        if (!exp.ok()) return;   // 退避失敗は致命にしない
        PrefixSnapshot snap;
        snap.tokens = L;
        snap.tick = ++prefix_clock;
        snap.stages = std::move(exp.value());
        if (std::getenv("SHELF_DEBUG_PREFIX") != nullptr)
            std::fprintf(stderr, "[prefix] store tokens=%zu hash=%016llx\n", snap.tokens,
                         static_cast<unsigned long long>(state_hash(snap.stages)));
        snapshots[parent] = std::move(snap);
        while (snapshots.size() > prefix_snapshot_limit && !snapshots.empty()) {
            auto victim = snapshots.begin();
            for (auto it = snapshots.begin(); it != snapshots.end(); ++it)
                if (it->second.tick < victim->second.tick) victim = it;
            snapshots.erase(victim);
        }
        std::fprintf(stderr, "[prefix] snapshot tokens=%zu\n", L);
    }

    // prefill 完了時に、計算済みの full block を prefix cache へ登録する。
    Status register_prefix_blocks(std::size_t slot, const std::string& rid) {
        if (!prefix) return Status::make_ok();
        const auto it = slot_prefix.find(rid);
        if (it == slot_prefix.end()) return Status::make_ok();
        auto& sp = it->second;
        const auto& prompt = sched->request(slot).prompt_ids;
        const std::size_t B = prefix_bs;
        const std::size_t full = prompt.size() / B;
        for (std::size_t p = sp.matched / B; p < full; ++p) {
            if (p >= sp.blocks.size() || p >= sp.registered.size()) break;
            if (sp.registered[p]) continue;
            if (auto st = prefix->insert(sp.blocks[p], prompt, p); !st.ok()) return st;
            sp.registered[p] = 1;
        }
        if (full != 0)
            std::fprintf(stderr, "[prefix] register rid=%s blocks=%zu\n", rid.c_str(), full);
        return Status::make_ok();
    }

    // 要求終了時に block の参照を返す。
    void release_prefix(const std::string& rid) {
        if (!prefix) return;
        const auto it = slot_prefix.find(rid);
        if (it == slot_prefix.end()) return;
        for (const auto blk : it->second.blocks) prefix->release(blk);
        slot_prefix.erase(it);
    }

    // これから書く位置まで page を確保する。
    Status ensure_pages(std::size_t slot, const std::string& rid, std::uint32_t max_pos) {
        if (!prefix) return Status::make_ok();
        const auto it = slot_prefix.find(rid);
        if (it == slot_prefix.end()) return Status::make_ok();
        return ensure_page(slot, it->second, max_pos / prefix_bs);
    }

    void run_prefill(const runtime::generate::PrefillAction& pa) {
        const std::size_t slot = pa.slot;
        // 新しい要求がスロットに入るとき（先頭チャンク）は状態と scratch を 0 に戻す。
        // prefix cache 有効時は prepare_prefix が初期化済み。
        if (pa.begin == 0 && !prefix) {
            if (auto rst = app::pipeline_reset_slot(pipeline, slot); !rst.ok()) {
                fail_request(slot, rst);
                return;
            }
        }
        const std::size_t vocab = cfg.vocab_size;
        const std::size_t chunk = std::max<std::size_t>(prefill_rows, 1);
        const std::string rid = sched->request(slot).request_id;
        if (std::getenv("SHELF_DEBUG_PREFIX") != nullptr)
            debug_kv("before", slot, pa.base_pos / std::max<std::size_t>(prefix_bs, 1u));
        if (std::getenv("SHELF_DEBUG_PREFIX") != nullptr)
            std::fprintf(stderr,
                         "[prefix] prefill rid=%s slot=%zu begin=%zu rows=%zu base=%u\n",
                         rid.c_str(), slot, pa.begin, pa.rows, pa.base_pos);
        std::vector<float> lg;
        Status st = Status::make_ok();
        if (prefix) {
            const std::uint32_t last_pos =
                pa.base_pos + static_cast<std::uint32_t>(pa.rows ? pa.rows - 1 : 0);
            if (auto es = ensure_pages(slot, rid, last_pos); !es.ok()) {
                fail_request(slot, es);
                return;
            }
        }
        if (pa.rows == chunk) {
            chunk_tokens.assign(pa.tokens, pa.tokens + pa.rows);
            row_slots.assign(pa.rows, static_cast<std::uint32_t>(slot));
            row_positions.resize(pa.rows);
            for (std::size_t i = 0; i < pa.rows; ++i)
                row_positions[i] = pa.base_pos + static_cast<std::uint32_t>(i);
            backend::StageExecutor::RowBatch rows{row_slots.data(), row_positions.data()};
            st = app::pipeline_step_batch_prefill(pipeline, chunk_tokens.data(), pa.rows, rows,
                                                  &lg);
        } else {
            // 端数は 1 行ずつ decode グラフで積む（直列エンジンと同じ手順）。
            for (std::size_t i = 0; i < pa.rows; ++i) {
                row_slots.assign(1, static_cast<std::uint32_t>(slot));
                row_positions.assign(1, pa.base_pos + static_cast<std::uint32_t>(i));
                backend::StageExecutor::RowBatch rows{row_slots.data(), row_positions.data()};
                st = app::pipeline_step_batch_decode(pipeline, pa.tokens + i, 1, rows, &lg);
                if (!st.ok()) break;
            }
        }
        if (!st.ok()) {
            fail_request(slot, st);
            return;
        }
        sched->prefill_progress(slot, pa.rows);
        if (std::getenv("SHELF_DEBUG_PREFIX") != nullptr) {
            auto hx = app::pipeline_export_gdn(pipeline, slot);
            if (hx.ok()) {
                std::uint64_t h = 1469598103934665603ull;
                for (const auto& part : hx.value())
                    for (const float v : part) {
                        std::uint32_t b = 0;
                        std::memcpy(&b, &v, sizeof(b));
                        h = (h ^ b) * 1099511628211ull;
                    }
                std::fprintf(stderr, "[prefix] state rid=%s prefilled=%zu hash=%016llx\n",
                             rid.c_str(), sched->slot(slot).prefilled,
                             static_cast<unsigned long long>(h));
                std::uint64_t hc = 0, hr = 0;
                for (auto& st : pipeline.stages) {
                    const auto sh = st.exec->debug_state_hashes(slot);
                    if (!sh.ok()) break;
                    hc = (hc ^ sh.value().first) * 1099511628211ull;
                    hr = (hr ^ sh.value().second) * 1099511628211ull;
                }
                if (const char* dump = std::getenv("SHELF_DEBUG_PREFIX_DUMP")) {
                    std::string path = std::string(dump) + ".slot" + std::to_string(slot) +
                                       "." + std::to_string(sched->slot(slot).prefilled);
                    if (FILE* f = std::fopen(path.c_str(), "wb")) {
                        for (const auto& part : hx.value())
                            std::fwrite(part.data(), sizeof(float), part.size(), f);
                        std::fclose(f);
                    }
                }
                std::fprintf(stderr, "[prefix] kind conv=%016llx rec=%016llx\n",
                             static_cast<unsigned long long>(hc),
                             static_cast<unsigned long long>(hr));
                debug_kv("after", slot, sched->slot(slot).prefilled / std::max<std::size_t>(
                                                                    prefix_bs, 1u));
            }
        }
        maybe_store_snapshot(slot);
        if (sched->slot(slot).state == runtime::generate::SlotState::decode) {
            if (auto rs = register_prefix_blocks(slot, rid); !rs.ok()) {
                fail_request(slot, rs);
                return;
            }
        }
        if (std::getenv("SHELF_DEBUG_BATCH") != nullptr)
            std::fprintf(stderr, "[batch] prefill slot=%zu rows=%zu lg=%zu state=%d\n", slot,
                         pa.rows, lg.size(), static_cast<int>(sched->slot(slot).state));
        if (sched->slot(slot).state != runtime::generate::SlotState::decode) return;
        if (vocab == 0 || lg.size() < vocab || lg.size() % vocab != 0) {
            fail_request(slot, Status::invalid_state("prefill: logits の長さが語彙数と不一致",
                                                     __FILE__, __LINE__));
            return;
        }
        // 最後の行の logits で最初の生成トークンを抽選する。
        if (std::getenv("SHELF_DEBUG_BATCH_LOGITS") != nullptr && !lg.empty()) {
            const float* row = lg.data() + (lg.size() - vocab);
            std::uint32_t t1 = 0, t2 = 0;
            float v1 = -3.4e38f, v2 = -3.4e38f;
            for (std::uint32_t v = 0; v < vocab; ++v) {
                if (row[v] > v1) { t2 = t1; v2 = v1; t1 = v; v1 = row[v]; }
                else if (row[v] > v2) { t2 = v; v2 = row[v]; }
            }
            std::uint64_t h = 1469598103934665603ull;
            for (std::uint32_t v = 0; v < vocab; ++v) {
                std::uint32_t b = 0;
                std::memcpy(&b, &row[v], sizeof(b));
                h = (h ^ b) * 1099511628211ull;
            }
            std::fprintf(stderr,
                         "[batch-logits] prefill slot=%zu pos=%u top1=%u(%.6f) top2=%u(%.6f) "
                         "margin=%.3e hash=%016llx\n",
                         slot, sched->slot(slot).position, t1, v1, t2, v2,
                         static_cast<double>(v1) - static_cast<double>(v2),
                         static_cast<unsigned long long>(h));
        }
        sample_row(slot, lg.data() + (lg.size() - vocab), sched->slot(slot).position,
                   /*from_prefill=*/true);
    }

    void run_decode(const runtime::generate::DecodeAction& da) {
        const std::size_t rows = da.rows.size();
        const std::size_t vocab = cfg.vocab_size;
        row_slots.resize(rows);
        row_positions.resize(rows);
        row_tokens.resize(rows);
        for (std::size_t i = 0; i < rows; ++i) {
            row_slots[i] = static_cast<std::uint32_t>(da.rows[i].slot);
            row_positions[i] = da.rows[i].position;
            row_tokens[i] = da.rows[i].token;
        }
        if (prefix) {
            for (const auto& row : da.rows) {
                const std::string rid = sched->request(row.slot).request_id;
                if (auto es = ensure_pages(row.slot, rid, row.position); !es.ok()) {
                    fail_request(row.slot, es);
                    return;
                }
            }
        }
        if (std::getenv("SHELF_DEBUG_DECODE_SLOTS") != nullptr) {
            for (const auto& row : da.rows) {
                std::uint64_t hc = 0, hr = 0, hk = 0;
                for (auto& st : pipeline.stages) {
                    if (const auto sh = st.exec->debug_state_hashes(row.slot); sh.ok()) {
                        hc = (hc ^ sh.value().first) * 1099511628211ull;
                        hr = (hr ^ sh.value().second) * 1099511628211ull;
                    }
                    // 書込み済みのページだけを比べる（未書込みページは次の step で
                    // 書くので、この時点では中身が不定）。
                    const std::size_t pages = std::min<std::size_t>(
                        row.position / std::max<std::size_t>(prefix_bs, 1u) + 1u,
                        st.exec->kv_pages_per_sequence());
                    if (const auto kv = st.exec->debug_kv_hash(
                            row.slot, pages, static_cast<std::size_t>(row.position) + 1u);
                        kv.ok())
                        hk = (hk ^ kv.value()) * 1099511628211ull;
                }
                if (auto tb = pipeline.stages[0].exec->debug_block_table(row.slot, 12); tb.ok()) {
                    std::fprintf(stderr, "[decode-slots] table slot=%zu:", row.slot);
                    for (const auto b : tb.value()) std::fprintf(stderr, " %u", b);
                    std::fprintf(stderr, "\n");
                } else {
                    std::fprintf(stderr, "[decode-slots] table slot=%zu: %s\n", row.slot,
                                 tb.status().message().c_str());
                }
                std::fprintf(stderr, "[decode-slots] rid=%s slot=%zu pos=%u tok=%d\n",
                             sched->request(row.slot).request_id.c_str(), row.slot,
                             row.position, row.token);
                std::fprintf(stderr,
                             "[decode-slots] rid=%s slot=%zu pos=%u conv=%016llx rec=%016llx "
                             "kv=%016llx\n",
                             sched->request(row.slot).request_id.c_str(), row.slot,
                             row.position, static_cast<unsigned long long>(hc),
                             static_cast<unsigned long long>(hr),
                             static_cast<unsigned long long>(hk));
            }
        }
        backend::StageExecutor::RowBatch rb{row_slots.data(), row_positions.data()};
        std::vector<float> lg;
        const auto st = app::pipeline_step_batch_decode(pipeline, row_tokens.data(), rows, rb,
                                                        &lg);
        if (!st.ok()) {
            for (const auto& row : da.rows) {
                const std::string rid = sched->request(row.slot).request_id;
                write_event(api::Event{
                    api::error_from_status(st, api::InvalidStateKind::state_error, rid)});
                sched->fail_slot(row.slot);
            }
            emit_finished();
            return;
        }
        if (vocab == 0 || lg.size() != rows * vocab) {
            for (const auto& row : da.rows) {
                const std::string rid = sched->request(row.slot).request_id;
                write_event(api::Event{api::error_from_status(
                    Status::invalid_state("decode: logits の長さが語彙数と不一致", __FILE__,
                                          __LINE__),
                    api::InvalidStateKind::state_error, rid)});
                sched->fail_slot(row.slot);
            }
            emit_finished();
            return;
        }
        if (std::getenv("SHELF_DEBUG_BATCH_LOGITS") != nullptr) {
            for (std::size_t i = 0; i < rows; ++i) {
                const float* row = lg.data() + i * vocab;
                std::uint32_t t1 = 0, t2 = 0;
                float v1 = -3.4e38f, v2 = -3.4e38f;
                for (std::uint32_t v = 0; v < vocab; ++v) {
                    if (row[v] > v1) {
                        t2 = t1; v2 = v1; t1 = v; v1 = row[v];
                    } else if (row[v] > v2) {
                        t2 = v; v2 = row[v];
                    }
                }
                std::uint64_t h = 1469598103934665603ull;
                for (std::uint32_t v = 0; v < vocab; ++v) {
                    std::uint32_t b = 0;
                    std::memcpy(&b, &row[v], sizeof(b));
                    h = (h ^ b) * 1099511628211ull;
                }
                std::fprintf(stderr,
                             "[batch-logits] slot=%zu pos=%u top1=%u(%.6f) top2=%u(%.6f) "
                             "margin=%.3e hash=%016llx\n",
                             da.rows[i].slot, da.rows[i].position, t1, v1, t2, v2,
                             static_cast<double>(v1) - static_cast<double>(v2),
                             static_cast<unsigned long long>(h));
            }
        }
        for (std::size_t i = 0; i < rows; ++i)
            sample_row(da.rows[i].slot, lg.data() + i * vocab, da.rows[i].position);
    }

    // 次の 1 ステップを実行する。実行できなければ false。
    bool step_once() {
        if (!sched) return false;
        if (prefix) {
            for (std::size_t i = 0; i < sched->max_slots(); ++i) {
                const auto info = sched->slot(i);
                if (info.state != runtime::generate::SlotState::prefill || info.prefilled != 0)
                    continue;
                const std::string rid(info.request_id);
                if (auto st = prepare_prefix(i, rid); !st.ok()) fail_request(i, st);
            }
        }
        // N06: prefill と decode が両方あるときは **交互** に流す。従来は
        // prefill を優先していたため、長い prefill が終わるまで既存の decode が
        // 止まり、トークン間隔が数百 ms 開いた（実測: 1024 tok の prefill で
        // 622 ms）。1 チャンク進めたら 1 回 decode を挟むことで、待ちをチャンク
        // 1 個分に抑える。片方しか無いときはそのまま進める。
        runtime::generate::PrefillAction pa;
        const bool prefill_ready = sched->next_prefill(&pa);
        runtime::generate::DecodeAction da;
        const bool decode_ready = sched->next_decode(&da);
        if (interleave && prefill_ready && decode_ready) {
            if (last_prefill_step_) {
                run_decode(da);
                last_prefill_step_ = false;
            } else {
                run_prefill(pa);
                last_prefill_step_ = true;
            }
            return true;
        }
        if (prefill_ready) {
            run_prefill(pa);
            last_prefill_step_ = true;
            return true;
        }
        if (decode_ready) {
            run_decode(da);
            last_prefill_step_ = false;
            return true;
        }
        return false;
    }
};

int usage() {
    std::fprintf(stderr,
                 "usage: shelf-compute [--model DIR] [--gpu N] [--pipeline-parallel K]\n"
                 "                   [--prefill-rows N] [--max-context N]\n"
                 "                   [--eos-token ID]... [--no-eos] [--no-interleave]\n"
                 "                   [--offload-weights-mib N] [--vram-budget-mib N]\n"
                 "                   [--allow-host-weights] [--kv-in-host] [--allow-host-kv]\n"
                 "                   [--online-kv-pages N]\n"
                 "                   [--max-batch N]  （連続バッチングのスロット数。既定 4）\n"
                 "                   [--decode1-max-rows N]  （この行数以下は行ごとに decode1。既定 4）\n"
                 "                   [--prefix-cache-blocks N] [--prefix-snapshot-tokens N]\n"
                 "                   [--prefix-snapshot-limit N]\n"
                 "  stdin/stdout: 1 行 1 オブジェクトの JSONL（engine_contract §1/§2）\n"
                 "  --model を省くと ping だけ応答する（プロトコル検証用）\n"
                 "  停止 ID は generation_config.json の eos_token_id を主とし、\n"
                 "  無ければ config.json の text_config.eos_token_id を使う（--eos-token で上書き）\n"
                 "  複数の generate を同じプロセスで続けて処理できる（reset はストリーム順）\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    Server srv;
    std::string device = "hip";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        // 参照実装（phaseshift-compute）と同じ綴りも受ける。
        if (a == "--model" || a == "--model-dir") { srv.model_dir = next("dir"); srv.has_model = true; }
        else if (a == "--gpu") srv.gpu = std::atoi(next("n").c_str());
        else if (a == "--pipeline-parallel") srv.pp = static_cast<std::size_t>(std::atoi(next("K").c_str()));
        else if (a == "--prefill-rows") srv.prefill_rows = static_cast<std::size_t>(std::atoi(next("N").c_str()));
        else if (a == "--max-context") srv.max_context = static_cast<std::size_t>(std::atoll(next("N").c_str()));
        else if (a == "--device") device = next("cpu|hip");
        else if (a == "--max-batch") {
            const long v = std::atol(next("N").c_str());
            if (v < 1 || v > 64) {
                std::fprintf(stderr, "--max-batch は 1..64\n");
                return usage();
            }
            srv.max_batch = static_cast<std::size_t>(v);
        }
        else if (a == "--prefix-cache-blocks") {
            srv.prefix_cache_blocks = std::atoll(next("N").c_str());
            if (srv.prefix_cache_blocks < 0) {
                std::fprintf(stderr, "--prefix-cache-blocks は 0 以上\n");
                return usage();
            }
        } else if (a == "--prefix-snapshot-tokens") {
            srv.prefix_snapshot_tokens = static_cast<std::size_t>(std::atoll(next("N").c_str()));
        } else if (a == "--prefix-snapshot-limit") {
            srv.prefix_snapshot_limit = static_cast<std::size_t>(std::atoll(next("N").c_str()));
        }
        // 停止 ID の上書き（既定は config.json / generation_config.json から解決）。
        // 同じ指定を繰り返すと複数 EOS になる。
        else if (a == "--eos-token") {
            srv.eos_override.push_back(
                static_cast<std::int32_t>(std::atoll(next("ID").c_str())));
            srv.has_eos_override = true;
        } else if (a == "--no-interleave") {
            srv.interleave = false;
        } else if (a == "--no-eos") {
            srv.ignore_eos = true;
        } else if (a == "--offload-weights-mib") {
            srv.offload_mib = std::atoll(next("MiB").c_str());
        } else if (a == "--vram-budget-mib") {
            srv.vram_budget_mib = std::atoll(next("MiB").c_str());
        } else if (a == "--allow-host-weights") {
            srv.allow_host_weights = true;
        } else if (a == "--kv-in-host") {
            srv.kv_in_host = true;
        } else if (a == "--allow-host-kv") {
            srv.allow_host_kv = true;
        } else if (a == "--online-kv-pages") {
            srv.online_kv_pages =
                static_cast<std::size_t>(std::strtoull(next("N").c_str(), nullptr, 10));
        } else if (a == "--decode1-max-rows") {
            const long v = std::atol(next("N").c_str());
            if (v < 0 || v > 64) {
                std::fprintf(stderr, "--decode1-max-rows は 0..64\n");
                return usage();
            }
            srv.decode1_max_rows = static_cast<std::uint32_t>(v);
        } else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); return usage(); }
    }
    if (!srv.has_model) {
        // プロトコルだけのモード（ping 応答・モデル不要）。
        std::fprintf(stderr, "%s: model なしモード（ping のみ）\n", kVersion);
    }

    if (srv.has_model) {
        const std::string config_text = app::read_file_str(srv.model_dir + "/config.json");
        auto cfg_r = model::qwen35::load_text_config_json(config_text);
        if (!cfg_r.ok()) {
            std::fprintf(stderr, "shelf-compute: config: %s\n", cfg_r.status().message().c_str());
            return 1;
        }
        srv.cfg = cfg_r.release();
        auto fmt_r = app::read_fp8_format(config_text);
        if (!fmt_r.ok()) {
            std::fprintf(stderr, "shelf-compute: quantization_config: %s\n",
                         fmt_r.status().message().c_str());
            return 1;
        }
        srv.fmt = fmt_r.release();

        // 停止 ID（T01）: generation_config を主、モデル設定を従にして解決する。
        const std::string gen_text =
            app::read_file_str(srv.model_dir + "/generation_config.json");
        // tokenizer の eos も見る（checkpoint の generation_config が
        // `<|endoftext|>` だけを宣言しているモデルの取りこぼしを防ぐ）。
        const std::string tok_cfg_text =
            app::read_file_str(srv.model_dir + "/tokenizer_config.json");
        const std::string tok_json_text =
            app::read_file_str(srv.model_dir + "/tokenizer.json");
        auto stops_r = runtime::generate::resolve_stop_tokens(config_text, gen_text,
                                                              tok_cfg_text, tok_json_text);
        if (!stops_r.ok()) {
            std::fprintf(stderr, "shelf-compute: stop token: %s\n",
                         stops_r.status().message().c_str());
            return 1;
        }
        srv.stops = stops_r.release();

        // 活性化量子化（T09）: MXFP4 重みは W4A8（E4M3、group 128）で走る。
        if (srv.fmt.mxfp4)
            std::fprintf(stderr, "shelf-compute: activation = FP8 (W4A8 group=%u)（%s）\n",
                         srv.fmt.act_group, srv.fmt.declared.c_str());
        else if (srv.fmt.quantized)
            std::fprintf(stderr, "shelf-compute: activation = FP8 W8A8（%s）\n",
                         srv.fmt.declared.c_str());

        if (srv.has_eos_override) srv.stops.ids = srv.eos_override;
        if (srv.ignore_eos) srv.stops.ids.clear();
        if (srv.stops.empty()) {
            std::fprintf(stderr, "shelf-compute: 停止 ID なし（max_new_tokens だけで停止する）\n");
        } else {
            std::fprintf(stderr, "shelf-compute: stop ids =");
            for (const std::int32_t id : srv.stops.ids) std::fprintf(stderr, " %d", id);
            std::fprintf(stderr, "\n");
        }

        app::BootRequest breq;
        breq.device_execution = (device == "hip");
        breq.first_device = static_cast<std::size_t>(srv.gpu);
        breq.pipeline_parallel = srv.pp;
        if (breq.device_execution) breq.required.add(backend::Feature::device_execution);
        if (srv.pp > 1) breq.required.add(backend::Feature::pipeline_parallel);
        if (srv.fmt.quantized) breq.required.add(backend::Feature::fp8_w8a8);
        // stdout は JSONL 専用（engine 契約）。起動要約は stderr に出す。
        const auto booted = app::boot(breq, /*print=*/false);
        std::fprintf(stderr, "boot: %s\n", booted.summary.c_str());
        if (!booted.accepted) {
            std::fprintf(stderr, "shelf-compute: 起動条件: %s\n", booted.reason.c_str());
            return 1;
        }
        app::ShardSet shards;
        if (auto st = app::open_shards(srv.model_dir, &shards); !st.ok()) {
            std::fprintf(stderr, "shelf-compute: %s\n", st.message().c_str());
            return 1;
        }
        auto cat = app::build_catalog(shards);
        if (!cat.ok()) {
            std::fprintf(stderr, "shelf-compute: catalog: %s\n", cat.status().message().c_str());
            return 1;
        }
        if (srv.max_context == 0) srv.max_context = 4096;
        // T16: prefix cache 用のグローバル KV ブロック数。
        std::size_t kv_bs = 16;
        if (const char* e = std::getenv("SHELF_KV_PAGE")) {
            const long v = std::strtol(e, nullptr, 10);
            if (v == 16 || v == 32 || v == 64 || v == 128) kv_bs = static_cast<std::size_t>(v);
        }
        srv.prefix_bs = kv_bs;
        const std::size_t pages_per_seq =
            (static_cast<std::size_t>(srv.max_context) + 4 + kv_bs - 1) / kv_bs;
        app::PipelineOptions po;
        for (std::size_t k = 0; k < srv.pp; ++k)
            po.devices.push_back(srv.gpu + static_cast<int>(k));
        po.kv_capacity = srv.max_context + 4;
        po.chunk_rows = srv.prefill_rows;
        po.logits_last_row_only = true;
        po.max_sequences = srv.max_batch;
        if (srv.prefix_cache_blocks > 0) {
            srv.kv_block_count =
                pages_per_seq * srv.max_batch + static_cast<std::size_t>(srv.prefix_cache_blocks);
            po.kv_blocks = srv.kv_block_count;
            srv.prefix = std::make_unique<runtime::state::PrefixCache>(
                srv.kv_block_count, srv.prefix_bs);
            std::fprintf(stderr,
                         "shelf-compute: prefix cache blocks=%zu (pages/seq=%zu, extra=%lld) "
                         "snapshot_tokens=%zu limit=%zu\n",
                         srv.kv_block_count, pages_per_seq, srv.prefix_cache_blocks,
                         srv.prefix_snapshot_tokens, srv.prefix_snapshot_limit);
        }
        po.rope_theta = srv.cfg.rope_theta;
        po.rope_rotary_dim = srv.cfg.rotary_dim;
        po.fp8_act_group = srv.fmt.act_group;
        po.decode1_max_rows = srv.decode1_max_rows;
        po.offload_weights_bytes = static_cast<std::uint64_t>(srv.offload_mib) << 20;
        po.allow_host_weights = srv.allow_host_weights;
        po.kv_pinned_host = srv.kv_in_host;
        po.allow_host_kv = srv.allow_host_kv;
        po.online_kv_pages = srv.online_kv_pages;
        if (srv.vram_budget_mib > 0)
            po.device_free_bytes = static_cast<std::uint64_t>(srv.vram_budget_mib) << 20;
        po.verbose = false;
        // デバッグ: アリーナの使い回しを止める（primitive ごとの値を正しく読むため）。
        if (std::getenv("SHELF_ARENA_NO_REUSE") != nullptr) po.arena_reuse = false;
        auto pl = app::make_pipeline(srv.cfg, shards, cat.value(), srv.fmt, po);
        if (!pl.ok()) {
            std::fprintf(stderr, "shelf-compute: pipeline: %s\n", pl.status().message().c_str());
            return 1;
        }
        srv.pipeline = pl.release();
        std::fprintf(stderr,
                     "shelf-compute: ready (model=%s, max_context=%zu, device=%.0f MiB, "
                     "host_weights=%.0f MiB, host_kv=%.0f MiB)\n",
                     srv.model_dir.c_str(), srv.max_context,
                     static_cast<double>(srv.pipeline.device_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(srv.pipeline.host_weight_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(srv.pipeline.host_kv_bytes) / (1024.0 * 1024.0));
    }


    // T15: 連続バッチング。受付 → prefill チャンク → decode バッチ → 終了/再利用。
    srv.sched = std::make_unique<runtime::generate::BatchScheduler>(
        srv.max_batch, std::max<std::size_t>(srv.prefill_rows, 1));
    std::string line;
    for (;;) {
        // 1) 届いている制御行を読み切る（generate / cancel / ping）。
        srv.drain_input();
        srv.emit_finished();
        // 2) 1 ステップ実行（prefill 優先、無ければ decode バッチ）。
        if (srv.step_once()) continue;
        // 3) 何も実行できない: 入力を 1 行待つ（実行中の要求が無いときだけブロック）。
        if (srv.idle()) {
            if (srv.stdin_eof) break;
            if (!read_line(&line)) {
                srv.stdin_eof = true;
                continue;
            }
            srv.process_line(line);
            srv.emit_finished();
            continue;
        }
        if (srv.stdin_eof) break;
        if (!read_line(&line)) {
            srv.stdin_eof = true;
            continue;
        }
        srv.process_line(line);
        srv.emit_finished();
    }
    // T17: MoE の分担（CPU / GPU）の実測。既定は黙る。
    if (std::getenv("SHELF_MOE_STATS") != nullptr && !srv.pipeline.stages.empty()) {
        std::size_t cpu = 0, gpu = 0, syncs = 0;
        double cpu_ms = 0.0;
        std::uint64_t copy_bytes = 0;
        for (auto& st : srv.pipeline.stages) {
            if (!st.exec) continue;
            cpu += st.exec->moe_cpu_experts();
            gpu += st.exec->moe_gpu_experts();
            cpu_ms += st.exec->moe_cpu_ms();
            syncs += st.exec->moe_router_syncs();
            copy_bytes += st.exec->moe_router_copy_bytes();
        }
        std::fprintf(stderr,
                     "[moe] cpu_experts=%zu gpu_experts=%zu cpu_ms=%.1f "
                     "router_syncs=%zu router_copy_kib=%.1f\n",
                     cpu, gpu, cpu_ms, syncs, static_cast<double>(copy_bytes) / 1024.0);
    }
    // T16/N03: prefix cache の実効性（hit / miss / LRU 退避）を 1 行で残す。
    if (srv.prefix) {
        std::fprintf(stderr, "[prefix] stats hits=%zu misses=%zu evictions=%zu blocks=%zu\n",
                     srv.prefix->hits(), srv.prefix->misses(), srv.prefix->evictions(),
                     srv.prefix->num_blocks());
    }
    write_event(api::Event{api::ShutdownEvent{}});
    return 0;
}
