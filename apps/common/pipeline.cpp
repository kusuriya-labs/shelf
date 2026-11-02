// shelf apps/common — pipeline-parallel の組み立て（HIP のみ）。
//
// 実行はエンジン（runtime/pipeline）が担う。ここはステージの**生成**だけを行う。
#include "pipeline.hpp"

#include <algorithm>
#include <cstdio>

#include <shelf/compiler/arena_offsets.hpp>
#include <shelf/foundation/dtype.hpp>
#include <shelf/ir/graph.hpp>
#include <shelf/placement/budget.hpp>
#include <string>
#include <utility>

#ifdef SHELF_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef SHELF_HAVE_HIP
namespace shelf::app {

Result<Pipeline> make_pipeline(const model::qwen35::TextConfig& cfg, const ShardSet& shards,
                               const weights::TensorCatalog& catalog, const Fp8ModelFormat& fmt,
                               const PipelineOptions& opts) {
    if (opts.devices.empty())
        return Status::invalid_argument("pipeline: devices が空", __FILE__, __LINE__);
    if (opts.devices.size() > cfg.num_layers)
        return Status::invalid_argument("pipeline: ステージ数が層数を超えている", __FILE__,
                                        __LINE__);
    const auto rrs = runtime::pipeline::even_stage_ranges(cfg.num_layers, opts.devices.size());
    std::vector<model::qwen35::StageRange> ranges;
    ranges.reserve(rrs.size());
    for (const auto& r : rrs) ranges.push_back(model::qwen35::StageRange{r.lo, r.hi});
    if (ranges.empty())
        return Status::invalid_argument("pipeline: レイヤ範囲を作れない", __FILE__, __LINE__);

    Pipeline p;
    p.hidden_size = cfg.hidden_size;
    p.stages.reserve(ranges.size());

    // ---- 予算判定（D2）: 確保の前に「載るか」を既知量の算術で決める ----
    //   重み = 実体化計画のバイト数、状態 = 宣言済み state の f32、scratch = アリーナ。
    //   実確保（hipMalloc）はこの判定を通ってから行う。
    if (opts.check_budget) {
        placement::Footprint fp;
        for (std::size_t s = 0; s < ranges.size(); ++s) {
            const auto& r = ranges[s];
            const bool final_stage = (s + 1 == ranges.size());
            auto dp = model::qwen35::build_step_graph(cfg, opts.kv_capacity, 1, r);
            if (!dp.ok()) continue;   // 形の誤りは後段の本構築が報告する
            auto pp = model::qwen35::build_step_graph(cfg, opts.kv_capacity,
                                                     std::max<std::size_t>(opts.chunk_rows, 1), r);
            const auto& declare_plan = pp.ok() ? pp.value() : dp.value();
            auto mp = plan_materialization(catalog, declare_plan, fmt);
            placement::StageCost cost;
            if (mp.ok()) cost.weight_bytes = mp.value().bytes;
            for (const auto& stt : declare_plan.states) {
                std::size_t n = 1;
                for (const auto d : stt.shape) n *= d;
                cost.state_bytes += static_cast<std::uint64_t>(n) * sizeof(float);
            }
            // アリーナは declare 相の見積り。`--arena-mib` で固定した場合はその値を
            // 使う（固定値の方が大きいことがある。過小見積りで予算を偽らない）。
            cost.scratch_bytes = opts.arena_bytes != 0
                                     ? opts.arena_bytes
                                     : compiler::plan_arena(declare_plan.graph, opts.arena_reuse)
                                           .peak_bytes;
            fp.add_stage(cost);
        }
        placement::Demand dem;
        dem.stages = ranges.size();
        const std::size_t full_layers = static_cast<std::size_t>(std::count_if(
            cfg.layer_kinds.begin(), cfg.layer_kinds.end(),
            [](model::qwen35::LayerKind k) { return k == model::qwen35::LayerKind::full_attention; }));
        dem.kv_bytes_per_token =
            placement::kv_bytes_per_token(full_layers, cfg.num_kv_heads, cfg.head_dim);
        placement::Limits lim;
        lim.device_free_bytes = opts.device_free_bytes;
        if (lim.device_free_bytes == 0) {
            // 呼び手が渡していない場合は 1 台目の空きを実測する（不明なら判定しない）。
            std::size_t free_b = 0, total_b = 0;
            if (hipSetDevice(opts.devices.front()) == hipSuccess &&
                hipMemGetInfo(&free_b, &total_b) == hipSuccess)
                lim.device_free_bytes = free_b;
        }
        const auto report = placement::plan_budget(fp, dem, lim);
        p.budget = report;
        if (opts.verbose) std::printf("  %s\n", report.describe().c_str());
        // T14: 予算超過でも host へ落として続行できる（pinned 正本）。明示 offload も同様。
        if (!report.ok() && !opts.allow_host_weights && opts.offload_weights_bytes == 0)
            return Status::insufficient_memory("pipeline: " + report.describe(), __FILE__,
                                               __LINE__);
    }


    for (std::size_t s = 0; s < ranges.size(); ++s) {
        const auto& r = ranges[s];
        auto dr = model::qwen35::build_step_graph(cfg, opts.kv_capacity, 1, r);
        if (!dr.ok())
            return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                             " の decode グラフ: " + dr.status().message(),
                                         __FILE__, __LINE__);
        PipelineStage st;
        st.layer_lo = r.layer_lo;
        st.layer_hi = r.layer_hi;
        st.decode = dr.release();
        if (opts.chunk_rows > 1) {
            model::qwen35::GraphOptions go;
            go.logits_last_row_only = opts.logits_last_row_only;
            auto pr = model::qwen35::build_step_graph(cfg, opts.kv_capacity, opts.chunk_rows, r,
                                                     go);
            if (!pr.ok())
                return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                                 " の prefill グラフ: " +
                                                 pr.status().message(),
                                             __FILE__, __LINE__);
            st.prefill = std::make_unique<model::qwen35::StepGraphPlan>(pr.release());
        }
        // T15: rows = 2..max_sequences の decode グラフ（行ごとの logits が要る）。
        if (opts.max_sequences > 1u) {
            st.decode_batches.reserve(opts.max_sequences - 1u);
            for (std::size_t rows = 2; rows <= opts.max_sequences; ++rows) {
                model::qwen35::GraphOptions go;
                go.logits_last_row_only = false;
                auto pr = model::qwen35::build_step_graph(cfg, opts.kv_capacity, rows, r, go);
                if (!pr.ok())
                    return Status::invalid_state(
                        "pipeline: ステージ " + std::to_string(s) + " の " +
                            std::to_string(rows) + " 行 decode グラフ: " +
                            pr.status().message(),
                        __FILE__, __LINE__);
                st.decode_batches.push_back(
                    std::make_unique<model::qwen35::StepGraphPlan>(pr.release()));
            }
        }

        // 宣言は行数の大きいグラフで行う（arena の自動算出を prefill 側に合わせる）。
        const auto& declare_plan = st.prefill ? *st.prefill : st.decode;

        // T14: KV を pinned host に置くか（強制 or 予算不足で自動）。
        std::uint64_t kv_state_bytes = 0, other_state_bytes = 0;
        for (const auto& stt : declare_plan.states) {
            std::size_t n = 1;
            for (const auto d : stt.shape) n *= d;
            const std::uint64_t b = static_cast<std::uint64_t>(n) * sizeof(float);
            if (stt.kind == ir::StateKind::kv_cache) kv_state_bytes += b;
            else other_state_bytes += b;
        }
        const std::uint64_t scratch_est =
            opts.arena_bytes != 0
                ? opts.arena_bytes
                : compiler::plan_arena(declare_plan.graph, opts.arena_reuse).peak_bytes;
        bool host_kv = opts.kv_pinned_host || opts.online_kv_pages != 0u;
        if (!host_kv && opts.allow_host_kv && opts.device_free_bytes != 0 && kv_state_bytes != 0 &&
            kv_state_bytes + other_state_bytes + scratch_est > opts.device_free_bytes) {
            host_kv = true;
        }

        backend::hip::HipStepExecutorOptions ho;
        ho.device = opts.devices[s];
        ho.rope_theta = opts.rope_theta;
        ho.rope_rotary_dim = opts.rope_rotary_dim;
        ho.arena_bytes = opts.arena_bytes;
        ho.arena_reuse = opts.arena_reuse;
        ho.fp8_act_group = opts.fp8_act_group;
        ho.decode1_max_rows = opts.decode1_max_rows;
        ho.max_sequences = std::max<std::size_t>(opts.max_sequences, 1u);
        ho.kv_blocks = opts.kv_blocks;
        // T17: `--allow-host-weights` は「収まらなければ host へ」の許可なので、
        // 実行時の device 確保失敗も pinned へ落として続行する（縮退）。
        ho.allow_host_fallback = opts.allow_host_weights;
        ho.kv_pinned_host = host_kv;
        ho.online_kv_pages = host_kv ? opts.online_kv_pages : 0u;
        // T14: online KV はスロットごとのリングと窓を使う（max_sequences まで）。
        // 窓は executor が行の位置からスロット単位で作る（`slot_window_end`）。
        st.exec = std::make_unique<backend::hip::HipStepExecutor>(ho);

        // ---- T14: 重みの置き場（VRAM / pinned host）を予算から決める ----
        // 候補はグラフの重み宣言順。`place_fitting_budget` に「使える device 量」を
        // 渡し、収まらない分を pinned host にする。状態とアリーナは device に残すので、
        // その分を先に差し引く。
        std::vector<char> host_flags;
        const bool want_placement = opts.offload_weights_bytes > 0 || opts.allow_host_weights;
        if (want_placement && !declare_plan.weights.empty()) {
            auto mp_stage = plan_materialization(catalog, declare_plan, fmt);
            if (!mp_stage.ok()) return mp_stage.status();
            std::vector<placement::Placement> cand;
            cand.reserve(declare_plan.weights.size());
            std::uint64_t weight_total = 0;
            for (const auto& w : declare_plan.weights) {
                std::uint64_t bytes = 0;
                if (const auto* spec = catalog.find(w.checkpoint_name)) {
                    const auto n = spec->shape.num_elements();
                    if (n.ok())
                        bytes = static_cast<std::uint64_t>(n.value()) *
                                element_size(spec->logical_dtype);
                }
                cand.push_back(placement::Placement{
                    w.checkpoint_name, bytes, backend::MemoryKind::device});
                weight_total += bytes;
            }
            // KV を host に置く場合、KV の分は device 予算から差し引かない。
            const std::uint64_t state_bytes = other_state_bytes + (host_kv ? 0 : kv_state_bytes);
            const std::uint64_t scratch_bytes = scratch_est;
            std::uint64_t usable = 0;
            if (opts.offload_weights_bytes > 0) {
                usable = weight_total > opts.offload_weights_bytes
                             ? weight_total - opts.offload_weights_bytes
                             : 1;
            } else {
                (void)hipSetDevice(opts.devices[s]);
                std::uint64_t free_b = opts.device_free_bytes;   // 0 = 実測
                if (free_b == 0) {
                    std::size_t fb = 0, total_b = 0;
                    if (hipMemGetInfo(&fb, &total_b) == hipSuccess) free_b = fb;
                }
                const std::uint64_t need = state_bytes + scratch_bytes;
                usable = free_b > need ? free_b - need : 1;
            }
            placement::Limits lim;
            lim.device_free_bytes = usable == 0 ? 1 : usable;
            lim.headroom = 0.0;   // 差し引き済みの量をそのまま使う
            placement::PlacementPlan plan;
            auto ps = placement::place_fitting_budget(cand, lim, &plan);
            if (!ps.ok()) return ps;
            host_flags.assign(cand.size(), 0);
            std::uint64_t offload_bytes = 0;
            std::size_t offload_count = 0;
            for (std::size_t i = 0; i < cand.size(); ++i) {
                if (plan.weights[i].kind != backend::MemoryKind::device) {
                    host_flags[i] = 1;
                    offload_bytes += cand[i].bytes;
                    ++offload_count;
                }
            }
            if (opts.verbose && offload_count != 0)
                std::printf("  stage %zu: offload %zu tensors / %.0f MB -> pinned host\n", s,
                            offload_count,
                            static_cast<double>(offload_bytes) / (1024.0 * 1024.0));
        }

        MaterializationPlan mp;
        auto up = upload_plan(*st.exec, declare_plan, catalog, shards, fmt, &mp, &host_flags);
        if (!up.ok())
            return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                             " の重み: " + up.message(),
                                         __FILE__, __LINE__);
        if (s == 0) p.arena_bytes = st.exec->arena_bytes();
        p.weight_bytes += mp.bytes;
        p.host_weight_bytes += st.exec->host_weight_bytes();
        p.host_kv_bytes += st.exec->host_kv_bytes();
        p.device_bytes += st.exec->device_bytes();
        if (opts.verbose) {
            std::printf("  stage %zu: gpu %d layers [%zu, %zu) %s%s weights %zu arena %.0f MB "
                        "device %.0f MB\n",
                        s, opts.devices[s], r.layer_lo, r.layer_hi,
                        (r.layer_lo == 0 ? "embed+" : ""),
                        (r.layer_hi == cfg.num_layers ? "+head" : ""),
                        st.decode.weights.size(),
                        static_cast<double>(st.exec->arena_bytes()) / (1024.0 * 1024.0),
                        static_cast<double>(st.exec->device_bytes()) / (1024.0 * 1024.0));
        }
        p.stages.push_back(std::move(st));
    }

    // エンジンへ編成を渡す（ポインタは p.stages の要素＝ヒープ上の実体を指す）。
    p.engine.hidden_size = p.hidden_size;
    p.max_sequences = opts.max_sequences;
    auto prep = runtime::pipeline::prepare(
        p.engine, std::max(opts.chunk_rows, opts.max_sequences));
    if (!prep.ok()) return prep;
    p.engine.stages.reserve(p.stages.size());
    for (auto& st : p.stages) {
        runtime::pipeline::Stage es;
        es.graphs.decode = &st.decode.graph;
        es.graphs.prefill = st.prefill ? &st.prefill->graph : nullptr;
        es.graphs.hidden_out_decode = st.decode.hidden_out;
        es.graphs.hidden_out_prefill = st.prefill ? st.prefill->hidden_out : ir::ValueId{};
        for (auto& plan : st.decode_batches) {
            es.graphs.decode_batch.push_back(&plan->graph);
            es.graphs.hidden_out_batch.push_back(plan->hidden_out);
        }
        es.exec = st.exec.get();
        es.layer_lo = st.layer_lo;
        es.layer_hi = st.layer_hi;
        p.engine.stages.push_back(es);
    }
    p.engine.hidden_size = p.hidden_size;
    p.engine.arena_bytes = p.arena_bytes;

    // 実 device の空きも出す（accounting は HIP コンテキストを含まないので、
    // 実使用量はこれより数百 MB 大きい）。
    if (opts.verbose) {
        for (std::size_t s = 0; s < p.stages.size(); ++s) {
            (void)hipSetDevice(opts.devices[s]);
            std::size_t f = 0u, t = 0u;
            if (hipMemGetInfo(&f, &t) == hipSuccess) {
                std::printf("  stage %zu: gpu %d 実 device 空き %.0f MiB / 総 %.0f MiB "
                            "(計上 %.0f MiB)\n",
                            s, opts.devices[s], static_cast<double>(f >> 20),
                            static_cast<double>(t >> 20),
                            static_cast<double>(p.stages[s].exec->device_bytes() >> 20));
            }
        }
    }
    return p;
}

Status pipeline_step(Pipeline& p, const std::int32_t* tokens, std::size_t n,
                     std::uint32_t base_pos, std::vector<float>* logits) {
    Status st = runtime::pipeline::step(p.engine, tokens, n, base_pos, logits);
    p.last_step_ms = p.engine.last_step_ms;
    return st;
}

}  // namespace shelf::app
#endif  // SHELF_HAVE_HIP
