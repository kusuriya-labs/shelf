// shelf-run — end-to-end text driver for the Qwen3.5-family CPU reference
// path. Loads safetensors shards, replays the single-step Model Graph
// token-by-token (prefill == decode here; slow but exact-by-construction),
// greedy-samples and dumps final logits for oracle parity checks.
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <shelf/artifact/safetensors/reader.hpp>
#include <shelf/backend/cpu/reference_executor.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

#include "common/app_support.hpp"
#include "common/pipeline.hpp"

#if defined(SHELF_HAVE_HIP)
#include <memory>
#endif
#ifdef SHELF_HAVE_HIP
#include <shelf/backend/hip/step_executor.hpp>
#endif

namespace {

using namespace shelf;

int fail(const std::string& msg) {
    std::fprintf(stderr, "shelf-run: %s\n", msg.c_str());
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_dir, ids_csv, out_dir = "artifacts", device = "cpu";
    int gpu = 0;
    int steps = 16;
    long long arena_mib = 0;  // 0 = declare_states のグラフから自動算出
    std::string dump_acts;    // 指定すると prefill 後の linear 入力（f32）を保存する
    int fp8_act_group = -1;   // FP8 活性化の量子化粒度。-1 = config.json の宣言に従う
    int prefill_rows = 64;    // prefill の 1 チャンク行数（VRAM が足りないとき落とす）
    int pp = 1;               // pipeline-parallel のステージ数（1 = 単一 GPU）
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") model_dir = next("model dir");
        else if (a == "--ids") ids_csv = next("ids");
        else if (a == "--steps") steps = std::atoi(next("steps").c_str());
        else if (a == "--out") out_dir = next("out");
        else if (a == "--device") device = next("cpu|hip");
        else if (a == "--gpu") gpu = std::atoi(next("gpu index").c_str());
        else if (a == "--arena-mib") arena_mib = std::atoll(next("arena MiB").c_str());
        else if (a == "--dump-acts") dump_acts = next("dir");
        else if (a == "--fp8-act-group") fp8_act_group = std::atoi(next("auto|0|128").c_str());
        else if (a == "--prefill-rows") prefill_rows = std::atoi(next("rows").c_str());
        else if (a == "--pipeline-parallel") pp = std::atoi(next("K").c_str());
        else return fail("usage: shelf-run --model DIR --ids 1,2,3 [--steps N] [--out DIR] "
                         "[--device cpu|hip] [--gpu N] [--arena-mib N] [--dump-acts DIR] "
                         "[--fp8-act-group auto|0|128] [--prefill-rows N] [--pipeline-parallel K]");
    }
    if (fp8_act_group != -1 && fp8_act_group != 0 && fp8_act_group != 128)
        return fail("--fp8-act-group は auto（省略）か 0 か 128");
    if (model_dir.empty() || ids_csv.empty()) return fail("--model and --ids required");
    if (arena_mib < 0) return fail("--arena-mib must be >= 0");
    if (prefill_rows < 1 || prefill_rows > 64)
        return fail("--prefill-rows must be in [1, 64]");
    if (pp < 1) return fail("--pipeline-parallel は 1 以上");
    if (pp > 1 && device != "hip") return fail("--pipeline-parallel > 1 は hip のみ");

    // ---- config ----
    const std::string config_text = app::read_file_str(model_dir + "/config.json");
    auto cfg_r = model::qwen35::load_text_config_json(config_text);
    if (!cfg_r.ok()) return fail(cfg_r.status().message());
    const auto& cfg = cfg_r.value();
    // FP8 の形は checkpoint の宣言（config.json）が正。CLI は実験用の上書きで、
    // 使った場合は明示して印字する（隠さない）。
    auto fmt_r = app::read_fp8_format(config_text);
    if (!fmt_r.ok()) return fail(fmt_r.status().message());
    app::Fp8ModelFormat fmt = fmt_r.value();
    if (fp8_act_group >= 0) fmt.act_group = static_cast<std::uint32_t>(fp8_act_group);

    std::vector<std::int64_t> ids;
    {
        std::stringstream ss(ids_csv);
        std::string part;
        while (std::getline(ss, part, ',')) if (!part.empty()) ids.push_back(std::stoll(part));
    }
    if (ids.empty()) return fail("empty ids");
    const std::size_t capacity = ids.size() + static_cast<std::size_t>(steps) + 4;

    // ---- graph ----
    const std::size_t pf_rows =
        (device == "hip") ? std::min<std::size_t>(ids.size(),
                                            static_cast<std::size_t>(prefill_rows))
                                  : 1;
    auto plan_r = model::qwen35::build_step_graph(cfg, capacity);
    if (!plan_r.ok()) return fail(plan_r.status().message());
    auto& plan = plan_r.value();
    std::unique_ptr<model::qwen35::StepGraphPlan> pfplan;
    if (pf_rows > 1) {
        auto pr = model::qwen35::build_step_graph(cfg, capacity, pf_rows);
        if (!pr.ok()) return fail("prefill graph: " + pr.status().message());
        pfplan = std::make_unique<model::qwen35::StepGraphPlan>(pr.release());
    }
    std::printf("graph: %zu primitives, %zu weights, %zu states, kv_capacity=%zu\n",
                plan.graph.size(), plan.weights.size(), plan.states.size(), capacity);

    // ---- shards ----
    app::ShardSet shards;
    if (auto sst = app::open_shards(model_dir, &shards); !sst.ok())
        return fail(sst.message());

    // ---- weights ----
#ifdef SHELF_HAVE_HIP
    app::Pipeline pipeline;   // PP=1 のときは 1 ステージ（単一 GPU）
#endif
    backend::cpu::HostStorage storage;
    if (device == "hip") {
#ifdef SHELF_HAVE_HIP
        if (static_cast<std::size_t>(pp) > cfg.num_layers)
            return fail("--pipeline-parallel は層数以下にしてください（層 " +
                        std::to_string(cfg.num_layers) + "）");
        app::PipelineOptions po;
        for (int k = 0; k < pp; ++k) po.devices.push_back(gpu + k);
        po.kv_capacity = capacity;
        po.chunk_rows = (pf_rows > 1) ? pf_rows : 1;
        po.arena_bytes = arena_mib > 0 ? static_cast<std::size_t>(arena_mib) << 20 : 0;
        po.rope_theta = cfg.rope_theta;
        po.rope_rotary_dim = cfg.rotary_dim;
        po.fp8_act_group = fmt.act_group;
        po.verbose = true;
        auto pl = app::make_pipeline(cfg, shards, fmt, po);
        if (!pl.ok()) return fail("pipeline: " + pl.status().message());
        pipeline = pl.release();
        std::printf("fp8: %s (act_group=%u%s)\n",
                    fmt.quantized ? fmt.declared.c_str() : "量子化なし（bf16 のまま）",
                    fmt.act_group,
                    fp8_act_group >= 0 ? " ※--fp8-act-group で上書き" : " ※config.json の宣言");
        std::printf("pipeline: %d stage (gpu %d..%d), arena %.0f MiB (%s)\n", pp, gpu,
                    gpu + pp - 1,
                    static_cast<double>(pipeline.arena_bytes) / (1024.0 * 1024.0),
                    arena_mib > 0 ? "pinned" : "auto");
#else
        return fail("built without HIP support");
#endif
    } else {
        for (const auto& w : plan.weights) {
            auto t = app::read_f32(shards.where, w.checkpoint_name);
            if (!t.ok()) return fail(t.status().message());
            if (t.value().size() != app::numel(w.shape)) {
                return fail("shape mismatch for " + w.checkpoint_name + ": file has " +
                            std::to_string(t.value().size()) + ", IR contract needs " +
                            std::to_string(app::numel(w.shape)));
            }
            auto st = storage.set_weight(w.id, t.release());
            if (!st.ok()) return fail("set_weight " + w.checkpoint_name + ": " + st.message());
        }
        for (const auto& s : plan.states) {
            auto st = storage.init_state(s.id, std::vector<float>(app::numel(s.shape), 0.0f));
            if (!st.ok()) return fail("init_state: " + st.message());
        }
    }

    // ---- rope tables (text path: all mrope channels equal -> identity; HF
    //      computes inv_freq/cos/sin in f32 — we compute in double and round) ----
    const std::size_t rd = cfg.rotary_dim;
    std::vector<double> inv(rd / 2);
    for (std::size_t j = 0; j < rd / 2; ++j)
        inv[j] = std::pow(cfg.rope_theta, -2.0 * static_cast<double>(j) / static_cast<double>(rd));

    auto run_step = [&](std::int64_t token, std::size_t pos, std::vector<float>* logits_out)
        -> Status {
#ifdef SHELF_HAVE_HIP
        if (!pipeline.stages.empty()) {
            const std::int32_t t = static_cast<std::int32_t>(token);
            SHELF_RETURN_IF_ERROR(app::pipeline_step(pipeline, &t, 1,
                                                     static_cast<std::uint32_t>(pos),
                                                     logits_out));
            std::printf("  hip step: %.2f ms\n", pipeline.last_step_ms);
            return Status::make_ok();
        }
#endif
        storage.reset_values();
        auto st = storage.set_value_int(plan.tokens, {static_cast<std::int32_t>(token)});
        if (!st.ok()) return st;
        std::vector<float> cosv(rd), sinv(rd);
        for (std::size_t j = 0; j < rd / 2; ++j) {
            const double f = static_cast<double>(pos) * inv[j];
            cosv[j] = cosv[j + rd / 2] = static_cast<float>(std::cos(f));
            sinv[j] = sinv[j + rd / 2] = static_cast<float>(std::sin(f));
        }
        SHELF_RETURN_IF_ERROR(storage.set_value(plan.cos, std::move(cosv)));
        SHELF_RETURN_IF_ERROR(storage.set_value(plan.sin, std::move(sinv)));
        SHELF_RETURN_IF_ERROR(backend::cpu::ReferenceExecutor::run(plan.graph, storage));
        if (logits_out) {
            const auto* lv = storage.value(plan.logits);
            if (!lv) return Status::invalid_state("logits missing after run", __FILE__, __LINE__);
            *logits_out = *lv;
        }
        return Status::make_ok();
    };

    // ---- replay prompt, then greedy decode ----
    std::vector<std::int64_t> seq = ids;
    std::vector<float> last_logits;
    const auto gpu_prefill = [&]() -> int {
        std::vector<std::int32_t> win;
        for (std::size_t c = 0; c < ids.size(); c += pf_rows) {
            const std::size_t n = std::min(pf_rows, ids.size() - c);
            win.clear();
            for (std::size_t i = 0; i < n; ++i)
                win.push_back(static_cast<std::int32_t>(ids[c + i]));
            const bool last_chunk = c + n == ids.size();
#ifdef SHELF_HAVE_HIP
            std::vector<float> lg;
            auto st = app::pipeline_step(pipeline, win.data(), n,
                                         static_cast<std::uint32_t>(c),
                                         last_chunk ? &lg : nullptr);
            if (!st.ok()) return fail("prefill chunk " + std::to_string(c) + ": " + st.message());
            std::printf("prefill chunk rows=%zu base=%zu\n", n, c);
            if (last_chunk) {
                // [n,V]: 最終行を採用
                last_logits.assign(lg.end() - static_cast<std::ptrdiff_t>(lg.size() / n),
                                   lg.end());
            }
#else
            (void)last_chunk;
            (void)c;
#endif
        }
        return 0;
    };
#ifdef SHELF_HAVE_HIP
    if (!pipeline.stages.empty() && pfplan) { if (int rc = gpu_prefill()) return rc; } else
#endif
    for (std::size_t p = 0; p < ids.size(); ++p) {
        auto st = run_step(seq[p], p, p + 1 == ids.size() ? &last_logits : nullptr);
        if (!st.ok()) return fail("prefill step " + std::to_string(p) + ": " + st.message());
        std::printf("prefill pos=%zu tok=%lld\n", p, static_cast<long long>(seq[p]));
    }
#ifdef SHELF_HAVE_HIP
    // --dump-acts: prefill 直後の linear 入力（f32）を保存する。
    // 量子化粒度の比較（token vs 128 グループ）や T2 の基準取りに使う開発用道具。
    if (!dump_acts.empty() && !pipeline.stages.empty()) {
        auto& st0 = pipeline.stages.front();
        auto* hip_exec = st0.exec.get();
        std::filesystem::create_directories(dump_acts);
        const auto& pgraph = (pfplan && st0.prefill) ? st0.prefill->graph : st0.decode.graph;
        std::ofstream manifest(dump_acts + "/manifest.json");
        manifest << "[\n";
        std::size_t idx = 0, dumped = 0;
        for (const auto& p : pgraph.primitives()) {
            if (p.kind != shelf::ir::Kind::linear) { ++idx; continue; }
            if (p.inputs.empty() || p.weights.empty()) { ++idx; continue; }
            std::string wname;
            for (const auto& w : st0.decode.weights)
                if (w.id == p.weights[0]) { wname = w.checkpoint_name; break; }
            auto lg = hip_exec->fetch_value_debug(pgraph, p.inputs[0]);
            if (!lg.ok()) { std::printf("dump skip #%zu: %s\n", idx, lg.status().message().c_str()); ++idx; continue; }
            const auto v = lg.value();
            const auto* vd = pgraph.find_value(p.inputs[0]);
            const std::size_t cols = vd ? vd->shape.dim(vd->shape.rank() - 1) : 0u;
            const std::size_t rows = cols ? v.size() / cols : 0u;
            char fname[64];
            std::snprintf(fname, sizeof fname, "act_%04zu.bin", idx);
            std::ofstream f(dump_acts + "/" + fname, std::ios::binary);
            f.write(reinterpret_cast<const char*>(v.data()),
                    static_cast<std::streamsize>(v.size() * sizeof(float)));
            if (dumped) manifest << ",\n";
            manifest << "  {\"i\":" << idx << ",\"weight\":\"" << wname << "\",\"rows\":" << rows
                     << ",\"cols\":" << cols << ",\"file\":\"" << fname << "\"}";
            ++dumped;
            ++idx;
        }
        manifest << "\n]\n";
        std::printf("dump-acts: %zu tensors -> %s\n", dumped, dump_acts.c_str());
    }
#endif
    // position of token seq[i] is i; the token sampled from prefill logits
    // occupies position T, so step s>=1 feeds seq.back() at position T+s-1.
    for (int s = 0; s < steps; ++s) {
        if (s > 0) {
            auto st = run_step(seq.back(), ids.size() + static_cast<std::size_t>(s) - 1,
                               &last_logits);
            if (!st.ok()) return fail("decode step " + std::to_string(s) + ": " + st.message());
        }
        std::size_t best = 0;
        for (std::size_t i = 1; i < last_logits.size(); ++i)
            if (last_logits[i] > last_logits[best]) best = i;
        seq.push_back(static_cast<std::int64_t>(best));
        std::printf("decode pos=%zu tok=%zu\n", ids.size() + static_cast<std::size_t>(s), best);
        if (static_cast<std::int64_t>(best) == cfg.eos_token_id) break;
    }

    // ---- outputs ----
    std::string ids_line;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        if (i) ids_line += ",";
        ids_line += std::to_string(seq[i]);
    }
    std::printf("seq: %s\n", ids_line.c_str());
    if (!last_logits.empty()) {
        std::filesystem::create_directories(out_dir);
        std::ofstream f(out_dir + "/logits_f32.bin", std::ios::binary);
        if (!f) return fail("cannot write " + out_dir + "/logits_f32.bin");
        f.write(reinterpret_cast<const char*>(last_logits.data()),
                static_cast<std::streamsize>(last_logits.size() * sizeof(float)));
        std::printf("wrote %s/logits_f32.bin (%zu floats)\n", out_dir.c_str(), last_logits.size());
    }
    return 0;
}
