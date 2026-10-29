// shelf apps/common — pipeline-parallel の実装（HIP のみ）。
#include "pipeline.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <utility>

#ifdef SHELF_HAVE_HIP
namespace shelf::app {
namespace {

using model::qwen35::StageRange;

}  // namespace

std::vector<StageRange> even_stage_ranges(std::size_t num_layers, std::size_t k) {
    std::vector<StageRange> out;
    if (k == 0 || k > num_layers) return out;
    const std::size_t per = num_layers / k;
    std::size_t lo = 0;
    for (std::size_t s = 0; s < k; ++s) {
        const std::size_t hi = (s + 1 == k) ? num_layers : lo + per;
        out.push_back(StageRange{lo, hi});
        lo = hi;
    }
    return out;
}

Result<Pipeline> make_pipeline(const model::qwen35::TextConfig& cfg, const ShardSet& shards,
                               const Fp8ModelFormat& fmt, const PipelineOptions& opts) {
    if (opts.devices.empty())
        return Status::invalid_argument("pipeline: devices が空", __FILE__, __LINE__);
    if (opts.devices.size() > cfg.num_layers)
        return Status::invalid_argument("pipeline: ステージ数が層数を超えている", __FILE__,
                                        __LINE__);
    const auto ranges = even_stage_ranges(cfg.num_layers, opts.devices.size());
    if (ranges.empty())
        return Status::invalid_argument("pipeline: レイヤ範囲を作れない", __FILE__, __LINE__);

    Pipeline p;
    p.hidden_size = cfg.hidden_size;
    p.handoff.assign(cfg.hidden_size * (opts.chunk_rows > 1 ? opts.chunk_rows : 1), bf16_t{0u});
    p.stages.reserve(ranges.size());

    for (std::size_t s = 0; s < ranges.size(); ++s) {
        const StageRange& r = ranges[s];
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
            auto pr = model::qwen35::build_step_graph(cfg, opts.kv_capacity, opts.chunk_rows, r);
            if (!pr.ok())
                return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                                 " の prefill グラフ: " + pr.status().message(),
                                             __FILE__, __LINE__);
            st.prefill = std::make_unique<model::qwen35::StepGraphPlan>(pr.release());
        }

        backend::hip::HipStepExecutorOptions ho;
        ho.device = opts.devices[s];
        ho.rope_theta = opts.rope_theta;
        ho.rope_rotary_dim = opts.rope_rotary_dim;
        ho.arena_bytes = opts.arena_bytes;
        ho.fp8_act_group = opts.fp8_act_group;
        st.exec = std::make_unique<backend::hip::HipStepExecutor>(ho);

        // 宣言は行数の大きいグラフで行う（arena の自動算出を prefill 側に合わせる）。
        const auto& declare_plan = st.prefill ? *st.prefill : st.decode;
        auto up = upload_plan(*st.exec, declare_plan, shards, fmt);
        if (!up.ok())
            return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                             " の重み: " + up.message(),
                                         __FILE__, __LINE__);
        if (s == 0) p.arena_bytes = st.exec->arena_bytes();
        if (opts.verbose) {
            std::printf("  stage %zu: gpu %d layers [%zu, %zu) %s%s weights %zu arena %.0f MB\n",
                        s, opts.devices[s], r.layer_lo, r.layer_hi,
                        (r.layer_lo == 0 ? "embed+" : ""),
                        (r.layer_hi == cfg.num_layers ? "+head" : ""),
                        st.decode.weights.size(),
                        static_cast<double>(st.exec->arena_bytes()) / (1024.0 * 1024.0));
        }
        p.stages.push_back(std::move(st));
    }
    return p;
}

Status pipeline_reset(Pipeline& p) {
    for (auto& st : p.stages) SHELF_RETURN_IF_ERROR(st.exec->reset_states());
    return Status::make_ok();
}

Status pipeline_step(Pipeline& p, const std::int32_t* tokens, std::size_t n,
                     std::uint32_t base_pos, std::vector<float>* logits) {
    if (p.stages.empty())
        return Status::invalid_state("pipeline_step: ステージが無い", __FILE__, __LINE__);
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t s = 0; s < p.stages.size(); ++s) {
        PipelineStage& st = p.stages[s];
        const model::qwen35::StepGraphPlan& plan =
            (n > 1 && st.prefill) ? *st.prefill : st.decode;
        const auto& g = plan.graph;
        const bool first = (s == 0);
        const bool final = (s + 1 == p.stages.size());
        const bf16_t* hidden = first ? nullptr : p.handoff.data();
        auto rs = st.exec->run_stage(g, first ? tokens : nullptr, hidden, n, base_pos);
        if (!rs.ok())
            return Status::invalid_state("pipeline: ステージ " + std::to_string(s) + ": " +
                                             rs.message(),
                                         __FILE__, __LINE__);
        if (!final) {
            const auto& out_id = plan.hidden_out;
            if (!out_id.valid())
                return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                                 " に hidden_out が無い",
                                             __FILE__, __LINE__);
            auto hb = st.exec->fetch_value_bf16(g, out_id);
            if (!hb.ok())
                return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                                 " の hidden 読み出し: " + hb.status().message(),
                                             __FILE__, __LINE__);
            if (hb.value().size() > p.handoff.size())
                return Status::invalid_state("pipeline: hand-off バッファが小さい", __FILE__,
                                             __LINE__);
            std::copy(hb.value().begin(), hb.value().end(), p.handoff.begin());
        } else if (logits != nullptr) {
            auto lg = st.exec->fetch_logits(g);
            if (!lg.ok()) return lg.status();
            *logits = std::move(lg.value());
        }
    }
    p.last_step_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
    return Status::make_ok();
}

}  // namespace shelf::app
#endif  // SHELF_HAVE_HIP
