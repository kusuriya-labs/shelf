// shelf apps/common — pipeline-parallel の組み立て（HIP のみ）。
//
// 実行はエンジン（runtime/pipeline）が担う。ここはステージの**生成**だけを行う。
#include "pipeline.hpp"

#include <cstdio>

#include <shelf/ir/graph.hpp>
#include <string>
#include <utility>

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
            auto pr = model::qwen35::build_step_graph(cfg, opts.kv_capacity, opts.chunk_rows, r);
            if (!pr.ok())
                return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                                 " の prefill グラフ: " +
                                                 pr.status().message(),
                                             __FILE__, __LINE__);
            st.prefill = std::make_unique<model::qwen35::StepGraphPlan>(pr.release());
        }

        backend::hip::HipStepExecutorOptions ho;
        ho.device = opts.devices[s];
        ho.rope_theta = opts.rope_theta;
        ho.rope_rotary_dim = opts.rope_rotary_dim;
        ho.arena_bytes = opts.arena_bytes;
        ho.arena_reuse = opts.arena_reuse;
        ho.fp8_act_group = opts.fp8_act_group;
        st.exec = std::make_unique<backend::hip::HipStepExecutor>(ho);

        // 宣言は行数の大きいグラフで行う（arena の自動算出を prefill 側に合わせる）。
        const auto& declare_plan = st.prefill ? *st.prefill : st.decode;
        MaterializationPlan mp;
        auto up = upload_plan(*st.exec, declare_plan, catalog, shards, fmt, &mp);
        if (!up.ok())
            return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                             " の重み: " + up.message(),
                                         __FILE__, __LINE__);
        if (s == 0) p.arena_bytes = st.exec->arena_bytes();
        p.weight_bytes += mp.bytes;
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

    // エンジンへ編成を渡す（ポインタは p.stages の要素＝ヒープ上の実体を指す）。
    p.engine.hidden_size = p.hidden_size;
    auto prep = runtime::pipeline::prepare(p.engine, opts.chunk_rows);
    if (!prep.ok()) return prep;
    p.engine.stages.reserve(p.stages.size());
    for (auto& st : p.stages) {
        runtime::pipeline::Stage es;
        es.graphs.decode = &st.decode.graph;
        es.graphs.prefill = st.prefill ? &st.prefill->graph : nullptr;
        es.graphs.hidden_out_decode = st.decode.hidden_out;
        es.graphs.hidden_out_prefill = st.prefill ? st.prefill->hidden_out : ir::ValueId{};
        es.exec = st.exec.get();
        es.layer_lo = st.layer_lo;
        es.layer_hi = st.layer_hi;
        p.engine.stages.push_back(es);
    }
    p.engine.hidden_size = p.hidden_size;
    p.engine.arena_bytes = p.arena_bytes;
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
