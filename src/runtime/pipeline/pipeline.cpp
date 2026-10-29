// shelf runtime/pipeline — ステージの直列 hand-off。
#include <shelf/runtime/pipeline/pipeline.hpp>

#include <algorithm>
#include <chrono>
#include <string>

namespace shelf::runtime::pipeline {

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

Status prepare(Pipeline& p, std::size_t chunk_rows) {
    if (p.hidden_size == 0u)
        return Status::invalid_argument("pipeline: hidden_size が未設定", __FILE__, __LINE__);
    const std::size_t rows = chunk_rows == 0u ? 1u : chunk_rows;
    p.handoff.assign(p.hidden_size * rows, bf16_t{0u});
    return Status::make_ok();
}

Status reset(Pipeline& p) {
    for (auto& st : p.stages) {
        if (st.exec == nullptr)
            return Status::invalid_state("pipeline: 実行器が無い", __FILE__, __LINE__);
        SHELF_RETURN_IF_ERROR(st.exec->reset_states());
    }
    return Status::make_ok();
}

Status step(Pipeline& p, const std::int32_t* tokens, std::size_t n, std::uint32_t base_pos,
            std::vector<float>* logits) {
    if (p.stages.empty())
        return Status::invalid_state("pipeline: ステージが無い", __FILE__, __LINE__);
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t s = 0; s < p.stages.size(); ++s) {
        const Stage& st = p.stages[s];
        if (st.exec == nullptr || st.graphs.decode == nullptr)
            return Status::invalid_state("pipeline: ステージ " + std::to_string(s) + " が未設定",
                                         __FILE__, __LINE__);
        const bool use_prefill = (n > 1 && st.graphs.prefill != nullptr);
        const ir::Graph& graph = use_prefill ? *st.graphs.prefill : *st.graphs.decode;
        const ir::ValueId hidden_out =
            use_prefill ? st.graphs.hidden_out_prefill : st.graphs.hidden_out_decode;
        const bool first = (s == 0);
        const bool final = (s + 1 == p.stages.size());

        auto rs = st.exec->run_stage(graph, first ? tokens : nullptr,
                                     first ? nullptr : p.handoff.data(), n, base_pos);
        if (!rs.ok())
            return Status::invalid_state(
                "pipeline: ステージ " + std::to_string(s) + ": " + rs.message(), __FILE__,
                __LINE__);
        if (!final) {
            if (!hidden_out.valid())
                return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                                 " に hidden_out が無い",
                                             __FILE__, __LINE__);
            auto hb = st.exec->fetch_value_bf16(graph, hidden_out);
            if (!hb.ok())
                return Status::invalid_state("pipeline: ステージ " + std::to_string(s) +
                                                 " の hidden 読み出し: " + hb.status().message(),
                                             __FILE__, __LINE__);
            if (hb.value().size() > p.handoff.size())
                return Status::invalid_state("pipeline: hand-off バッファが小さい", __FILE__,
                                             __LINE__);
            std::copy(hb.value().begin(), hb.value().end(), p.handoff.begin());
        } else if (logits != nullptr) {
            auto lg = st.exec->fetch_logits(graph);
            if (!lg.ok()) return lg.status();
            *logits = std::move(lg.value());
        }
    }
    p.last_step_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    return Status::make_ok();
}

}  // namespace shelf::runtime::pipeline
