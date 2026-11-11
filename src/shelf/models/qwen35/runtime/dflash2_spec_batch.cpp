#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/models/qwen35/kernels/dflash2/candidate_selector.h>

#include <vector>
#include <type_traits>

namespace shelf::qwen35::runtime {

Result<DFlash2SpecBatchContext> create_dflash2_spec_batch_context(
    const std::vector<DFlash2GdnRankBinding>& bindings,
    uint32_t max_rows, uint32_t tree_budget) {
    if (max_rows == 0u) {
        return Status::invalid_argument("batch context max_rows is zero", __FILE__, __LINE__);
    }
    if (bindings.empty()) {
        return Status::invalid_argument("batch context requires a binding", __FILE__, __LINE__);
    }
    DFlash2SpecBatchContext ctx;
    ctx.max_rows = max_rows;
    ctx.tree_budget = tree_budget;
    ctx.history.resize(bindings.size());
    ctx.compact.resize(bindings.size());
    if (tree_budget != 0u) {
        ctx.tree.resize(bindings.size());
        ctx.tree_parent_host.resize(max_rows);
        ctx.tree_depth_host.resize(max_rows);
        ctx.tree_offsets_host.resize(static_cast<std::size_t>(max_rows) + 1u);
        ctx.tree_ancestors_host.resize(static_cast<std::size_t>(max_rows) * tree_budget);
        ctx.tree_rope_host.resize(max_rows);
        ctx.tree_committed_end_host.resize(max_rows);
        ctx.tree_replay_host.resize(max_rows);
    }
    {
        gpu::GpuArena& leader_arena = *bindings.front().arena;
        auto scope = shelf::gpu::ScopedDevice::create(bindings.front().device);
        if (!scope.ok()) return scope.status();
        auto verify_alloc = leader_arena.allocate_aligned(
            static_cast<std::size_t>(max_rows) * sizeof(int32_t), 256u);
        if (!verify_alloc.ok()) return verify_alloc.status();
        ctx.verify_token_ids_device =
            static_cast<int32_t*>(verify_alloc.release().data());
        auto decision_alloc = leader_arena.allocate_aligned(
            2u * static_cast<std::size_t>(max_rows) * sizeof(int32_t), 256u);
        if (!decision_alloc.ok()) return decision_alloc.status();
        ctx.decision_staging_device =
            static_cast<int32_t*>(decision_alloc.release().data());
    }
    for (std::size_t r = 0; r < bindings.size(); ++r) {
        gpu::GpuArena& arena = *bindings[r].arena;
        GdnStatePool& pool = *bindings[r].pool;
        auto scope = shelf::gpu::ScopedDevice::create(bindings[r].device);
        if (!scope.ok()) return scope.status();

        if (tree_budget != 0u) {
            DFlash2BatchTreeMetadata& tree = ctx.tree[r];
            const auto allocate = [&](std::size_t count, auto*& ptr) -> Status {
                auto allocation = arena.allocate_aligned(count * sizeof(*ptr), 256u);
                if (!allocation.ok()) return allocation.status();
                ptr = static_cast<std::remove_reference_t<decltype(*ptr)>*>(allocation.release().data());
                return Status::make_ok();
            };
            Status st = allocate(max_rows, tree.parent);
            if (st.ok()) st = allocate(max_rows, tree.depth);
            if (st.ok()) st = allocate(static_cast<std::size_t>(max_rows) + 1u, tree.ancestor_offsets);
            if (st.ok()) st = allocate(static_cast<std::size_t>(max_rows) * tree_budget, tree.ancestor_positions);
            if (st.ok()) st = allocate(max_rows, tree.rope_positions);
            if (st.ok()) st = allocate(max_rows, tree.committed_end);
            if (st.ok()) st = allocate(max_rows, tree.replay_rows);
            if (st.ok()) st = allocate(static_cast<std::size_t>(max_rows) *
                sizeof(shelf::kernel::DFlash2TreeProbabilities) / sizeof(float), tree.probability_scratch);
            if (!st.ok()) return st;
        }
        auto history = create_gdn_spec_history(arena, pool, max_rows, false);
        if (!history.ok()) return history.status();
        ctx.history[r] = history.release();
        ctx.history_bytes += static_cast<uint64_t>(max_rows) *
            static_cast<uint64_t>(ctx.history[r].conv_state_bytes +
                                  ctx.history[r].recurrent_state_bytes);

        const GdnStatePoolDeviceView view = pool.device_view();
        const uint64_t states = view.num_gdn_states;
        const uint64_t vdim = static_cast<uint64_t>(view.num_v_heads) * view.head_v;
        const uint64_t qkdim = static_cast<uint64_t>(view.num_v_heads) * view.head_k;
        const uint64_t dper = static_cast<uint64_t>(max_rows) * vdim;
        const uint64_t kper = static_cast<uint64_t>(max_rows) * qkdim;
        const uint64_t aper = static_cast<uint64_t>(max_rows) * view.num_v_heads;
        auto d_alloc = arena.allocate_aligned(states * dper * sizeof(float), 256u);
        if (!d_alloc.ok()) return d_alloc.status();
        auto k_alloc = arena.allocate_aligned(states * kper * sizeof(float), 256u);
        if (!k_alloc.ok()) return k_alloc.status();
        auto a_alloc = arena.allocate_aligned(states * aper * sizeof(float), 256u);
        if (!a_alloc.ok()) return a_alloc.status();
        ctx.compact[r].delta = static_cast<float*>(d_alloc.release().data());
        ctx.compact[r].k = static_cast<float*>(k_alloc.release().data());
        ctx.compact[r].a = static_cast<float*>(a_alloc.release().data());
        ctx.compact[r].delta_layer_stride = dper;
        ctx.compact[r].k_layer_stride = kper;
        ctx.compact[r].a_layer_stride = aper;
        ctx.compact_bytes += states * (dper + kper + aper) * sizeof(float);
    }
    ctx.initialized = true;
    return ctx;
}

Status shutdown_dflash2_spec_batch_context(DFlash2SpecBatchContext& context) noexcept {
    Status first_error = Status::make_ok();
    for (GdnSpecHistory& history : context.history) {
        const Status st = shutdown_gdn_spec_history(history);
        if (!st.ok() && first_error.ok()) first_error = st;
    }
    context.history.clear();
    context.compact.clear();
    context.tree.clear();
    context.tree_parent_host.clear();
    context.tree_depth_host.clear();
    context.tree_offsets_host.clear();
    context.tree_ancestors_host.clear();
    context.tree_rope_host.clear();
    context.tree_committed_end_host.clear();
    context.tree_replay_host.clear();
    context.max_rows = 0u;
    context.tree_budget = 0u;
    context.history_bytes = 0u;
    context.compact_bytes = 0u;
    context.verify_token_ids_device = nullptr;
    context.decision_staging_device = nullptr;
    context.initialized = false;
    return first_error;
}

}  // namespace shelf::qwen35::runtime
