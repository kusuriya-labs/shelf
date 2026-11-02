#pragma once

namespace ps::kernel::detail {

__device__ __forceinline__ ProgramStatus exec_embedding(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    if (env.context == nullptr)
        return ProgramStatus::INVALID_BINDING;
    ResolvedValue out{};
    if (!resolve_value(env, b.output_slots[0], out))
        return ProgramStatus::INVALID_BINDING;
    if (b.weight_index >= env.program->weight_slot_count || env.program->weight_table == nullptr)
        return ProgramStatus::INVALID_BINDING;
    const auto* wtable = static_cast<const WeightSlot*>(env.program->weight_table);
    const WeightSlot we = wtable[b.weight_index];
    const uint32_t hidden = out.features;
    const uint32_t rows = value_rows(env, out);
    if (we.encoding == 0u) {
        if (we.device_ptr == nullptr)
            return ProgramStatus::INVALID_BINDING;
        const bf16* table = static_cast<const bf16*>(we.device_ptr);
        for (size_t idx = global_tid(env); idx < static_cast<size_t>(rows) * hidden;
             idx += global_stride(env)) {
            const uint32_t m = static_cast<uint32_t>(idx / hidden);
            const uint32_t h = static_cast<uint32_t>(idx % hidden);
            const uint32_t tok =
                env.context->token_ids ? static_cast<uint32_t>(env.context->token_ids[m]) : 0u;
            static_cast<bf16*>(out.ptr)[static_cast<size_t>(m) * out.row_stride + h] =
                table[static_cast<size_t>(tok) * hidden + h];
        }
        return ProgramStatus::COMPLETE;
    }
    return ProgramStatus::INVALID_PROGRAM;
}

}  // namespace ps::kernel::detail
