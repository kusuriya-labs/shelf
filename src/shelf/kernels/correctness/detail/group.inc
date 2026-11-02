#pragma once

namespace ps::kernel::detail {

__device__ __forceinline__ ProgramStatus exec_group_mean(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    const uint32_t group = b.group_size;
    if (group == 0u || out.features == 0u) return ProgramStatus::INVALID_BINDING;
    if (in.features != out.features * group) return ProgramStatus::INVALID_BINDING;
    const uint32_t rows = value_rows(env, out);
    const uint32_t f = out.features;
    const uint32_t stride = global_stride(env);
    const size_t total = static_cast<size_t>(rows) * f;
    const float inv_group = 1.0f / static_cast<float>(group);
    for (size_t idx = global_tid(env); idx < total; idx += stride) {
        const size_t r = idx / f;
        const uint32_t c = static_cast<uint32_t>(idx % f);
        float acc = 0.f;
        const size_t base = r * in.row_stride + c;
        for (uint32_t g = 0; g < group; ++g) {
            acc += load_elem(in, base + static_cast<size_t>(g) * f);
        }
        store_elem(out, r * out.row_stride + c, acc * inv_group);
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ ProgramStatus exec_group_broadcast_mul(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in_a{};
    ResolvedValue in_b{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in_a)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[1], in_b)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    const uint32_t group = b.group_size;
    if (group == 0u || in_a.features == 0u) return ProgramStatus::INVALID_BINDING;
    if (in_b.features != group || out.features != in_a.features * group)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t rows = value_rows(env, out);
    const uint32_t f = in_a.features;
    const uint32_t of = out.features;
    const uint32_t stride = global_stride(env);
    const size_t total = static_cast<size_t>(rows) * of;
    for (size_t idx = global_tid(env); idx < total; idx += stride) {
        const size_t r = idx / of;
        const uint32_t j = static_cast<uint32_t>(idx % of);
        const uint32_t g = j / f;
        const uint32_t h = j % f;
        const float av = load_elem(in_a, r * in_a.row_stride + h);
        const float bv = load_elem(in_b, r * in_b.row_stride + g);
        store_elem(out, r * out.row_stride + j, av * bv);
    }
    return ProgramStatus::COMPLETE;
}

}  // namespace ps::kernel::detail
