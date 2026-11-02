#pragma once

namespace ps::kernel::detail {

__device__ __forceinline__ ProgramStatus exec_unary(
    ModelDispatchEnv& env, KernelId kind, const DispatchBinding& b) {
    ResolvedValue in{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    if (in.features != out.features) return ProgramStatus::INVALID_BINDING;
    const uint32_t rows = value_rows(env, out);
    const uint32_t f = out.features;
    const uint32_t stride = global_stride(env);
    const size_t total = static_cast<size_t>(rows) * f;
    for (size_t idx = global_tid(env); idx < total; idx += stride) {
        const size_t r = idx / f;
        const size_t c = idx % f;
        const float x = load_elem(in, r * in.row_stride + c);
        float y = x;
        if (kind == KernelId::SILU) y = x / (1.f + expf(-x));
        else if (kind == KernelId::SIGMOID) y = 1.f / (1.f + expf(-x));
        else if (kind == KernelId::SCALE) y = x * b.scalar_a;
        store_elem(out, r * out.row_stride + c, y);
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ ProgramStatus exec_binary(
    ModelDispatchEnv& env, KernelId kind, const DispatchBinding& b) {
    ResolvedValue in_a{};
    ResolvedValue in_b{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in_a)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[1], in_b)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    if (in_a.features != out.features || in_b.features != out.features)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t rows = value_rows(env, out);
    const uint32_t f = out.features;
    const uint32_t stride = global_stride(env);
    const size_t total = static_cast<size_t>(rows) * f;
    for (size_t idx = global_tid(env); idx < total; idx += stride) {
        const size_t r = idx / f;
        const size_t c = idx % f;
        const float av = load_elem(in_a, r * in_a.row_stride + c);
        const float bv = load_elem(in_b, r * in_b.row_stride + c);
        float y = 0.f;
        if (kind == KernelId::RESIDUAL_ADD) y = av + bv;
        else if (kind == KernelId::MUL) y = av * bv;
        else if (kind == KernelId::SWIGLU) y = bv * (av / (1.f + expf(-av)));
        store_elem(out, r * out.row_stride + c, y);
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ ProgramStatus exec_concat(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in_a{};
    ResolvedValue in_b{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in_a)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[1], in_b)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    const uint32_t fa = b.group_size;
    const uint32_t fb = in_b.features;
    if (in_a.features != fa) return ProgramStatus::INVALID_BINDING;
    if (out.features != fa + fb) return ProgramStatus::INVALID_BINDING;
    const uint32_t rows = value_rows(env, out);
    const uint32_t f = out.features;
    const uint32_t stride = global_stride(env);
    const size_t total = static_cast<size_t>(rows) * f;
    for (size_t idx = global_tid(env); idx < total; idx += stride) {
        const size_t r = idx / f;
        const uint32_t c = static_cast<uint32_t>(idx % f);
        const float y = (c < fa)
            ? load_elem(in_a, r * in_a.row_stride + c)
            : load_elem(in_b, r * in_b.row_stride + (c - fa));
        store_elem(out, r * out.row_stride + c, y);
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ ProgramStatus exec_split(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    ResolvedValue o0{};
    ResolvedValue o1{};
    if (!resolve_value(env, b.input_slots[0], in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], o0)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[1], o1)) return ProgramStatus::INVALID_BINDING;
    if (o0.features != o1.features || in.features < o0.features + o1.features)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t rows = value_rows(env, o0);
    const uint32_t f0 = o0.features;
    const uint32_t stride = global_stride(env);
    const size_t total = static_cast<size_t>(rows) * f0;
    if (b.flags == static_cast<uint32_t>(ps::runtime::SplitLayout::INTERLEAVED_HEADS)) {
        const uint32_t hd = b.group_size;
        if (hd == 0u || f0 % hd != 0u || in.features < 2u * hd)
            return ProgramStatus::INVALID_BINDING;
        for (size_t idx = global_tid(env); idx < total; idx += stride) {
            const uint32_t m = static_cast<uint32_t>(idx / f0);
            const uint32_t j = static_cast<uint32_t>(idx % f0);
            const size_t base = static_cast<size_t>(m) * in.row_stride +
                                (j / hd) * 2u * hd + (j % hd);
            store_elem(o0, static_cast<size_t>(m) * o0.row_stride + j, load_elem(in, base));
            store_elem(o1, static_cast<size_t>(m) * o1.row_stride + j, load_elem(in, base + hd));
        }
        return ProgramStatus::COMPLETE;
    }
    for (size_t idx = global_tid(env); idx < total; idx += stride) {
        const uint32_t m = static_cast<uint32_t>(idx / f0);
        const uint32_t j = static_cast<uint32_t>(idx % f0);
        const size_t src = static_cast<size_t>(m) * in.row_stride + j;
        store_elem(o0, static_cast<size_t>(m) * o0.row_stride + j, load_elem(in, src));
        store_elem(o1, static_cast<size_t>(m) * o1.row_stride + j, load_elem(in, src + f0));
    }
    return ProgramStatus::COMPLETE;
}

}  // namespace ps::kernel::detail
