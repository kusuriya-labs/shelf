#pragma once

namespace ps::kernel::detail {

__device__ __forceinline__ ProgramStatus exec_row_norm_like(
    ModelDispatchEnv& env, const DispatchBinding& b, bool l2_mode) {
    ResolvedValue in{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    if (in.features != out.features) return ProgramStatus::INVALID_BINDING;
    const uint32_t f = out.features;
    const uint32_t group = b.group_size != 0 && b.group_size <= f ? b.group_size : f;
    if (f % group != 0) return ProgramStatus::INVALID_BINDING;
    uint32_t weight_elements = 0;
    ValueDType weight_dtype = ValueDType::BF16;
    const void* weight_raw =
        resolve_parameter(env, b.parameter_index0, &weight_elements, &weight_dtype);
    const bool want_weight = !l2_mode && b.parameter_index0 != ::ps::runtime::kNoParameter;
    if (want_weight && weight_raw == nullptr) return ProgramStatus::INVALID_BINDING;
    const bool weight_per_row = want_weight && weight_elements == f;
    const bool weight_per_group = want_weight && weight_elements == group;
    if (want_weight && !weight_per_row && !weight_per_group)
        return ProgramStatus::INVALID_BINDING;
    const bool one_plus = b.flags == 0u;
    const uint32_t rows = value_rows(env, out);
    for (uint32_t r = blockIdx.x; r < rows; r += gridDim.x) {
        for (uint32_t g0 = 0; g0 < f; g0 += group) {
            if (threadIdx.x == 0) {
                float sum = 0.f;
                for (uint32_t h = 0; h < group; ++h) {
                    const float x =
                        load_elem(in, static_cast<size_t>(r) * in.row_stride + g0 + h);
                    sum += x * x;
                }
                float inv;
                if (l2_mode) {
                    inv = rsqrtf(sum + b.scalar_a);
                } else {
                    inv = rsqrtf(sum / static_cast<float>(group) + b.scalar_a);
                }
                env.scratch[r] = inv;
            }
            __syncthreads();
            const float inv = env.scratch[r];
            for (uint32_t h = threadIdx.x; h < group; h += blockDim.x) {
                const size_t idx = static_cast<size_t>(r) * in.row_stride + g0 + h;
                float y = load_elem(in, idx) * inv;
                if (!l2_mode && in.dtype == ValueDType::F32)
                    y = __bfloat162float(__float2bfloat16(y));
                if (want_weight) {
                    const uint32_t widx = weight_per_row ? (g0 + h) : h;
                    const float w = load_parameter_elem(weight_raw, weight_dtype, widx);
                    y *= one_plus ? (1.f + w) : w;
                }
                store_elem(out, static_cast<size_t>(r) * out.row_stride + g0 + h, y);
            }
            __syncthreads();
        }
    }
    return ProgramStatus::COMPLETE;
}

}  // namespace ps::kernel::detail
