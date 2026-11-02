#pragma once

namespace ps::kernel::detail {

__device__ __forceinline__ ProgramStatus exec_linear_weight(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in))
        return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out))
        return ProgramStatus::INVALID_BINDING;
    if (b.weight_index >= env.program->weight_slot_count ||
        env.program->weight_table == nullptr)
        return ProgramStatus::INVALID_BINDING;
    const auto* wtable = static_cast<const WeightSlot*>(env.program->weight_table);
    const WeightSlot e = wtable[b.weight_index];
    const uint32_t out_features = e.rows;
    const uint32_t col_offset = e.output_col_offset;
    const uint32_t expected_out =
        e.output_col_total != 0u ? e.output_col_total : e.rows;
    const uint32_t kdim = e.cols;
    if (out.features != expected_out || kdim != in.features)
        return ProgramStatus::INVALID_BINDING;
    if (e.output_col_total != 0u && col_offset + out_features > e.output_col_total)
        return ProgramStatus::INVALID_BINDING;
    if (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32)
        return ProgramStatus::INVALID_BINDING;
    if (e.encoding == 0u) {
        if (e.device_ptr == nullptr || e.codes != nullptr || e.scales != nullptr)
            return ProgramStatus::INVALID_PROGRAM;
        const bf16* w = static_cast<const bf16*>(e.device_ptr);
        const uint32_t rows = value_rows(env, out);
        const size_t total = static_cast<size_t>(rows) * out_features;
        const uint32_t stride = global_stride(env);
        for (size_t idx = global_tid(env); idx < total; idx += stride) {
            const uint32_t m = static_cast<uint32_t>(idx / out_features);
            const uint32_t o = static_cast<uint32_t>(idx % out_features);
            float acc = 0.f;
            const bf16* wrow = w + static_cast<size_t>(o) * kdim;
            const size_t xin_base = static_cast<size_t>(m) * in.row_stride;
            for (uint32_t kk = 0; kk < kdim; ++kk)
                acc += __bfloat162float(wrow[kk]) * load_elem(in, xin_base + kk);
            store_elem(out, static_cast<size_t>(m) * out.row_stride + col_offset + o,
                       acc);
        }
        return ProgramStatus::COMPLETE;
    }
    return ProgramStatus::INVALID_PROGRAM;
}

__device__ __forceinline__ ProgramStatus exec_linear_block_scaled(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue out{};
    if (!resolve_value(env, b.output_slots[0], out))
        return ProgramStatus::INVALID_BINDING;
    if (b.weight_index >= env.program->weight_slot_count ||
        env.program->weight_table == nullptr)
        return ProgramStatus::INVALID_BINDING;
    if (b.workspace_slot >= env.program->workspace_range_count)
        return ProgramStatus::INVALID_PROGRAM;
    if (b.activation_kp < 32u || (b.activation_kp & 31u) != 0u ||
        b.activation_max_rows == 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const auto* wtable = static_cast<const WeightSlot*>(env.program->weight_table);
    const WeightSlot e = wtable[b.weight_index];
    if (e.encoding != 1u)
        return ProgramStatus::INVALID_PROGRAM;
    if (e.codes == nullptr || e.scales == nullptr)
        return ProgramStatus::INVALID_PROGRAM;
    if (e.weight_scale_group != 128u)
        return ProgramStatus::INVALID_PROGRAM;
    if (e.rows != out.features)
        return ProgramStatus::INVALID_BINDING;
    if (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t kdim = e.cols;
    const uint32_t wkp = e.k_padded;
    if (wkp < kdim || (wkp & 127u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint32_t sstride = e.storage_scale_stride_bytes;
    if (sstride != (wkp >> 7) * 4u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint32_t akp = b.activation_kp;
    if (akp != wkp)
        return ProgramStatus::INVALID_PROGRAM;
    const uint32_t nb = wkp >> 5;
    const uint32_t code_stride = b.activation_code_stride;
    const uint32_t scale_stride = b.activation_scale_stride;
    if (code_stride < akp || (code_stride & 63u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    if (scale_stride < 4u || (scale_stride & 15u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint64_t ws_off = env.program->workspace_ranges[b.workspace_slot].offset;
    const uint8_t* acodes = reinterpret_cast<const uint8_t*>(env.workspace_base + ws_off);
    const uint8_t* asbase = reinterpret_cast<const uint8_t*>(env.workspace_base + ws_off +
        a8_ws_scales_offset(code_stride, b.activation_max_rows));
    const uint32_t rows = value_rows(env, out);
    if (rows > b.activation_max_rows)
        return ProgramStatus::INVALID_PROGRAM;
    if (rows == 0u)
        return ProgramStatus::COMPLETE;
    const uint8_t* wcodes = static_cast<const uint8_t*>(e.codes);
    const uint8_t* wscales = static_cast<const uint8_t*>(e.scales);
    const uint32_t out_features = e.rows;
    const size_t total = static_cast<size_t>(rows) * out_features;
    const uint32_t stride = global_stride(env);
    for (size_t idx = global_tid(env); idx < total; idx += stride) {
        const uint32_t m = static_cast<uint32_t>(idx / out_features);
        const uint32_t o = static_cast<uint32_t>(idx % out_features);
        const uint8_t* wtile = wcodes +
            static_cast<size_t>(o >> 4) * (static_cast<size_t>(wkp) * 16u);
        const float* wsrow_fp8 = reinterpret_cast<const float*>(wscales) +
            static_cast<size_t>(o >> 7) * (wkp >> 7);
        const uint8_t* atile = acodes +
            static_cast<size_t>(m >> 4) * (static_cast<size_t>(code_stride) * 16u);
        const float* asrow = reinterpret_cast<const float*>(
            asbase + static_cast<size_t>(m) * scale_stride);
        const uint32_t o_slot = o & 15u;
        const uint32_t m_slot = m & 15u;
        float acc = 0.f;
        for (uint32_t ib = 0; ib < nb; ++ib) {
            const float ws = wsrow_fp8[ib >> 2];
            const float as = asrow[ib >> 2];
            float p = 0.f;
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t kc = ib * 32u + j;
                const size_t off = static_cast<size_t>(kc >> 6) *
                                       ::ps::quantization::fp8::kFragBlockBytes +
                                   ::ps::quantization::fp8::fp8_frag_offset(o_slot, kc & 63u);
                const size_t aoff = static_cast<size_t>(kc >> 6) *
                                        ::ps::quantization::fp8::kFragBlockBytes +
                                    ::ps::quantization::fp8::fp8_frag_offset(m_slot, kc & 63u);
                p += ::ps::quantization::codec::device::e4m3_decode_f32(wtile[off]) *
                     ::ps::quantization::codec::device::e4m3_decode_f32(atile[aoff]);
            }
            acc += p * ws * as;
        }
        store_elem(out, idx, acc);
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ float act_load_f32(const void* row, uint32_t i, bool f32_in) {
    const uint8_t* base = static_cast<const uint8_t*>(row);
    if (f32_in) {
        float v;
        __builtin_memcpy(&v, base + i * sizeof(float), sizeof(float));
        return v;
    }
    bf16 h;
    __builtin_memcpy(&h, base + i * sizeof(uint16_t), sizeof(uint16_t));
    return __bfloat162float(h);
}


__device__ __forceinline__ ProgramStatus exec_activation_quantize_fp8(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    if (!resolve_value(env, b.input_slots[0], in))
        return ProgramStatus::INVALID_BINDING;
    if (in.dtype != ValueDType::BF16 && in.dtype != ValueDType::F32)
        return ProgramStatus::INVALID_PROGRAM;
    if (b.workspace_slot >= env.program->workspace_range_count)
        return ProgramStatus::INVALID_PROGRAM;
    if (in.features == 0u)
        return ProgramStatus::INVALID_PROGRAM;
    if (b.activation_kp < in.features || (b.activation_kp & 31u) != 0u ||
        b.activation_max_rows == 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint32_t rows = value_rows(env, in);
    if (rows > b.activation_max_rows)
        return ProgramStatus::INVALID_PROGRAM;
    if (rows == 0u)
        return ProgramStatus::COMPLETE;
    const uint32_t k = in.features;
    const uint32_t kp = b.activation_kp;
    const uint32_t code_stride = b.activation_code_stride;
    const uint32_t scale_stride = b.activation_scale_stride;
    if (code_stride < kp || (code_stride & 63u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    if (scale_stride < 4u || (scale_stride & 15u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const bool f32_in = in.dtype == ValueDType::F32;
    const uint64_t ws_off = env.program->workspace_ranges[b.workspace_slot].offset;
    uint8_t* acodes = reinterpret_cast<uint8_t*>(env.workspace_base + ws_off);
    uint8_t* ascales = reinterpret_cast<uint8_t*>(env.workspace_base + ws_off +
        a8_ws_scales_offset(code_stride, b.activation_max_rows));
    const uint32_t stride = global_stride(env);
    const void* xin = in.ptr;
    const uint32_t groups = (kp + 127u) / 128u;
    const size_t row_bytes =
        static_cast<size_t>(in.row_stride) *
        ::ps::runtime::value_dtype_bytes(in.dtype);
    for (uint32_t m = global_tid(env); m < rows; m += stride) {
        const void* xrow = static_cast<const uint8_t*>(xin) +
            static_cast<size_t>(m) * row_bytes;
        uint8_t* codes = acodes +
            static_cast<size_t>(m >> 4) * (code_stride * 16u);
        const uint32_t m_slot = m & 15u;
        float* row_scales =
            reinterpret_cast<float*>(ascales + static_cast<size_t>(m) * scale_stride);
        for (uint32_t g = 0u; g < groups; ++g) {
            const uint32_t i0 = g * 128u;
            const uint32_t k_end = i0 + 128u < k ? i0 + 128u : k;
            float peak = 0.f;
            for (uint32_t i = i0; i < k_end; ++i) {
                const float v = act_load_f32(xrow, i, f32_in);
                if (std::isfinite(v) && std::fabs(v) > peak) peak = std::fabs(v);
            }
            const float scale = peak > 0.f ? peak / 448.0f : 0.f;
            const float inv = scale > 0.f ? 1.0f / scale : 0.f;
            row_scales[g] = scale;
            const uint32_t kp_end = i0 + 128u < kp ? i0 + 128u : kp;
            for (uint32_t i = i0; i < kp_end; ++i) {
                float v = 0.f;
                if (i < k) {
                    const float x = act_load_f32(xrow, i, f32_in);
                    v = std::isfinite(x) ? x : 0.f;
                }
                codes[static_cast<size_t>(i >> 6) *
                          ::ps::quantization::fp8::kFragBlockBytes +
                      ::ps::quantization::fp8::fp8_frag_offset(m_slot, i & 63u)] =
                    scale > 0.f
                        ? ::ps::quantization::codec::device::e4m3_encode_u8(v * inv)
                        : 0u;
            }
        }
    }
    return ProgramStatus::COMPLETE;
}

}  // namespace ps::kernel::detail
