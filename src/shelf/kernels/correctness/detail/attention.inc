#pragma once

namespace shelf::kernel::detail {

__device__ __forceinline__ ProgramStatus exec_rope(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    if (in.features != out.features) return ProgramStatus::INVALID_BINDING;
    const uint32_t f = out.features;
    const uint32_t head_dim = b.group_size != 0 ? b.group_size : f;
    if (f % head_dim != 0) return ProgramStatus::INVALID_BINDING;
    const uint32_t rotary =
        b.rotary_dim != 0 && (b.rotary_dim & 1u) == 0 && b.rotary_dim <= head_dim
            ? b.rotary_dim
            : (head_dim & ~1u);
    const uint32_t half = rotary / 2;
    const uint32_t rows = value_rows(env, out);
    const double theta_base = static_cast<double>(b.scalar_b > 0.f ? b.scalar_b : 10000000.f);
    const uint32_t stride = global_stride(env);
    const size_t total = static_cast<size_t>(rows) * f;
    for (size_t idx = global_tid(env); idx < total; idx += stride) {
        const uint32_t r = static_cast<uint32_t>(idx / f);
        const uint32_t col = static_cast<uint32_t>(idx % f);
        const uint32_t head_col = col % head_dim;
        const size_t row_base = static_cast<size_t>(r) * in.row_stride;
        float x = load_elem(in, row_base + col);
        float y = x;
        if (head_col < rotary) {
            const uint32_t rotary_col = head_col % half;
            const uint32_t partner_head_col =
                head_col < half ? head_col + half : head_col - half;
            const uint32_t position =
                (env.context && env.context->rope_positions)
                    ? env.context->rope_positions[r]
                    : (env.context && env.context->row_positions
                           ? env.context->row_positions[r]
                           : r);
            const double* inv_freq_table =
                env.context != nullptr ? env.context->rope_inv_freq : nullptr;
            const double freq = inv_freq_table != nullptr
                ? inv_freq_table[rotary_col]
                : 1.0 / pow(theta_base,
                            static_cast<double>(rotary_col) / static_cast<double>(half));
            const float attention_factor =
                b.scalar_a > 0.0f ? b.scalar_a : 1.0f;
            const double angle = static_cast<double>(position) * freq;
            const float cs = static_cast<float>(cos(angle)) * attention_factor;
            const float sn = static_cast<float>(sin(angle)) * attention_factor;
            const float partner =
                load_elem(in, row_base + (col - head_col) + partner_head_col);
            y = head_col < half ? x * cs - partner * sn : x * cs + partner * sn;
        }
        store_elem(out, static_cast<size_t>(r) * out.row_stride + col, y);
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ ProgramStatus exec_kv_append(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    if (env.context == nullptr || env.qwen_state == nullptr)
        return ProgramStatus::INVALID_PROGRAM;
    const auto* sb = find_state(env, b.descriptor_slot);
    if (sb == nullptr ||
        sb->kind != static_cast<uint8_t>(shelf::runtime::PrimitiveStateKind::KV_CACHE))
        return ProgramStatus::INVALID_BINDING;
    const uint32_t layer = sb->state_index;
    const auto& bt = env.qwen_state->block_table;
    ResolvedValue k_in{};
    ResolvedValue v_in{};
    if (!resolve_value(env, b.input_slots[0], k_in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[1], v_in)) return ProgramStatus::INVALID_BINDING;
    if (k_in.dtype != ValueDType::BF16 || v_in.dtype != ValueDType::BF16)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t rows = env.context->actual_rows;
    if (bt.block_tables == nullptr || bt.block_table_stride == 0)
        return ProgramStatus::INVALID_BINDING;
    const bool fp8 = env.qwen_state->kv_dtype == 1u;
    if (fp8) {
        const auto& kv = env.qwen_state->kv_fp8;
        if (layer >= kv.num_attention_layers || kv.k_pool == nullptr || kv.v_pool == nullptr ||
            kv.k_scale_pool == nullptr || kv.v_scale_pool == nullptr || kv.page_tokens == 0 ||
            k_in.features != kv.kv_heads * kv.head_dim)
            return ProgramStatus::INVALID_BINDING;
        for (uint32_t row = global_tid(env); row < rows; row += global_stride(env)) {
            if (env.context->row_sequence_slots[row] >= bt.max_sequences)
                return ProgramStatus::INVALID_BINDING;
            const uint32_t slot = env.context->row_sequence_slots[row];
            const uint32_t pos = env.context->row_positions[row];
            const uint32_t lb = pos / kv.page_tokens;
            const uint32_t off = pos % kv.page_tokens;
            if (lb >= bt.block_table_stride) return ProgramStatus::INVALID_BINDING;
            const uint32_t page = bt.block_tables[slot * bt.block_table_stride + lb];
            if (page >= kv.num_pages) return ProgramStatus::INVALID_BINDING;
            const size_t base = static_cast<size_t>(layer) * kv.elems_per_layer +
                                static_cast<size_t>(page) * kv.elems_per_page +
                                static_cast<size_t>(off) * kv.elems_per_token;
            const auto* ks = static_cast<const bf16*>(k_in.ptr) +
                             static_cast<size_t>(row) * k_in.row_stride;
            const auto* vs = static_cast<const bf16*>(v_in.ptr) +
                             static_cast<size_t>(row) * v_in.row_stride;
            for (uint32_t h = 0; h < kv.kv_heads; ++h) {
                const bf16* kh = ks + static_cast<size_t>(h) * kv.head_dim;
                const bf16* vh = vs + static_cast<size_t>(h) * kv.head_dim;
                float k_amax = 0.f, v_amax = 0.f;
                for (uint32_t i = 0; i < kv.head_dim; ++i) {
                    const float kf = __bfloat162float(kh[i]);
                    const float vf = __bfloat162float(vh[i]);
                    const float ka = kf < 0.f ? -kf : kf;
                    const float va = vf < 0.f ? -vf : vf;
                    k_amax = ka > k_amax ? ka : k_amax;
                    v_amax = va > v_amax ? va : v_amax;
                }
                constexpr float kFp8Max = 448.0f;
                const float k_scale = k_amax == 0.f ? 1.f : k_amax / kFp8Max;
                const float v_scale = v_amax == 0.f ? 1.f : v_amax / kFp8Max;
                const size_t si = ((static_cast<size_t>(layer) * kv.num_pages + page) *
                                       kv.page_tokens +
                                   off) *
                                      kv.kv_heads +
                                  h;
                kv.k_scale_pool[si] = k_scale;
                kv.v_scale_pool[si] = v_scale;
                const size_t hb = base + static_cast<size_t>(h) * kv.head_dim;
                for (uint32_t i = 0; i < kv.head_dim; ++i) {
                    const float kf = __bfloat162float(kh[i]);
                    const float vf = __bfloat162float(vh[i]);
                    kv.k_pool[hb + i] =
                        __hip_cvt_float_to_fp8(kf / k_scale, __HIP_SATFINITE, __HIP_E4M3);
                    kv.v_pool[hb + i] =
                        __hip_cvt_float_to_fp8(vf / v_scale, __HIP_SATFINITE, __HIP_E4M3);
                }
            }
        }
        return ProgramStatus::COMPLETE;
    }
    const auto& kv = env.qwen_state->kv_bf16;
    if (layer >= kv.num_attention_layers || kv.k_pool == nullptr || kv.v_pool == nullptr ||
        kv.page_tokens == 0 || k_in.features != kv.kv_heads * kv.head_dim)
        return ProgramStatus::INVALID_BINDING;
    for (uint32_t row = global_tid(env); row < rows; row += global_stride(env)) {
        if (env.context->row_sequence_slots[row] >= bt.max_sequences)
            return ProgramStatus::INVALID_BINDING;
        const uint32_t slot = env.context->row_sequence_slots[row];
        const uint32_t pos = env.context->row_positions[row];
        const uint32_t lb = pos / kv.page_tokens;
        const uint32_t off = pos % kv.page_tokens;
        if (lb >= bt.block_table_stride) return ProgramStatus::INVALID_BINDING;
        const uint32_t page = bt.block_tables[slot * bt.block_table_stride + lb];
        if (page >= kv.num_pages) return ProgramStatus::INVALID_BINDING;
        const size_t base = static_cast<size_t>(layer) * kv.elems_per_layer +
                            static_cast<size_t>(page) * kv.elems_per_page +
                            static_cast<size_t>(off) * kv.elems_per_token;
        const auto* ks = static_cast<const bf16*>(k_in.ptr) +
                         static_cast<size_t>(row) * k_in.row_stride;
        const auto* vs = static_cast<const bf16*>(v_in.ptr) +
                         static_cast<size_t>(row) * v_in.row_stride;
        auto* kd = reinterpret_cast<bf16*>(kv.k_pool) + base;
        auto* vd = reinterpret_cast<bf16*>(kv.v_pool) + base;
        const auto* ksp = reinterpret_cast<const bf16*>(ks);
        const auto* vsp = reinterpret_cast<const bf16*>(vs);
        for (uint32_t e = 0; e < kv.elems_per_token; ++e) {
            kd[e] = ksp[e];
            vd[e] = vsp[e];
        }
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ ProgramStatus exec_paged_attention(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    if (env.context == nullptr || env.qwen_state == nullptr)
        return ProgramStatus::INVALID_PROGRAM;
    const auto* sb = find_state(env, b.descriptor_slot);
    if (sb == nullptr ||
        sb->kind != static_cast<uint8_t>(shelf::runtime::PrimitiveStateKind::KV_CACHE))
        return ProgramStatus::INVALID_BINDING;
    const uint32_t layer = sb->state_index;
    const auto& bt = env.qwen_state->block_table;
    ResolvedValue q_in{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], q_in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    if (q_in.dtype != ValueDType::BF16 ||
        (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32))
        return ProgramStatus::INVALID_BINDING;
    const uint32_t head_dim = b.group_size;
    if (head_dim == 0u || head_dim > 256u || q_in.features % head_dim != 0u)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t num_q = q_in.features / head_dim;
    const float scale = b.scalar_a;
    const uint32_t rows = env.context->actual_rows;
    const bool fp8 = env.qwen_state->kv_dtype == 1u;
    uint32_t kv_heads = 0;
    uint32_t kv_head_dim = 0;
    if (fp8) {
        kv_heads = env.qwen_state->kv_fp8.kv_heads;
        kv_head_dim = env.qwen_state->kv_fp8.head_dim;
    } else {
        kv_heads = env.qwen_state->kv_bf16.kv_heads;
        kv_head_dim = env.qwen_state->kv_bf16.head_dim;
    }
    if (kv_heads == 0u || num_q % kv_heads != 0u || kv_head_dim != head_dim)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t group = num_q / kv_heads;
    if (bt.block_tables == nullptr || bt.block_table_stride == 0)
        return ProgramStatus::INVALID_BINDING;
    const size_t total = static_cast<size_t>(rows) * num_q;
    for (size_t idx = global_tid(env); idx < total; idx += global_stride(env)) {
        const uint32_t row = static_cast<uint32_t>(idx / num_q);
        const uint32_t h = static_cast<uint32_t>(idx % num_q);
        if (env.context->row_sequence_slots[row] >= bt.max_sequences)
            return ProgramStatus::INVALID_BINDING;
        const uint32_t slot = env.context->row_sequence_slots[row];
        const uint32_t kv_head = h / group;
        const uint32_t visible = env.context->row_positions[row] + 1u;
        const uint32_t* block_table =
            bt.block_tables + static_cast<size_t>(slot) * bt.block_table_stride;
        const bf16* q = static_cast<const bf16*>(q_in.ptr) +
                        static_cast<size_t>(row) * q_in.row_stride +
                        static_cast<size_t>(h) * head_dim;
        float qv[256];
        for (uint32_t i = 0; i < head_dim; ++i) qv[i] = __bfloat162float(q[i]);
        float max_score = -3.402823466e+38f;
        float sum_exp = 0.f;
        float acc[256];
        for (uint32_t i = 0; i < head_dim; ++i) acc[i] = 0.f;
        if (!fp8) {
            const auto& kv = env.qwen_state->kv_bf16;
            if (layer >= kv.num_attention_layers) return ProgramStatus::INVALID_BINDING;
            for (uint32_t pass = 0; pass < 2; ++pass) {
                sum_exp = 0.f;
                for (uint32_t i = 0; i < head_dim; ++i) acc[i] = 0.f;
                for (uint32_t p = 0; p < visible; ++p) {
                    const uint32_t lb = p / kv.page_tokens;
                    const uint32_t off = p % kv.page_tokens;
                    if (lb >= bt.block_table_stride) return ProgramStatus::INVALID_BINDING;
                    const uint32_t page_id = block_table[lb];
                    if (page_id >= kv.num_pages) return ProgramStatus::INVALID_BINDING;
                    const size_t kb = static_cast<size_t>(layer) * kv.elems_per_layer +
                                      static_cast<size_t>(page_id) * kv.elems_per_page +
                                      static_cast<size_t>(off) * kv.elems_per_token +
                                      static_cast<size_t>(kv_head) * head_dim;
                    float sc = 0.f;
                    const bf16* kb16 = reinterpret_cast<const bf16*>(kv.k_pool);
                    const bf16* vb16 = reinterpret_cast<const bf16*>(kv.v_pool);
                    for (uint32_t i = 0; i < head_dim; ++i)
                        sc += qv[i] * __bfloat162float(kb16[kb + i]);
                    sc *= scale;
                    if (pass == 0) {
                        if (sc > max_score) max_score = sc;
                    } else {
                        const float pv = __expf(sc - max_score);
                        sum_exp += pv;
                        for (uint32_t i = 0; i < head_dim; ++i)
                            acc[i] += pv * __bfloat162float(vb16[kb + i]);
                    }
                }
            }
        } else {
            const auto& kv = env.qwen_state->kv_fp8;
            if (layer >= kv.num_attention_layers) return ProgramStatus::INVALID_BINDING;
            for (uint32_t pass = 0; pass < 2; ++pass) {
                sum_exp = 0.f;
                for (uint32_t i = 0; i < head_dim; ++i) acc[i] = 0.f;
                for (uint32_t p = 0; p < visible; ++p) {
                    const uint32_t lb = p / kv.page_tokens;
                    const uint32_t off = p % kv.page_tokens;
                    if (lb >= bt.block_table_stride) return ProgramStatus::INVALID_BINDING;
                    const uint32_t page_id = block_table[lb];
                    if (page_id >= kv.num_pages) return ProgramStatus::INVALID_BINDING;
                    const size_t kb = static_cast<size_t>(layer) * kv.elems_per_layer +
                                      static_cast<size_t>(page_id) * kv.elems_per_page +
                                      static_cast<size_t>(off) * kv.elems_per_token +
                                      static_cast<size_t>(kv_head) * head_dim;
                    const size_t si = ((static_cast<size_t>(layer) * kv.num_pages + page_id) *
                                           kv.page_tokens +
                                       off) *
                                          kv.kv_heads +
                                      kv_head;
                    const float k_scale = kv.k_scale_pool[si];
                    const float v_scale = kv.v_scale_pool[si];
                    float sc = 0.f;
                    for (uint32_t i = 0; i < head_dim; ++i)
                        sc += qv[i] * shelf::quantization::codec::device::e4m3_decode_f32(
                                          kv.k_pool[kb + i]);
                    sc *= k_scale * scale;
                    if (pass == 0) {
                        if (sc > max_score) max_score = sc;
                    } else {
                        const float pv = __expf(sc - max_score);
                        sum_exp += pv;
                        const float w = pv * v_scale;
                        for (uint32_t i = 0; i < head_dim; ++i)
                            acc[i] += w * shelf::quantization::codec::device::e4m3_decode_f32(
                                              kv.v_pool[kb + i]);
                    }
                }
            }
        }
        for (uint32_t i = 0; i < head_dim; ++i)
            store_elem(out, static_cast<size_t>(row) * out.row_stride +
                               static_cast<size_t>(h) * head_dim + i,
                       acc[i] / sum_exp);
    }
    return ProgramStatus::COMPLETE;
}

}  // namespace shelf::kernel::detail
