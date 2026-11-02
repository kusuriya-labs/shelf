#pragma once

namespace shelf::kernel::detail {

__device__ __forceinline__ ProgramStatus exec_gdn_conv1d(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    if (env.context == nullptr || env.qwen_state == nullptr)
        return ProgramStatus::INVALID_PROGRAM;
    const auto* sb = find_state(env, b.descriptor_slot);
    if (sb == nullptr ||
        sb->kind != static_cast<uint8_t>(shelf::runtime::PrimitiveStateKind::GDN_CONV_STATE))
        return ProgramStatus::INVALID_BINDING;
    const auto& gdn = env.qwen_state->gdn;
    if (gdn.conv_base == nullptr || sb->state_index >= gdn.num_gdn_states ||
        gdn.conv_history != 3u)
        return ProgramStatus::INVALID_BINDING;
    uint32_t w_elements = 0;
    ValueDType w_dtype = ValueDType::BF16;
    const auto* conv_w = static_cast<const bf16*>(
        resolve_parameter(env, b.parameter_index0, &w_elements, &w_dtype));
    if (conv_w == nullptr || w_dtype != ValueDType::BF16 ||
        w_elements < gdn.conv_dim * 4u)
        return ProgramStatus::INVALID_BINDING;
    ResolvedValue in{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    if (in.dtype != ValueDType::BF16 ||
        (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32) ||
        in.features != gdn.conv_dim || out.features != gdn.conv_dim)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t requests = env.context->num_requests;
    const size_t total = static_cast<size_t>(requests) * gdn.conv_dim;
    for (size_t task = global_tid(env); task < total; task += global_stride(env)) {
        const uint32_t r = static_cast<uint32_t>(task / gdn.conv_dim);
        const uint32_t ch = static_cast<uint32_t>(task % gdn.conv_dim);
        const auto req = env.context->requests[r];
        const uint32_t slot = req.request_handle.slot;
        if (slot >= gdn.max_sequences) return ProgramStatus::INVALID_BINDING;
        auto* state_row = reinterpret_cast<bf16*>(gdn.conv_base) +
                          static_cast<size_t>(slot) * gdn.conv_slot_stride +
                          static_cast<size_t>(sb->state_index) * gdn.conv_layer_stride;
        const uint64_t hist_stride = gdn.conv_history_stride;
        const bf16* w = conv_w + static_cast<size_t>(ch) * 4u;
        float s0 = __bfloat162float(state_row[ch]);
        float s1 = __bfloat162float(state_row[hist_stride + ch]);
        float s2 = __bfloat162float(state_row[(size_t)2 * hist_stride + ch]);
        const float w0 = __bfloat162float(w[0]);
        const float w1 = __bfloat162float(w[1]);
        const float w2 = __bfloat162float(w[2]);
        const float w3 = __bfloat162float(w[3]);
        for (uint32_t t = 0; t < req.row_count; ++t) {
            const uint32_t row = req.row_begin + t;
            if (row >= env.context->actual_rows) continue;
            const size_t in_idx = static_cast<size_t>(row) * in.row_stride + ch;
            const size_t out_idx = static_cast<size_t>(row) * out.row_stride + ch;
            const float x = __bfloat162float(static_cast<const bf16*>(in.ptr)[in_idx]);
            const float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
            s0 = s1;
            s1 = s2;
            s2 = x;
            store_elem(out, out_idx, y);
        }
        state_row[ch] = __float2bfloat16(s0);
        state_row[hist_stride + ch] = __float2bfloat16(s1);
        state_row[(size_t)2 * hist_stride + ch] = __float2bfloat16(s2);
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ float gdn_ld_f16(const fp16_t* p) {
    return static_cast<float>(p->value);
}

__device__ __forceinline__ void gdn_st_f16(fp16_t* p, float v) {
    p->value = static_cast<_Float16>(v);
}

__device__ __forceinline__ ProgramStatus exec_gdn_recurrence(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    if (env.context == nullptr || env.qwen_state == nullptr)
        return ProgramStatus::INVALID_PROGRAM;
    const auto* sb = find_state(env, b.descriptor_slot);
    if (sb == nullptr ||
        sb->kind != static_cast<uint8_t>(shelf::runtime::PrimitiveStateKind::GDN_RECURRENCE_STATE))
        return ProgramStatus::INVALID_BINDING;
    const auto& gdn = env.qwen_state->gdn;
    if (gdn.recurrent_base == nullptr || sb->state_index >= gdn.num_gdn_states ||
        gdn.head_k == 0u || gdn.head_v == 0u || gdn.head_k > 128u)
        return ProgramStatus::INVALID_BINDING;
    uint32_t dt_elements = 0, al_elements = 0;
    ValueDType dt_dtype = ValueDType::BF16, al_dtype = ValueDType::BF16;
    const auto* dt_bias = static_cast<const bf16*>(
        resolve_parameter(env, b.parameter_index0, &dt_elements, &dt_dtype));
    const auto* a_log = static_cast<const bf16*>(
        resolve_parameter(env, b.parameter_index1, &al_elements, &al_dtype));
    if (dt_bias == nullptr || a_log == nullptr || dt_dtype != ValueDType::BF16 ||
        al_dtype != ValueDType::BF16 || dt_elements < gdn.num_v_heads ||
        al_elements < gdn.num_v_heads)
        return ProgramStatus::INVALID_BINDING;
    ResolvedValue q_in{}, k_in{}, v_in{}, a_in{}, b_in{}, out{};
    if (!resolve_value(env, b.input_slots[0], q_in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[1], k_in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[2], v_in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[3], a_in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[4], b_in)) return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    const uint32_t value_dim = gdn.num_v_heads * gdn.head_v;
    const uint32_t qk_dim = q_in.features;
    if (v_in.dtype != ValueDType::BF16 || b_in.dtype != ValueDType::BF16 ||
        a_in.dtype != ValueDType::BF16 || out.dtype != ValueDType::F32 ||
        qk_dim == 0u || qk_dim % gdn.head_k != 0u || v_in.features != value_dim ||
        b_in.features != gdn.num_v_heads || a_in.features != gdn.num_v_heads ||
        out.features != value_dim)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t key_heads = qk_dim / gdn.head_k;
    const uint32_t repeat = gdn.num_v_heads / key_heads;
    if (repeat == 0u || gdn.num_v_heads % key_heads != 0u)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t requests = env.context->num_requests;
    const size_t total = static_cast<size_t>(requests) * gdn.num_v_heads * gdn.head_v;
    for (size_t task = global_tid(env); task < total; task += global_stride(env)) {
        const uint32_t r = static_cast<uint32_t>(task / (gdn.num_v_heads * gdn.head_v));
        const uint32_t h =
            static_cast<uint32_t>((task / gdn.head_v) % gdn.num_v_heads);
        const uint32_t vcol = static_cast<uint32_t>(task % gdn.head_v);
        const auto req = env.context->requests[r];
        const uint32_t slot = req.request_handle.slot;
        if (slot >= gdn.max_sequences) return ProgramStatus::INVALID_BINDING;
        const uint32_t qk_head = h / repeat;
        fp16_t* sbase = gdn.recurrent_base +
                       static_cast<size_t>(slot) * gdn.recurrent_slot_stride +
                       static_cast<size_t>(sb->state_index) * gdn.recurrent_layer_stride +
                       static_cast<size_t>(h) * gdn.head_k * gdn.head_v + vcol;
        for (uint32_t t = 0; t < req.row_count; ++t) {
            const uint32_t row = req.row_begin + t;
            if (row >= env.context->actual_rows) continue;
            const float beta =
                1.0f / (1.0f + __expf(-__bfloat162float(
                            static_cast<const bf16*>(b_in.ptr)
                                [static_cast<size_t>(row) * b_in.row_stride + h])));
            const float a_val = __bfloat162float(
                static_cast<const bf16*>(a_in.ptr)
                    [static_cast<size_t>(row) * a_in.row_stride + h]);
            const float g = -__expf(__bfloat162float(a_log[h])) *
                            gdn_softplus(a_val + __bfloat162float(dt_bias[h]));
            const float d_t = __expf(g);
            const size_t qrow = static_cast<size_t>(row) * q_in.row_stride +
                                static_cast<size_t>(qk_head) * gdn.head_k;
            const size_t krow = static_cast<size_t>(row) * k_in.row_stride +
                                static_cast<size_t>(qk_head) * gdn.head_k;
            const bf16 v_val = static_cast<const bf16*>(v_in.ptr)
                                   [static_cast<size_t>(row) * v_in.row_stride +
                                    static_cast<size_t>(h) * gdn.head_v + vcol];
            float kv_mem = 0.f;
            for (uint32_t kk = 0; kk < gdn.head_k; ++kk) {
                kv_mem += gdn_ld_f16(sbase + kk * gdn.head_v) * d_t *
                          load_elem(k_in, krow + kk);
            }
            const float delta = (__bfloat162float(v_val) - kv_mem) * beta;
            for (uint32_t kk = 0; kk < gdn.head_k; ++kk) {
                fp16_t* sc = sbase + kk * gdn.head_v;
                gdn_st_f16(sc, gdn_ld_f16(sc) * d_t + load_elem(k_in, krow + kk) * delta);
            }
            float acc = 0.f;
            for (uint32_t kk = 0; kk < gdn.head_k; ++kk) {
                acc += gdn_ld_f16(sbase + kk * gdn.head_v) * load_elem(q_in, qrow + kk);
            }
            static_cast<float*>(out.ptr)
                [static_cast<size_t>(row) * out.row_stride +
                 static_cast<size_t>(h) * gdn.head_v + vcol] = acc;
        }
    }
    return ProgramStatus::COMPLETE;
}

}  // namespace shelf::kernel::detail
