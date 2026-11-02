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
    if (we.encoding == 2u) {
        if (we.device_ptr != nullptr || we.codes == nullptr || we.scales == nullptr)
            return ProgramStatus::INVALID_PROGRAM;
        const int8_t* wcodes = static_cast<const int8_t*>(we.codes);
        const uint8_t* wscales = static_cast<const uint8_t*>(we.scales);
        const uint32_t kp = we.k_padded;
        if (kp < hidden || (kp & 31u) != 0u || we.cols != hidden)
            return ProgramStatus::INVALID_PROGRAM;
        if (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32)
            return ProgramStatus::INVALID_BINDING;
        for (size_t idx = global_tid(env); idx < static_cast<size_t>(rows) * hidden;
             idx += global_stride(env)) {
            const uint32_t m = static_cast<uint32_t>(idx / hidden);
            const uint32_t h = static_cast<uint32_t>(idx % hidden);
            const uint32_t tok =
                env.context->token_ids ? static_cast<uint32_t>(env.context->token_ids[m]) : 0u;
            const uint32_t blk = h >> 5;
            const uint32_t lok = h & 31u;
            const int8_t cv = wcodes[static_cast<size_t>(tok >> 4) * we.codes_row_stride_bytes +
                static_cast<size_t>(blk) * 512u + static_cast<size_t>(lok >> 4) * 256u +
                static_cast<size_t>(tok & 15u) * 16u + static_cast<size_t>(lok & 15u)];
            const float wv =
                static_cast<float>(cv) *
                ps::quantization::codec::device::ue4m3_decode_f32(
                    wscales[static_cast<size_t>(tok) * (kp >> 5) + (h >> 5)]);
            store_elem(out, static_cast<size_t>(m) * out.row_stride + h, wv);
        }
        return ProgramStatus::COMPLETE;
    }
    if (we.encoding == 4u) {
        if (we.device_ptr != nullptr || we.codes == nullptr || we.scales == nullptr)
            return ProgramStatus::INVALID_PROGRAM;
        const uint8_t* wcodes = static_cast<const uint8_t*>(we.codes);
        const uint8_t* wscales = static_cast<const uint8_t*>(we.scales);
        const uint32_t kp = we.k_padded;
        if (kp < hidden || (kp & 31u) != 0u || we.cols != hidden)
            return ProgramStatus::INVALID_PROGRAM;
        if (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32)
            return ProgramStatus::INVALID_BINDING;
        const uint32_t sstride = we.storage_scale_stride_bytes != 0u
                                     ? we.storage_scale_stride_bytes : kp;
        for (size_t idx = global_tid(env); idx < static_cast<size_t>(rows) * hidden;
             idx += global_stride(env)) {
            const uint32_t m = static_cast<uint32_t>(idx / hidden);
            const uint32_t h = static_cast<uint32_t>(idx % hidden);
            const uint32_t tok =
                env.context->token_ids ? static_cast<uint32_t>(env.context->token_ids[m]) : 0u;
            const uint32_t blk = h >> 5;
            const uint32_t lok = h & 31u;
            const uint8_t cv = wcodes[static_cast<size_t>(tok >> 4) * we.codes_row_stride_bytes +
                static_cast<size_t>(blk) * 512u + static_cast<size_t>(lok >> 4) * 256u +
                static_cast<size_t>(tok & 15u) * 16u + static_cast<size_t>(lok & 15u)];
            const uint8_t* sb = wscales +
                static_cast<size_t>(tok >> 4) * sstride +
                static_cast<size_t>(tok & 15u) * 2u +
                static_cast<size_t>(blk) * 32u;
            const uint16_t bits = static_cast<uint16_t>(sb[0]) |
                                  (static_cast<uint16_t>(sb[1]) << 8u);
            const float wv = ::ps::quantization::codec::device::e4m3_decode_f32(cv) *
                             __bfloat162float(__ushort_as_bfloat16(bits));
            store_elem(out, static_cast<size_t>(m) * out.row_stride + h, wv);
        }
        return ProgramStatus::COMPLETE;
    }
    if (we.encoding == 6u) {
        if (we.device_ptr != nullptr || we.codes == nullptr || we.scales == nullptr)
            return ProgramStatus::INVALID_PROGRAM;
        const uint8_t* wcodes = static_cast<const uint8_t*>(we.codes);
        const uint8_t* wscales = static_cast<const uint8_t*>(we.scales);
        const uint32_t kp = we.k_padded;
        if (kp < hidden || (kp & 31u) != 0u || we.cols != hidden)
            return ProgramStatus::INVALID_PROGRAM;
        if (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32)
            return ProgramStatus::INVALID_BINDING;
        const uint32_t nb = kp >> 5;
        for (size_t idx = global_tid(env); idx < static_cast<size_t>(rows) * hidden;
             idx += global_stride(env)) {
            const uint32_t m = static_cast<uint32_t>(idx / hidden);
            const uint32_t h = static_cast<uint32_t>(idx % hidden);
            const uint32_t tok =
                env.context->token_ids ? static_cast<uint32_t>(env.context->token_ids[m]) : 0u;
            const uint32_t ob = tok >> 5;
            const uint32_t ot = (tok & 31u) >> 4;
            const uint32_t oi = tok & 15u;
            const uint32_t full_row = tok & 31u;
            const uint32_t ib = h >> 5;
            const uint32_t lok = h & 31u;
            const size_t cbase = static_cast<size_t>(ob * nb + ib) * 1024u;
            const uint8_t cv = wcodes[cbase + static_cast<size_t>(ot) * 512u +
                static_cast<size_t>(oi) * 16u + (lok & 15u) + ((lok >= 16u) ? 256u : 0u)];
            const size_t soff = static_cast<size_t>(ob * nb + ib) * 64u +
                static_cast<size_t>(full_row) * 2u;
            const uint16_t sbits =
                static_cast<uint16_t>(wscales[soff]) |
                (static_cast<uint16_t>(wscales[soff + 1u]) << 8u);
            const float wv = static_cast<float>(static_cast<int8_t>(cv)) *
                            __bfloat162float(__ushort_as_bfloat16(sbits));
            store_elem(out, static_cast<size_t>(m) * out.row_stride + h, wv);
        }
        return ProgramStatus::COMPLETE;
    }
    return ProgramStatus::INVALID_PROGRAM;
}

}  // namespace ps::kernel::detail
