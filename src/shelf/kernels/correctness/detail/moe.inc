#pragma once

#include <shelf/kernels/psq5_device.h>
#include <shelf/quantization/codec/e4m3_detail.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/quantization/psq5/psq5.h>

namespace ps::kernel::detail {

namespace pq = ::ps::quantization::psq5;

constexpr uint32_t kMoeMaxExperts = 512;
constexpr uint32_t kMoeMaxIntermediate = 1024;

__device__ __forceinline__ ProgramStatus exec_moe_router(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    ResolvedValue idx_out{};
    ResolvedValue w_out{};
    if (!resolve_value(env, b.input_slots[0], in))
        return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], idx_out))
        return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[1], w_out))
        return ProgramStatus::INVALID_BINDING;
    const uint32_t experts = b.group_size;
    const uint32_t top_k = b.rotary_dim;
    if (experts == 0u || experts > kMoeMaxExperts) return ProgramStatus::INVALID_BINDING;
    if (top_k == 0u || top_k > experts) return ProgramStatus::INVALID_BINDING;
    if (idx_out.dtype != ValueDType::I32 || w_out.dtype != ValueDType::F32)
        return ProgramStatus::INVALID_BINDING;
    if (idx_out.features != top_k || w_out.features != top_k)
        return ProgramStatus::INVALID_BINDING;
    if (in.dtype != ValueDType::BF16 && in.dtype != ValueDType::F32)
        return ProgramStatus::INVALID_BINDING;
    if (b.weight_index >= env.program->weight_slot_count ||
        env.program->weight_table == nullptr)
        return ProgramStatus::INVALID_BINDING;
    const auto* wtable = static_cast<const WeightSlot*>(env.program->weight_table);
    const WeightSlot w = wtable[b.weight_index];
    if (w.encoding != 0u) return ProgramStatus::INVALID_PROGRAM;
    if (w.device_ptr == nullptr || w.codes != nullptr || w.scales != nullptr)
        return ProgramStatus::INVALID_PROGRAM;
    if (w.rows != experts || w.cols != in.features)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t hidden = in.features;
    const uint32_t rows = value_rows(env, in);
    const bool renorm = (b.flags & 1u) != 0u;
    const uint32_t stride = global_stride(env);
    const bf16* wtable_all = static_cast<const bf16*>(w.device_ptr);
    auto* idx_base = static_cast<int32_t*>(idx_out.ptr);
    auto* wgt_base = static_cast<float*>(w_out.ptr);
    for (uint32_t t = global_tid(env); t < rows; t += stride) {
        float probs[kMoeMaxExperts];
        const size_t xbase = static_cast<size_t>(t) * in.row_stride;
        for (uint32_t e = 0; e < experts; ++e) {
            const bf16* wrow = wtable_all + static_cast<size_t>(e) * hidden;
            float acc = 0.f;
            for (uint32_t h = 0; h < hidden; ++h)
                acc += load_elem(in, xbase + h) * __bfloat162float(wrow[h]);
            probs[e] = acc;
        }
        float mx = probs[0];
        for (uint32_t e = 1; e < experts; ++e)
            if (probs[e] > mx) mx = probs[e];
        float sum = 0.f;
        for (uint32_t e = 0; e < experts; ++e) {
            probs[e] = expf(probs[e] - mx);
            sum += probs[e];
        }
        for (uint32_t e = 0; e < experts; ++e) probs[e] /= sum;
        int32_t* idx_row = idx_base + static_cast<size_t>(t) * idx_out.row_stride;
        float* wgt_row = wgt_base + static_cast<size_t>(t) * w_out.row_stride;
        for (uint32_t s = 0; s < top_k; ++s) {
            uint32_t best = 0u;
            float best_v = -1.f;
            for (uint32_t e = 0; e < experts; ++e) {
                const float v = probs[e];
                if (v < 0.f) continue;
                if (v > best_v) {
                    best_v = v;
                    best = e;
                }
            }
            probs[best] = -1.f;
            idx_row[s] = static_cast<int32_t>(best);
            wgt_row[s] = best_v;
        }
        if (renorm) {
            float wsum = 0.f;
            for (uint32_t s = 0; s < top_k; ++s) wsum += wgt_row[s];
            for (uint32_t s = 0; s < top_k; ++s) wgt_row[s] /= wsum;
        }
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ float psq5_dot(const WeightSlot& w, uint32_t row,
                                          const ResolvedValue& x, size_t xoff,
                                          uint32_t k_limit) {
    const uint32_t nb = w.k_padded / pq::kBlockK;
    const uint32_t tile = row >> 4;
    const uint32_t in_row = row & 15u;
    const auto* codes = static_cast<const uint8_t*>(w.codes);
    const auto* signs = static_cast<const uint8_t*>(w.sign);
    const auto* scales = static_cast<const uint8_t*>(w.scales);
    const uint32_t blocks = (k_limit + pq::kBlockK - 1u) / pq::kBlockK;
    float acc = 0.f;
    for (uint32_t ib = 0; ib < blocks; ++ib) {
        const size_t slot =
            (static_cast<size_t>(tile) * nb + ib) * pq::kTileRows + in_row;
        const uint8_t* mp = codes + slot * pq::kMagnitudeBytesPerBlock;
        const uint8_t* sp = signs + slot * pq::kSignBytesPerBlock;
        const float ws = psq5_scale_f32(scales[slot * pq::kScaleBytesPerBlock]);
        for (uint32_t g = 0; g < 4u; ++g) {
            const uint32_t k0 = ib * pq::kBlockK + g * 8u;
            if (k0 >= k_limit) break;
            const int32x2_t w8 = psq5_unpack_e4m3(
                *reinterpret_cast<const uint32_t*>(mp + g * 4u), sp[g]);
            for (uint32_t j = 0; j < 8u; ++j) {
                const uint32_t kk = k0 + j;
                if (kk >= k_limit) break;
                const uint32_t word = static_cast<uint32_t>(w8[j >> 2]);
                const uint8_t code = static_cast<uint8_t>(word >> ((j & 3u) * 8u));
                const float wv =
                    ::ps::quantization::codec::device::e4m3_decode_f32(code) * ws;
                acc += load_elem(x, xoff + kk) * wv;
            }
        }
    }
    return acc;
}

__device__ __forceinline__ ProgramStatus exec_moe_experts(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    ResolvedValue idx_in{};
    ResolvedValue w_in{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in))
        return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[1], idx_in))
        return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.input_slots[2], w_in))
        return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out))
        return ProgramStatus::INVALID_BINDING;
    const uint32_t experts = b.group_size;
    const uint32_t top_k = b.rotary_dim;
    if (experts == 0u || top_k == 0u || top_k > experts)
        return ProgramStatus::INVALID_BINDING;
    if (idx_in.dtype != ValueDType::I32 || w_in.dtype != ValueDType::F32 ||
        out.dtype != ValueDType::F32)
        return ProgramStatus::INVALID_BINDING;
    if (idx_in.features != top_k || w_in.features != top_k)
        return ProgramStatus::INVALID_BINDING;
    if (in.dtype != ValueDType::BF16 && in.dtype != ValueDType::F32)
        return ProgramStatus::INVALID_BINDING;
    if (b.weight_index >= env.program->weight_slot_count ||
        env.program->weight_table == nullptr)
        return ProgramStatus::INVALID_BINDING;
    if (static_cast<uint32_t>(b.weight_index) + 2u * experts >
        env.program->weight_slot_count)
        return ProgramStatus::INVALID_BINDING;
    const auto* wtable = static_cast<const WeightSlot*>(env.program->weight_table);
    const WeightSlot gu0 = wtable[b.weight_index];
    const WeightSlot dn0 = wtable[b.weight_index + experts];
    if (gu0.encoding != dn0.encoding) return ProgramStatus::INVALID_PROGRAM;
    const bool psq5 =
        gu0.encoding ==
        static_cast<uint8_t>(::ps::quantization::manifest::QuantizedEncoding::Psq5);
    if (!psq5 && gu0.encoding != 0u) return ProgramStatus::INVALID_PROGRAM;
    if (psq5) {
        if (gu0.codes == nullptr || gu0.sign == nullptr || gu0.scales == nullptr ||
            dn0.codes == nullptr || dn0.sign == nullptr || dn0.scales == nullptr ||
            !gu0.preshuffled || !dn0.preshuffled)
            return ProgramStatus::INVALID_PROGRAM;
        if (gu0.k_padded < gu0.cols || dn0.k_padded < dn0.cols ||
            (gu0.k_padded % pq::kBlockK) != 0u || (dn0.k_padded % pq::kBlockK) != 0u)
            return ProgramStatus::INVALID_BINDING;
    } else {
        if (gu0.device_ptr == nullptr || dn0.device_ptr == nullptr ||
            gu0.codes != nullptr || gu0.scales != nullptr || dn0.codes != nullptr ||
            dn0.scales != nullptr)
            return ProgramStatus::INVALID_PROGRAM;
    }
    const uint32_t inter = dn0.cols;
    if (inter == 0u || inter > kMoeMaxIntermediate)
        return ProgramStatus::INVALID_BINDING;
    if (gu0.rows != 2u * inter || gu0.cols != in.features)
        return ProgramStatus::INVALID_BINDING;
    if (dn0.rows != in.features) return ProgramStatus::INVALID_BINDING;
    const uint32_t hidden = in.features;
    const uint32_t rows = value_rows(env, out);
    const uint32_t stride = global_stride(env);
    auto* idx_base = static_cast<const int32_t*>(idx_in.ptr);
    auto* wgt_base = static_cast<const float*>(w_in.ptr);
    ResolvedValue gate_x{};
    gate_x.dtype = ValueDType::F32;
    for (uint32_t t = global_tid(env); t < rows; t += stride) {
        float gate[kMoeMaxIntermediate];
        float up[kMoeMaxIntermediate];
        const size_t xbase = static_cast<size_t>(t) * in.row_stride;
        const int32_t* idx_row = idx_base + static_cast<size_t>(t) * idx_in.row_stride;
        const float* wgt_row = wgt_base + static_cast<size_t>(t) * w_in.row_stride;
        float* out_row = static_cast<float*>(out.ptr) +
                         static_cast<size_t>(t) * out.row_stride;
        for (uint32_t h = 0; h < hidden; ++h) out_row[h] = 0.f;
        gate_x.ptr = gate;
        for (uint32_t s = 0; s < top_k; ++s) {
            const int32_t e = idx_row[s];
            if (e < 0 || static_cast<uint32_t>(e) >= experts)
                return ProgramStatus::INVALID_BINDING;
            const WeightSlot gu = wtable[b.weight_index + e];
            const WeightSlot dn = wtable[b.weight_index + experts + e];
            if (gu.encoding != gu0.encoding || dn.encoding != dn0.encoding ||
                gu.rows != 2u * inter || gu.cols != hidden || dn.rows != hidden ||
                dn.cols != inter)
                return ProgramStatus::INVALID_BINDING;
            if (psq5 &&
                (gu.codes == nullptr || gu.sign == nullptr || gu.scales == nullptr ||
                 dn.codes == nullptr || dn.sign == nullptr || dn.scales == nullptr ||
                 gu.k_padded < hidden || dn.k_padded < inter ||
                 (gu.k_padded % pq::kBlockK) != 0u ||
                 (dn.k_padded % pq::kBlockK) != 0u))
                return ProgramStatus::INVALID_PROGRAM;
            const float wgt = wgt_row[s];
            if (psq5) {
                for (uint32_t i = 0; i < inter; ++i)
                    gate[i] = psq5_dot(gu, i, in, xbase, hidden);
                for (uint32_t i = 0; i < inter; ++i)
                    up[i] = psq5_dot(gu, inter + i, in, xbase, hidden);
            } else {
                const bf16* gu_all = static_cast<const bf16*>(gu.device_ptr);
                for (uint32_t i = 0; i < inter; ++i) {
                    const bf16* grow = gu_all + static_cast<size_t>(i) * hidden;
                    float acc = 0.f;
                    for (uint32_t h = 0; h < hidden; ++h)
                        acc += load_elem(in, xbase + h) * __bfloat162float(grow[h]);
                    gate[i] = acc;
                }
                for (uint32_t i = 0; i < inter; ++i) {
                    const bf16* urow =
                        gu_all + static_cast<size_t>(inter + i) * hidden;
                    float acc = 0.f;
                    for (uint32_t h = 0; h < hidden; ++h)
                        acc += load_elem(in, xbase + h) * __bfloat162float(urow[h]);
                    up[i] = acc;
                }
            }
            for (uint32_t i = 0; i < inter; ++i) {
                const float g = gate[i];
                gate[i] = g * (1.0f / (1.0f + expf(-g))) * up[i];
            }
            if (psq5) {
                for (uint32_t h = 0; h < hidden; ++h)
                    out_row[h] += wgt * psq5_dot(dn, h, gate_x, 0u, inter);
            } else {
                const bf16* dn_all = static_cast<const bf16*>(dn.device_ptr);
                for (uint32_t h = 0; h < hidden; ++h) {
                    const bf16* drow = dn_all + static_cast<size_t>(h) * inter;
                    float acc = 0.f;
                    for (uint32_t i = 0; i < inter; ++i)
                        acc += gate[i] * __bfloat162float(drow[i]);
                    out_row[h] += wgt * acc;
                }
            }
        }
    }
    return ProgramStatus::COMPLETE;
}

}  // namespace ps::kernel::detail
