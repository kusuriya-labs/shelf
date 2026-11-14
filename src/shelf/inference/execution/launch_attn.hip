#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdlib>

namespace shelf::execution {
namespace {
using namespace ::shelf::runtime;
using bf16 = __hip_bfloat16;

__global__ void attn_q_preprocess(const bf16* input, const bf16* weight, bf16* gate,
                                  bf16* output, const uint32_t* positions,
                                  const double* freq, uint32_t input_stride,
                                  uint32_t gate_stride, uint32_t output_stride,
                                  float eps, float factor) {
    const uint32_t row = blockIdx.x / 24u;
    const uint32_t head = blockIdx.x % 24u;
    const uint32_t tid = threadIdx.x;
    const bf16* src = input + size_t(row) * input_stride + head * 512u;
    bf16* g = gate + size_t(row) * gate_stride + head * 256u;
    bf16* dst = output + size_t(row) * output_stride + head * 256u;
    float x[4];
    float s0 = 0.0f, s1 = 0.0f;
    for (uint32_t j = 0; j < 4; ++j) x[j] = __bfloat162float(src[tid * 4u + j]);
    s0 += x[0] * x[0]; s1 += x[1] * x[1];
    s0 += x[2] * x[2]; s1 += x[3] * x[3];
    float sum = s0 + s1;
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down(sum, offset, 32);
    __shared__ float sums[2];
    __shared__ float normalized[256];
    if ((tid & 31u) == 0) sums[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32u) {
        float v = tid < 2u ? sums[tid] : 0.f;
        for (int offset = 16; offset; offset >>= 1) v += __shfl_down(v, offset, 32);
        if (tid == 0u) sums[0] = rsqrtf(v / 256.f + eps);
    }
    __syncthreads();
    for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t col = tid * 4u + j;
        normalized[col] = (x[j] * sums[0]) * (1.f + __bfloat162float(weight[col]));
        g[col] = src[256u + col];
    }
    __syncthreads();
    if (tid < 32u) {
        const uint32_t pos = positions ? positions[row] : row;
        double sn, cs;
        sincos(static_cast<double>(pos) * freq[tid], &sn, &cs);
        const float c = static_cast<float>(cs) * factor;
        const float s = static_cast<float>(sn) * factor;
        const float a = normalized[tid], b = normalized[tid + 32u];
        dst[tid] = __float2bfloat16(__fmaf_rn(a, c, -__fmul_rn(b, s)));
        dst[tid + 32u] = __float2bfloat16(__fmaf_rn(b, c, __fmul_rn(a, s)));
    }
    for (uint32_t col = 64u + tid; col < 256u; col += 64u)
        dst[col] = __float2bfloat16(normalized[col]);
}


__global__ void attn_kv_preprocess(const bf16* input, const bf16* weight,
                                   const double* freq, const uint32_t* positions,
                                   ::shelf::kernel::KvAppendCommonArgs a,
                                   bf16* kp, bf16* vp, uint32_t input_stride,
                                   float eps, float factor) {
    const uint32_t row = blockIdx.x / 4u, head = blockIdx.x % 4u;
    const uint32_t tid = threadIdx.x;
    const bf16* src = input + size_t(row) * input_stride + head * 256u;
    float x[4];
    float s0 = 0.f, s1 = 0.f;
    for (uint32_t j = 0; j < 4; ++j) x[j] = __bfloat162float(src[tid * 4u + j]);
    s0 += x[0] * x[0]; s1 += x[1] * x[1];
    s0 += x[2] * x[2]; s1 += x[3] * x[3];
    float sum = s0 + s1;
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down(sum, offset, 32);
    __shared__ float sums[2];
    __shared__ float normalized[256];
    if ((tid & 31u) == 0u) sums[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32u) {
        float v = tid < 2u ? sums[tid] : 0.f;
        for (int offset = 16; offset; offset >>= 1) v += __shfl_down(v, offset, 32);
        if (tid == 0u) sums[0] = rsqrtf(v / 256.f + eps);
    }
    __syncthreads();
    for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t col = tid * 4u + j;
        normalized[col] = (x[j] * sums[0]) * (1.f + __bfloat162float(weight[col]));
    }
    __syncthreads();
    const uint32_t slot = a.row_sequence_slots[row];
    if (slot >= a.max_sequences) return;
    const uint32_t pos = a.row_positions[row];
    const uint32_t logical = pos / a.page_tokens;
    if (logical >= a.block_table_stride) return;
    const uint32_t page = a.block_tables[size_t(slot) * a.block_table_stride + logical];
    if (page >= a.num_pages) return;
    const uint64_t base = uint64_t(a.layer) * a.elems_per_layer +
        uint64_t(page) * a.elems_per_page + uint64_t(pos % a.page_tokens) * a.elems_per_token + head * 256u;
    const bf16* vsrc = reinterpret_cast<const bf16*>(a.v_input) + size_t(row) * a.v_row_stride + head * 256u;
    for (uint32_t j = 0; j < 4; ++j) vp[base + tid * 4u + j] = vsrc[tid * 4u + j];
    if (tid < 32u) {
        const uint32_t rope_pos = positions ? positions[row] : row;
        double sn, cs;
        sincos(static_cast<double>(rope_pos) * freq[tid], &sn, &cs);
        const float c = static_cast<float>(cs) * factor;
        const float s = static_cast<float>(sn) * factor;
        const float x0 = normalized[tid], x1 = normalized[tid + 32u];
        kp[base + tid] = __float2bfloat16(__fmaf_rn(x0, c, -__fmul_rn(x1, s)));
        kp[base + tid + 32u] = __float2bfloat16(__fmaf_rn(x1, c, __fmul_rn(x0, s)));
    }
    for (uint32_t col = 64u + tid; col < 256u; col += 64u)
        kp[base + col] = __float2bfloat16(normalized[col]);
}

bool disjoint(uint64_t a, size_t an, uint64_t b, size_t bn) {
    return a && b && (a + an <= b || b + bn <= a);
}
}

Result<OptimizedLaunchResult> try_launch_attn_q(
    const Program& p, size_t i, size_t end, HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult none{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_ATTN_Q_FUSION") == nullptr;
    if (!enabled || ctx.value_trace || ctx.layer_override_count || ctx.actual_rows == 0 ||
        ctx.actual_rows > 16 || ctx.num_prefill_requests || ctx.num_requests > 1 || i == 0 || i + 6 >= end)
        return none;
    const auto& s = p.dispatches[i];
    const auto& n = p.dispatches[i + 1];
    const auto& r = p.dispatches[i + 2];
    const auto& kn = p.dispatches[i + 3];
    const auto& kr = p.dispatches[i + 4];
    const auto& kv = p.dispatches[i + 5];
    const auto& att = p.dispatches[i + 6];
    if (p.dispatches[i - 1].kernel_id != KernelId::LINEAR_EXL3_INT8 ||
        s.kernel_id != KernelId::SPLIT || n.kernel_id != KernelId::RMS_NORM ||
        r.kernel_id != KernelId::ROPE || kn.kernel_id != KernelId::RMS_NORM ||
        kr.kernel_id != KernelId::ROPE || kv.kernel_id != KernelId::KV_APPEND ||
        att.kernel_id != KernelId::PAGED_ATTENTION || s.input_count != 1 || s.output_count != 2 ||
        n.input_count != 1 || n.output_count != 1 || r.input_count != 1 || r.output_count != 1 ||
        n.input_slots[0] != s.output_slots[0] || r.input_slots[0] != n.output_slots[0] ||
        att.input_count < 1 || att.input_slots[0] != r.output_slots[0]) return none;
    uint32_t gdn = 0, attention = 0;
    for (const auto& d : p.dispatches) {
        if (d.kernel_id == KernelId::MOE_ROUTER) return none;
        gdn += d.kernel_id == KernelId::GDN_RECURRENCE;
        attention += d.kernel_id == KernelId::PAGED_ATTENTION;
    }
    if (gdn != 48u || attention != 16u) return none;
    for (size_t j = i + 3; j < p.dispatches.size(); ++j)
        for (uint32_t k = 0; k < p.dispatches[j].input_count; ++k)
            if (p.dispatches[j].input_slots[k] == s.output_slots[0] ||
                p.dispatches[j].input_slots[k] == n.output_slots[0]) return none;
    PhysicalElementwiseLaunch split{};
    PhysicalRmsNormLaunch norm{};
    PhysicalRopeLaunch rope{};
    PhysicalKvAppendLaunch append{};
    auto shape = ctx;
    shape.actual_rows = 1;
    shape.actual_outputs = 1;
    auto a = resolve_elementwise_physical(p, s, shape, split);
    auto b = resolve_rmsnorm_physical(p, n, shape, norm);
    auto c = resolve_rope_physical(p, r, shape, rope);
    auto d = resolve_kv_append_physical(p, i + 5, ctx, append);
    if (!a.ok() || !b.ok() || !c.ok() || !d.ok() ||
        a.value() != PhysicalResolveStatus::Applicable || b.value() != PhysicalResolveStatus::Applicable ||
        c.value() != PhysicalResolveStatus::Applicable || d.value() != PhysicalResolveStatus::Applicable ||
        split.profile != ElementwiseProfile::SplitBf16InterleavedHeads || split.features != 6144 ||
        split.aux != 256 || norm.variant != PhysicalRmsNormVariant::Bf16F32PgOnePlus ||
        norm.group_size != 256 || norm.features != 6144 || norm.workgroup != 64 ||
        rope.variant != PhysicalRopeVariant::F32Bf16Pair || rope.head_dim != 256 ||
        rope.rotary_dim != 64 || append.variant != PhysicalKvAppendVariant::Bf16 ||
        append.common.kv_heads != 4 || append.common.head_dim != 256) return none;
    auto in = resolve_host_value(p, ValueId{s.input_slots[0]}, ctx);
    auto gate = resolve_host_value(p, ValueId{s.output_slots[1]}, ctx);
    auto out = resolve_host_value(p, ValueId{r.output_slots[0]}, ctx);
    if (!in.ok() || !gate.ok() || !out.ok() || in.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        gate.value().row_domain != ValueRowDomain::TOKEN_ROWS || out.value().row_domain != ValueRowDomain::TOKEN_ROWS)
        return none;
    const auto* ib = p.find_value(ValueId{s.input_slots[0]});
    const auto* gb = p.find_value(ValueId{s.output_slots[1]});
    const auto* ob = p.find_value(ValueId{r.output_slots[0]});
    if (!ib || !gb || !ob || ib->storage != ValueStorage::WORKSPACE ||
        gb->storage != ValueStorage::WORKSPACE || ob->storage != ValueStorage::WORKSPACE) return none;
    const size_t ic = ib->bytes / (size_t(split.in0_stride) * 2u);
    const size_t gc = gb->bytes / (size_t(split.out1_stride) * 2u);
    const size_t oc = ob->bytes / (size_t(rope.output_row_stride) * 2u);
    if (ic < ctx.actual_rows || gc < ctx.actual_rows || oc < ctx.actual_rows ||
        ic != gc || ic != oc) return none;
    const size_t isz = ((ic - 1) * split.in0_stride + 12288u) * 2u;
    const size_t gsz = ((ic - 1) * split.out1_stride + 6144u) * 2u;
    const size_t osz = ((ic - 1) * rope.output_row_stride + 6144u) * 2u;
    if (!disjoint(split.in0, isz, split.out1, gsz) ||
        !disjoint(split.in0, isz, rope.output, osz) ||
        !disjoint(split.out1, gsz, rope.output, osz)) return none;
    attn_q_preprocess<<<ctx.actual_rows * 24u, 64, 0, stream>>>(
        reinterpret_cast<const bf16*>(split.in0), reinterpret_cast<const bf16*>(norm.weight),
        reinterpret_cast<bf16*>(split.out1), reinterpret_cast<bf16*>(rope.output),
        reinterpret_cast<const uint32_t*>(rope.positions), reinterpret_cast<const double*>(rope.inv_freq),
        split.in0_stride, split.out1_stride, rope.output_row_stride, norm.eps, rope.attention_factor);
    const auto err = hipGetLastError();
    if (err != hipSuccess) return Status::hip_error("attention q preprocessing", hipGetErrorString(err), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 3};
}

Result<OptimizedLaunchResult> try_launch_attn_kv(
    const Program& p, size_t i, size_t end, HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult none{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_ATTN_KV_FUSION") == nullptr;
    if (!enabled || ctx.value_trace || ctx.layer_override_count || ctx.actual_rows == 0 ||
        ctx.actual_rows > 16 || ctx.num_prefill_requests || ctx.num_requests > 1 || i < 3 || i + 3 >= end) return none;
    const auto& split = p.dispatches[i - 3];
    const auto& qn = p.dispatches[i - 2];
    const auto& qr = p.dispatches[i - 1];
    const auto& n = p.dispatches[i];
    const auto& r = p.dispatches[i + 1];
    const auto& kv = p.dispatches[i + 2];
    const auto& att = p.dispatches[i + 3];
    if (split.kernel_id != KernelId::SPLIT || qn.kernel_id != KernelId::RMS_NORM ||
        qr.kernel_id != KernelId::ROPE || n.kernel_id != KernelId::RMS_NORM ||
        r.kernel_id != KernelId::ROPE || kv.kernel_id != KernelId::KV_APPEND ||
        att.kernel_id != KernelId::PAGED_ATTENTION || n.input_count != 1 || n.output_count != 1 ||
        r.input_count != 1 || r.output_count != 1 || kv.input_count != 2 ||
        r.input_slots[0] != n.output_slots[0] || kv.input_slots[0] != r.output_slots[0]) return none;
    uint32_t gdn = 0, attention = 0;
    for (const auto& d : p.dispatches) {
        if (d.kernel_id == KernelId::MOE_ROUTER) return none;
        gdn += d.kernel_id == KernelId::GDN_RECURRENCE;
        attention += d.kernel_id == KernelId::PAGED_ATTENTION;
    }
    if (gdn != 48u || attention != 16u) return none;
    for (size_t j = i + 3; j < p.dispatches.size(); ++j)
        for (uint32_t k = 0; k < p.dispatches[j].input_count; ++k)
            if (p.dispatches[j].input_slots[k] == n.output_slots[0] ||
                p.dispatches[j].input_slots[k] == r.output_slots[0]) return none;
    PhysicalRmsNormLaunch norm{};
    PhysicalRopeLaunch rope{};
    PhysicalKvAppendLaunch append{};
    auto shape = ctx;
    shape.actual_rows = 1;
    shape.actual_outputs = 1;
    auto a = resolve_rmsnorm_physical(p, n, shape, norm);
    auto b = resolve_rope_physical(p, r, shape, rope);
    auto c = resolve_kv_append_physical(p, i + 2, ctx, append);
    if (!a.ok() || !b.ok() || !c.ok() || a.value() != PhysicalResolveStatus::Applicable ||
        b.value() != PhysicalResolveStatus::Applicable || c.value() != PhysicalResolveStatus::Applicable ||
        norm.variant != PhysicalRmsNormVariant::Bf16F32PgOnePlus || norm.group_size != 256 ||
        norm.features != 1024 || norm.workgroup != 64 || rope.variant != PhysicalRopeVariant::F32Bf16Pair ||
        rope.head_dim != 256 || rope.rotary_dim != 64 || append.variant != PhysicalKvAppendVariant::Bf16 ||
        append.common.kv_heads != 4 || append.common.head_dim != 256) return none;
    auto in = resolve_host_value(p, ValueId{n.input_slots[0]}, ctx);
    auto v = resolve_host_value(p, ValueId{kv.input_slots[1]}, ctx);
    if (!in.ok() || !v.ok() || in.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        v.value().row_domain != ValueRowDomain::TOKEN_ROWS) return none;
    const auto* ib = p.find_value(ValueId{n.input_slots[0]});
    const auto* vb = p.find_value(ValueId{kv.input_slots[1]});
    if (!ib || !vb || ib->storage != ValueStorage::WORKSPACE || vb->storage != ValueStorage::WORKSPACE)
        return none;
    const size_t ic = ib->bytes / (size_t(norm.input_row_stride) * 2u);
    const size_t vc = vb->bytes / (size_t(append.common.v_row_stride) * 2u);
    if (ic < ctx.actual_rows || vc != ic) return none;
    const size_t isz = ((ic - 1) * norm.input_row_stride + 1024u) * 2u;
    const size_t vsz = ((vc - 1) * append.common.v_row_stride + 1024u) * 2u;
    const size_t pool = size_t(append.common.elems_per_layer) * append.common.num_attention_layers * 2u;
    if (!disjoint(norm.input, isz, append.k_pool, pool) ||
        !disjoint(norm.input, isz, append.v_pool, pool) ||
        !disjoint(reinterpret_cast<uint64_t>(append.common.v_input), vsz, append.k_pool, pool) ||
        !disjoint(reinterpret_cast<uint64_t>(append.common.v_input), vsz, append.v_pool, pool) ||
        !disjoint(append.k_pool, pool, append.v_pool, pool)) return none;
    attn_kv_preprocess<<<ctx.actual_rows * 4u, 64, 0, stream>>>(
        reinterpret_cast<const bf16*>(norm.input), reinterpret_cast<const bf16*>(norm.weight),
        reinterpret_cast<const double*>(rope.inv_freq), reinterpret_cast<const uint32_t*>(rope.positions),
        append.common, reinterpret_cast<bf16*>(append.k_pool), reinterpret_cast<bf16*>(append.v_pool),
        norm.input_row_stride, norm.eps, rope.attention_factor);
    const auto err = hipGetLastError();
    if (err != hipSuccess) return Status::hip_error("attention kv preprocessing", hipGetErrorString(err), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 3};
}

}
