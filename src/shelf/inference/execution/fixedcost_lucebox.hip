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

__global__ void gdn_norm_gate(const float* input, const bf16* weight, const bf16* gate,
                              bf16* output, uint32_t features, uint32_t group_size,
                              uint32_t input_stride, uint32_t gate_stride, uint32_t output_stride, float eps) {
    const uint32_t lane = threadIdx.x;
    const size_t row = blockIdx.x / (features / group_size);
    const uint32_t base = (blockIdx.x % (features / group_size)) * group_size;
    input += row * input_stride;
    gate += row * gate_stride;
    output += row * output_stride;
    float sum = 0.f;
    for (uint32_t i = lane; i < group_size; i += 32) {
        const float x = input[base + i];
        sum += x * x;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    const float inv = rsqrtf(sum / group_size + eps);
    for (uint32_t i = lane; i < group_size; i += 32) {
        const float z = __bfloat162float(gate[base + i]);
        const float rounded = __bfloat162float(__float2bfloat16(input[base + i] * inv));
        const float norm = rounded * __bfloat162float(weight[i]);
        output[base + i] = __float2bfloat16(norm * (z / (1.f + expf(-z))));
    }
}
__global__ void gdn_qk_norm(const bf16* q, const bf16* k, float* qout, float* kout,
                            uint32_t features, uint32_t q_stride, uint32_t k_stride,
                            uint32_t qo_stride, uint32_t ko_stride, float eps, float scale) {
    const size_t row = blockIdx.x / (2 * (features / 128));
    const uint32_t head = blockIdx.x % (2 * (features / 128));
    q += row * q_stride;
    k += row * k_stride;
    qout += row * qo_stride;
    kout += row * ko_stride;
    const uint32_t lane = threadIdx.x;
    const bool is_key = head >= features / 128;
    const uint32_t base = (is_key ? head - features / 128 : head) * 128;
    const bf16* input = (is_key ? k : q) + base;
    float sum = 0.f;
    float x[4];
    for (uint32_t j = 0; j < 4; ++j) {
        x[j] = __bfloat162float(input[lane * 4 + j]);
        sum += x[j] * x[j];
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    const float factor = rsqrtf(sum + eps) * (is_key ? 1.f : scale);
    float* out = (is_key ? kout : qout) + base;
    for (uint32_t j = 0; j < 4; ++j) out[lane * 4 + j] = x[j] * factor;
}

}

Result<OptimizedLaunchResult> try_fixedcost_lucebox(
    const Program& p, size_t i, size_t end, HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult none{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_LUCEBOX_FUSION") == nullptr;
    if (!enabled || ctx.value_trace != nullptr ||
        ctx.layer_override_count != 0 || i + 3 >= end || i == 0) return none;
    const auto& a = p.dispatches[i];
    const auto& b = p.dispatches[i + 1];
    const auto& c = p.dispatches[i + 2];
    const auto& d = p.dispatches[i + 3];
    if (p.dispatches[i - 1].kernel_id != KernelId::GDN_RECURRENCE ||
        a.kernel_id != KernelId::RMS_NORM || b.kernel_id != KernelId::SILU ||
        c.kernel_id != KernelId::MUL || d.kernel_id != KernelId::LINEAR_EXL3_INT8 ||
        a.input_count != 1 || a.output_count != 1 || b.input_count != 1 ||
        b.output_count != 1 || c.input_count != 2 || c.output_count != 1 ||
        c.input_slots[0] != a.output_slots[0] || c.input_slots[1] != b.output_slots[0] ||
        d.input_count == 0 || d.input_slots[0] != c.output_slots[0]) return none;
    for (size_t j = i + 3; j < p.dispatches.size(); ++j) {
        for (uint32_t k = 0; k < p.dispatches[j].input_count; ++k) {
            const auto v = p.dispatches[j].input_slots[k];
            if (v == a.output_slots[0] || v == b.output_slots[0]) return none;
        }
    }
    PhysicalRmsNormLaunch norm{};
    auto shape_ctx = ctx;
    shape_ctx.actual_rows = 1;
    shape_ctx.actual_outputs = 1;
    auto resolved = resolve_rmsnorm_physical(p, a, shape_ctx, norm);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable ||
        norm.variant != PhysicalRmsNormVariant::F32PgDirect || norm.group_size != 128 ||
        norm.features != 6144) return none;
    auto input = resolve_host_value(p, ValueId{a.input_slots[0]}, ctx);
    auto gate = resolve_host_value(p, ValueId{b.input_slots[0]}, ctx);
    auto out = resolve_host_value(p, ValueId{c.output_slots[0]}, ctx);
    if (!input.ok() || !gate.ok() || !out.ok()) return none;
    const uint32_t rows = input.value().row_domain == ValueRowDomain::OUTPUT_ROWS
        ? ctx.actual_outputs : (ctx.actual_rows != 0 ? ctx.actual_rows : 1u);
    if (rows == 0 || gate.value().row_domain != input.value().row_domain ||
        out.value().row_domain != input.value().row_domain || norm.input_row_stride < norm.features ||
        gate.value().row_stride < norm.features || out.value().row_stride < norm.features) return none;
    if (gate.value().dtype != ValueDType::BF16 || out.value().dtype != ValueDType::BF16 ||
        gate.value().feature_count != norm.features || out.value().feature_count != norm.features ||
        gate.value().ptr == nullptr || out.value().ptr == nullptr ||
        gate.value().ptr == out.value().ptr) return none;
    const uintptr_t dst = reinterpret_cast<uintptr_t>(out.value().ptr);
    const uintptr_t gate_ptr = reinterpret_cast<uintptr_t>(gate.value().ptr);
    const auto overlaps = [](uintptr_t x, size_t n, uintptr_t y, size_t m) {
        return x < y + m && y < x + n;
    };
    const auto* out_binding = p.find_value(ValueId{c.output_slots[0]});
    if (out_binding == nullptr || out_binding->storage != ValueStorage::WORKSPACE) return none;
    const size_t capacity = out_binding->bytes / (size_t(out.value().row_stride) * 2u);
    if (capacity < rows) return none;
    const size_t output_bytes = (capacity - 1) * out.value().row_stride * 2u + norm.features * 2u;
    const size_t gate_bytes = ((capacity - 1) * gate.value().row_stride + norm.features) * 2u;
    const size_t input_bytes = ((capacity - 1) * norm.input_row_stride + norm.features) * 4u;
    if (overlaps(dst, output_bytes, gate_ptr, gate_bytes) ||
        overlaps(dst, output_bytes, norm.input, input_bytes)) return none;
    gdn_norm_gate<<<rows * (norm.features / norm.group_size), 32, 0, stream>>>(
        reinterpret_cast<const float*>(norm.input), reinterpret_cast<const bf16*>(norm.weight),
        static_cast<const bf16*>(gate.value().ptr), static_cast<bf16*>(out.value().ptr),
        norm.features, norm.group_size, norm.input_row_stride, gate.value().row_stride,
        out.value().row_stride, norm.eps);
    const auto err = hipGetLastError();
    if (err != hipSuccess) return Status::hip_error("gdn norm gate", hipGetErrorString(err), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 3};
}
Result<OptimizedLaunchResult> try_fixedcost_lucebox_qk(
    const Program& p, size_t i, size_t end, HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult none{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_LUCEBOX_FUSION") == nullptr;
    if (!enabled || ctx.value_trace != nullptr ||
        ctx.layer_override_count != 0 || i == 0 || i + 5 >= end) return none;
    const auto& q = p.dispatches[i];
    const auto& k = p.dispatches[i + 1];
    const auto& scale = p.dispatches[i + 2];
    const auto& rec = p.dispatches[i + 3];
    if (p.dispatches[i - 1].kernel_id != KernelId::SILU ||
        q.kernel_id != KernelId::L2_NORMALIZE || k.kernel_id != KernelId::L2_NORMALIZE ||
        scale.kernel_id != KernelId::SCALE || rec.kernel_id != KernelId::GDN_RECURRENCE ||
        p.dispatches[i + 4].kernel_id != KernelId::RMS_NORM ||
        p.dispatches[i + 5].kernel_id != KernelId::SILU ||
        q.input_count != 1 || q.output_count != 1 || k.input_count != 1 ||
        k.output_count != 1 || scale.input_count != 1 || scale.output_count != 1 ||
        scale.input_slots[0] != q.output_slots[0] || rec.input_count < 2 ||
        rec.input_slots[0] != scale.output_slots[0] || rec.input_slots[1] != k.output_slots[0]) return none;
    bool exl3 = false;
    for (size_t j = i + 6; j < end && j < i + 10; ++j)
        exl3 |= p.dispatches[j].kernel_id == KernelId::LINEAR_EXL3_INT8;
    if (!exl3) return none;
    for (size_t j = i + 3; j < p.dispatches.size(); ++j)
        for (uint32_t n = 0; n < p.dispatches[j].input_count; ++n)
            if (p.dispatches[j].input_slots[n] == q.output_slots[0]) return none;
    PhysicalL2NormalizeLaunch lq{}, lk{};
    auto shape_ctx = ctx;
    shape_ctx.actual_rows = 1;
    shape_ctx.actual_outputs = 1;
    auto rq = resolve_l2_normalize_physical(p, q, shape_ctx, lq);
    auto rk = resolve_l2_normalize_physical(p, k, shape_ctx, lk);
    if (!rq.ok()) return rq.status();
    if (!rk.ok()) return rk.status();
    if (rq.value() != PhysicalResolveStatus::Applicable || rk.value() != PhysicalResolveStatus::Applicable ||
        lq.features != 2048 || lk.features != lq.features ||
        lq.group_size != 128 || lk.group_size != 128 || lq.eps != lk.eps) return none;
    auto qi = resolve_host_value(p, ValueId{q.input_slots[0]}, ctx);
    auto ki = resolve_host_value(p, ValueId{k.input_slots[0]}, ctx);
    auto ko = resolve_host_value(p, ValueId{k.output_slots[0]}, ctx);
    auto qo = resolve_host_value(p, ValueId{scale.output_slots[0]}, ctx);
    if (!qi.ok() || !ki.ok() || !ko.ok() || !qo.ok() || qo.value().dtype != ValueDType::F32 ||
        qo.value().feature_count != lq.features || qo.value().ptr == nullptr ||
        qo.value().row_stride < lq.features || qi.value().row_domain != ko.value().row_domain ||
        ki.value().row_domain != ko.value().row_domain || qo.value().row_domain != ko.value().row_domain) return none;
    const uint32_t rows = ko.value().row_domain == ValueRowDomain::OUTPUT_ROWS
        ? ctx.actual_outputs : (ctx.actual_rows != 0 ? ctx.actual_rows : 1u);
    if (rows == 0) return none;
    const auto overlaps = [](uintptr_t x, size_t n, uintptr_t y, size_t m) {
        return x < y + m && y < x + n;
    };
    const uintptr_t out = reinterpret_cast<uintptr_t>(qo.value().ptr);
    const auto* qo_binding = p.find_value(ValueId{scale.output_slots[0]});
    const auto* ko_binding = p.find_value(ValueId{k.output_slots[0]});
    if (qo_binding == nullptr || ko_binding == nullptr ||
        qo_binding->storage != ValueStorage::WORKSPACE ||
        ko_binding->storage != ValueStorage::WORKSPACE) return none;
    const size_t qo_capacity = qo_binding->bytes / (size_t(qo.value().row_stride) * 4u);
    const size_t ko_capacity = ko_binding->bytes / (size_t(lk.output_row_stride) * 4u);
    if (qo_capacity != ko_capacity || qo_capacity < rows) return none;
    const size_t capacity = qo_capacity;
    const size_t qi_bytes = ((capacity - 1) * lq.input_row_stride + lq.features) * 2u;
    const size_t ki_bytes = ((capacity - 1) * lk.input_row_stride + lq.features) * 2u;
    const size_t qo_bytes = ((capacity - 1) * qo.value().row_stride + lq.features) * 4u;
    const size_t ko_bytes = ((capacity - 1) * lk.output_row_stride + lq.features) * 4u;
    if (overlaps(out, qo_bytes, lk.output, ko_bytes) ||
        overlaps(out, qo_bytes, lq.input, qi_bytes) ||
        overlaps(out, qo_bytes, lk.input, ki_bytes) ||
        overlaps(lk.output, ko_bytes, lq.input, qi_bytes) ||
        overlaps(lk.output, ko_bytes, lk.input, ki_bytes)) return none;
    gdn_qk_norm<<<rows * (2 * lq.features / 128), 32, 0, stream>>>(
        reinterpret_cast<const bf16*>(lq.input), reinterpret_cast<const bf16*>(lk.input),
        static_cast<float*>(qo.value().ptr), reinterpret_cast<float*>(lk.output),
        lq.features, lq.input_row_stride, lk.input_row_stride, qo.value().row_stride,
        lk.output_row_stride, lq.eps, scale.scalar_a);
    const auto err = hipGetLastError();
    if (err != hipSuccess) return Status::hip_error("gdn qk norm", hipGetErrorString(err), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 3};
}

}
