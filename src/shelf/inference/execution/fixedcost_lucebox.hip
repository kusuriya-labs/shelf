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
                              bf16* output, uint32_t group_size, float eps) {
    const uint32_t lane = threadIdx.x;
    const uint32_t base = blockIdx.x * group_size;
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
}

Result<OptimizedLaunchResult> try_fixedcost_lucebox(
    const Program& p, size_t i, size_t end, HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult none{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_LUCEBOX_FUSION") == nullptr;
    if (!enabled || ctx.actual_rows != 1 || ctx.value_trace != nullptr ||
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
    auto resolved = resolve_rmsnorm_physical(p, a, ctx, norm);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable ||
        norm.variant != PhysicalRmsNormVariant::F32PgDirect || norm.group_size != 128 ||
        norm.features != 6144 || norm.rows != 1) return none;
    auto gate = resolve_host_value(p, ValueId{b.input_slots[0]}, ctx);
    auto out = resolve_host_value(p, ValueId{c.output_slots[0]}, ctx);
    if (!gate.ok() || !out.ok()) return none;
    if (gate.value().dtype != ValueDType::BF16 || out.value().dtype != ValueDType::BF16 ||
        gate.value().feature_count != norm.features || out.value().feature_count != norm.features ||
        gate.value().ptr == nullptr || out.value().ptr == nullptr ||
        gate.value().ptr == out.value().ptr) return none;
    const uintptr_t dst = reinterpret_cast<uintptr_t>(out.value().ptr);
    const uintptr_t gate_ptr = reinterpret_cast<uintptr_t>(gate.value().ptr);
    const auto overlaps = [](uintptr_t x, size_t n, uintptr_t y, size_t m) {
        return x < y + m && y < x + n;
    };
    if (overlaps(dst, norm.features * 2u, gate_ptr, norm.features * 2u) ||
        overlaps(dst, norm.features * 2u, norm.input, norm.features * 4u)) return none;
    gdn_norm_gate<<<norm.features / norm.group_size, 32, 0, stream>>>(
        reinterpret_cast<const float*>(norm.input), reinterpret_cast<const bf16*>(norm.weight),
        static_cast<const bf16*>(gate.value().ptr), static_cast<bf16*>(out.value().ptr),
        norm.group_size, norm.eps);
    const auto err = hipGetLastError();
    if (err != hipSuccess) return Status::hip_error("gdn norm gate", hipGetErrorString(err), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 3};
}
}
