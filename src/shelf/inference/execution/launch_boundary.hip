#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/boundary_deferred.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include "../../kernels/optimized/detail/rmsnorm_device.h"
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdlib>

namespace shelf::execution {
namespace {
using namespace ::shelf::runtime;
using bf16 = __hip_bfloat16;

__global__ void residual_norm(const bf16* a, const bf16* b, bf16* residual,
                              const bf16* weight, bf16* output, uint32_t as,
                              uint32_t bs, uint32_t rs, uint32_t os, float eps) {
    __shared__ bf16 x[5120];
    __shared__ bf16 y[5120];
    const uint32_t row = blockIdx.y;
    for (uint32_t i = threadIdx.x; i < 5120; i += blockDim.x)
        x[i] = __float2bfloat16(__bfloat162float(a[row * as + i]) +
                                __bfloat162float(b[row * bs + i]));
    __syncthreads();
    ::shelf::kernel::detail::rmsnorm_row_body<
        ::shelf::kernel::RmsNormDataType::BF16, ::shelf::kernel::RmsNormDataType::BF16,
        ::shelf::kernel::RmsNormDataType::BF16, ::shelf::kernel::RmsNormWeightLayout::PER_FEATURE,
        ::shelf::kernel::RmsNormWeightMode::ONE_PLUS>(x, weight, y, 0, 0, 5120, 5120, 5120, eps);
    __syncthreads();
    for (uint32_t i = blockIdx.x * 1024 + threadIdx.x; i < (blockIdx.x + 1) * 1024; i += blockDim.x) {
        residual[row * rs + i] = x[i];
        output[row * os + i] = y[i];
    }
}
}

hipError_t launch_boundary_out_rotate(const ::shelf::kernel::PlamoExl3ProjectionGroup&, const BoundaryDeferred&,
    const bf16*, bf16*, const bf16*, bf16*, uint32_t, uint32_t, uint32_t, float, bool, bool, hipStream_t);

hipError_t launch_boundary_rotate_fast(const ::shelf::kernel::PlamoExl3ProjectionGroup&,
    const bf16*, const bf16*, bf16*, const bf16*, bf16*, uint32_t, uint32_t, uint32_t, uint32_t, float, hipStream_t);

hipError_t launch_boundary_out_rotate_fast(const ::shelf::kernel::PlamoExl3ProjectionGroup&, const BoundaryDeferred&,
    const bf16*, bf16*, const bf16*, bf16*, uint32_t, uint32_t, uint32_t, float, bool, hipStream_t);

Result<OptimizedLaunchResult> try_launch_boundary_core(
    const Program& p, size_t i, size_t end, HostExecutionContext& ctx, hipStream_t stream,
    const ::shelf::kernel::PlamoExl3ProjectionGroup* group, const BoundaryDeferred* source) {
    const OptimizedLaunchResult none{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_BOUNDARY_I1") == nullptr;
    static const bool rotate_enabled = std::getenv("SHELF_DISABLE_BOUNDARY_I8") == nullptr;
    static const bool out_enabled = std::getenv("SHELF_DISABLE_BOUNDARY_I9") == nullptr;
    if ((source && !out_enabled) || (group && !rotate_enabled) || !enabled || ctx.value_trace || ctx.layer_override_count || i == 0 || i + 3 >= end ||
        ctx.actual_rows == 0 || ctx.actual_rows > 16 || ctx.tp_active) return none;
    const auto& add = p.dispatches[i];
    const auto& norm = p.dispatches[i + 1];
    const auto& proj = p.dispatches[i + 2];
    const auto& proj2 = p.dispatches[i + 3];
    if (p.dispatches[i - 1].kernel_id != KernelId::LINEAR_EXL3_INT8 ||
        add.kernel_id != KernelId::RESIDUAL_ADD || norm.kernel_id != KernelId::RMS_NORM ||
        proj.kernel_id != KernelId::LINEAR_EXL3_INT8 || proj2.kernel_id != KernelId::LINEAR_EXL3_INT8 ||
        add.output_count != 1 || norm.input_count != 1 || norm.output_count != 1 ||
        proj.input_count == 0 || proj2.input_count == 0 ||
        add.output_slots[0] != norm.input_slots[0] || norm.output_slots[0] != proj.input_slots[0] ||
        norm.output_slots[0] != proj2.input_slots[0]) return none;
    bool gdn = false;
    for (const auto& d : p.dispatches) {
        if (d.kernel_id == KernelId::MOE_ROUTER) return none;
        gdn |= d.kernel_id == KernelId::GDN_RECURRENCE;
    }
    if (!gdn) return none;
    auto shape = ctx;
    shape.actual_rows = 1;
    shape.actual_outputs = 1;
    PhysicalElementwiseLaunch a{};
    PhysicalRmsNormLaunch n{};
    auto ar = resolve_elementwise_physical(p, add, shape, a);
    auto nr = resolve_rmsnorm_physical(p, norm, shape, n);
    if (!ar.ok() || !nr.ok() || ar.value() != PhysicalResolveStatus::Applicable ||
        nr.value() != PhysicalResolveStatus::Applicable || a.profile != ElementwiseProfile::ResidualBf16 ||
        a.features != 5120 || n.features != 5120 || n.group_size != 5120 ||
        n.variant != PhysicalRmsNormVariant::Bf16PfOnePlus || a.out0 != n.input || !n.weight) return none;
    auto av = resolve_host_value(p, ValueId{add.input_slots[0]}, ctx);
    auto bv = resolve_host_value(p, ValueId{add.input_slots[1]}, ctx);
    auto rv = resolve_host_value(p, ValueId{add.output_slots[0]}, ctx);
    auto nv = resolve_host_value(p, ValueId{norm.output_slots[0]}, ctx);
    if (!av.ok() || !bv.ok() || !rv.ok() || !nv.ok() ||
        av.value().row_domain != rv.value().row_domain || bv.value().row_domain != rv.value().row_domain ||
        nv.value().row_domain != rv.value().row_domain) return none;
    const uint32_t rows = rv.value().row_domain == ValueRowDomain::OUTPUT_ROWS ? ctx.actual_outputs : ctx.actual_rows;
    if (rows == 0 || rows > 16) return none;
    const auto overlap = [](uint64_t x, size_t xn, uint64_t y, size_t yn) {
        return x < y + yn && y < x + xn;
    };
    const size_t ab = ((rows - 1) * size_t(a.in0_stride) + 5120) * 2;
    const size_t bb = ((rows - 1) * size_t(a.in1_stride) + 5120) * 2;
    const size_t rb = ((rows - 1) * size_t(a.out0_stride) + 5120) * 2;
    const size_t nb = ((rows - 1) * size_t(n.output_row_stride) + 5120) * 2;
    if (overlap(a.out0, rb, a.in0, ab) || overlap(a.out0, rb, a.in1, bb) ||
        overlap(n.output, nb, a.in0, ab) || overlap(n.output, nb, a.in1, bb) ||
        overlap(n.output, nb, a.out0, rb)) return none;
    hipError_t err;
    if (group) {
        if (group->rows != rows || group->k != 5120 || group->count < 2 ||
            reinterpret_cast<uint64_t>(group->input) != n.output) return none;
        if (source) {
            if (p.dispatches[i - 1].output_count != 1 || add.input_count != 2 ||
                (add.input_slots[0] != p.dispatches[i - 1].output_slots[0] &&
                 add.input_slots[1] != p.dispatches[i - 1].output_slots[0]) ||
                source->rows != rows || source->features != 5120 || !source->yprime || !source->svh ||
                (source->output != reinterpret_cast<void*>(a.in0) && source->output != reinterpret_cast<void*>(a.in1)) ||
                a.in0 == a.in1) return none;
            const auto* produced = p.find_value(ValueId{p.dispatches[i - 1].output_slots[0]});
            if (!produced || (produced->storage != ValueStorage::WORKSPACE &&
                              produced->storage != ValueStorage::WORKSPACE_VIEW)) return none;
            for (size_t j = i + 1; j < p.dispatches.size(); ++j)
                for (uint32_t k = 0; k < p.dispatches[j].input_count; ++k)
                    if (p.dispatches[j].input_slots[k] == p.dispatches[i - 1].output_slots[0]) return none;
            const bool first = source->output == reinterpret_cast<void*>(a.in0);
            static const bool serial = std::getenv("SHELF_BOUNDARY_I9_SERIAL") != nullptr;
            if (serial) err = launch_boundary_out_rotate(*group, *source,
                reinterpret_cast<const bf16*>(first ? a.in1 : a.in0), reinterpret_cast<bf16*>(a.out0),
                reinterpret_cast<const bf16*>(n.weight), reinterpret_cast<bf16*>(n.output),
                first ? a.in1_stride : a.in0_stride, a.out0_stride, n.output_row_stride, n.eps, first, true, stream);
            else err = launch_boundary_out_rotate_fast(*group, *source,
                reinterpret_cast<const bf16*>(first ? a.in1 : a.in0), reinterpret_cast<bf16*>(a.out0),
                reinterpret_cast<const bf16*>(n.weight), reinterpret_cast<bf16*>(n.output),
                first ? a.in1_stride : a.in0_stride, a.out0_stride, n.output_row_stride, n.eps, first, stream);
        } else err = launch_boundary_rotate_fast(*group, reinterpret_cast<const bf16*>(a.in0),
            reinterpret_cast<const bf16*>(a.in1), reinterpret_cast<bf16*>(a.out0),
            reinterpret_cast<const bf16*>(n.weight), reinterpret_cast<bf16*>(n.output),
            a.in0_stride, a.in1_stride, a.out0_stride, n.output_row_stride, n.eps, stream);
    } else {
        residual_norm<<<dim3(5, rows), 256, 0, stream>>>(
            reinterpret_cast<const bf16*>(a.in0), reinterpret_cast<const bf16*>(a.in1),
            reinterpret_cast<bf16*>(a.out0), reinterpret_cast<const bf16*>(n.weight),
            reinterpret_cast<bf16*>(n.output), a.in0_stride, a.in1_stride, a.out0_stride,
            n.output_row_stride, n.eps);
        err = hipGetLastError();
    }
    if (err != hipSuccess) return Status::hip_error("boundary i1", hipGetErrorString(err), __FILE__, __LINE__);
    if (!group) record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 2};
}
Result<OptimizedLaunchResult> try_launch_boundary_i1(
    const Program& p, size_t i, size_t end, HostExecutionContext& ctx, hipStream_t stream) {
    return try_launch_boundary_core(p, i, end, ctx, stream, nullptr, nullptr);
}
Result<OptimizedLaunchResult> try_launch_boundary_i8(
    const Program& p, size_t i, size_t end, HostExecutionContext& ctx, hipStream_t stream,
    const ::shelf::kernel::PlamoExl3ProjectionGroup& group) {
    return try_launch_boundary_core(p, i, end, ctx, stream, &group, nullptr);
}
Result<OptimizedLaunchResult> try_launch_boundary_i9(
    const Program& p, size_t i, size_t end, HostExecutionContext& ctx, hipStream_t stream,
    const ::shelf::kernel::PlamoExl3ProjectionGroup& group, const BoundaryDeferred& source) {
    return try_launch_boundary_core(p, i, end, ctx, stream, &group, &source);
}

}
