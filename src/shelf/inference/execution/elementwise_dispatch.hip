#include <shelf/inference/execution/elementwise_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/selectors/elementwise_selector.h>
#include <shelf/inference/execution/selectors/embedding_selector.h>
#include <shelf/inference/execution/selectors/gdn_conv_selector.h>
#include <shelf/inference/execution/selectors/gdn_recurrence_selector.h>
#include <shelf/inference/execution/selectors/kv_append_selector.h>
#include <shelf/inference/execution/selectors/l2_normalize_selector.h>
#include <shelf/inference/execution/selectors/linear_selector.h>
#include <shelf/inference/execution/selectors/paged_attention_selector.h>
#include <shelf/inference/execution/selectors/rmsnorm_selector.h>
#include <shelf/inference/execution/selectors/rope_selector.h>
#include <shelf/inference/execution/selectors/sampling_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstddef>

namespace shelf::execution {
namespace {

using ::shelf::kernel::ElementwiseSplitLayout;
using ::shelf::runtime::ValueDType;
using ::shelf::runtime::ValueRowDomain;

constexpr OptimizedLaunchResult kNone{OptimizedLaunchKind::NotApplicable};

uint32_t rows_for(const HostResolvedValue& v, const HostExecutionContext& ctx) {
    if (v.row_domain == ValueRowDomain::OUTPUT_ROWS) return ctx.actual_outputs;
    return ctx.actual_rows != 0 ? ctx.actual_rows : 1u;
}

size_t elem_bytes(const HostResolvedValue& v) {
    return (v.dtype == ValueDType::F32 || v.dtype == ValueDType::I32) ? 4u : 2u;
}

struct ByteRange {
    uintptr_t start = 0;
    uintptr_t end = 0;
};

ByteRange value_range(const HostResolvedValue& v, uint32_t rows) {
    const size_t eb = elem_bytes(v);
    const size_t extent =
        (static_cast<size_t>(rows - 1u) * v.row_stride + v.feature_count) * eb;
    const uintptr_t base = reinterpret_cast<uintptr_t>(v.ptr);
    return ByteRange{base, base + extent};
}

bool ranges_ok(const ByteRange& a, const ByteRange& b) {
    const bool same = (a.start == b.start && a.end == b.end);
    const bool disjoint = (a.end <= b.start) || (b.end <= a.start);
    return same || disjoint;
}

bool all_alias_ok(const ByteRange* rs, size_t n) {
    for (size_t i = 0u; i < n; ++i)
        for (size_t j = i + 1u; j < n; ++j)
            if (!ranges_ok(rs[i], rs[j])) return false;
    return true;
}

bool same_shape(const HostResolvedValue& x, const HostResolvedValue& y) {
    return x.feature_count == y.feature_count && x.row_domain == y.row_domain;
}

void fill_geometry(PhysicalElementwiseLaunch& d) {
    d.grid_x = ::shelf::kernel::elementwise_grid_x(d.features);
    d.grid_y = d.rows;
    d.workgroup = ::shelf::kernel::kElementwiseBlock;
}

PhysicalResolveStatus select_profile(const PhysicalElementwiseLaunch& d) {
    ElementwiseSelectorInput si;
    si.profile = d.profile;
    si.rows = d.rows;
    si.features = d.features;
    si.aux = (d.profile == ElementwiseProfile::SplitBf16InterleavedHeads) ? d.aux : 0u;
    const ElementwiseImplementation impl = select_elementwise_implementation(si);
    return impl == ElementwiseImplementation::Optimized
               ? PhysicalResolveStatus::Applicable
               : PhysicalResolveStatus::NotApplicable;
}

Result<PhysicalResolveStatus> resolve_unary_bf16_to_f32(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, ElementwiseProfile profile,
    PhysicalElementwiseLaunch& out) {
    auto in_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue in = in_res.value();
    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue dst = out_res.value();
    if (in.dtype != ValueDType::BF16 || dst.dtype != ValueDType::F32)
        return PhysicalResolveStatus::NotApplicable;
    if (!same_shape(in, dst)) return PhysicalResolveStatus::NotApplicable;
    if (in.feature_count == 0u || in.row_stride < in.feature_count ||
        dst.row_stride < in.feature_count) {
        return PhysicalResolveStatus::NotApplicable;
    }
    const uint32_t rows = rows_for(dst, ctx);
    if (rows == 0u) return PhysicalResolveStatus::NotApplicable;
    const ByteRange ranges[2] = {value_range(in, rows), value_range(dst, rows)};
    if (!all_alias_ok(ranges, 2)) return PhysicalResolveStatus::NotApplicable;
    out.profile = profile;
    out.in0 = reinterpret_cast<uint64_t>(in.ptr);
    out.in0_stride = in.row_stride;
    out.out0 = reinterpret_cast<uint64_t>(dst.ptr);
    out.out0_stride = dst.row_stride;
    out.rows = rows;
    out.features = in.feature_count;
    fill_geometry(out);
    return select_profile(out);
}

Result<PhysicalResolveStatus> resolve_unary_f32_to_bf16(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, PhysicalElementwiseLaunch& out) {
    auto in_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue in = in_res.value();
    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue dst = out_res.value();
    if (in.dtype != ValueDType::F32 || dst.dtype != ValueDType::BF16)
        return PhysicalResolveStatus::NotApplicable;
    if (!same_shape(in, dst)) return PhysicalResolveStatus::NotApplicable;
    if (in.feature_count == 0u || in.row_stride < in.feature_count ||
        dst.row_stride < in.feature_count) {
        return PhysicalResolveStatus::NotApplicable;
    }
    const uint32_t rows = rows_for(dst, ctx);
    if (rows == 0u) return PhysicalResolveStatus::NotApplicable;
    const ByteRange ranges[2] = {value_range(in, rows), value_range(dst, rows)};
    if (!all_alias_ok(ranges, 2)) return PhysicalResolveStatus::NotApplicable;
    out.profile = ElementwiseProfile::SiluF32ToBf16;
    out.in0 = reinterpret_cast<uint64_t>(in.ptr);
    out.in0_stride = in.row_stride;
    out.out0 = reinterpret_cast<uint64_t>(dst.ptr);
    out.out0_stride = dst.row_stride;
    out.rows = rows;
    out.features = in.feature_count;
    fill_geometry(out);
    return select_profile(out);
}

Result<PhysicalResolveStatus> resolve_unary_f32_scale(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, PhysicalElementwiseLaunch& out) {
    auto in_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue in = in_res.value();
    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue dst = out_res.value();
    if (in.dtype != ValueDType::F32 || dst.dtype != ValueDType::F32)
        return PhysicalResolveStatus::NotApplicable;
    if (!same_shape(in, dst)) return PhysicalResolveStatus::NotApplicable;
    if (in.feature_count == 0u || in.row_stride < in.feature_count ||
        dst.row_stride < in.feature_count) {
        return PhysicalResolveStatus::NotApplicable;
    }
    const uint32_t rows = rows_for(dst, ctx);
    if (rows == 0u) return PhysicalResolveStatus::NotApplicable;
    const ByteRange ranges[2] = {value_range(in, rows), value_range(dst, rows)};
    if (!all_alias_ok(ranges, 2)) return PhysicalResolveStatus::NotApplicable;
    out.profile = ElementwiseProfile::ScaleF32;
    out.in0 = reinterpret_cast<uint64_t>(in.ptr);
    out.in0_stride = in.row_stride;
    out.out0 = reinterpret_cast<uint64_t>(dst.ptr);
    out.out0_stride = dst.row_stride;
    out.rows = rows;
    out.features = in.feature_count;
    out.scalar_a = b.scalar_a;
    fill_geometry(out);
    return select_profile(out);
}

Result<PhysicalResolveStatus> resolve_binary_bf16(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, ElementwiseProfile profile,
    PhysicalElementwiseLaunch& out) {
    auto a_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!a_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue a = a_res.value();
    auto b_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[1]}, ctx);
    if (!b_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue bb = b_res.value();
    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue dst = out_res.value();
    if (a.dtype != ValueDType::BF16 || bb.dtype != ValueDType::BF16 ||
        dst.dtype != ValueDType::BF16) {
        return PhysicalResolveStatus::NotApplicable;
    }
    if (!same_shape(a, dst) || !same_shape(bb, dst))
        return PhysicalResolveStatus::NotApplicable;
    if (dst.feature_count == 0u || a.row_stride < dst.feature_count ||
        bb.row_stride < dst.feature_count || dst.row_stride < dst.feature_count) {
        return PhysicalResolveStatus::NotApplicable;
    }
    const uint32_t rows = rows_for(dst, ctx);
    if (rows == 0u) return PhysicalResolveStatus::NotApplicable;
    const ByteRange ranges[3] = {value_range(a, rows), value_range(bb, rows),
                                 value_range(dst, rows)};
    if (!all_alias_ok(ranges, 3)) return PhysicalResolveStatus::NotApplicable;
    out.profile = profile;
    out.in0 = reinterpret_cast<uint64_t>(a.ptr);
    out.in0_stride = a.row_stride;
    out.in1 = reinterpret_cast<uint64_t>(bb.ptr);
    out.in1_stride = bb.row_stride;
    out.out0 = reinterpret_cast<uint64_t>(dst.ptr);
    out.out0_stride = dst.row_stride;
    out.rows = rows;
    out.features = dst.feature_count;
    fill_geometry(out);
    return select_profile(out);
}

Result<PhysicalResolveStatus> resolve_binary_f32_to_bf16(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, PhysicalElementwiseLaunch& out) {
    auto a_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!a_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue a = a_res.value();
    auto b_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[1]}, ctx);
    if (!b_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue bb = b_res.value();
    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue dst = out_res.value();
    if (a.dtype != ValueDType::F32 || bb.dtype != ValueDType::F32 ||
        dst.dtype != ValueDType::BF16) {
        return PhysicalResolveStatus::NotApplicable;
    }
    if (!same_shape(a, dst) || !same_shape(bb, dst))
        return PhysicalResolveStatus::NotApplicable;
    if (dst.feature_count == 0u || a.row_stride < dst.feature_count ||
        bb.row_stride < dst.feature_count || dst.row_stride < dst.feature_count) {
        return PhysicalResolveStatus::NotApplicable;
    }
    const uint32_t rows = rows_for(dst, ctx);
    if (rows == 0u) return PhysicalResolveStatus::NotApplicable;
    const ByteRange ranges[3] = {value_range(a, rows), value_range(bb, rows),
                                 value_range(dst, rows)};
    if (!all_alias_ok(ranges, 3)) return PhysicalResolveStatus::NotApplicable;
    out.profile = ElementwiseProfile::MulF32F32ToBf16;
    out.in0 = reinterpret_cast<uint64_t>(a.ptr);
    out.in0_stride = a.row_stride;
    out.in1 = reinterpret_cast<uint64_t>(bb.ptr);
    out.in1_stride = bb.row_stride;
    out.out0 = reinterpret_cast<uint64_t>(dst.ptr);
    out.out0_stride = dst.row_stride;
    out.rows = rows;
    out.features = dst.feature_count;
    fill_geometry(out);
    return select_profile(out);
}

Result<PhysicalResolveStatus> resolve_split_desc(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, PhysicalElementwiseLaunch& out) {
    if (b.input_count != 1u || b.output_count != 2u)
        return PhysicalResolveStatus::NotApplicable;
    auto in_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue in = in_res.value();
    auto o0_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!o0_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue o0 = o0_res.value();
    auto o1_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[1]}, ctx);
    if (!o1_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue o1 = o1_res.value();
    if (in.dtype != ValueDType::BF16 || o0.dtype != ValueDType::BF16 ||
        o1.dtype != ValueDType::BF16) {
        return PhysicalResolveStatus::NotApplicable;
    }
    if (o0.feature_count != o1.feature_count || o0.row_domain != o1.row_domain)
        return PhysicalResolveStatus::NotApplicable;
    if (o0.row_domain != in.row_domain || o1.row_domain != in.row_domain)
        return PhysicalResolveStatus::NotApplicable;
    const uint32_t f0 = o0.feature_count;
    if (f0 == 0u || in.feature_count != 2u * f0)
        return PhysicalResolveStatus::NotApplicable;
    if (in.row_stride < in.feature_count || o0.row_stride < f0 || o1.row_stride < f0)
        return PhysicalResolveStatus::NotApplicable;
    const bool interleaved =
        (b.flags == static_cast<uint32_t>(::shelf::runtime::SplitLayout::INTERLEAVED_HEADS));
    const uint32_t head_dim = b.group_size;
    if (interleaved) {
        if (head_dim == 0u || f0 % head_dim != 0u)
            return PhysicalResolveStatus::NotApplicable;
    }
    const uint32_t rows = rows_for(o0, ctx);
    if (rows == 0u) return PhysicalResolveStatus::NotApplicable;
    const ByteRange ranges[3] = {value_range(in, rows), value_range(o0, rows),
                                 value_range(o1, rows)};
    if (!all_alias_ok(ranges, 3)) return PhysicalResolveStatus::NotApplicable;
    out.profile = interleaved ? ElementwiseProfile::SplitBf16InterleavedHeads
                              : ElementwiseProfile::SplitBf16Halves;
    out.in0 = reinterpret_cast<uint64_t>(in.ptr);
    out.in0_stride = in.row_stride;
    out.out0 = reinterpret_cast<uint64_t>(o0.ptr);
    out.out0_stride = o0.row_stride;
    out.out1 = reinterpret_cast<uint64_t>(o1.ptr);
    out.out1_stride = o1.row_stride;
    out.rows = rows;
    out.features = f0;
    out.aux = head_dim;
    fill_geometry(out);
    return select_profile(out);
}

Result<PhysicalResolveStatus> resolve_silu_desc(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, PhysicalElementwiseLaunch& out) {
    auto in_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue in = in_res.value();
    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue dst = out_res.value();
    if (in.dtype == ValueDType::BF16 && dst.dtype == ValueDType::F32)
        return resolve_unary_bf16_to_f32(
            program, b, ctx, ElementwiseProfile::SiluBf16ToF32, out);
    if (in.dtype == ValueDType::F32 && dst.dtype == ValueDType::BF16)
        return resolve_unary_f32_to_bf16(program, b, ctx, out);
    return PhysicalResolveStatus::NotApplicable;
}

Result<PhysicalResolveStatus> resolve_concat_desc(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, PhysicalElementwiseLaunch& out) {
    if (b.input_count != 2u || b.output_count != 1u)
        return PhysicalResolveStatus::NotApplicable;
    auto a_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!a_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue a = a_res.value();
    auto b_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[1]}, ctx);
    if (!b_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue bb = b_res.value();
    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue dst = out_res.value();
    if (a.dtype != ValueDType::BF16 || bb.dtype != ValueDType::BF16 ||
        dst.dtype != ValueDType::BF16) {
        return PhysicalResolveStatus::NotApplicable;
    }
    if (a.row_domain != dst.row_domain || bb.row_domain != dst.row_domain)
        return PhysicalResolveStatus::NotApplicable;
    const uint32_t fa = a.feature_count;
    const uint32_t fb = bb.feature_count;
    if (fa == 0u || fb == 0u || dst.feature_count != fa + fb)
        return PhysicalResolveStatus::NotApplicable;
    if (a.row_stride < fa || bb.row_stride < fb || dst.row_stride < dst.feature_count)
        return PhysicalResolveStatus::NotApplicable;
    const uint32_t rows = rows_for(dst, ctx);
    if (rows == 0u) return PhysicalResolveStatus::NotApplicable;
    const ByteRange ranges[3] = {value_range(a, rows), value_range(bb, rows),
                                 value_range(dst, rows)};
    if (!all_alias_ok(ranges, 3)) return PhysicalResolveStatus::NotApplicable;
    out.profile = ElementwiseProfile::Count;
    out.in0 = reinterpret_cast<uint64_t>(a.ptr);
    out.in0_stride = a.row_stride;
    out.in1 = reinterpret_cast<uint64_t>(bb.ptr);
    out.in1_stride = bb.row_stride;
    out.out0 = reinterpret_cast<uint64_t>(dst.ptr);
    out.out0_stride = dst.row_stride;
    out.rows = rows;
    out.features = fa;
    out.features_b = fb;
    fill_geometry(out);
    out.grid_x = ::shelf::kernel::elementwise_grid_x(fa + fb);
    return PhysicalResolveStatus::Applicable;
}

hipError_t launch_elementwise_physical(const PhysicalElementwiseLaunch& d,
                                       hipStream_t stream) {
    switch (d.profile) {
        case ElementwiseProfile::ResidualBf16:
            return ::shelf::kernel::launch_residual_add_bf16(
                reinterpret_cast<const ::shelf::bf16_t*>(d.in0), d.in0_stride,
                reinterpret_cast<const ::shelf::bf16_t*>(d.in1), d.in1_stride,
                reinterpret_cast<::shelf::bf16_t*>(d.out0), d.out0_stride, d.rows,
                d.features, stream);
        case ElementwiseProfile::SwigluBf16:
            return ::shelf::kernel::launch_swiglu_bf16(
                reinterpret_cast<const ::shelf::bf16_t*>(d.in0), d.in0_stride,
                reinterpret_cast<const ::shelf::bf16_t*>(d.in1), d.in1_stride,
                reinterpret_cast<::shelf::bf16_t*>(d.out0), d.out0_stride, d.rows,
                d.features, stream);
        case ElementwiseProfile::SigmoidBf16ToF32:
            return ::shelf::kernel::launch_sigmoid_bf16_f32(
                reinterpret_cast<const ::shelf::bf16_t*>(d.in0), d.in0_stride,
                reinterpret_cast<float*>(d.out0), d.out0_stride, d.rows, d.features,
                stream);
        case ElementwiseProfile::SiluBf16ToF32:
            return ::shelf::kernel::launch_silu_bf16_f32(
                reinterpret_cast<const ::shelf::bf16_t*>(d.in0), d.in0_stride,
                reinterpret_cast<float*>(d.out0), d.out0_stride, d.rows, d.features,
                stream);
        case ElementwiseProfile::SiluF32ToBf16:
            return ::shelf::kernel::launch_silu_f32_bf16(
                reinterpret_cast<const float*>(d.in0), d.in0_stride,
                reinterpret_cast<::shelf::bf16_t*>(d.out0), d.out0_stride, d.rows,
                d.features, stream);
        case ElementwiseProfile::MulF32F32ToBf16:
            return ::shelf::kernel::launch_mul_f32_f32_bf16(
                reinterpret_cast<const float*>(d.in0), d.in0_stride,
                reinterpret_cast<const float*>(d.in1), d.in1_stride,
                reinterpret_cast<::shelf::bf16_t*>(d.out0), d.out0_stride, d.rows,
                d.features, stream);
        case ElementwiseProfile::ScaleF32:
            return ::shelf::kernel::launch_scale_f32(
                reinterpret_cast<const float*>(d.in0), d.in0_stride,
                reinterpret_cast<float*>(d.out0), d.out0_stride, d.rows, d.features,
                d.scalar_a, stream);
        case ElementwiseProfile::SplitBf16Halves:
        case ElementwiseProfile::SplitBf16InterleavedHeads:
            return ::shelf::kernel::launch_split_bf16(
                reinterpret_cast<const ::shelf::bf16_t*>(d.in0), d.in0_stride,
                reinterpret_cast<::shelf::bf16_t*>(d.out0), d.out0_stride,
                reinterpret_cast<::shelf::bf16_t*>(d.out1), d.out1_stride, d.rows,
                d.features,
                d.profile == ElementwiseProfile::SplitBf16InterleavedHeads
                    ? ElementwiseSplitLayout::InterleavedHeads
                    : ElementwiseSplitLayout::Halves,
                d.aux, stream);
        case ElementwiseProfile::Count:
            return ::shelf::kernel::launch_concat_bf16(
                reinterpret_cast<const ::shelf::bf16_t*>(d.in0), d.in0_stride,
                reinterpret_cast<const ::shelf::bf16_t*>(d.in1), d.in1_stride,
                reinterpret_cast<::shelf::bf16_t*>(d.out0), d.out0_stride, d.rows,
                d.features, d.features_b, stream);
        default:
            return hipErrorNotSupported;
    }
}

}  // namespace

Result<PhysicalResolveStatus> resolve_elementwise_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalElementwiseLaunch& out) {
    out = PhysicalElementwiseLaunch{};
    switch (b.kernel_id) {
        case ::shelf::runtime::KernelId::RESIDUAL_ADD:
            if (b.input_count != 2u || b.output_count != 1u)
                return PhysicalResolveStatus::NotApplicable;
            return resolve_binary_bf16(program, b, ctx,
                                       ElementwiseProfile::ResidualBf16, out);
        case ::shelf::runtime::KernelId::SWIGLU:
            if (b.input_count != 2u || b.output_count != 1u)
                return PhysicalResolveStatus::NotApplicable;
            return resolve_binary_bf16(program, b, ctx,
                                       ElementwiseProfile::SwigluBf16, out);
        case ::shelf::runtime::KernelId::SIGMOID:
            if (b.input_count != 1u || b.output_count != 1u)
                return PhysicalResolveStatus::NotApplicable;
            return resolve_unary_bf16_to_f32(
                program, b, ctx, ElementwiseProfile::SigmoidBf16ToF32, out);
        case ::shelf::runtime::KernelId::SILU:
            if (b.input_count != 1u || b.output_count != 1u)
                return PhysicalResolveStatus::NotApplicable;
            return resolve_silu_desc(program, b, ctx, out);
        case ::shelf::runtime::KernelId::MUL:
            if (b.input_count != 2u || b.output_count != 1u)
                return PhysicalResolveStatus::NotApplicable;
            return resolve_binary_f32_to_bf16(program, b, ctx, out);
        case ::shelf::runtime::KernelId::SCALE:
            if (b.input_count != 1u || b.output_count != 1u)
                return PhysicalResolveStatus::NotApplicable;
            return resolve_unary_f32_scale(program, b, ctx, out);
        case ::shelf::runtime::KernelId::SPLIT:
            return resolve_split_desc(program, b, ctx, out);
        case ::shelf::runtime::KernelId::CONCAT:
            return resolve_concat_desc(program, b, ctx, out);
        default:
            return PhysicalResolveStatus::NotApplicable;
    }
}

Result<OptimizedLaunchResult> try_launch_elementwise(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    if (dispatch_index >= program.dispatches.size()) return kNone;
    const auto& b = program.dispatches[dispatch_index];
    PhysicalElementwiseLaunch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_elementwise_physical(program, b, ctx, launch);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable) return kNone;
    const hipError_t herr = launch_elementwise_physical(launch, stream);
    if (herr == hipErrorNotSupported) return kNone;
    if (herr != hipSuccess)
        return Status::hip_error("elementwise optimized launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
