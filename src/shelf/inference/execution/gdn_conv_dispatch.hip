#include <shelf/inference/execution/gdn_conv_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/kernels/optimized/gdn/conv1d.h>
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

#include <cstddef>
#include <cstdint>

namespace ps::execution {
namespace {

using ::ps::runtime::KernelId;
using ::ps::runtime::ValueDType;
using ::ps::runtime::ValueRowDomain;

constexpr OptimizedLaunchResult kNone{OptimizedLaunchKind::NotApplicable, 0};

const ::ps::runtime::StateBinding* find_conv_state_binding(
    const ::ps::runtime::Program& program, uint32_t descriptor_slot) {
    if (descriptor_slot == 0u) return nullptr;
    for (const auto& s : program.states)
        if (s.slot + 1u == descriptor_slot) return &s;
    return nullptr;
}

}  // namespace

Result<PhysicalResolveStatus> resolve_gdn_conv1d_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalGdnConv1dLaunch& out) {
    out = PhysicalGdnConv1dLaunch{};
    if (b.input_count != 1u || b.output_count != 1u)
        return PhysicalResolveStatus::NotApplicable;

    const auto* sb = find_conv_state_binding(program, b.descriptor_slot);
    if (sb == nullptr ||
        sb->kind != ::ps::runtime::PrimitiveStateKind::GDN_CONV_STATE)
        return PhysicalResolveStatus::NotApplicable;
    const uint32_t state_index = sb->state_index;

    const ::ps::state::GdnStatePoolDeviceView& gdn = ctx.model_state.gdn;
    if (gdn.conv_base == nullptr || gdn.num_gdn_states == 0u ||
        gdn.max_sequences == 0u || gdn.conv_dim == 0u || gdn.conv_history != 3u)
        return PhysicalResolveStatus::NotApplicable;
    if (state_index >= gdn.num_gdn_states)
        return PhysicalResolveStatus::NotApplicable;
    if (gdn.conv_slot_stride == 0u || gdn.conv_layer_stride == 0u ||
        gdn.conv_history_stride < gdn.conv_dim)
        return PhysicalResolveStatus::NotApplicable;

    auto in_res =
        resolve_host_value(program, ::ps::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    auto out_res =
        resolve_host_value(program, ::ps::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue dst = out_res.value();

    if (in.dtype != ValueDType::BF16 || in.row_domain != ValueRowDomain::TOKEN_ROWS)
        return PhysicalResolveStatus::NotApplicable;
    if (dst.dtype != ValueDType::F32 || dst.row_domain != ValueRowDomain::TOKEN_ROWS)
        return PhysicalResolveStatus::NotApplicable;
    if (in.feature_count != gdn.conv_dim || dst.feature_count != gdn.conv_dim)
        return PhysicalResolveStatus::NotApplicable;
    if (in.ptr == nullptr || dst.ptr == nullptr)
        return PhysicalResolveStatus::NotApplicable;
    if (in.row_stride < gdn.conv_dim || dst.row_stride < gdn.conv_dim)
        return PhysicalResolveStatus::NotApplicable;

    if (b.parameter_index0 == ::ps::runtime::kNoParameter ||
        ctx.host_parameters == nullptr ||
        b.parameter_index0 >= ctx.host_parameter_count)
        return PhysicalResolveStatus::NotApplicable;
    const auto& w_slot = ctx.host_parameters[b.parameter_index0];
    const uint64_t need_w = static_cast<uint64_t>(gdn.conv_dim) * 4u;
    if (w_slot.device_ptr == nullptr || w_slot.dtype != ValueDType::BF16 ||
        w_slot.elements < need_w)
        return PhysicalResolveStatus::NotApplicable;

    const ::ps::runtime::DeviceRequestDescriptor* requests = ctx.staged_requests;
    const uint32_t num_requests = ctx.staged_request_count;
    const uint32_t verify_requests = ctx.staged_verify_request_count;
    const bool all_verify_requests =
        verify_requests != 0u && verify_requests == num_requests;
    const uint32_t actual_rows = ctx.actual_rows;
    if (requests == nullptr || num_requests == 0u || actual_rows == 0u)
        return PhysicalResolveStatus::NotApplicable;
    uint32_t max_request_rows = 0u;
    for (uint32_t r = 0u; r < num_requests; ++r) {
        const auto& req = requests[r];
        if (req.request_handle.slot >= gdn.max_sequences)
            return PhysicalResolveStatus::NotApplicable;
        if (req.row_count > max_request_rows) max_request_rows = req.row_count;
    }

    GdnConvSelectorInput si;
    si.input_dtype = in.dtype;
    si.output_dtype = dst.dtype;
    si.conv_dim = gdn.conv_dim;
    si.history = gdn.conv_history;
    si.rows = actual_rows;
    si.num_requests = num_requests;
    si.max_request_rows = max_request_rows;
    if (select_gdn_conv_implementation(si) != GdnConvImplementation::Optimized)
        return PhysicalResolveStatus::NotApplicable;

    ::ps::kernel::GdnConv1dArgs args;
    args.input = static_cast<const ::ps::bf16_t*>(in.ptr);
    args.output = static_cast<float*>(dst.ptr);
    args.conv_weight = static_cast<const ::ps::bf16_t*>(w_slot.device_ptr);
    args.conv_state = gdn.conv_base;
    if (ctx.device_requests == nullptr)
        return PhysicalResolveStatus::NotApplicable;
    args.requests = ctx.device_requests;
    args.num_requests = num_requests;
    args.actual_rows = actual_rows;
    args.max_request_rows = max_request_rows;
    args.input_row_stride = in.row_stride;
    args.output_row_stride = dst.row_stride;
    args.conv_dim = gdn.conv_dim;
    args.conv_history = gdn.conv_history;
    args.state_index = state_index;
    args.max_sequences = gdn.max_sequences;
    args.conv_slot_stride = gdn.conv_slot_stride;
    args.conv_layer_stride = gdn.conv_layer_stride;
    args.conv_history_stride = gdn.conv_history_stride;
    const GdnSpecHistoryDeviceView& hist = ctx.spec.gdn_spec_history;
    const bool mixed_verify = verify_requests != 0u && !all_verify_requests;
    if ((ctx.role == ::ps::runtime::ExecutionRole::Verify || mixed_verify) &&
        hist.conv_history != nullptr && hist.capture_rows != 0u) {
        args.conv_history_store = hist.conv_history;
        args.conv_history_row_stride = hist.conv_row_stride;
        args.capture_rows = hist.capture_rows;
        args.capture_verify_only = mixed_verify ? 1u : 0u;
    }
    args.row_parent = ctx.spec.tree_row_parent;
    if (ctx.spec.tree_row_parent != nullptr) {
        if (args.conv_history_store == nullptr) {
            return Status::invalid_state("gdn-conv tree requires spec history capture",
                                         __FILE__, __LINE__);
        }
        if (args.capture_rows < actual_rows) {
            return Status::invalid_state("gdn-conv tree capture rows too small", __FILE__,
                                         __LINE__);
        }
    }

    out.args = args;
    out.row_tile = ::ps::kernel::gdn_conv1d_row_tile();
    const uint32_t tiled_rows =
        max_request_rows != 0u ? max_request_rows : actual_rows;
    out.tiles = ::ps::kernel::gdn_conv1d_tiles(out.row_tile, tiled_rows);
    out.grid_x = ::ps::kernel::gdn_conv1d_blocks_x(gdn.conv_dim);
    out.grid_y = num_requests;
    out.grid_z = out.tiles;
    out.workgroup = ::ps::kernel::kGdnConv1dThreads;
    return PhysicalResolveStatus::Applicable;
}

Result<OptimizedLaunchResult> try_launch_gdn_conv1d(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (dispatch_index >= program.dispatches.size()) return kNone;
    const auto& b = program.dispatches[dispatch_index];
    if (b.kernel_id != KernelId::STATEFUL_CAUSAL_CONV1D) return kNone;

    PhysicalGdnConv1dLaunch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_gdn_conv1d_physical(program, b, ctx, launch);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable) return kNone;

    const hipError_t herr =
        ::ps::kernel::launch_gdn_conv1d_bf16_f32(launch.args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("gdn-conv1d optimized launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 1};
}

}  // namespace ps::execution
