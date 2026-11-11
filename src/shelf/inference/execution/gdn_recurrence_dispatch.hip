#include <shelf/inference/execution/gdn_recurrence_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/optimized_dispatch.h>
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
#include <cstdlib>

namespace shelf::execution {
namespace {

using ::shelf::runtime::KernelId;

const ::shelf::runtime::StateBinding* find_state_binding_for_descriptor(
    const ::shelf::runtime::Program& program, uint32_t descriptor_slot) {
    if (descriptor_slot == 0u) return nullptr;
    for (const auto& s : program.states)
        if (s.slot + 1u == descriptor_slot) return &s;
    return nullptr;
}

}  // namespace

Result<GdnRecurrenceResolvedArgs> resolve_gdn_recurrence_args(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx) {
    if (b.kernel_id != KernelId::GDN_RECURRENCE)
        return Status::invalid_state("gdn-recurrence: not a recurrence dispatch", __FILE__, __LINE__);
    if (b.input_count != 5u || b.output_count != 1u)
        return Status::invalid_state("gdn-recurrence: io count", __FILE__, __LINE__);

    const auto* sb = find_state_binding_for_descriptor(program, b.descriptor_slot);
    if (sb == nullptr || sb->kind != ::shelf::runtime::PrimitiveStateKind::GDN_RECURRENCE_STATE)
        return Status::invalid_state("gdn-recurrence: recurrence state not found", __FILE__, __LINE__);
    const uint32_t state_index = sb->state_index;

    auto q_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!q_res.ok()) return q_res.status();
    const HostResolvedValue q = q_res.value();
    auto k_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[1]}, ctx);
    if (!k_res.ok()) return k_res.status();
    const HostResolvedValue k = k_res.value();
    auto v_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[2]}, ctx);
    if (!v_res.ok()) return v_res.status();
    const HostResolvedValue v = v_res.value();
    auto a_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[3]}, ctx);
    if (!a_res.ok()) return a_res.status();
    const HostResolvedValue a = a_res.value();
    auto bin_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[4]}, ctx);
    if (!bin_res.ok()) return bin_res.status();
    const HostResolvedValue bin = bin_res.value();
    auto out_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue out = out_res.value();

    const ::shelf::state::GdnStatePoolDeviceView& gdn = ctx.model_state.gdn;
    if (gdn.recurrent_base == nullptr)
        return Status::invalid_state("gdn-recurrence: recurrent_base null", __FILE__, __LINE__);
    if (state_index >= gdn.num_gdn_states)
        return Status::invalid_state("gdn-recurrence: state_index out of range", __FILE__, __LINE__);
    if (gdn.num_gdn_states == 0u || gdn.max_sequences == 0u)
        return Status::invalid_state("gdn-recurrence: gdn state metadata", __FILE__, __LINE__);
    const uint32_t num_v_heads = gdn.num_v_heads;
    const uint32_t head_k = gdn.head_k;
    const uint32_t head_v = gdn.head_v;
    if (num_v_heads == 0u || head_k == 0u || head_v == 0u)
        return Status::invalid_state("gdn-recurrence: gdn geometry zero", __FILE__, __LINE__);
    if (gdn.recurrent_slot_stride == 0u || gdn.recurrent_layer_stride == 0u)
        return Status::invalid_state("gdn-recurrence: gdn stride zero", __FILE__, __LINE__);

    if (q.dtype != ::shelf::runtime::ValueDType::F32 ||
        q.row_domain != ::shelf::runtime::ValueRowDomain::TOKEN_ROWS ||
        q.feature_count == 0u || q.row_stride < q.feature_count)
        return Status::invalid_state("gdn-recurrence: Q invalid", __FILE__, __LINE__);
    if (k.dtype != ::shelf::runtime::ValueDType::F32 || k.feature_count != q.feature_count ||
        k.row_domain != q.row_domain || k.row_stride < k.feature_count)
        return Status::invalid_state("gdn-recurrence: K invalid", __FILE__, __LINE__);
    if (v.dtype != ::shelf::runtime::ValueDType::BF16 ||
        v.row_domain != ::shelf::runtime::ValueRowDomain::TOKEN_ROWS)
        return Status::invalid_state("gdn-recurrence: V dtype/domain", __FILE__, __LINE__);
    if (a.dtype != ::shelf::runtime::ValueDType::BF16 ||
        a.row_domain != ::shelf::runtime::ValueRowDomain::TOKEN_ROWS)
        return Status::invalid_state("gdn-recurrence: A dtype/domain", __FILE__, __LINE__);
    if (bin.dtype != ::shelf::runtime::ValueDType::BF16 ||
        bin.row_domain != ::shelf::runtime::ValueRowDomain::TOKEN_ROWS)
        return Status::invalid_state("gdn-recurrence: B dtype/domain", __FILE__, __LINE__);
    if (out.dtype != ::shelf::runtime::ValueDType::F32 ||
        out.row_domain != ::shelf::runtime::ValueRowDomain::TOKEN_ROWS || out.row_stride < out.feature_count)
        return Status::invalid_state("gdn-recurrence: output invalid", __FILE__, __LINE__);

    const uint32_t qk_dim = q.feature_count;
    if (head_k > 128u || qk_dim % head_k != 0u)
        return Status::invalid_state("gdn-recurrence: qk geometry", __FILE__, __LINE__);
    const uint32_t key_heads = qk_dim / head_k;
    if (key_heads == 0u || num_v_heads < key_heads || num_v_heads % key_heads != 0u)
        return Status::invalid_state("gdn-recurrence: repeat invalid", __FILE__, __LINE__);

    const uint64_t value_dim = static_cast<uint64_t>(num_v_heads) * head_v;
    if (v.feature_count != value_dim || bin.feature_count != num_v_heads ||
        a.feature_count != num_v_heads || out.feature_count != value_dim)
        return Status::invalid_state("gdn-recurrence: feature mismatch", __FILE__, __LINE__);

    const uint64_t expected_layer = static_cast<uint64_t>(num_v_heads) * head_k * head_v;
    if (gdn.recurrent_layer_stride < expected_layer)
        return Status::invalid_state("gdn-recurrence: recurrent_layer_stride", __FILE__, __LINE__);

    if (b.parameter_index0 == ::shelf::runtime::kNoParameter || ctx.host_parameters == nullptr ||
        b.parameter_index0 >= ctx.host_parameter_count)
        return Status::invalid_state("gdn-recurrence: dt_bias parameter missing", __FILE__, __LINE__);
    const auto& dt_slot = ctx.host_parameters[b.parameter_index0];
    if (dt_slot.device_ptr == nullptr || dt_slot.dtype != ::shelf::runtime::ValueDType::BF16 ||
        dt_slot.elements < num_v_heads)
        return Status::invalid_state("gdn-recurrence: dt_bias invalid", __FILE__, __LINE__);
    if (b.parameter_index1 == ::shelf::runtime::kNoParameter ||
        b.parameter_index1 >= ctx.host_parameter_count)
        return Status::invalid_state("gdn-recurrence: a_log parameter missing", __FILE__, __LINE__);
    const auto& al_slot = ctx.host_parameters[b.parameter_index1];
    if (al_slot.device_ptr == nullptr || al_slot.dtype != ::shelf::runtime::ValueDType::BF16 ||
        al_slot.elements < num_v_heads)
        return Status::invalid_state("gdn-recurrence: a_log invalid", __FILE__, __LINE__);

    const ::shelf::runtime::DeviceRequestDescriptor* requests = ctx.staged_requests;
    const uint32_t num_requests = ctx.staged_request_count;
    const uint32_t verify_requests = ctx.staged_verify_request_count;
    const bool all_verify_requests =
        verify_requests != 0u && verify_requests == num_requests;
    const uint32_t actual_rows = ctx.actual_rows;
    if (requests == nullptr)
        return Status::invalid_state("gdn-recurrence: host request metadata missing",
                                     __FILE__, __LINE__);
    if (num_requests == 0u || actual_rows == 0u)
        return Status::invalid_state("gdn-recurrence: requests/rows == 0", __FILE__, __LINE__);
    uint32_t max_request_rows = 0u;
    for (uint32_t r = 0u; r < num_requests; ++r)
        if (requests[r].row_count > max_request_rows)
            max_request_rows = requests[r].row_count;

    GdnRecurrenceResolvedArgs ra;
    ra.rows = actual_rows;
    ra.num_requests = num_requests;
    ra.max_request_rows = max_request_rows;
    ::shelf::kernel::GdnRecurrenceArgs& c = ra.kernel;
    c.q = static_cast<const float*>(q.ptr);
    c.k = static_cast<const float*>(k.ptr);
    c.v = static_cast<const bf16_t*>(v.ptr);
    c.a = static_cast<const bf16_t*>(a.ptr);
    c.b = static_cast<const bf16_t*>(bin.ptr);
    c.dt_bias = static_cast<const bf16_t*>(dt_slot.device_ptr);
    c.a_log = static_cast<const bf16_t*>(al_slot.device_ptr);
    c.recurrent_state = gdn.recurrent_base;
    c.output = static_cast<float*>(out.ptr);
    if (ctx.device_requests == nullptr)
        return Status::invalid_state("gdn-recurrence: device requests null",
                                     __FILE__, __LINE__);
    c.requests = ctx.device_requests;
    c.num_requests = num_requests;
    c.actual_rows = actual_rows;
    c.q_row_stride = q.row_stride;
    c.k_row_stride = k.row_stride;
    c.v_row_stride = v.row_stride;
    c.a_row_stride = a.row_stride;
    c.b_row_stride = bin.row_stride;
    c.output_row_stride = out.row_stride;
    c.state_index = state_index;
    c.max_sequences = gdn.max_sequences;
    c.num_gdn_states = gdn.num_gdn_states;
    c.key_heads = key_heads;
    c.num_v_heads = num_v_heads;
    c.head_k = head_k;
    c.head_v = head_v;
    c.recurrent_slot_stride = gdn.recurrent_slot_stride;
    c.recurrent_layer_stride = gdn.recurrent_layer_stride;
    const GdnSpecHistoryDeviceView& hist = ctx.spec.gdn_spec_history;
    const bool mixed_verify = verify_requests != 0u && !all_verify_requests;
    if ((ctx.role == ::shelf::runtime::ExecutionRole::Verify || mixed_verify) &&
        hist.recurrent_history != nullptr && hist.capture_rows != 0u) {
        c.recurrent_history = hist.recurrent_history;
        c.recurrent_history_row_stride = hist.recurrent_row_stride;
        c.capture_rows = hist.capture_rows;
        c.recurrent_capture_mask = hist.recurrent_capture_mask;
        c.capture_verify_only = mixed_verify ? 1u : 0u;
    }
    c.row_parent = ctx.spec.tree_row_parent;
    if (ctx.spec.tree_row_parent != nullptr && c.recurrent_history == nullptr) {
        return Status::invalid_state("gdn recurrence tree requires spec history capture",
                                     __FILE__, __LINE__);
    }
    if (ctx.spec.tree_row_parent != nullptr && c.capture_rows < c.actual_rows) {
        return Status::invalid_state("gdn recurrence tree capture rows too small", __FILE__,
                                     __LINE__);
    }
    const GdnCompactLogDeviceView& compact = ctx.spec.gdn_compact;
    if (compact.delta != nullptr && compact.k != nullptr && compact.a != nullptr &&
        compact.delta_layer_stride != 0u && compact.k_layer_stride != 0u &&
        compact.a_layer_stride != 0u) {
        c.compact_delta = compact.delta;
        c.compact_k = compact.k;
        c.compact_a = compact.a;
        c.compact_delta_layer_stride = compact.delta_layer_stride;
        c.compact_k_layer_stride = compact.k_layer_stride;
        c.compact_a_layer_stride = compact.a_layer_stride;
        c.skip_state_store = compact.defer_state_store;
    }
    return ra;
}

Result<PhysicalResolveStatus> resolve_gdn_recurrence_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalGdnRecurrenceLaunch& out) {
    out = PhysicalGdnRecurrenceLaunch{};
    const auto res = resolve_gdn_recurrence_args(program, b, ctx);
    if (!res.ok()) return PhysicalResolveStatus::NotApplicable;
    const GdnRecurrenceResolvedArgs ra = res.value();

    GdnRecurrenceSelectorInput si;
    si.rows = ra.rows;
    si.num_requests = ra.num_requests;
    si.max_request_rows = ra.max_request_rows;
    si.key_heads = ra.kernel.key_heads;
    si.num_v_heads = ra.kernel.num_v_heads;
    si.head_k = ra.kernel.head_k;
    si.head_v = ra.kernel.head_v;
    if (select_gdn_recurrence_implementation(si) !=
        GdnRecurrenceImplementation::Optimized)
        return PhysicalResolveStatus::NotApplicable;
    if (ra.rows == 0u || ra.num_requests == 0u || ra.max_request_rows == 0u)
        return PhysicalResolveStatus::NotApplicable;
    if (ra.kernel.key_heads == 0u || ra.kernel.num_v_heads == 0u ||
        ra.kernel.head_k == 0u || ra.kernel.head_v == 0u)
        return PhysicalResolveStatus::NotApplicable;

    const bool geo_ok = ra.kernel.head_k == ::shelf::kernel::kGdnRecurrenceWmmaK &&
                        ra.kernel.head_v %
                                ::shelf::kernel::kGdnRecurrenceWmmaVGroup ==
                            0u;

    PhysicalGdnRecurrenceVariant variant = PhysicalGdnRecurrenceVariant::F32;
    if (geo_ok) {
        variant = (ctx.role == ::shelf::runtime::ExecutionRole::Verify)
                      ? PhysicalGdnRecurrenceVariant::WmmaVerify
                      : PhysicalGdnRecurrenceVariant::Wmma;
    }

    out.kernel = ra.kernel;
    out.rows = ra.rows;
    out.num_requests = ra.num_requests;
    out.max_request_rows = ra.max_request_rows;
    out.variant = variant;
    out.lossy = ::shelf::kernel::gdn_recurrence_lossy_enabled();
    out.split_regions =
        variant == PhysicalGdnRecurrenceVariant::WmmaVerify &&
        ctx.num_verify_requests > 0u &&
        ctx.num_verify_requests < ctx.num_requests;
    if (variant == PhysicalGdnRecurrenceVariant::Wmma ||
        variant == PhysicalGdnRecurrenceVariant::WmmaVerify) {
        out.grid_x = ::shelf::kernel::gdn_recurrence_wmma_grid_x(
            ra.num_requests, ra.kernel.num_v_heads, ra.kernel.head_v);
        out.grid_y = 1u;
        out.grid_z = 1u;
        out.workgroup = ::shelf::kernel::kGdnRecurrenceWmmaThreads;
        if (out.grid_x == 0u)
            return PhysicalResolveStatus::NotApplicable;
        if (ra.kernel.num_v_heads % ra.kernel.key_heads != 0u)
            return PhysicalResolveStatus::NotApplicable;
    }
    return PhysicalResolveStatus::Applicable;
}

Result<OptimizedLaunchResult> try_launch_gdn_recurrence(
    const ::shelf::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (dispatch_index >= program.dispatches.size())
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    const auto& b = program.dispatches[dispatch_index];
    if (b.kernel_id != KernelId::GDN_RECURRENCE)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};

    PhysicalGdnRecurrenceLaunch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_gdn_recurrence_physical(program, b, ctx, launch);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};

    bool ran_f32 = launch.variant == PhysicalGdnRecurrenceVariant::F32;
    hipError_t herr = hipSuccess;
    uint32_t physical_launches = 0u;
    auto launch_variant = [&](PhysicalGdnRecurrenceVariant variant,
                              uint32_t execution_class_filter) -> hipError_t {
        ::shelf::kernel::GdnRecurrenceArgs args = launch.kernel;
        args.execution_class_filter = execution_class_filter;
        switch (variant) {
            case PhysicalGdnRecurrenceVariant::WmmaVerify:
                return ::shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(args, stream);
            case PhysicalGdnRecurrenceVariant::Wmma:
                return ::shelf::kernel::launch_gdn_recurrence_f32_wmma(args, stream);
            default:
                return ::shelf::kernel::launch_gdn_recurrence_f32(args, stream);
        }
    };

    if (launch.split_regions) {
        herr = launch_variant(PhysicalGdnRecurrenceVariant::WmmaVerify, 1u);
        if (herr == hipSuccess) {
            ++physical_launches;
            herr = launch_variant(PhysicalGdnRecurrenceVariant::Wmma, 2u);
            if (herr == hipSuccess) ++physical_launches;
        }
    } else {
        herr = launch_variant(launch.variant, 0u);
        if (herr == hipSuccess) ++physical_launches;
    }

    if (herr == hipErrorNotSupported) {
        herr = launch_variant(PhysicalGdnRecurrenceVariant::F32, 0u);
        ran_f32 = true;
        physical_launches = 1u;
    }
    if (herr != hipSuccess)
        return Status::hip_error("gdn-recurrence optimized launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    if (ran_f32)
        record_degraded_dispatch(b.kernel_id);
    record_physical_launch(physical_launches);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
