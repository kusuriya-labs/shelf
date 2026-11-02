#include <shelf/core/gpu/cleanup.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/kv_calib_dump.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <vector>
namespace ps::execution {

using ps::Result;
using ps::Status;
using ::ps::runtime::StagingProgramRefs;
using ::ps::runtime::StagingSectionOffsets;
using ::ps::runtime::staging_section_offsets;
using ::ps::runtime::staging_bytes_for;

Status fill_staging_image(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& binding,
    const HostExecutionContext& ctx,
    ::ps::runtime::DeviceProgramView* prog_out,
    ::ps::runtime::DispatchBinding* binding_out,
    ::ps::runtime::DeviceValueBinding* values_out,
    StagingProgramRefs* refs_out) {
    const auto& plan = program;
    uint32_t n = 0;
    for (uint32_t i = 0; i < binding.input_count; ++i) {
        const ::ps::runtime::ValueId vid{binding.input_slots[i]};
        const auto* found = plan.find_value(vid);
        if (found == nullptr)
            return Status::invalid_argument("dispatch input value unbound", __FILE__, __LINE__);
        auto& d = values_out[n++];
        d.logical_value = vid.id;
        d.storage = found->storage;
        d.dtype = found->dtype;
        d.row_domain = found->row_domain;
        d.slot = found->slot;
        d.feature_count = found->feature_count;
        d.row_stride = found->row_stride;
        d.bytes = found->bytes;
        if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_INPUT) {
            if (found->slot >= ctx.external_input_count || ctx.external_inputs == nullptr)
                return Status::invalid_argument("external input slot out of range", __FILE__, __LINE__);
            d.offset = reinterpret_cast<uint64_t>(ctx.external_inputs[found->slot]);
        } else if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT) {
            if (found->slot >= ctx.external_output_count || ctx.external_outputs == nullptr)
                return Status::invalid_argument("external output slot out of range", __FILE__, __LINE__);
            d.offset = reinterpret_cast<uint64_t>(ctx.external_outputs[found->slot]);
        } else {
            d.offset = found->offset;
        }
    }
    for (uint32_t i = 0; i < binding.output_count; ++i) {
        const ::ps::runtime::ValueId vid{binding.output_slots[i]};
        const auto* found = plan.find_value(vid);
        if (found == nullptr)
            return Status::invalid_argument("dispatch output value unbound", __FILE__, __LINE__);
        auto& d = values_out[n++];
        d.logical_value = vid.id;
        d.storage = found->storage;
        d.dtype = found->dtype;
        d.row_domain = found->row_domain;
        d.slot = found->slot;
        d.feature_count = found->feature_count;
        d.row_stride = found->row_stride;
        d.bytes = found->bytes;
        if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_INPUT) {
            if (found->slot >= ctx.external_input_count || ctx.external_inputs == nullptr)
                return Status::invalid_argument("external input slot out of range", __FILE__, __LINE__);
            d.offset = reinterpret_cast<uint64_t>(ctx.external_inputs[found->slot]);
        } else if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT) {
            if (found->slot >= ctx.external_output_count || ctx.external_outputs == nullptr)
                return Status::invalid_argument("external output slot out of range", __FILE__, __LINE__);
            d.offset = reinterpret_cast<uint64_t>(ctx.external_outputs[found->slot]);
        } else {
            d.offset = found->offset;
        }
    }
    if (refs_out != nullptr) {
        const ::ps::runtime::ProgramStagingMeta* meta = ctx.program_meta;
        refs_out->ranges = (meta != nullptr) ? meta->ranges : nullptr;
        refs_out->range_count =
            (meta != nullptr) ? meta->range_count
                              : static_cast<uint32_t>(plan.workspace.ranges.size());
        refs_out->states = (meta != nullptr) ? meta->states : nullptr;
        refs_out->state_count =
            (meta != nullptr) ? meta->state_count
                              : static_cast<uint32_t>(plan.states.size());
    }
    *binding_out = binding;
    prog_out->values = static_cast<const ::ps::runtime::DeviceValueBinding*>(nullptr);
    prog_out->value_count = n;
    prog_out->weight_table = ctx.weights;
    prog_out->weight_slot_count = ctx.weight_count;
    prog_out->parameter_table = ctx.parameters;
    prog_out->parameter_slot_count = ctx.parameter_count;
    prog_out->workspace_ranges = static_cast<const ::ps::runtime::DeviceWorkspaceRange*>(nullptr);
    prog_out->workspace_range_count = static_cast<uint32_t>(plan.workspace.ranges.size());
    prog_out->states = static_cast<const ::ps::runtime::DeviceStateBinding*>(nullptr);
    prog_out->state_count = static_cast<uint32_t>(plan.states.size());
    prog_out->workspace_base = reinterpret_cast<uint64_t>(ctx.workspace);
    return Status::make_ok();
}

Status validate_host_weight_slot(const ::ps::runtime::WeightSlot& s, size_t index) {
    const uint8_t enc = s.encoding;
    if (s.rows == 0 || s.cols == 0) {
        return Status::invalid_argument(
            ("weight table slot has zero rows/cols: index " + std::to_string(index)).c_str(),
            __FILE__, __LINE__);
    }
    if (enc == 0u) {
        if (s.device_ptr == nullptr || s.codes != nullptr || s.scales != nullptr) {
            return Status::invalid_argument(
                ("bf16 weight table slot needs device_ptr and no codes/scales: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (enc == 2u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.scales == nullptr) {
            return Status::invalid_argument(
                ("rocm fp8 weight table slot needs codes+scales and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.k_padded < s.cols || (s.k_padded & 31u) != 0u) {
            return Status::invalid_argument(
                ("rocm fp8 weight table slot k_padded invalid: index " + std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (enc == 5u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.scales == nullptr) {
            return Status::invalid_argument(
                ("block scaled weight table slot needs codes+scales and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        const uint32_t k_align = 127u;
        if (s.k_padded < s.cols || (s.k_padded & k_align) != 0u) {
            return Status::invalid_argument(
                ("block scaled weight table slot k_padded invalid: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    return Status::invalid_argument(
        ("unsupported weight table encoding: index " + std::to_string(index)).c_str(),
        __FILE__, __LINE__);
}


using ::ps::Result;
using ::ps::Status;
using ::ps::runtime::WeightSlot;
using ::ps::runtime::StaticParameterSlot;

Result<HostResolvedValue> resolve_host_value(
    const ::ps::runtime::Program& program,
    ::ps::runtime::ValueId id,
    const HostExecutionContext& ctx) {
    const auto* found = program.find_value(id);
    if (found == nullptr)
        return Status::invalid_argument("value unbound", __FILE__, __LINE__);
    HostResolvedValue out{};
    out.row_stride = found->row_stride;
    out.feature_count = found->feature_count;
    out.dtype = found->dtype;
    out.row_domain = found->row_domain;
    if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_INPUT) {
        if (found->slot >= ctx.external_input_count || ctx.external_inputs == nullptr)
            return Status::invalid_argument("external input slot out of range", __FILE__, __LINE__);
        out.ptr = const_cast<void*>(ctx.external_inputs[found->slot]);
        return out;
    }
    if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT) {
        if (found->slot >= ctx.external_output_count || ctx.external_outputs == nullptr)
            return Status::invalid_argument("external output slot out of range", __FILE__, __LINE__);
        out.ptr = ctx.external_outputs[found->slot];
        return out;
    }
    if (ctx.workspace == nullptr)
        return Status::invalid_state("workspace not bound", __FILE__, __LINE__);
    out.ptr = ctx.workspace + found->offset;
    return out;
}

namespace {
const bool g_op_dump_enabled = [] {
    const char* e = std::getenv("SHELF_OP_DUMP");
    return e != nullptr && e[0] != '\0';
}();
const bool g_op_sync_enabled = [] {
    const char* e = std::getenv("SHELF_OP_SYNC");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}();
struct OpDumpTimer {
    std::chrono::steady_clock::time_point t0;
    const ::ps::runtime::KernelId id;
    explicit OpDumpTimer(::ps::runtime::KernelId k) : t0(std::chrono::steady_clock::now()), id(k) {}
    ~OpDumpTimer() {
        if (!g_op_dump_enabled) return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "OPDUMP %s %.4f\n", ::ps::runtime::kernel_to_name(id), ms);
    }
};
}

static int bucket_to_index(::ps::runtime::RowBucket bucket) {
    switch (bucket) {
        case ::ps::runtime::RowBucket::R16: return 0;
        case ::ps::runtime::RowBucket::R32: return 1;
        case ::ps::runtime::RowBucket::R64: return 2;
        case ::ps::runtime::RowBucket::R128: return 3;
        case ::ps::runtime::RowBucket::R256: return 4;
        case ::ps::runtime::RowBucket::R512: return 5;
        case ::ps::runtime::RowBucket::R1024: return 6;
        case ::ps::runtime::RowBucket::R2048: return 7;
        case ::ps::runtime::RowBucket::R4096: return 8;
        case ::ps::runtime::RowBucket::R8192: return 9;
    }
    return -1;
}

Status launch_host_binding(
    const ::ps::runtime::DispatchBinding& binding,
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    const bool sync_each_op = g_op_dump_enabled || g_op_sync_enabled;
    if (ctx.batch_context == nullptr)
        return Status::invalid_state("batch context not bound", __FILE__, __LINE__);
    if (ctx.scratch == nullptr || ctx.scratch_floats < 2048)
        return Status::invalid_state("dispatch scratch not bound", __FILE__, __LINE__);
    if (ctx.error_word == nullptr)
        return Status::invalid_state("error word not bound", __FILE__, __LINE__);
    const uint64_t nval = static_cast<uint64_t>(binding.input_count + binding.output_count);
    const auto seco = staging_section_offsets(nval);
#ifdef SHELF_HIP_GRAPH
    bool stream_capturing = false;
    {
        hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(stream, &cap_status) == hipSuccess)
            stream_capturing = (cap_status == hipStreamCaptureStatusActive);
    }
#endif
    uint8_t* dev_base = nullptr;
    bool via_pool = false;
    if (ctx.staging_pool != nullptr && ctx.staging_pool->device != nullptr) {
        const int bidx = bucket_to_index(program.row_bucket);
        const uint32_t bindx =
            static_cast<uint32_t>(&binding - program.dispatches.data());
        if (bidx >= 0 && bindx < program.dispatches.size()) {
            const uint64_t slot = ctx.staging_pool->slot(static_cast<uint32_t>(bidx), bindx);
            if (slot < ctx.staging_pool->bytes.size() && ctx.staging_pool->bytes[slot] > 0) {
                dev_base = static_cast<uint8_t*>(ctx.staging_pool->device)
                           + ctx.staging_pool->offsets[slot];
                via_pool = true;
                if (slot < ctx.staging_pool->run_seen.size())
                    ctx.staging_pool->run_seen[slot] = 1;
                if (!ctx.staging_pool->uploaded[slot]) {
#ifdef SHELF_HIP_GRAPH
                    if (stream_capturing)
                        return Status::invalid_state(
                            "staging pool upload during capture", __FILE__, __LINE__);
#endif
                    if (ctx.host_staging == nullptr || ctx.staging_bytes < ctx.staging_pool->bytes[slot])
                        return Status::invalid_state(
                            "dispatch staging not bound for pool upload", __FILE__, __LINE__);
                    auto* host_base = static_cast<uint8_t*>(ctx.host_staging);
                    auto* binding_host = reinterpret_cast<::ps::runtime::DispatchBinding*>(host_base);
                    auto* values_host = reinterpret_cast<::ps::runtime::DeviceValueBinding*>(host_base + seco.values);
                    auto* refs_host = reinterpret_cast<StagingProgramRefs*>(host_base + seco.refs);
                    ::ps::runtime::DeviceProgramView prog_host{};
                    Status fst = fill_staging_image(
                        program, binding, ctx, &prog_host, binding_host, values_host,
                        refs_host);
                    if (!fst.ok()) return fst;
                    hipError_t uherr = hipMemcpyAsync(dev_base, ctx.host_staging,
                                                      ctx.staging_pool->bytes[slot],
                                                      hipMemcpyHostToDevice, stream);
                    if (uherr != hipSuccess)
                        return Status::hip_error("hipMemcpyAsync staging pool",
                                                 hipGetErrorString(uherr), __FILE__, __LINE__);
                    uherr = hipStreamSynchronize(stream);
                    if (uherr != hipSuccess)
                        return Status::hip_error("staging pool upload sync",
                                                 hipGetErrorString(uherr), __FILE__, __LINE__);
                    ctx.staging_pool->uploaded[slot] = 1;
                }
            }
        }
    }
    if (dev_base == nullptr) {
#ifdef SHELF_HIP_GRAPH
        if (stream_capturing)
            return Status::invalid_state(
                "non-pool dispatch staging during capture", __FILE__, __LINE__);
#endif
        if (ctx.staging == nullptr || ctx.host_staging == nullptr || ctx.staging_bytes == 0)
            return Status::invalid_state("dispatch staging not bound", __FILE__, __LINE__);
        const uint64_t need = staging_bytes_for(binding);
        if (need > ctx.staging_bytes)
            return Status::insufficient_memory("dispatch staging too small", __FILE__, __LINE__);
        auto* host_base = static_cast<uint8_t*>(ctx.host_staging);
        auto* binding_host =
            reinterpret_cast<::ps::runtime::DispatchBinding*>(host_base);
        auto* values_host = reinterpret_cast<::ps::runtime::DeviceValueBinding*>(host_base + seco.values);
        auto* refs_host = reinterpret_cast<StagingProgramRefs*>(host_base + seco.refs);
        ::ps::runtime::DeviceProgramView prog_host{};
        Status fst = fill_staging_image(
            program, binding, ctx, &prog_host, binding_host, values_host, refs_host);
        if (!fst.ok()) return fst;
        hipError_t herr = hipMemcpyAsync(ctx.staging, ctx.host_staging, need,
                                         hipMemcpyHostToDevice, stream);
        if (herr != hipSuccess)
            return Status::hip_error("hipMemcpyAsync dispatch staging",
                                     hipGetErrorString(herr), __FILE__, __LINE__);
        dev_base = static_cast<uint8_t*>(ctx.staging);
    }
    const auto* refs_dev = reinterpret_cast<const StagingProgramRefs*>(dev_base + seco.refs);
    ::ps::runtime::DeviceProgramView prog_dev{};
    prog_dev.values = reinterpret_cast<const ::ps::runtime::DeviceValueBinding*>(dev_base + seco.values);
    prog_dev.value_count = static_cast<uint32_t>(nval);
    prog_dev.weight_table = ctx.weights;
    prog_dev.weight_slot_count = ctx.weight_count;
    prog_dev.parameter_table = ctx.parameters;
    prog_dev.parameter_slot_count = ctx.parameter_count;
    prog_dev.workspace_ranges =
        reinterpret_cast<const ::ps::runtime::DeviceWorkspaceRange*>(refs_dev->ranges);
    prog_dev.workspace_range_count = refs_dev->range_count;
    prog_dev.states =
        reinterpret_cast<const ::ps::runtime::DeviceStateBinding*>(refs_dev->states);
    prog_dev.state_count = refs_dev->state_count;
    prog_dev.workspace_base = reinterpret_cast<uint64_t>(ctx.workspace);
    Status lstat = ::ps::kernel::launch_model_dispatch_correctness(
        prog_dev, binding, ctx.batch_context, ctx.model_state, ctx.scratch,
        ctx.error_word, stream);
    if (!lstat.ok()) return lstat;
    hipError_t herr = hipSuccess;
    if (!via_pool || sync_each_op ||
        binding.kernel_id == ::ps::runtime::KernelId::EMBEDDING_LOOKUP) {
#ifdef SHELF_HIP_GRAPH
        if (stream_capturing && via_pool)
            return Status::make_ok();
#endif
        herr = hipStreamSynchronize(stream);
        if (herr != hipSuccess)
            return Status::hip_error("host_dispatch_kernel sync",
                                     hipGetErrorString(herr), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

namespace {

uint32_t probe_rows(const HostResolvedValue& v, const HostExecutionContext& ctx) {
    if (v.row_domain == ::ps::runtime::ValueRowDomain::OUTPUT_ROWS) return ctx.actual_outputs;
    return ctx.actual_rows != 0 ? ctx.actual_rows : 1u;
}

Status record_value_trace(const ::ps::runtime::Program& program, std::size_t first,
                          uint32_t count, const HostExecutionContext& ctx,
                          hipStream_t stream) {
    auto* sink = ctx.value_trace;
    if (sink == nullptr || sink->on_value == nullptr) return Status::make_ok();
    for (uint32_t j = 0; j < count; ++j) {
        const auto& binding = program.dispatches[first + j];
        for (uint32_t o = 0; o < binding.output_count; ++o) {
            const uint32_t vid = binding.output_slots[o];
            auto resolved = resolve_host_value(program, ::ps::runtime::ValueId{vid}, ctx);
            if (!resolved.ok()) return resolved.status();
            const HostResolvedValue v = resolved.value();
            if (v.ptr == nullptr || v.feature_count == 0) continue;
            sink->on_value(vid, v.ptr, v.row_stride, v.feature_count, v.dtype,
                           probe_rows(v, ctx), stream);
        }
    }
    return Status::make_ok();
}

}

Status execute_program_range(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream,
    size_t begin,
    size_t end) {
    hipError_t herr = hipSuccess;
    if (ctx.layer_overrides != nullptr && ctx.layer_override_count != 0) {
        for (uint32_t oi = 0; oi < ctx.layer_override_count; ++oi) {
            ctx.layer_override_injected[oi] = 0;
        }
    }
    auto mode_result = read_kernel_mode();
    if (!mode_result.ok()) return mode_result.status();
    const KernelMode mode = mode_result.value();
    const bool sync_each_op = g_op_dump_enabled || g_op_sync_enabled;
    size_t i = begin;
    while (i < end) {
        const auto& binding = program.dispatches[i];
        if (ctx.layer_overrides != nullptr && ctx.layer_override_count != 0) {
            for (uint32_t oi = 0; oi < ctx.layer_override_count; ++oi) {
                if (ctx.layer_override_injected[oi] != 0) continue;
                const LayerInputOverride& ov = ctx.layer_overrides[oi];
                bool consumes = false;
                for (uint32_t s = 0; s < binding.input_count &&
                                    s < ::ps::runtime::kMaxIoSlots; ++s) {
                    if (binding.input_slots[s] == ov.value_id.id) {
                        consumes = true;
                        break;
                    }
                }
                if (!consumes) continue;
                ctx.layer_override_injected[oi] = 1;
                auto resolved = resolve_host_value(program, ov.value_id, ctx);
                if (!resolved.ok()) return resolved.status();
                const HostResolvedValue v = resolved.value();
                const uint32_t rows = probe_rows(v, ctx);
                if (v.ptr == nullptr || rows != ov.rows || v.feature_count != ov.hidden ||
                    v.row_stride != ov.hidden) {
                    if (std::getenv("SHELF_LAYER_INPUT_DEBUG") != nullptr) {
                        std::fprintf(stderr,
                                     "OVERRIDE skip id=%u rows=%u stride=%u feat=%u\n",
                                     ov.value_id.id, rows, v.row_stride, v.feature_count);
                    }
                    continue;
                }
                if (std::getenv("SHELF_LAYER_INPUT_DEBUG") != nullptr) {
                    const auto* spec = program.find_value(ov.value_id);
                    std::fprintf(stderr,
                                 "OVERRIDE id=%u ptr=%p off=%llu rows=%u stride=%u "
                                 "feat=%u i=%zu begin=%zu\n",
                                 ov.value_id.id, v.ptr,
                                 spec != nullptr
                                     ? static_cast<unsigned long long>(spec->offset)
                                     : 0ull,
                                 rows, v.row_stride, v.feature_count, i, begin);
                }
                const std::size_t bytes =
                    static_cast<std::size_t>(ov.rows) * ov.hidden * sizeof(std::uint16_t);
                herr = hipMemcpyAsync(v.ptr, ov.host_bf16, bytes, hipMemcpyHostToDevice, stream);
                if (herr != hipSuccess) {
                    return Status::hip_error("layer input override copy",
                                             hipGetErrorString(herr), __FILE__, __LINE__);
                }
            }
        }
        kv_calib_dump_maybe(program, i, ctx, stream);
        OpDumpTimer op_timer(binding.kernel_id);
        const bool sem_on = semantic_timing_enabled();
        const uint32_t sem_seq = sem_on ? semantic_timing_start(stream) : 0xFFFFFFFFu;
        if (mode == KernelMode::Auto) {
            OptimizedLaunchResult opt_result{};
            auto optimized = try_launch_optimized(program, i, ctx, stream);
            if (!optimized.ok()) return optimized.status();
            opt_result = optimized.value();
            const OptimizedLaunchKind opt_kind = opt_result.kind;
            if (opt_kind == OptimizedLaunchKind::Skipped) {
                const uint32_t consumed = opt_result.consumed_dispatches;
                if (consumed == 0 || i + consumed > end)
                    return Status::invalid_state(
                        "invalid skipped dispatch consumption", __FILE__, __LINE__);
                uint32_t covered_kids[4] = {0, 0, 0, 0};
                for (uint32_t j = 0; j < consumed && j < 4u; ++j)
                    covered_kids[j] = static_cast<uint32_t>(program.dispatches[i + j].kernel_id);
                record_logical_covered(covered_kids, consumed < 4u ? consumed : 4u);
                i += consumed;
                continue;
            }
            if (opt_kind == OptimizedLaunchKind::Launched) {
                const uint32_t consumed = opt_result.consumed_dispatches;
                if (consumed == 0 || i + consumed > end)
                    return Status::invalid_state(
                        "invalid optimized dispatch consumption", __FILE__, __LINE__);
                if (sync_each_op) {
                    herr = hipStreamSynchronize(stream);
                    if (herr != hipSuccess)
                        return Status::hip_error(
                            "optimized dispatch sync", hipGetErrorString(herr),
                            __FILE__, __LINE__);
                }
                uint32_t covered_kids[4] = {0, 0, 0, 0};
                for (uint32_t j = 0; j < consumed && j < 4u; ++j)
                    covered_kids[j] = static_cast<uint32_t>(program.dispatches[i + j].kernel_id);
                record_logical_covered(covered_kids, consumed < 4u ? consumed : 4u);
                if (sem_on)
                    semantic_timing_finish(stream, sem_seq, static_cast<uint32_t>(binding.kernel_id),
                                           static_cast<uint32_t>(SemPath::Optimized),
                                           consumed);
                {
                    Status tst = record_value_trace(program, i, consumed, ctx, stream);
                    if (!tst.ok()) return tst;
                }
                i += consumed;
                continue;
            }
        }
        Status st = launch_host_binding(binding, program, ctx, stream);
        if (!st.ok()) return st;
        record_correctness_fallback(binding.kernel_id);
        record_physical_launch(1);
        const uint32_t single_kid = static_cast<uint32_t>(binding.kernel_id);
        record_logical_covered(&single_kid, 1);
        if (sem_on)
            semantic_timing_finish(stream, sem_seq, single_kid,
                                   static_cast<uint32_t>(SemPath::Correctness), 1);
        {
            Status tst = record_value_trace(program, i, 1, ctx, stream);
            if (!tst.ok()) return tst;
        }
        ++i;
    }
#ifdef SHELF_HIP_GRAPH
    if (ctx.staging_pool != nullptr) {
        const int bidx = bucket_to_index(program.row_bucket);
        if (bidx >= 0 &&
            static_cast<std::size_t>(bidx) < ctx.staging_pool->bucket_ran.size())
            ctx.staging_pool->bucket_ran[static_cast<std::size_t>(bidx)] = 1;
    }
#endif
    return Status::make_ok();
}

Status execute_program_tp_ranges(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream,
    const ::ps::runtime::TpExecutionSchedule& schedule,
    ::ps::runtime::TpBarrierHook* barrier,
    int rank) {
    if (barrier == nullptr) {
        return Status::invalid_state("tp barrier hook not bound", __FILE__, __LINE__);
    }
    if (schedule.segments.empty()) {
        return Status::invalid_argument("tp execution schedule is empty", __FILE__, __LINE__);
    }
    if (schedule.segments.front().dispatch_begin != 0 ||
        schedule.segments.back().dispatch_end != program.dispatches.size()) {
        return Status::invalid_argument(
            "tp execution schedule does not cover the program", __FILE__, __LINE__);
    }
    uint32_t next_begin = 0;
    for (const ::ps::runtime::TpProgramSegment& segment : schedule.segments) {
        if (segment.dispatch_begin != next_begin ||
            segment.dispatch_end < segment.dispatch_begin ||
            segment.dispatch_end > program.dispatches.size()) {
            return Status::invalid_argument("tp segment range is not contiguous",
                                            __FILE__, __LINE__);
        }
        next_begin = segment.dispatch_end;
        const bool combines = segment.barrier_after != ::ps::runtime::TpBarrierKind::None;
        HostResolvedValue value{};
        uint32_t producer_begin = segment.dispatch_begin;
        ctx.tp_peer_armed = false;
        ctx.tp_peer_out = nullptr;
        ctx.tp_peer_written = false;
        ctx.tp_peer_value = 0;
        if (combines) {
            auto resolved = resolve_host_value(program, segment.combine_value, ctx);
            if (!resolved.ok()) return resolved.status();
            value = resolved.value();
            if (value.ptr == nullptr || value.feature_count == 0 || value.row_stride == 0) {
                return Status::invalid_state("tp combine value is empty", __FILE__, __LINE__);
            }
            const uint64_t arm_rows = probe_rows(value, ctx);
            const uint64_t arm_bytes = arm_rows * static_cast<uint64_t>(value.row_stride) *
                                       ::ps::runtime::value_dtype_bytes(value.dtype);
            if (arm_bytes != 0) {
                void* peer = nullptr;
                Status peer_status = barrier->peer_write_target(
                    rank, static_cast<std::size_t>(arm_bytes), peer);
                if (!peer_status.ok()) return peer_status;
                if (peer != nullptr) {
                    ctx.tp_peer_value = segment.combine_value.id;
                    ctx.tp_peer_out = peer;
                    ctx.tp_peer_armed = true;
                }
            }
            if (segment.combine_local_cols != 0 &&
                segment.combine_local_cols < value.feature_count) {
                uint32_t producer = segment.dispatch_end;
                for (uint32_t i = segment.dispatch_end; i > segment.dispatch_begin; --i) {
                    const auto& binding = program.dispatches[i - 1];
                    bool writes_combine = false;
                    for (uint32_t o = 0; o < binding.output_count; ++o) {
                        if (binding.output_slots[o] == segment.combine_value.id) {
                            writes_combine = true;
                            break;
                        }
                    }
                    if (writes_combine) {
                        producer = i - 1;
                        break;
                    }
                }
                if (producer == segment.dispatch_end) {
                    return Status::invalid_state("tp combine producer dispatch not found",
                                                 __FILE__, __LINE__);
                }
                producer_begin = producer;
                if (producer_begin != segment.dispatch_begin) {
                    Status head = execute_program_range(program, ctx, stream,
                                                        segment.dispatch_begin,
                                                        producer_begin);
                    if (!head.ok()) return head;
                }
                const uint64_t rows = probe_rows(value, ctx);
                const uint64_t bytes = rows * static_cast<uint64_t>(value.row_stride) *
                                       ::ps::runtime::value_dtype_bytes(value.dtype);
                if (bytes != 0) {
                    const hipError_t memset_err =
                        hipMemsetAsync(value.ptr, 0, static_cast<size_t>(bytes), stream);
                    if (memset_err != hipSuccess) {
                        return Status::hip_error("tp combine zero fill",
                                                 hipGetErrorString(memset_err), __FILE__,
                                                 __LINE__);
                    }
                }
            }
        }
        Status st = execute_program_range(program, ctx, stream, producer_begin,
                                          segment.dispatch_end);
        if (!st.ok()) return st;
        if (!combines) continue;
        ::ps::runtime::TpSumTarget target;
        target.ptr = value.ptr;
        target.rows = probe_rows(value, ctx);
        target.row_stride = value.row_stride;
        target.feature_count = value.feature_count;
        target.dtype = value.dtype;
        if (std::getenv("SHELF_TP_SUM_DEBUG") != nullptr) {
            std::fprintf(stderr,
                         "SUM rank=%d id=%u ptr=%p rows=%u stride=%u feat=%u dtype=%d "
                         "actual_rows=%u actual_out=%u\n",
                         rank, segment.combine_value.id, target.ptr, target.rows,
                         target.row_stride, target.feature_count,
                         static_cast<int>(target.dtype), ctx.actual_rows,
                         ctx.actual_outputs);
        }
        st = barrier->arrive(rank, target, stream,
                             ctx.tp_peer_armed && ctx.tp_peer_written);
        ctx.tp_peer_armed = false;
        ctx.tp_peer_out = nullptr;
        ctx.tp_peer_written = false;
        ctx.tp_peer_value = 0;
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

Status execute_program(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    if (ctx.error_word == nullptr)
        return Status::invalid_state("error word not bound", __FILE__, __LINE__);
    const hipError_t herr =
        hipMemsetAsync(ctx.error_word, 0, sizeof(uint32_t), stream);
    if (herr != hipSuccess)
        return Status::hip_error("hipMemsetAsync error word",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    return execute_program_range(program, ctx, stream, 0u,
                                 program.dispatches.size());
}

Status upload_host_weight_table(
    const ::ps::runtime::WeightTableView& weights,
    WeightSlot** out,
    hipStream_t stream) {
    if (out == nullptr)
        return Status::invalid_argument("weight table out is null", __FILE__, __LINE__);
    *out = nullptr;
    std::vector<WeightSlot> host(weights.count);
    for (size_t i = 0; i < weights.count; ++i) {
        auto s = weights.at(i);
        Status vst = validate_host_weight_slot(s, i);
        if (!vst.ok()) return vst;
        host[i].device_ptr = s.device_ptr;
        host[i].rows = s.rows;
        host[i].cols = s.cols;
        host[i].k_padded = s.k_padded;
        host[i].encoding = s.encoding;
        host[i].codes = s.codes;
        host[i].scales = s.scales;
        host[i].storage_scale_stride_bytes = s.storage_scale_stride_bytes;
        host[i].compute_codes = s.compute_codes;
        host[i].compute_scales_bf16 = s.compute_scales_bf16;
        host[i].codes_row_stride_bytes = s.codes_row_stride_bytes;
        host[i].scale_row_stride_bytes = s.scale_row_stride_bytes;
        host[i].weight_scale_group = s.weight_scale_group;
        host[i].preshuffled = s.preshuffled;
        host[i].output_col_offset = s.output_col_offset;
        host[i].output_col_total = s.output_col_total;
    }
    WeightSlot* dev = nullptr;
    hipError_t err =
        hipMalloc(reinterpret_cast<void**>(&dev), sizeof(WeightSlot) * host.size());
    if (err != hipSuccess)
        return Status::hip_error("hipMalloc weight table", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    err = hipMemcpyWithStream(dev, host.data(), sizeof(WeightSlot) * host.size(),
                              hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        ps::gpu::discard_cleanup_result(hipFree(dev));
        return Status::hip_error("hipMemcpy weight table", hipGetErrorString(err), __FILE__,
                                  __LINE__);
    }
    *out = dev;
    return Status::make_ok();
}

Status upload_host_parameter_table(
    const ::ps::runtime::StaticParameterTableView& params,
    ::ps::runtime::StaticParameterSlot** out,
    hipStream_t stream) {
    if (out == nullptr)
        return Status::invalid_argument("parameter table out is null", __FILE__, __LINE__);
    *out = nullptr;
    std::vector<::ps::runtime::StaticParameterSlot> host(params.count);
    for (size_t i = 0; i < params.count; ++i) {
        host[i] = params.slots[i];
    }
    ::ps::runtime::StaticParameterSlot* dev = nullptr;
    const size_t bytes =
        sizeof(::ps::runtime::StaticParameterSlot) * host.size();
    hipError_t err = hipMalloc(reinterpret_cast<void**>(&dev), bytes);
    if (err != hipSuccess)
        return Status::hip_error("hipMalloc parameter table", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    err = hipMemcpyWithStream(dev, host.data(), bytes, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        ps::gpu::discard_cleanup_result(hipFree(dev));
        return Status::hip_error("hipMemcpy parameter table", hipGetErrorString(err), __FILE__,
                                  __LINE__);
    }
    *out = dev;
    return Status::make_ok();
}

}
