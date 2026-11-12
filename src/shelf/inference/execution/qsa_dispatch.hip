#include <shelf/inference/execution/qsa_dispatch.h>

#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/model_dispatch_state.h>
#include <shelf/kernels/optimized/qsa.h>

#include <algorithm>

namespace shelf::execution {

using ::shelf::runtime::DispatchBinding;
using ::shelf::runtime::KernelId;
using ::shelf::runtime::PrimitiveStateKind;
using ::shelf::runtime::Program;
using ::shelf::runtime::ValueDType;
using ::shelf::runtime::ValueId;
using ::shelf::runtime::ValueRowDomain;

namespace {

Status bad(const char* msg) {
    return Status::invalid_state(msg, __FILE__, __LINE__);
}

const ::shelf::runtime::StateBinding* find_state_binding(const Program& program,
                                                         uint32_t descriptor_slot) {
    if (descriptor_slot == 0u) return nullptr;
    for (const auto& state : program.states) {
        if (state.slot + 1u == descriptor_slot) return &state;
    }
    return nullptr;
}

bool qsa_view_ready(const ::shelf::state::QsaKeyStatePoolDeviceView& view) {
    return view.pooled_base != nullptr && view.partial_base != nullptr &&
           view.pool_count_base != nullptr && view.partial_count_base != nullptr &&
           view.max_sequences != 0u && view.max_blocks != 0u &&
           view.compress_ratio != 0u && view.head_dim != 0u;
}

}  // namespace

Result<OptimizedLaunchResult> try_launch_qsa_key_cache(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 0u)
        return bad("qsa key cache: bad io count");
    const auto* state = find_state_binding(program, b.descriptor_slot);
    if (state == nullptr || state->kind != PrimitiveStateKind::QSA_KEY_STATE)
        return bad("qsa key cache: state binding missing");
    const auto& view = ctx.model_state.qsa;
    if (!qsa_view_ready(view)) return bad("qsa key cache: state pool missing");
    if (b.group_size != view.compress_ratio || b.flags != view.max_blocks)
        return bad("qsa key cache: geometry differs from the state pool");
    auto in = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in.ok()) return in.status();
    if (in.value().dtype != ValueDType::BF16 ||
        in.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        in.value().feature_count != view.head_dim || in.value().ptr == nullptr) {
        return bad("qsa key cache: value contract mismatch");
    }
    if (b.parameter_index0 >= ctx.host_parameter_count || ctx.host_parameters == nullptr)
        return bad("qsa key cache: norm parameter missing");
    const auto& weight = ctx.host_parameters[b.parameter_index0];
    if (weight.device_ptr == nullptr || weight.dtype != ValueDType::BF16 ||
        weight.elements < view.head_dim) {
        return bad("qsa key cache: norm parameter invalid");
    }
    const auto* requests =
        ctx.staged_requests != nullptr ? ctx.staged_requests : ctx.device_requests;
    const uint32_t num_requests = ctx.staged_request_count != 0u ? ctx.staged_request_count
                                                                : ctx.num_requests;
    if (requests == nullptr || num_requests == 0u || ctx.actual_rows == 0u ||
        ctx.row_positions == nullptr) {
        return bad("qsa key cache: requests missing");
    }

    ::shelf::kernel::QsaKeyCacheBatchArgs args;
    args.raw_keys = static_cast<const ::shelf::bf16_t*>(in.value().ptr);
    args.norm_weight = static_cast<const ::shelf::bf16_t*>(weight.device_ptr);
    args.positions = ctx.row_positions;
    args.requests = requests;
    args.pooled_base = view.pooled_base;
    args.partial_base = view.partial_base;
    args.pool_count_base = view.pool_count_base;
    args.partial_count_base = view.partial_count_base;
    args.partial_history = ctx.spec.qsa_spec_history.partial;
    args.count_history = ctx.spec.qsa_spec_history.counts;
    args.history_rows = ctx.spec.qsa_spec_history.rows;
    args.history_partial_state_stride =
        ctx.spec.qsa_spec_history.partial_state_stride;
    args.history_count_state_stride = ctx.spec.qsa_spec_history.count_state_stride;
    args.num_requests = num_requests;
    args.actual_rows = ctx.actual_rows;
    args.max_sequences = view.max_sequences;
    args.state_index = state->state_index;
    args.max_blocks = view.max_blocks;
    args.head_dim = view.head_dim;
    args.compress_ratio = b.group_size;
    args.rotary_dim = b.rotary_dim;
    args.raw_row_stride = in.value().row_stride;
    args.pooled_slot_stride = view.pooled_slot_stride;
    args.pooled_layer_stride = view.pooled_layer_stride;
    args.partial_slot_stride = view.partial_slot_stride;
    args.partial_layer_stride = view.partial_layer_stride;
    args.count_slot_stride = view.count_slot_stride;
    args.count_layer_stride = view.count_layer_stride;
    args.theta = b.scalar_a;
    args.position_scale = b.scalar_b;
    args.inv_freq = ctx.rope_inv_freq;
    // Prefill blocks are independent: one CTA per compressed block removes the serial
    // per-token loop. Speculative snapshots still need the ordered path.
    const bool parallel_blocks = args.partial_history == nullptr &&
                                 args.actual_rows >= 256u;
    const hipError_t error = parallel_blocks
                                 ? ::shelf::kernel::launch_qsa_key_cache_blocks(args, stream)
                                 : ::shelf::kernel::launch_qsa_key_cache_batch(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("qsa key cache launch", hipGetErrorString(error),
                                 __FILE__, __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_qsa_select(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 1u)
        return bad("qsa select: bad io count");
    const auto* state = find_state_binding(program, b.descriptor_slot);
    if (state == nullptr || state->kind != PrimitiveStateKind::QSA_KEY_STATE)
        return bad("qsa select: state binding missing");
    const auto& view = ctx.model_state.qsa;
    if (!qsa_view_ready(view)) return bad("qsa select: state pool missing");
    if (b.flags != view.max_blocks ||
        static_cast<uint32_t>(b.scalar_b) != view.compress_ratio)
        return bad("qsa select: geometry differs from the state pool");
    auto q = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!q.ok()) return q.status();
    auto mask = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!mask.ok()) return mask.status();
    if (q.value().dtype != ValueDType::BF16 ||
        q.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        q.value().feature_count !=
            static_cast<uint64_t>(b.group_size) * b.rotary_dim ||
        mask.value().dtype != ValueDType::U8 ||
        mask.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        mask.value().feature_count * 8u < b.flags ||
        q.value().ptr == nullptr || mask.value().ptr == nullptr) {
        return bad("qsa select: value contract mismatch");
    }
    if (ctx.row_sequence_slots == nullptr || ctx.row_positions == nullptr)
        return bad("qsa select: row metadata missing");
    const uint32_t rows = ctx.actual_rows;
    if (rows == 0u) return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};

    ::shelf::kernel::QsaSelectBatchArgs args;
    args.q = static_cast<const ::shelf::bf16_t*>(q.value().ptr);
    args.pooled_base = view.pooled_base;
    args.block_mask = static_cast<uint8_t*>(mask.value().ptr);
    args.row_sequence_slots = ctx.row_sequence_slots;
    args.positions = ctx.row_positions;
    args.rows = rows;
    args.blocks = b.flags;
    args.n_heads = b.group_size;
    args.head_dim = b.rotary_dim;
    args.compress_ratio = static_cast<uint32_t>(b.scalar_b);
    args.state_index = state->state_index;
    args.max_sequences = view.max_sequences;
    args.top_k = b.compute_spec;
    args.q_row_stride = q.value().row_stride;
    args.mask_row_bytes = mask.value().row_stride;
    args.pooled_slot_stride = view.pooled_slot_stride;
    args.pooled_layer_stride = view.pooled_layer_stride;
    args.scale = b.scalar_a;
    if (ctx.qsa_scores != nullptr && ctx.qsa_topk_scores != nullptr &&
        ctx.qsa_topk_indices != nullptr && ctx.staged_requests != nullptr &&
        ctx.staged_request_count != 0u && args.rows >= 1u &&
        args.n_heads == 4u && args.head_dim == ::shelf::kernel::kQsaHeadDim) {
        constexpr uint32_t kMaxSlices = 256u;
        struct Slice {
            uint32_t begin = 0;
            uint32_t count = 0;
            uint32_t blocks = 0;
        };
        const auto* reqs = ctx.staged_requests;
        const uint32_t req_count = ctx.staged_request_count;
        if (req_count <= kMaxSlices) {
            uint32_t launches = 0u;
            for (uint32_t rc = 0u; rc < args.rows; rc += ::shelf::kernel::kQsaScoreRowTile) {
                const uint32_t rc_rows = args.rows - rc < ::shelf::kernel::kQsaScoreRowTile
                    ? args.rows - rc
                    : ::shelf::kernel::kQsaScoreRowTile;
                Slice slices[kMaxSlices];
                uint32_t slice_count = 0u;
                uint32_t max_blocks = 0u;
                for (uint32_t i = 0u; i < req_count; ++i) {
                    const auto& req = reqs[i];
                    const uint32_t begin = std::max(req.row_begin, rc);
                    const uint32_t end = std::min(req.row_begin + req.row_count, rc + rc_rows);
                    if (end <= begin) continue;
                    const uint32_t blocks = std::min(
                        args.blocks, req.sequence_length / args.compress_ratio);
                    slices[slice_count++] = Slice{begin, end - begin, blocks};
                    if (blocks > max_blocks) max_blocks = blocks;
                }
                if (max_blocks == 0u) {
                    ::shelf::kernel::QsaTopkArgs ta{};
                    ta.scores = ctx.qsa_scores;
                    ta.state_scores = ctx.qsa_topk_scores;
                    ta.state_indices = ctx.qsa_topk_indices;
                    ta.block_mask = args.block_mask +
                        static_cast<uint64_t>(rc) * args.mask_row_bytes;
                    ta.selected = args.selected != nullptr
                        ? args.selected +
                              static_cast<uint64_t>(rc) * args.selected_row_stride
                        : nullptr;
                    ta.positions = args.positions + rc;
                    ta.rows = rc_rows;
                    ta.block_begin = 0u;
                    ta.block_count = 0u;
                    ta.score_row_stride = ::shelf::kernel::kQsaScoreBlockTile;
                    ta.blocks = args.blocks;
                    ta.compress_ratio = args.compress_ratio;
                    ta.top_k = args.top_k;
                    ta.mask_row_bytes = args.mask_row_bytes;
                    ta.selected_row_stride = args.selected_row_stride;
                    ta.first = 1u;
                    ta.last = 1u;
                    const hipError_t error =
                        ::shelf::kernel::launch_qsa_topk(ta, stream);
                    if (error != hipSuccess)
                        return Status::hip_error("qsa topk launch",
                                                 hipGetErrorString(error), __FILE__,
                                                 __LINE__);
                    ++launches;
                    continue;
                }
                for (uint32_t bt = 0u; bt < max_blocks;
                     bt += ::shelf::kernel::kQsaScoreBlockTile) {
                    const uint32_t tile_blocks =
                        max_blocks - bt < ::shelf::kernel::kQsaScoreBlockTile
                            ? max_blocks - bt
                            : ::shelf::kernel::kQsaScoreBlockTile;
                    for (uint32_t si = 0u; si < slice_count; ++si) {
                        const Slice& slice = slices[si];
                        if (slice.blocks <= bt) continue;
                        ::shelf::kernel::QsaScoreArgs sa{};
                        sa.q = args.q + static_cast<uint64_t>(slice.begin) * args.q_row_stride;
                        sa.pooled_base = args.pooled_base;
                        sa.scores = ctx.qsa_scores;
                        sa.row_sequence_slots = args.row_sequence_slots + slice.begin;
                        sa.rows = slice.count;
                        sa.row_begin = slice.begin - rc;
                        sa.block_begin = bt;
                        sa.blocks = std::min(tile_blocks, slice.blocks - bt);
                        sa.block_tile = args.rows < 64u
                                            ? 64u
                                            : (args.rows < 256u
                                                   ? 256u
                                                   : ::shelf::kernel::kQsaScoreBlocksPerCta);
                        sa.score_row_stride = ::shelf::kernel::kQsaScoreBlockTile;
                        sa.n_heads = args.n_heads;
                        sa.head_dim = args.head_dim;
                        sa.compress_ratio = args.compress_ratio;
                        sa.state_index = args.state_index;
                        sa.max_sequences = args.max_sequences;
                        sa.q_row_stride = args.q_row_stride;
                        sa.pooled_slot_stride = args.pooled_slot_stride;
                        sa.pooled_layer_stride = args.pooled_layer_stride;
                        sa.scale = args.scale;
                        const hipError_t error =
                            ::shelf::kernel::launch_qsa_score_wmma(sa, stream);
                        if (error != hipSuccess)
                            return Status::hip_error("qsa score launch",
                                                     hipGetErrorString(error), __FILE__,
                                                     __LINE__);
                        ++launches;
                    }
                    ::shelf::kernel::QsaTopkArgs ta{};
                    ta.scores = ctx.qsa_scores;
                    ta.state_scores = ctx.qsa_topk_scores;
                    ta.state_indices = ctx.qsa_topk_indices;
                    ta.block_mask = args.block_mask +
                        static_cast<uint64_t>(rc) * args.mask_row_bytes;
                    ta.selected = args.selected != nullptr
                        ? args.selected +
                              static_cast<uint64_t>(rc) * args.selected_row_stride
                        : nullptr;
                    ta.positions = args.positions + rc;
                    ta.rows = rc_rows;
                    ta.block_begin = bt;
                    ta.block_count = tile_blocks;
                    ta.score_row_stride = ::shelf::kernel::kQsaScoreBlockTile;
                    ta.blocks = args.blocks;
                    ta.compress_ratio = args.compress_ratio;
                    ta.top_k = args.top_k;
                    ta.mask_row_bytes = args.mask_row_bytes;
                    ta.selected_row_stride = args.selected_row_stride;
                    ta.first = bt == 0u ? 1u : 0u;
                    ta.last = bt + tile_blocks >= max_blocks ? 1u : 0u;
                    const hipError_t error =
                        ::shelf::kernel::launch_qsa_topk(ta, stream);
                    if (error != hipSuccess)
                        return Status::hip_error("qsa topk launch",
                                                 hipGetErrorString(error), __FILE__,
                                                 __LINE__);
                    ++launches;
                }
            }
            record_physical_launch(launches);
            record_optimized_op();
            return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
        }
    }
    const hipError_t error = ::shelf::kernel::launch_qsa_select_batch(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("qsa select launch", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_qsa_rope(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 1u)
        return bad("qsa rope: bad io count");
    auto in = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in.ok()) return in.status();
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();
    if (in.value().dtype != ValueDType::BF16 || out.value().dtype != ValueDType::BF16 ||
        in.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        out.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        in.value().feature_count != static_cast<uint64_t>(b.group_size) * b.flags ||
        out.value().feature_count != in.value().feature_count ||
        in.value().ptr == nullptr || out.value().ptr == nullptr) {
        return bad("qsa rope: value contract mismatch");
    }
    if (ctx.row_positions == nullptr) return bad("qsa rope: positions missing");
    const uint32_t rows = ctx.actual_rows;
    if (rows == 0u) return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    // The kernel rotates in place; the graph gives the rope its own output buffer, so the
    // caller hands the source over by copying it first. The two rows can have different
    // aligned strides.
    if (in.value().ptr != out.value().ptr) {
        const std::size_t width = static_cast<std::size_t>(in.value().feature_count) *
                                  sizeof(::shelf::bf16_t);
        const hipError_t copy_error = hipMemcpy2DAsync(
            out.value().ptr, static_cast<std::size_t>(out.value().row_stride) *
                                 sizeof(::shelf::bf16_t),
            in.value().ptr,
            static_cast<std::size_t>(in.value().row_stride) * sizeof(::shelf::bf16_t),
            width, rows, hipMemcpyDeviceToDevice, stream);
        if (copy_error != hipSuccess)
            return Status::hip_error("qsa rope copy", hipGetErrorString(copy_error),
                                     __FILE__, __LINE__);
    }

    ::shelf::kernel::QsaRopeArgs args;
    args.x = static_cast<::shelf::bf16_t*>(out.value().ptr);
    args.positions = ctx.row_positions;
    args.inv_freq = ctx.rope_inv_freq;
    args.rows = rows;
    args.heads = b.group_size;
    args.head_dim = b.flags;
    args.rotary_dim = b.rotary_dim;
    args.theta = b.scalar_a;
    args.position_scale = b.scalar_b;
    const hipError_t error = ::shelf::kernel::launch_qsa_rope(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("qsa rope launch", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
