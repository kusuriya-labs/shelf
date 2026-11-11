#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/runtime/spec_decode.h>
#include <shelf/models/qwen35/runtime/dflash2_tree.h>
#include <shelf/models/qwen35/runtime/dflash2_tree_budget.h>
#include <shelf/models/qwen35/kernels/dflash2/candidate_selector.h>
#include <shelf/models/qwen35/kernels/dflash2/feature_concat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace shelf::qwen35::runtime {

namespace {

bool env_flag_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;
    return value[0] != '0';
}

uint32_t max_target_tap(const dflash2::DFlash2Config& config) {
    uint32_t max_tap = 0u;
    for (std::size_t i = 0; i < config.num_target_layer_ids; ++i) {
        max_tap = std::max(max_tap, static_cast<uint32_t>(config.target_layer_ids[i]));
    }
    return max_tap;
}

ScheduledBatch make_prefill_batch(
    PagedSequenceState& sequence,
    std::vector<ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t prefix_tokens,
    bool compute_logits,
    const SamplingConfig* sampling = nullptr,
    uint64_t sampling_index = 0u) {
    ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? ::shelf::runtime::ExecutionClass::DECODE
                                           : ::shelf::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = prefix_tokens;
    req.compute_logits = compute_logits;
    req.sample = compute_logits;
    req.num_output_rows = 1u;
    if (sampling != nullptr) {
        req.sampling = *sampling;
        req.sampling_index = sampling_index;
    }
    requests.clear();
    requests.push_back(req);

    ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = num_tokens == 1u ? 1u : 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = num_tokens == 1u ? 0u : 1u;
    batch.speculative_verify = false;
    return batch;
}

}  // namespace

Result<DFlash2SpecDecoder> create_dflash2_spec_decoder(
    Executor& target,
    dflash2::DFlash2Executor& draft,
    dflash2::DFlash2ContextState& context,
    PagedSequenceState& sequence,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* tp_ranks,
    TpBatchHook* tp_hook,
    DFlash2SpecWorkspace* workspace) {
    if (target.model.config == nullptr || target.model.embed_tokens == nullptr) {
        return Status::invalid_state("create_dflash2_spec_decoder: target model is null",
                                     __FILE__, __LINE__);
    }
    if (!draft.initialized || draft.config == nullptr) {
        return Status::invalid_state("create_dflash2_spec_decoder: draft executor not initialized",
                                     __FILE__, __LINE__);
    }
    if (!context.initialized) {
        return Status::invalid_state("create_dflash2_spec_decoder: context not initialized",
                                     __FILE__, __LINE__);
    }
    if (!sequence.is_allocated()) {
        return Status::invalid_state("create_dflash2_spec_decoder: sequence not allocated",
                                     __FILE__, __LINE__);
    }

    const dflash2::DFlash2Config& dcfg = *draft.config;
    const shelf::qwen_hybrid::QwenHybridConfig& tc = *target.model.config;
    const shelf::weights::MatrixWeight& embed = *target.model.embed_tokens;

    if (embed.cols != dcfg.hidden_size) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: target hidden_size mismatch", __FILE__, __LINE__);
    }
    if (embed.rows != dcfg.vocab_size) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: target vocab_size mismatch", __FILE__, __LINE__);
    }
    if (tc.num_hidden_layers <= max_target_tap(dcfg)) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: target layer count too small for taps", __FILE__,
            __LINE__);
    }
    if (dcfg.mask_token_id >= dcfg.vocab_size) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: mask_token_id exceeds vocab_size", __FILE__,
            __LINE__);
    }
    if (draft.target_embed_tokens == nullptr || draft.target_lm_head == nullptr) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: draft is missing target embed_tokens/lm_head",
            __FILE__, __LINE__);
    }
    if (draft.target_lm_head->rows != dcfg.vocab_size ||
        draft.target_lm_head->cols != dcfg.hidden_size) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: target lm_head shape mismatch", __FILE__, __LINE__);
    }
    if (config.num_drafts == 0u || config.num_drafts > dcfg.max_draft_tokens()) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: num_drafts out of range", __FILE__, __LINE__);
    }

    DFlash2SpecDecoder decoder;
    decoder.target = &target;
    decoder.draft = &draft;
    decoder.context = &context;
    decoder.sequence = &sequence;
    decoder.stream = stream;
    decoder.tp_hook = tp_hook;
    decoder.config = config;
    decoder.hidden_size = static_cast<uint32_t>(dcfg.hidden_size);
    decoder.device_token_bridge =
        env_flag_enabled("SHELF_DFLASH2_DEVICE_TOKEN_BRIDGE", true);
    decoder.host_proposal_visibility =
        env_flag_enabled("SHELF_DFLASH2_HOST_PROPOSAL_D2H", false);
    if (tp_hook != nullptr) decoder.device_token_bridge = false;
    decoder.gdn_key_heads =
        static_cast<uint32_t>(tc.linear_num_key_heads != 0 ? tc.linear_num_key_heads : 16u);

    DFlash2SpecWorkspace local_workspace;
    DFlash2SpecWorkspace* bound = workspace;
    if (bound == nullptr) {
        auto workspace_result = create_dflash2_spec_workspace(
            target, draft, gdn_pool, arena, config, stream, tp_ranks);
        if (!workspace_result.ok()) return workspace_result.status();
        local_workspace = workspace_result.release();
        bound = &local_workspace;
        decoder.owns_workspace = true;
    }
    const Status bind_st = bind_dflash2_spec_workspace(decoder, *bound);
    if (!bind_st.ok()) {
        if (decoder.owns_workspace) {
            (void)shutdown_dflash2_spec_workspace(local_workspace);
        }
        return bind_st;
    }

    decoder.initialized = true;
    return decoder;
}

namespace {

Result<DFlash2SpecWorkspace> create_spec_workspace_impl(
    Executor& target,
    uint32_t draft_max_drafts,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* tp_ranks,
    bool step_buffers,
    const std::vector<void*>* shared_rec_snapshots) {
    if (target.model.config == nullptr || target.model.embed_tokens == nullptr) {
        return Status::invalid_state("create_dflash2_spec_workspace: target model is null",
                                     __FILE__, __LINE__);
    }
    if (config.num_drafts == 0u || config.num_drafts > draft_max_drafts) {
        return Status::invalid_argument(
            "create_dflash2_spec_workspace: num_drafts out of range", __FILE__, __LINE__);
    }

    DFlash2SpecWorkspace workspace;
    std::vector<DFlash2GdnRankBinding> bindings;
    if (tp_ranks != nullptr && !tp_ranks->empty()) {
        bindings = *tp_ranks;
    } else {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) dev = 0;
        bindings.push_back(DFlash2GdnRankBinding{&gdn_pool, &arena, stream, dev,
                                                 &target.host_model_state, &target});
    }
    workspace.gdn_ranks.resize(bindings.size());
    for (std::size_t r = 0; r < bindings.size(); ++r) {
        DFlash2GdnRankState& g = workspace.gdn_ranks[r];
        g.pool = bindings[r].pool;
        g.stream = bindings[r].stream;
        g.model_state = bindings[r].model_state;
        g.device = bindings[r].device;
    }

    {
        gpu::GpuArena& leader_arena = *bindings.front().arena;
        auto leader_scope = shelf::gpu::ScopedDevice::create(bindings.front().device);
        if (!leader_scope.ok()) return leader_scope.status();
        auto verify_alloc = leader_arena.allocate_aligned(
            kDFlash2SpecMaxVerifyRows * sizeof(int32_t), 256u);
        if (!verify_alloc.ok()) return verify_alloc.status();
        workspace.verify_token_ids_device =
            static_cast<int32_t*>(verify_alloc.release().data());
        auto decision_alloc = leader_arena.allocate_aligned(
            2u * kDFlash2SpecMaxVerifyRows * sizeof(int32_t), 256u);
        if (!decision_alloc.ok()) return decision_alloc.status();
        workspace.decision_staging_device =
            static_cast<int32_t*>(decision_alloc.release().data());
    }

    {
        hipEvent_t events[4] = {nullptr, nullptr, nullptr, nullptr};
        for (int i = 0; i < 4; ++i) {
            hipError_t err = hipEventCreateWithFlags(&events[i], hipEventDefault);
            if (err != hipSuccess) {
                for (int j = 0; j < i; ++j) {
                    shelf::gpu::discard_cleanup_result(hipEventDestroy(events[j]));
                }
                return Status::hip_error("create_dflash2_spec_workspace hipEventCreate",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
        workspace.draft_start_event = events[0];
        workspace.draft_stop_event = events[1];
        workspace.verify_start_event = events[2];
        workspace.verify_stop_event = events[3];
    }

    const uint32_t history_rows = config.num_drafts + (config.tree ? 1u + config.tree_extra : 0u);
    const shelf::qwen_hybrid::QwenHybridConfig& tc = *target.model.config;
    workspace.gdn_key_heads =
        static_cast<uint32_t>(tc.linear_num_key_heads != 0 ? tc.linear_num_key_heads : 16u);

    for (std::size_t r = 0; r < bindings.size(); ++r) {
        DFlash2GdnRankState& g = workspace.gdn_ranks[r];
        gpu::GpuArena& rank_arena = *bindings[r].arena;
        GdnStatePool& rank_pool = *bindings[r].pool;
        auto dev_scope = shelf::gpu::ScopedDevice::create(g.device);
        if (!dev_scope.ok()) {
            (void)shutdown_dflash2_spec_workspace(workspace);
            return dev_scope.status();
        }

        g.conv_bytes = spec_gdn_conv_bytes(rank_pool);
        g.rec_bytes = spec_gdn_recurrent_bytes(rank_pool);
        const GdnStatePoolDeviceView pool_view = rank_pool.device_view();

        if (g.conv_bytes != 0u) {
            auto conv_alloc = rank_arena.allocate_aligned(g.conv_bytes, 256u);
            if (!conv_alloc.ok()) {
                (void)shutdown_dflash2_spec_workspace(workspace);
                return conv_alloc.status();
            }
            g.conv_snapshot = conv_alloc.release().data();
            workspace.gdn_snapshot_bytes += g.conv_bytes;
        }
        if (g.rec_bytes != 0u) {
            if (shared_rec_snapshots != nullptr && r < shared_rec_snapshots->size()) {
                // The recurrent snapshot is only used by serialized paths (prefix
                // shadow prefill); sessions share one buffer per rank.
                g.rec_snapshot = (*shared_rec_snapshots)[r];
            } else {
                auto rec_alloc = rank_arena.allocate_aligned(g.rec_bytes, 256u);
                if (!rec_alloc.ok()) {
                    (void)shutdown_dflash2_spec_workspace(workspace);
                    return rec_alloc.status();
                }
                g.rec_snapshot = rec_alloc.release().data();
                workspace.gdn_snapshot_bytes += g.rec_bytes;
            }
        }
        // The batched serve path keeps history / compact log in the shared batch
        // context, so per-session sessions skip them (step_buffers == false).
        if (step_buffers) {
            auto history_result = create_gdn_spec_history(rank_arena, rank_pool, history_rows,
                                                          config.tree);
            if (!history_result.ok()) {
                (void)shutdown_dflash2_spec_workspace(workspace);
                return history_result.status();
            }
            g.history = history_result.release();
            if (config.tree) {
                const size_t rows = history_rows;
                auto metadata = rank_arena.allocate_aligned(
                    (3u * rows + 1u + rows * (rows + 1u) / 2u) * sizeof(uint32_t), 256u);
                if (!metadata.ok()) {
                    (void)shutdown_dflash2_spec_workspace(workspace);
                    return metadata.status();
                }
                auto* base = static_cast<uint32_t*>(metadata.release().data());
                g.tree_parent = reinterpret_cast<int32_t*>(base);
                g.tree_offsets = base + rows;
                g.tree_rope = base + 2u * rows + 1u;
                g.tree_ancestors = base + 3u * rows + 1u;
                auto selector = rank_arena.allocate_aligned((2u * rows + 16u * rows) * sizeof(uint32_t), 256u);
                if (!selector.ok()) {
                    (void)shutdown_dflash2_spec_workspace(workspace);
                    return selector.status();
                }
                auto* selector_base = static_cast<int32_t*>(selector.release().data());
                g.tree_predecessors = selector_base;
                g.tree_selected = selector_base + rows;
                g.tree_scores = reinterpret_cast<float*>(selector_base + 2u * rows);
            }

            const uint64_t states = pool_view.num_gdn_states;
            const uint64_t log_rows = config.tree &&
                (env_flag_enabled("SHELF_DFLASH2_TREE_COMPACT_BRANCH", false) ||
                 (env_flag_enabled("SHELF_GDN_VERIFY_CHUNKED", false) &&
                  env_flag_enabled("SHELF_GDN_TREE_CHUNKED", false)))
                ? history_rows : static_cast<uint64_t>(config.num_drafts) + 1u;
            const uint64_t vdim =
                static_cast<uint64_t>(pool_view.num_v_heads) * pool_view.head_v;
            const uint64_t qkdim = pool_view.num_v_heads * pool_view.head_k;
            const uint64_t dper = log_rows * vdim;
            const uint64_t kper = log_rows * qkdim;
            const uint64_t aper = log_rows * pool_view.num_v_heads;
            auto d_alloc = rank_arena.allocate_aligned(states * dper * sizeof(float), 256u);
            if (!d_alloc.ok()) {
                (void)shutdown_dflash2_spec_workspace(workspace);
                return d_alloc.status();
            }
            g.compact_delta = static_cast<float*>(d_alloc.release().data());
            auto k_alloc = rank_arena.allocate_aligned(states * kper * sizeof(float), 256u);
            if (!k_alloc.ok()) {
                (void)shutdown_dflash2_spec_workspace(workspace);
                return k_alloc.status();
            }
            g.compact_k = static_cast<float*>(k_alloc.release().data());
            auto a_alloc = rank_arena.allocate_aligned(states * aper * sizeof(float), 256u);
            if (!a_alloc.ok()) {
                (void)shutdown_dflash2_spec_workspace(workspace);
                return a_alloc.status();
            }
            g.compact_a = static_cast<float*>(a_alloc.release().data());
            g.compact_delta_layer_stride = dper;
            g.compact_k_layer_stride = kper;
            g.compact_a_layer_stride = aper;
            workspace.gdn_compact_bytes = states * (dper + kper + aper) * sizeof(float);
        }

        auto req_alloc = rank_arena.allocate_aligned(
            sizeof(::shelf::runtime::DeviceRequestDescriptor), 256u);
        if (!req_alloc.ok()) {
            (void)shutdown_dflash2_spec_workspace(workspace);
            return req_alloc.status();
        }
        g.commit_request =
            static_cast<::shelf::runtime::DeviceRequestDescriptor*>(req_alloc.release().data());
    }

    workspace.initialized = true;
    return workspace;
}

}  // namespace

Result<DFlash2SpecWorkspace> create_dflash2_spec_workspace_for_drafts(
    Executor& target,
    uint32_t draft_max_drafts,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* tp_ranks,
    bool step_buffers,
    const std::vector<void*>* shared_rec_snapshots) {
    return create_spec_workspace_impl(target, draft_max_drafts, gdn_pool, arena, config,
                                      stream, tp_ranks, step_buffers, shared_rec_snapshots);
}

Result<DFlash2SpecWorkspace> create_dflash2_spec_workspace(
    Executor& target,
    dflash2::DFlash2Executor& draft,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* tp_ranks) {
    if (!draft.initialized || draft.config == nullptr) {
        return Status::invalid_state("create_dflash2_spec_workspace: draft not initialized",
                                     __FILE__, __LINE__);
    }
    return create_spec_workspace_impl(target, static_cast<uint32_t>(draft.config->max_draft_tokens()),
                                      gdn_pool, arena, config, stream, tp_ranks, true, nullptr);
}

Status bind_dflash2_spec_workspace(
    DFlash2SpecDecoder& decoder,
    const DFlash2SpecWorkspace& workspace) {
    if (!workspace.initialized) {
        return Status::invalid_state("bind_dflash2_spec_workspace: workspace not initialized",
                                     __FILE__, __LINE__);
    }
    if (decoder.sequence == nullptr) {
        return Status::invalid_state("bind_dflash2_spec_workspace: decoder has no sequence",
                                     __FILE__, __LINE__);
    }
    decoder.gdn_ranks = workspace.gdn_ranks;
    decoder.gdn_snapshot_bytes = workspace.gdn_snapshot_bytes;
    decoder.gdn_compact_bytes = workspace.gdn_compact_bytes;
    decoder.gdn_key_heads = workspace.gdn_key_heads;
    decoder.verify_token_ids_device = workspace.verify_token_ids_device;
    decoder.decision_staging_device = workspace.decision_staging_device;
    decoder.draft_start_event = workspace.draft_start_event;
    decoder.draft_stop_event = workspace.draft_stop_event;
    decoder.verify_start_event = workspace.verify_start_event;
    decoder.verify_stop_event = workspace.verify_stop_event;

    for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
        if (g.commit_request == nullptr) continue;
        auto scope = shelf::gpu::ScopedDevice::create(g.device);
        if (!scope.ok()) return scope.status();
        ::shelf::runtime::DeviceRequestDescriptor req{};
        req.request_handle.slot = decoder.sequence->slot;
        req.row_begin = 0u;
        req.row_count = 0u;
        const hipError_t err =
            hipMemcpy(g.commit_request, &req, sizeof(req), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            return Status::hip_error("bind_dflash2_spec_workspace gdn commit request",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Status shutdown_dflash2_spec_workspace(DFlash2SpecWorkspace& workspace) noexcept {
    Status first_error = Status::make_ok();
    auto destroy_event = [&](hipEvent_t& event, const char* tag) noexcept {
        if (event == nullptr) return;
        hipError_t err = hipEventDestroy(event);
        if (first_error.ok() && err != hipSuccess) {
            first_error = Status::hip_error(tag, hipGetErrorString(err), __FILE__, __LINE__);
        }
        event = nullptr;
    };
    destroy_event(workspace.draft_start_event, "hipEventDestroy draft_start");
    destroy_event(workspace.draft_stop_event, "hipEventDestroy draft_stop");
    destroy_event(workspace.verify_start_event, "hipEventDestroy verify_start");
    destroy_event(workspace.verify_stop_event, "hipEventDestroy verify_stop");
    for (DFlash2GdnRankState& g : workspace.gdn_ranks) {
        const Status history_st = shutdown_gdn_spec_history(g.history);
        if (!history_st.ok() && first_error.ok()) first_error = history_st;
        g.conv_snapshot = nullptr;
        g.rec_snapshot = nullptr;
        g.compact_delta = nullptr;
        g.compact_k = nullptr;
        g.compact_a = nullptr;
        g.commit_request = nullptr;
    }
    workspace.gdn_ranks.clear();
    workspace.verify_token_ids_device = nullptr;
    workspace.decision_staging_device = nullptr;
    workspace.gdn_snapshot_bytes = 0u;
    workspace.gdn_compact_bytes = 0u;
    workspace.gdn_key_heads = 0u;
    workspace.initialized = false;
    return first_error;
}

void dflash2_spec_decoder_set_sampling(DFlash2SpecDecoder& decoder,
                                       const SamplingConfig& sampling) {
    decoder.sampling = sampling;
    decoder.sample_index = 0u;
}

Status dflash2_spec_decoder_shutdown(DFlash2SpecDecoder& decoder) noexcept {
    Status first_error = Status::make_ok();
    if (decoder.owns_workspace) {
        auto destroy_event = [&](hipEvent_t& event, const char* tag) noexcept {
            if (event == nullptr) return;
            hipError_t err = hipEventDestroy(event);
            if (first_error.ok() && err != hipSuccess) {
                first_error =
                    Status::hip_error(tag, hipGetErrorString(err), __FILE__, __LINE__);
            }
            event = nullptr;
        };
        destroy_event(decoder.draft_start_event, "hipEventDestroy draft_start");
        destroy_event(decoder.draft_stop_event, "hipEventDestroy draft_stop");
        destroy_event(decoder.verify_start_event, "hipEventDestroy verify_start");
        destroy_event(decoder.verify_stop_event, "hipEventDestroy verify_stop");
        for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
            if (g.history.initialized) {
                const Status history_st = shutdown_gdn_spec_history(g.history);
                if (!history_st.ok() && first_error.ok()) first_error = history_st;
            }
        }
        decoder.owns_workspace = false;
    }
    decoder.verify_token_ids_device = nullptr;
    decoder.decision_staging_device = nullptr;
    decoder.device_token_bridge = false;
    decoder.host_proposal_visibility = false;
    decoder.token_history.clear();
    for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
        g.conv_snapshot = nullptr;
        g.rec_snapshot = nullptr;
        g.conv_bytes = 0u;
        g.rec_bytes = 0u;
        g.compact_delta = nullptr;
        g.compact_k = nullptr;
        g.compact_a = nullptr;
        g.compact_delta_layer_stride = 0u;
        g.compact_k_layer_stride = 0u;
        g.compact_a_layer_stride = 0u;
        g.commit_request = nullptr;
        g.pool = nullptr;
        g.stream = nullptr;
    }
    decoder.gdn_ranks.clear();
    decoder.gdn_key_heads = 0u;
    decoder.hidden_size = 0u;
    decoder.target = nullptr;
    decoder.draft = nullptr;
    decoder.context = nullptr;
    decoder.sequence = nullptr;
    decoder.initialized = false;
    return first_error;
}

std::vector<ExecuteBatchOptions> build_gdn_options(DFlash2SpecDecoder& decoder,
                                                  uint32_t capture_rows) {
    std::vector<ExecuteBatchOptions> options(decoder.gdn_ranks.size());
    for (std::size_t r = 0; r < decoder.gdn_ranks.size(); ++r) {
        const DFlash2GdnRankState& g = decoder.gdn_ranks[r];
        options[r].gdn_spec_history = gdn_spec_history_view(g.history, capture_rows);
        options[r].gdn_compact.delta = g.compact_delta;
        options[r].gdn_compact.k = g.compact_k;
        options[r].gdn_compact.a = g.compact_a;
        options[r].gdn_compact.delta_layer_stride = g.compact_delta_layer_stride;
        options[r].gdn_compact.k_layer_stride = g.compact_k_layer_stride;
        options[r].gdn_compact.a_layer_stride = g.compact_a_layer_stride;
    }
    return options;
}

Result<BatchExecutionOutput> run_target_batch(DFlash2SpecDecoder& decoder,
                                              const ScheduledBatch& batch,
                                              const std::vector<ExecuteBatchOptions>& options) {
    if (decoder.tp_hook != nullptr) {
        return decoder.tp_hook->on_execute(batch, options.data());
    }
    return execute_batch(*decoder.target, batch, decoder.stream, options[0]);
}

Result<DFlash2PrefillOutput> dflash2_spec_prefill(
    DFlash2SpecDecoder& decoder,
    const int32_t* prompt_tokens,
    uint32_t prompt_count,
    uint32_t prefix_tokens,
    uint32_t page_tokens,
    const std::function<void(uint32_t)>& on_chunk_complete) {
    if (!decoder.initialized) {
        return Status::invalid_state("dflash2_spec_prefill: not initialized", __FILE__,
                                     __LINE__);
    }
    if (prompt_tokens == nullptr || prompt_count == 0u) {
        return Status::invalid_argument("dflash2_spec_prefill: empty prompt", __FILE__,
                                        __LINE__);
    }
    if (prefix_tokens >= prompt_count) {
        return Status::invalid_argument(
            "dflash2_spec_prefill: prefix must leave a suffix to prefill", __FILE__,
            __LINE__);
    }
    const uint32_t chunk_limit = decoder.target->config.max_batch_rows;
    if (chunk_limit == 0u) {
        return Status::invalid_state("dflash2_spec_prefill: target has no token capacity",
                                     __FILE__, __LINE__);
    }
    const uint32_t tap_count =
        static_cast<uint32_t>(decoder.draft->config->num_target_layer_ids);
    if (tap_count == 0u || tap_count != shelf::kernel::kDFlash2TargetTaps) {
        return Status::invalid_state("dflash2_spec_prefill: draft tap count mismatch", __FILE__,
                                     __LINE__);
    }

    DFlash2PrefillOutput out;
    std::vector<ScheduledRequest> requests;
    uint32_t offset = prefix_tokens;
    const uint32_t last_boundary =
        page_tokens == 0u ? prompt_count
                          : (prompt_count / page_tokens) * page_tokens;
    while (offset < prompt_count) {
        uint32_t n = std::min(chunk_limit, prompt_count - offset);
        if (page_tokens != 0u && offset < last_boundary &&
            offset + n > last_boundary) {
            n = last_boundary - offset;
        }
        const bool final_chunk = (offset + n == prompt_count);
        ScheduledBatch batch = make_prefill_batch(
            *decoder.sequence, requests, prompt_tokens + offset, n, offset, final_chunk,
            final_chunk ? &decoder.sampling : nullptr, decoder.sample_index);
        const std::vector<ExecuteBatchOptions> poptions = build_gdn_options(decoder, 1u);
        struct TimingEvents {
            hipEvent_t begin = nullptr;
            hipEvent_t target_end = nullptr;
            hipEvent_t append_end = nullptr;
            ~TimingEvents() {
                if (begin != nullptr) (void)hipEventDestroy(begin);
                if (target_end != nullptr) (void)hipEventDestroy(target_end);
                if (append_end != nullptr) (void)hipEventDestroy(append_end);
            }
        } timing;
        const bool timing_on = env_flag_enabled("SHELF_DFLASH_PREFILL_TIMING", false);
        if (timing_on && (hipEventCreate(&timing.begin) != hipSuccess ||
                          hipEventCreate(&timing.target_end) != hipSuccess ||
                          hipEventCreate(&timing.append_end) != hipSuccess ||
                          hipEventRecord(timing.begin, decoder.stream) != hipSuccess))
            return Status::invalid_state("dflash2 prefill timing event setup failed", __FILE__, __LINE__);
        auto executed = run_target_batch(decoder, batch, poptions);
        if (timing_on && hipEventRecord(timing.target_end, decoder.stream) != hipSuccess)
            return Status::invalid_state("dflash2 prefill target event failed", __FILE__, __LINE__);
        if (!executed.ok()) return executed.status();
        BatchExecutionOutput output = executed.release();

        std::array<const bf16_t*, shelf::kernel::kDFlash2TargetTaps> taps{};
        for (uint32_t t = 0; t < tap_count; ++t) {
            taps[t] = decoder.target->dflash_target_hidden[t].data<bf16_t>();
        }
        Status appended = dflash2::dflash2_append_target_taps(
            *decoder.draft, *decoder.context, taps.data(), tap_count, n, offset,
            decoder.stream, true);
        if (!appended.ok()) return appended;
        if (timing_on) {
            float target_ms = 0.0f, append_ms = 0.0f;
            if (hipEventRecord(timing.append_end, decoder.stream) != hipSuccess ||
                hipEventSynchronize(timing.append_end) != hipSuccess ||
                hipEventElapsedTime(&target_ms, timing.begin, timing.target_end) != hipSuccess ||
                hipEventElapsedTime(&append_ms, timing.target_end, timing.append_end) != hipSuccess)
                return Status::invalid_state("dflash2 prefill timing event failed", __FILE__, __LINE__);
            std::fprintf(stderr, "DFLASH_PREFILL_TIMING offset=%u rows=%u target_ms=%.3f append_ms=%.3f\n",
                         offset, n, target_ms, append_ms);
        }

        if (final_chunk) {
            const hipError_t err = hipMemcpy(
                &out.pending_token, output.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                return Status::hip_error("dflash2_spec_prefill sample copy",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
            if (out.pending_token < 0) {
                return Status::invalid_state("dflash2_spec_prefill produced no token",
                                             __FILE__, __LINE__);
            }
            decoder.sample_index += 1u;
        }
        offset += n;
        if (on_chunk_complete) {
            on_chunk_complete(offset);
        }
    }
    decoder.token_history.assign(prompt_tokens, prompt_tokens + prompt_count);
    decoder.token_history.push_back(out.pending_token);
    return out;
}

namespace {

struct ScopedTimer {
    std::chrono::steady_clock::time_point t0;
    double* out;
    explicit ScopedTimer(double* o) noexcept
        : t0(std::chrono::steady_clock::now()), out(o) {}
    ~ScopedTimer() {
        if (out != nullptr) {
            *out += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        }
    }
};

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

void commit_history_tokens(DFlash2SpecDecoder& decoder, const int32_t* tokens,
                           uint32_t count) {
    for (uint32_t i = 0u; i < count; ++i) {
        decoder.token_history.push_back(tokens[i]);
    }
    constexpr std::size_t kHistoryCap = 8192u;
    if (decoder.token_history.size() > kHistoryCap) {
        decoder.token_history.erase(decoder.token_history.begin(),
                                    decoder.token_history.end() - kHistoryCap / 2u);
    }
}

ScheduledBatch make_verify_batch(
    PagedSequenceState& sequence,
    std::vector<ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t prefix_tokens,
    uint32_t output_rows,
    bool speculative_verify,
    TokenIdsLocation token_location = TokenIdsLocation::Host,
    const SamplingConfig* sampling = nullptr,
    uint64_t sampling_index = 0u) {
    ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u
        ? ::shelf::runtime::ExecutionClass::DECODE
        : (speculative_verify ? ::shelf::runtime::ExecutionClass::SPEC_VERIFY
                              : ::shelf::runtime::ExecutionClass::PREFILL);
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = prefix_tokens;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = output_rows;
    if (sampling != nullptr) {
        req.sampling = *sampling;
        req.sampling_index = sampling_index;
    }
    requests.clear();
    requests.push_back(req);

    ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.token_ids_location = token_location;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = num_tokens == 1u ? 1u : 0u;
    batch.num_verify_requests =
        (speculative_verify && num_tokens != 1u) ? 1u : 0u;
    batch.num_prefill_requests =
        (!speculative_verify && num_tokens != 1u) ? 1u : 0u;
    batch.speculative_verify = speculative_verify;
    return batch;
}

void collect_target_taps(
    const Executor& target,
    std::array<const bf16_t*, shelf::kernel::kDFlash2TargetTaps>& taps) {
    for (uint32_t t = 0; t < shelf::kernel::kDFlash2TargetTaps; ++t) {
        taps[t] = target.dflash_target_hidden[t].data<bf16_t>();
    }
}

Status append_committed(
    DFlash2SpecDecoder& decoder,
    uint32_t position,
    uint32_t rows) {
    std::array<const bf16_t*, shelf::kernel::kDFlash2TargetTaps> taps{};
    collect_target_taps(*decoder.target, taps);
    return dflash2::dflash2_append_target_taps(
        *decoder.draft, *decoder.context, taps.data(), shelf::kernel::kDFlash2TargetTaps,
        rows, position, decoder.stream);
}

Status gather_tree_path(DFlash2SpecDecoder& decoder, uint32_t position,
                        std::span<const uint32_t> rows) {
    auto& pool = *decoder.sequence->kv_pool();
    const uint32_t page_tokens = pool.page_tokens();
    const auto& pages = decoder.sequence->block_table;
    const std::size_t heads = pool.kv_heads();
    const std::size_t width = heads * pool.head_dim();
    for (uint32_t dst_row = 0; dst_row < rows.size(); ++dst_row) {
        const uint32_t src_row = rows[dst_row];
        if (src_row == dst_row) continue;
        const uint32_t src_pos = position + src_row;
        const uint32_t dst_pos = position + dst_row;
        if (src_pos / page_tokens >= pages.size() || dst_pos / page_tokens >= pages.size())
            return Status::out_of_range("dflash2 tree KV page out of range", __FILE__, __LINE__);
        const std::size_t src_page = pages[src_pos / page_tokens];
        const std::size_t dst_page = pages[dst_pos / page_tokens];
        const std::size_t src_slot = src_page * page_tokens + src_pos % page_tokens;
        const std::size_t dst_slot = dst_page * page_tokens + dst_pos % page_tokens;
        const std::size_t slots = static_cast<std::size_t>(pool.num_pages()) * page_tokens;
        const auto copy = [&](void* dst, const void* src, std::size_t bytes) -> Status {
            const hipError_t err = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, decoder.stream);
            if (err != hipSuccess) return Status::hip_error("dflash2 tree gather", hipGetErrorString(err), __FILE__, __LINE__);
            return Status::make_ok();
        };
        for (uint32_t layer = 0; layer < pool.num_attention_layers(); ++layer) {
            const std::size_t src = layer * slots + src_slot;
            const std::size_t dst = layer * slots + dst_slot;
            if (pool.dtype() == shelf::kernel::KVCacheDType::BF16) {
                auto* k = pool.k_pool().data<bf16_t>();
                auto* v = pool.v_pool().data<bf16_t>();
                Status st = copy(k + dst * width, k + src * width, width * sizeof(bf16_t));
                if (!st.ok()) return st;
                st = copy(v + dst * width, v + src * width, width * sizeof(bf16_t));
                if (!st.ok()) return st;
            } else if (pool.dtype() == shelf::kernel::KVCacheDType::FP8_E4M3) {
                auto* k = pool.k_pool().data<shelf::kernel::fp8e4m3_storage_t>();
                auto* v = pool.v_pool().data<shelf::kernel::fp8e4m3_storage_t>();
                Status st = copy(k + dst * width, k + src * width, width * sizeof(*k));
                if (!st.ok()) return st;
                st = copy(v + dst * width, v + src * width, width * sizeof(*v));
                if (!st.ok()) return st;
                auto* ks = pool.k_scale_pool().data<float>();
                auto* vs = pool.v_scale_pool().data<float>();
                st = copy(ks + dst * heads, ks + src * heads, heads * sizeof(float));
                if (!st.ok()) return st;
                st = copy(vs + dst * heads, vs + src * heads, heads * sizeof(float));
                if (!st.ok()) return st;
            } else if (pool.dtype() == shelf::kernel::KVCacheDType::OCTAVE) {
                const auto view = shelf::state::make_paged_kv_octave_view(pool);
                const std::size_t kb = view.k_bytes_per_token;
                const std::size_t vb = view.v_bytes_per_token;
                Status st = copy(view.k_pool + dst * kb, view.k_pool + src * kb, kb);
                if (!st.ok()) return st;
                st = copy(view.v_pool + dst * vb, view.v_pool + src * vb, vb);
                if (!st.ok()) return st;
            } else {
                return Status::unsupported("dflash2 tree KV dtype", __FILE__, __LINE__);
            }
        }
        for (uint32_t t = 0; t < shelf::kernel::kDFlash2TargetTaps; ++t) {
            auto* tap = decoder.target->dflash_target_hidden[t].data<bf16_t>();
            Status st = copy(tap + static_cast<std::size_t>(dst_row) * decoder.hidden_size,
                             tap + static_cast<std::size_t>(src_row) * decoder.hidden_size,
                             decoder.hidden_size * sizeof(bf16_t));
            if (!st.ok()) return st;
        }
    }
    return Status::make_ok();
}

Status run_single_target(
    DFlash2SpecDecoder& decoder,
    int32_t token,
    DFlash2SpecIterationOutput& out,
    int32_t forced_token = -1) {
    DFlash2SpecTiming* tm = decoder.timing;
    const uint32_t position = decoder.sequence->position;
    std::vector<ScheduledRequest> requests;
    int32_t single = token;
    ScheduledBatch batch = make_verify_batch(
        *decoder.sequence, requests, &single, 1u, position, 1u, false,
        TokenIdsLocation::Host, &decoder.sampling, decoder.sample_index);
    const std::vector<ExecuteBatchOptions> soptions = build_gdn_options(decoder, 1u);
    auto executed = run_target_batch(decoder, batch, soptions);
    if (!executed.ok()) return executed.status();
    BatchExecutionOutput output = executed.release();
    int32_t sampled = -1;
    const hipError_t err =
        hipMemcpy(&sampled, output.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                  hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 single decode sample copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (sampled < 0) {
        return Status::invalid_state("sampled token is invalid", __FILE__, __LINE__);
    }
    decoder.sample_index += 1u;
    std::array<const bf16_t*, shelf::kernel::kDFlash2TargetTaps> taps{};
    collect_target_taps(*decoder.target, taps);
    Status st = dflash2::dflash2_append_target_taps(
        *decoder.draft, *decoder.context, taps.data(), shelf::kernel::kDFlash2TargetTaps, 1u,
        position, decoder.stream);
    if (!st.ok()) return st;
    out.emitted[0] = forced_token >= 0 ? forced_token : sampled;
    out.emitted_count = 1u;
    out.pending_token = out.emitted[0];
    if (tm != nullptr) ++tm->generated_tokens;
    commit_history_tokens(decoder, out.emitted.data(), 1u);
    return Status::make_ok();
}

}  // namespace

static Result<DFlash2SpecIterationOutput> dflash2_spec_step_impl(
    DFlash2SpecDecoder& decoder,
    int32_t pending_token,
    uint32_t remaining_tokens) {
    if (!decoder.initialized) {
        return Status::invalid_state("dflash2_spec_step: not initialized", __FILE__, __LINE__);
    }
    if (remaining_tokens == 0u) {
        return Status::invalid_argument("dflash2_spec_step: remaining_tokens is zero", __FILE__,
                                        __LINE__);
    }
    if (qwen_hybrid::is_stop_token(decoder.config.eos_tokens, pending_token)) {
        // The stop token was already emitted by the caller; do not draft past it.
        DFlash2SpecIterationOutput stopped;
        stopped.pending_token = pending_token;
        stopped.finished = true;
        return stopped;
    }
    if (decoder.target == nullptr || decoder.sequence == nullptr || decoder.gdn_ranks.empty() ||
        decoder.draft == nullptr || decoder.context == nullptr) {
        return Status::invalid_state("dflash2_spec_step: state pointers are null", __FILE__,
                                     __LINE__);
    }

    DFlash2SpecIterationOutput out;
    DFlash2SpecTiming* tm = decoder.timing;
    ScopedTimer round_timer(tm != nullptr ? &tm->round_ms : nullptr);
    if (tm != nullptr) ++tm->rounds;

    const DFlash2TargetSnapshot snapshot{
        decoder.sequence->position,
        static_cast<uint32_t>(decoder.sequence->block_table.size()), true};
    const uint32_t position = snapshot.position;
    const uint32_t room = decoder.sequence->max_seq_len - position;
    uint32_t room_max = 0u;
    if (remaining_tokens > 1u && room >= 2u) {
        room_max = std::min(remaining_tokens - 1u, room - 1u);
    }
    const uint32_t dflash_k = std::min(decoder.config.num_drafts, room_max);

    if (dflash_k == 0u) {
        Status st = run_single_target(decoder, pending_token, out);
        if (!st.ok()) return st;
        if (qwen_hybrid::is_stop_token(decoder.config.eos_tokens, out.pending_token)) out.finished = true;
        return out;
    }

    const bool bridge = !decoder.config.tree && decoder.device_token_bridge &&
                        decoder.verify_token_ids_device != nullptr &&
                        decoder.decision_staging_device != nullptr;
    const bool want_timing = tm != nullptr;
    int32_t* proposal_device = decoder.draft->proposal_tokens.data<int32_t>();
    std::array<int32_t, dflash2::kMaxBlockSize - 1u> drafts{};
    std::array<int32_t, kDFlash2SpecMaxVerifyRows> sampled{};
    std::array<int32_t, kDFlash2SpecMaxVerifyRows> verify_host{};
    std::array<int32_t, 2u * kDFlash2SpecMaxVerifyRows> decision_host{};
    bool draft_events_recorded = false;
    bool verify_events_recorded = false;

    {
        ScopedTimer t(want_timing ? &tm->draft_ms : nullptr);
        if (bridge) {
            const hipError_t anchor_err = hipMemcpyAsync(
                decoder.verify_token_ids_device, &pending_token, sizeof(int32_t),
                hipMemcpyHostToDevice, decoder.stream);
            if (anchor_err != hipSuccess) {
                return Status::hip_error("dflash2 anchor upload",
                                         hipGetErrorString(anchor_err), __FILE__, __LINE__);
            }
        }
        if (want_timing) {
            const hipError_t start_err =
                hipEventRecord(decoder.draft_start_event, decoder.stream);
            if (start_err != hipSuccess) {
                return Status::hip_error("dflash2 draft start event",
                                         hipGetErrorString(start_err), __FILE__, __LINE__);
            }
            draft_events_recorded = true;
        }
        Status st = dflash2::dflash2_propose_cached(
            *decoder.draft, *decoder.context, pending_token, dflash_k, proposal_device,
            decoder.stream);
        if (!st.ok()) return st;
        if (bridge) {
            const hipError_t relay_err = hipMemcpyAsync(
                decoder.verify_token_ids_device + 1, proposal_device,
                static_cast<std::size_t>(dflash_k) * sizeof(int32_t),
                hipMemcpyDeviceToDevice, decoder.stream);
            if (relay_err != hipSuccess) {
                return Status::hip_error("dflash2 draft relay",
                                         hipGetErrorString(relay_err), __FILE__, __LINE__);
            }
        }
        if (want_timing) {
            const hipError_t stop_err =
                hipEventRecord(decoder.draft_stop_event, decoder.stream);
            if (stop_err != hipSuccess) {
                return Status::hip_error("dflash2 draft stop event",
                                         hipGetErrorString(stop_err), __FILE__, __LINE__);
            }
        }
    }

    const int32_t* sampled_device_ptr = nullptr;
    const bool need_host_seed = decoder.host_proposal_visibility || decoder.config.tree;
    if (!bridge || need_host_seed) {
        ScopedTimer t(want_timing ? &tm->draft_d2h_ms : nullptr);
        {
            ScopedTimer tw(want_timing ? &tm->proposal_wait_ms : nullptr);
            if (hipStreamSynchronize(decoder.stream) != hipSuccess) {
                return Status::hip_error(
                    "dflash2 draft sync", hipGetErrorString(hipGetLastError()), __FILE__,
                    __LINE__);
            }
        }
        {
            ScopedTimer tc(want_timing ? &tm->proposal_copy_ms : nullptr);
            const hipError_t err = hipMemcpy(
                drafts.data(), proposal_device,
                static_cast<std::size_t>(dflash_k) * sizeof(int32_t),
                hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                return Status::hip_error("dflash2 draft copy", hipGetErrorString(err),
                                         __FILE__, __LINE__);
            }
        }
    }

    DFlash2Tree tree;
    uint32_t verify_rows = dflash_k + 1u;
    const bool tree_compact = decoder.config.tree &&
        env_flag_enabled("SHELF_DFLASH2_TREE_COMPACT_BRANCH", false);
    if (decoder.config.tree) {
        ScopedTimer t(want_timing ? &tm->tree_select_ms : nullptr);
        if (decoder.config.tree_budget != 0u) {
            const uint32_t top_k = static_cast<uint32_t>(decoder.draft->config->selector_top_k);
            const uint32_t vocab = static_cast<uint32_t>(decoder.draft->config->vocab_size);
            const uint32_t retained = std::min(top_k, shelf::kernel::kDflash2TreeRetained);
            using Probabilities = shelf::kernel::DFlash2TreeProbabilities;
            auto* device_probabilities = reinterpret_cast<Probabilities*>(decoder.gdn_ranks.front().tree_scores);
            std::vector<Probabilities> probabilities(dflash_k);
            hipError_t err = shelf::kernel::launch_dflash2_tree_probabilities(
                decoder.draft->proposal_logits.data<float>(), decoder.draft->proposal_topk_ids.data<int32_t>(),
                decoder.draft->proposal_topk_logits.data<float>(), device_probabilities,
                dflash_k, vocab, top_k, decoder.stream);
            if (err == hipSuccess) err = hipMemcpyAsync(probabilities.data(), device_probabilities,
                probabilities.size() * sizeof(Probabilities), hipMemcpyDeviceToHost, decoder.stream);
            if (err == hipSuccess) err = hipStreamSynchronize(decoder.stream);
            if (err != hipSuccess) return Status::hip_error("dflash2 budget probabilities copy", hipGetErrorString(err), __FILE__, __LINE__);
            std::vector<std::vector<DFlash2TreeCandidate>> choices(dflash_k);
            for (uint32_t i = 0; i < dflash_k; ++i) {
                for (uint32_t j = 0; j < retained; ++j) {
                    choices[i].push_back({probabilities[i].ids[j], probabilities[i].probabilities[j]});
                }
                if (tm != nullptr) { tm->tree_retained_mass += 1.0 - probabilities[i].residual; ++tm->tree_probability_rows; }
            }
            if (env_flag_enabled("SHELF_DFLASH2_TREE_TRACE_SHAPE", false)) {
                std::fprintf(stderr, "DFLASH2_TREE_SHAPE %u %d", position, pending_token);
                std::fprintf(stderr, " drafts");
                for (uint32_t i = 0; i < dflash_k; ++i) std::fprintf(stderr, " %d", drafts[i]);
                for (const auto& row : choices) {
                    std::fprintf(stderr, " ;");
                    for (const auto& candidate : row)
                        std::fprintf(stderr, " %d:%.9g", candidate.token, candidate.probability);
                }
                std::fprintf(stderr, "\n");
            }
            if (!make_budgeted_dflash2_tree(pending_token, std::min(decoder.config.tree_budget, room),
                    dflash_k, position, [&](uint32_t depth, int32_t) { return choices[depth]; }, tree))
                return Status::invalid_state("dflash2 budget tree selection failed", __FILE__, __LINE__);
            if (env_flag_enabled("SHELF_DFLASH2_TREE_TRACE_CHAIN", false)) {
                tree = make_dflash2_tree(pending_token,
                    std::span<const int32_t>(drafts.data(), dflash_k), {}, 0u, position);
            }
            if (env_flag_enabled("SHELF_DFLASH2_TREE_CHECK_CPU", false)) {
                std::vector<float> logits(static_cast<size_t>(dflash_k) * vocab);
                err = hipMemcpy(logits.data(), decoder.draft->proposal_logits.data<float>(),
                                logits.size() * sizeof(float), hipMemcpyDeviceToHost);
                if (err != hipSuccess) return Status::hip_error("dflash2 budget reference copy", hipGetErrorString(err), __FILE__, __LINE__);
                std::vector<std::vector<DFlash2TreeCandidate>> reference(dflash_k);
                for (uint32_t i = 0; i < dflash_k; ++i) {
                    const float* row = logits.data() + static_cast<size_t>(i) * vocab;
                    const float maximum = *std::max_element(row, row + vocab);
                    double sum = 0.0;
                    for (uint32_t j = 0; j < vocab; ++j) sum += std::exp(static_cast<double>(row[j] - maximum));
                    for (uint32_t j = 0; j < retained; ++j) {
                        const int32_t id = probabilities[i].ids[j];
                        reference[i].push_back({id, std::exp(static_cast<double>(row[id] - maximum)) / sum});
                    }
                }
                DFlash2Tree expected;
                if (!make_budgeted_dflash2_tree(pending_token, std::min(decoder.config.tree_budget, room),
                        dflash_k, position, [&](uint32_t depth, int32_t) { return reference[depth]; }, expected))
                    return Status::invalid_state("dflash2 budget reference selection failed", __FILE__, __LINE__);
                if (tree.tokens != expected.tokens || tree.parent != expected.parent)
                    std::fprintf(stderr, "DFLASH2_TREE_CPU_MISMATCH position=%u budget=%u\n", position, decoder.config.tree_budget);
            }
            verify_rows = static_cast<uint32_t>(tree.tokens.size());
        } else {
            std::vector<int32_t> alternatives(dflash_k, -1);
            const uint32_t extra = std::min({decoder.config.tree_extra, 2u,
                                             room - verify_rows, dflash_k});
            if (extra != 0u) {
                auto& g = decoder.gdn_ranks.front();
                const uint32_t top_k = static_cast<uint32_t>(decoder.draft->config->selector_top_k);
                if (top_k < 2u || top_k > shelf::kernel::kDflash2SelectorMaxTopK)
                    return Status::unsupported("dflash2 tree requires selector top-k between 2 and 16", __FILE__, __LINE__);
                std::vector<int32_t> predecessors(dflash_k);
                predecessors[0] = pending_token;
                for (uint32_t i = 1; i < dflash_k; ++i) predecessors[i] = drafts[i - 1u];
                hipError_t err = hipMemcpyAsync(g.tree_predecessors, predecessors.data(),
                    dflash_k * sizeof(int32_t), hipMemcpyHostToDevice, decoder.stream);
                if (err != hipSuccess) return Status::hip_error("dflash2 tree predecessor upload", hipGetErrorString(err), __FILE__, __LINE__);
                shelf::kernel::DFlash2CandidateSelectorArgs args;
                args.hidden_proj = decoder.draft->selector_hidden_proj.data<bf16_t>();
                args.topk_ids = decoder.draft->proposal_topk_ids.data<int32_t>();
                args.topk_logits = decoder.draft->proposal_topk_logits.data<float>();
                args.predecessor_codebook = decoder.draft->weights->selector.predecessor_codebook.data<bf16_t>();
                args.successor_codebook = decoder.draft->weights->selector.successor_codebook.data<bf16_t>();
                args.row_predecessors = g.tree_predecessors;
                args.output_tokens = g.tree_selected;
                args.output_scores = g.tree_scores;
                args.draft_rows = dflash_k;
                args.rank = static_cast<uint32_t>(decoder.draft->config->selector_rank);
                args.top_k = top_k;
                args.vocab_size = static_cast<uint32_t>(decoder.draft->config->vocab_size);
                err = shelf::kernel::launch_dflash2_candidate_selector(args, decoder.stream);
                if (err != hipSuccess) return Status::hip_error("dflash2 tree selector", hipGetErrorString(err), __FILE__, __LINE__);
                std::vector<int32_t> ids(dflash_k * top_k);
                std::vector<float> scores(dflash_k * top_k);
                err = hipMemcpyAsync(ids.data(), args.topk_ids, ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost, decoder.stream);
                if (err == hipSuccess) err = hipMemcpyAsync(scores.data(), g.tree_scores, scores.size() * sizeof(float), hipMemcpyDeviceToHost, decoder.stream);
                if (err == hipSuccess) err = hipStreamSynchronize(decoder.stream);
                if (err != hipSuccess) return Status::hip_error("dflash2 tree scores copy", hipGetErrorString(err), __FILE__, __LINE__);
                for (uint32_t i = 0; i < extra; ++i) {
                    float best = -std::numeric_limits<float>::infinity();
                    for (uint32_t j = 0; j < top_k; ++j) {
                        const uint32_t n = i * top_k + j;
                        if (ids[n] == drafts[i]) continue;
                        if (scores[n] > best || (scores[n] == best && (alternatives[i] < 0 || ids[n] < alternatives[i]))) {
                            alternatives[i] = ids[n];
                            best = scores[n];
                        }
                    }
                }
            }
            tree = make_dflash2_tree(pending_token,
                std::span<const int32_t>(drafts.data(), dflash_k), alternatives, extra, position);
            verify_rows = static_cast<uint32_t>(tree.tokens.size());
        }
    }

    bool chunk_geometry = decoder.gdn_ranks.size() == 1u && decoder.gdn_key_heads == 16u;
    if (chunk_geometry) {
        const auto view = decoder.gdn_ranks.front().pool->device_view();
        chunk_geometry = view.head_k == 128u && view.head_v == 128u &&
            (view.num_v_heads == 32u || view.num_v_heads == 48u);
    }
    const bool tree_chunk = chunk_geometry && decoder.config.tree && verify_rows <= 32u &&
        env_flag_enabled("SHELF_GDN_VERIFY_CHUNKED", false) &&
        env_flag_enabled("SHELF_GDN_TREE_CHUNKED", false);
    const bool chain_chunk = chunk_geometry && !decoder.config.tree && verify_rows <= 32u &&
        env_flag_enabled("SHELF_GDN_VERIFY_CHUNKED", false);

    out.num_dflash_drafts = dflash_k;
    const uint32_t total_k = dflash_k;
    out.num_drafts = total_k;


    {
        ScopedTimer t(tm != nullptr ? &tm->gdn_snapshot_ms : nullptr);
        for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
            auto scope = shelf::gpu::ScopedDevice::create(g.device);
            if (!scope.ok()) return scope.status();
            Status st = spec_gdn_snapshot(*g.pool, decoder.sequence->slot,
                                          g.conv_snapshot, g.rec_snapshot, g.conv_bytes,
                                          g.rec_bytes, g.stream);
            if (!st.ok()) return st;
        }
    }

    std::vector<ScheduledRequest> requests;
    const uint64_t round_sample_index = decoder.sample_index;
    decoder.sample_index += static_cast<uint64_t>(total_k) + 1u;
    static thread_local bool semantic_verify_recorded = false;
    const char* semantic_csv = std::getenv("SHELF_DFLASH2_VERIFY_TIMING_CSV");
    const bool semantic_verify = semantic_csv != nullptr && semantic_csv[0] != '\0' &&
        shelf::execution::semantic_timing_enabled() && !semantic_verify_recorded;
    if (semantic_verify) {
        shelf::execution::semantic_timing_begin(decoder.stream, 8192);
        semantic_verify_recorded = true;
    }
    {
        ScopedTimer t(want_timing ? &tm->verify_ms : nullptr);
        ScheduledBatch batch;
        if (bridge) {
            batch = make_verify_batch(
                *decoder.sequence, requests, decoder.verify_token_ids_device, verify_rows,
                position, verify_rows, true, TokenIdsLocation::Device,
                &decoder.sampling, round_sample_index);
        } else {
            verify_host[0] = pending_token;
            for (uint32_t i = 0; i < total_k; ++i) verify_host[i + 1u] = drafts[i];
            if (decoder.config.tree) std::copy(tree.tokens.begin(), tree.tokens.end(), verify_host.begin());
            batch = make_verify_batch(
                *decoder.sequence, requests, verify_host.data(), verify_rows, position,
                verify_rows, true, TokenIdsLocation::Host, &decoder.sampling,
                round_sample_index);
        }
        std::vector<ExecuteBatchOptions> options = build_gdn_options(
            decoder, decoder.config.tree ? verify_rows : total_k);
        if (chain_chunk) {
            for (auto& option : options) option.gdn_compact.defer_state_store = 1u;
        }
        if (decoder.config.tree) {
            for (std::size_t r = 0; r < options.size(); ++r) {
                auto& g = decoder.gdn_ranks[r];
                auto scope = shelf::gpu::ScopedDevice::create(g.device);
                if (!scope.ok()) return scope.status();
                const auto copy = [&](void* dst, const void* src, size_t bytes) {
                    return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
                };
                if (copy(g.tree_parent, tree.parent.data(), tree.parent.size() * sizeof(int32_t)) != hipSuccess ||
                    copy(g.tree_offsets, tree.ancestor_offsets.data(), tree.ancestor_offsets.size() * sizeof(uint32_t)) != hipSuccess ||
                    copy(g.tree_ancestors, tree.ancestor_positions.data(), tree.ancestor_positions.size() * sizeof(uint32_t)) != hipSuccess ||
                    copy(g.tree_rope, tree.rope_positions.data(), tree.rope_positions.size() * sizeof(uint32_t)) != hipSuccess) {
                    return Status::hip_error("dflash2 tree metadata upload", hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
                }
                options[r].tree_row_parent = g.tree_parent;
                options[r].tree_anc_offsets = g.tree_offsets;
                options[r].tree_anc_positions = g.tree_ancestors;
                options[r].rope_positions = g.tree_rope;
                options[r].tree_committed_end = position;
                if (tree_chunk) {
                    options[r].gdn_spec_history.recurrent_history = nullptr;
                    options[r].gdn_compact.defer_state_store = 1u;
                    continue;
                }
                if (!tree_compact) {
                    options[r].gdn_compact = {};
                    continue;
                }
                // A non-adjacent child reloads its parent's state. Adjacent
                // children continue from registers and need no snapshot.
                std::vector<uint32_t> capture(verify_rows, 0u);
                for (uint32_t row = 1; row < verify_rows; ++row) {
                    const int32_t parent = tree.parent[row];
                    if (parent >= 0 && parent != static_cast<int32_t>(row) - 1)
                        capture[static_cast<uint32_t>(parent)] = 1u;
                }
                if (copy(g.tree_selected, capture.data(), capture.size() * sizeof(uint32_t)) != hipSuccess)
                    return Status::hip_error("dflash2 tree snapshot mask upload", hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
                options[r].gdn_spec_history.recurrent_capture_mask = reinterpret_cast<const uint32_t*>(g.tree_selected);
            }
        }
        if (want_timing) {
            const hipError_t start_err =
                hipEventRecord(decoder.verify_start_event, decoder.stream);
            if (start_err != hipSuccess) {
                return Status::hip_error("dflash2 verify start event",
                                         hipGetErrorString(start_err), __FILE__, __LINE__);
            }
            verify_events_recorded = true;
        }
        if (decoder.tp_hook != nullptr) {
            auto executed = decoder.tp_hook->on_execute(batch, options.data());
            if (!executed.ok()) return executed.status();
            BatchExecutionOutput hook_output = executed.release();
            if (want_timing) {
                const hipError_t stop_err =
                    hipEventRecord(decoder.verify_stop_event, decoder.stream);
                if (stop_err != hipSuccess) {
                    return Status::hip_error("dflash2 verify stop event",
                                             hipGetErrorString(stop_err), __FILE__,
                                             __LINE__);
                }
            }
            sampled_device_ptr = hook_output.sampled_tokens.data<int32_t>();
            {
                ScopedTimer tc(want_timing ? &tm->decision_copy_ms : nullptr);
                const hipError_t err = hipMemcpy(
                    sampled.data(), hook_output.sampled_tokens.data<int32_t>(),
                    static_cast<std::size_t>(verify_rows) * sizeof(int32_t),
                    hipMemcpyDeviceToHost);
                if (err != hipSuccess) {
                    return Status::hip_error("dflash2 verify sample copy",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
        } else {
            auto submitted = submit_batch(*decoder.target, batch, decoder.stream, options[0]);
            if (!submitted.ok()) return submitted.status();
            PendingBatch pending = submitted.release();
            if (want_timing) {
                const hipError_t stop_err =
                    hipEventRecord(decoder.verify_stop_event, decoder.stream);
                if (stop_err != hipSuccess) {
                    return Status::hip_error("dflash2 verify stop event",
                                             hipGetErrorString(stop_err), __FILE__, __LINE__);
                }
            }
        sampled_device_ptr = pending.output.sampled_tokens.data<int32_t>();
        if (bridge) {
            const int32_t* sampled_device = sampled_device_ptr;
            hipError_t d_err = hipMemcpyAsync(
                decoder.decision_staging_device, proposal_device,
                static_cast<std::size_t>(dflash_k) * sizeof(int32_t),
                hipMemcpyDeviceToDevice, decoder.stream);
            if (d_err != hipSuccess) {
                return Status::hip_error("dflash2 decision draft relay",
                                         hipGetErrorString(d_err), __FILE__, __LINE__);
            }
            d_err = hipMemcpyAsync(
                decoder.decision_staging_device + kDFlash2SpecMaxVerifyRows,
                sampled_device, static_cast<std::size_t>(verify_rows) * sizeof(int32_t),
                hipMemcpyDeviceToDevice, decoder.stream);
            if (d_err != hipSuccess) {
                return Status::hip_error("dflash2 decision sample relay",
                                         hipGetErrorString(d_err), __FILE__, __LINE__);
            }
            {
                ScopedTimer tw(want_timing ? &tm->decision_wait_ms : nullptr);
                if (hipStreamSynchronize(decoder.stream) != hipSuccess) {
                    return Status::hip_error(
                        "dflash2 decision sync", hipGetErrorString(hipGetLastError()),
                        __FILE__, __LINE__);
                }
            }
            {
                ScopedTimer tc(want_timing ? &tm->decision_copy_ms : nullptr);
                const hipError_t err = hipMemcpy(
                    decision_host.data(), decoder.decision_staging_device,
                    decision_host.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
                if (err != hipSuccess) {
                    return Status::hip_error("dflash2 decision copy",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
            for (uint32_t i = 0; i < dflash_k; ++i) drafts[i] = decision_host[i];
            for (uint32_t i = 0; i < verify_rows; ++i) {
                sampled[i] = decision_host[kDFlash2SpecMaxVerifyRows + i];
            }
            auto completed =
                complete_batch(*decoder.target, std::move(pending), decoder.stream);
            if (!completed.ok()) return completed.status();
        } else {
            BatchExecutionOutput output;
            {
                ScopedTimer tw(want_timing ? &tm->decision_wait_ms : nullptr);
                auto completed =
                    complete_batch(*decoder.target, std::move(pending), decoder.stream);
                if (!completed.ok()) return completed.status();
                output = completed.release();
            }
            {
                ScopedTimer tc(want_timing ? &tm->decision_copy_ms : nullptr);
                const hipError_t err = hipMemcpy(
                    sampled.data(), output.sampled_tokens.data<int32_t>(),
                    static_cast<std::size_t>(verify_rows) * sizeof(int32_t),
                    hipMemcpyDeviceToHost);
                if (err != hipSuccess) {
                    return Status::hip_error("dflash2 verify sample copy",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
        }
        }
    }

    if (semantic_verify) shelf::execution::semantic_timing_end(decoder.stream, semantic_csv);

    if (want_timing && draft_events_recorded) {
        float elapsed = 0.0f;
        if (hipEventElapsedTime(&elapsed, decoder.draft_start_event,
                                decoder.draft_stop_event) == hipSuccess) {
            tm->draft_gpu_ms += static_cast<double>(elapsed);
        }
    }
    if (want_timing && verify_events_recorded) {
        float elapsed = 0.0f;
        if (hipEventElapsedTime(&elapsed, decoder.verify_start_event,
                                decoder.verify_stop_event) == hipSuccess) {
            tm->verify_gpu_ms += static_cast<double>(elapsed);
        }
    }

    SpecVerifyResult result;
    if (sampled[0] < 0) {
        return Status::invalid_state("verify produced no sampled token", __FILE__,
                                     __LINE__);
    }
    DFlash2TreeWalk walk;
    if (decoder.config.tree) {
        {
            ScopedTimer t(want_timing ? &tm->tree_walk_ms : nullptr);
            if (!walk_dflash2_tree(tree.tokens, tree.parent,
                                   std::span<const int32_t>(sampled.data(), verify_rows), walk)) {
                return Status::invalid_state("dflash2 tree walk failed", __FILE__, __LINE__);
            }
        }
        const uint32_t accepted = static_cast<uint32_t>(walk.rows.size() - 1u);
        result.num_drafts = total_k;
        result.num_accepted_drafts = accepted;
        for (uint32_t i = 1u; i < walk.rows.size(); ++i)
            result.emitted_tokens.push_back(tree.tokens[walk.rows[i]]);
        if (accepted == total_k) {
            result.bonus_token_valid = true;
            result.bonus_token = walk.correction;
        } else {
            result.first_reject_index = accepted;
            result.correction_valid = true;
            result.correction_token = walk.correction;
        }
        result.emitted_tokens.push_back(walk.correction);
        result.num_emitted_tokens = static_cast<uint32_t>(result.emitted_tokens.size());
        result.num_committed_target_tokens = accepted + 1u;
    } else {
        Status accept = spec_greedy_accept(drafts.data(), total_k, sampled.data(), true, result);
        if (!accept.ok()) return accept;
    }
    out.finished = spec_bound_result(result, remaining_tokens, decoder.config.eos_tokens);
    const uint32_t accepted = result.num_accepted_drafts;
    out.num_accepted = accepted;
    out.emitted_count = std::min(static_cast<uint32_t>(result.emitted_tokens.size()),
                                 remaining_tokens);
    for (uint32_t i = 0; i < out.emitted_count; ++i) {
        out.emitted[i] = result.emitted_tokens[i];
    }
    if (result.correction_valid) {
        out.pending_token = result.correction_token;
    } else if (result.bonus_token_valid) {
        out.pending_token = result.bonus_token;
    } else {
        out.pending_token = -1;
    }
    const uint32_t page_tokens = decoder.sequence->kv_pool()->page_tokens();
    if (page_tokens == 0u) {
        return Status::invalid_state("dflash2_spec_step: page_tokens is zero", __FILE__,
                                     __LINE__);
    }

    if (decoder.config.tree) {
        ScopedTimer t(want_timing ? &tm->tree_commit_ms : nullptr);
        const bool restore_needed = walk.rows[accepted] != verify_rows - 1u;
        const bool profile_commit = want_timing && decoder.gdn_ranks.size() == 1u &&
            env_flag_enabled("SHELF_DFLASH2_TREE_PROFILE_COMMIT", false);
        if (profile_commit && hipEventRecord(decoder.draft_start_event, decoder.stream) != hipSuccess)
            return Status::hip_error("dflash2 restore event", hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
        {
            ScopedTimer restore_timer(want_timing ? &tm->tree_restore_ms : nullptr);
            if (tree_chunk) {
                for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
                    auto scope = shelf::gpu::ScopedDevice::create(g.device);
                    if (!scope.ok()) return scope.status();
                    Status st = restore_gdn_spec_conv(*g.pool, decoder.sequence->slot, g.history,
                                                      walk.rows[accepted], g.stream);
                    if (!st.ok()) return st;
                    std::vector<int32_t> replay_rows;
                    for (uint32_t i = 0u; i <= accepted; ++i)
                        replay_rows.push_back(static_cast<int32_t>(walk.rows[i]));
                    const hipError_t err = hipMemcpy(g.tree_selected, replay_rows.data(),
                        replay_rows.size() * sizeof(int32_t), hipMemcpyHostToDevice);
                    if (err != hipSuccess)
                        return Status::hip_error("dflash2 tree replay rows upload", hipGetErrorString(err), __FILE__, __LINE__);
                    GdnCompactLogDeviceView compact{};
                    compact.delta = g.compact_delta;
                    compact.k = g.compact_k;
                    compact.a = g.compact_a;
                    compact.delta_layer_stride = g.compact_delta_layer_stride;
                    compact.k_layer_stride = g.compact_k_layer_stride;
                    compact.a_layer_stride = g.compact_a_layer_stride;
                    st = commit_gdn_compact_log(*g.pool, decoder.sequence->slot, compact,
                        g.commit_request, static_cast<uint32_t>(replay_rows.size()),
                        decoder.gdn_key_heads, g.stream, g.tree_selected, verify_rows);
                    if (!st.ok()) return st;
                }
            } else if (restore_needed && !tree_compact) {
                for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
                    auto scope = shelf::gpu::ScopedDevice::create(g.device);
                    if (!scope.ok()) return scope.status();
                    Status st = restore_gdn_spec_history(*g.pool, decoder.sequence->slot,
                                                         g.history, walk.rows[accepted], g.stream);
                    if (!st.ok()) return st;
                }
            } else if (restore_needed) {
                // The last non-adjacent edge has a captured parent. Replay
                // only the suffix after it, or the whole path from the round
                // snapshot when the path has no such edge.
                uint32_t suffix = 0u;
                for (uint32_t i = 1u; i <= accepted; ++i)
                    if (walk.rows[i] != walk.rows[i - 1u] + 1u) suffix = i;
                for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
                    auto scope = shelf::gpu::ScopedDevice::create(g.device);
                    if (!scope.ok()) return scope.status();
                    Status st = suffix != 0u
                        ? restore_gdn_spec_history(*g.pool, decoder.sequence->slot, g.history,
                                                   walk.rows[suffix - 1u], g.stream)
                        : spec_gdn_restore(*g.pool, decoder.sequence->slot, g.conv_snapshot,
                                           g.rec_snapshot, g.conv_bytes, g.rec_bytes, g.stream);
                    if (!st.ok()) return st;
                    st = restore_gdn_spec_conv(*g.pool, decoder.sequence->slot, g.history,
                                               walk.rows[accepted], g.stream);
                    if (!st.ok()) return st;
                    std::vector<int32_t> replay_rows;
                    for (uint32_t i = suffix; i <= accepted; ++i)
                        replay_rows.push_back(static_cast<int32_t>(walk.rows[i]));
                    const hipError_t err = hipMemcpy(g.tree_selected, replay_rows.data(),
                        replay_rows.size() * sizeof(int32_t), hipMemcpyHostToDevice);
                    if (err != hipSuccess)
                        return Status::hip_error("dflash2 tree replay rows upload", hipGetErrorString(err), __FILE__, __LINE__);
                    GdnCompactLogDeviceView compact{};
                    compact.delta = g.compact_delta;
                    compact.k = g.compact_k;
                    compact.a = g.compact_a;
                    compact.delta_layer_stride = g.compact_delta_layer_stride;
                    compact.k_layer_stride = g.compact_k_layer_stride;
                    compact.a_layer_stride = g.compact_a_layer_stride;
                    st = commit_gdn_compact_log(*g.pool, decoder.sequence->slot, compact,
                        g.commit_request, static_cast<uint32_t>(replay_rows.size()),
                        decoder.gdn_key_heads, g.stream, g.tree_selected, verify_rows);
                    if (!st.ok()) return st;
                }
            }
        }
        if (profile_commit && hipEventRecord(decoder.draft_stop_event, decoder.stream) != hipSuccess)
            return Status::hip_error("dflash2 restore event", hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
        if (decoder.config.tree_extra != 0u || decoder.config.tree_budget != 0u) {
            if (profile_commit && hipEventRecord(decoder.verify_start_event, decoder.stream) != hipSuccess)
                return Status::hip_error("dflash2 gather event", hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
            ScopedTimer gather_timer(want_timing ? &tm->tree_gather_ms : nullptr);
            Status st = gather_tree_path(decoder, position,
                std::span<const uint32_t>(walk.rows.data(), accepted + 1u));
            if (!st.ok()) return st;
            if (profile_commit) {
                hipError_t err = hipEventRecord(decoder.verify_stop_event, decoder.stream);
                if (err == hipSuccess) err = hipEventSynchronize(decoder.verify_stop_event);
                float restore_ms = 0.0f, gather_ms = 0.0f;
                if (err == hipSuccess) err = hipEventElapsedTime(&restore_ms, decoder.draft_start_event, decoder.draft_stop_event);
                if (err == hipSuccess) err = hipEventElapsedTime(&gather_ms, decoder.verify_start_event, decoder.verify_stop_event);
                if (err != hipSuccess) return Status::hip_error("dflash2 commit timing", hipGetErrorString(err), __FILE__, __LINE__);
                std::fprintf(stderr, "DFLASH2_TREE_COMMIT_GPU restore_ms=%.6f gather_ms=%.6f restore_skipped=%u\n", restore_ms, gather_ms, restore_needed || tree_chunk ? 0u : 1u);
            }
        }
        ScopedTimer append_timer(want_timing ? &tm->tree_append_ms : nullptr);
        decoder.sequence->position = position + accepted + 1u;
        Status st = append_committed(decoder, position, accepted + 1u);
        if (!st.ok()) return st;
        st = rollback_sequence_append(*decoder.sequence,
            ceil_div_u32(decoder.sequence->position, page_tokens), decoder.stream);
        if (!st.ok()) return st;
        if (tm != nullptr) {
            if (accepted == total_k) ++tm->full_accepts;
            else ++tm->partial_accepts;
        }
    } else if (accepted == total_k) {
        ScopedTimer t(tm != nullptr ? &tm->dflash_commit_ms : nullptr);
        if (chain_chunk) {
            for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
                auto scope = shelf::gpu::ScopedDevice::create(g.device);
                if (!scope.ok()) return scope.status();
                GdnCompactLogDeviceView compact{};
                compact.delta = g.compact_delta;
                compact.k = g.compact_k;
                compact.a = g.compact_a;
                compact.delta_layer_stride = g.compact_delta_layer_stride;
                compact.k_layer_stride = g.compact_k_layer_stride;
                compact.a_layer_stride = g.compact_a_layer_stride;
                Status st = commit_gdn_compact_log(*g.pool, decoder.sequence->slot, compact,
                    g.commit_request, total_k + 1u, decoder.gdn_key_heads, g.stream);
                if (!st.ok()) return st;
            }
        }
        Status st = append_committed(decoder, position, total_k + 1u);
        if (!st.ok()) return st;
        const uint32_t required = ceil_div_u32(decoder.sequence->position, page_tokens);
        st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
        if (!st.ok()) return st;
        if (tm != nullptr) ++tm->full_accepts;
    } else {
        {
            ScopedTimer t(tm != nullptr ? &tm->gdn_restore_ms : nullptr);
            for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
                auto scope = shelf::gpu::ScopedDevice::create(g.device);
                if (!scope.ok()) return scope.status();
                Status st = spec_gdn_restore(*g.pool, decoder.sequence->slot,
                                             g.conv_snapshot, g.rec_snapshot, g.conv_bytes,
                                             g.rec_bytes, g.stream);
                if (!st.ok()) return st;
                st = restore_gdn_spec_conv(*g.pool, decoder.sequence->slot, g.history,
                                           accepted, g.stream);
                if (!st.ok()) return st;
            }
        }
        {
            ScopedTimer t(tm != nullptr ? &tm->rerun_ms : nullptr);
            for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
                auto scope = shelf::gpu::ScopedDevice::create(g.device);
                if (!scope.ok()) return scope.status();
                GdnCompactLogDeviceView compact{};
                compact.delta = g.compact_delta;
                compact.k = g.compact_k;
                compact.a = g.compact_a;
                compact.delta_layer_stride = g.compact_delta_layer_stride;
                compact.k_layer_stride = g.compact_k_layer_stride;
                compact.a_layer_stride = g.compact_a_layer_stride;
                Status st = commit_gdn_compact_log(
                    *g.pool, decoder.sequence->slot, compact, g.commit_request,
                    accepted + 1u, decoder.gdn_key_heads, g.stream);
                if (!st.ok()) return st;
            }
        }
        decoder.sequence->position = position + accepted + 1u;
        {
            ScopedTimer t(tm != nullptr ? &tm->dflash_commit_ms : nullptr);
            Status st = append_committed(decoder, position, accepted + 1u);
            if (!st.ok()) return st;
            const uint32_t required = ceil_div_u32(decoder.sequence->position, page_tokens);
            st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
            if (!st.ok()) return st;
        }
        if (tm != nullptr) ++tm->partial_accepts;
    }

    if (tm != nullptr) {
        tm->accepted_drafts += accepted;
        tm->generated_tokens += out.emitted_count;
        tm->verify_rows_total += verify_rows;
        if (accepted >= dflash_k) ++tm->dflash_prefix_full_accepts;
        tm->gdn_history_bytes = 0u;
        for (const DFlash2GdnRankState& g : decoder.gdn_ranks) {
            if (!g.history.initialized) continue;
            tm->gdn_history_bytes +=
                static_cast<uint64_t>(g.history.rows) *
                static_cast<uint64_t>(g.history.conv_state_bytes +
                                      g.history.recurrent_state_bytes);
        }
    }

    for (uint32_t i = 0; i < out.emitted_count; ++i) {
        if (qwen_hybrid::is_stop_token(decoder.config.eos_tokens, out.emitted[i])) {
            out.emitted_count = i + 1u;
            out.finished = true;
            break;
        }
    }
    if (qwen_hybrid::is_stop_token(decoder.config.eos_tokens, out.pending_token)) out.finished = true;
    commit_history_tokens(decoder, out.emitted.data(), out.emitted_count);
    return out;
}

Result<DFlash2SpecIterationOutput> dflash2_spec_step(
    DFlash2SpecDecoder& decoder, int32_t pending_token, uint32_t remaining_tokens) {
    if (decoder.config.tree) return Status::invalid_state("chain step with tree workspace", __FILE__, __LINE__);
    return dflash2_spec_step_impl(decoder, pending_token, remaining_tokens);
}

Result<DFlash2SpecIterationOutput> dflash2_spec_step_tree(
    DFlash2SpecDecoder& decoder, int32_t pending_token, uint32_t remaining_tokens) {
    if (!decoder.config.tree) return Status::invalid_state("tree step without tree workspace", __FILE__, __LINE__);
    return dflash2_spec_step_impl(decoder, pending_token, remaining_tokens);
}

namespace {

std::vector<ExecuteBatchOptions> build_batch_gdn_options(
    DFlash2SpecBatchContext& ctx,
    uint32_t capture_rows) {
    std::vector<ExecuteBatchOptions> options(ctx.history.size());
    for (std::size_t r = 0; r < options.size(); ++r) {
        options[r].gdn_spec_history = gdn_spec_history_view(ctx.history[r], capture_rows);
        options[r].gdn_compact.delta = ctx.compact[r].delta;
        options[r].gdn_compact.k = ctx.compact[r].k;
        options[r].gdn_compact.a = ctx.compact[r].a;
        options[r].gdn_compact.delta_layer_stride = ctx.compact[r].delta_layer_stride;
        options[r].gdn_compact.k_layer_stride = ctx.compact[r].k_layer_stride;
        options[r].gdn_compact.a_layer_stride = ctx.compact[r].a_layer_stride;
        // Verify keeps the live recurrent state at the base; the batch commit below
        // replays the accepted prefix from the compact log.
        options[r].gdn_compact.defer_state_store = 1u;
    }
    return options;
}

std::array<const bf16_t*, shelf::kernel::kDFlash2TargetTaps> collect_batch_target_taps(
    const DFlash2SpecDecoder& decoder) {
    std::array<const bf16_t*, shelf::kernel::kDFlash2TargetTaps> taps{};
    for (uint32_t t = 0; t < shelf::kernel::kDFlash2TargetTaps; ++t) {
        taps[t] = decoder.target->dflash_target_hidden[t].data<bf16_t>();
    }
    return taps;
}

std::array<const bf16_t*, shelf::kernel::kDFlash2TargetTaps> batch_target_taps(
    const DFlash2SpecDecoder& decoder,
    uint32_t row_begin) {
    std::array<const bf16_t*, shelf::kernel::kDFlash2TargetTaps> taps =
        collect_batch_target_taps(decoder);
    for (uint32_t t = 0; t < shelf::kernel::kDFlash2TargetTaps; ++t) {
        taps[t] += static_cast<std::size_t>(row_begin) * decoder.hidden_size;
    }
    return taps;
}


}  // namespace


Result<DFlash2SpecBatchOutput> dflash2_spec_step_batch(
    DFlash2SpecBatchContext& context,
    std::span<const DFlash2SpecBatchSlot> slots) {
    if (!context.initialized) {
        return Status::invalid_state("batch context is not initialized", __FILE__, __LINE__);
    }
    if (slots.empty()) {
        return Status::invalid_argument("batch step requires a slot", __FILE__, __LINE__);
    }

    struct SlotPlan {
        DFlash2SpecDecoder* decoder = nullptr;
        int32_t anchor = 0;
        uint32_t drafts = 0u;
        uint32_t remaining = 0u;
        uint32_t position = 0u;
        uint32_t row_begin = 0u;
        uint32_t total_rows = 0u;
        uint64_t sample_index = 0u;
        uint32_t thinking_allowance = kDFlash2NoThinkingAllowance;
        int32_t thinking_close_token = -1;
        bool batched = false;
        bool stopped = false;
    };

    std::vector<SlotPlan> plans;
    plans.reserve(slots.size());
    uint32_t row_cursor = 0u;
    for (const DFlash2SpecBatchSlot& slot : slots) {
        DFlash2SpecDecoder* decoder = slot.decoder;
        if (decoder == nullptr || !decoder->initialized || decoder->sequence == nullptr ||
            decoder->target == nullptr || decoder->draft == nullptr ||
            decoder->context == nullptr || decoder->gdn_ranks.empty()) {
            return Status::invalid_argument("batch slot has an invalid decoder", __FILE__,
                                            __LINE__);
        }
        const uint32_t position = decoder->sequence->position;
        const uint32_t room =
            decoder->sequence->max_seq_len > position
                ? decoder->sequence->max_seq_len - position
                : 0u;
        uint32_t room_max = 0u;
        if (slot.remaining_tokens > 1u && room >= 2u) {
            room_max = std::min(slot.remaining_tokens - 1u, room - 1u);
        }
        const uint32_t drafts_limit =
            slot.thinking_allowance != kDFlash2NoThinkingAllowance
                ? std::min(slot.num_drafts, slot.thinking_allowance)
                : slot.num_drafts;
        uint32_t drafts = std::min(drafts_limit, room_max);
        if (slot.page_tokens != 0u && drafts > 0u) {
            const uint32_t to_boundary =
                slot.page_tokens - (position % slot.page_tokens);
            if (drafts + 1u > to_boundary) {
                drafts = to_boundary - 1u;
            }
        }
        SlotPlan plan;
        plan.decoder = decoder;
        plan.anchor = slot.anchor_token;
        plan.drafts = drafts;
        plan.remaining = slot.remaining_tokens;
        plan.position = position;
        plan.sample_index = decoder->sample_index;
        plan.thinking_allowance = slot.thinking_allowance;
        plan.thinking_close_token = slot.thinking_close_token;
        plan.batched = drafts > 0u;
        if (qwen_hybrid::is_stop_token(decoder->config.eos_tokens, slot.anchor_token)) {
            // The stop token was already emitted by the caller; this slot runs nothing.
            plan.stopped = true;
            plan.batched = false;
            plan.drafts = 0u;
        }
        plan.total_rows = drafts + 1u;
        if (plan.batched) {
            plan.row_begin = row_cursor;
            row_cursor += plan.total_rows;
        }
        plans.push_back(plan);
    }

    const uint32_t total_rows = row_cursor;
    if (total_rows > context.max_rows) {
        return Status::invalid_argument("batch rows exceed context capacity", __FILE__,
                                        __LINE__);
    }
    if (total_rows > kDFlash2SpecMaxVerifyRows) {
        return Status::invalid_argument("batch rows exceed verify row cap", __FILE__, __LINE__);
    }

    std::vector<std::array<int32_t, dflash2::kMaxBlockSize - 1u>> drafts(plans.size());
    bool batch_timing = false;
    for (const SlotPlan& plan : plans) {
        if (plan.decoder->timing != nullptr) {
            batch_timing = true;
            break;
        }
    }
    bool bridge = context.verify_token_ids_device != nullptr &&
                  context.decision_staging_device != nullptr;
    if (bridge) {
        for (const SlotPlan& plan : plans) {
            if (!plan.batched) continue;
            if (!plan.decoder->device_token_bridge) {
                bridge = false;
                break;
            }
        }
    }
    const auto round_t0 = std::chrono::steady_clock::now();
    bool verify_events_recorded = false;
    std::vector<dflash2::DFlash2ProposeRequest> propose_requests;
    std::vector<int32_t*> propose_outputs;
    dflash2::DFlash2Executor* propose_executor = nullptr;
    DFlash2SpecDecoder* propose_leader = nullptr;
    hipStream_t propose_stream = nullptr;
    for (std::size_t i = 0; i < plans.size(); ++i) {
        SlotPlan& plan = plans[i];
        if (!plan.batched) continue;
        DFlash2SpecDecoder& decoder = *plan.decoder;
        if (propose_executor == nullptr) {
            propose_executor = decoder.draft;
            propose_leader = &decoder;
            propose_stream = decoder.stream;
        } else if (decoder.draft != propose_executor || decoder.stream != propose_stream) {
            return Status::unsupported(
                "batch proposal requires a shared drafter executor and stream", __FILE__,
                __LINE__);
        }
        dflash2::DFlash2ProposeRequest request;
        request.context = decoder.context;
        request.anchor_token = plan.anchor;
        request.num_drafts = plan.drafts;
        propose_requests.push_back(request);
        propose_outputs.push_back(context.verify_token_ids_device + plan.row_begin + 1u);
        if (bridge) {
            const hipError_t anchor_err = hipMemcpyAsync(
                context.verify_token_ids_device + plan.row_begin, &plan.anchor,
                sizeof(int32_t), hipMemcpyHostToDevice, decoder.stream);
            if (anchor_err != hipSuccess) {
                return Status::hip_error("batch anchor upload",
                                         hipGetErrorString(anchor_err), __FILE__, __LINE__);
            }
        }
    }
    if (propose_executor != nullptr) {
        if (batch_timing) {
            const hipError_t start_err =
                hipEventRecord(propose_leader->draft_start_event, propose_stream);
            if (start_err != hipSuccess) {
                return Status::hip_error("batch draft start event",
                                         hipGetErrorString(start_err), __FILE__, __LINE__);
            }
        }
        const Status st = dflash2::dflash2_propose_batch(
            *propose_executor, propose_requests.data(), propose_outputs.data(),
            static_cast<uint32_t>(propose_requests.size()), propose_stream);
        if (!st.ok()) return st;
        if (batch_timing) {
            const hipError_t stop_err =
                hipEventRecord(propose_leader->draft_stop_event, propose_stream);
            if (stop_err != hipSuccess) {
                return Status::hip_error("batch draft stop event",
                                         hipGetErrorString(stop_err), __FILE__, __LINE__);
            }
        }
    }
    if (!bridge) {
        for (std::size_t i = 0; i < plans.size(); ++i) {
            SlotPlan& plan = plans[i];
            if (!plan.batched) continue;
            const hipError_t err = hipMemcpy(
                drafts[i].data(), context.verify_token_ids_device + plan.row_begin + 1u,
                static_cast<std::size_t>(plan.drafts) * sizeof(int32_t),
                hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                return Status::hip_error("batch draft copy", hipGetErrorString(err), __FILE__,
                                         __LINE__);
            }
        }
    }

    for (std::size_t i = 0; i < plans.size(); ++i) {
        SlotPlan& plan = plans[i];
        if (!plan.batched) continue;
        DFlash2SpecDecoder& decoder = *plan.decoder;
        for (DFlash2GdnRankState& state : decoder.gdn_ranks) {
            auto scope = shelf::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            const Status st = spec_gdn_snapshot_conv(
                *state.pool, decoder.sequence->slot, state.conv_snapshot,
                state.conv_bytes, state.stream);
            if (!st.ok()) return st;
            if (state.commit_request != nullptr) {
                ::shelf::runtime::DeviceRequestDescriptor req{};
                req.request_handle.slot = decoder.sequence->slot;
                req.row_begin = plan.row_begin;
                req.row_count = 0u;
                const hipError_t err = hipMemcpy(
                    state.commit_request, &req, sizeof(req), hipMemcpyHostToDevice);
                if (err != hipSuccess) {
                    return Status::hip_error("batch commit descriptor", hipGetErrorString(err),
                                             __FILE__, __LINE__);
                }
            }
        }
    }

    std::vector<int32_t> token_ids;
    token_ids.reserve(total_rows);
    std::vector<ScheduledRequest> requests;
    requests.reserve(plans.size());
    for (std::size_t i = 0; i < plans.size(); ++i) {
        SlotPlan& plan = plans[i];
        if (!plan.batched) continue;
        DFlash2SpecDecoder& decoder = *plan.decoder;
        if (!bridge) {
            token_ids.push_back(plan.anchor);
            for (uint32_t j = 0; j < plan.drafts; ++j) {
                token_ids.push_back(drafts[i][j]);
            }
        }
        ScheduledRequest req;
        req.sequence = decoder.sequence;
        req.handle = decoder.sequence->request_handle();
        req.execution_class = ::shelf::runtime::ExecutionClass::SPEC_VERIFY;
        req.token_begin = plan.row_begin;
        req.num_tokens = plan.total_rows;
        req.prefix_tokens = plan.position;
        req.compute_logits = true;
        req.sample = true;
        req.num_output_rows = plan.total_rows;
        req.sampling = decoder.sampling;
        req.sampling_index = plan.sample_index;
        requests.push_back(req);
    }

    ScheduledBatch batch;
    batch.token_ids = bridge ? context.verify_token_ids_device : token_ids.data();
    batch.token_ids_location = bridge ? TokenIdsLocation::Device : TokenIdsLocation::Host;
    batch.requests = requests;
    batch.num_tokens = total_rows;
    batch.num_requests = static_cast<uint32_t>(requests.size());
    batch.num_decode_requests = 0u;
    batch.num_verify_requests = static_cast<uint32_t>(requests.size());
    batch.num_prefill_requests = 0u;
    batch.speculative_verify = true;

    std::vector<int32_t> sampled;
    std::vector<int32_t> decision_host;
    if (total_rows > 0u) {
        DFlash2SpecDecoder& leader = *plans.front().decoder;
        std::vector<ExecuteBatchOptions> options = build_batch_gdn_options(context, total_rows);
        if (batch_timing) {
            const hipError_t start_err =
                hipEventRecord(leader.verify_start_event, leader.stream);
            if (start_err != hipSuccess) {
                return Status::hip_error("batch verify start event",
                                         hipGetErrorString(start_err), __FILE__, __LINE__);
            }
            verify_events_recorded = true;
        }
        auto executed = leader.tp_hook != nullptr
            ? leader.tp_hook->on_execute(batch, options.data())
            : execute_batch(*leader.target, batch, leader.stream, options[0]);
        if (!executed.ok()) return executed.status();
        if (batch_timing) {
            const hipError_t stop_err =
                hipEventRecord(leader.verify_stop_event, leader.stream);
            if (stop_err != hipSuccess) {
                return Status::hip_error("batch verify stop event",
                                         hipGetErrorString(stop_err), __FILE__, __LINE__);
            }
        }
        BatchExecutionOutput output = executed.release();
        if (output.num_outputs < total_rows) {
            return Status::invalid_state("batch verify output rows mismatch", __FILE__,
                                         __LINE__);
        }
        sampled.resize(total_rows);
        if (bridge) {
            decision_host.assign(2u * context.max_rows, 0);
            hipError_t relay = hipMemcpyAsync(
                context.decision_staging_device, context.verify_token_ids_device,
                static_cast<std::size_t>(total_rows) * sizeof(int32_t),
                hipMemcpyDeviceToDevice, leader.stream);
            if (relay != hipSuccess) {
                return Status::hip_error("batch decision draft relay",
                                         hipGetErrorString(relay), __FILE__, __LINE__);
            }
            relay = hipMemcpyAsync(
                context.decision_staging_device + context.max_rows,
                output.sampled_tokens.data<int32_t>(),
                static_cast<std::size_t>(total_rows) * sizeof(int32_t),
                hipMemcpyDeviceToDevice, leader.stream);
            if (relay != hipSuccess) {
                return Status::hip_error("batch decision sample relay",
                                         hipGetErrorString(relay), __FILE__, __LINE__);
            }
            if (hipStreamSynchronize(leader.stream) != hipSuccess) {
                return Status::hip_error("batch decision sync",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
            const hipError_t err = hipMemcpy(
                decision_host.data(), context.decision_staging_device,
                decision_host.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                return Status::hip_error("batch decision copy", hipGetErrorString(err),
                                         __FILE__, __LINE__);
            }
            for (std::size_t i = 0; i < plans.size(); ++i) {
                const SlotPlan& plan = plans[i];
                if (!plan.batched) continue;
                for (uint32_t j = 0; j < plan.drafts; ++j) {
                    drafts[i][j] = decision_host[plan.row_begin + 1u + j];
                }
                for (uint32_t t = 0; t < plan.total_rows; ++t) {
                    sampled[plan.row_begin + t] =
                        decision_host[context.max_rows + plan.row_begin + t];
                }
            }
        } else {
            const hipError_t err = hipMemcpy(
                sampled.data(), output.sampled_tokens.data<int32_t>(),
                static_cast<std::size_t>(total_rows) * sizeof(int32_t),
                hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                return Status::hip_error("batch verify sample copy", hipGetErrorString(err),
                                         __FILE__, __LINE__);
            }
        }
    }

    DFlash2SpecBatchOutput out;
    out.slots.resize(plans.size());
    out.total_rows = total_rows;
    out.rounds = 1u;
    if (batch_timing) {
        double draft_gpu_ms = 0.0;
        if (propose_leader != nullptr) {
            float elapsed = 0.0f;
            if (hipEventElapsedTime(&elapsed, propose_leader->draft_start_event,
                                    propose_leader->draft_stop_event) == hipSuccess) {
                draft_gpu_ms = static_cast<double>(elapsed);
            }
        }
        if (verify_events_recorded) {
            float elapsed = 0.0f;
            if (hipEventElapsedTime(&elapsed, plans.front().decoder->verify_start_event,
                                    plans.front().decoder->verify_stop_event) == hipSuccess) {
                out.verify_gpu_ms = static_cast<double>(elapsed);
            }
        }
        out.draft_gpu_ms = draft_gpu_ms;
        out.round_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - round_t0).count();
    }

    for (std::size_t i = 0; i < plans.size(); ++i) {
        SlotPlan& plan = plans[i];
        DFlash2SpecDecoder& decoder = *plan.decoder;
        DFlash2SpecIterationOutput& result_out = out.slots[i];

        if (plan.stopped) {
            result_out.pending_token = plan.anchor;
            result_out.finished = true;
            continue;
        }
        if (!plan.batched) {
            const int32_t forced =
                plan.thinking_allowance == 0u ? plan.thinking_close_token : -1;
            const Status st = run_single_target(decoder, plan.anchor, result_out, forced);
            if (!st.ok()) return st;
            if (qwen_hybrid::is_stop_token(decoder.config.eos_tokens, result_out.pending_token)) {
                result_out.finished = true;
            }
            continue;
        }

        SpecVerifyResult verify;
        const Status accept_st = spec_greedy_accept(
            drafts[i].data(), plan.drafts, sampled.data() + plan.row_begin, true, verify);
        if (!accept_st.ok()) return accept_st;
        result_out.finished = spec_bound_result(verify, plan.remaining, decoder.config.eos_tokens);
        const uint32_t accepted = verify.num_accepted_drafts;
        result_out.num_drafts = plan.drafts;
        result_out.num_dflash_drafts = plan.drafts;
        result_out.num_accepted = accepted;
        result_out.emitted_count = std::min(
            static_cast<uint32_t>(verify.emitted_tokens.size()), plan.remaining);
        for (uint32_t j = 0; j < result_out.emitted_count; ++j) {
            result_out.emitted[j] = verify.emitted_tokens[j];
        }
        if (verify.correction_valid) {
            result_out.pending_token = verify.correction_token;
        } else if (verify.bonus_token_valid) {
            result_out.pending_token = verify.bonus_token;
        } else {
            result_out.pending_token = -1;
        }

        if (!result_out.finished &&
            plan.thinking_allowance != kDFlash2NoThinkingAllowance &&
            plan.thinking_close_token >= 0 && accepted == plan.thinking_allowance &&
            result_out.emitted_count == accepted + 1u) {
            result_out.emitted[accepted] = plan.thinking_close_token;
            result_out.pending_token = plan.thinking_close_token;
        }

        const uint32_t page_tokens = decoder.sequence->kv_pool()->page_tokens();
        if (page_tokens == 0u) {
            return Status::invalid_state("batch step page_tokens is zero", __FILE__, __LINE__);
        }

        // Verify deferred the recurrent state write, so the live slot still holds the
        // base state: replay the accepted prefix (all K+1 rows on a full accept).
        for (std::size_t r = 0; r < decoder.gdn_ranks.size(); ++r) {
            DFlash2GdnRankState& state = decoder.gdn_ranks[r];
            auto scope = shelf::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            Status st = Status::make_ok();
            if (accepted != plan.drafts) {
                st = spec_gdn_restore_conv(
                    *state.pool, decoder.sequence->slot, state.conv_snapshot,
                    state.conv_bytes, state.stream);
                if (!st.ok()) return st;
                st = restore_gdn_spec_conv(
                    *state.pool, decoder.sequence->slot, context.history[r],
                    plan.row_begin + accepted, state.stream);
                if (!st.ok()) return st;
            }
            GdnCompactLogDeviceView compact{};
            compact.delta = context.compact[r].delta;
            compact.k = context.compact[r].k;
            compact.a = context.compact[r].a;
            compact.delta_layer_stride = context.compact[r].delta_layer_stride;
            compact.k_layer_stride = context.compact[r].k_layer_stride;
            compact.a_layer_stride = context.compact[r].a_layer_stride;
            st = commit_gdn_compact_log(
                *state.pool, decoder.sequence->slot, compact, state.commit_request,
                accepted + 1u, decoder.gdn_key_heads, state.stream);
            if (!st.ok()) return st;
        }

        if (accepted == plan.drafts) {
            const auto taps = batch_target_taps(decoder, plan.row_begin);
            Status st = dflash2::dflash2_append_target_taps(
                *decoder.draft, *decoder.context, taps.data(),
                shelf::kernel::kDFlash2TargetTaps, plan.total_rows, plan.position,
                decoder.stream);
            if (!st.ok()) return st;
            const uint32_t required =
                ceil_div_u32(decoder.sequence->position, page_tokens);
            st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
            if (!st.ok()) return st;
        } else {
            decoder.sequence->position = plan.position + accepted + 1u;
            const auto taps = batch_target_taps(decoder, plan.row_begin);
            Status st = dflash2::dflash2_append_target_taps(
                *decoder.draft, *decoder.context, taps.data(),
                shelf::kernel::kDFlash2TargetTaps, accepted + 1u, plan.position,
                decoder.stream);
            if (!st.ok()) return st;
            const uint32_t required =
                ceil_div_u32(decoder.sequence->position, page_tokens);
            st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
            if (!st.ok()) return st;
        }

        decoder.sample_index = plan.sample_index + plan.total_rows;

        if (decoder.timing != nullptr) {
            DFlash2SpecTiming* tm = decoder.timing;
            ++tm->rounds;
            tm->accepted_drafts += accepted;
            tm->generated_tokens += result_out.emitted_count;
            tm->verify_rows_total += plan.total_rows;
            if (accepted >= plan.drafts) ++tm->full_accepts;
            else ++tm->partial_accepts;
            tm->gdn_history_bytes = 0u;
            for (const DFlash2GdnRankState& state : decoder.gdn_ranks) {
                if (!state.history.initialized) continue;
                tm->gdn_history_bytes +=
                    static_cast<uint64_t>(state.history.rows) *
                    static_cast<uint64_t>(state.history.conv_state_bytes +
                                          state.history.recurrent_state_bytes);
            }
        }

        for (uint32_t j = 0; j < result_out.emitted_count; ++j) {
            if (qwen_hybrid::is_stop_token(decoder.config.eos_tokens, result_out.emitted[j])) {
                result_out.emitted_count = j + 1u;
                result_out.finished = true;
                break;
            }
        }
        if (qwen_hybrid::is_stop_token(decoder.config.eos_tokens, result_out.pending_token)) {
            result_out.finished = true;
        }
        commit_history_tokens(decoder, result_out.emitted.data(), result_out.emitted_count);
    }

    return out;
}

}  // namespace shelf::qwen35::runtime
