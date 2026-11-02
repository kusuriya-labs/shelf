#include <phaseshift/models/qwen35/runtime/dflash2_spec_decoder.h>

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/models/qwen35/runtime/scheduling_types.h>
#include <phaseshift/models/qwen35/runtime/spec_decode.h>
#include <phaseshift/models/qwen35/kernels/dflash2/feature_concat.h>
#include <phaseshift/models/qwen35/eagle4/context_state.h>
#include <phaseshift/models/qwen35/eagle4/executor.h>
#include <phaseshift/models/qwen35/kernels/eagle4/ops.h>
#include <phaseshift/kernels/optimized/kv_relocate.h>
#include <phaseshift/models/qwen35/runtime/eagle4_tree.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <vector>

namespace ps::qwen35::runtime {

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
    req.execution_class = num_tokens == 1u ? ::ps::runtime::ExecutionClass::DECODE
                                           : ::ps::runtime::ExecutionClass::PREFILL;
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
    if (target.model == nullptr) {
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
    const Qwen35TextConfig& tc = target.model->text_config();
    const ps::weights::MatrixWeight& embed = target.model->weights().embed_tokens;

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
        env_flag_enabled("PHASESHIFT_DFLASH2_DEVICE_TOKEN_BRIDGE", true);
    decoder.host_proposal_visibility =
        env_flag_enabled("PHASESHIFT_DFLASH2_HOST_PROPOSAL_D2H", false);
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
    const std::vector<DFlash2GdnRankBinding>* tp_ranks) {
    if (target.model == nullptr) {
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
        auto leader_scope = ps::gpu::ScopedDevice::create(bindings.front().device);
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
                    ps::gpu::discard_cleanup_result(hipEventDestroy(events[j]));
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

    const uint32_t history_rows =
        std::max(config.num_drafts, config.eagle4_tree_width);
    const Qwen35TextConfig& tc = target.model->text_config();
    workspace.gdn_key_heads =
        static_cast<uint32_t>(tc.linear_num_key_heads != 0 ? tc.linear_num_key_heads : 16u);

    for (std::size_t r = 0; r < bindings.size(); ++r) {
        DFlash2GdnRankState& g = workspace.gdn_ranks[r];
        gpu::GpuArena& rank_arena = *bindings[r].arena;
        GdnStatePool& rank_pool = *bindings[r].pool;
        auto dev_scope = ps::gpu::ScopedDevice::create(g.device);
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
            auto rec_alloc = rank_arena.allocate_aligned(g.rec_bytes, 256u);
            if (!rec_alloc.ok()) {
                (void)shutdown_dflash2_spec_workspace(workspace);
                return rec_alloc.status();
            }
            g.rec_snapshot = rec_alloc.release().data();
            workspace.gdn_snapshot_bytes += g.rec_bytes;
        }
        auto history_result = create_gdn_spec_history(
            rank_arena, rank_pool, history_rows, config.eagle4_tree_width >= 2u);
        if (!history_result.ok()) {
            (void)shutdown_dflash2_spec_workspace(workspace);
            return history_result.status();
        }
        g.history = history_result.release();

        const uint64_t states = pool_view.num_gdn_states;
        const uint64_t log_rows = std::max(
            static_cast<uint64_t>(config.num_drafts) + 1u,
            static_cast<uint64_t>(config.eagle4_tree_width));
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

        auto req_alloc = rank_arena.allocate_aligned(
            sizeof(::ps::runtime::DeviceRequestDescriptor), 256u);
        if (!req_alloc.ok()) {
            (void)shutdown_dflash2_spec_workspace(workspace);
            return req_alloc.status();
        }
        g.commit_request =
            static_cast<::ps::runtime::DeviceRequestDescriptor*>(req_alloc.release().data());

        if (config.eagle4_tree_width >= 2u) {
            auto tree_alloc = [&](std::size_t bytes, void** out) -> Status {
                auto v = rank_arena.allocate_aligned(bytes, 256u);
                if (!v.ok()) return v.status();
                *out = v.release().data();
                return Status::make_ok();
            };
            void* p = nullptr;
            Status ts = Status::make_ok();
            ts = ts.ok() ? tree_alloc(16u * sizeof(std::uint32_t), &p) : ts;
            g.tree_rope_positions = static_cast<std::uint32_t*>(p);
            p = nullptr;
            ts = ts.ok() ? tree_alloc(16u * sizeof(int32_t), &p) : ts;
            g.tree_row_parent = static_cast<int32_t*>(p);
            p = nullptr;
            ts = ts.ok() ? tree_alloc(17u * sizeof(int32_t), &p) : ts;
            g.tree_anc_offsets = static_cast<int32_t*>(p);
            p = nullptr;
            ts = ts.ok() ? tree_alloc(256u * sizeof(std::uint32_t), &p) : ts;
            g.tree_anc_positions = static_cast<std::uint32_t*>(p);
            p = nullptr;
            ts = ts.ok() ? tree_alloc(16u * sizeof(int32_t), &p) : ts;
            g.tree_commit_rows = static_cast<int32_t*>(p);
            p = nullptr;
            ts = ts.ok() ? tree_alloc(16u * sizeof(std::uint32_t), &p) : ts;
            g.tree_kv_src = static_cast<std::uint32_t*>(p);
            p = nullptr;
            ts = ts.ok() ? tree_alloc(16u * sizeof(std::uint32_t), &p) : ts;
            g.tree_kv_dst = static_cast<std::uint32_t*>(p);
            if (!ts.ok()) {
                (void)shutdown_dflash2_spec_workspace(workspace);
                return ts;
            }
        }
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
    const std::vector<DFlash2GdnRankBinding>* tp_ranks) {
    return create_spec_workspace_impl(target, draft_max_drafts, gdn_pool, arena, config,
                                      stream, tp_ranks);
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
                                      gdn_pool, arena, config, stream, tp_ranks);
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
        auto scope = ps::gpu::ScopedDevice::create(g.device);
        if (!scope.ok()) return scope.status();
        ::ps::runtime::DeviceRequestDescriptor req{};
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

Result<DFlash2SpecDecoder> create_eagle4_spec_decoder(
    Executor& target,
    eagle4::Eagle4Executor& draft,
    eagle4::Eagle4ContextState& context,
    PagedSequenceState& sequence,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* tp_ranks,
    TpBatchHook* tp_hook,
    DFlash2SpecWorkspace* workspace,
    const std::vector<eagle4::Eagle4RankDraft>* eagle_ranks) {
    if (target.model == nullptr) {
        return Status::invalid_state("create_eagle4_spec_decoder: target model is null",
                                     __FILE__, __LINE__);
    }
    if (!draft.initialized || draft.config == nullptr) {
        return Status::invalid_state(
            "create_eagle4_spec_decoder: draft executor not initialized", __FILE__, __LINE__);
    }
    if (!context.initialized) {
        return Status::invalid_state("create_eagle4_spec_decoder: context not initialized",
                                     __FILE__, __LINE__);
    }
    if (!sequence.is_allocated()) {
        return Status::invalid_state("create_eagle4_spec_decoder: sequence not allocated",
                                     __FILE__, __LINE__);
    }
    const eagle4::Eagle4Config& dcfg = *draft.config;
    const Qwen35TextConfig& tc = target.model->text_config();
    if (config.num_drafts == 0u || config.num_drafts > 15u) {
        return Status::invalid_argument("create_eagle4_spec_decoder: num_drafts out of range",
                                        __FILE__, __LINE__);
    }
    if (config.eagle4_tree_width > 16u) {
        return Status::invalid_argument("create_eagle4_spec_decoder: tree width out of range",
                                        __FILE__, __LINE__);
    }
    if (dcfg.aux_layer_ids[dcfg.num_aux_layers - 1u] >= tc.num_hidden_layers) {
        return Status::invalid_argument(
            "create_eagle4_spec_decoder: target layer count too small for taps", __FILE__,
            __LINE__);
    }
    if (draft.embed_tokens == nullptr || draft.lm_head == nullptr) {
        return Status::invalid_argument(
            "create_eagle4_spec_decoder: draft has no embed_tokens/lm_head", __FILE__,
            __LINE__);
    }

    DFlash2SpecDecoder decoder;
    decoder.target = &target;
    decoder.eagle = &draft;
    decoder.eagle_context = &context;
    if (eagle_ranks != nullptr && !eagle_ranks->empty()) {
        decoder.eagle_ranks = *eagle_ranks;
    } else {
        eagle4::Eagle4RankDraft single;
        single.executor = &draft;
        single.context = &context;
        single.stream = stream;
        if (tp_ranks != nullptr && !tp_ranks->empty()) {
            single.device = (*tp_ranks)[0].device;
        }
        decoder.eagle_ranks.push_back(single);
    }
    for (std::size_t r = 0; r < decoder.eagle_ranks.size(); ++r) {
        if (tp_ranks != nullptr && r < tp_ranks->size() &&
            (*tp_ranks)[r].executor != nullptr) {
            decoder.eagle_ranks[r].target = (*tp_ranks)[r].executor;
            continue;
        }
        if (r == 0u) {
            decoder.eagle_ranks[r].target = &target;
            continue;
        }
        return Status::invalid_state(
            "create_eagle4_spec_decoder: rank target executor is missing", __FILE__,
            __LINE__);
    }
    decoder.sequence = &sequence;
    decoder.stream = stream;
    decoder.config = config;
    decoder.tp_hook = tp_hook;
    decoder.hidden_size = static_cast<uint32_t>(dcfg.hidden_size);
    decoder.device_token_bridge = false;
    decoder.host_proposal_visibility = false;
    decoder.eagle_num_candidates = draft.num_candidates;
    decoder.eagle_candidate_topk = 1u;
    decoder.gdn_key_heads =
        static_cast<uint32_t>(tc.linear_num_key_heads != 0 ? tc.linear_num_key_heads : 16u);

    DFlash2SpecWorkspace local_workspace;
    DFlash2SpecWorkspace* bound = workspace;
    if (bound == nullptr) {
        auto workspace_result = create_spec_workspace_impl(
            target, 15u, gdn_pool, arena, config, stream, tp_ranks);
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

    {
        auto drafts_alloc = arena.allocate_aligned(16u * sizeof(int32_t), 256u);
        if (!drafts_alloc.ok()) {
            (void)shutdown_dflash2_spec_workspace(local_workspace);
            return drafts_alloc.status();
        }
        decoder.eagle_draft_tokens_device =
            static_cast<int32_t*>(drafts_alloc.release().data());
    }

    decoder.initialized = true;
    return decoder;
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
    decoder.eagle = nullptr;
    decoder.eagle_context = nullptr;
    decoder.eagle_draft_tokens_device = nullptr;
    decoder.eagle_num_candidates = 0u;
    decoder.eagle_root_ready = false;
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
    uint32_t prefix_tokens) {
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
    const uint32_t chunk_limit = decoder.target->config.max_scheduled_tokens;
    if (chunk_limit == 0u) {
        return Status::invalid_state("dflash2_spec_prefill: target has no token capacity",
                                     __FILE__, __LINE__);
    }
    const bool eagle = decoder.eagle != nullptr;
    uint32_t tap_count = 0u;
    if (eagle) {
        tap_count = static_cast<uint32_t>(decoder.eagle->config->num_aux_layers);
    } else {
        tap_count = static_cast<uint32_t>(decoder.draft->config->num_target_layer_ids);
    }
    if (tap_count == 0u || (!eagle && tap_count != ps::kernel::kDFlash2TargetTaps)) {
        return Status::invalid_state("dflash2_spec_prefill: draft tap count mismatch", __FILE__,
                                     __LINE__);
    }

    DFlash2PrefillOutput out;
    std::vector<ScheduledRequest> requests;
    uint32_t offset = prefix_tokens;
    while (offset < prompt_count) {
        uint32_t n = std::min(chunk_limit, prompt_count - offset);
        const bool final_chunk = (offset + n == prompt_count);
        ScheduledBatch batch = make_prefill_batch(
            *decoder.sequence, requests, prompt_tokens + offset, n, offset, final_chunk,
            final_chunk ? &decoder.sampling : nullptr, decoder.sample_index);
        const std::vector<ExecuteBatchOptions> poptions = build_gdn_options(decoder, 1u);
        auto executed = run_target_batch(decoder, batch, poptions);
        if (!executed.ok()) return executed.status();
        BatchExecutionOutput output = executed.release();

        std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps{};
        for (uint32_t t = 0; t < tap_count; ++t) {
            taps[t] = decoder.target->dflash_target_hidden[t].data<bf16_t>();
        }
        if (eagle) {
            const uint32_t extend_rows = final_chunk ? (n > 1u ? n - 1u : 0u) : n;
            uint32_t row = 0u;
            while (row < extend_rows) {
                const uint32_t slice = std::min<uint32_t>(
                    eagle4::kEagle4ExtendRowsMax, extend_rows - row);
                std::array<int32_t, eagle4::kEagle4ExtendRowsMax> shift_tokens{};
                for (uint32_t r = 0; r < slice; ++r) {
                    shift_tokens[r] = prompt_tokens[offset + row + r + 1u];
                }
                Status extended = eagle4::eagle4_extend_rows_tp(
                    *decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks,
                    taps.data(), tap_count, row, shift_tokens.data(), slice, offset + row,
                    decoder.eagle_candidate_topk, decoder.eagle_draft_tokens_device,
                    decoder.stream);
                if (!extended.ok()) return extended;
                row += slice;
            }
        } else {
            Status appended = dflash2::dflash2_append_target_taps(
                *decoder.draft, *decoder.context, taps.data(), tap_count, n, offset,
                decoder.stream);
            if (!appended.ok()) return appended;
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
            if (eagle) {
                Status extended = eagle4::eagle4_extend_rows_tp(
                    *decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks,
                    taps.data(), tap_count, n - 1u, &out.pending_token, 1u, offset + n - 1u,
                    decoder.eagle_candidate_topk, decoder.eagle_draft_tokens_device,
                    decoder.stream);
                if (!extended.ok()) return extended;
                decoder.eagle_root_ready = true;
            }
        }
        offset += n;
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
        ? ::ps::runtime::ExecutionClass::DECODE
        : (speculative_verify ? ::ps::runtime::ExecutionClass::SPEC_VERIFY
                              : ::ps::runtime::ExecutionClass::PREFILL);
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
    std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps>& taps) {
    for (uint32_t t = 0; t < ps::kernel::kDFlash2TargetTaps; ++t) {
        taps[t] = target.dflash_target_hidden[t].data<bf16_t>();
    }
}

Status append_committed(
    DFlash2SpecDecoder& decoder,
    uint32_t position,
    uint32_t rows,
    const int32_t* sampled_host) {
    std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps{};
    collect_target_taps(*decoder.target, taps);
    if (decoder.eagle == nullptr) {
        return dflash2::dflash2_append_target_taps(
            *decoder.draft, *decoder.context, taps.data(), ps::kernel::kDFlash2TargetTaps,
            rows, position, decoder.stream);
    }
    if (sampled_host == nullptr) {
        return Status::invalid_state("eagle4 commit: sampled token buffer is null",
                                     __FILE__, __LINE__);
    }
    Status st = eagle4::eagle4_extend_rows_tp(
        *decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks, taps.data(),
        static_cast<uint32_t>(decoder.eagle->config->num_aux_layers), 0u, sampled_host, rows,
        position, decoder.eagle_candidate_topk, decoder.eagle_draft_tokens_device,
        decoder.stream);
    if (!st.ok()) return st;
    decoder.eagle_root_ready = true;
    return Status::make_ok();
}

Status run_single_target(
    DFlash2SpecDecoder& decoder,
    int32_t token,
    DFlash2SpecIterationOutput& out) {
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
    std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps{};
    collect_target_taps(*decoder.target, taps);
    Status st = Status::make_ok();
    if (decoder.eagle != nullptr) {
        st = eagle4::eagle4_extend_rows_tp(
            *decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks, taps.data(),
            static_cast<uint32_t>(decoder.eagle->config->num_aux_layers), 0u, &sampled, 1u,
            position, decoder.eagle_candidate_topk, decoder.eagle_draft_tokens_device,
            decoder.stream);
        if (!st.ok()) return st;
        decoder.eagle_root_ready = true;
    } else {
        st = dflash2::dflash2_append_target_taps(
            *decoder.draft, *decoder.context, taps.data(), ps::kernel::kDFlash2TargetTaps, 1u,
            position, decoder.stream);
    }
    if (!st.ok()) return st;
    out.emitted[0] = sampled;
    out.emitted_count = 1u;
    out.pending_token = sampled;
    if (tm != nullptr) ++tm->generated_tokens;
    commit_history_tokens(decoder, out.emitted.data(), 1u);
    return Status::make_ok();
}


struct Eagle4TreeRoundState {
    Eagle4DraftTree tree;
    int32_t pending_token = 0;
    uint32_t remaining_tokens = 0;
    uint32_t position = 0;
    uint32_t rows = 0;
    int32_t plan_depth = 1;
    uint64_t sample_index = 0;
    double draft_ms0 = 0.0;
    double verify_ms0 = 0.0;
    bool done = false;
    std::array<int32_t, 16> verify_inputs{};
    std::array<uint32_t, 16> rope_positions{};
    std::array<int32_t, 16> row_parent{};
    std::vector<int32_t> anc_offsets;
    std::vector<uint32_t> anc_positions;
};

Status eagle4_tree_prepare_round(DFlash2SpecDecoder& decoder,
                          int32_t pending_token,
                          uint32_t remaining_tokens,
                          uint32_t position,
                          uint32_t node_cap,
                          DFlash2SpecIterationOutput& out,
                          Eagle4TreeRoundState& state) {
    DFlash2SpecTiming* tm = decoder.timing;
    const bool want_timing = tm != nullptr;
    const uint32_t budget_width = decoder.config.eagle4_tree_width;
    const int32_t max_depth = static_cast<int32_t>(decoder.config.num_drafts) - 1;
    const uint32_t room = decoder.sequence->max_seq_len - position;

    out = DFlash2SpecIterationOutput{};
    if (room == 0u) {
        Status st = run_single_target(decoder, pending_token, out);
        if (!st.ok()) return st;
        if (is_stop_token(decoder.config.eos_tokens, out.pending_token)) out.finished = true;
        state.done = true;
        return Status::make_ok();
    }
    state.draft_ms0 = want_timing ? tm->draft_ms : 0.0;
    state.verify_ms0 = want_timing ? tm->verify_ms : 0.0;
    uint32_t budget = budget_width < room ? budget_width : room;
    int32_t plan_depth = max_depth;
    if (decoder.config.eagle4_tree_dynamic != 0u && decoder.tree_cost.frozen) {
        const Eagle4TreePlan plan =
            eagle4_cost_choose(decoder.tree_cost, max_depth,
                               static_cast<int32_t>(budget_width));
        plan_depth = plan.depth;
        if (plan.width > 0 && static_cast<uint32_t>(plan.width) < budget) {
            budget = static_cast<uint32_t>(plan.width);
        }
    }

    state.pending_token = pending_token;
    state.remaining_tokens = remaining_tokens;
    state.position = position;
    if (node_cap != 0u && budget > node_cap) budget = node_cap;
    Eagle4DraftTree& tree = state.tree;
    eagle4_tree_reset(tree, static_cast<int32_t>(budget), plan_depth);
    std::array<int32_t, 16> creator_row{};
    std::array<int32_t, 16 * 16> cand_ids{};
    std::array<float, 16 * 16> cand_probs{};
    std::array<int32_t, kEagle4TreeMaxNodes> path_scratch{};
    std::vector<int32_t> anc_nodes_host;
    std::vector<uint32_t> anc_positions_host;
    std::vector<int32_t> anc_offsets_host;

    if (!decoder.eagle_root_ready) {
        return Status::invalid_state("dflash2_tree_round: eagle4 root not ready", __FILE__,
                                     __LINE__);
    }

    ScopedTimer draft_timer(want_timing ? &tm->draft_ms : nullptr);
    if (want_timing) {
        const hipError_t e0 = hipEventRecord(decoder.draft_start_event, decoder.stream);
        if (e0 != hipSuccess) {
            return Status::hip_error("eagle4 tree draft start event",
                                     hipGetErrorString(e0), __FILE__, __LINE__);
        }
    }
    {
        if (hipStreamSynchronize(decoder.stream) != hipSuccess) {
            return Status::hip_error("eagle4 tree root sync",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        int32_t root_token = -1;
        if (hipMemcpy(&root_token, decoder.eagle_draft_tokens_device, sizeof(int32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            return Status::hip_error("eagle4 tree root copy",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        if (root_token < 0) {
            return Status::invalid_state("eagle4 tree root token invalid", __FILE__,
                                         __LINE__);
        }
        const Status root_st = eagle4_tree_insert_root(tree, root_token, 1.0f);
        if (!root_st.ok()) return root_st;
    }

    const uint32_t depth_steps =
        static_cast<uint32_t>(plan_depth > 0 ? plan_depth : 1);
    const uint32_t child_k =
        std::max(1u, (budget_width - 1u + depth_steps - 1u) / depth_steps);
    for (int32_t step = 0; step <= plan_depth && eagle4_tree_step_available(tree);
         ++step) {
        const uint32_t rows = static_cast<uint32_t>(tree.frontier_count);
        const int32_t depth = tree.depth[tree.frontier[0]];
        std::array<int32_t, 16> tree_tokens{};
        std::array<int32_t, 16> tree_nodes{};
        std::array<int32_t, 16> tree_src_rows{};
        for (uint32_t r = 0; r < rows; ++r) {
            const int32_t node = tree.frontier[r];
            tree_tokens[r] = tree.token[node];
            tree_nodes[r] = node;
            tree_src_rows[r] = step == 0 ? 0 : creator_row[node];
        }
        anc_offsets_host.clear();
        anc_nodes_host.clear();
        anc_offsets_host.push_back(0);
        for (uint32_t r = 0; r < rows; ++r) {
            const std::size_t base = anc_nodes_host.size();
            anc_nodes_host.resize(base + static_cast<std::size_t>(kEagle4TreeMaxNodes));
            const int32_t path_len = static_cast<int32_t>(
                eagle4_tree_node_path(tree, tree.frontier[r], anc_nodes_host.data() + base));
            anc_nodes_host.resize(base + static_cast<std::size_t>(path_len));
            anc_offsets_host.push_back(static_cast<int32_t>(anc_nodes_host.size()));
        }
        const uint32_t step_position = position + static_cast<uint32_t>(depth);
        Status st = eagle4::eagle4_tree_draft_step_tp(
            *decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks,
            tree_tokens.data(), tree_nodes.data(), tree_src_rows.data(), step == 0,
            anc_offsets_host.data(), anc_nodes_host.data(),
            static_cast<uint32_t>(anc_offsets_host.size()),
            static_cast<uint32_t>(anc_nodes_host.size()), rows, step_position, position,
            child_k, decoder.stream);
        if (!st.ok()) return st;
        if (hipStreamSynchronize(decoder.stream) != hipSuccess) {
            return Status::hip_error("eagle4 tree draft sync",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        const std::size_t cand_count = static_cast<std::size_t>(rows) * child_k;
        if (hipMemcpy(cand_ids.data(), decoder.eagle->topk_ids.data<int32_t>(),
                      cand_count * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(cand_probs.data(), decoder.eagle->topk_probs.data<float>(),
                      cand_count * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
            return Status::hip_error("eagle4 tree candidates",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        int32_t best_child = -1;
        float best_prob = -1.0f;
        for (uint32_t r = 0; r < rows; ++r) {
            const std::size_t base = static_cast<std::size_t>(r) * child_k;
            const int32_t before = tree.num_nodes;
            (void)eagle4_tree_insert_children(tree, tree.frontier[r], cand_ids.data() + base,
                                              cand_probs.data() + base, child_k);
            for (int32_t c = before; c < tree.num_nodes; ++c) {
                creator_row[static_cast<std::size_t>(c)] = static_cast<int32_t>(r);
                if (tree.score[static_cast<std::size_t>(c)] > best_prob) {
                    best_prob = tree.score[static_cast<std::size_t>(c)];
                    best_child = c;
                }
            }
        }
        tree.frontier_count = 0;
        if (best_child >= 0) {
            tree.frontier[0] = best_child;
            tree.frontier_count = 1;
        }
    }
    if (want_timing) {
        const hipError_t e1 = hipEventRecord(decoder.draft_stop_event, decoder.stream);
        if (e1 != hipSuccess) {
            return Status::hip_error("eagle4 tree draft stop event",
                                     hipGetErrorString(e1), __FILE__, __LINE__);
        }
        float elapsed = 0.0f;
        if (hipEventElapsedTime(&elapsed, decoder.draft_start_event,
                                decoder.draft_stop_event) == hipSuccess) {
            tm->draft_gpu_ms += static_cast<double>(elapsed);
        }
    }

    const uint32_t rows = static_cast<uint32_t>(tree.num_nodes);
    if (rows == 0u) {
        return Status::invalid_state("eagle4 tree round produced no nodes", __FILE__,
                                     __LINE__);
    }


    {
        ScopedTimer t(want_timing ? &tm->gdn_snapshot_ms : nullptr);
        for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
            auto scope = ps::gpu::ScopedDevice::create(g.device);
            if (!scope.ok()) return scope.status();
            Status st = spec_gdn_snapshot(*g.pool, decoder.sequence->slot,
                                          g.conv_snapshot, g.rec_snapshot, g.conv_bytes,
                                          g.rec_bytes, g.stream);
            if (!st.ok()) return st;
        }
    }

    std::array<int32_t, 16>& verify_inputs = state.verify_inputs;
    std::array<uint32_t, 16>& rope_positions = state.rope_positions;
    std::array<int32_t, 16>& row_parent = state.row_parent;
    verify_inputs[0] = pending_token;
    rope_positions[0] = position;
    row_parent[0] = -1;
    anc_offsets_host.clear();
    anc_positions_host.clear();
    anc_offsets_host.push_back(0);
    for (uint32_t n = 0; n < rows; ++n) {
        const int32_t path_len = static_cast<int32_t>(
            eagle4_tree_node_path(tree, static_cast<int32_t>(n), path_scratch.data()));
        for (int32_t i = 0; i < path_len; ++i) {
            anc_positions_host.push_back(
                position + static_cast<uint32_t>(path_scratch[static_cast<std::size_t>(i)]));
        }
        anc_offsets_host.push_back(static_cast<int32_t>(anc_positions_host.size()));
        if (n > 0u) {
            verify_inputs[n] = tree.token[tree.parent[n]];
            rope_positions[n] = position + static_cast<uint32_t>(tree.depth[n]);
            row_parent[n] = tree.parent[n];
        }
    }
    for (std::size_t r = 0; r < decoder.gdn_ranks.size(); ++r) {
        DFlash2GdnRankState& g = decoder.gdn_ranks[r];
        auto scope = ps::gpu::ScopedDevice::create(g.device);
        if (!scope.ok()) return scope.status();
        if (hipMemcpyAsync(g.tree_rope_positions, rope_positions.data(),
                           rows * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                           g.stream) != hipSuccess ||
            hipMemcpyAsync(g.tree_row_parent, row_parent.data(),
                           rows * sizeof(int32_t), hipMemcpyHostToDevice, g.stream) !=
                hipSuccess ||
            hipMemcpyAsync(g.tree_anc_offsets, anc_offsets_host.data(),
                           anc_offsets_host.size() * sizeof(int32_t),
                           hipMemcpyHostToDevice, g.stream) != hipSuccess ||
            hipMemcpyAsync(g.tree_anc_positions, anc_positions_host.data(),
                           anc_positions_host.size() * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, g.stream) != hipSuccess) {
            return Status::hip_error("eagle4 tree verify inputs",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }
    state.rows = static_cast<uint32_t>(tree.num_nodes);
    state.plan_depth = plan_depth;
    state.anc_offsets = std::move(anc_offsets_host);
    state.anc_positions = std::move(anc_positions_host);
    return Status::make_ok();
}

Status eagle4_tree_finish_round(DFlash2SpecDecoder& decoder,
                         Eagle4TreeRoundState& state,
                         const int32_t* sampled,
                         uint32_t row_base,
                         DFlash2SpecIterationOutput& out) {
    DFlash2SpecTiming* tm = decoder.timing;
    const bool want_timing = tm != nullptr;
    const int32_t pending_token = state.pending_token;
    const uint32_t remaining_tokens = state.remaining_tokens;
    const uint32_t position = state.position;
    const uint32_t rows = state.rows;
    const int32_t plan_depth = state.plan_depth;
    const double draft_ms0 = state.draft_ms0;
    const double verify_ms0 = state.verify_ms0;
    Eagle4DraftTree& tree = state.tree;
    if (sampled[0] < 0) {
        return Status::invalid_state("eagle4 tree verify produced no sampled token",
                                     __FILE__, __LINE__);
    }



    if (std::getenv("PHASESHIFT_EAGLE4_DEBUG_ROUND") != nullptr) {
        std::fprintf(stderr, "EAGLE4_TREE_ROUND pos=%u nodes=", position);
        for (uint32_t i = 0; i < rows; ++i) std::fprintf(stderr, "%d,", tree.token[i]);
        std::fprintf(stderr, " parents=");
        for (uint32_t i = 0; i < rows; ++i) std::fprintf(stderr, "%d,", tree.parent[i]);
        std::fprintf(stderr, " sampled=");
        for (uint32_t i = 0; i < rows; ++i) std::fprintf(stderr, "%d,", sampled[i]);
        std::fprintf(stderr, "\n");
    }

    Eagle4TreeAcceptResult accept;
    Status accept_st = spec_tree_greedy_accept(tree, sampled, accept);
    if (!accept_st.ok()) return accept_st;
    out.num_accepted = accept.num_accepted;
    const uint32_t emitted_total = static_cast<uint32_t>(accept.emitted_tokens.size());
    out.emitted_count = emitted_total < remaining_tokens ? emitted_total : remaining_tokens;
    for (uint32_t i = 0; i < out.emitted_count; ++i) {
        out.emitted[i] = accept.emitted_tokens[i];
    }
    out.pending_token =
        out.emitted_count > 0u ? out.emitted[out.emitted_count - 1u] : pending_token;
    out.num_drafts = rows;

    const uint32_t page_tokens = decoder.sequence->kv_pool()->page_tokens();
    if (page_tokens == 0u) {
        return Status::invalid_state("eagle4 tree round: page_tokens is zero", __FILE__,
                                     __LINE__);
    }
    const uint32_t committed = accept.num_committed_target_tokens;
    if (committed == 0u || committed > rows) {
        return Status::invalid_state("eagle4 tree round: committed rows out of range",
                                     __FILE__, __LINE__);
    }

    std::array<int32_t, kEagle4TreeMaxNodes> commit_rows{};
    for (uint32_t i = 0; i < accept.path_count; ++i) {
        commit_rows[i] = accept.path[i];
    }
    if (accept.correction_valid) {
        commit_rows[accept.path_count] = accept.correction_row;
    }
    const uint32_t commit_row_count =
        accept.path_count + (accept.correction_valid ? 1u : 0u);

    {
        ScopedTimer t(want_timing ? &tm->gdn_restore_ms : nullptr);
        for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
            auto scope = ps::gpu::ScopedDevice::create(g.device);
            if (!scope.ok()) return scope.status();
            Status st = spec_gdn_restore(*g.pool, decoder.sequence->slot,
                                         g.conv_snapshot, g.rec_snapshot, g.conv_bytes,
                                         g.rec_bytes, g.stream);
            if (!st.ok()) return st;
            const uint32_t conv_row =
                accept.correction_valid
                    ? static_cast<uint32_t>(accept.correction_row)
                    : accept.path[accept.path_count - 1u];
            st = restore_gdn_spec_conv(*g.pool, decoder.sequence->slot, g.history,
                                       conv_row + row_base, g.stream);
            if (!st.ok()) return st;
        }
    }

    {
        ScopedTimer t(want_timing ? &tm->dflash_commit_ms : nullptr);
        for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
            auto scope = ps::gpu::ScopedDevice::create(g.device);
            if (!scope.ok()) return scope.status();
            std::array<int32_t, kEagle4TreeMaxNodes> upload_rows{};
            for (uint32_t i = 0; i < commit_row_count; ++i) {
                upload_rows[i] = commit_rows[i] + static_cast<int32_t>(row_base);
            }
            if (hipMemcpyAsync(g.tree_commit_rows, upload_rows.data(),
                               commit_row_count * sizeof(int32_t),
                               hipMemcpyHostToDevice, g.stream) != hipSuccess) {
                return Status::hip_error("eagle4 tree commit rows",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
            GdnCompactLogDeviceView compact{};
            compact.delta = g.compact_delta;
            compact.k = g.compact_k;
            compact.a = g.compact_a;
            compact.delta_layer_stride = g.compact_delta_layer_stride;
            compact.k_layer_stride = g.compact_k_layer_stride;
            compact.a_layer_stride = g.compact_a_layer_stride;
            Status st = commit_gdn_compact_log(
                *g.pool, decoder.sequence->slot, compact, g.commit_request,
                commit_row_count, decoder.gdn_key_heads, g.stream, g.tree_commit_rows,
                g.history.rows);
            if (!st.ok()) return st;
        }

        decoder.sequence->position = position + committed;
        std::array<uint32_t, kEagle4TreeMaxNodes> kv_src{};
        std::array<uint32_t, kEagle4TreeMaxNodes> kv_dst{};
        for (uint32_t i = 0; i < committed; ++i) {
            const uint32_t row_i = static_cast<uint32_t>(
                i < accept.path_count ? accept.path[i] : accept.correction_row);
            kv_src[i] = position + row_i;
            kv_dst[i] = position + i;
        }
        const uint32_t slot = decoder.sequence->slot;
        for (std::size_t r = 0; r < decoder.gdn_ranks.size(); ++r) {
            DFlash2GdnRankState& g = decoder.gdn_ranks[r];
            auto scope = ps::gpu::ScopedDevice::create(g.device);
            if (!scope.ok()) return scope.status();
            if (g.model_state == nullptr) {
                return Status::invalid_state("eagle4 tree rank model state missing",
                                             __FILE__, __LINE__);
            }
            if (hipMemcpyAsync(g.tree_kv_src, kv_src.data(),
                               committed * sizeof(std::uint32_t),
                               hipMemcpyHostToDevice, g.stream) != hipSuccess ||
                hipMemcpyAsync(g.tree_kv_dst, kv_dst.data(),
                               committed * sizeof(std::uint32_t),
                               hipMemcpyHostToDevice, g.stream) != hipSuccess) {
                return Status::hip_error("eagle4 tree kv relocation inputs",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
            const auto& state_view = *g.model_state;
        if (state_view.kv_dtype == 1u) {
            ps::kernel::KvRelocateFp8Args reloc;
            reloc.common.src_positions = g.tree_kv_src;
            reloc.common.dst_positions = g.tree_kv_dst;
            reloc.common.rows = committed;
            reloc.common.slot = slot;
            reloc.common.block_tables = state_view.block_table.block_tables;
            reloc.common.block_table_stride = state_view.block_table.block_table_stride;
            reloc.common.max_sequences = state_view.block_table.max_sequences;
            reloc.common.page_tokens = state_view.kv_fp8.page_tokens;
            reloc.common.num_pages = state_view.kv_fp8.num_pages;
            reloc.common.num_attention_layers = state_view.kv_fp8.num_attention_layers;
            reloc.common.kv_heads = state_view.kv_fp8.kv_heads;
            reloc.common.head_dim = state_view.kv_fp8.head_dim;
            reloc.common.elems_per_token = state_view.kv_fp8.elems_per_token;
            reloc.common.elems_per_page = state_view.kv_fp8.elems_per_page;
            reloc.common.elems_per_layer = state_view.kv_fp8.elems_per_layer;
            reloc.k_pool = state_view.kv_fp8.k_pool;
            reloc.v_pool = state_view.kv_fp8.v_pool;
            reloc.k_scale_pool = state_view.kv_fp8.k_scale_pool;
            reloc.v_scale_pool = state_view.kv_fp8.v_scale_pool;
            const hipError_t rh = ps::kernel::launch_kv_relocate_fp8_e4m3(reloc, g.stream);
            if (rh != hipSuccess) {
                return Status::hip_error("eagle4 tree kv relocation (fp8)",
                                         hipGetErrorString(rh), __FILE__, __LINE__);
            }
        } else {
            ps::kernel::KvRelocateBf16Args reloc;
            reloc.common.src_positions = g.tree_kv_src;
            reloc.common.dst_positions = g.tree_kv_dst;
            reloc.common.rows = committed;
            reloc.common.slot = slot;
            reloc.common.block_tables = state_view.block_table.block_tables;
            reloc.common.block_table_stride = state_view.block_table.block_table_stride;
            reloc.common.max_sequences = state_view.block_table.max_sequences;
            reloc.common.page_tokens = state_view.kv_bf16.page_tokens;
            reloc.common.num_pages = state_view.kv_bf16.num_pages;
            reloc.common.num_attention_layers = state_view.kv_bf16.num_attention_layers;
            reloc.common.kv_heads = state_view.kv_bf16.kv_heads;
            reloc.common.head_dim = state_view.kv_bf16.head_dim;
            reloc.common.elems_per_token = state_view.kv_bf16.elems_per_token;
            reloc.common.elems_per_page = state_view.kv_bf16.elems_per_page;
            reloc.common.elems_per_layer = state_view.kv_bf16.elems_per_layer;
            reloc.k_pool = state_view.kv_bf16.k_pool;
            reloc.v_pool = state_view.kv_bf16.v_pool;
            const hipError_t rh = ps::kernel::launch_kv_relocate_bf16(reloc, g.stream);
            if (rh != hipSuccess) {
                return Status::hip_error("eagle4 tree kv relocation (bf16)",
                                         hipGetErrorString(rh), __FILE__, __LINE__);
            }
        }
        }

        std::array<int32_t, kEagle4TreeMaxNodes> extend_tokens{};
        std::array<int32_t, kEagle4TreeMaxNodes> tap_rows_host{};
        for (uint32_t i = 0; i < commit_row_count; ++i) {
            extend_tokens[i] = sampled[commit_rows[i]];
            tap_rows_host[i] = commit_rows[i] + static_cast<int32_t>(row_base);
        }

        Status es = eagle4::eagle4_extend_rows_tp(
            *decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks, nullptr,
            static_cast<uint32_t>(decoder.eagle->config->num_aux_layers), 0u,
            extend_tokens.data(), commit_row_count, position,
            decoder.eagle_candidate_topk, decoder.eagle_draft_tokens_device,
            decoder.stream, tap_rows_host.data(), commit_row_count);
        if (!es.ok()) return es;

        const uint32_t required = ceil_div_u32(decoder.sequence->position, page_tokens);
        Status rb = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
        if (!rb.ok()) return rb;
    }

    if (decoder.config.eagle4_tree_dynamic != 0u && want_timing) {
        Eagle4RoundSample sample;
        sample.draft_us = static_cast<float>((tm->draft_ms - draft_ms0) * 1000.0);
        sample.verify_us = static_cast<float>((tm->verify_ms - verify_ms0) * 1000.0);
        sample.accepted = accept.num_accepted;
        sample.depth_cap = plan_depth;
        sample.width = static_cast<int32_t>(rows);
        eagle4_cost_observe(decoder.tree_cost, sample);
    }

    if (tm != nullptr) {
        tm->accepted_drafts += accept.num_accepted;
        tm->generated_tokens += out.emitted_count;
        tm->verify_rows_total += rows;
        if (accept.correction_valid) {
            ++tm->partial_accepts;
        } else if (accept.leaf_accept) {
            ++tm->full_accepts;
        }
    }

    for (uint32_t i = 0; i < out.emitted_count; ++i) {
        if (is_stop_token(decoder.config.eos_tokens, out.emitted[i])) {
            out.emitted_count = i + 1u;
            out.finished = true;
            break;
        }
    }
    if (is_stop_token(decoder.config.eos_tokens, out.pending_token)) out.finished = true;
    commit_history_tokens(decoder, out.emitted.data(), out.emitted_count);
    return Status::make_ok();
}

Status dflash2_tree_round(DFlash2SpecDecoder& decoder,
                          int32_t pending_token,
                          uint32_t remaining_tokens,
                          uint32_t position,
                          DFlash2SpecIterationOutput& out) {
    DFlash2SpecTiming* tm = decoder.timing;
    const bool want_timing = tm != nullptr;
    Eagle4TreeRoundState state;
    Status prep = eagle4_tree_prepare_round(decoder, pending_token, remaining_tokens,
                                            position, 0u, out, state);
    if (!prep.ok()) return prep;
    if (state.done) return Status::make_ok();
    const Eagle4DraftTree& tree = state.tree;
    const uint32_t rows = state.rows;
    const std::array<int32_t, 16>& verify_inputs = state.verify_inputs;
    std::vector<ScheduledRequest> requests;
    std::array<int32_t, kDFlash2SpecMaxVerifyRows> sampled{};
    state.sample_index = decoder.sample_index;
    decoder.sample_index += rows;
    {
        ScopedTimer t(want_timing ? &tm->verify_ms : nullptr);
        ScheduledBatch batch = make_verify_batch(
            *decoder.sequence, requests, verify_inputs.data(), rows, position, rows,
            true, TokenIdsLocation::Host, &decoder.sampling, state.sample_index);
        std::vector<ExecuteBatchOptions> options = build_gdn_options(decoder, rows);
        bool gdn_parent_needed = false;
        for (uint32_t n = 1u; n < rows; ++n) {
            if (tree.parent[n] != static_cast<int32_t>(n) - 1) gdn_parent_needed = true;
        }
        for (std::size_t r = 0; r < options.size(); ++r) {
            ExecuteBatchOptions& o = options[r];
            DFlash2GdnRankState& g = decoder.gdn_ranks[r];
            o.tree_anc_offsets = reinterpret_cast<const uint32_t*>(g.tree_anc_offsets);
            o.tree_anc_positions = g.tree_anc_positions;
            o.tree_committed_end = position;
            if (gdn_parent_needed) {
                o.tree_row_parent = g.tree_row_parent;
            }
            o.rope_positions = g.tree_rope_positions;
        }
        if (want_timing) {
            const hipError_t e0 = hipEventRecord(decoder.verify_start_event, decoder.stream);
            if (e0 != hipSuccess) {
                return Status::hip_error("eagle4 tree verify start event",
                                         hipGetErrorString(e0), __FILE__, __LINE__);
            }
        }
        if (decoder.tp_hook != nullptr) {
            auto executed = decoder.tp_hook->on_execute(batch, options.data());
            if (!executed.ok()) return executed.status();
            BatchExecutionOutput hook_output = executed.release();
            const hipError_t err =
                hipMemcpy(sampled.data(), hook_output.sampled_tokens.data<int32_t>(),
                          rows * sizeof(int32_t), hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                return Status::hip_error("eagle4 tree verify sample copy",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
        } else {
            auto submitted =
                submit_batch(*decoder.target, batch, decoder.stream, options[0]);
            if (!submitted.ok()) return submitted.status();
            PendingBatch pending_batch = submitted.release();
            BatchExecutionOutput output;
            {
                auto completed = complete_batch(*decoder.target,
                                                std::move(pending_batch), decoder.stream);
                if (!completed.ok()) return completed.status();
                output = completed.release();
            }
            const hipError_t err = hipMemcpy(sampled.data(),
                                              output.sampled_tokens.data<int32_t>(),
                                              rows * sizeof(int32_t),
                                              hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                return Status::hip_error("eagle4 tree verify sample copy",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
        if (want_timing) {
            const hipError_t e1 = hipEventRecord(decoder.verify_stop_event, decoder.stream);
            if (e1 != hipSuccess) {
                return Status::hip_error("eagle4 tree verify stop event",
                                         hipGetErrorString(e1), __FILE__, __LINE__);
            }
            float elapsed = 0.0f;
            if (hipEventElapsedTime(&elapsed, decoder.verify_start_event,
                                    decoder.verify_stop_event) == hipSuccess) {
                tm->verify_gpu_ms += static_cast<double>(elapsed);
            }
        }
    }


    return eagle4_tree_finish_round(decoder, state, sampled.data(), 0u, out);
}

}  // namespace

Result<DFlash2SpecIterationOutput> dflash2_spec_step(
    DFlash2SpecDecoder& decoder,
    int32_t pending_token,
    uint32_t remaining_tokens) {
    if (!decoder.initialized) {
        return Status::invalid_state("dflash2_spec_step: not initialized", __FILE__, __LINE__);
    }
    const bool eagle = decoder.eagle != nullptr;
    if (decoder.target == nullptr || decoder.sequence == nullptr || decoder.gdn_ranks.empty() ||
        (eagle ? decoder.eagle_context == nullptr
               : (decoder.draft == nullptr || decoder.context == nullptr))) {
        return Status::invalid_state("dflash2_spec_step: state pointers are null", __FILE__,
                                     __LINE__);
    }
    if (remaining_tokens == 0u) {
        return Status::invalid_argument("dflash2_spec_step: remaining_tokens is zero", __FILE__,
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
    if (eagle && decoder.config.eagle4_tree_width >= 2u) {
        Status tree_st = dflash2_tree_round(decoder, pending_token, remaining_tokens,
                                            position, out);
        if (!tree_st.ok()) return tree_st;
        return out;
    }
    const uint32_t room = decoder.sequence->max_seq_len - position;
    uint32_t room_max = 0u;
    if (remaining_tokens > 1u && room >= 2u) {
        room_max = std::min(remaining_tokens - 1u, room - 1u);
    }
    const uint32_t dflash_k = std::min(decoder.config.num_drafts, room_max);

    if (dflash_k == 0u) {
        Status st = run_single_target(decoder, pending_token, out);
        if (!st.ok()) return st;
        if (is_stop_token(decoder.config.eos_tokens, out.pending_token)) out.finished = true;
        return out;
    }

    const bool bridge = decoder.device_token_bridge &&
                        decoder.verify_token_ids_device != nullptr &&
                        decoder.decision_staging_device != nullptr;
    const bool want_timing = tm != nullptr;
    int32_t* proposal_device = eagle ? decoder.eagle_draft_tokens_device
                                     : decoder.draft->proposal_tokens.data<int32_t>();
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
        Status st = Status::make_ok();
        if (eagle) {
            if (!decoder.eagle_root_ready) {
                return Status::invalid_state(
                    "dflash2_spec_step: eagle4 root state is not ready", __FILE__, __LINE__);
            }
            st = eagle4::eagle4_draft_steps_tp(*decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks, position,
                                            dflash_k, proposal_device, decoder.stream);
        } else {
            st = dflash2::dflash2_propose_cached(
                *decoder.draft, *decoder.context, pending_token, dflash_k, proposal_device,
                decoder.stream);
        }
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
    const bool need_host_seed = decoder.host_proposal_visibility;
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

    out.num_dflash_drafts = dflash_k;
    const uint32_t total_k = dflash_k;
    out.num_drafts = total_k;


    {
        ScopedTimer t(tm != nullptr ? &tm->gdn_snapshot_ms : nullptr);
        for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
            auto scope = ps::gpu::ScopedDevice::create(g.device);
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
    {
        ScopedTimer t(want_timing ? &tm->verify_ms : nullptr);
        ScheduledBatch batch;
        if (bridge) {
            batch = make_verify_batch(
                *decoder.sequence, requests, decoder.verify_token_ids_device, total_k + 1u,
                position, total_k + 1u, true, TokenIdsLocation::Device,
                &decoder.sampling, round_sample_index);
        } else {
            verify_host[0] = pending_token;
            for (uint32_t i = 0; i < total_k; ++i) verify_host[i + 1u] = drafts[i];
            batch = make_verify_batch(
                *decoder.sequence, requests, verify_host.data(), total_k + 1u, position,
                total_k + 1u, true, TokenIdsLocation::Host, &decoder.sampling,
                round_sample_index);
        }
        std::vector<ExecuteBatchOptions> options = build_gdn_options(decoder, total_k);
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
                    static_cast<std::size_t>(total_k + 1u) * sizeof(int32_t),
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
                sampled_device, static_cast<std::size_t>(total_k + 1u) * sizeof(int32_t),
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
            for (uint32_t i = 0; i <= total_k; ++i) {
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
                    static_cast<std::size_t>(total_k + 1u) * sizeof(int32_t),
                    hipMemcpyDeviceToHost);
                if (err != hipSuccess) {
                    return Status::hip_error("dflash2 verify sample copy",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
        }
        }
    }

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

    if (eagle && std::getenv("PHASESHIFT_EAGLE4_DEBUG_ROUND") != nullptr) {
        std::fprintf(stderr, "EAGLE4_ROUND pos=%u drafts=", position);
        for (uint32_t i = 0; i < total_k; ++i) std::fprintf(stderr, "%d,", drafts[i]);
        std::fprintf(stderr, " sampled=");
        for (uint32_t i = 0; i <= total_k; ++i) std::fprintf(stderr, "%d,", sampled[i]);
        std::fprintf(stderr, "\n");
    }



    SpecVerifyResult result;
    if (sampled[0] < 0) {
        return Status::invalid_state("verify produced no sampled token", __FILE__,
                                     __LINE__);
    }
    Status accept = spec_greedy_accept(drafts.data(), total_k, sampled.data(), true, result);
    if (!accept.ok()) return accept;
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

    if (accepted == total_k) {
        ScopedTimer t(tm != nullptr ? &tm->dflash_commit_ms : nullptr);
        Status st = append_committed(decoder, position, total_k + 1u, sampled.data());
        if (!st.ok()) return st;
        const uint32_t required = ceil_div_u32(decoder.sequence->position, page_tokens);
        st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
        if (!st.ok()) return st;
        if (tm != nullptr) ++tm->full_accepts;
    } else {
        {
            ScopedTimer t(tm != nullptr ? &tm->gdn_restore_ms : nullptr);
            for (DFlash2GdnRankState& g : decoder.gdn_ranks) {
                auto scope = ps::gpu::ScopedDevice::create(g.device);
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
                auto scope = ps::gpu::ScopedDevice::create(g.device);
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
            Status st = append_committed(decoder, position, accepted + 1u, sampled.data());
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
        tm->verify_rows_total += total_k + 1u;
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
        if (is_stop_token(decoder.config.eos_tokens, out.emitted[i])) {
            out.emitted_count = i + 1u;
            out.finished = true;
            break;
        }
    }
    if (is_stop_token(decoder.config.eos_tokens, out.pending_token)) out.finished = true;
    commit_history_tokens(decoder, out.emitted.data(), out.emitted_count);
    return out;
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
    }
    return options;
}

std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> collect_batch_target_taps(
    const DFlash2SpecDecoder& decoder) {
    std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps{};
    for (uint32_t t = 0; t < ps::kernel::kDFlash2TargetTaps; ++t) {
        taps[t] = decoder.target->dflash_target_hidden[t].data<bf16_t>();
    }
    return taps;
}

std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> batch_target_taps(
    const DFlash2SpecDecoder& decoder,
    uint32_t row_begin) {
    std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps =
        collect_batch_target_taps(decoder);
    for (uint32_t t = 0; t < ps::kernel::kDFlash2TargetTaps; ++t) {
        taps[t] += static_cast<std::size_t>(row_begin) * decoder.hidden_size;
    }
    return taps;
}


}  // namespace


Result<DFlash2SpecBatchOutput> eagle4_tree_spec_step_batch(
    DFlash2SpecBatchContext& context,
    std::span<const DFlash2SpecBatchSlot> slots) {
    if (!context.initialized) {
        return Status::invalid_state("eagle4 tree batch context is not initialized",
                                     __FILE__, __LINE__);
    }
    if (slots.empty()) {
        return Status::invalid_argument("eagle4 tree batch requires a slot", __FILE__,
                                        __LINE__);
    }
    const std::size_t ctx_ranks = context.tree_rope_positions_device.size();
    if (ctx_ranks == 0u) {
        return Status::invalid_state("eagle4 tree batch has no rank buffers", __FILE__,
                                     __LINE__);
    }

    struct TreeSlotPlan {
        DFlash2SpecDecoder* decoder = nullptr;
        int32_t anchor = 0;
        uint32_t remaining = 0;
        uint32_t position = 0;
        uint64_t sample_index = 0;
        uint32_t row_begin = 0;
        uint32_t rows = 0;
        bool done = false;
        Eagle4TreeRoundState state;
        DFlash2SpecIterationOutput out{};
    };

    const uint32_t total_cap = std::min(context.max_rows, kDFlash2SpecMaxVerifyRows);
    std::vector<TreeSlotPlan> plans;
    plans.reserve(slots.size());
    uint32_t row_cursor = 0u;
    for (const DFlash2SpecBatchSlot& slot : slots) {
        DFlash2SpecDecoder* decoder = slot.decoder;
        if (decoder == nullptr || !decoder->initialized || decoder->sequence == nullptr ||
            decoder->target == nullptr || decoder->eagle == nullptr ||
            decoder->eagle_context == nullptr || decoder->gdn_ranks.empty()) {
            return Status::invalid_argument("eagle4 tree batch slot has an invalid decoder",
                                            __FILE__, __LINE__);
        }
        const uint32_t position = decoder->sequence->position;
        TreeSlotPlan plan;
        plan.decoder = decoder;
        plan.anchor = slot.anchor_token;
        plan.remaining = slot.remaining_tokens;
        plan.position = position;
        const uint32_t node_cap =
            row_cursor < total_cap ? (total_cap - row_cursor) : 1u;
        const Status prep = eagle4_tree_prepare_round(*decoder, plan.anchor,
                                                      plan.remaining, position, node_cap,
                                                      plan.out, plan.state);
        if (!prep.ok()) return prep;
        plan.done = plan.state.done;
        if (!plan.done) {
            plan.rows = plan.state.rows;
            plan.sample_index = decoder->sample_index;
            decoder->sample_index += plan.rows;
            plan.row_begin = row_cursor;
            row_cursor += plan.rows;
            if (row_cursor > total_cap) {
                return Status::invalid_argument("eagle4 tree batch rows exceed cap",
                                                __FILE__, __LINE__);
            }
        }
        plans.push_back(std::move(plan));
    }
    const uint32_t total_rows = row_cursor;

    for (TreeSlotPlan& plan : plans) {
        if (plan.done) continue;
        DFlash2SpecDecoder& decoder = *plan.decoder;
        for (DFlash2GdnRankState& state : decoder.gdn_ranks) {
            auto scope = ps::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            const Status st = spec_gdn_snapshot(
                *state.pool, decoder.sequence->slot, state.conv_snapshot,
                state.rec_snapshot, state.conv_bytes, state.rec_bytes, state.stream);
            if (!st.ok()) return st;
            if (state.commit_request != nullptr) {
                ::ps::runtime::DeviceRequestDescriptor req{};
                req.request_handle.slot = decoder.sequence->slot;
                req.row_begin = plan.row_begin;
                req.row_count = 0u;
                const hipError_t err = hipMemcpy(
                    state.commit_request, &req, sizeof(req), hipMemcpyHostToDevice);
                if (err != hipSuccess) {
                    return Status::hip_error("eagle4 tree batch commit descriptor",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
        }
    }

    if (total_rows > 0u) {
        std::vector<uint32_t>& rope_h = context.tree_rope_positions_host;
        std::vector<int32_t>& parent_h = context.tree_row_parent_host;
        std::vector<int32_t>& off_h = context.tree_anc_offsets_host;
        std::vector<uint32_t>& pos_h = context.tree_anc_positions_host;
        std::vector<uint32_t>& commit_h = context.tree_committed_end_host;
        uint32_t anc_cursor = 0u;
        for (TreeSlotPlan& plan : plans) {
            if (plan.done) continue;
            const uint32_t rb = plan.row_begin;
            const uint32_t rows = plan.rows;
            if (static_cast<std::size_t>(rb + rows) > rope_h.size() ||
                static_cast<std::size_t>(anc_cursor + plan.state.anc_positions.size()) >
                    pos_h.size()) {
                return Status::invalid_argument("eagle4 tree batch staging overflow",
                                                __FILE__, __LINE__);
            }
            for (uint32_t j = 0; j < rows; ++j) {
                rope_h[rb + j] = plan.state.rope_positions[j];
                parent_h[rb + j] = plan.state.row_parent[j];
                commit_h[rb + j] = plan.position;
            }
            for (uint32_t j = 0; j <= rows; ++j) {
                off_h[rb + j] =
                    static_cast<int32_t>(anc_cursor) + plan.state.anc_offsets[j];
            }
            for (uint32_t j = 0; j < plan.state.anc_positions.size(); ++j) {
                pos_h[anc_cursor + j] = plan.state.anc_positions[j];
            }
            anc_cursor += static_cast<uint32_t>(plan.state.anc_positions.size());
        }
        DFlash2SpecDecoder& leader = *plans.front().decoder;
        if (ctx_ranks != leader.gdn_ranks.size()) {
            return Status::invalid_argument("eagle4 tree batch rank count mismatch",
                                            __FILE__, __LINE__);
        }
        for (std::size_t r = 0; r < ctx_ranks; ++r) {
            DFlash2GdnRankState& rank = leader.gdn_ranks[r];
            auto scope = ps::gpu::ScopedDevice::create(rank.device);
            if (!scope.ok()) return scope.status();
            const hipError_t up1 = hipMemcpyAsync(
                context.tree_rope_positions_device[r], rope_h.data(),
                static_cast<std::size_t>(total_rows) * sizeof(uint32_t),
                hipMemcpyHostToDevice, rank.stream);
            const hipError_t up2 = hipMemcpyAsync(
                context.tree_row_parent_device[r], parent_h.data(),
                static_cast<std::size_t>(total_rows) * sizeof(int32_t),
                hipMemcpyHostToDevice, rank.stream);
            const hipError_t up3 = hipMemcpyAsync(
                context.tree_anc_offsets_device[r], off_h.data(),
                (static_cast<std::size_t>(total_rows) + 1u) * sizeof(int32_t),
                hipMemcpyHostToDevice, rank.stream);
            const hipError_t up4 = hipMemcpyAsync(
                context.tree_anc_positions_device[r], pos_h.data(),
                static_cast<std::size_t>(anc_cursor) * sizeof(uint32_t),
                hipMemcpyHostToDevice, rank.stream);
            const hipError_t up5 = hipMemcpyAsync(
                context.tree_committed_end_rows_device[r], commit_h.data(),
                static_cast<std::size_t>(total_rows) * sizeof(uint32_t),
                hipMemcpyHostToDevice, rank.stream);
            if (up1 != hipSuccess || up2 != hipSuccess || up3 != hipSuccess ||
                up4 != hipSuccess || up5 != hipSuccess) {
                return Status::hip_error("eagle4 tree batch staging upload",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
    }

    std::vector<int32_t> token_ids;
    token_ids.reserve(total_rows);
    std::vector<ScheduledRequest> requests;
    requests.reserve(plans.size());
    for (TreeSlotPlan& plan : plans) {
        if (plan.done) continue;
        for (uint32_t j = 0; j < plan.rows; ++j) {
            token_ids.push_back(plan.state.verify_inputs[j]);
        }
        ScheduledRequest req;
        req.sequence = plan.decoder->sequence;
        req.handle = plan.decoder->sequence->request_handle();
        req.execution_class = ::ps::runtime::ExecutionClass::SPEC_VERIFY;
        req.token_begin = plan.row_begin;
        req.num_tokens = plan.rows;
        req.prefix_tokens = plan.position;
        req.compute_logits = true;
        req.sample = true;
        req.num_output_rows = plan.rows;
        req.sampling = plan.decoder->sampling;
        req.sampling_index = plan.sample_index;
        requests.push_back(req);
    }

    ScheduledBatch batch;
    batch.token_ids = token_ids.data();
    batch.token_ids_location = TokenIdsLocation::Host;
    batch.requests = requests;
    batch.num_tokens = total_rows;
    batch.num_requests = static_cast<uint32_t>(requests.size());
    batch.num_decode_requests = 0u;
    batch.num_verify_requests = static_cast<uint32_t>(requests.size());
    batch.num_prefill_requests = 0u;
    batch.speculative_verify = true;

    DFlash2SpecBatchOutput out;
    out.slots.resize(plans.size());
    out.total_rows = total_rows;
    out.rounds = 1u;

    if (total_rows > 0u) {
        DFlash2SpecDecoder& leader = *plans.front().decoder;
        std::vector<ExecuteBatchOptions> options =
            build_batch_gdn_options(context, total_rows);
        uint32_t max_position = 0u;
        for (TreeSlotPlan& plan : plans) {
            if (plan.done) continue;
            if (plan.position > max_position) max_position = plan.position;
        }
        if (options.size() != ctx_ranks) {
            return Status::invalid_argument("eagle4 tree batch option rank mismatch",
                                            __FILE__, __LINE__);
        }
        for (std::size_t r = 0; r < options.size(); ++r) {
            ExecuteBatchOptions& o = options[r];
            o.tree_anc_offsets = reinterpret_cast<const uint32_t*>(
                context.tree_anc_offsets_device[r]);
            o.tree_anc_positions = context.tree_anc_positions_device[r];
            o.tree_committed_end = max_position;
            o.tree_committed_end_rows = context.tree_committed_end_rows_device[r];
            o.rope_positions = context.tree_rope_positions_device[r];
            o.tree_row_parent = context.tree_row_parent_device[r];
        }
        const bool batch_timing = leader.timing != nullptr;
        bool verify_events_recorded = false;
        const auto round_t0 = std::chrono::steady_clock::now();
        if (batch_timing) {
            const hipError_t start_err =
                hipEventRecord(leader.verify_start_event, leader.stream);
            if (start_err != hipSuccess) {
                return Status::hip_error("eagle4 tree batch verify start event",
                                         hipGetErrorString(start_err), __FILE__,
                                         __LINE__);
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
                return Status::hip_error("eagle4 tree batch verify stop event",
                                         hipGetErrorString(stop_err), __FILE__, __LINE__);
            }
        }
        BatchExecutionOutput output = executed.release();
        if (output.num_outputs < total_rows) {
            return Status::invalid_state("eagle4 tree batch verify output rows mismatch",
                                         __FILE__, __LINE__);
        }
        std::vector<int32_t> sampled(total_rows);
        const hipError_t err = hipMemcpy(
            sampled.data(), output.sampled_tokens.data<int32_t>(),
            static_cast<std::size_t>(total_rows) * sizeof(int32_t),
            hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            return Status::hip_error("eagle4 tree batch verify sample copy",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        if (batch_timing) {
            double verify_gpu_ms = 0.0;
            if (verify_events_recorded) {
                float elapsed = 0.0f;
                if (hipEventElapsedTime(&elapsed, leader.verify_start_event,
                                        leader.verify_stop_event) == hipSuccess) {
                    verify_gpu_ms = static_cast<double>(elapsed);
                }
            }
            out.verify_gpu_ms = verify_gpu_ms;
            out.round_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - round_t0)
                               .count();
        }

        for (std::size_t i = 0; i < plans.size(); ++i) {
            TreeSlotPlan& plan = plans[i];
            if (plan.done) {
                out.slots[i] = plan.out;
                continue;
            }
            const Status fin = eagle4_tree_finish_round(
                *plan.decoder, plan.state, sampled.data() + plan.row_begin,
                plan.row_begin, plan.out);
            if (!fin.ok()) return fin;
            out.slots[i] = plan.out;
        }
    } else {
        for (std::size_t i = 0; i < plans.size(); ++i) {
            out.slots[i] = plans[i].out;
        }
    }
    return out;
}
Result<DFlash2SpecBatchOutput> eagle4_tree_spec_step_batch(
    DFlash2SpecBatchContext& context,
    std::span<const DFlash2SpecBatchSlot> slots);

Result<DFlash2SpecBatchOutput> eagle4_spec_step_batch(
    DFlash2SpecBatchContext& context,
    std::span<const DFlash2SpecBatchSlot> slots) {
    if (!slots.empty() && slots.front().decoder != nullptr &&
        slots.front().decoder->config.eagle4_tree_width >= 2u) {
        return eagle4_tree_spec_step_batch(context, slots);
    }
    if (!context.initialized) {
        return Status::invalid_state("eagle4 batch context is not initialized", __FILE__,
                                     __LINE__);
    }
    if (slots.empty()) {
        return Status::invalid_argument("eagle4 batch step requires a slot", __FILE__,
                                        __LINE__);
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
        bool batched = false;
    };

    std::vector<SlotPlan> plans;
    plans.reserve(slots.size());
    uint32_t row_cursor = 0u;
    for (const DFlash2SpecBatchSlot& slot : slots) {
        DFlash2SpecDecoder* decoder = slot.decoder;
        if (decoder == nullptr || !decoder->initialized || decoder->sequence == nullptr ||
            decoder->target == nullptr || decoder->eagle == nullptr ||
            decoder->eagle_context == nullptr || decoder->gdn_ranks.empty()) {
            return Status::invalid_argument("eagle4 batch slot has an invalid decoder",
                                            __FILE__, __LINE__);
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
        const uint32_t drafts = std::min(slot.num_drafts, room_max);
        SlotPlan plan;
        plan.decoder = decoder;
        plan.anchor = slot.anchor_token;
        plan.drafts = drafts;
        plan.remaining = slot.remaining_tokens;
        plan.position = position;
        plan.sample_index = decoder->sample_index;
        plan.batched = drafts > 0u;
        plan.total_rows = drafts + 1u;
        if (plan.batched) {
            plan.row_begin = row_cursor;
            row_cursor += plan.total_rows;
        }
        plans.push_back(plan);
    }

    const uint32_t total_rows = row_cursor;
    if (total_rows > context.max_rows) {
        return Status::invalid_argument("eagle4 batch rows exceed context capacity", __FILE__,
                                        __LINE__);
    }
    if (total_rows > kDFlash2SpecMaxVerifyRows) {
        return Status::invalid_argument("eagle4 batch rows exceed verify row cap", __FILE__,
                                        __LINE__);
    }

    std::vector<std::array<int32_t, 16>> drafts(plans.size());
    bool batch_timing = false;
    for (const SlotPlan& plan : plans) {
        if (plan.decoder->timing != nullptr) {
            batch_timing = true;
            break;
        }
    }

    const auto round_t0 = std::chrono::steady_clock::now();
    bool verify_events_recorded = false;
    eagle4::Eagle4Executor* draft_executor = nullptr;
    DFlash2SpecDecoder* draft_leader = nullptr;
    hipStream_t draft_stream = nullptr;
    for (std::size_t i = 0; i < plans.size(); ++i) {
        SlotPlan& plan = plans[i];
        if (!plan.batched) continue;
        DFlash2SpecDecoder& decoder = *plan.decoder;
        if (!decoder.eagle_root_ready) {
            return Status::invalid_state("eagle4 batch slot root not ready", __FILE__,
                                         __LINE__);
        }
        if (draft_executor == nullptr) {
            draft_executor = decoder.eagle;
            draft_leader = &decoder;
            draft_stream = decoder.stream;
        } else if (decoder.eagle != draft_executor || decoder.stream != draft_stream) {
            return Status::unsupported(
                "eagle4 batch draft requires a shared drafter executor and stream",
                __FILE__, __LINE__);
        }
        if (batch_timing) {
            const hipError_t start_err =
                hipEventRecord(decoder.draft_start_event, decoder.stream);
            if (start_err != hipSuccess) {
                return Status::hip_error("eagle4 batch draft start event",
                                         hipGetErrorString(start_err), __FILE__, __LINE__);
            }
        }
        const Status st = eagle4::eagle4_draft_steps_tp(
            *decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks, plan.position, plan.drafts,
            decoder.eagle_draft_tokens_device, decoder.stream);
        if (!st.ok()) return st;
        if (batch_timing) {
            const hipError_t stop_err =
                hipEventRecord(decoder.draft_stop_event, decoder.stream);
            if (stop_err != hipSuccess) {
                return Status::hip_error("eagle4 batch draft stop event",
                                         hipGetErrorString(stop_err), __FILE__, __LINE__);
            }
        }
        if (hipMemcpy(drafts[i].data(), decoder.eagle_draft_tokens_device,
                      static_cast<std::size_t>(plan.drafts) * sizeof(int32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            return Status::hip_error("eagle4 batch draft copy",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }

    for (std::size_t i = 0; i < plans.size(); ++i) {
        SlotPlan& plan = plans[i];
        if (!plan.batched) continue;
        DFlash2SpecDecoder& decoder = *plan.decoder;
        for (DFlash2GdnRankState& state : decoder.gdn_ranks) {
            auto scope = ps::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            const Status st = spec_gdn_snapshot(
                *state.pool, decoder.sequence->slot, state.conv_snapshot,
                state.rec_snapshot, state.conv_bytes, state.rec_bytes, state.stream);
            if (!st.ok()) return st;
            if (state.commit_request != nullptr) {
                ::ps::runtime::DeviceRequestDescriptor req{};
                req.request_handle.slot = decoder.sequence->slot;
                req.row_begin = plan.row_begin;
                req.row_count = 0u;
                const hipError_t err = hipMemcpy(
                    state.commit_request, &req, sizeof(req), hipMemcpyHostToDevice);
                if (err != hipSuccess) {
                    return Status::hip_error("eagle4 batch commit descriptor",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
        }
    }

    std::vector<int32_t> token_ids;
    token_ids.reserve(total_rows);
    std::vector<ScheduledRequest> requests;
    requests.reserve(plans.size());
    uint32_t verify_request_count = 0u;
    for (std::size_t i = 0; i < plans.size(); ++i) {
        SlotPlan& plan = plans[i];
        if (!plan.batched) continue;
        DFlash2SpecDecoder& decoder = *plan.decoder;
        token_ids.push_back(plan.anchor);
        for (uint32_t j = 0; j < plan.drafts; ++j) {
            token_ids.push_back(drafts[i][j]);
        }
        ScheduledRequest req;
        req.sequence = decoder.sequence;
        req.handle = decoder.sequence->request_handle();
        req.execution_class = ::ps::runtime::ExecutionClass::SPEC_VERIFY;
        ++verify_request_count;
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
    batch.token_ids = token_ids.data();
    batch.token_ids_location = TokenIdsLocation::Host;
    batch.requests = requests;
    batch.num_tokens = total_rows;
    batch.num_requests = static_cast<uint32_t>(requests.size());
    batch.num_decode_requests = 0u;
    batch.num_verify_requests = static_cast<uint32_t>(requests.size());
    batch.num_prefill_requests = 0u;
    batch.speculative_verify = true;

    const bool mixed_verify_batch = verify_request_count != requests.size();
    if (mixed_verify_batch && verify_request_count > 1u) {
        return Status::unsupported(
            "eagle4 mixed verify batch supports a single verify request", __FILE__,
            __LINE__);
    }

    std::vector<int32_t> sampled;
    if (total_rows > 0u) {
        DFlash2SpecDecoder& leader = *plans.front().decoder;
        std::vector<ExecuteBatchOptions> options = build_batch_gdn_options(context, total_rows);
        if (batch_timing) {
            const hipError_t start_err =
                hipEventRecord(leader.verify_start_event, leader.stream);
            if (start_err != hipSuccess) {
                return Status::hip_error("eagle4 batch verify start event",
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
                return Status::hip_error("eagle4 batch verify stop event",
                                         hipGetErrorString(stop_err), __FILE__, __LINE__);
            }
        }
        BatchExecutionOutput output = executed.release();
        if (output.num_outputs < total_rows) {
            return Status::invalid_state("eagle4 batch verify output rows mismatch",
                                         __FILE__, __LINE__);
        }
        sampled.resize(total_rows);
        const hipError_t err = hipMemcpy(
            sampled.data(), output.sampled_tokens.data<int32_t>(),
            static_cast<std::size_t>(total_rows) * sizeof(int32_t),
            hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            return Status::hip_error("eagle4 batch verify sample copy",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
    }

    DFlash2SpecBatchOutput out;
    out.slots.resize(plans.size());
    out.total_rows = total_rows;
    out.rounds = 1u;
    if (batch_timing) {
        if (draft_leader != nullptr) {
            float elapsed = 0.0f;
            if (hipEventElapsedTime(&elapsed, draft_leader->draft_start_event,
                                    draft_leader->draft_stop_event) == hipSuccess) {
                out.draft_gpu_ms = static_cast<double>(elapsed);
            }
        }
        if (verify_events_recorded) {
            float elapsed = 0.0f;
            if (hipEventElapsedTime(&elapsed, plans.front().decoder->verify_start_event,
                                    plans.front().decoder->verify_stop_event) == hipSuccess) {
                out.verify_gpu_ms = static_cast<double>(elapsed);
            }
        }
        out.round_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - round_t0)
                           .count();
    }

    for (std::size_t i = 0; i < plans.size(); ++i) {
        SlotPlan& plan = plans[i];
        DFlash2SpecDecoder& decoder = *plan.decoder;
        DFlash2SpecIterationOutput& result_out = out.slots[i];

        if (!plan.batched) {
            const Status st = run_single_target(decoder, plan.anchor, result_out);
            if (!st.ok()) return st;
            if (is_stop_token(decoder.config.eos_tokens, result_out.pending_token)) {
                result_out.finished = true;
            }
            continue;
        }

        SpecVerifyResult verify;
        const Status accept_st = spec_greedy_accept(
            drafts[i].data(), plan.drafts, sampled.data() + plan.row_begin, true, verify);
        if (!accept_st.ok()) return accept_st;
        const uint32_t accepted = verify.num_accepted_drafts;
        result_out.num_drafts = plan.drafts;
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

        const uint32_t page_tokens = decoder.sequence->kv_pool()->page_tokens();
        if (page_tokens == 0u) {
            return Status::invalid_state("eagle4 batch step page_tokens is zero", __FILE__,
                                         __LINE__);
        }

        const auto taps = collect_batch_target_taps(decoder);
        if (accepted == plan.drafts) {
            std::array<int32_t, 16> extend_tokens{};
            for (uint32_t j = 0; j < plan.total_rows; ++j) {
                extend_tokens[j] = sampled[plan.row_begin + j];
            }
            Status st = eagle4::eagle4_extend_rows_tp(
                *decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks, taps.data(), 3u,
                plan.row_begin, extend_tokens.data(), plan.total_rows, plan.position,
                decoder.eagle_candidate_topk, decoder.eagle_draft_tokens_device,
                decoder.stream);
            if (!st.ok()) return st;
            decoder.eagle_root_ready = true;
            const uint32_t required =
                ceil_div_u32(decoder.sequence->position, page_tokens);
            st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
            if (!st.ok()) return st;
        } else {
            for (std::size_t r = 0; r < decoder.gdn_ranks.size(); ++r) {
                DFlash2GdnRankState& state = decoder.gdn_ranks[r];
                auto scope = ps::gpu::ScopedDevice::create(state.device);
                if (!scope.ok()) return scope.status();
                Status st = spec_gdn_restore(
                    *state.pool, decoder.sequence->slot, state.conv_snapshot,
                    state.rec_snapshot, state.conv_bytes, state.rec_bytes, state.stream);
                if (!st.ok()) return st;
                st = restore_gdn_spec_conv(
                    *state.pool, decoder.sequence->slot, context.history[r],
                    (mixed_verify_batch ? 0u : plan.row_begin) + accepted, state.stream);
                if (!st.ok()) return st;
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
            decoder.sequence->position = plan.position + accepted + 1u;
            std::array<int32_t, 16> extend_tokens{};
            for (uint32_t j = 0; j < accepted + 1u; ++j) {
                extend_tokens[j] = sampled[plan.row_begin + j];
            }
            Status st = eagle4::eagle4_extend_rows_tp(
                *decoder.eagle, *decoder.eagle_context, &decoder.eagle_ranks, taps.data(), 3u,
                plan.row_begin, extend_tokens.data(), accepted + 1u, plan.position,
                decoder.eagle_candidate_topk, decoder.eagle_draft_tokens_device,
                decoder.stream);
            if (!st.ok()) return st;
            decoder.eagle_root_ready = true;
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
        }

        for (uint32_t j = 0; j < result_out.emitted_count; ++j) {
            if (is_stop_token(decoder.config.eos_tokens, result_out.emitted[j])) {
                result_out.emitted_count = j + 1u;
                result_out.finished = true;
                break;
            }
        }
        if (is_stop_token(decoder.config.eos_tokens, result_out.pending_token)) {
            result_out.finished = true;
        }
        commit_history_tokens(decoder, result_out.emitted.data(), result_out.emitted_count);
    }

    return out;
}

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
        bool batched = false;
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
        const uint32_t drafts = std::min(slot.num_drafts, room_max);
        SlotPlan plan;
        plan.decoder = decoder;
        plan.anchor = slot.anchor_token;
        plan.drafts = drafts;
        plan.remaining = slot.remaining_tokens;
        plan.position = position;
        plan.sample_index = decoder->sample_index;
        plan.batched = drafts > 0u;
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
            auto scope = ps::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            const Status st = spec_gdn_snapshot(
                *state.pool, decoder.sequence->slot, state.conv_snapshot,
                state.rec_snapshot, state.conv_bytes, state.rec_bytes, state.stream);
            if (!st.ok()) return st;
            if (state.commit_request != nullptr) {
                ::ps::runtime::DeviceRequestDescriptor req{};
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
    uint32_t verify_request_count = 0u;
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
        req.execution_class = ::ps::runtime::ExecutionClass::SPEC_VERIFY;
        ++verify_request_count;
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

    const bool mixed_verify_batch = verify_request_count != requests.size();
    if (mixed_verify_batch && verify_request_count > 1u) {
        return Status::unsupported(
            "mixed verify batch supports a single verify request", __FILE__, __LINE__);
    }

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

        if (!plan.batched) {
            const Status st = run_single_target(decoder, plan.anchor, result_out);
            if (!st.ok()) return st;
            if (is_stop_token(decoder.config.eos_tokens, result_out.pending_token)) {
                result_out.finished = true;
            }
            continue;
        }

        SpecVerifyResult verify;
        const Status accept_st = spec_greedy_accept(
            drafts[i].data(), plan.drafts, sampled.data() + plan.row_begin, true, verify);
        if (!accept_st.ok()) return accept_st;
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

        const uint32_t page_tokens = decoder.sequence->kv_pool()->page_tokens();
        if (page_tokens == 0u) {
            return Status::invalid_state("batch step page_tokens is zero", __FILE__, __LINE__);
        }

        if (accepted == plan.drafts) {
            const auto taps = batch_target_taps(decoder, plan.row_begin);
            Status st = dflash2::dflash2_append_target_taps(
                *decoder.draft, *decoder.context, taps.data(),
                ps::kernel::kDFlash2TargetTaps, plan.total_rows, plan.position,
                decoder.stream);
            if (!st.ok()) return st;
            const uint32_t required =
                ceil_div_u32(decoder.sequence->position, page_tokens);
            st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
            if (!st.ok()) return st;
        } else {
            for (std::size_t r = 0; r < decoder.gdn_ranks.size(); ++r) {
                DFlash2GdnRankState& state = decoder.gdn_ranks[r];
                auto scope = ps::gpu::ScopedDevice::create(state.device);
                if (!scope.ok()) return scope.status();
                Status st = spec_gdn_restore(
                    *state.pool, decoder.sequence->slot, state.conv_snapshot,
                    state.rec_snapshot, state.conv_bytes, state.rec_bytes, state.stream);
                if (!st.ok()) return st;
                st = restore_gdn_spec_conv(
                    *state.pool, decoder.sequence->slot, context.history[r],
                    (mixed_verify_batch ? 0u : plan.row_begin) + accepted,
                    state.stream);
                if (!st.ok()) return st;
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
            decoder.sequence->position = plan.position + accepted + 1u;
            const auto taps = batch_target_taps(decoder, plan.row_begin);
            Status st = dflash2::dflash2_append_target_taps(
                *decoder.draft, *decoder.context, taps.data(),
                ps::kernel::kDFlash2TargetTaps, accepted + 1u, plan.position,
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
            if (is_stop_token(decoder.config.eos_tokens, result_out.emitted[j])) {
                result_out.emitted_count = j + 1u;
                result_out.finished = true;
                break;
            }
        }
        if (is_stop_token(decoder.config.eos_tokens, result_out.pending_token)) {
            result_out.finished = true;
        }
        commit_history_tokens(decoder, result_out.emitted.data(), result_out.emitted_count);
    }

    return out;
}

}  // namespace ps::qwen35::runtime
