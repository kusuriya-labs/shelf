#include <shelf/runtime/tp/pipeline_parallel.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/size_format.h>
#include <shelf/runtime/gdn_state_layout.h>
#include <shelf/models/pipeline_stage_model.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/runtime/executor_footprint.h>
#include <shelf/weights/weight_bytes.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/runtime/request_handle.h>

#include <algorithm>
#include <cstdio>
#include <span>
#include <sstream>
#include <utility>

namespace shelf {
namespace runtime {

namespace {

std::uint32_t ceil_div(std::uint32_t a, std::uint32_t b) {
    return (a + b - 1u) / b;
}

std::uint32_t count_attention_layers(const shelf::qwen_hybrid::QwenHybridConfig& tc,
                                     const PipelineStage& stage) {
    return stage_attention_layers(tc, stage);
}

}  // namespace

PipelineStageRuntime::~PipelineStageRuntime() noexcept {
    if (device_id_ >= 0) {
        auto scope = shelf::gpu::ScopedDevice::create(device_id_);
        (void)scope;
    }
    if (executor_ready_) {
        executor_shutdown(executor_);
        executor_ready_ = false;
    }
}

Result<std::unique_ptr<PipelineStageRuntime>> PipelineStageRuntime::create(
    const PipelineStageRuntimeConfig& config) {
    if (!config.stage.enabled ||
        config.stage.layers.layer_lo >= config.stage.layers.layer_hi) {
        return Status::invalid_argument("invalid pipeline stage", __FILE__, __LINE__);
    }
    if (config.max_scheduled_requests == 0 || config.max_seq_len == 0) {
        return Status::invalid_argument(
            "max_scheduled_requests and max_seq_len must be > 0", __FILE__, __LINE__);
    }

    auto scope = shelf::gpu::ScopedDevice::create(config.device_id);
    if (!scope.ok()) return scope.status();

    std::unique_ptr<PipelineStageRuntime> stage(new PipelineStageRuntime());
    stage->device_id_ = config.device_id;
    stage->stage_ = config.stage;

    if (!config.tp.devices.empty()) {
        TpCoordinatorConfig tp_config = config.tp;
        tp_config.pipeline_stage = config.stage;
        if (tp_config.build_rank_model == nullptr) {
            Status selected = shelf::models::select_tp_rank_model(tp_config);
            if (!selected.ok()) return selected;
        }
        auto tp_result = TpCoordinator::create(tp_config);
        if (!tp_result.ok()) return tp_result.status();
        stage->tp_ = tp_result.release();
        stage->device_id_ = stage->tp_->rank(0).device_id();
        stage->stream_ = stage->tp_->rank(0).stream();
        return stage;
    }

    if (config.arena == nullptr || config.stream == nullptr) {
        return Status::invalid_argument(
            "pipeline stage requires a borrowed arena and stream", __FILE__, __LINE__);
    }
    stage->arena_ = config.arena;
    stage->stream_ = config.stream;

    if (config.build_stage_model == nullptr) {
        return Status::invalid_argument("pipeline stage runtime needs a stage model builder",
                                        __FILE__, __LINE__);
    }
    auto model_result = config.build_stage_model(config.model_dir, config.stage,
                                                 *stage->arena_, stage->stream_,
                                                 config.load_options);
    if (!model_result.ok()) return model_result.status();
    stage->model_ = std::move(model_result.value());

    const shelf::qwen_hybrid::QwenHybridConfig& tc = stage->model_->hybrid_config();
    const std::uint32_t max_blocks_per_sequence =
        ceil_div(config.max_seq_len + 1u, config.page_tokens) + 1u;
    const std::uint32_t attention_layers = count_attention_layers(tc, config.stage);
    if (attention_layers == 0) {
        return Status::unsupported(
            "pipeline stage has no full-attention layer", __FILE__, __LINE__);
    }
    const std::uint32_t kv_heads = static_cast<std::uint32_t>(tc.num_key_value_heads);
    const std::uint32_t head_dim = static_cast<std::uint32_t>(tc.attention_head_dim);

    const std::size_t kv_page_bytes = PagedKVPool::page_bytes(
        config.page_tokens, attention_layers, kv_heads, head_dim, config.kv_cache_dtype,
        config.octave_kv_format);
    if (kv_page_bytes == 0u) {
        return Status::invalid_state("KV page geometry is zero", __FILE__, __LINE__);
    }

    auto gdn_layout_result =
        make_gdn_state_layout_stage(tc, config.stage, 1u);
    if (!gdn_layout_result.ok()) return gdn_layout_result.status();

    auto seq_result = SequenceSlotPool::create(
        *stage->arena_, config.max_scheduled_requests, max_blocks_per_sequence);
    if (!seq_result.ok()) return seq_result.status();
    stage->seq_pool_.emplace(seq_result.release());

    auto gdn_result = GdnStatePool::create(*stage->arena_,
                                           config.max_scheduled_requests,
                                           gdn_layout_result.value());
    if (!gdn_result.ok()) return gdn_result.status();
    stage->gdn_pool_.emplace(gdn_result.release());

    const std::size_t arena_free_now =
        stage->arena_->capacity() - stage->arena_->used();
    std::size_t reserved = config.working_set_reserve_bytes;
    ExecutorConfig exec_config;
    exec_config.max_batch_rows = config.max_batch_rows;
    exec_config.max_scheduled_requests = config.max_scheduled_requests;
    exec_config.max_scheduled_output_rows = config.max_scheduled_output_rows;
    exec_config.rope_profile = config.rope_profile;
    exec_config.stage = config.stage;
    exec_config.max_context_tokens = config.max_seq_len + 1u;
    if (reserved == 0u) {
        reserved = executor_reserve_bytes(
            tc, stage->model_->token_row_ffn_size(), exec_config, 1u,
            stage->model_->uses_exl3_linear());
        reserved += stage->model_->ffn_workspace_reserve_bytes(config.max_batch_rows);
        // The KV pool is sized from what the arena leaves after this reserve, so an estimate
        // that is short of the executor's real footprint turns into a failed allocation. The
        // auto path keeps slack back; an explicit --working-set-reserve is taken as given.
        const std::size_t slack = std::max<std::size_t>(64ull << 20ull, arena_free_now / 100ull);
        reserved += slack;
        std::fprintf(stderr,
                     "MEMORY_PLAN auto pipeline stage %u..%u: working set reserve %s "
                     "(row bucket %u, incl. %s slack)\n",
                     config.stage.layers.layer_lo, config.stage.layers.layer_hi,
                     gpu::format_gib(reserved).c_str(),
                     static_cast<unsigned>(config.max_batch_rows),
                     gpu::format_gib(slack).c_str());
    }
    if (arena_free_now <= reserved) {
        return Status::insufficient_memory(
            "pipeline stage arena has " + gpu::format_gib(arena_free_now) +
                " free after weights but the runtime reserves " +
                gpu::format_gib(reserved),
            __FILE__, __LINE__);
    }

    std::uint32_t kv_pool_pages = config.kv_pool_pages;
    if (kv_pool_pages == 0u) {
        constexpr std::size_t kKvSafetyBytes = 2ull * 1024ull * 1024ull * 1024ull;
        const std::size_t kv_budget = arena_free_now > reserved + kKvSafetyBytes
            ? arena_free_now - reserved - kKvSafetyBytes
            : 0u;
        const std::uint32_t pages_fit = static_cast<std::uint32_t>(
            kv_budget / kv_page_bytes);
        if (pages_fit <= 1u) {
            return Status::insufficient_memory(
                "pipeline stage arena cannot hold any KV page", __FILE__, __LINE__);
        }
        const std::uint32_t fit_pages = pages_fit - 1u;
        const std::uint64_t single_request_tokens =
            static_cast<std::uint64_t>(config.max_seq_len) + 1u;
        const std::uint32_t required_pages =
            ceil_div(static_cast<std::uint32_t>(single_request_tokens), config.page_tokens);
        if (fit_pages < required_pages) {
            const std::uint32_t max_supportable =
                fit_pages * config.page_tokens - 1u;
            return Status::insufficient_memory(
                "pipeline stage arena fits " +
                    std::to_string(fit_pages * config.page_tokens) +
                    " KV tokens but max_seq_len " + std::to_string(config.max_seq_len) +
                    " needs " + std::to_string(single_request_tokens) +
                    " (max supportable max_seq_len " +
                    std::to_string(max_supportable) + ")",
                __FILE__, __LINE__);
        }
        kv_pool_pages = fit_pages + 1u;
    }
    auto kv_result = PagedKVPool::create(
        *stage->arena_, kv_pool_pages, config.page_tokens, attention_layers,
        kv_heads, head_dim, config.kv_cache_dtype, config.octave_kv_format);
    if (!kv_result.ok()) return kv_result.status();
    stage->kv_pool_.emplace(kv_result.release());

    const std::size_t executor_before = stage->arena_->used();
    auto exec_result = stage->model_->create_executor(
        stage->seq_pool_.value(), stage->gdn_pool_.value(), stage->kv_pool_.value(),
        *stage->arena_, exec_config);
    std::fprintf(stderr,
                 "MEMORY_PLAN pipeline stage %u..%u device=%d executor footprint: %s\n",
                 config.stage.layers.layer_lo, config.stage.layers.layer_hi,
                 config.device_id,
                 gpu::format_gib(stage->arena_->used() - executor_before).c_str());
    if (!exec_result.ok()) return exec_result.status();
    stage->executor_ = exec_result.release();
    stage->executor_ready_ = true;

    return stage;
}

std::uint32_t PipelineStageRuntime::kv_pool_pages() const {
    if (tp_) return const_cast<TpCoordinator*>(tp_.get())->rank(0).kv_pool().num_pages();
    return kv_pool_ ? kv_pool_->num_pages() : 0u;
}

const shelf::qwen_hybrid::QwenHybridConfig& PipelineStageRuntime::hybrid_config() const {
    if (tp_) return const_cast<TpCoordinator*>(tp_.get())->rank(0).hybrid_config();
    return model_->hybrid_config();
}

std::size_t PipelineStageRuntime::host_pinned_bytes() const {
    if (tp_) {
        const auto* rank_model = const_cast<TpCoordinator*>(tp_.get())->rank(0).rank_model();
        return rank_model != nullptr ? rank_model->host_pinned_bytes() : 0u;
    }
    return model_->host_pinned_bytes();
}

PipelineParallelCoordinator::~PipelineParallelCoordinator() noexcept {
    for (auto& stage : stages_) stage.reset();
    stages_.clear();
    mirrors_.clear();
    if (!devices_.empty()) {
        auto scope = shelf::gpu::ScopedDevice::create(devices_.front());
        (void)scope;
    }
    if (host_hidden_ != nullptr) {
        shelf::gpu::discard_cleanup_result(hipHostFree(host_hidden_));
        host_hidden_ = nullptr;
    }
    if (host_sampled_ != nullptr) {
        shelf::gpu::discard_cleanup_result(hipHostFree(host_sampled_));
        host_sampled_ = nullptr;
    }
}

Result<std::unique_ptr<PipelineParallelCoordinator>> PipelineParallelCoordinator::create(
    const PipelineParallelConfig& config) {
    if (config.stages.empty() || config.stages.size() != config.devices.size()) {
        return Status::invalid_argument(
            "pipeline stages and devices must be non-empty and the same size",
            __FILE__, __LINE__);
    }
    if (config.max_batch_rows == 0 || config.max_scheduled_requests == 0) {
        return Status::invalid_argument(
            "max_batch_rows and max_scheduled_requests must be > 0",
            __FILE__, __LINE__);
    }

    std::unique_ptr<PipelineParallelCoordinator> coordinator(
        new PipelineParallelCoordinator());
    const Status init = coordinator->initialize(config);
    if (!init.ok()) return init;
    return coordinator;
}

Status PipelineParallelCoordinator::initialize(const PipelineParallelConfig& config) {
    devices_ = config.devices;
    if (config.memory == nullptr) {
        return Status::invalid_argument(
            "pipeline coordinator requires a GpuMemoryManager", __FILE__, __LINE__);
    }

    std::uint32_t kv_pages = 0;
    stages_.reserve(config.stages.size());
    for (std::size_t i = 0; i < config.stages.size(); ++i) {
        auto arena = config.memory->arena_for(config.devices[i]);
        if (!arena.ok()) return arena.status();
        auto stream = config.memory->stream_for(config.devices[i]);
        if (!stream.ok()) return stream.status();
        PipelineStageRuntimeConfig stage_config;
        stage_config.model_dir = config.model_dir;
        stage_config.stage.enabled = true;
        stage_config.stage.layers = config.stages[i];
        if (i < config.stage_tp.size()) stage_config.tp = config.stage_tp[i];
        stage_config.arena = arena.value();
        stage_config.stream = stream.value();
        stage_config.tp.memory = config.memory;
        stage_config.device_id = config.devices[i];
        stage_config.max_batch_rows = config.max_batch_rows;
        stage_config.max_scheduled_requests = config.max_scheduled_requests;
        stage_config.max_scheduled_output_rows = config.max_scheduled_output_rows;
        stage_config.max_seq_len = config.max_seq_len;
        stage_config.page_tokens = config.page_tokens;
        stage_config.kv_cache_dtype = config.kv_cache_dtype;
        stage_config.octave_kv_format = config.octave_kv_format;
        stage_config.working_set_reserve_bytes = config.working_set_reserve_bytes;
        stage_config.kv_pool_pages = i == 0 ? 0u : kv_pages;
        if (!stage_config.tp.devices.empty() && i != 0 &&
            stage_config.tp.kv_pool_pages == 0u) {
            stage_config.tp.kv_pool_pages = kv_pages;
        }
        stage_config.rope_profile = config.rope_profile;
        stage_config.load_options = config.load_options;
        stage_config.build_stage_model = config.build_stage_model;
        auto stage_result = PipelineStageRuntime::create(stage_config);
        if (!stage_result.ok()) return stage_result.status();
        stages_.push_back(stage_result.release());
        if (i == 0) kv_pages = stages_.back()->kv_pool_pages();
    }

    const shelf::qwen_hybrid::QwenHybridConfig& tc =
        stages_.front()->hybrid_config();
    hidden_size_ = config.pipeline_hidden_width != 0u
        ? config.pipeline_hidden_width
        : static_cast<std::uint32_t>(tc.hidden_size);
    max_rows_ = config.max_batch_rows;
    max_outputs_ = config.max_scheduled_output_rows > config.max_scheduled_requests
        ? config.max_scheduled_output_rows
        : config.max_scheduled_requests;
    if (max_outputs_ > max_rows_) max_outputs_ = max_rows_;

    host_hidden_bytes_ = static_cast<std::size_t>(max_rows_) * hidden_size_ * sizeof(uint16_t);
    if (host_hidden_bytes_ > 0) {
        void* ptr = nullptr;
        hipError_t err = hipHostMalloc(&ptr, host_hidden_bytes_);
        if (err != hipSuccess) {
            return Status::hip_error("hipHostMalloc pipeline hidden",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        host_hidden_ = ptr;
    }
    host_sampled_bytes_ = static_cast<std::size_t>(max_outputs_) * sizeof(int32_t);
    if (host_sampled_bytes_ > 0) {
        void* ptr = nullptr;
        hipError_t err = hipHostMalloc(&ptr, host_sampled_bytes_);
        if (err != hipSuccess) {
            return Status::hip_error("hipHostMalloc pipeline sampled",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        host_sampled_ = ptr;
    }

    std::fprintf(stderr,
                 "PIPELINE parallel: %zu stages, devices",
                 stages_.size());
    for (int device : devices_) std::fprintf(stderr, " %d", device);
    std::fprintf(stderr, "\n");
    return Status::make_ok();
}

Status PipelineParallelCoordinator::on_sequence_created(const PagedSequenceState& source) {
    const std::uint64_t key = ::shelf::runtime::request_handle_key(source.request_handle());
    if (mirrors_.find(key) != mirrors_.end()) {
        return Status::invalid_state("pipeline mirror sequence already exists",
                                     __FILE__, __LINE__);
    }
    if (stages_[0]->tp() != nullptr) {
        Status st = stages_[0]->tp()->on_sequence_created(source);
        if (!st.ok()) return st;
    }
    MirrorSequences mirror;
    mirror.states.reserve(stages_.size() - 1);
    for (std::size_t i = 1; i < stages_.size(); ++i) {
        PipelineStageRuntime& stage = *stages_[i];
        auto scope = shelf::gpu::ScopedDevice::create(stage.device_id());
        if (!scope.ok()) {
            mirror.states.clear();
            return scope.status();
        }
        auto created = create_paged_sequence_state(
            stage.sequence_pool(), stage.gdn_pool(), stage.kv_pool(), source.max_seq_len,
            stage.stream());
        if (!created.ok()) {
            mirror.states.clear();
            return created.status();
        }
        PagedSequenceState state = created.release();
        if (stage.tp() != nullptr) {
            Status st = stage.tp()->on_sequence_created(state);
            if (!st.ok()) {
                mirror.states.clear();
                return st;
            }
        }
        if (::shelf::runtime::request_handle_key(state.request_handle()) != key) {
            mirror.states.clear();
            return Status::invalid_state(
                "pipeline mirror sequence slot diverged from stage0", __FILE__, __LINE__);
        }
        mirror.states.push_back(std::move(state));
    }
    mirrors_.emplace(key, std::move(mirror));
    return Status::make_ok();
}

Status PipelineParallelCoordinator::on_sequence_released(const PagedSequenceState& source) {
    const std::uint64_t key = ::shelf::runtime::request_handle_key(source.request_handle());
    auto it = mirrors_.find(key);
    if (it == mirrors_.end()) {
        return Status::invalid_state("pipeline mirror sequence not found", __FILE__,
                                     __LINE__);
    }
    Status first_error = Status::make_ok();
    if (stages_[0]->tp() != nullptr) {
        Status st = stages_[0]->tp()->on_sequence_released(source);
        if (!st.ok()) first_error = st;
    }
    for (std::size_t i = 1; i < stages_.size(); ++i) {
        PipelineStageRuntime& stage = *stages_[i];
        auto scope = shelf::gpu::ScopedDevice::create(stage.device_id());
        if (!scope.ok()) {
            if (first_error.ok()) first_error = scope.status();
            continue;
        }
        if (stage.tp() != nullptr) {
            Status st = stage.tp()->on_sequence_released(it->second.states[i - 1]);
            if (!st.ok() && first_error.ok()) first_error = st;
        }
        Status st = release_paged_sequence_state(
            it->second.states[i - 1], stage.stream());
        if (!st.ok() && first_error.ok()) first_error = st;
    }
    mirrors_.erase(it);
    return first_error;
}

Status PipelineParallelCoordinator::sync_mirror_state(const ScheduledBatch& batch) {
    for (const ScheduledRequest& req : batch.requests) {
        if (req.sequence == nullptr) continue;
        const std::uint64_t key = ::shelf::runtime::request_handle_key(req.handle);
        auto it = mirrors_.find(key);
        if (it == mirrors_.end()) continue;
        const std::size_t want_blocks = req.sequence->block_table.size();
        for (std::size_t i = 0; i < it->second.states.size(); ++i) {
            PagedSequenceState& mirror = it->second.states[i];
            PipelineStageRuntime& stage = *stages_[i + 1];
            auto scope = shelf::gpu::ScopedDevice::create(stage.device_id());
            if (!scope.ok()) return scope.status();
            Status st = import_sequence_block_table(mirror, *req.sequence, stage.stream());
            if (!st.ok()) return st;
            if (mirror.block_table.size() != want_blocks) {
                return Status::invalid_state(
                    "pipeline mirror block table size mismatch", __FILE__, __LINE__);
            }
            mirror.position = req.sequence->position;
        }
    }
    return Status::make_ok();
}

Result<BatchExecutionOutput> PipelineParallelCoordinator::on_execute(
    const ScheduledBatch& batch) {
    return on_execute(batch, nullptr);
}

Result<BatchExecutionOutput> PipelineParallelCoordinator::on_execute(
    const ScheduledBatch& batch, const ExecuteBatchOptions* per_rank) {
    for (const auto& request : batch.requests) {
        if (request.token_bitmask != nullptr) {
            return Status::unsupported("token bitmask requires single-device execution", __FILE__, __LINE__);
        }
    }
    if (stages_.size() < 2) {
        return Status::invalid_state("pipeline coordinator has no stages", __FILE__,
                                     __LINE__);
    }
    if (batch.token_ids_location != TokenIdsLocation::Host) {
        return Status::unsupported(
            "pipeline execution requires host token ids", __FILE__, __LINE__);
    }
    if (batch.speculative_verify || per_rank != nullptr) {
        return Status::unsupported(
            "pipeline execution does not support speculative verify", __FILE__,
            __LINE__);
    }
    if (batch.num_decode_requests > 0u && batch.num_prefill_requests > 0u) {
        return Status::unsupported(
            "pipeline execution does not support a mixed decode/prefill batch",
            __FILE__, __LINE__);
    }
    const Status sync_st = sync_mirror_state(batch);
    if (!sync_st.ok()) return sync_st;

    const std::uint32_t rows = batch.num_tokens;
    if (rows == 0 || rows > max_rows_) {
        return Status::invalid_argument("pipeline batch rows out of range", __FILE__,
                                        __LINE__);
    }

    std::vector<std::vector<ScheduledRequest>> request_storage(stages_.size());
    std::vector<ScheduledBatch> stage_batches(stages_.size());
    for (std::size_t i = 0; i < stages_.size(); ++i) {
        stage_batches[i] = batch;
        request_storage[i].assign(batch.requests.begin(), batch.requests.end());
        if (i > 0) {
            for (ScheduledRequest& request : request_storage[i]) {
                const std::uint64_t key =
                    ::shelf::runtime::request_handle_key(request.handle);
                auto it = mirrors_.find(key);
                if (it == mirrors_.end()) {
                    return Status::invalid_state(
                        "pipeline mirror sequence missing for scheduled request",
                        __FILE__, __LINE__);
                }
                if (i - 1 >= it->second.states.size()) {
                    return Status::invalid_state(
                        "pipeline mirror sequence stage out of range", __FILE__,
                        __LINE__);
                }
                request.sequence = &it->second.states[i - 1];
            }
        }
        stage_batches[i].requests =
            std::span<const ScheduledRequest>(request_storage[i]);
    }

    BatchExecutionOutput output;
    for (std::size_t i = 0; i < stages_.size(); ++i) {
        const uint16_t* input_hidden =
            i == 0 ? nullptr : static_cast<const uint16_t*>(host_hidden_);
        Status st = run_stage(i, stage_batches[i], input_hidden, rows, output);
        if (!st.ok()) return st;
        if (i + 1 < stages_.size()) {
            PipelineStageRuntime& stage = *stages_[i];
            auto scope = shelf::gpu::ScopedDevice::create(stage.device_id());
            if (!scope.ok()) return scope.status();
            const std::size_t bytes =
                static_cast<std::size_t>(rows) * hidden_size_ * sizeof(uint16_t);
            if (bytes > host_hidden_bytes_) {
                return Status::invalid_state("pipeline hidden buffer too small",
                                             __FILE__, __LINE__);
            }
            hipError_t err = hipMemcpyAsync(
                host_hidden_, output.final_hidden.data(), bytes,
                hipMemcpyDeviceToHost, stage.stream());
            if (err != hipSuccess) {
                return Status::hip_error("pipeline hidden copy D2H",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipStreamSynchronize(stage.stream());
            if (err != hipSuccess) {
                return Status::hip_error("pipeline hidden sync",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
    }

    const std::uint32_t num_outputs = output.num_outputs;
    if (num_outputs > max_outputs_) {
        return Status::invalid_state("pipeline sampled output too large", __FILE__,
                                     __LINE__);
    }
    if (num_outputs > 0) {
        PipelineStageRuntime& last = *stages_.back();
        {
            auto scope = shelf::gpu::ScopedDevice::create(last.device_id());
            if (!scope.ok()) return scope.status();
            const std::size_t bytes =
                static_cast<std::size_t>(num_outputs) * sizeof(int32_t);
            hipError_t err = hipMemcpyAsync(
                host_sampled_, output.sampled_tokens.data<int32_t>(), bytes,
                hipMemcpyDeviceToHost, last.stream());
            if (err != hipSuccess) {
                return Status::hip_error("pipeline sampled copy D2H",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipStreamSynchronize(last.stream());
            if (err != hipSuccess) {
                return Status::hip_error("pipeline sampled sync",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
        PipelineStageRuntime& first = *stages_.front();
        auto scope = shelf::gpu::ScopedDevice::create(first.device_id());
        if (!scope.ok()) return scope.status();
        const std::size_t bytes =
            static_cast<std::size_t>(num_outputs) * sizeof(int32_t);
        hipError_t err = hipMemcpyAsync(
            first.executor().sampled_tokens.data<int32_t>(), host_sampled_, bytes,
            hipMemcpyHostToDevice, first.stream());
        if (err != hipSuccess) {
            return Status::hip_error("pipeline sampled copy H2D",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        err = hipStreamSynchronize(first.stream());
        if (err != hipSuccess) {
            return Status::hip_error("pipeline sampled sync stage0",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        auto sampled = first.executor().sampled_tokens.slice(0, 0, num_outputs);
        if (!sampled.ok()) return sampled.status();
        output.sampled_tokens = sampled.release();
    }
    return output;
}

Status PipelineParallelCoordinator::run_stage(
    std::size_t index, const ScheduledBatch& batch, const uint16_t* input_hidden,
    std::size_t rows, BatchExecutionOutput& output) {
    PipelineStageRuntime& stage = *stages_[index];
    auto scope = shelf::gpu::ScopedDevice::create(stage.device_id());
    if (!scope.ok()) return scope.status();
    if (index > 0) {
        if (input_hidden == nullptr) {
            return Status::invalid_state("pipeline stage has no hidden input",
                                         __FILE__, __LINE__);
        }
        const std::size_t bytes = rows * hidden_size_ * sizeof(uint16_t);
        if (stage.tp() != nullptr) {
            for (int r = 0; r < stage.tp()->tp_size(); ++r) {
                TpRankRuntime& rank = stage.tp()->rank(static_cast<std::size_t>(r));
                auto rank_scope = shelf::gpu::ScopedDevice::create(rank.device_id());
                if (!rank_scope.ok()) return rank_scope.status();
                hipError_t err = hipMemcpyAsync(
                    const_cast<void*>(rank.executor().pipeline_input_hidden.data()),
                    input_hidden, bytes, hipMemcpyHostToDevice, rank.stream());
                if (err != hipSuccess) {
                    return Status::hip_error("pipeline hidden copy H2D",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
        } else {
            hipError_t err = hipMemcpyAsync(
                const_cast<void*>(stage.executor().pipeline_input_hidden.data()),
                input_hidden, bytes, hipMemcpyHostToDevice, stage.stream());
            if (err != hipSuccess) {
                return Status::hip_error("pipeline hidden copy H2D",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
    }
    auto result = stage.tp() != nullptr
        ? stage.tp()->on_execute(batch)
        : execute_batch(stage.executor(), batch, stage.stream());
    if (!result.ok()) return result.status();
    output = result.release();
    return Status::make_ok();
}

std::string PipelineParallelCoordinator::debug_report() const {
    std::ostringstream out;
    out << "pipeline_stages=" << stages_.size() << " hidden=" << hidden_size_ << "\n";
    for (std::size_t i = 0; i < stages_.size(); ++i) {
        const PipelineStageRuntime& stage = *stages_[i];
        out << "  stage" << i
            << " device=" << stage.device_id()
            << " layers=[" << stage.stage().layers.layer_lo << ","
            << stage.stage().layers.layer_hi << ")"
            << " weight_bytes="
            << (stage.tp() != nullptr
                    ? const_cast<TpCoordinator*>(stage.tp())->rank(0).weight_bytes()
                    : stage.model().weight_bytes())
            << " kv_pages=" << stage.kv_pool_pages() << "\n";
    }
    return out.str();
}

}  // namespace runtime
}  // namespace shelf
