#pragma once

#include "support/resident_model_fixture.h"

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/runtime/continuous_batcher.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/models/qwen35/runtime/tensor_parallel.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace tp_harness {

namespace gpu = ps::gpu;

using ps::qwen35::Executor;
using ps::qwen35::ExecutorConfig;
using ps::state::GdnStatePool;
using ps::state::GdnStatePoolLayout;
using ps::kernel::KVCacheDType;
using ps::state::PagedKVPool;
using ps::qwen35::Qwen35LoadOptions;
using ps::qwen35::Qwen35Model;
using ps::qwen35::Qwen35ModelWeights;
using ps::qwen35::Qwen35TextConfig;
using ps::state::SequenceSlotPool;
using ps::qwen35::create_model_executor;
using ps::qwen35::executor_shutdown;
using ps::qwen35::runtime::ContinuousBatcher;
using ps::qwen35::runtime::ContinuousBatcherConfig;
using ps::qwen35::runtime::RuntimeRequest;
using ps::qwen35::runtime::TpCoordinator;
using ps::qwen35::runtime::TpCoordinatorConfig;
using ps::qwen35::runtime::TpRankRuntime;
using ps::qwen35::kMaxTargetHiddenTaps;
using ps::qwen35::runtime::ValueTraceSink;

inline std::vector<int> tp_devices_from_env() {
    if (const char* env = std::getenv("SHELF_TP_DEVICES")) {
        std::vector<int> out;
        const char* p = env;
        while (*p != '\0' && out.size() < 2u) {
            char* end = nullptr;
            const long value = std::strtol(p, &end, 10);
            if (end == p) break;
            if (value >= 0) out.push_back(static_cast<int>(value));
            p = end;
            while (*p == ',' || *p == ' ') ++p;
        }
        if (out.size() == 2u) return out;
    }
    return {0, 1};
}

struct RunOptions {
    std::string model_dir;
    std::uint64_t arena_bytes = 1024ull * 1024ull * 1024ull;
    std::uint32_t max_scheduled_tokens = 16;
    std::uint32_t max_scheduled_requests = 4;
    std::uint32_t max_seq_len = 64;
    std::uint32_t page_tokens = 16;
    std::vector<int> taps{0};
    bool verify_crc = true;
    std::vector<int32_t> prompt = {3, 11, 7, 42, 5};
    std::uint32_t max_new_tokens = 8;
};

struct StepCapture {
    std::vector<float> hidden_row0;
    std::vector<float> logits_row0;
    std::vector<float> tap0_row0;
    std::vector<float> tap1_row0;
    std::vector<std::vector<float>> all_taps;
};

inline bool download_row0(const gpu::Tensor& tensor, int device,
                          std::vector<float>& out) {
    if (tensor.ndim() < 2) return false;
    std::size_t span = 0;
    if (!tensor.required_span_bytes(span).ok()) return false;
    if (span == 0) return false;
    std::vector<std::uint8_t> raw(span);
    {
        auto scope = ps::gpu::ScopedDevice::create(device);
        if (!scope.ok()) return false;
        if (hipMemcpy(raw.data(), tensor.data<void>(), span, hipMemcpyDeviceToHost) !=
            hipSuccess) {
            return false;
        }
    }
    const std::size_t feature = static_cast<std::size_t>(tensor.dim(1));
    const float* row = reinterpret_cast<const float*>(raw.data());
    out.assign(row, row + feature);
    return true;
}

inline int argmax(const std::vector<float>& v) {
    int best = 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = static_cast<int>(i);
    }
    return best;
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    float worst = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

struct TraceCapture {
    int device = 0;
    std::map<std::uint32_t, std::vector<float>> rows0;
    ValueTraceSink sink;

    void bind() {
        sink.on_value = [this](std::uint32_t value_id, const void* device_ptr,
                               std::uint32_t row_stride, std::uint32_t feature_count,
                               ps::runtime::ValueDType dtype, std::uint32_t rows,
                               hipStream_t stream) {
            (void)row_stride;
            if (rows == 0 || feature_count == 0) return;
            const std::size_t elem = ps::runtime::value_dtype_bytes(dtype);
            if (elem == 0 || elem > 4) return;
            const std::size_t n = static_cast<std::size_t>(feature_count);
            std::vector<std::uint8_t> host(n * elem);
            auto scope = ps::gpu::ScopedDevice::create(this->device);
            if (!scope.ok()) return;
            if (hipMemcpyAsync(host.data(), device_ptr, n * elem,
                               hipMemcpyDeviceToHost, stream) != hipSuccess) {
                return;
            }
            if (hipStreamSynchronize(stream) != hipSuccess) return;
            std::vector<float> decoded;
            if (elem == 2) {
                const auto* u = reinterpret_cast<const std::uint16_t*>(host.data());
                decoded.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    const std::uint32_t bits = static_cast<std::uint32_t>(u[i]) << 16;
                    float f = 0.0f;
                    std::memcpy(&f, &bits, sizeof(f));
                    decoded[i] = f;
                }
            } else if (elem == 4) {
                const float* f = reinterpret_cast<const float*>(host.data());
                decoded.assign(f, f + n);
            } else {
                return;
            }
            this->rows0[value_id] = std::move(decoded);
        };
    }
};

inline void compare_traces(const std::map<std::uint32_t, std::vector<float>>& reference,
                           const std::map<std::uint32_t, std::vector<float>>& candidate,
                           const char* tag) {
    int reported = 0;
    for (const auto& entry : reference) {
        auto it = candidate.find(entry.first);
        if (it == candidate.end()) continue;
        if (it->second.size() != entry.second.size()) continue;
        float worst = 0.0f;
        float scale = 1.0f;
        for (std::size_t i = 0; i < entry.second.size(); ++i) {
            worst = std::max(worst, std::fabs(entry.second[i] - it->second[i]));
            scale = std::max(scale, std::fabs(entry.second[i]));
        }
        const float rel = worst / scale;
        if (rel > 0.05f && reported < 8) {
            std::printf("dbg %s diverging value id=%u features=%zu rel=%g ref0=%g cand0=%g\n",
                        tag, entry.first, entry.second.size(), static_cast<double>(rel),
                        static_cast<double>(entry.second[0]),
                        static_cast<double>(it->second[0]));
            ++reported;
        }
    }
}

struct RunResult {
    bool ok = false;
    std::string error;
    std::vector<int32_t> tokens;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    std::size_t decode_steps = 0;
    StepCapture capture;
    StepCapture rank1_capture;
    TraceCapture trace;
    TraceCapture rank1_trace;
    std::string report;
    std::uint64_t weight_bytes = 0;
    std::uint32_t gate_rows = 0;
    std::uint32_t down_cols = 0;
    std::uint32_t q_rows = 0;
    std::uint32_t kv_rows = 0;
    std::uint32_t gdn_qkv_rows = 0;
    std::uint32_t kv_pool_heads = 0;
    std::uint32_t gdn_conv_dim = 0;
};

inline std::uint32_t ceil_div_u32(std::uint32_t a, std::uint32_t b) {
    return (a + b - 1u) / b;
}

inline std::uint32_t count_attention_layers(const Qwen35TextConfig& tc) {
    if (!tc.layer_types.empty()) {
        std::uint32_t n = 0;
        for (int lt : tc.layer_types) {
            if (lt == 1) ++n;
        }
        return n;
    }
    std::uint32_t n = 0;
    for (std::size_t l = 0; l < tc.num_hidden_layers; ++l) {
        const bool gdn = (tc.full_attention_interval == 0)
            ? (l % 4 < 3)
            : (l % tc.full_attention_interval < tc.full_attention_interval - 1);
        if (!gdn) ++n;
    }
    return n;
}

inline ContinuousBatcherConfig batcher_config(const Qwen35TextConfig& tc,
                                              const RunOptions& opts) {
    ContinuousBatcherConfig config;
    config.max_scheduled_tokens = opts.max_scheduled_tokens;
    config.max_scheduled_requests = opts.max_scheduled_requests;
    config.max_seq_len = opts.max_seq_len;
    config.page_tokens = opts.page_tokens;
    config.eos_token_ids = tc.stop_tokens;
    return config;
}

inline void fill_capture_from_executor(const Executor& executor, int device,
                                       StepCapture& capture) {
    download_row0(executor.output_hidden, device, capture.hidden_row0);
    download_row0(executor.logits, device, capture.logits_row0);
    capture.all_taps.assign(executor.config.target_hidden_tap_count, {});
    for (std::uint32_t t = 0; t < executor.config.target_hidden_tap_count; ++t) {
        download_row0(executor.dflash_target_hidden[t], device, capture.all_taps[t]);
        if (t == 0) capture.tap0_row0 = capture.all_taps[t];
        if (t == 1) capture.tap1_row0 = capture.all_taps[t];
    }
}

inline void collect_tokens(const ContinuousBatcher& batcher, std::uint64_t id,
                           std::vector<std::int32_t>& tokens) {
    const RuntimeRequest* request = batcher.request(id);
    if (request == nullptr) return;
    tokens = request->generated;
}

inline RunResult run_tp1(const RunOptions& opts) {
    RunResult result;
    const int tp1_device = tp_devices_from_env()[0];
    if (hipSetDevice(tp1_device) != hipSuccess) {
        result.error = "tp1 hipSetDevice";
        return result;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        result.error = "tp1 stream create";
        return result;
    }
    Qwen35LoadOptions load_options;
    load_options.verify_quantized_payload_crc = opts.verify_crc;

    ::ps::resident::ResidentTestRequest resident_request;
    resident_request.qwen_model_dir = opts.model_dir;
    resident_request.qwen_options = load_options;
    resident_request.device = tp1_device;
    resident_request.arena_bytes = opts.arena_bytes;
    auto resident_result = ::ps::resident::ResidentModelFixture::acquire(resident_request);
    if (!resident_result.ok()) {
        result.error = "tp1 resident acquire: " + resident_result.status().message();
        return result;
    }
    ::ps::resident::ResidentModelFixture resident = resident_result.release();
    resident.set_release_on_destroy(true);

    auto arena_result = gpu::GpuArena::create(tp1_device, resident.arena_bytes());
    if (!arena_result.ok()) {
        result.error = "tp1 arena: " + arena_result.status().message();
        return result;
    }
    gpu::GpuArena arena = arena_result.release();

    auto model_result = resident.enabled()
        ? resident.take_qwen35()
        : Qwen35Model::load_from_safetensors(opts.model_dir, arena, stream, load_options);
    if (!model_result.ok()) {
        result.error = "tp1 model load: " + model_result.status().message();
        return result;
    }
    Qwen35Model model = model_result.release();

    const std::uint32_t max_blocks =
        ceil_div_u32(opts.max_seq_len + 1u, opts.page_tokens) + 1u;
    const std::uint32_t kv_pages =
        ceil_div_u32(opts.max_seq_len + 1u, opts.page_tokens) + 1u;

    auto seq_result =
        SequenceSlotPool::create(arena, opts.max_scheduled_requests, max_blocks);
    if (!seq_result.ok()) {
        result.error = "tp1 sequence pool: " + seq_result.status().message();
        return result;
    }
    SequenceSlotPool seq_pool = seq_result.release();

    auto gdn_layout = ps::qwen35::make_qwen35_gdn_state_layout(model.text_config());
    auto gdn_result =
        GdnStatePool::create(arena, opts.max_scheduled_requests, gdn_layout);
    if (!gdn_result.ok()) {
        result.error = "tp1 gdn pool: " + gdn_result.status().message();
        return result;
    }
    GdnStatePool gdn_pool = gdn_result.release();

    auto kv_result = PagedKVPool::create(
        arena, kv_pages, opts.page_tokens, count_attention_layers(model.text_config()),
        static_cast<std::uint32_t>(model.text_config().num_key_value_heads),
        static_cast<std::uint32_t>(model.text_config().attention_head_dim),
        KVCacheDType::BF16);
    if (!kv_result.ok()) {
        result.error = "tp1 kv pool: " + kv_result.status().message();
        return result;
    }
    PagedKVPool kv_pool = kv_result.release();

    ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = opts.max_scheduled_tokens;
    exec_config.max_scheduled_requests = opts.max_scheduled_requests;
    for (std::size_t i = 0; i < opts.taps.size() && i < kMaxTargetHiddenTaps; ++i) {
        exec_config.target_hidden_taps[i] = static_cast<std::uint32_t>(opts.taps[i]);
    }
    exec_config.target_hidden_tap_count =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            opts.taps.size(), static_cast<std::size_t>(kMaxTargetHiddenTaps)));
    auto exec_result =
        create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        result.error = "tp1 executor: " + exec_result.status().message();
        return result;
    }
    Executor executor = exec_result.release();
    result.trace.device = tp1_device;
    result.trace.bind();
    executor.value_trace = &result.trace.sink;

    ContinuousBatcher batcher(executor, seq_pool, gdn_pool, kv_pool,
                              batcher_config(model.text_config(), opts), stream);
    auto id_result = batcher.submit(opts.prompt, opts.max_new_tokens);
    if (!id_result.ok()) {
        result.error = "tp1 submit: " + id_result.status().message();
        return result;
    }
    const std::uint64_t id = id_result.value();

    const auto t_prefill = std::chrono::steady_clock::now();
    auto step = batcher.step();
    if (!step.ok()) {
        result.error = "tp1 prefill step: " + step.status().message();
        return result;
    }
    result.prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_prefill).count();
    fill_capture_from_executor(executor, tp1_device, result.capture);
    executor.value_trace = nullptr;

    const auto t_decode = std::chrono::steady_clock::now();
    while (batcher.has_pending()) {
        auto decode_step = batcher.step();
        ++result.decode_steps;
        if (!decode_step.ok()) {
            result.error = "tp1 decode step: " + decode_step.status().message();
            return result;
        }
    }
    result.decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_decode).count();
    collect_tokens(batcher, id, result.tokens);

    const Qwen35ModelWeights& weights = model.weights();
    if (!weights.layers.empty()) {
        result.gate_rows = weights.layers[0].mlp_gate_proj.rows;
        result.down_cols = weights.layers[0].mlp_down_proj.cols;
        result.gdn_qkv_rows = weights.layers[0].attn_in_proj_qkv.rows;
    }
    if (weights.layers.size() > 1) {
        result.q_rows = weights.layers[1].attn_q_proj.rows;
        result.kv_rows = weights.layers[1].attn_k_proj.rows;
    }
    result.kv_pool_heads = kv_pool.kv_heads();
    result.gdn_conv_dim = gdn_pool.layout().conv_dim;

    (void)batcher.cancel_all();
    executor_shutdown(executor);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    arena.shutdown();

    result.ok = true;
    return result;
}

inline RunResult run_tp2(const RunOptions& opts) {
    RunResult result;
    TpCoordinatorConfig config;
    config.model_dir = opts.model_dir;
    config.model_host_socket = ::ps::models::model_host_socket_from_env();
    config.devices = tp_devices_from_env();
    config.max_scheduled_tokens = opts.max_scheduled_tokens;
    config.max_scheduled_requests = opts.max_scheduled_requests;
    config.max_seq_len = opts.max_seq_len;
    config.page_tokens = opts.page_tokens;
    config.arena_bytes = opts.arena_bytes;
    for (std::size_t i = 0; i < opts.taps.size() && i < kMaxTargetHiddenTaps; ++i) {
        config.target_hidden_taps[i] = static_cast<std::uint32_t>(opts.taps[i]);
    }
    config.target_hidden_tap_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(opts.taps.size(), static_cast<std::size_t>(kMaxTargetHiddenTaps)));
    config.load_options.verify_quantized_payload_crc = opts.verify_crc;

    auto coordinator_result = TpCoordinator::create(config);
    if (!coordinator_result.ok()) {
        result.error = "tp2 coordinator: " + coordinator_result.status().message();
        return result;
    }
    std::unique_ptr<TpCoordinator> coordinator = coordinator_result.release();
    result.report = coordinator->debug_report();

    TpRankRuntime& rank0 = coordinator->rank(0);
    TpRankRuntime& rank1 = coordinator->rank(1);

    {
        const Qwen35ModelWeights& weights = rank0.model().weights();
        if (!weights.layers.empty()) {
            result.gate_rows = weights.layers[0].mlp_gate_proj.rows;
            result.down_cols = weights.layers[0].mlp_down_proj.cols;
            result.gdn_qkv_rows = weights.layers[0].attn_in_proj_qkv.rows;
        }
        if (weights.layers.size() > 1) {
            result.q_rows = weights.layers[1].attn_q_proj.rows;
            result.kv_rows = weights.layers[1].attn_k_proj.rows;
        }
        result.kv_pool_heads = rank0.kv_pool().kv_heads();
        result.gdn_conv_dim = rank0.gdn_pool().layout().conv_dim;
        result.weight_bytes = coordinator->rank(0).weight_bytes();
    }

    ContinuousBatcher batcher(rank0.executor(), rank0.sequence_pool(), rank0.gdn_pool(),
                              rank0.kv_pool(),
                              batcher_config(rank0.text_config(), opts), rank0.stream());
    batcher.set_tp_batch_hook(coordinator.get());

    result.trace.device = rank0.device_id();
    result.trace.bind();
    result.rank1_trace.device = rank1.device_id();
    result.rank1_trace.bind();
    rank0.executor().value_trace = &result.trace.sink;
    rank1.executor().value_trace = &result.rank1_trace.sink;

    auto id_result = batcher.submit(opts.prompt, opts.max_new_tokens);
    if (!id_result.ok()) {
        result.error = "tp2 submit: " + id_result.status().message();
        return result;
    }
    const std::uint64_t id = id_result.value();

    const auto t_prefill = std::chrono::steady_clock::now();
    auto step = batcher.step();
    if (!step.ok()) {
        result.error = "tp2 prefill step: " + step.status().message();
        return result;
    }
    result.prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_prefill).count();
    fill_capture_from_executor(rank0.executor(), rank0.device_id(), result.capture);
    fill_capture_from_executor(rank1.executor(), rank1.device_id(), result.rank1_capture);
    rank0.executor().value_trace = nullptr;
    rank1.executor().value_trace = nullptr;

    const auto t_decode = std::chrono::steady_clock::now();
    while (batcher.has_pending()) {
        auto decode_step = batcher.step();
        ++result.decode_steps;
        if (!decode_step.ok()) {
            result.error = "tp2 decode step: " + decode_step.status().message();
            return result;
        }
    }
    result.decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_decode).count();
    collect_tokens(batcher, id, result.tokens);

    (void)batcher.cancel_all();
    result.ok = true;
    return result;
}

}
