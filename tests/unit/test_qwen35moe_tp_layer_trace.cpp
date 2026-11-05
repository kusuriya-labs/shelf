#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/runtime/tensor_parallel.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/runtime/tp/tp_barrier_group.h>
#include <shelf/runtime/tp/tp_transports.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace q35 = shelf::qwen35;
namespace q35r = shelf::qwen35::runtime;
namespace qhy = shelf::qwen_hybrid;
namespace moe = shelf::qwen35moe;
namespace mods = shelf::models;
namespace rt = shelf::runtime;
namespace st = shelf::state;
namespace gpu = shelf::gpu;

namespace {

using shelf::execution::ValueTraceSink;

float bf16_to_float(std::uint16_t bits) {
    std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float f = 0.0f;
    std::memcpy(&f, &wide, sizeof(f));
    return f;
}

float fp16_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t fraction = bits & 0x3ffu;
    std::uint32_t wide = 0;
    if (exponent == 0u) {
        if (fraction == 0u) {
            wide = sign;
        } else {
            std::uint32_t e = 0u;
            std::uint32_t f = fraction;
            while ((f & 0x400u) == 0u) {
                f <<= 1;
                ++e;
            }
            wide = sign | ((127u - 15u - e) << 23) | ((f & 0x3ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        wide = sign | 0x7f800000u | (fraction << 13);
    } else {
        wide = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &wide, sizeof(f));
    return f;
}

// The same PSKLDTOK corpus the quality benchmark reads, so the trace run and the
// teacher-forced measurement see the same tokens.
std::vector<int32_t> read_token_corpus(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return {};
    char magic[8] = {};
    uint32_t version = 0;
    uint64_t count = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "PSKLDTOK", 8) == 0;
    ok = ok && std::fread(&version, 4, 1, f) == 1 && version == 1u;
    ok = ok && std::fread(&count, 8, 1, f) == 1 && count > 0u && count < (1u << 22);
    std::vector<int32_t> tokens;
    if (ok) {
        tokens.resize(static_cast<std::size_t>(count));
        ok = std::fread(tokens.data(), sizeof(int32_t), tokens.size(), f) == tokens.size();
    }
    std::fclose(f);
    return ok ? tokens : std::vector<int32_t>{};
}

struct TracedValue {
    std::uint32_t rows = 0;
    std::uint32_t features = 0;
    std::vector<float> row0;
};

struct TraceCapture {
    int device = 0;
    std::map<std::uint32_t, TracedValue> values;
    ValueTraceSink sink;

    void bind() {
        sink.on_value = [this](std::uint32_t value_id, const void* device_ptr,
                               std::uint32_t row_stride, std::uint32_t feature_count,
                               shelf::runtime::ValueDType dtype, std::uint32_t rows,
                               hipStream_t stream) {
            (void)row_stride;
            if (rows == 0 || feature_count == 0) return;
            const std::size_t elem = shelf::runtime::value_dtype_bytes(dtype);
            if (elem == 0 || elem > 4) return;
            const std::size_t n = static_cast<std::size_t>(feature_count);
            std::vector<std::uint8_t> host(n * elem);
            auto scope = gpu::ScopedDevice::create(this->device);
            if (!scope.ok()) return;
            if (hipMemcpyAsync(host.data(), device_ptr, n * elem, hipMemcpyDeviceToHost,
                               stream) != hipSuccess) {
                return;
            }
            if (hipStreamSynchronize(stream) != hipSuccess) return;
            TracedValue entry;
            entry.rows = rows;
            entry.features = feature_count;
            entry.row0.resize(n);
            if (elem == 2 && dtype == shelf::runtime::ValueDType::BF16) {
                const auto* u = reinterpret_cast<const std::uint16_t*>(host.data());
                for (std::size_t i = 0; i < n; ++i) entry.row0[i] = bf16_to_float(u[i]);
            } else if (elem == 4 && dtype == shelf::runtime::ValueDType::F32) {
                const float* f = reinterpret_cast<const float*>(host.data());
                entry.row0.assign(f, f + n);
            } else {
                return;
            }
            this->values[value_id] = std::move(entry);
        };
    }
};

// The internal state a rank carries out of the prefill: what the next step that reads history
// will see. Rank-local channels are stored in the rank's own order; the comparison reassembles
// the global order from the two halves.
struct StateSnapshot {
    bool valid = false;
    std::uint32_t position = 0;
    std::vector<std::uint32_t> pages;

    std::uint32_t gdn_layers = 0;
    std::uint32_t conv_rows = 0;
    std::uint32_t conv_dim = 0;
    // The rank's own share of the conv channels: [q_half | k_half | v_half], so the global
    // channel order can be rebuilt without guessing where the halves sit.
    std::uint32_t conv_qk_local = 0;
    std::uint32_t conv_v_local = 0;
    std::uint32_t tp_rank = 0;
    std::uint32_t tp_size = 1;
    std::vector<float> conv;
    std::uint32_t v_heads = 0;
    std::uint32_t head_k = 0;
    std::uint32_t head_v = 0;
    std::vector<float> recurrent;

    std::uint32_t kv_layers = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t kv_tokens = 0;
    std::uint32_t head_dim = 0;
    std::vector<float> kv_k;
    std::vector<float> kv_v;
};

StateSnapshot snapshot_state(q35r::TpRankRuntime& rank, const st::PagedSequenceState& sequence,
                             std::uint32_t tokens) {
    StateSnapshot snap;
    st::GdnStatePool& gdn = rank.gdn_pool();
    st::PagedKVPool& kv = rank.kv_pool();
    const st::GdnStatePoolDeviceView gview = gdn.device_view();
    const std::size_t slot = sequence.slot;

    snap.position = sequence.position;
    snap.pages = sequence.block_table;

    snap.gdn_layers = gview.num_gdn_states;
    snap.conv_rows = static_cast<std::uint32_t>(gview.conv_history);
    snap.conv_dim = gview.conv_dim;
    snap.v_heads = gview.num_v_heads;
    snap.head_k = gview.head_k;
    snap.head_v = gview.head_v;
    const qhy::TensorParallelContext& geometry = rank.context();
    snap.tp_rank = geometry.tp_rank;
    snap.tp_size = geometry.tp_size;
    snap.conv_qk_local = geometry.local_gdn_key_heads * snap.head_k;
    snap.conv_v_local = geometry.local_gdn_value_heads * snap.head_v;
    snap.kv_layers = static_cast<std::uint32_t>(kv.num_attention_layers());
    snap.kv_heads = kv.kv_heads();
    snap.head_dim = kv.head_dim();
    snap.kv_tokens = tokens;

    auto conv_scope = gpu::ScopedDevice::create(rank.device_id());
    if (!conv_scope.ok()) return snap;

    snap.conv.assign(static_cast<std::size_t>(snap.gdn_layers) * snap.conv_rows * snap.conv_dim,
                     0.0f);
    for (std::uint32_t li = 0; li < snap.gdn_layers; ++li) {
        for (std::uint32_t h = 0; h < snap.conv_rows; ++h) {
            std::vector<std::uint16_t> row(snap.conv_dim);
            const std::size_t elems =
                slot * gview.conv_slot_stride + static_cast<std::size_t>(li) *
                                                     gview.conv_layer_stride +
                static_cast<std::size_t>(h) * gview.conv_history_stride;
            if (hipMemcpy(row.data(), gview.conv_base + elems, row.size() * sizeof(std::uint16_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                return snap;
            }
            const std::size_t base =
                (static_cast<std::size_t>(li) * snap.conv_rows + h) * snap.conv_dim;
            for (std::size_t c = 0; c < row.size(); ++c)
                snap.conv[base + c] = bf16_to_float(row[c]);
        }
    }

    const std::size_t per_layer =
        static_cast<std::size_t>(snap.v_heads) * snap.head_k * snap.head_v;
    snap.recurrent.assign(static_cast<std::size_t>(snap.gdn_layers) * per_layer, 0.0f);
    std::vector<std::uint16_t> rec(per_layer);
    for (std::uint32_t li = 0; li < snap.gdn_layers; ++li) {
        const std::size_t elems =
            slot * gview.recurrent_slot_stride + static_cast<std::size_t>(li) *
                                                     gview.recurrent_layer_stride;
        if (hipMemcpy(rec.data(), gview.recurrent_base + elems, rec.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            return snap;
        }
        for (std::size_t i = 0; i < rec.size(); ++i)
            snap.recurrent[static_cast<std::size_t>(li) * per_layer + i] = fp16_to_float(rec[i]);
    }

    if (kv.dtype() != shelf::kernel::KVCacheDType::BF16) return snap;
    const std::size_t page_elems =
        static_cast<std::size_t>(kv.page_tokens()) * kv.kv_heads() * kv.head_dim();
    const std::size_t layer_elems = page_elems * kv.num_pages();
    auto copy_kv = [&](const gpu::Tensor& pool, std::vector<float>& out) -> bool {
        out.assign(static_cast<std::size_t>(snap.kv_layers) * tokens * kv.kv_heads() *
                       kv.head_dim(),
                   0.0f);
        std::vector<std::uint16_t> page(page_elems);
        for (std::uint32_t li = 0; li < snap.kv_layers; ++li) {
            for (std::uint32_t t = 0; t < tokens; ++t) {
                const std::uint32_t page_index = t / kv.page_tokens();
                if (page_index >= sequence.block_table.size()) return false;
                const std::size_t base =
                    static_cast<std::size_t>(li) * layer_elems +
                    static_cast<std::size_t>(sequence.block_table[page_index]) * page_elems +
                    static_cast<std::size_t>(t % kv.page_tokens()) * kv.kv_heads() * kv.head_dim();
                if (hipMemcpy(page.data(), pool.data<shelf::bf16_t>() + base,
                              static_cast<std::size_t>(kv.kv_heads()) * kv.head_dim() *
                                  sizeof(std::uint16_t),
                              hipMemcpyDeviceToHost) != hipSuccess) {
                    return false;
                }
                const std::size_t out_base =
                    (static_cast<std::size_t>(li) * tokens + t) * kv.kv_heads() * kv.head_dim();
                const std::size_t n = static_cast<std::size_t>(kv.kv_heads()) * kv.head_dim();
                for (std::size_t i = 0; i < n; ++i)
                    out[out_base + i] = bf16_to_float(page[i]);
            }
        }
        return true;
    };
    if (!copy_kv(kv.k_pool(), snap.kv_k)) return snap;
    if (!copy_kv(kv.v_pool(), snap.kv_v)) return snap;

    snap.valid = true;
    return snap;
}

// The decode-step value ids only mean something next to the op that wrote them, so the tool
// prints the decode program's dispatch list: op index, kernel, and the values it writes.
void dump_program_ops(const q35::Executor& ex, std::uint32_t limit) {
    for (const auto& program : ex.program_set.programs) {
        if (program.execution_class != rt::ExecutionClass::DECODE) continue;
        std::printf("=== program row_bucket=%d class=DECODE dispatches=%zu ===\n",
                    static_cast<int>(program.row_bucket), program.dispatches.size());
        std::map<std::uint32_t, std::uint32_t> slot_of_value;
        for (const auto& binding : program.values)
            slot_of_value[binding.logical_value.id] = binding.slot;
        std::uint32_t printed = 0;
        for (std::size_t i = 0; i < program.dispatches.size() && printed < limit; ++i, ++printed) {
            const auto& op = program.dispatches[i];
            std::printf("op %4zu  %-38s out=", i, shelf::runtime::kernel_to_name(op.kernel_id));
            for (std::uint32_t s = 0; s < op.output_count; ++s) {
                const std::uint32_t value_id = op.output_slots[s];
                const auto* binding = program.find_value(rt::ValueId{value_id});
                if (binding == nullptr) {
                    std::printf(" v%u(?)", value_id);
                    continue;
                }
                std::printf(" v%u(f%u,dom%u,st%u)", value_id, binding->feature_count,
                            static_cast<unsigned>(binding->row_domain),
                            static_cast<unsigned>(binding->storage));
            }
            std::printf("  in=");
            for (std::uint32_t s = 0; s < op.input_count; ++s)
                std::printf(" v%u", op.input_slots[s]);
            std::printf("\n");
        }
        return;
    }
    std::printf("no DECODE program in the program set\n");
}

struct RunResult {
    bool ok = false;
    std::string error;
    std::vector<int32_t> sampled;
    std::map<std::uint32_t, TracedValue> values;
    StateSnapshot state;
};

// Teacher-forced: one prefill of `prompt` rows, then one single-row step. The single-row step is
// the first step that reads history, and its trace is what the two geometries are compared on.
// The state snapshot is taken between the two, so it is exactly the state the failing step reads.
RunResult run_rank(const std::string& model_dir, int device, std::uint32_t tp_size,
                   std::uint32_t tp_rank, std::uint32_t ctx, const std::vector<int32_t>& prompt,
                   rt::TpBarrierGroup* barrier, bool dump_state, bool dump_ops = false) {
    RunResult result;
    auto scope = gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        result.error = "scoped device: " + scope.status().message();
        return result;
    }

    q35r::TpRankRuntimeConfig config;
    config.model_dir = model_dir;
    config.build_rank_model = &mods::build_moe_tp_rank_model;
    config.tp_size = tp_size;
    config.tp_rank = tp_rank;
    config.device_id = device;
    config.max_batch_rows = ctx;
    config.max_scheduled_requests = 1u;
    config.max_scheduled_output_rows = 1u;
    config.max_seq_len = ctx + 8u;
    config.page_tokens = 16u;
    config.kv_cache_dtype = shelf::kernel::KVCacheDType::BF16;
    config.arena_bytes = 0u;
    // A fixed working-set reserve: the auto estimate competes with a 23 GiB checkpoint for the
    // last hundred MiB of one GPU, and the trace run only needs `ctx` tokens of history.
    config.working_set_reserve_bytes = 2ull * 1024ull * 1024ull * 1024ull;

    auto rank_result = q35r::TpRankRuntime::create(config);
    if (!rank_result.ok()) {
        result.error = "TpRankRuntime: " + rank_result.status().message();
        return result;
    }
    std::unique_ptr<q35r::TpRankRuntime> rank = rank_result.release();

    q35::Executor& ex = rank->executor();
    if (barrier != nullptr) ex.tp_barrier = barrier;

    if (dump_ops) dump_program_ops(ex, 60u);

    TraceCapture capture;
    capture.device = device;
    capture.bind();
    ex.value_trace = &capture.sink;

    const hipStream_t stream = rank->stream();
    auto seq_state = st::create_paged_sequence_state(rank->sequence_pool(), rank->gdn_pool(),
                                                     rank->kv_pool(), ctx + 8u, stream);
    if (!seq_state.ok()) {
        result.error = "sequence: " + seq_state.status().message();
        return result;
    }
    st::PagedSequenceState sequence = seq_state.release();

    std::vector<q35::ScheduledRequest> requests;
    auto submit = [&](const int32_t* tokens, uint32_t count) -> bool {
        q35::ScheduledRequest req;
        req.sequence = &sequence;
        req.handle = sequence.request_handle();
        req.execution_class = count == 1u ? rt::ExecutionClass::DECODE : rt::ExecutionClass::PREFILL;
        req.token_begin = 0u;
        req.num_tokens = count;
        req.prefix_tokens = sequence.position;
        req.compute_logits = true;
        req.sample = true;
        req.num_output_rows = 1u;
        requests.clear();
        requests.push_back(req);
        q35::ScheduledBatch batch;
        batch.token_ids = tokens;
        batch.requests = requests;
        batch.num_tokens = count;
        batch.num_requests = 1u;
        batch.num_decode_requests = count == 1u ? 1u : 0u;
        batch.num_verify_requests = 0u;
        batch.num_prefill_requests = count == 1u ? 0u : 1u;
        auto out = q35::execute_batch(ex, batch, stream);
        if (!out.ok()) {
            result.error = "execute_batch rows=" + std::to_string(count) + ": " +
                           out.status().message();
            return false;
        }
        q35::BatchExecutionOutput exec = out.release();
        int32_t token = 0;
        if (exec.sampled_tokens.ndim() > 0 && exec.sampled_tokens.dim(0) > 0)
            (void)hipMemcpy(&token, exec.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                            hipMemcpyDeviceToHost);
        result.sampled.push_back(token);
        return true;
    };

    if (!submit(prompt.data(), static_cast<std::uint32_t>(prompt.size()))) {
        result.values = std::move(capture.values);
        (void)st::release_paged_sequence_state(sequence, stream);
        return result;
    }
    if (dump_state) result.state = snapshot_state(*rank, sequence, static_cast<std::uint32_t>(prompt.size()));

    std::vector<int32_t> next(1, prompt.back());
    (void)submit(next.data(), 1u);

    ex.value_trace = nullptr;
    (void)st::release_paged_sequence_state(sequence, stream);
    result.values = std::move(capture.values);
    result.ok = result.error.empty();
    return result;
}

void compare_traces(const RunResult& reference, const RunResult& candidate, const char* tag) {
    std::printf("--- %s: per-value trace of the single-row step ---\n", tag);
    std::printf("%8s  %5s %5s  %9s  %9s  %10s  %10s  %6s %6s\n", "value_id", "rows", "feats",
                "rel_err", "max|d|", "ref0", "cand0", "nan_r", "nan_c");
    int compared = 0;
    int printed = 0;
    int first_diverged_id = -1;
    int last_executed_id = -1;
    for (const auto& entry : reference.values) {
        if (entry.second.rows != 1u) continue;
        auto it = candidate.values.find(entry.first);
        if (it == candidate.values.end()) continue;
        if (it->second.rows != entry.second.rows ||
            it->second.features != entry.second.features) {
            continue;
        }
        ++compared;
        last_executed_id = static_cast<int>(entry.first);
        float worst = 0.0f;
        float scale = 1e-6f;
        int nan_ref = 0;
        int nan_cand = 0;
        for (std::size_t i = 0; i < entry.second.row0.size(); ++i) {
            const float a = entry.second.row0[i];
            const float b = it->second.row0[i];
            if (!std::isfinite(a)) ++nan_ref;
            if (!std::isfinite(b)) ++nan_cand;
            worst = std::max(worst, std::fabs(a - b));
            scale = std::max(scale, std::fabs(a));
        }
        const double rel = worst / scale;
        if (rel > 5e-3 || nan_ref + nan_cand != 0) {
            if (first_diverged_id < 0) first_diverged_id = static_cast<int>(entry.first);
            if (printed < 40) {
                std::printf("%8u  %5u %5u  %9.4g  %9.4g  %10.4g  %10.4g  %6d %6d\n", entry.first,
                            entry.second.rows, entry.second.features, rel,
                            static_cast<double>(worst), static_cast<double>(entry.second.row0[0]),
                            static_cast<double>(it->second.row0[0]), nan_ref, nan_cand);
                ++printed;
            }
        }
    }
    std::printf("%s: decode-step values compared=%d first diverged value_id=%d last traced "
                "value_id=%d\n",
                tag, compared, first_diverged_id, last_executed_id);
}

// Rebuilds the global channel order of one rank's state. The GDN split halves Q, K and V
// separately, so the local layout is [q_half | k_half | v_half] and each half lands at its own
// global offset. Attention value heads and KV heads are contiguous head ranges per rank.
struct GlobalState {
    std::uint32_t conv_dim = 0;
    std::vector<float> conv;
    std::uint32_t v_heads = 0;
    std::vector<float> recurrent;
    std::uint32_t kv_heads = 0;
    std::vector<float> kv_k;
    std::vector<float> kv_v;
};

void place_conv(const StateSnapshot& src, GlobalState& dst) {
    const std::uint32_t tp_size = src.tp_size;
    const std::uint32_t rank = src.tp_rank;
    const std::uint32_t local_qk = src.conv_qk_local;
    const std::uint32_t local_value = src.conv_v_local;
    const std::uint32_t qk_dim = local_qk * tp_size;
    const std::uint32_t v_dim = local_value * tp_size;
    dst.conv_dim = 2u * qk_dim + v_dim;
    if (dst.conv.empty())
        dst.conv.assign(static_cast<std::size_t>(src.gdn_layers) * src.conv_rows * dst.conv_dim,
                        0.0f);
    for (std::uint32_t li = 0; li < src.gdn_layers; ++li) {
        for (std::uint32_t h = 0; h < src.conv_rows; ++h) {
            const std::size_t src_base =
                (static_cast<std::size_t>(li) * src.conv_rows + h) * src.conv_dim;
            const std::size_t dst_base =
                (static_cast<std::size_t>(li) * src.conv_rows + h) * dst.conv_dim;
            for (std::uint32_t c = 0; c < local_qk; ++c) {
                dst.conv[dst_base + rank * local_qk + c] = src.conv[src_base + c];
                dst.conv[dst_base + qk_dim + rank * local_qk + c] = src.conv[src_base + local_qk + c];
            }
            for (std::uint32_t c = 0; c < local_value; ++c) {
                dst.conv[dst_base + 2u * qk_dim + rank * local_value + c] =
                    src.conv[src_base + 2u * local_qk + c];
            }
        }
    }
}

void place_recurrent(const StateSnapshot& src, GlobalState& dst) {
    const std::size_t per_layer_src =
        static_cast<std::size_t>(src.v_heads) * src.head_k * src.head_v;
    if (dst.recurrent.empty()) {
        dst.v_heads = src.v_heads * src.tp_size;
        const std::size_t per_layer =
            static_cast<std::size_t>(dst.v_heads) * src.head_k * src.head_v;
        dst.recurrent.assign(static_cast<std::size_t>(src.gdn_layers) * per_layer, 0.0f);
    }
    const std::size_t per_layer = static_cast<std::size_t>(dst.v_heads) * src.head_k * src.head_v;
    for (std::uint32_t li = 0; li < src.gdn_layers; ++li) {
        for (std::uint32_t vh = 0; vh < src.v_heads; ++vh) {
            const std::size_t src_base = li * per_layer_src + static_cast<std::size_t>(vh) *
                                                                  src.head_k * src.head_v;
            const std::size_t dst_base = li * per_layer +
                                         static_cast<std::size_t>(src.tp_rank * src.v_heads + vh) *
                                             src.head_k * src.head_v;
            for (std::size_t i = 0; i < src.head_k * src.head_v; ++i)
                dst.recurrent[dst_base + i] = src.recurrent[src_base + i];
        }
    }
}

void place_kv(const std::vector<float>& src, std::uint32_t rank, const StateSnapshot& snap,
              std::uint32_t tp_size, std::vector<float>& dst) {
    const std::uint32_t global_heads = snap.kv_heads * tp_size;
    if (dst.empty())
        dst.assign(static_cast<std::size_t>(snap.kv_layers) * snap.kv_tokens * global_heads *
                       snap.head_dim,
                   0.0f);
    for (std::uint32_t li = 0; li < snap.kv_layers; ++li) {
        for (std::uint32_t t = 0; t < snap.kv_tokens; ++t) {
            const std::size_t src_base =
                (static_cast<std::size_t>(li) * snap.kv_tokens + t) * snap.kv_heads * snap.head_dim;
            const std::size_t dst_base =
                (static_cast<std::size_t>(li) * snap.kv_tokens + t) * global_heads * snap.head_dim;
            for (std::uint32_t h = 0; h < snap.kv_heads; ++h) {
                const std::size_t global_head = rank * snap.kv_heads + h;
                for (std::uint32_t d = 0; d < snap.head_dim; ++d) {
                    dst[dst_base + static_cast<std::size_t>(global_head) * snap.head_dim + d] =
                        src[src_base + static_cast<std::size_t>(h) * snap.head_dim + d];
                }
            }
        }
    }
}

void compare_state(const StateSnapshot& reference, const StateSnapshot& rank0,
                   const StateSnapshot& rank1) {
    std::printf("--- state after prefill, TP=1 vs reassembled TP=2 ---\n");
    std::printf("tp1 position=%u pages=%zu | tp2 rank0 position=%u pages=%zu rank1 position=%u "
                "pages=%zu\n",
                reference.position, reference.pages.size(), rank0.position, rank0.pages.size(),
                rank1.position, rank1.pages.size());

    auto report = [&](const char* name, std::uint32_t units, const std::vector<float>& ref,
                      const std::vector<float>& cand, std::size_t unit_elems) {
        if (ref.size() != cand.size() || unit_elems == 0u) {
            std::printf("%s: size mismatch ref=%zu cand=%zu\n", name, ref.size(), cand.size());
            return;
        }
        for (std::uint32_t u = 0; u < units; ++u) {
            float worst = 0.0f;
            float scale = 1e-6f;
            int nan_ref = 0;
            int nan_cand = 0;
            const std::size_t base = static_cast<std::size_t>(u) * unit_elems;
            for (std::size_t i = 0; i < unit_elems; ++i) {
                const float a = ref[base + i];
                const float b = cand[base + i];
                if (!std::isfinite(a)) ++nan_ref;
                if (!std::isfinite(b)) ++nan_cand;
                worst = std::max(worst, std::fabs(a - b));
                scale = std::max(scale, std::fabs(a));
            }
            std::printf("  %-14s unit %2u: rel=%9.4g max|d|=%10.4g scale=%9.4g nan_ref=%d "
                        "nan_cand=%d\n",
                        name, u, static_cast<double>(worst / scale), static_cast<double>(worst),
                        static_cast<double>(scale), nan_ref, nan_cand);
        }
    };

    GlobalState ref_global;
    place_conv(reference, ref_global);
    GlobalState tp2_global;
    place_conv(rank0, tp2_global);
    place_conv(rank1, tp2_global);
    report("gdn conv", reference.gdn_layers, ref_global.conv, tp2_global.conv,
           static_cast<std::size_t>(reference.conv_rows) * ref_global.conv_dim);

    GlobalState ref_rec;
    place_recurrent(reference, ref_rec);
    GlobalState tp2_rec;
    place_recurrent(rank0, tp2_rec);
    place_recurrent(rank1, tp2_rec);
    report("gdn recurrent", reference.gdn_layers, ref_rec.recurrent, tp2_rec.recurrent,
           static_cast<std::size_t>(reference.v_heads) * reference.tp_size * reference.head_k *
               reference.head_v);

    GlobalState ref_kv;
    place_kv(reference.kv_k, 0u, reference, reference.tp_size, ref_kv.kv_k);
    place_kv(reference.kv_v, 0u, reference, reference.tp_size, ref_kv.kv_v);
    GlobalState tp2_kv;
    place_kv(rank0.kv_k, 0u, rank0, rank0.tp_size, tp2_kv.kv_k);
    place_kv(rank1.kv_k, 1u, rank1, rank1.tp_size, tp2_kv.kv_k);
    report("kv K", reference.kv_layers, ref_kv.kv_k, tp2_kv.kv_k,
           static_cast<std::size_t>(reference.kv_tokens) * reference.kv_heads *
               reference.tp_size * reference.head_dim);

    GlobalState tp2_kv_v;
    place_kv(rank0.kv_v, 0u, rank0, rank0.tp_size, tp2_kv_v.kv_v);
    place_kv(rank1.kv_v, 1u, rank1, rank1.tp_size, tp2_kv_v.kv_v);
    report("kv V", reference.kv_layers, ref_kv.kv_v, tp2_kv_v.kv_v,
           static_cast<std::size_t>(reference.kv_tokens) * reference.kv_heads *
               reference.tp_size * reference.head_dim);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::string tokens_file;
    std::uint32_t ctx = 192u;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (arg == "--tokens-file" && i + 1 < argc) tokens_file = argv[++i];
        else if (arg == "--context" && i + 1 < argc)
            ctx = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    }
    if (model_dir.empty()) {
        for (const char* key : {"SHELF_MODEL_DIR_MOE", "SHELF_MODEL_DIR"}) {
            if (const char* env = std::getenv(key); env != nullptr) {
                model_dir = env;
                break;
            }
        }
    }
    if (model_dir.empty()) {
        std::printf("SKIP: --model-dir or SHELF_MODEL_DIR_MOE / SHELF_MODEL_DIR is required\n");
        return 77;
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: 2 GPUs required\n");
        return 77;
    }

    auto config_result = moe::read_qwen35_moe_text_config(model_dir);
    if (!config_result.ok()) {
        std::printf("FAIL config: %s\n", config_result.status().message().c_str());
        return 1;
    }
    const moe::Qwen35MoeTextConfig config = config_result.release();

    std::vector<int32_t> corpus;
    if (!tokens_file.empty()) corpus = read_token_corpus(tokens_file);
    if (corpus.size() < ctx + 1u) {
        std::printf("SKIP: --tokens-file with a PSKLDTOK corpus of at least %u tokens is "
                    "required (see tools/quality_gate.py)\n",
                    ctx + 1u);
        return 77;
    }
    std::vector<int32_t> prompt(corpus.begin(), corpus.begin() + ctx);

    std::printf("=== MoE TP=1 vs TP=2 layer trace, ctx=%u, hidden=%zu, vocab=%zu ===\n", ctx,
                config.hidden_size, config.vocab_size);

    RunResult solo = run_rank(model_dir, 0, 1u, 0u, ctx, prompt, nullptr, true, true);
    std::printf("tp1 sampled:");
    for (int32_t t : solo.sampled) std::printf(" %d", t);
    std::printf("  traced values=%zu  state.valid=%d  error=%s\n", solo.values.size(),
                static_cast<int>(solo.state.valid), solo.error.c_str());
    if (solo.values.empty()) {
        std::printf("FAIL tp1: no trace\n");
        return 1;
    }

    auto transport = rt::create_tp_transport({0, 1});
    if (!transport.ok()) {
        std::printf("FAIL transport: %s\n", transport.status().message().c_str());
        return 1;
    }
    rt::TpTransportSelection selection = transport.release();
    rt::TpBarrierGroup group(2, std::vector<int>{0, 1}, selection.transport.get());
    if (!group.initialize().ok()) {
        std::printf("FAIL barrier initialize\n");
        return 1;
    }

    RunResult rank0;
    RunResult rank1;
    std::thread t0([&] { rank0 = run_rank(model_dir, 0, 2u, 0u, ctx, prompt, &group, true); });
    std::thread t1([&] { rank1 = run_rank(model_dir, 1, 2u, 1u, ctx, prompt, &group, true); });
    t0.join();
    t1.join();
    group.shutdown();

    std::printf("tp2 rank0 sampled:");
    for (int32_t t : rank0.sampled) std::printf(" %d", t);
    std::printf("  traced values=%zu state.valid=%d error=%s\n", rank0.values.size(),
                static_cast<int>(rank0.state.valid), rank0.error.c_str());
    std::printf("tp2 rank1 sampled:");
    for (int32_t t : rank1.sampled) std::printf(" %d", t);
    std::printf("  traced values=%zu state.valid=%d error=%s\n", rank1.values.size(),
                static_cast<int>(rank1.state.valid), rank1.error.c_str());

    if (solo.state.valid && rank0.state.valid && rank1.state.valid)
        compare_state(solo.state, rank0.state, rank1.state);
    else
        std::printf("state snapshot incomplete: tp1=%d rank0=%d rank1=%d\n",
                    static_cast<int>(solo.state.valid), static_cast<int>(rank0.state.valid),
                    static_cast<int>(rank1.state.valid));

    compare_traces(solo, rank0, "TP=1 vs TP=2 rank0");
    compare_traces(rank0, rank1, "rank0 vs rank1");
    return 0;
}
