#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/kernels/fused/detail/radix_select.h>
#include <shelf/models/hy4_flash/generate.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <numeric>
#include <shelf/models/hy4_flash/target.h>
#include <exception>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
constexpr std::array<int, 5> sources{25, 29, 33, 37, 41};
__device__ uint64_t mix(uint64_t v) {
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ull;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebull;
    return v ^ (v >> 31);
}
__global__ void best_ids(const float *logits, int *tokens, int position, uint64_t seed, float temperature,
                         const float *subtract = nullptr, const Hy4StepParams *params = nullptr) {
    __shared__ float vals[256];
    __shared__ int ids[256];
    int tid = threadIdx.x, row = blockIdx.x, id = -1;
    if (params) {
        position += params->position;
        seed ^= params->seed;
    }
    float best = -INFINITY;
    for (int c = tid; c < 120832; c += 256) {
        float v = logits[size_t(row) * 120832 + c];
        if (subtract) {
            float mass = v - subtract[c];
            if (mass <= 0)
                continue;
            v = logf(mass);
        }
        if (temperature > 0) {
            uint64_t bits = mix((uint64_t(position + row) << 32 | uint32_t(c)) ^ seed);
            float uniform = (float(bits >> 41) + 0.5f) * 0x1p-23f;
            v -= temperature * logf(-logf(uniform));
        }
        if (v > best) {
            best = v;
            id = c;
        }
    }
    vals[tid] = best;
    ids[tid] = id;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (tid < d && (vals[tid + d] > vals[tid] || (vals[tid + d] == vals[tid] && ids[tid + d] < ids[tid]))) {
            vals[tid] = vals[tid + d];
            ids[tid] = ids[tid + d];
        }
        __syncthreads();
    }
    if (!tid)
        tokens[row] = ids[0];
}
__global__ void truncate_logits(float *logits, int k, float p, float temperature) {
    __shared__ uint32_t histogram[256], state[4];
    __shared__ int ids[64];
    __shared__ float values[64], masses[64];
    float *row = logits + size_t(blockIdx.x) * 120832;
    ::shelf::kernel::detail::radix_select<3>(row, 120832, k, histogram, state, ids, values, 0);
    const int t = threadIdx.x;
    if (t < k) {
        float peak = values[0];
        for (int j = 1; j < k; ++j)
            peak = fmaxf(peak, values[j]);
        masses[t] = expf((values[t] - peak) / temperature);
    }
    __syncthreads();
    for (int i = t; i < 120832; i += blockDim.x)
        row[i] = -INFINITY;
    __syncthreads();
    if (t < k) {
        double before = 0, total = 0;
        for (int j = 0; j < k; ++j) {
            total += masses[j];
            if (values[j] > values[t] || (values[j] == values[t] && ids[j] < ids[t]))
                before += masses[j];
        }
        if (p == 1 || before < p * total)
            row[ids[t]] = values[t];
    }
}
__global__ void probabilities(const float *logits, float *out, float temp) {
    __shared__ float scratch[256];
    int t = threadIdx.x;
    logits += size_t(blockIdx.x) * 120832;
    out += size_t(blockIdx.x) * 120832;
    float peak = -INFINITY;
    for (int c = t; c < 120832; c += 256)
        peak = fmaxf(peak, logits[c]);
    scratch[t] = peak;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (t < n)
            scratch[t] = fmaxf(scratch[t], scratch[t + n]);
        __syncthreads();
    }
    peak = scratch[0];
    __syncthreads();
    float sum = 0;
    for (int c = t; c < 120832; c += 256) {
        out[c] = expf((logits[c] - peak) / temp);
        sum += out[c];
    }
    scratch[t] = sum;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (t < n)
            scratch[t] += scratch[t + n];
        __syncthreads();
    }
    for (int c = t; c < 120832; c += 256)
        out[c] /= scratch[0];
}
__global__ void test_drafts(const float *target, const float *draft, const int *ids, int position,
                            uint64_t seed, int *accept) {
    int i = threadIdx.x, token = ids[i];
    size_t offset = size_t(i) * 120832 + token;
    uint64_t bits = mix((uint64_t(position + i) << 32) ^ seed ^ 0xc343d97fb3efull);
    double u = (double(bits >> 12) + 0.5) * 0x1p-52;
    accept[i] = u * double(draft[offset]) <= double(target[offset]);
}
__global__ void fuse_anchor(Mat anchor, Mat fusion, const uint16_t *gate) {
    int i = blockIdx.x * 256 + threadIdx.x;
    anchor.p[i] = f32_to_bf16_rne_device(bf16_bits_to_f32(anchor.p[i]) +
                                         bf16_bits_to_f32(fusion.p[i]) * bf16_bits_to_f32(gate[i]));
}

template <size_t N, typename Body>
GraphPlan *capture_cached(Ops &op, std::array<GraphPlan, N> &slots, size_t &victim, bool &failed,
                          const GraphKey &key, Body body) {
    for (auto &slot : slots)
        if (slot.tail && slot.key == key)
            return &slot;
    auto &slot = slots[victim++ % N];
    std::string errors;
    if (slot.tail)
        slot.release(op.gpu, errors);
    if (!errors.empty())
        throw std::runtime_error(errors);
    GraphCapture capture(op.gpu, slot);
    auto last = std::to_array(op.sparse_last_layer);
    op.params = &op.gpu.graph_input->step;
    op.max_kv_len = key.bucket;
    op.capture = &capture;
    try {
        capture.begin(0);
        body(slot);
        capture.finish(slot.tail);
        slot.key = key;
    } catch (const std::exception &e) {
        capture.cancel(errors);
        slot.release(op.gpu, errors);
        failed = true;
        (void)hipGetLastError();
        fprintf(stderr, "Hy4 Graph の capture に失敗したため eager に切り替えます: %s\n", e.what());
    }
    op.params = nullptr;
    op.capture = nullptr;
    op.max_kv_len = 0;
    std::copy(last.begin(), last.end(), op.sparse_last_layer);
    if (!errors.empty())
        throw std::runtime_error(errors);
    return failed ? nullptr : &slot;
}

bool graph_enabled(const char *name) {
    auto value = std::getenv(name);
    return !value || std::string_view(value) != "0";
}
}

Target::Target(Residency &gpu, float temp, uint64_t rng, int k, float p, int rank)
    : op(gpu, rank), temperature(temp), top_k(k), top_p(p), seed(rng) {
    op.arena.cursor = 0;
    if (gpu.model.has_draft && temperature > 0) {
        target_probs = static_cast<float *>(op.arena.next(6 * 120832 * 4));
        draft_probs = static_cast<float *>(op.arena.next(5 * 120832 * 4));
    }
    checked(hipMemsetAsync(gpu.moe_counts[rank], 0, 4 * sizeof(uint64_t), op.stream));
    if (gpu.tp_size == 1)
        gpu.routed_rows.fill(0);
    else
        gpu.routed_rows[rank] = 0;
    gpu.expert_ms[rank] = 0;
    if (rank == 0) {
        gpu.attention_ms = gpu.ffn_ms = gpu.transfer_ms = 0;
        gpu.transfer_bytes = {};
    }
    state = op.mat(6, 3072);
    base = op.arena.cursor;
    bool room = gpu.graph_room && !gpu.expert_begin[rank];
    target_graph = graph_topology(gpu.ep_size, gpu.tp_size, gpu.push, room, gpu.peer) && graph_enabled("SHELF_HY4_GRAPH");
    draft_graph = gpu.tp_size == 1 && gpu.model.has_draft && room && graph_enabled("SHELF_HY4_DRAFT_GRAPH");
    if ((target_graph || draft_graph) && !gpu.graph_input) {
        gpu.graph_input = static_cast<GraphInput *>(op.reserve_persistent(sizeof(GraphInput)));
        if (gpu.tokens > 2048)
            for (int layer = 0; layer < 42; ++layer)
                if (gpu.index_keys[0][layer] && !gpu.sparse_top[0][layer])
                    gpu.sparse_top[0][layer] = static_cast<int *>(op.reserve_persistent(size_t(max_prefill_chunk) * 511 * sizeof(int)));
    }
}
float *Target::head_logits(Mat hidden) {
    auto codes = static_cast<int8_t *>(op.arena.next(size_t(hidden.rows) * 3072));
    auto scales = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 24 * 4));
    auto partial = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 120832 * 4));
    auto logits = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 120832 * 4));
    checked(shelf_hy4_head(hidden.p, static_cast<const uint16_t *>(op.param("lm_head.suh").p),
                           static_cast<const uint16_t *>(op.param("lm_head.svh").p),
                           static_cast<const uint32_t *>(op.param("lm_head.trellis").p), codes, scales,
                           partial, logits, hidden.rows, op.stream));
    return logits;
}
std::vector<int> Target::predict(Mat hidden, int position, bool draft, const Path &dump,
                                 float *distribution, float *logits) {
    std::chrono::steady_clock::time_point begin;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0])
        begin = std::chrono::steady_clock::now();
    size_t mark = op.arena.cursor;
    if (!logits)
        logits = head_logits(hidden);
    if (!dump.empty()) {
        std::vector<float> host(120832);
        op.copy(host.data(), logits + size_t(hidden.rows - 1) * 120832, host.size() * 4,
                hipMemcpyDeviceToHost);
        std::ofstream file(dump, std::ios::binary);
        if (!file.write(reinterpret_cast<const char *>(host.data()), host.size() * 4))
            throw std::runtime_error("logits を保存できません");
    }
    auto device_ids = static_cast<int *>(op.arena.next(hidden.rows * sizeof(int)));
    sample(logits, hidden.rows, position, seed ^ (draft ? 0x735a7ec51b09ull : 0ull), device_ids, distribution);
    std::vector<int> result(hidden.rows);
    op.copy(result.data(), device_ids, result.size() * 4, hipMemcpyDeviceToHost);
    op.arena.cursor = mark;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0])
        head_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (std::find(result.begin(), result.end(), -1) != result.end())
        throw std::runtime_error("有効な出力 logit がありません");
    return result;
}
void Target::sample(float *logits, int rows, int position, uint64_t rng, int *ids, float *distribution,
                    const Hy4StepParams *params) {
    if (temperature > 0 && top_k > 0) {
        truncate_logits<<<rows, 256, 0, op.stream>>>(logits, top_k, top_p, temperature);
        checked(hipGetLastError());
    }
    if (distribution) {
        probabilities<<<rows, 256, 0, op.stream>>>(logits, distribution, temperature);
        checked(hipGetLastError());
    }
    best_ids<<<rows, 256, 0, op.stream>>>(logits, ids, position, rng, temperature, nullptr, params);
    checked(hipGetLastError());
}
Mat Target::layers(const std::vector<int> &ids, bool verify, bool need_head) {
    int start = history.size();
    Mat x = op.capture ? op.embed_device(op.params->ids, ids.size(), true) : op.embed(ids, true);
    for (int layer = 0; layer < 42; ++layer) {
        Selection selection;
        decoder_layer(op, layer, start, x, history, ids, selection);
        if (!layer && op.capture) {
            op.capture->finish(op.capture->plan.entry);
            op.capture->begin(0);
        }
        auto it = std::find(sources.begin(), sources.end(), layer);
        if (!op.capture && it != sources.end()) {
            if (!verify)
                selection = Selection{std::move(selection.back())};
            tapped[it - sources.begin()] = std::move(selection);
        }
    }
    if (!need_head)
        return {};
    if (!verify)
        x = x.row(x.rows - 1);
    float *head_post;
    Mat h = op.collapse_norm(x, "model.hc_head.hc_head_", head_post, "model.norm.weight");
    checked(hipMemcpyAsync(state.p, h.p, h.bytes(), hipMemcpyDeviceToDevice, op.stream));
    return h;
}
std::vector<int> Target::forward(const std::vector<int> &ids, bool verify, bool need_head, const Path &dump) {
    const int rows = int(ids.size()), start = int(history.size());
    if (ids.empty() || history.size() + ids.size() > op.gpu.tokens || (verify && rows > 6))
        throw std::runtime_error("入力列が KV の範囲外です");
    if (std::any_of(ids.begin(), ids.end(), [](int id) { return id < 0 || id >= 120832; }))
        throw std::runtime_error("token ID が範囲外です");
    auto &gpu = op.gpu;
    op.arena.cursor = base;
    if (target_graph && !gpu.target_capture_failed && need_head && Hy4MoePlan::supports(rows) && (verify || rows == 1)) {
        const int dense = std::clamp(2048 - start, 0, rows);
        int bucket = std::max(2048, (start + rows + 1023) / 1024 * 1024);
        GraphKey key{rows, dense, bucket, verify, base, 0, 0, 0};
        auto plan = capture_cached(op, gpu.target_graphs, gpu.target_victim, gpu.target_capture_failed, key,
            [&](GraphPlan &slot) {
                slot.logits = head_logits(layers(ids, verify, true));
                slot.cursor = op.arena.cursor;
            });
        if (plan) {
            GraphInput input{};
            input.step.position = start;
            input.step.rows = rows;
            std::copy(ids.begin(), ids.end(), input.step.ids);
            checked(hipMemcpyAsync(&gpu.graph_input->step, &input.step, sizeof(input.step),
                                   hipMemcpyHostToDevice, op.stream));
            checked(hipGraphLaunch(plan->entry, op.stream));
            for (int g = 0; g < 4; ++g) {
                auto packed = gather_engram(gpu.model, 1 + g * 10, rows, ids, history);
                std::copy(packed.begin(), packed.end(), input.engram[g]);
            }
            checked(hipMemcpyAsync(gpu.graph_input->engram, input.engram, sizeof(input.engram),
                                   hipMemcpyHostToDevice, op.stream));
            plan->launch(gpu);
            ++gpu.target_replays;
            for (size_t k = 0; k < sources.size(); ++k) {
                op.sparse_last_layer[sources[k]] = rows - 1;
                tapped[k].assign(rows, {});
                for (int i = 0; i < dense; ++i) {
                    tapped[k][i].resize(start + i + 1);
                    std::iota(tapped[k][i].begin(), tapped[k][i].end(), 0);
                }
            }
            history.insert(history.end(), ids.begin(), ids.end());
            op.arena.cursor = plan->cursor;
            return predict(state.row(0, rows), start + 1, false, dump, verify ? target_probs : nullptr, plan->logits);
        }
        op.arena.cursor = base;
    }
    std::vector<int> prior;
    if (history.size() > 3)
        prior.assign(history.end() - 3, history.end());
    else
        prior = history;
    auto prefetch = std::getenv("SHELF_HY4_ENGRAM_PREFETCH");
    if (!prefetch || prefetch[0] != '0')
        op.prefetch_engram(ids, prior);
    Mat h = layers(ids, verify, need_head);
    history.insert(history.end(), ids.begin(), ids.end());
    return need_head ? predict(h, history.size() - h.rows + 1, false, dump, verify ? target_probs : nullptr)
                     : std::vector<int>{};
}
std::vector<int> Target::draft_steps(int anchor, int *device_ids) {
    op.arena.cursor = base;
    Mat x;
    if (device_ids)
        x = op.embed_device(device_ids, 6, false);
    else {
        std::vector<int> block(6, 120009);
        block[0] = anchor;
        x = op.embed(block, false);
    }
    std::string p = "model.dfly_layers.";
    auto e = op.norm(x.row(0), p + "mtp_fusion.enorm.weight");
    auto h = op.norm(state.row(0), p + "mtp_fusion.hnorm.weight");
    auto fused = op.linear(p + "mtp_fusion.eh_proj", op.join(e, h));
    fuse_anchor<<<12, 256, 0, op.stream>>>(x.row(0), fused,
                             static_cast<const uint16_t *>(op.param(p + "mtp_fusion.gate").p));
    checked(hipGetLastError());
    int start = history.size();
    size_t save = op.arena.cursor;
    for (int i = 0; i < 5; ++i) {
        std::string prefix = p + std::to_string(i) + ".";
        Selection select;
        if (!device_ids) {
            select.assign(6, tapped[i][0]);
            for (int row = 0; row < 6; ++row)
                for (int j = 0; j <= row; ++j)
                    select[row].push_back(start + j);
        }
        auto a = op.norm(x, prefix + "input_layernorm.weight");
        auto y = op.add(x, attention(op, prefix, sources[i], start, a, select, true));
        auto f = feedforward(op, prefix + "mlp", op.norm(y, prefix + "post_attention_layernorm.weight"));
        op.add_into(y, f, x);
        op.arena.cursor = save;
    }
    x = op.norm(x, p + "final_norm.weight");
    std::string correction = p + "hidden_correction.";
    int previous = anchor;
    std::vector<int> draft;
    for (int i = 1; i <= 5; ++i) {
        auto q = op.norm(x.row(i), correction + "hidden_norm.weight");
        Mat embedding;
        if (device_ids) {
            embedding = op.embed_device(device_ids + i - 1, 1, false);
        } else {
            embedding = op.embed({previous}, false);
        }
        auto k = op.norm(embedding, correction + "embed_norm.weight");
        auto corrected = op.add(x.row(i), feedforward(op, p + "hidden_correction", op.join(q, k)));
        float *distribution = draft_probs ? draft_probs + (i - 1) * 120832 : nullptr;
        if (device_ids) {
            size_t mark = op.arena.cursor;
            sample(head_logits(corrected), 1, i, 0x735a7ec51b09ull, device_ids + i, distribution, op.params);
            op.arena.cursor = mark;
        } else {
            previous = predict(corrected, start + i, true, {}, distribution)[0];
            draft.push_back(previous);
        }
    }
    return draft;
}
std::vector<int> Target::propose(int anchor) {
    if (anchor < 0 || anchor >= 120832)
        throw std::runtime_error("token ID が範囲外です");
    auto &gpu = op.gpu;
    if (gpu.draft_capture_failed || !draft_graph)
        return draft_steps(anchor, nullptr);
    int start = history.size();
    bool dense = start <= 2048;
    int bucket = dense ? 2048 : (start + 6 + 1023) / 1024 * 1024;
    GraphKey key{6, int(dense), bucket, false, base, temperature, top_p, top_k};
    auto plan = capture_cached(op, gpu.draft_graphs, gpu.draft_victim, gpu.draft_capture_failed, key,
        [&](GraphPlan &) { draft_steps(anchor, gpu.graph_input->step.ids); });
    if (!plan)
        return draft_steps(anchor, nullptr);
    Hy4StepParams params{};
    params.position = start;
    params.rows = 6;
    params.seed = seed;
    params.draft_last = op.sparse_last_layer[sources[0]];
    params.ids[0] = anchor;
    std::fill_n(params.ids + 1, 5, 120009);
    checked(hipMemcpyAsync(&gpu.graph_input->step, &params, sizeof(params), hipMemcpyHostToDevice, op.stream));
    checked(hipGraphLaunch(plan->tail, op.stream));
    ++gpu.draft_replays;
    std::vector<int> result(5);
    op.copy(result.data(), gpu.graph_input->step.ids + 1, 5 * sizeof(int), hipMemcpyDeviceToHost);
    if (std::find(result.begin(), result.end(), -1) != result.end())
        throw std::runtime_error("有効な出力 logit がありません");
    return result;
}
void Target::accept(int keep, int submitted) {
    history.resize(history.size() - submitted + keep);
    if (keep > 1)
        checked(hipMemcpyAsync(state.p, state.row(keep - 1).p, 3072 * 2, hipMemcpyDeviceToDevice, op.stream));
    for (auto layer : sources)
        op.sparse_last_layer[layer] = keep - 1;
    for (auto &s : tapped)
        s = Selection{std::move(s[keep - 1])};
}

int verify_speculation(Ops &op, const float *target_probs, const float *draft_probs, int position,
                       uint64_t seed, const std::vector<int> &draft, std::vector<int> &actual) {
    if (!target_probs) {
        int keep = 1;
        while (keep < 6 && actual[keep - 1] == draft[keep - 1])
            ++keep;
        return keep;
    }
    auto ids = op.indices(draft);
    auto out = static_cast<int *>(op.arena.next(5 * sizeof(int)));
    test_drafts<<<1, 5, 0, op.stream>>>(target_probs, draft_probs, ids, position, seed, out);
    checked(hipGetLastError());
    int accepted[5];
    op.copy(accepted, out, sizeof(accepted), hipMemcpyDeviceToHost);
    for (int i = 0; i < 5; ++i) {
        if (accepted[i]) {
            actual[i] = draft[i];
            continue;
        }
        best_ids<<<1, 256, 0, op.stream>>>(target_probs + i * 120832, out, position + i, seed ^ 0xc1f8efb6703ull, 1.f,
                             draft_probs + i * 120832);
        checked(hipGetLastError());
        int replacement;
        op.copy(&replacement, out, sizeof(int), hipMemcpyDeviceToHost);
        if (replacement >= 0)
            actual[i] = replacement;
        return i + 1;
    }
    return 6;
}


void decoder_layer(Ops &op, int layer, int start, Mat x, const std::vector<int> &history,
                   const std::vector<int> &ids, Selection &selection) {
    auto now = std::chrono::steady_clock::now;
    std::chrono::steady_clock::time_point start_time;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0]) {
        checked(hipStreamSynchronize(op.stream));
        start_time = now();
    }
    size_t save = op.arena.cursor;
    auto p = "model.layers." + std::to_string(layer) + ".";
    if (layer % 10 == 1 && layer <= 31) {
        std::vector<int> prior;
        if (history.size() > 3)
            prior.assign(history.end() - 3, history.end());
        else
            prior = history;
        engram(op, layer, x, prior, ids);
    }
    float *post;
    Mat a = op.collapse_norm(x, p + "hc_attn_layer.hc_pre.hc_", post, p + "input_layernorm.weight");
    a = attention(op, p, layer, start, a, selection, false);
    op.expand(x, a, post);
    op.arena.cursor = save;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0]) {
        checked(hipStreamSynchronize(op.stream));
        auto end = now();
        op.gpu.attention_ms += std::chrono::duration<double, std::milli>(end - start_time).count();
        start_time = end;
    }
    Mat f = op.collapse_norm(x, p + "hc_mlp_layer.hc_pre.hc_", post, p + "post_attention_layernorm.weight");
    if (layer == 0)
        f = op.norm(feedforward(op, p + "mlp", f), p + "mlp.post_fc2_norm.weight");
    else
        f = moe(op, layer, f);
    op.expand(x, f, post);
    op.arena.cursor = save;
    if (op.gpu.tp_size == 1 && op.rank == 0 && op.gpu.expert_begin[0]) {
        checked(hipStreamSynchronize(op.stream));
        op.gpu.ffn_ms += std::chrono::duration<double, std::milli>(now() - start_time).count();
    }
}
std::vector<int> generate(Residency &gpu, const std::vector<int> &ids, int count, int chunk, float temperature,
              uint64_t seed, const Path &logits_file, int top_k, float top_p, const std::function<void(int)> &emit) {
    if (ids.empty() || !count || ids.size() + count > gpu.tokens || gpu.tokens > 1048576 || chunk < 1 ||
        chunk > max_prefill_chunk)
        throw std::runtime_error("入力・出力長または prefill chunk が範囲外です");
    if (top_k < 0 || top_k > 64 || !(top_p > 0 && top_p <= 1) || (temperature > 0 && top_p < 1 && !top_k))
        throw std::invalid_argument("top_k は 0〜64、top_p を使う場合は top_k が必要です");
    struct Lane {
        std::vector<int> generated;
        double prefill = 0, decode = 0, draft = 0, head = 0;
        double pp_attn = 0, pp_ffn = 0, pp_transfer = 0, pp_head = 0;
        uint64_t pp_routes = 0;
        double pp_expert = 0;
        std::array<uint64_t, 4> pp_routes_all{};
        std::array<double, 4> pp_expert_all{};
        int rounds = 0, accepted = 0;
        std::array<int, 6> histogram{};
    };
    std::array<Lane, 4> lane;
    std::exception_ptr failure;
    bool failure_is_abort = false;
    std::mutex failure_mutex;
    std::vector<int> shared_tokens;
    int shared_keep = 0;
    gpu.reset_collectives();
    auto run = [&](int rank) {
        try {
            checked(hipSetDevice(gpu.physical[rank]));
            auto now = std::chrono::steady_clock::now;
            Target target(gpu, temperature, seed, top_k, top_p, rank);
            auto begin = now();
            int next = 0;
            for (size_t first = 0; first < ids.size(); first += chunk) {
                auto end = std::min(first + chunk, ids.size());
                auto output = target.forward({ids.begin() + first, ids.begin() + end}, false,
                                             end == ids.size(), rank == 0 ? logits_file : Path{});
                if (!output.empty())
                    next = output.back();
            }
            gpu.collect_route_counts(rank);
            auto prefill = now();
            lane[rank].pp_attn = gpu.attention_ms;
            lane[rank].pp_ffn = gpu.ffn_ms;
            lane[rank].pp_transfer = gpu.transfer_ms;
            lane[rank].pp_head = target.head_ms;
            lane[rank].pp_routes = gpu.routed_rows[rank];
            lane[rank].pp_expert = gpu.expert_ms[rank];
            if (rank == 0 && gpu.tp_size == 1) {
                lane[0].pp_routes_all = gpu.routed_rows;
                lane[0].pp_expert_all = gpu.expert_ms;
            }
            auto broadcast = [&](std::vector<int> &tokens, int &keep) {
                if (gpu.tp_size == 1) return;
                if (rank == 0) {
                    shared_tokens = tokens;
                    shared_keep = keep;
                }
                gpu.all_barrier->arrive_and_wait();
                tokens = shared_tokens;
                keep = shared_keep;
                gpu.all_barrier->arrive_and_wait();
            };
            std::vector<int> initial{next};
            int shared_value = 0;
            if (rank == 0 && emit) emit(next);
            broadcast(initial, shared_value);
            std::vector<int> generated{initial[0]};
            auto append = [&](int id) {
                generated.push_back(id);
                if (rank == 0 && emit) emit(id);
            };
            auto &stats = lane[rank];
            while (int(generated.size()) < count && generated.back() != 120025) {
                int anchor = generated.back();
                if (!gpu.model.has_draft || target.history.size() + 6 > gpu.tokens ||
                    int(generated.size()) + 1 == count) {
                    auto output = target.forward({anchor}, false);
                    if (rank == 0 && emit) emit(output[0]);
                    broadcast(output, shared_value);
                    generated.push_back(output[0]);
                    continue;
                }
                std::chrono::steady_clock::time_point draft_begin;
                if (gpu.tp_size == 1 && target.op.rank == 0 && gpu.expert_begin[0])
                    draft_begin = now();
                auto draft = target.propose(anchor);
                if (gpu.tp_size == 1 && target.op.rank == 0 && gpu.expert_begin[0])
                    stats.draft += std::chrono::duration<double, std::milli>(now() - draft_begin).count();
                broadcast(draft, shared_value);
                std::vector<int> block{anchor};
                block.insert(block.end(), draft.begin(), draft.end());
                auto actual = target.forward(block, true);
                int keep = verify_speculation(target.op, target.target_probs, target.draft_probs,
                                              target.history.size() - 5, seed, draft, actual);
                broadcast(actual, keep);
                ++stats.rounds;
                stats.accepted += keep - 1;
                ++stats.histogram[keep - 1];
                target.accept(keep, 6);
                for (int i = 0; i < keep && int(generated.size()) < count; ++i) {
                    append(actual[i]);
                    if (actual[i] == 120025)
                        break;
                }
            }
            gpu.collect_route_counts(rank);
            stats.generated = std::move(generated);
            stats.head = target.head_ms;
            stats.prefill = std::chrono::duration<double>(prefill - begin).count();
            stats.decode = std::chrono::duration<double>(now() - prefill).count();
        } catch (...) {
            auto error = std::current_exception();
            bool aborted = false;
            try { std::rethrow_exception(error); }
            catch (const CollectiveAborted &) { aborted = true; }
            catch (...) {}
            gpu.abort_collectives();
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure || (failure_is_abort && !aborted)) {
                failure = error;
                failure_is_abort = aborted;
            }
        }
    };
    if (gpu.tp_size == 1) {
        run(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(gpu.ranks);
        try {
            for (int r = 0; r < gpu.ranks; ++r)
                threads.emplace_back(run, r);
        } catch (...) {
            gpu.abort_collectives();
            for (auto &thread : threads)
                thread.join();
            throw;
        }
        for (auto &thread : threads)
            thread.join();
    }
    if (failure)
        std::rethrow_exception(failure);
    const auto &stats = lane[0];
    fprintf(stderr,
            "prefill_tokens=%zu prefill_s=%.6f decode_tokens=%zu decode_s=%.6f draft_rounds=%d accepted=%d\n",
            ids.size(), stats.prefill, stats.generated.size() - 1, stats.decode, stats.rounds, stats.accepted);
    fprintf(stderr, "accepted_histogram=%d,%d,%d,%d,%d,%d\n", stats.histogram[0], stats.histogram[1],
            stats.histogram[2], stats.histogram[3], stats.histogram[4], stats.histogram[5]);
    for (int r = 0; r < gpu.ranks; ++r) {
        const uint64_t prefill_routes = gpu.tp_size == 1 ? stats.pp_routes_all[r] : lane[r].pp_routes;
        const double prefill_expert = gpu.tp_size == 1 ? stats.pp_expert_all[r] : lane[r].pp_expert;
        fprintf(stderr, "rank=%d prefill_choices=%llu decode_choices=%llu", r,
                static_cast<unsigned long long>(prefill_routes),
                static_cast<unsigned long long>(gpu.routed_rows[r] - prefill_routes));
        if (gpu.expert_begin[r])
            fprintf(stderr, " prefill_expert_ms=%.4f decode_expert_ms=%.4f", prefill_expert,
                    gpu.expert_ms[r] - prefill_expert);
        fputc('\n', stderr);
    }
    if (gpu.expert_begin[0])
        fprintf(stderr,
                "profile_pp_ms attention=%.3f ffn=%.3f transfer=%.3f head=%.3f\n"
                "profile_tg_ms attention=%.3f ffn=%.3f transfer=%.3f head=%.3f draft=%.3f\n",
                stats.pp_attn, stats.pp_ffn, stats.pp_transfer, stats.pp_head,
                gpu.attention_ms - stats.pp_attn, gpu.ffn_ms - stats.pp_ffn,
                gpu.transfer_ms - stats.pp_transfer, stats.head - stats.pp_head, stats.draft);
    return lane[0].generated;
}
}
