#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/models/hy4_flash/generate.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>

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
                         const float *subtract = nullptr) {
    __shared__ float vals[256];
    __shared__ int ids[256];
    int tid = threadIdx.x, row = blockIdx.x, id = -1;
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
struct Target {
    Ops op;
    std::vector<int> history;
    std::array<Selection, 5> tapped;
    Mat state{};
    size_t base = 0;
    float *target_probs = nullptr;
    float *draft_probs = nullptr;
    float temperature;
    uint64_t seed;
    Target(Residency &gpu, float temp, uint64_t rng) : op(gpu), temperature(temp), seed(rng) {
        op.arena.cursor = 0;
        if (gpu.model.has_draft && temperature > 0) {
            target_probs = static_cast<float *>(op.arena.next(6 * 120832 * 4));
            draft_probs = static_cast<float *>(op.arena.next(5 * 120832 * 4));
        }
        gpu.routed_rows = {};
        gpu.transfer_bytes = {};
        state = op.mat(6, 3072);
        base = op.arena.cursor;
    }
    std::vector<int> predict(Mat hidden, int position, bool draft = false, const Path &dump = {},
                             float *distribution = nullptr) {
        size_t mark = op.arena.cursor;
        auto codes = static_cast<int8_t *>(op.arena.next(size_t(hidden.rows) * 3072));
        auto scales = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 24 * 4));
        auto partial = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 120832 * 4));
        auto logits = static_cast<float *>(op.arena.next(size_t(hidden.rows) * 120832 * 4));
        checked(shelf_hy4_head(hidden.p, static_cast<const uint16_t *>(op.param("lm_head.suh").p),
                               static_cast<const uint16_t *>(op.param("lm_head.svh").p),
                               static_cast<const uint32_t *>(op.param("lm_head.trellis").p), codes, scales,
                               partial, logits, hidden.rows));
        if (!dump.empty()) {
            std::vector<float> host(120832);
            op.copy(host.data(), logits + size_t(hidden.rows - 1) * 120832, host.size() * 4,
                    hipMemcpyDeviceToHost);
            std::ofstream file(dump, std::ios::binary);
            if (!file.write(reinterpret_cast<const char *>(host.data()), host.size() * 4))
                throw std::runtime_error("logits を保存できません");
        }
        if (distribution) {
            probabilities<<<hidden.rows, 256>>>(logits, distribution, temperature);
            checked(hipGetLastError());
        }
        int *device_ids = static_cast<int *>(op.arena.next(hidden.rows * 4));
        best_ids<<<hidden.rows, 256>>>(logits, device_ids, position, seed ^ (draft ? 0x735a7ec51b09ull : 0ull),
                                       temperature);
        checked(hipGetLastError());
        std::vector<int> result(hidden.rows);
        op.copy(result.data(), device_ids, result.size() * 4, hipMemcpyDeviceToHost);
        op.arena.cursor = mark;
        if (std::find(result.begin(), result.end(), -1) != result.end())
            throw std::runtime_error("有効な出力 logit がありません");
        return result;
    }
    std::vector<int> forward(const std::vector<int> &ids, bool verify, bool need_head = true,
                             const Path &dump = {}) {
        op.arena.cursor = base;
        int start = history.size();
        if (ids.empty() || start + ids.size() > op.gpu.tokens || (verify && ids.size() > 6))
            throw std::runtime_error("入力列が KV の範囲外です");
        Mat x = op.embed(ids, true);
        for (int layer = 0; layer < 42; ++layer) {
            Selection selection;
            decoder_layer(op, layer, start, x, history, ids, selection);
            auto it = std::find(sources.begin(), sources.end(), layer);
            if (it != sources.end()) {
                if (!verify)
                    selection = Selection{std::move(selection.back())};
                tapped[it - sources.begin()] = std::move(selection);
            }
        }
        history.insert(history.end(), ids.begin(), ids.end());
        if (!need_head)
            return {};
        if (!verify)
            x = x.row(x.rows - 1);
        float *unused;
        Mat h = op.norm(op.collapse(x, "model.hc_head.hc_head_", unused), "model.norm.weight");
        checked(hipMemcpyAsync(state.p, h.p, h.bytes(), hipMemcpyDeviceToDevice));
        return predict(h, history.size() - h.rows + 1, false, dump, verify ? target_probs : nullptr);
    }
    std::vector<int> propose(int anchor) {
        op.arena.cursor = base;
        std::vector<int> block(6, 120009);
        block[0] = anchor;
        Mat x = op.embed(block, false);
        std::string p = "model.dfly_layers.";
        auto e = op.norm(x.row(0), p + "mtp_fusion.enorm.weight");
        auto h = op.norm(state.row(0), p + "mtp_fusion.hnorm.weight");
        auto fused = op.linear(p + "mtp_fusion.eh_proj", op.join(e, h));
        fuse_anchor<<<12, 256>>>(x.row(0), fused,
                                 static_cast<const uint16_t *>(op.param(p + "mtp_fusion.gate").p));
        checked(hipGetLastError());
        int start = history.size();
        size_t save = op.arena.cursor;
        for (int i = 0; i < 5; ++i) {
            std::string prefix = p + std::to_string(i) + ".";
            Selection select(6, tapped[i][0]);
            for (int row = 0; row < 6; ++row)
                for (int j = 0; j <= row; ++j)
                    select[row].push_back(start + j);
            auto a = op.norm(x, prefix + "input_layernorm.weight");
            auto y = op.add(x, attention(op, prefix, sources[i], start, a, select, true));
            auto f = feedforward(op, prefix + "mlp", op.norm(y, prefix + "post_attention_layernorm.weight"));
            auto next = op.add(y, f);
            checked(hipMemcpyAsync(x.p, next.p, next.bytes(), hipMemcpyDeviceToDevice));
            op.arena.cursor = save;
        }
        x = op.norm(x, p + "final_norm.weight");
        std::string correction = p + "hidden_correction.";
        int previous = anchor;
        std::vector<int> draft;
        for (int i = 1; i <= 5; ++i) {
            auto q = op.norm(x.row(i), correction + "hidden_norm.weight");
            auto k = op.norm(op.embed({previous}, false), correction + "embed_norm.weight");
            auto corrected = op.add(x.row(i), feedforward(op, p + "hidden_correction", op.join(q, k)));
            previous = predict(corrected, start + i, true, {},
                               draft_probs ? draft_probs + (i - 1) * 120832 : nullptr)[0];
            draft.push_back(previous);
        }
        return draft;
    }
    void accept(int keep, int submitted) {
        history.resize(history.size() - submitted + keep);
        if (keep > 1)
            checked(hipMemcpyAsync(state.p, state.row(keep - 1).p, 3072 * 2, hipMemcpyDeviceToDevice));
        for (auto &s : tapped)
            s = Selection{std::move(s[keep - 1])};
    }
};
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
    test_drafts<<<1, 5>>>(target_probs, draft_probs, ids, position, seed, out);
    checked(hipGetLastError());
    int accepted[5];
    op.copy(accepted, out, sizeof(accepted), hipMemcpyDeviceToHost);
    for (int i = 0; i < 5; ++i) {
        if (accepted[i]) {
            actual[i] = draft[i];
            continue;
        }
        best_ids<<<1, 256>>>(target_probs + i * 120832, out, position + i, seed ^ 0xc1f8efb6703ull, 1.f,
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
    size_t save = op.arena.cursor;
    auto p = "model.layers." + std::to_string(layer) + ".";
    if (layer % 10 == 1 && layer <= 31)
        engram(op, layer, x, history, ids);
    float *post;
    Mat a = op.collapse(x, p + "hc_attn_layer.hc_pre.hc_", post);
    a = op.norm(a, p + "input_layernorm.weight");
    a = attention(op, p, layer, start, a, selection, false);
    Mat z = op.expand(x, a, post);
    checked(hipMemcpyAsync(x.p, z.p, z.bytes(), hipMemcpyDeviceToDevice));
    op.arena.cursor = save;
    Mat f = op.collapse(x, p + "hc_mlp_layer.hc_pre.hc_", post);
    f = op.norm(f, p + "post_attention_layernorm.weight");
    if (layer == 0)
        f = op.norm(feedforward(op, p + "mlp", f), p + "mlp.post_fc2_norm.weight");
    else
        f = moe(op, layer, f);
    z = op.expand(x, f, post);
    checked(hipMemcpyAsync(x.p, z.p, z.bytes(), hipMemcpyDeviceToDevice));
    op.arena.cursor = save;
}
void generate(Residency &gpu, const std::vector<int> &ids, int count, int chunk, float temperature,
              uint64_t seed, const Path &logits_file) {
    if (ids.empty() || !count || ids.size() + count > gpu.tokens || gpu.tokens > 1048576 || chunk < 1 ||
        chunk > 128)
        throw std::runtime_error("入力・出力長または prefill chunk が範囲外です");
    auto now = std::chrono::steady_clock::now;
    Target target(gpu, temperature, seed);
    auto begin = now();
    int next = 0;
    for (size_t first = 0; first < ids.size(); first += chunk) {
        auto end = std::min(first + chunk, ids.size());
        auto output =
            target.forward({ids.begin() + first, ids.begin() + end}, false, end == ids.size(), logits_file);
        if (!output.empty())
            next = output.back();
    }
    auto prefill = now();
    auto pp_routes = gpu.routed_rows;
    auto pp_bytes = gpu.transfer_bytes;
    std::vector<int> generated{next};
    int rounds = 0, accepted = 0;
    std::array<int, 6> histogram{};
    while (int(generated.size()) < count && generated.back() != 120025) {
        int anchor = generated.back();
        if (!gpu.model.has_draft || target.history.size() + 6 > gpu.tokens ||
            int(generated.size()) + 1 == count) {
            generated.push_back(target.forward({anchor}, false)[0]);
            continue;
        }
        auto draft = target.propose(anchor);
        std::vector<int> block{anchor};
        block.insert(block.end(), draft.begin(), draft.end());
        auto actual = target.forward(block, true);
        int keep = verify_speculation(target.op, target.target_probs, target.draft_probs,
                                      target.history.size() - 5, seed, draft, actual);
        ++rounds;
        accepted += keep - 1;
        ++histogram[keep - 1];
        target.accept(keep, 6);
        for (int i = 0; i < keep && int(generated.size()) < count; ++i) {
            generated.push_back(actual[i]);
            if (actual[i] == 120025)
                break;
        }
    }
    double pp = std::chrono::duration<double>(prefill - begin).count();
    double tg = std::chrono::duration<double>(now() - prefill).count();
    fprintf(stderr,
            "prefill_tokens=%zu prefill_s=%.6f decode_tokens=%zu decode_s=%.6f draft_rounds=%d accepted=%d\n",
            ids.size(), pp, generated.size() - 1, tg, rounds, accepted);
    fprintf(stderr, "accepted_histogram=%d,%d,%d,%d,%d,%d\n", histogram[0], histogram[1], histogram[2],
            histogram[3], histogram[4], histogram[5]);
    for (int r = 0; r < 3; ++r) {
        fprintf(stderr, "rank=%d prefill_choices=%llu decode_choices=%llu", r,
                static_cast<unsigned long long>(pp_routes[r]),
                static_cast<unsigned long long>(gpu.routed_rows[r] - pp_routes[r]));
        if (r)
            fprintf(stderr, " prefill_pcie_bytes=%llu decode_pcie_bytes=%llu",
                    static_cast<unsigned long long>(pp_bytes[0][r] + pp_bytes[r][0]),
                    static_cast<unsigned long long>(gpu.transfer_bytes[0][r] + gpu.transfer_bytes[r][0] -
                                                    pp_bytes[0][r] - pp_bytes[r][0]));
        fputc('\n', stderr);
    }
    printf("GENERATED_IDS=");
    for (size_t i = 0; i < generated.size(); ++i)
        printf("%s%d", i ? "," : "", generated[i]);
    puts("");
}
}
