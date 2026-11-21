#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Options {
    std::vector<uint32_t> rows = {1u, 8u, 64u};
    std::vector<uint32_t> blocks = {16384u, 32768u, 65536u};
    uint32_t top_k = 512u;
    uint32_t compress = 4u;
    int warmup = 20;
    int samples = 20;
    int launches = 10;
    int device = 0;
    uint32_t legacy_merge = 0u;
    std::string out;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: shelf-bench qsa-select [options]\n"
                 "  --rows 1,8,...\n"
                 "  --blocks 16384,32768,65536   score blocks (context / compress)\n"
                 "  --top-k N\n"
                 "  --compress N\n"
                 "  --warmup N\n"
                 "  --samples N\n"
                 "  --launches N\n"
                 "  --device N\n"
                 "  --legacy-merge 0|1           use the bitonic sort + merge merge path\n"
                 "  --out PATH\n");
}

bool parse_list(const std::string& s, std::vector<uint32_t>& out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        size_t j = i;
        while (j < s.size() && std::isdigit((unsigned char)s[j])) ++j;
        if (j == i) return false;
        out.push_back((uint32_t)std::stoul(s.substr(i, j - i)));
        i = j;
        if (i < s.size() && s[i] == ',') ++i;
    }
    return !out.empty();
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p * (double)(v.size() - 1u);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min(lo + 1u, v.size() - 1u);
    const double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

uint64_t h64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

template <typename T>
void release(T*& ptr) {
    if (ptr != nullptr) {
        shelf::gpu::discard_cleanup_result(hipFree(ptr));
        ptr = nullptr;
    }
}

}  // namespace

int run_qsa_select(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        }
        if (i + 1 >= argc) return 2;
        const char* v = argv[++i];
        if (!std::strcmp(a, "--rows")) {
            if (!parse_list(v, o.rows)) return 2;
        } else if (!std::strcmp(a, "--blocks")) {
            if (!parse_list(v, o.blocks)) return 2;
        } else if (!std::strcmp(a, "--top-k")) {
            o.top_k = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--compress")) {
            o.compress = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--warmup")) {
            o.warmup = std::stoi(v);
        } else if (!std::strcmp(a, "--samples")) {
            o.samples = std::stoi(v);
        } else if (!std::strcmp(a, "--launches")) {
            o.launches = std::stoi(v);
        } else if (!std::strcmp(a, "--device")) {
            o.device = std::stoi(v);
        } else if (!std::strcmp(a, "--legacy-merge")) {
            o.legacy_merge = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--out")) {
            o.out = v;
        } else {
            return 2;
        }
    }
    if (o.rows.empty() || o.blocks.empty() || o.top_k == 0u || o.compress == 0u ||
        o.launches <= 0 || o.samples <= 0 || o.warmup < 0) {
        return 2;
    }
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    constexpr uint32_t kHeadDim = shelf::kernel::kQsaHeadDim;
    constexpr uint32_t kHeads = 4u;
    constexpr uint32_t kWindow = shelf::kernel::kQsaScoreBlockTile;
    uint32_t half = shelf::kernel::kQsaSelectChunk;
    while (half < o.top_k) half <<= 1u;
    const uint32_t state_count = half * 2u;

    int rc = 0;
    for (uint32_t blocks : o.blocks) {
        shelf::bf16_t* q = nullptr;
        shelf::bf16_t* pooled = nullptr;
        uint32_t* slots = nullptr;
        uint32_t* positions = nullptr;
        float* scores = nullptr;
        float* state_scores = nullptr;
        int32_t* state_indices = nullptr;
        int32_t* selected = nullptr;
        uint8_t* mask = nullptr;
        for (uint32_t rows : o.rows) {
            const uint32_t mask_row_bytes = ((blocks + 31u) / 32u) * 4u;
            bool ok = hipMalloc(&q, (size_t)rows * kHeads * kHeadDim * sizeof(shelf::bf16_t)) == hipSuccess;
            ok = ok && hipMalloc(&pooled, (size_t)2u * blocks * kHeadDim * sizeof(shelf::bf16_t)) == hipSuccess;
            ok = ok && hipMalloc(&slots, (size_t)rows * sizeof(uint32_t)) == hipSuccess;
            ok = ok && hipMalloc(&positions, (size_t)rows * sizeof(uint32_t)) == hipSuccess;
            ok = ok && hipMalloc(&scores, (size_t)rows * kWindow * sizeof(float)) == hipSuccess;
            ok = ok && hipMalloc(&state_scores, (size_t)rows * state_count * sizeof(float)) == hipSuccess;
            ok = ok && hipMalloc(&state_indices, (size_t)rows * state_count * sizeof(int32_t)) == hipSuccess;
            ok = ok && hipMalloc(&selected, (size_t)rows * o.top_k * sizeof(int32_t)) == hipSuccess;
            ok = ok && hipMalloc(&mask, (size_t)rows * mask_row_bytes) == hipSuccess;
            if (!ok) {
                rc = 1;
                break;
            }
            {
                std::vector<shelf::bf16_t> hq((size_t)rows * kHeads * kHeadDim);
                std::vector<shelf::bf16_t> hp((size_t)2u * blocks * kHeadDim);
                std::vector<uint32_t> hs(rows, 0u);
                std::vector<uint32_t> hpos(rows, blocks * o.compress - 1u);
                uint64_t seed = 0x9E3779B97F4A7C15ull;
                for (auto& value : hq) {
                    seed = h64(seed);
                    value.data = shelf::f32_to_bf16_rne(((int32_t)(seed % 2001u) - 1000) / 1000.0f);
                }
                for (auto& value : hp) {
                    seed = h64(seed);
                    value.data = shelf::f32_to_bf16_rne(((int32_t)(seed % 2001u) - 1000) / 1000.0f);
                }
                ok = hipMemcpy(q, hq.data(), hq.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice) == hipSuccess;
                ok = ok && hipMemcpy(pooled, hp.data(), hp.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice) == hipSuccess;
                ok = ok && hipMemcpy(slots, hs.data(), hs.size() * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess;
                ok = ok && hipMemcpy(positions, hpos.data(), hpos.size() * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess;
                if (!ok) {
                    rc = 1;
                    break;
                }
            }
            const uint32_t block_tile = rows < 64u
                ? 64u
                : (rows < 256u ? 256u : shelf::kernel::kQsaScoreBlocksPerCta);
            const float scale = 1.0f / std::sqrt((float)kHeadDim);
            const uint32_t mask_row_bytes_ = mask_row_bytes;

            auto run_windows = [&](bool score_pass) {
                for (uint32_t bt = 0u; bt < blocks; bt += kWindow) {
                    const uint32_t window = std::min(kWindow, blocks - bt);
                    if (score_pass) {
                        shelf::kernel::QsaScoreArgs sa{};
                        sa.q = q;
                        sa.pooled_base = pooled;
                        sa.scores = scores;
                        sa.row_sequence_slots = slots;
                        sa.rows = rows;
                        sa.row_begin = 0u;
                        sa.block_begin = bt;
                        sa.blocks = window;
                        sa.block_tile = block_tile;
                        sa.score_row_stride = kWindow;
                        sa.n_heads = kHeads;
                        sa.head_dim = kHeadDim;
                        sa.compress_ratio = o.compress;
                        sa.state_index = 1u;
                        sa.max_sequences = 1u;
                        sa.q_row_stride = kHeads * kHeadDim;
                        sa.pooled_slot_stride = (uint64_t)2u * blocks * kHeadDim;
                        sa.pooled_layer_stride = (uint64_t)blocks * kHeadDim;
                        sa.scale = scale;
                        if (shelf::kernel::launch_qsa_score_wmma(sa, stream) != hipSuccess) return false;
                    } else {
                        shelf::kernel::QsaTopkArgs ta{};
                        ta.scores = scores;
                        ta.state_scores = state_scores;
                        ta.state_indices = state_indices;
                        ta.block_mask = mask;
                        ta.selected = selected;
                        ta.positions = positions;
                        ta.rows = rows;
                        ta.block_begin = bt;
                        ta.block_count = window;
                        ta.score_row_stride = kWindow;
                        ta.blocks = blocks;
                        ta.compress_ratio = o.compress;
                        ta.top_k = o.top_k;
                        ta.mask_row_bytes = mask_row_bytes_;
                        ta.selected_row_stride = o.top_k;
                        ta.first = bt == 0u ? 1u : 0u;
                        ta.last = bt + window >= blocks ? 1u : 0u;
                        ta.legacy_merge = o.legacy_merge;
                        if (shelf::kernel::launch_qsa_topk(ta, stream) != hipSuccess) return false;
                    }
                }
                return true;
            };
            if (!run_windows(true) || !run_windows(false) ||
                hipStreamSynchronize(stream) != hipSuccess) {
                rc = 1;
                break;
            }

            auto measure = [&](bool score_pass, std::vector<double>& per) {
                hipEvent_t ev0 = nullptr;
                hipEvent_t ev1 = nullptr;
                if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) {
                    return false;
                }
                bool ok_run = true;
                for (int s = 0; s < o.samples; ++s) {
                    if (s == 0) {
                        for (int w = 0; w < o.warmup; ++w) {
                            if (!run_windows(score_pass)) ok_run = false;
                        }
                    }
                    if (hipEventRecord(ev0, stream) != hipSuccess) { ok_run = false; break; }
                    for (int l = 0; l < o.launches; ++l) {
                        if (!run_windows(score_pass)) { ok_run = false; break; }
                    }
                    if (hipEventRecord(ev1, stream) != hipSuccess) { ok_run = false; break; }
                    if (hipEventSynchronize(ev1) != hipSuccess) { ok_run = false; break; }
                    float ms = 0.0f;
                    if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) { ok_run = false; break; }
                    per.push_back(ms * 1000.0 / (double)o.launches);
                }
                shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
                shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
                return ok_run;
            };

            std::vector<double> score_us;
            std::vector<double> topk_us;
            if (!measure(true, score_us) || !measure(false, topk_us)) {
                rc = 1;
                break;
            }
            if (score_us.empty() || topk_us.empty()) {
                rc = 1;
                break;
            }
            const double score_p50 = percentile(score_us, 0.5);
            const double score_mean = [&] {
                double sum = 0.0;
                for (double x : score_us) sum += x;
                return sum / (double)score_us.size();
            }();
            const double topk_p50 = percentile(topk_us, 0.5);
            const double topk_mean = [&] {
                double sum = 0.0;
                for (double x : topk_us) sum += x;
                return sum / (double)topk_us.size();
            }();
            if (!o.out.empty()) {
                FILE* f = std::fopen(o.out.c_str(), "a");
                if (f) {
                    if (std::ftell(f) == 0) {
                        std::fprintf(f,
                                     "rows,blocks,top_k,compress,window,score_p50_us,"
                                     "score_mean_us,topk_p50_us,topk_mean_us,"
                                     "score_mb,topk_mb\n");
                    }
                    const double mb = (double)rows * (double)blocks * 4.0 / (1024.0 * 1024.0);
                    std::fprintf(f, "%u,%u,%u,%u,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                                 rows, blocks, o.top_k, o.compress, kWindow, score_p50,
                                 score_mean, topk_p50, topk_mean, mb, mb);
                    std::fclose(f);
                }
            }
            std::printf(
                "qsa-select rows=%-4u blocks=%-6u top_k=%-4u score_p50=%8.2fus "
                "topk_p50=%8.2fus score_mb=%.1f\n",
                rows, blocks, o.top_k, score_p50, topk_p50,
                (double)rows * (double)blocks * 4.0 / (1024.0 * 1024.0));
        }
        release(q);
        release(pooled);
        release(slots);
        release(positions);
        release(scores);
        release(state_scores);
        release(state_indices);
        release(selected);
        release(mask);
        if (rc != 0) break;
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
