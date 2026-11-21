#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/attention/qsa_sparse.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Options {
    std::vector<uint32_t> rows = {1u, 4u, 8u, 64u, 128u, 512u, 2048u};
    std::vector<uint32_t> visible = {2048u, 8192u, 32768u};
    uint32_t selected = 512u;
    std::string kv_dtype = "bf16";
    uint32_t page_tokens = 16u;
    uint32_t splits = 0u;
    uint32_t q_heads = 24u;
    uint32_t kv_heads = 2u;
    uint32_t seed = 20261121u;
    int warmup = 20;
    int samples = 20;
    int launches = 10;
    int device = 0;
    bool scatter = true;
    std::string out;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: shelf-bench qsa-attention [options]\n"
                 "  --rows 1,4,...\n"
                 "  --visible 2048,8192,...      visible tokens per row\n"
                 "  --selected N                 selected blocks per row\n"
                 "  --kv-dtype bf16|fp8\n"
                 "  --page-tokens N\n"
                 "  --splits N                   (0 = auto)\n"
                 "  --q-heads N --kv-heads N     (q_heads/kv_heads must be 6 or 12)\n"
                 "  --contiguous                 mark selected blocks contiguously\n"
                 "  --warmup N --samples N --launches N --device N\n"
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

}  // namespace

int run_qsa_attention(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        }
        if (!std::strcmp(a, "--contiguous")) {
            o.scatter = false;
            continue;
        }
        if (i + 1 >= argc) return 2;
        const char* v = argv[++i];
        if (!std::strcmp(a, "--rows")) {
            if (!parse_list(v, o.rows)) return 2;
        } else if (!std::strcmp(a, "--visible")) {
            if (!parse_list(v, o.visible)) return 2;
        } else if (!std::strcmp(a, "--selected")) {
            o.selected = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--kv-dtype")) {
            o.kv_dtype = v;
        } else if (!std::strcmp(a, "--page-tokens")) {
            o.page_tokens = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--splits")) {
            o.splits = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--q-heads")) {
            o.q_heads = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--kv-heads")) {
            o.kv_heads = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--warmup")) {
            o.warmup = std::stoi(v);
        } else if (!std::strcmp(a, "--samples")) {
            o.samples = std::stoi(v);
        } else if (!std::strcmp(a, "--launches")) {
            o.launches = std::stoi(v);
        } else if (!std::strcmp(a, "--device")) {
            o.device = std::stoi(v);
        } else if (!std::strcmp(a, "--out")) {
            o.out = v;
        } else {
            return 2;
        }
    }
    const bool fp8 = o.kv_dtype == "fp8";
    if (!fp8 && o.kv_dtype != "bf16") return 2;
    if (o.rows.empty() || o.visible.empty() || o.selected == 0u ||
        o.page_tokens == 0u || (o.page_tokens % 4u) != 0u || o.warmup < 0) {
        return 2;
    }
    const uint32_t q_per_kv = o.q_heads / o.kv_heads;
    if (o.kv_heads == 0u || (q_per_kv != 6u && q_per_kv != 12u)) return 2;
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    constexpr uint32_t kHeadDim = shelf::kernel::kQsaSparseHeadDim;
    constexpr uint32_t kBlockTokens = shelf::kernel::kQsaSparseBlockTokens;
    const uint32_t max_visible = *std::max_element(o.visible.begin(), o.visible.end());
    const uint32_t blocks_capacity = (max_visible + kBlockTokens - 1u) / kBlockTokens;
    const uint32_t mask_blocks = ((blocks_capacity + 31u) / 32u) * 32u;
    const uint32_t mask_stride = mask_blocks / 8u;
    const uint32_t max_pages = (max_visible + o.page_tokens - 1u) / o.page_tokens;
    const uint32_t elems_per_token = o.kv_heads * kHeadDim;
    const uint32_t elems_per_page = o.page_tokens * elems_per_token;
    const uint32_t elems_per_layer = max_pages * elems_per_page;
    const uint32_t block_table_stride = max_pages;
    const uint64_t pool_elems = (uint64_t)elems_per_layer;
    const uint64_t pool_bytes = fp8 ? pool_elems : pool_elems * sizeof(shelf::bf16_t);

    int rc = 0;
    for (uint32_t visible : o.visible) {
        const uint32_t visible_blocks = (visible + kBlockTokens - 1u) / kBlockTokens;
        const uint32_t selected = std::min(o.selected, visible_blocks);
        shelf::bf16_t* q_pool = nullptr;
        uint8_t* k_pool = nullptr;
        uint8_t* v_pool = nullptr;
        float* k_scales = nullptr;
        float* v_scales = nullptr;
        uint8_t* mask = nullptr;
        uint32_t* positions = nullptr;
        uint32_t* slots = nullptr;
        uint32_t* tables = nullptr;
        float* out_pool = nullptr;
        float* partials = nullptr;
        for (uint32_t rows : o.rows) {
            const uint32_t max_sequences = 1u;
            bool ok = hipMalloc(&q_pool, (size_t)rows * o.q_heads * kHeadDim *
                                            sizeof(shelf::bf16_t)) == hipSuccess;
            ok = ok && hipMalloc(&k_pool, pool_bytes) == hipSuccess;
            ok = ok && hipMalloc(&v_pool, pool_bytes) == hipSuccess;
            if (fp8) {
                ok = ok && hipMalloc(&k_scales,
                                     (size_t)elems_per_layer * sizeof(float)) == hipSuccess;
                ok = ok && hipMalloc(&v_scales,
                                     (size_t)elems_per_layer * sizeof(float)) == hipSuccess;
            }
            ok = ok && hipMalloc(&mask, (size_t)rows * mask_stride) == hipSuccess;
            ok = ok && hipMalloc(&positions, (size_t)rows * sizeof(uint32_t)) == hipSuccess;
            ok = ok && hipMalloc(&slots, (size_t)rows * sizeof(uint32_t)) == hipSuccess;
            ok = ok && hipMalloc(&tables, (size_t)max_sequences * block_table_stride *
                                              sizeof(uint32_t)) == hipSuccess;
            ok = ok && hipMalloc(&out_pool, (size_t)rows * o.q_heads * kHeadDim *
                                                sizeof(float)) == hipSuccess;
            if (!ok) {
                rc = 1;
                break;
            }
            uint32_t splits = o.splits;
            uint64_t partial_bytes = 0u;
            if (splits > 1u) {
                partial_bytes = shelf::kernel::qsa_sparse_partial_bytes(
                    rows, o.kv_heads, splits, q_per_kv);
            } else {
                splits = shelf::kernel::qsa_sparse_splits(
                    rows, o.kv_heads, 32ull << 20, q_per_kv);
                partial_bytes = shelf::kernel::qsa_sparse_partial_bytes(
                    rows, o.kv_heads, splits, q_per_kv);
            }
            partial_bytes = std::max<uint64_t>(partial_bytes, 1024u);
            ok = hipMalloc(&partials, partial_bytes) == hipSuccess;
            if (!ok) {
                rc = 1;
                break;
            }
            {
                std::vector<uint8_t> pool_data(pool_bytes, 0u);
                uint64_t seed = 0x9E3779B97F4A7C15ull ^ h64((uint64_t)visible);
                if (fp8) {
                    for (auto& byte : pool_data) {
                        seed = h64(seed);
                        byte = (uint8_t)(seed & 0x7Fu);
                    }
                } else {
                    auto* values = reinterpret_cast<shelf::bf16_t*>(pool_data.data());
                    for (uint64_t i = 0; i < pool_elems; ++i) {
                        seed = h64(seed);
                        values[i].data = shelf::f32_to_bf16_rne(
                            ((int32_t)(seed % 2001u) - 1000) / 1000.0f);
                    }
                }
                std::vector<shelf::bf16_t> q_data((size_t)rows * o.q_heads * kHeadDim);
                for (auto& value : q_data) {
                    seed = h64(seed);
                    value.data = shelf::f32_to_bf16_rne(
                        ((int32_t)(seed % 2001u) - 1000) / 1000.0f);
                }
                std::vector<uint8_t> mask_data((size_t)rows * mask_stride, 0u);
                for (uint32_t row = 0; row < rows; ++row) {
                    const uint32_t stride = o.scatter
                        ? (selected == 0u ? 1u : std::max(1u, visible_blocks / selected))
                        : 1u;
                    uint32_t marked = 0u;
                    for (uint32_t b = 0; b < visible_blocks && marked < selected; b += stride) {
                        mask_data[(size_t)row * mask_stride + (b >> 3u)] |=
                            (uint8_t)(1u << (b & 7u));
                        ++marked;
                    }
                }
                std::vector<uint32_t> pos_data(rows, visible - 1u);
                std::vector<uint32_t> slot_data(rows, 0u);
                std::vector<uint32_t> table_data(block_table_stride);
                for (uint32_t p = 0; p < block_table_stride; ++p) table_data[p] = p;
                if (fp8) {
                    std::vector<float> scale_data(elems_per_layer, 0.01f);
                    ok = hipMemcpy(k_scales, scale_data.data(),
                                   scale_data.size() * sizeof(float),
                                   hipMemcpyHostToDevice) == hipSuccess;
                    ok = ok && hipMemcpy(v_scales, scale_data.data(),
                                         scale_data.size() * sizeof(float),
                                         hipMemcpyHostToDevice) == hipSuccess;
                }
                ok = ok && hipMemcpy(q_pool, q_data.data(),
                                     q_data.size() * sizeof(shelf::bf16_t),
                                     hipMemcpyHostToDevice) == hipSuccess;
                ok = ok && hipMemcpy(k_pool, pool_data.data(), pool_bytes,
                                     hipMemcpyHostToDevice) == hipSuccess;
                ok = ok && hipMemcpy(v_pool, pool_data.data(), pool_bytes,
                                     hipMemcpyHostToDevice) == hipSuccess;
                ok = ok && hipMemcpy(mask, mask_data.data(), mask_data.size(),
                                     hipMemcpyHostToDevice) == hipSuccess;
                ok = ok && hipMemcpy(positions, pos_data.data(),
                                     pos_data.size() * sizeof(uint32_t),
                                     hipMemcpyHostToDevice) == hipSuccess;
                ok = ok && hipMemcpy(slots, slot_data.data(),
                                     slot_data.size() * sizeof(uint32_t),
                                     hipMemcpyHostToDevice) == hipSuccess;
                ok = ok && hipMemcpy(tables, table_data.data(),
                                     table_data.size() * sizeof(uint32_t),
                                     hipMemcpyHostToDevice) == hipSuccess;
                if (!ok) {
                    rc = 1;
                    break;
                }
            }

            shelf::kernel::PagedAttentionCommonArgs args{};
            args.q = q_pool;
            args.output = out_pool;
            args.output_dtype = shelf::kernel::PagedAttentionOutputDType::F32;
            args.q_row_stride = o.q_heads * kHeadDim;
            args.output_row_stride = o.q_heads * kHeadDim;
            args.row_sequence_slots = slots;
            args.row_positions = positions;
            args.block_tables = tables;
            args.block_table_stride = block_table_stride;
            args.max_sequences = max_sequences;
            args.rows = rows;
            args.layer = 0u;
            args.page_tokens = o.page_tokens;
            args.num_pages = max_pages;
            args.num_attention_layers = 1u;
            args.q_heads = o.q_heads;
            args.kv_heads = o.kv_heads;
            args.head_dim = kHeadDim;
            args.elems_per_token = elems_per_token;
            args.elems_per_page = elems_per_page;
            args.elems_per_layer = elems_per_layer;
            args.scale = 1.0f / std::sqrt((float)kHeadDim);
            args.block_mask = mask;
            args.mask_stride = mask_stride;
            args.mask_blocks = mask_blocks;
            args.mask_block_shift = 2u;

            auto run = [&]() -> bool {
                hipError_t error = fp8
                    ? shelf::kernel::launch_qsa_sparse_attention_fp8(
                          args, k_pool, v_pool, k_scales, v_scales, partials, splits,
                          stream, q_per_kv)
                    : shelf::kernel::launch_qsa_sparse_attention(
                          args, reinterpret_cast<const shelf::bf16_t*>(k_pool),
                          reinterpret_cast<const shelf::bf16_t*>(v_pool), partials,
                          splits, stream, q_per_kv);
                if (error != hipSuccess) return false;
                if (splits > 1u) {
                    if (shelf::kernel::launch_qsa_sparse_reduce(args, partials, splits,
                                                                stream, q_per_kv) !=
                        hipSuccess) {
                        return false;
                    }
                }
                return true;
            };
            if (!run() || hipStreamSynchronize(stream) != hipSuccess) {
                rc = 1;
                break;
            }
            hipEvent_t ev0 = nullptr;
            hipEvent_t ev1 = nullptr;
            bool measured = hipEventCreate(&ev0) == hipSuccess &&
                            hipEventCreate(&ev1) == hipSuccess;
            std::vector<double> per;
            if (measured) {
                for (int s = 0; s < o.samples && measured; ++s) {
                    if (s == 0) {
                        for (int w = 0; w < o.warmup; ++w) {
                            if (!run()) { measured = false; break; }
                        }
                    }
                    if (hipEventRecord(ev0, stream) != hipSuccess) { measured = false; break; }
                    for (int l = 0; l < o.launches; ++l) {
                        if (!run()) { measured = false; break; }
                    }
                    if (hipEventRecord(ev1, stream) != hipSuccess ||
                        hipEventSynchronize(ev1) != hipSuccess) {
                        measured = false;
                        break;
                    }
                    float ms = 0.0f;
                    if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) {
                        measured = false;
                        break;
                    }
                    per.push_back(ms * 1000.0 / (double)o.launches);
                }
                shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
                shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            }
            if (!measured || per.empty()) {
                rc = 1;
                break;
            }
            const double p50 = percentile(per, 0.5);
            double mean = 0.0;
            for (double x : per) mean += x;
            mean /= (double)per.size();
            if (!o.out.empty()) {
                FILE* f = std::fopen(o.out.c_str(), "a");
                if (f) {
                    if (std::ftell(f) == 0) {
                        std::fprintf(f, "kv_dtype,rows,visible,selected,splits,p50_us,mean_us\n");
                    }
                    std::fprintf(f, "%s,%u,%u,%u,%u,%.3f,%.3f\n", o.kv_dtype.c_str(), rows,
                                 visible, selected, splits, p50, mean);
                    std::fclose(f);
                }
            }
            std::printf("qsa-attention %-4s rows=%-5u visible=%-7u selected=%-4u splits=%-3u "
                        "p50=%8.2fus mean=%8.2fus\n",
                        o.kv_dtype.c_str(), rows, visible, selected, splits, p50, mean);
            shelf::gpu::discard_cleanup_result(hipFree(q_pool));
            shelf::gpu::discard_cleanup_result(hipFree(k_pool));
            shelf::gpu::discard_cleanup_result(hipFree(v_pool));
            if (fp8) {
                shelf::gpu::discard_cleanup_result(hipFree(k_scales));
                shelf::gpu::discard_cleanup_result(hipFree(v_scales));
            }
            shelf::gpu::discard_cleanup_result(hipFree(mask));
            shelf::gpu::discard_cleanup_result(hipFree(positions));
            shelf::gpu::discard_cleanup_result(hipFree(slots));
            shelf::gpu::discard_cleanup_result(hipFree(tables));
            shelf::gpu::discard_cleanup_result(hipFree(out_pool));
            shelf::gpu::discard_cleanup_result(hipFree(partials));
            q_pool = nullptr;
            k_pool = nullptr;
            v_pool = nullptr;
            k_scales = nullptr;
            v_scales = nullptr;
            mask = nullptr;
            positions = nullptr;
            slots = nullptr;
            tables = nullptr;
            out_pool = nullptr;
            partials = nullptr;
        }
        (void)q_pool;
        if (rc != 0) break;
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
