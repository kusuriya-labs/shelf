#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/fused/qsa_key_cache.h>

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
    std::vector<uint32_t> rows = {16u, 32u, 64u, 128u, 256u, 512u};
    uint32_t requests = 1u;
    uint32_t compress = 4u;
    std::string variant = "both";
    int warmup = 20;
    int samples = 20;
    int launches = 10;
    int device = 0;
    std::string out;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: shelf-bench qsa-key-cache [options]\n"
                 "  --rows 16,32,...         total actual rows\n"
                 "  --requests N\n"
                 "  --compress N             (1..4)\n"
                 "  --variant batch|blocks|both\n"
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

int run_qsa_key_cache(int argc, char** argv) {
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
        } else if (!std::strcmp(a, "--requests")) {
            o.requests = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--compress")) {
            o.compress = (uint32_t)std::stoul(v);
        } else if (!std::strcmp(a, "--variant")) {
            o.variant = v;
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
    if (o.rows.empty() || o.requests == 0u || o.compress == 0u || o.compress > 4u ||
        (o.variant != "batch" && o.variant != "blocks" && o.variant != "both")) {
        return 2;
    }
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    constexpr uint32_t kHeadDim = shelf::kernel::kQsaHeadDim;
    const uint32_t block_units =
        (o.rows.back() + o.compress - 1u) / o.compress + 1u;
    const uint32_t max_blocks = block_units + 1u;

    shelf::bf16_t* raw = nullptr;
    shelf::bf16_t* norm = nullptr;
    uint32_t* positions = nullptr;
    shelf::runtime::DeviceRequestDescriptor* requests = nullptr;
    shelf::bf16_t* pooled = nullptr;
    shelf::bf16_t* partial = nullptr;
    uint32_t* pool_counts = nullptr;
    uint32_t* partial_counts = nullptr;
    int rc = 0;
    bool ok = hipMalloc(&raw, (size_t)o.rows.back() * kHeadDim * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&norm, (size_t)kHeadDim * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&positions, (size_t)o.rows.back() * sizeof(uint32_t)) == hipSuccess;
    ok = ok && hipHostMalloc(reinterpret_cast<void**>(&requests),
                             (size_t)o.requests *
                                 sizeof(shelf::runtime::DeviceRequestDescriptor)) == hipSuccess;
    ok = ok && hipMalloc(&pooled, (size_t)o.requests * max_blocks * kHeadDim *
                                      sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&partial, (size_t)o.requests * o.compress * kHeadDim *
                                       sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&pool_counts, (size_t)o.requests * max_blocks *
                                           sizeof(uint32_t)) == hipSuccess;
    ok = ok && hipMalloc(&partial_counts, (size_t)o.requests * sizeof(uint32_t)) == hipSuccess;
    if (!ok) {
        if (rc == 0) rc = 1;
    }
    {
        uint64_t seed = 0x9E3779B97F4A7C15ull;
        std::vector<shelf::bf16_t> raw_host((size_t)o.rows.back() * kHeadDim);
        for (auto& value : raw_host) {
            seed = h64(seed);
            value.data = shelf::f32_to_bf16_rne(((int32_t)(seed % 2001u) - 1000) / 1000.0f);
        }
        std::vector<shelf::bf16_t> norm_host(kHeadDim);
        for (auto& value : norm_host) {
            seed = h64(seed);
            value.data = shelf::f32_to_bf16_rne(((int32_t)(seed % 201u) - 100) / 1000.0f);
        }
        std::vector<uint32_t> pos_host(o.rows.back());
        for (uint32_t i = 0; i < pos_host.size(); ++i) pos_host[i] = i;
        if (ok) {
            ok = hipMemcpy(raw, raw_host.data(),
                           raw_host.size() * sizeof(shelf::bf16_t),
                           hipMemcpyHostToDevice) == hipSuccess;
            ok = ok && hipMemcpy(norm, norm_host.data(),
                                 norm_host.size() * sizeof(shelf::bf16_t),
                                 hipMemcpyHostToDevice) == hipSuccess;
            ok = ok && hipMemcpy(positions, pos_host.data(),
                                 pos_host.size() * sizeof(uint32_t),
                                 hipMemcpyHostToDevice) == hipSuccess;
        }
        for (uint32_t r = 0; r < o.requests; ++r) {
            shelf::runtime::DeviceRequestDescriptor desc{};
            desc.request_handle.slot = r;
            desc.row_begin = (uint32_t)((uint64_t)o.rows.back() * r / o.requests);
            desc.row_count = (uint32_t)((uint64_t)o.rows.back() * (r + 1u) / o.requests) -
                             desc.row_begin;
            desc.sequence_length = desc.row_count;
            requests[r] = desc;
        }
        if (!ok) rc = 1;
    }
    if (rc != 0) {
        shelf::gpu::discard_cleanup_result(hipFree(raw));
        shelf::gpu::discard_cleanup_result(hipFree(norm));
        shelf::gpu::discard_cleanup_result(hipFree(positions));
        shelf::gpu::discard_cleanup_result(hipHostFree(requests));
        shelf::gpu::discard_cleanup_result(hipFree(pooled));
        shelf::gpu::discard_cleanup_result(hipFree(partial));
        shelf::gpu::discard_cleanup_result(hipFree(pool_counts));
        shelf::gpu::discard_cleanup_result(hipFree(partial_counts));
        shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return rc;
    }

    for (uint32_t rows : o.rows) {
        uint32_t offset = 0u;
        for (uint32_t r = 0; r < o.requests; ++r) {
            shelf::runtime::DeviceRequestDescriptor desc = requests[r];
            const uint32_t total = desc.row_begin + desc.row_count;
            const uint32_t begin = (uint32_t)((uint64_t)rows * r / o.requests);
            desc.row_begin = begin;
            desc.row_count = (uint32_t)((uint64_t)rows * (r + 1u) / o.requests) - begin;
            desc.sequence_length = desc.row_count;
            requests[r] = desc;
            offset = total;
        }
        (void)offset;
        shelf::kernel::QsaKeyCacheBatchArgs args{};
        args.raw_keys = raw;
        args.norm_weight = norm;
        args.positions = positions;
        args.requests = requests;
        args.pooled_base = pooled;
        args.partial_base = partial;
        args.pool_count_base = pool_counts;
        args.partial_count_base = partial_counts;
        args.num_requests = o.requests;
        args.actual_rows = rows;
        args.max_sequences = o.requests;
        args.state_index = 0u;
        args.max_blocks = max_blocks;
        args.head_dim = kHeadDim;
        args.compress_ratio = o.compress;
        args.rotary_dim = kHeadDim;
        args.raw_row_stride = kHeadDim;
        args.pooled_slot_stride = (uint64_t)max_blocks * kHeadDim;
        args.pooled_layer_stride = (uint64_t)max_blocks * kHeadDim;
        args.partial_slot_stride = (uint64_t)o.compress * kHeadDim;
        args.partial_layer_stride = (uint64_t)o.compress * kHeadDim;
        args.count_slot_stride = max_blocks;
        args.count_layer_stride = max_blocks;
        args.theta = 10000000.0f;
        args.position_scale = 1.0f;
        args.eps = 1e-6f;

        auto run_batch = [&]() {
            return shelf::kernel::launch_qsa_key_cache_batch(args, stream) == hipSuccess;
        };
        auto run_blocks = [&]() {
            return shelf::kernel::launch_qsa_key_cache_blocks(args, stream) == hipSuccess;
        };
        auto measure = [&](bool blocks, std::vector<double>& per) {
            hipEvent_t ev0 = nullptr;
            hipEvent_t ev1 = nullptr;
            if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) {
                return false;
            }
            bool ok_run = true;
            for (int s = 0; s < o.samples && ok_run; ++s) {
                if (s == 0) {
                    for (int w = 0; w < o.warmup; ++w) {
                        if (!(blocks ? run_blocks() : run_batch())) { ok_run = false; break; }
                    }
                }
                if (!ok_run || hipEventRecord(ev0, stream) != hipSuccess) { ok_run = false; break; }
                for (int l = 0; l < o.launches; ++l) {
                    if (!(blocks ? run_blocks() : run_batch())) { ok_run = false; break; }
                }
                if (!ok_run || hipEventRecord(ev1, stream) != hipSuccess ||
                    hipEventSynchronize(ev1) != hipSuccess) {
                    ok_run = false;
                    break;
                }
                float ms = 0.0f;
                if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) { ok_run = false; break; }
                per.push_back(ms * 1000.0 / (double)o.launches);
            }
            shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
            shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            return ok_run;
        };
        if (hipDeviceSynchronize() != hipSuccess) { rc = 1; break; }
        std::vector<double> per_batch;
        std::vector<double> per_blocks;
        if (o.variant != "blocks" && !measure(false, per_batch)) { rc = 1; break; }
        if (o.variant != "batch" && !measure(true, per_blocks)) { rc = 1; break; }
        const double batch_p50 = percentile(per_batch, 0.5);
        const double blocks_p50 = percentile(per_blocks, 0.5);
        if (!o.out.empty()) {
            FILE* f = std::fopen(o.out.c_str(), "a");
            if (f) {
                if (std::ftell(f) == 0) {
                    std::fprintf(f, "rows,requests,compress,batch_p50_us,blocks_p50_us\n");
                }
                std::fprintf(f, "%u,%u,%u,%.3f,%.3f\n", rows, o.requests, o.compress,
                             batch_p50, blocks_p50);
                std::fclose(f);
            }
        }
        if (o.variant == "batch") {
            std::printf("qsa-key-cache rows=%-5u req=%-3u cr=%u batch_p50=%8.2fus\n", rows,
                        o.requests, o.compress, batch_p50);
        } else if (o.variant == "blocks") {
            std::printf("qsa-key-cache rows=%-5u req=%-3u cr=%u blocks_p50=%8.2fus\n", rows,
                        o.requests, o.compress, blocks_p50);
        } else {
            std::printf("qsa-key-cache rows=%-5u req=%-3u cr=%u batch_p50=%8.2fus "
                        "blocks_p50=%8.2fus\n",
                        rows, o.requests, o.compress, batch_p50, blocks_p50);
        }
    }
    shelf::gpu::discard_cleanup_result(hipFree(raw));
    shelf::gpu::discard_cleanup_result(hipFree(norm));
    shelf::gpu::discard_cleanup_result(hipFree(positions));
    shelf::gpu::discard_cleanup_result(hipHostFree(requests));
    shelf::gpu::discard_cleanup_result(hipFree(pooled));
    shelf::gpu::discard_cleanup_result(hipFree(partial));
    shelf::gpu::discard_cleanup_result(hipFree(pool_counts));
    shelf::gpu::discard_cleanup_result(hipFree(partial_counts));
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
