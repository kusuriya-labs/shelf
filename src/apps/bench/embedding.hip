#include <shelf/kernels/optimized/embedding.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/program/program.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using V = ::shelf::runtime::DeviceValueBinding;
using P = ::shelf::runtime::DeviceProgramView;
using B = ::shelf::runtime::DispatchBinding;
using W = ::shelf::runtime::WeightSlot;
using DT = ::shelf::runtime::ValueDType;

struct Options {
    std::string variant = "optimized";
    std::string storage = "bf16";
    std::string token_pattern = "random";
    std::vector<uint32_t> rows = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    uint32_t vocab = 248320;
    uint32_t hidden = 2560;
    int warmup = 100;
    int samples = 50;
    int launches = 20;
    uint32_t seed = 1;
    int device = 0;
    std::string out;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: shelf-bench embedding [options]\n"
                 "  --variant correctness|optimized\n"
                 "  --storage bf16\n"
                 "  --token-pattern random|repeated|sequential\n"
                 "  --rows 1,2,4,...\n"
                 "  --vocab N\n"
                 "  --hidden N\n"
                 "  --warmup N\n"
                 "  --samples N\n"
                 "  --launches N\n"
                 "  --seed N\n"
                 "  --device N\n"
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

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    const double idx = p * (double)(v.size() - 1);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min((size_t)std::ceil(idx), (size_t)v.size() - 1);
    const double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct Table {
    void* bf16 = nullptr;
    ~Table() {
        if (bf16) shelf::gpu::discard_cleanup_result(hipFree(bf16));
    }
    bool alloc_fill(uint32_t vocab, uint32_t hidden, uint32_t seed) {
        if (hipMalloc(&bf16, (size_t)vocab * hidden * 2) != hipSuccess) return false;
        std::vector<uint16_t> tb((size_t)vocab * hidden);
        for (size_t i = 0; i < tb.size(); ++i)
            tb[i] = (uint16_t)(h32((uint64_t)seed * 41u + i) % 65536u);
        if (hipMemcpy(bf16, tb.data(), tb.size() * 2, hipMemcpyHostToDevice) != hipSuccess)
            return false;
        return true;
    }
};

struct Env {
    uint32_t vocab = 0;
    uint32_t hidden = 0;
    uint32_t rows = 0;
    void* table_dev = nullptr;
    void* toks = nullptr;
    void* out = nullptr;
    void* bc = nullptr;
    void* wslot = nullptr;
    void* scratch = nullptr;
    void* err = nullptr;
    void* vals = nullptr;
    P prog{};
    B binding{};
    ~Env() {
        for (auto* q : {toks, out, bc, wslot, scratch, err, vals})
            if (q) shelf::gpu::discard_cleanup_result(hipFree(q));
    }
};

bool build_env(Env& e, const Table& t, const std::string& pattern, uint32_t rows,
               uint32_t seed) {
    e.rows = rows;
    e.table_dev = t.bf16;
    if (hipMalloc(&e.toks, (size_t)rows * 4) != hipSuccess) return false;
    if (hipMalloc(&e.out, (size_t)rows * e.hidden * 2) != hipSuccess) return false;
    if (hipMalloc(&e.bc, sizeof(::shelf::runtime::DeviceBatchContext)) != hipSuccess) return false;
    if (hipMalloc(&e.wslot, sizeof(W)) != hipSuccess) return false;
    if (hipMalloc(&e.scratch, (size_t)256 * 4096 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&e.err, sizeof(uint32_t)) != hipSuccess) return false;
    if (hipMalloc(&e.vals, sizeof(V)) != hipSuccess) return false;

    std::vector<int32_t> toks(rows);
    for (uint32_t m = 0; m < rows; ++m) {
        if (pattern == "repeated") toks[m] = (int32_t)(m % 8u);
        else if (pattern == "sequential") toks[m] = (int32_t)(m % e.vocab);
        else toks[m] = (int32_t)(h32((uint64_t)seed * 1000003u + (uint64_t)m * 2654435761u) % e.vocab);
    }
    if (hipMemcpy(e.toks, toks.data(), (size_t)rows * 4, hipMemcpyHostToDevice) != hipSuccess)
        return false;

    ::shelf::runtime::DeviceBatchContext hbc{};
    hbc.actual_rows = rows;
    hbc.num_requests = 0;
    hbc.token_ids = (const int32_t*)e.toks;
    if (hipMemcpy(e.bc, &hbc, sizeof(hbc), hipMemcpyHostToDevice) != hipSuccess) return false;

    W ws{};
    ws.rows = e.vocab;
    ws.cols = e.hidden;
    ws.encoding = 0;
    ws.device_ptr = e.table_dev;
    if (hipMemcpy(e.wslot, &ws, sizeof(W), hipMemcpyHostToDevice) != hipSuccess) return false;

    V v{};
    v.logical_value = 1;
    v.storage = ::shelf::runtime::ValueStorage::WORKSPACE;
    v.dtype = DT::BF16;
    v.row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
    v.slot = 0;
    v.feature_count = e.hidden;
    v.row_stride = e.hidden;
    v.offset = (uint64_t)(uintptr_t)e.out;
    v.bytes = 0;
    if (hipMemcpy(e.vals, &v, sizeof(V), hipMemcpyHostToDevice) != hipSuccess) return false;

    e.prog = {};
    e.prog.values = (const V*)e.vals;
    e.prog.value_count = 1;
    e.prog.workspace_base = 0;
    e.prog.weight_table = (const W*)e.wslot;
    e.prog.weight_slot_count = 1;
    e.binding = {};
    e.binding.kernel_id = ::shelf::runtime::KernelId::EMBEDDING_LOOKUP;
    e.binding.input_slots[0] = 0;
    e.binding.output_slots[0] = 1;
    e.binding.input_count = 1;
    e.binding.output_count = 1;
    e.binding.weight_index = 0;
    return true;
}

hipError_t launch_correctness(const Env& e, hipStream_t s) {
    shelf::gpu::discard_cleanup_result(hipMemsetAsync(e.err, 0, sizeof(uint32_t), s));
    ::shelf::kernel::ModelDispatchStateView sv{};
    auto st = shelf::kernel::launch_model_dispatch_correctness(
        e.prog, e.binding, (::shelf::runtime::DeviceBatchContext*)e.bc, sv, (float*)e.scratch,
        (uint32_t*)e.err, s);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

hipError_t launch_optimized(const Env& e, hipStream_t s) {
    return ::shelf::kernel::launch_embedding_bf16(
        (::shelf::bf16_t*)e.table_dev, (const int32_t*)e.toks, (::shelf::bf16_t*)e.out, e.rows,
        e.vocab, e.hidden, e.hidden, (uint32_t*)e.err, s);
}

}  // namespace

int run_embedding(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing arg for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else if (a == "--variant") o.variant = next();
        else if (a == "--storage") o.storage = next();
        else if (a == "--token-pattern") o.token_pattern = next();
        else if (a == "--rows") { if (!parse_list(next(), o.rows)) std::exit(2); }
        else if (a == "--vocab") o.vocab = (uint32_t)std::stoul(next());
        else if (a == "--hidden") o.hidden = (uint32_t)std::stoul(next());
        else if (a == "--warmup") o.warmup = std::atoi(next());
        else if (a == "--samples") o.samples = std::atoi(next());
        else if (a == "--launches") o.launches = std::atoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::atoi(next());
        else if (a == "--out") o.out = next();
        else { usage(stderr); return 2; }
    }
    if (o.variant != "correctness" && o.variant != "optimized") { usage(stderr); return 2; }
    if (o.storage != "bf16") { usage(stderr); return 2; }
    if (o.samples <= 0 || o.launches <= 0) return 2;
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    const uint32_t hidden = o.hidden;
    Table table;
    if (!table.alloc_fill(o.vocab, hidden, o.seed)) {
        std::printf("table alloc fail\n");
        return 1;
    }
    if (hipDeviceSynchronize() != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f,
                             "variant,storage,vocab,hidden,rows,token_pattern,warmup,samples,"
                             "launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (uint32_t rows : o.rows) {
        Env e;
        e.vocab = o.vocab;
        e.hidden = hidden;
        if (!build_env(e, table, o.token_pattern, rows, o.seed)) {
            std::printf("build fail rows=%u\n", rows);
            rc = 1;
            continue;
        }
        if (hipDeviceSynchronize() != hipSuccess) { rc = 1; continue; }
        auto launch = [&](hipStream_t s) -> hipError_t {
            if (o.variant == "correctness") return launch_correctness(e, s);
            return launch_optimized(e, s);
        };
        for (int i = 0; i < o.warmup; ++i)
            if (launch(stream) != hipSuccess) rc = 1;
        if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; continue; }
        hipEvent_t ev0 = nullptr, ev1 = nullptr;
        if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) {
            rc = 1;
            continue;
        }
        std::vector<double> per;
        per.reserve(o.samples);
        for (int s = 0; s < o.samples; ++s) {
            if (hipEventRecord(ev0, stream) != hipSuccess) break;
            for (int l = 0; l < o.launches; ++l)
                if (launch(stream) != hipSuccess) break;
            if (hipEventRecord(ev1, stream) != hipSuccess) break;
            if (hipEventSynchronize(ev1) != hipSuccess) break;
            float ms = 0.0f;
            if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) break;
            per.push_back(ms * 1000.0 / (double)o.launches);
        }
        shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
        shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
        if (per.empty()) { rc = 1; continue; }
        const double mn = percentile(per, 0.0);
        const double p50 = percentile(per, 0.5);
        const double p95 = percentile(per, 0.95);
        double mean = 0.0;
        for (double x : per) mean += x;
        mean /= (double)per.size();
        if (!o.out.empty()) {
            FILE* f = std::fopen(o.out.c_str(), "a");
            if (f) {
                std::fprintf(f, "%s,%s,%u,%u,%u,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                             o.variant.c_str(), o.storage.c_str(), o.vocab, hidden, rows,
                             o.token_pattern.c_str(), o.warmup, o.samples, o.launches, mn, p50,
                             p95, mean);
                std::fclose(f);
            }
        }
        std::printf("%-11s %-8s v=%-6u h=%-5u rows=%-5u pat=%-10s p50=%.2fus mean=%.2fus\n",
                    o.variant.c_str(), o.storage.c_str(), o.vocab, hidden, rows,
                    o.token_pattern.c_str(), p50, mean);
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
