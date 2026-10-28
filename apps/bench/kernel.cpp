// shelf-bench kernel <name> — 1 つのカーネルの微計測。
//
// 計測規約は phaseshift-bench と同じ: warmup 回を破棄し、samples 回、
// それぞれ launches 回の発射を 1 イベント対で測って launches で割る。
// 主指標は µs/launch の p50。ストリームは bench 自前のものを作る
// （launch_* は stream を引数で受け取るため、実行器を経由しない）。
// 合成バッファはすべて 0 埋め（値はタイミングに影響せず、NaN も生まない）。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "bench_common.hpp"
#include <shelf/foundation/numeric.hpp>

#ifdef SHELF_HAVE_HIP
#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/attention/kv_append.h>
#include <shelf/backend/hip/kernels/attention/paged_attention.h>
#include <shelf/backend/hip/kernels/elementwise/elementwise.h>
#include <shelf/backend/hip/kernels/elementwise/rmsnorm.h>
#include <shelf/backend/hip/kernels/gemm/linear_bf16.h>
#include <shelf/backend/hip/kernels/gemm/linear_bf16_config.h>
#endif

namespace shelf::bench {
namespace {

struct KernelEntry {
    const char* name;
    const char* blurb;
};

const KernelEntry kKernels[] = {
    {"gemm", "linear (GEMV/GEMM): --rows --n --k"},
    {"rmsnorm", "rms normalization: --rows --features"},
    {"elementwise", "residual_add (bf16): --rows --features"},
    {"kv-append", "paged KV append: --rows --kv-heads --head-dim --page-tokens --num-pages --seq-len"},
    {"paged-attention", "paged GQA decode attention: --rows --q-heads --kv-heads --head-dim --page-tokens --num-pages --seq-len"},
};

void usage(std::FILE* out) {
    std::fprintf(out,
        "usage: shelf-bench kernel <name> [options]\n"
        "\n"
        "names:\n");
    for (const auto& e : kKernels)
        std::fprintf(out, "  %-16s %s\n", e.name, e.blurb);
    std::fprintf(out,
        "\n"
        "common options:\n"
        "  --warmup N       破棄する発射回数（gemm/rmsnorm/elementwise は既定 100、\n"
        "                   kv-append / paged-attention は既定 30）\n"
        "  --samples N      サンプル数（既定 50）\n"
        "  --launches N     1 サンプル内の発射回数（既定 20）\n"
        "  --gpu N          GPU 番号（既定 0）\n"
        "  --output PATH    CSV を追記（ファイルが無ければヘッダを書く）\n"
        "  --help, -h       このヘルプ（GPU を触らず exit 0）\n"
        "\n"
        "notes:\n"
        "  すべて合成バッファ（0 埋め）。モデルは不要。\n"
        "  gemm の --rows は [1,64]、--k は 8 の倍数（カーネル契約）。\n"
        "  exit codes: 0 = ok, 1 = runtime error, 2 = usage error\n");
}

#ifdef SHELF_HAVE_HIP

// ---- 共通の報告 -------------------------------------------------------------

struct KernelResult {
    std::string name;
    std::string shape;        // CSV 用（カンマ禁止）。例: 1x2048x2048
    Stats us;                 // µs / launch
    double throughput = 0.0;
    std::string unit;         // GFLOP/s | GB/s | 空
    std::vector<std::pair<std::string, std::string>> extra;  // stdout 用の追加 key=value
};

void report(const KernelResult& r, std::int64_t gpu, std::int64_t warmup,
            std::int64_t samples, std::int64_t launches, const std::string& output) {
    std::printf("shelf-bench kernel %s\n", r.name.c_str());
    kv_print("schema_version", static_cast<std::int64_t>(3));
    kv_print("kernel", r.name);
    kv_print("device", std::string("hip"));
    kv_print("gpu", gpu);
    kv_print("shape", r.shape);
    for (const auto& kv : r.extra) kv_print(kv.first.c_str(), kv.second);
    kv_print("warmup", warmup);
    kv_print("samples", samples);
    kv_print("launches", launches);
    kv_print("us_min", r.us.min);
    kv_print("us_p10", r.us.p10);
    kv_print("us_p50", r.us.p50);
    kv_print("us_p90", r.us.p90);
    kv_print("us_p95", r.us.p95);
    kv_print("us_mean", r.us.mean);
    kv_print("us_stddev", r.us.stddev);
    kv_print("throughput", r.throughput, 2);
    kv_print("throughput_unit", r.unit);

    if (output.empty()) return;
    const char* kHeader =
        "schema_version,device,gpu,kernel,shape,warmup,samples,launches,"
        "us_min,us_p10,us_p50,us_p90,us_p95,us_mean,us_stddev,throughput,throughput_unit";
    char row[512];
    std::snprintf(row, sizeof row,
                  "3,hip,%lld,%s,%s,%lld,%lld,%lld,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%s",
                  static_cast<long long>(gpu), r.name.c_str(), r.shape.c_str(),
                  static_cast<long long>(warmup), static_cast<long long>(samples),
                  static_cast<long long>(launches), r.us.min, r.us.p10, r.us.p50, r.us.p90,
                  r.us.p95, r.us.mean, r.us.stddev, r.throughput, r.unit.c_str());
    std::string err;
    if (!append_csv(output, kHeader, {row}, &err))
        std::fprintf(stderr, "kernel: %s\n", err.c_str());
}

// ---- 共通の下請け -----------------------------------------------------------

struct DevBuf {
    void* p = nullptr;
    std::size_t bytes = 0;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() {
        if (p) (void)hipFree(p);
    }
    bool alloc(std::size_t n, const char* what, std::string* error) {
        const hipError_t e = hipMalloc(&p, n);
        if (e != hipSuccess) {
            p = nullptr;
            if (error) *error = std::string(what) + ": " + hipGetErrorString(e);
            return false;
        }
        bytes = n;
        const hipError_t z = hipMemset(p, 0, bytes);  // 0 埋め（NaN を生まない）
        if (z != hipSuccess) {
            if (error) *error = std::string(what) + " memset: " + hipGetErrorString(z);
            return false;
        }
        return true;
    }
};

bool set_device(std::int64_t gpu, std::string* error) {
    const hipError_t e = hipSetDevice(static_cast<int>(gpu));
    if (e != hipSuccess) {
        if (error) *error = std::string("hipSetDevice: ") + hipGetErrorString(e);
        return false;
    }
    return true;
}

// 多くのカーネルに共通するオプション（warmup / samples / launches）。
bool common_iters(const Args& a, std::int64_t warmup_default, std::int64_t* warmup,
                  std::int64_t* samples, std::int64_t* launches, int argc) {
    bool ok = true;
    *warmup = a.i64("warmup", warmup_default, &ok);
    *samples = a.i64("samples", 50, &ok);
    *launches = a.i64("launches", 20, &ok);
    if (!ok) {
        std::fprintf(stderr, "kernel: numeric option has a bad value\n");
        return false;
    }
    if (*warmup < 0 || *samples <= 0 || *launches <= 0) {
        std::fprintf(stderr, "kernel: require warmup >= 0, samples >= 1, launches >= 1\n");
        return false;
    }
    (void)argc;
    return true;
}

// KV 系の共通 fixture: プール + ブロック表 + 行位置（合成・0 埋め）。
struct KvSetup {
    std::int64_t rows = 1;
    std::int64_t q_heads = 32;
    std::int64_t kv_heads = 8;
    std::int64_t head_dim = 256;
    std::int64_t page_tokens = 16;
    std::int64_t num_pages = 4;
    std::int64_t seq_len = 16;
};

struct KvBufs {
    DevBuf k_pool, v_pool, table, slots, pos;
    std::uint32_t stride = 1;
};

bool setup_kv(const KvSetup& s, KvBufs* b, std::string* error) {
    if (s.rows <= 0 || s.kv_heads <= 0 || s.head_dim <= 0 || s.page_tokens <= 0 ||
        s.num_pages <= 0 || s.seq_len <= 0) {
        *error = "bad KV option (all must be >= 1)";
        return false;
    }
    if (s.seq_len > s.page_tokens * s.num_pages) {
        *error = "--seq-len exceeds page-tokens * num-pages";
        return false;
    }
    const std::size_t elems_per_token =
        static_cast<std::size_t>(s.kv_heads) * static_cast<std::size_t>(s.head_dim);
    const std::size_t pool_elems =
        static_cast<std::size_t>(s.num_pages) * static_cast<std::size_t>(s.page_tokens) *
        elems_per_token;
    const std::size_t stride =
        (static_cast<std::size_t>(s.seq_len) + static_cast<std::size_t>(s.page_tokens) - 1) /
        static_cast<std::size_t>(s.page_tokens);
    b->stride = static_cast<std::uint32_t>(stride);

    if (!b->k_pool.alloc(pool_elems * sizeof(shelf::bf16_t), "k_pool", error)) return false;
    if (!b->v_pool.alloc(pool_elems * sizeof(shelf::bf16_t), "v_pool", error)) return false;
    if (!b->slots.alloc(static_cast<std::size_t>(s.rows) * sizeof(std::uint32_t), "slots", error))
        return false;
    if (!b->pos.alloc(static_cast<std::size_t>(s.rows) * sizeof(std::uint32_t), "pos", error))
        return false;
    if (!b->table.alloc(stride * sizeof(std::uint32_t), "table", error)) return false;

    // ブロック表は恒等（0,1,2,...）、行はすべてシーケンス 0・位置 seq_len-1。
    std::vector<std::uint32_t> host_table(stride);
    for (std::size_t i = 0; i < stride; ++i) host_table[i] = static_cast<std::uint32_t>(i);
    std::vector<std::uint32_t> host_slots(static_cast<std::size_t>(s.rows), 0u);
    std::vector<std::uint32_t> host_pos(static_cast<std::size_t>(s.rows),
                                        static_cast<std::uint32_t>(s.seq_len - 1));
    if (hipMemcpy(b->table.p, host_table.data(), stride * sizeof(std::uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(b->slots.p, host_slots.data(), host_slots.size() * sizeof(std::uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(b->pos.p, host_pos.data(), host_pos.size() * sizeof(std::uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        *error = "KV fixture upload failed";
        return false;
    }
    return true;
}

// ---- 各カーネル -------------------------------------------------------------

int bench_gemm(const Args& a, std::int64_t gpu, const std::string& output, int argc) {
    bool ok = true;
    const std::vector<std::int64_t> rows_list = a.i64_list("rows", {1, 16}, &ok);
    const std::int64_t n = a.i64("n", 2048, &ok);
    const std::int64_t k = a.i64("k", 2048, &ok);
    std::int64_t warmup = 0, samples = 0, launches = 0;
    if (!common_iters(a, 100, &warmup, &samples, &launches, argc)) return kExitUsage;
    if (!ok || n <= 0 || k <= 0 || (k % 8) != 0 || rows_list.empty()) {
        std::fprintf(stderr, "gemm: require n,k >= 1, k %% 8 == 0, and a --rows list\n");
        return kExitUsage;
    }
    for (std::int64_t r : rows_list) {
        if (r < 1 || r > 64) {
            std::fprintf(stderr, "gemm: --rows must be in [1,64] (got %lld)\n",
                         static_cast<long long>(r));
            return kExitUsage;
        }
    }
    std::string error;
    if (!set_device(gpu, &error)) {
        std::fprintf(stderr, "gemm: %s\n", error.c_str());
        return kExitRuntime;
    }

    const std::size_t max_rows =
        static_cast<std::size_t>(*std::max_element(rows_list.begin(), rows_list.end()));
    DevBuf weight, input, out;
    if (!weight.alloc(static_cast<std::size_t>(n) * static_cast<std::size_t>(k) *
                          sizeof(shelf::bf16_t),
                      "weight", &error))
        goto fail;
    if (!input.alloc(max_rows * static_cast<std::size_t>(k) * sizeof(shelf::bf16_t),
                     "input", &error))
        goto fail;
    if (!out.alloc(max_rows * static_cast<std::size_t>(n) * sizeof(shelf::bf16_t), "out", &error))
        goto fail;

    {
        StreamTimer timer;
        if (!timer.init(&error)) {
            std::fprintf(stderr, "gemm: %s\n", error.c_str());
            return kExitRuntime;
        }
        for (std::int64_t rows : rows_list) {
            const shelf::hipk::Bf16GemmConfig cfg{
                rows >= 2 ? shelf::hipk::Bf16GemmConfigId::ExactRowsDot2
                          : shelf::hipk::Bf16GemmConfigId::ExactRows,
                static_cast<std::uint8_t>(rows)};
            const auto launch = [&](hipStream_t s) {
                return shelf::hipk::launch_gemm_bf16(
                    cfg, static_cast<const shelf::bf16_t*>(weight.p),
                    static_cast<const shelf::bf16_t*>(input.p), out.p,
                    shelf::hipk::Bf16GemmOutputDType::BF16, static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(k),
                    static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n), s);
            };
            Stats us;
            if (!timer.measure(static_cast<int>(warmup), static_cast<int>(samples),
                               static_cast<int>(launches), launch, &us, &error)) {
                std::fprintf(stderr, "gemm: %s\n", error.c_str());
                return kExitRuntime;
            }
            KernelResult r;
            r.name = "gemm";
            r.shape = std::to_string(rows) + "x" + std::to_string(n) + "x" +
                      std::to_string(k);
            r.us = us;
            const double flops = 2.0 * static_cast<double>(rows) * static_cast<double>(n) *
                                 static_cast<double>(k);
            r.throughput = us.p50 > 0.0 ? flops / (us.p50 * 1e-6) / 1e9 : 0.0;
            r.unit = "GFLOP/s";
            r.extra = {{"rows", std::to_string(rows)},
                       {"n", std::to_string(n)},
                       {"k", std::to_string(k)}};
            report(r, gpu, warmup, samples, launches, output);
        }
    }
    return kExitOk;

fail:
    std::fprintf(stderr, "gemm: %s\n", error.c_str());
    return kExitRuntime;
}

int bench_rmsnorm(const Args& a, std::int64_t gpu, const std::string& output, int argc) {
    bool ok = true;
    const std::int64_t rows = a.i64("rows", 1024, &ok);
    const std::int64_t features = a.i64("features", 2560, &ok);
    std::int64_t warmup = 0, samples = 0, launches = 0;
    if (!common_iters(a, 100, &warmup, &samples, &launches, argc)) return kExitUsage;
    if (!ok || rows <= 0 || features <= 0) {
        std::fprintf(stderr, "rmsnorm: require --rows >= 1 and --features >= 1\n");
        return kExitUsage;
    }
    std::string error;
    if (!set_device(gpu, &error)) {
        std::fprintf(stderr, "rmsnorm: %s\n", error.c_str());
        return kExitRuntime;
    }
    DevBuf in, weight, o;
    const auto bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(features) *
                       sizeof(shelf::bf16_t);
    if (!in.alloc(bytes, "input", &error) ||
        !weight.alloc(static_cast<std::size_t>(features) * sizeof(shelf::bf16_t), "weight",
                      &error) ||
        !o.alloc(bytes, "output", &error)) {
        std::fprintf(stderr, "rmsnorm: %s\n", error.c_str());
        return kExitRuntime;
    }
    StreamTimer timer;
    if (!timer.init(&error)) {
        std::fprintf(stderr, "rmsnorm: %s\n", error.c_str());
        return kExitRuntime;
    }
    // 生産経路と同じ形: 入力/重力 bf16、出力 bf16、PER_FEATURE + DIRECT。
    const auto launch = [&](hipStream_t s) {
        return shelf::hipk::launch_rmsnorm(
            in.p, shelf::hipk::RmsNormDataType::BF16, weight.p,
            shelf::hipk::RmsNormDataType::BF16, shelf::hipk::RmsNormWeightLayout::PER_FEATURE,
            shelf::hipk::RmsNormWeightMode::DIRECT, o.p, shelf::hipk::RmsNormDataType::BF16,
            static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(features),
            static_cast<std::uint32_t>(features), static_cast<std::uint32_t>(features),
            static_cast<std::uint32_t>(features), 1e-5f, s);
    };
    Stats us;
    if (!timer.measure(static_cast<int>(warmup), static_cast<int>(samples),
                       static_cast<int>(launches), launch, &us, &error)) {
        std::fprintf(stderr, "rmsnorm: %s\n", error.c_str());
        return kExitRuntime;
    }
    KernelResult r;
    r.name = "rmsnorm";
    r.shape = std::to_string(rows) + "x" + std::to_string(features);
    r.us = us;
    // 読み取り + weight + write（bf16）
    const double bytes_moved = 2.0 * static_cast<double>(bytes) +
                               static_cast<double>(features) * sizeof(shelf::bf16_t);
    r.throughput = us.p50 > 0.0 ? bytes_moved / (us.p50 * 1e-6) / 1e9 : 0.0;
    r.unit = "GB/s";
    r.extra = {{"rows", std::to_string(rows)}, {"features", std::to_string(features)}};
    report(r, gpu, warmup, samples, launches, output);
    return kExitOk;
}

int bench_elementwise(const Args& a, std::int64_t gpu, const std::string& output, int argc) {
    bool ok = true;
    const std::int64_t rows = a.i64("rows", 1024, &ok);
    const std::int64_t features = a.i64("features", 2560, &ok);
    std::int64_t warmup = 0, samples = 0, launches = 0;
    if (!common_iters(a, 100, &warmup, &samples, &launches, argc)) return kExitUsage;
    if (!ok || rows <= 0 || features <= 0) {
        std::fprintf(stderr, "elementwise: require --rows >= 1 and --features >= 1\n");
        return kExitUsage;
    }
    std::string error;
    if (!set_device(gpu, &error)) {
        std::fprintf(stderr, "elementwise: %s\n", error.c_str());
        return kExitRuntime;
    }
    DevBuf x, y, o;
    const auto bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(features) *
                       sizeof(shelf::bf16_t);
    if (!x.alloc(bytes, "x", &error) || !y.alloc(bytes, "y", &error) ||
        !o.alloc(bytes, "out", &error)) {
        std::fprintf(stderr, "elementwise: %s\n", error.c_str());
        return kExitRuntime;
    }
    StreamTimer timer;
    if (!timer.init(&error)) {
        std::fprintf(stderr, "elementwise: %s\n", error.c_str());
        return kExitRuntime;
    }
    const auto launch = [&](hipStream_t s) {
        return shelf::hipk::launch_residual_add_bf16(
            static_cast<const shelf::bf16_t*>(x.p), static_cast<std::uint32_t>(features),
            static_cast<const shelf::bf16_t*>(y.p), static_cast<std::uint32_t>(features),
            static_cast<shelf::bf16_t*>(o.p), static_cast<std::uint32_t>(features),
            static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(features), s);
    };
    Stats us;
    if (!timer.measure(static_cast<int>(warmup), static_cast<int>(samples),
                       static_cast<int>(launches), launch, &us, &error)) {
        std::fprintf(stderr, "elementwise: %s\n", error.c_str());
        return kExitRuntime;
    }
    KernelResult r;
    r.name = "elementwise";
    r.shape = std::to_string(rows) + "x" + std::to_string(features);
    r.us = us;
    const double bytes_moved = 3.0 * static_cast<double>(bytes);  // 読み取り 2 + write 1
    r.throughput = us.p50 > 0.0 ? bytes_moved / (us.p50 * 1e-6) / 1e9 : 0.0;
    r.unit = "GB/s";
    r.extra = {{"rows", std::to_string(rows)},
               {"features", std::to_string(features)},
               {"op", std::string("residual_add_bf16")}};
    report(r, gpu, warmup, samples, launches, output);
    return kExitOk;
}

bool fill_kv_args(const Args& a, KvSetup* s, std::int64_t default_warmup,
                  std::int64_t* warmup, std::int64_t* samples, std::int64_t* launches,
                  std::int64_t* q_heads, int argc) {
    bool ok = true;
    s->rows = a.i64("rows", 1, &ok);
    s->kv_heads = a.i64("kv-heads", 8, &ok);
    s->head_dim = a.i64("head-dim", 256, &ok);
    s->page_tokens = a.i64("page-tokens", 16, &ok);
    s->num_pages = a.i64("num-pages", 4, &ok);
    s->seq_len = a.i64("seq-len", 16, &ok);
    if (q_heads) *q_heads = a.i64("q-heads", 32, &ok);
    if (!common_iters(a, default_warmup, warmup, samples, launches, argc)) return false;
    if (!ok) {
        std::fprintf(stderr, "kernel: numeric option has a bad value\n");
        return false;
    }
    if (s->head_dim > 256) {  // kPagedAttentionHeadLimit
        std::fprintf(stderr, "kernel: --head-dim must be <= 256\n");
        return false;
    }
    return true;
}

int bench_kv_append(const Args& a, std::int64_t gpu, const std::string& output, int argc) {
    KvSetup s;
    std::int64_t warmup = 0, samples = 0, launches = 0, unused_q = 0;
    if (!fill_kv_args(a, &s, 30, &warmup, &samples, &launches, &unused_q, argc))
        return kExitUsage;
    if (s.rows <= 0 || s.kv_heads <= 0 || s.head_dim <= 0 || s.page_tokens <= 0 ||
        s.num_pages <= 0 || s.seq_len <= 0) {
        std::fprintf(stderr, "kv-append: bad option value\n");
        return kExitUsage;
    }
    std::string error;
    if (!set_device(gpu, &error)) {
        std::fprintf(stderr, "kv-append: %s\n", error.c_str());
        return kExitRuntime;
    }
    KvBufs kv;
    if (!setup_kv(s, &kv, &error)) {
        std::fprintf(stderr, "kv-append: %s\n", error.c_str());
        return kExitUsage;
    }
    const std::size_t tok_elems = static_cast<std::size_t>(s.kv_heads) *
                                  static_cast<std::size_t>(s.head_dim);
    DevBuf k_in, v_in;
    const auto in_bytes = static_cast<std::size_t>(s.rows) * tok_elems * sizeof(shelf::bf16_t);
    if (!k_in.alloc(in_bytes, "k_input", &error) || !v_in.alloc(in_bytes, "v_input", &error)) {
        std::fprintf(stderr, "kv-append: %s\n", error.c_str());
        return kExitRuntime;
    }
    StreamTimer timer;
    if (!timer.init(&error)) {
        std::fprintf(stderr, "kv-append: %s\n", error.c_str());
        return kExitRuntime;
    }

    shelf::hipk::KvAppendCommonArgs common{};
    common.k_input = static_cast<const shelf::bf16_t*>(k_in.p);
    common.v_input = static_cast<const shelf::bf16_t*>(v_in.p);
    common.k_row_stride = static_cast<std::uint32_t>(tok_elems);
    common.v_row_stride = static_cast<std::uint32_t>(tok_elems);
    common.row_sequence_slots = static_cast<const std::uint32_t*>(kv.slots.p);
    common.row_positions = static_cast<const std::uint32_t*>(kv.pos.p);
    common.block_tables = static_cast<const std::uint32_t*>(kv.table.p);
    common.block_table_stride = kv.stride;
    common.max_sequences = 1u;
    common.rows = static_cast<std::uint32_t>(s.rows);
    common.layer = 0u;
    common.page_tokens = static_cast<std::uint32_t>(s.page_tokens);
    common.num_pages = static_cast<std::uint32_t>(s.num_pages);
    common.num_attention_layers = 1u;
    common.kv_heads = static_cast<std::uint32_t>(s.kv_heads);
    common.head_dim = static_cast<std::uint32_t>(s.head_dim);
    common.elems_per_token = static_cast<std::uint32_t>(tok_elems);
    common.elems_per_page = static_cast<std::uint32_t>(
        static_cast<std::size_t>(s.page_tokens) * tok_elems);
    common.elems_per_layer = static_cast<std::uint32_t>(
        static_cast<std::size_t>(s.num_pages) * static_cast<std::size_t>(s.page_tokens) *
        tok_elems);

    const auto launch = [&](hipStream_t s) {
        return shelf::hipk::launch_kv_append_bf16(
            common, static_cast<shelf::bf16_t*>(kv.k_pool.p),
            static_cast<shelf::bf16_t*>(kv.v_pool.p), s);
    };
    Stats us;
    if (!timer.measure(static_cast<int>(warmup), static_cast<int>(samples),
                       static_cast<int>(launches), launch, &us, &error)) {
        std::fprintf(stderr, "kv-append: %s\n", error.c_str());
        return kExitRuntime;
    }
    KernelResult r;
    r.name = "kv-append";
    r.shape = std::to_string(s.rows) + "x" + std::to_string(s.kv_heads) + "x" +
              std::to_string(s.head_dim);
    r.us = us;
    // K/V それぞれ 読み取り + write
    const double bytes_moved = 4.0 * static_cast<double>(in_bytes);
    r.throughput = us.p50 > 0.0 ? bytes_moved / (us.p50 * 1e-6) / 1e9 : 0.0;
    r.unit = "GB/s";
    r.extra = {{"rows", std::to_string(s.rows)},
               {"kv_heads", std::to_string(s.kv_heads)},
               {"head_dim", std::to_string(s.head_dim)},
               {"seq_len", std::to_string(s.seq_len)}};
    report(r, gpu, warmup, samples, launches, output);
    return kExitOk;
}

int bench_paged_attention(const Args& a, std::int64_t gpu, const std::string& output,
                          int argc) {
    KvSetup s;
    std::int64_t warmup = 0, samples = 0, launches = 0, q_heads = 0;
    if (!fill_kv_args(a, &s, 30, &warmup, &samples, &launches, &q_heads, argc))
        return kExitUsage;
    if (s.rows <= 0 || q_heads <= 0 || s.kv_heads <= 0 || (q_heads % s.kv_heads) != 0 ||
        s.head_dim <= 0 || s.page_tokens <= 0 || s.num_pages <= 0 || s.seq_len <= 0) {
        std::fprintf(stderr,
                     "paged-attention: require rows,q_heads,kv_heads,page-tokens,"
                     "num-pages,seq-len >= 1 and q-heads %% kv-heads == 0\n");
        return kExitUsage;
    }
    std::string error;
    if (!set_device(gpu, &error)) {
        std::fprintf(stderr, "paged-attention: %s\n", error.c_str());
        return kExitRuntime;
    }
    KvBufs kv;
    if (!setup_kv(s, &kv, &error)) {
        std::fprintf(stderr, "paged-attention: %s\n", error.c_str());
        return kExitUsage;
    }
    const std::size_t tok_elems = static_cast<std::size_t>(s.kv_heads) *
                                  static_cast<std::size_t>(s.head_dim);
    const std::size_t out_elems = static_cast<std::size_t>(s.rows) *
                                  static_cast<std::size_t>(q_heads) *
                                  static_cast<std::size_t>(s.head_dim);
    DevBuf q, o;
    if (!q.alloc(out_elems * sizeof(shelf::bf16_t), "q", &error) ||
        !o.alloc(out_elems * sizeof(shelf::bf16_t), "out", &error)) {
        std::fprintf(stderr, "paged-attention: %s\n", error.c_str());
        return kExitRuntime;
    }
    StreamTimer timer;
    if (!timer.init(&error)) {
        std::fprintf(stderr, "paged-attention: %s\n", error.c_str());
        return kExitRuntime;
    }

    shelf::hipk::PagedAttentionCommonArgs common{};
    common.q = static_cast<const shelf::bf16_t*>(q.p);
    common.output = o.p;
    common.output_dtype = shelf::hipk::PagedAttentionOutputDType::BF16;
    common.q_row_stride = static_cast<std::uint32_t>(q_heads * s.head_dim);
    common.output_row_stride = static_cast<std::uint32_t>(q_heads * s.head_dim);
    common.row_sequence_slots = static_cast<const std::uint32_t*>(kv.slots.p);
    common.row_positions = static_cast<const std::uint32_t*>(kv.pos.p);
    common.block_tables = static_cast<const std::uint32_t*>(kv.table.p);
    common.block_table_stride = kv.stride;
    common.max_sequences = 1u;
    common.rows = static_cast<std::uint32_t>(s.rows);
    common.layer = 0u;
    common.page_tokens = static_cast<std::uint32_t>(s.page_tokens);
    common.num_pages = static_cast<std::uint32_t>(s.num_pages);
    common.num_attention_layers = 1u;
    common.q_heads = static_cast<std::uint32_t>(q_heads);
    common.kv_heads = static_cast<std::uint32_t>(s.kv_heads);
    common.head_dim = static_cast<std::uint32_t>(s.head_dim);
    common.elems_per_token = static_cast<std::uint32_t>(tok_elems);
    common.elems_per_page = static_cast<std::uint32_t>(
        static_cast<std::size_t>(s.page_tokens) * tok_elems);
    common.elems_per_layer = static_cast<std::uint32_t>(
        static_cast<std::size_t>(s.num_pages) * static_cast<std::size_t>(s.page_tokens) *
        tok_elems);
    common.scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));

    const auto launch = [&](hipStream_t st) {
        return shelf::hipk::launch_attention_paged_bf16(
            common, static_cast<const shelf::bf16_t*>(kv.k_pool.p),
            static_cast<const shelf::bf16_t*>(kv.v_pool.p), st);
    };
    Stats us;
    if (!timer.measure(static_cast<int>(warmup), static_cast<int>(samples),
                       static_cast<int>(launches), launch, &us, &error)) {
        std::fprintf(stderr, "paged-attention: %s\n", error.c_str());
        return kExitRuntime;
    }
    KernelResult r;
    r.name = "paged-attention";
    r.shape = std::to_string(s.rows) + "x" + std::to_string(q_heads) + "x" +
              std::to_string(s.kv_heads) + "x" + std::to_string(s.head_dim);
    r.us = us;
    r.throughput = 0.0;  // カーネルの読み出し量は可視長に依存するため指標化しない
    r.unit = "";
    r.extra = {{"rows", std::to_string(s.rows)},
               {"q_heads", std::to_string(q_heads)},
               {"kv_heads", std::to_string(s.kv_heads)},
               {"head_dim", std::to_string(s.head_dim)},
               {"seq_len", std::to_string(s.seq_len)}};
    report(r, gpu, warmup, samples, launches, output);
    return kExitOk;
}

#endif  // SHELF_HAVE_HIP

}  // namespace

int run_kernel(int argc, char** argv) {
    Args a;
    std::string err;
    if (!a.parse(argc, argv, &err)) {
        std::fprintf(stderr, "kernel: %s\n", err.c_str());
        usage(stderr);
        return kExitUsage;
    }
    if (a.has("help") || a.has("h")) {
        usage(stdout);
        return kExitOk;
    }
    if (a.positional().empty()) {
        std::fprintf(stderr, "kernel: <name> is required\n");
        usage(stderr);
        return kExitUsage;
    }
    const std::string name = a.positional().front();
    if (a.positional().size() > 1) {
        std::fprintf(stderr, "kernel: unexpected argument %s\n", a.positional()[1].c_str());
        usage(stderr);
        return kExitUsage;
    }
    bool known = false;
    for (const auto& e : kKernels)
        if (name == e.name) known = true;
    if (!known) {
        std::fprintf(stderr, "kernel: unknown name: %s\n", name.c_str());
        usage(stderr);
        return kExitUsage;
    }

    // カーネル別の既知キー（それ以外は exit 2）。
    {
        std::vector<std::string> allowed = {"help", "h", "warmup", "samples",
                                            "launches", "gpu", "output"};
        const auto add = [&](std::initializer_list<const char*> extra) {
            for (const char* k : extra) allowed.emplace_back(k);
        };
        if (name == "gemm") {
            add({"rows", "n", "k"});
        } else if (name == "rmsnorm" || name == "elementwise") {
            add({"rows", "features"});
        } else if (name == "kv-append") {
            add({"rows", "kv-heads", "head-dim", "page-tokens", "num-pages", "seq-len"});
        } else if (name == "paged-attention") {
            add({"rows", "q-heads", "kv-heads", "head-dim", "page-tokens",
                 "num-pages", "seq-len"});
        }
        std::string bad;
        if (!only_known_keys(a, allowed, &bad)) {
            std::fprintf(stderr, "kernel: unknown option for %s: --%s\n",
                         name.c_str(), bad.c_str());
            usage(stderr);
            return kExitUsage;
        }
    }

    bool ok = true;
    const std::int64_t gpu = a.i64("gpu", 0, &ok);
    const std::string output = a.str("output");
    if (!ok || gpu < 0) {
        std::fprintf(stderr, "kernel: bad option value\n");
        usage(stderr);
        return kExitUsage;
    }

#ifndef SHELF_HAVE_HIP
    std::fprintf(stderr,
                  "kernel: this build has no HIP backend (rebuild with SHELF_ENABLE_HIP=ON)\n");
    return kExitRuntime;
#else
    if (name == "gemm") return bench_gemm(a, gpu, output, argc);
    if (name == "rmsnorm") return bench_rmsnorm(a, gpu, output, argc);
    if (name == "elementwise") return bench_elementwise(a, gpu, output, argc);
    if (name == "kv-append") return bench_kv_append(a, gpu, output, argc);
    if (name == "paged-attention") return bench_paged_attention(a, gpu, output, argc);
    return kExitUsage;  // unreachable (known check above)
#endif  // SHELF_HAVE_HIP
}

}  // namespace shelf::bench
