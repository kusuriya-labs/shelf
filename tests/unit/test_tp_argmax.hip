#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/tp/tp_argmax.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using shelf::runtime::TpArgmaxPair;
using shelf::runtime::launch_tp_argmax_pack;
using shelf::runtime::launch_tp_argmax_reduce;
using shelf::runtime::launch_tp_sum_bf16_4;
using shelf::runtime::launch_tp_sum_f32_4;
using shelf::runtime::launch_tp_sum_f32_to_bf16_4;
using shelf::runtime::tp_argmax_pack_host;
using shelf::runtime::tp_argmax_reduce_host;
using shelf::runtime::tp_reduce_four_input_shots;
using shelf::runtime::tp_sum_bf16_4_host;
using shelf::runtime::tp_sum_f32_4_host;

static int g_fail = 0;

static void fail(const char* msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg);
}

static void check(bool cond, const char* msg) {
    if (!cond) fail(msg);
}

static void test_kernel_argmax(int device) {
    constexpr uint32_t rows = 5u;
    constexpr uint32_t vocab = 40u;
    constexpr uint32_t ranks = 4u;
    constexpr uint32_t local_cols = vocab / ranks;
    constexpr uint32_t stride = rows + 2u;
    auto scope = shelf::gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        fail("scoped device for argmax kernel");
        return;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        fail("hipStreamCreate argmax");
        return;
    }

    std::vector<std::vector<float>> host(ranks, std::vector<float>(rows * vocab, -9.0f));
    for (uint32_t r = 0u; r < ranks; ++r) {
        for (uint32_t row = 0u; row < rows; ++row) {
            for (uint32_t c = 0u; c < local_cols; ++c) {
                host[r][row * vocab + c] =
                    static_cast<float>((row * 5u + c * 3u + r * 7u) % 19u);
            }
        }
    }
    host[1][2 * vocab + 4u] = 100.0f;
    host[2][2 * vocab + 3u] = 100.0f;

    std::vector<void*> logits_dev(ranks, nullptr);
    void* pairs_dev = nullptr;
    void* ids_dev = nullptr;
    for (uint32_t r = 0u; r < ranks; ++r) {
        if (hipMalloc(&logits_dev[r], rows * vocab * sizeof(float)) != hipSuccess) {
            fail("hipMalloc argmax logits");
            return;
        }
        if (hipMemcpy(logits_dev[r], host[r].data(), rows * vocab * sizeof(float),
                      hipMemcpyHostToDevice) != hipSuccess) {
            fail("upload argmax logits");
            return;
        }
    }
    if (hipMalloc(&pairs_dev, ranks * stride * sizeof(TpArgmaxPair)) != hipSuccess) {
        fail("hipMalloc dedicated pair buffer");
        return;
    }
    if (hipMalloc(&ids_dev, rows * sizeof(int32_t)) != hipSuccess) {
        fail("hipMalloc argmax ids");
        return;
    }
    check(pairs_dev != logits_dev[0], "device pair buffer is separate from logits");

    for (uint32_t r = 0u; r < ranks; ++r) {
        const hipError_t err =
            launch_tp_argmax_pack(static_cast<const float*>(logits_dev[r]),
                                  static_cast<TpArgmaxPair*>(pairs_dev) + r * stride, rows,
                                  vocab, local_cols, r * local_cols, stream);
        check(err == hipSuccess, "launch argmax pack");
    }
    const hipError_t reduce_err = launch_tp_argmax_reduce(
        static_cast<const TpArgmaxPair*>(pairs_dev), stride, ranks, rows,
        static_cast<int32_t*>(ids_dev), stream);
    check(reduce_err == hipSuccess, "launch argmax reduce");

    std::vector<TpArgmaxPair> got_pairs(ranks * stride);
    std::vector<int32_t> got_ids(rows, -100);
    if (hipMemcpy(got_pairs.data(), pairs_dev, got_pairs.size() * sizeof(TpArgmaxPair),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        fail("download argmax pairs");
        return;
    }
    if (hipMemcpy(got_ids.data(), ids_dev, rows * sizeof(int32_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        fail("download argmax ids");
        return;
    }

    std::vector<TpArgmaxPair> want_pairs(ranks * stride);
    for (uint32_t r = 0u; r < ranks; ++r) {
        tp_argmax_pack_host(host[r].data(), rows, vocab, local_cols, r * local_cols,
                            want_pairs.data() + r * stride);
    }
    std::vector<int32_t> want_ids(rows, -1);
    tp_argmax_reduce_host(want_pairs.data(), stride, ranks, rows, want_ids.data());

    for (uint32_t r = 0u; r < ranks; ++r) {
        for (uint32_t row = 0u; row < rows; ++row) {
            const std::size_t i = r * stride + row;
            check(got_pairs[i].value == want_pairs[i].value && got_pairs[i].id == want_pairs[i].id,
                  "device argmax pack matches host reference");
        }
    }
    for (uint32_t row = 0u; row < rows; ++row) {
        check(got_ids[row] == want_ids[row], "device argmax reduce matches host reference");
    }
    check(got_ids[2] == 14, "tie breaks toward the smaller global token id on device");

    for (void* ptr : logits_dev) shelf::gpu::discard_cleanup_result(hipFree(ptr));
    shelf::gpu::discard_cleanup_result(hipFree(pairs_dev));
    shelf::gpu::discard_cleanup_result(hipFree(ids_dev));
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
}

static void test_kernel_four_input_sum(int device) {
    constexpr std::size_t count = 1023u;
    auto scope = shelf::gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        fail("scoped device for four input sum");
        return;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        fail("hipStreamCreate four input sum");
        return;
    }

    std::vector<std::uint16_t> a(count), b(count), c(count), d(count), out_bf(count);
    std::vector<float> fa(count), fb(count), fc(count), fd(count), out_f(count), out_bf16(count);
    for (std::size_t i = 0u; i < count; ++i) {
        a[i] = shelf::f32_to_bf16_rne(static_cast<float>(i) * 0.25f);
        b[i] = shelf::f32_to_bf16_rne(static_cast<float>(i) * 0.5f + 1.0f);
        c[i] = shelf::f32_to_bf16_rne(static_cast<float>(i) * 0.125f + 2.0f);
        d[i] = shelf::f32_to_bf16_rne(static_cast<float>(i) * 0.75f + 3.0f);
        fa[i] = static_cast<float>(i) + 0.25f;
        fb[i] = static_cast<float>(i) + 0.5f;
        fc[i] = static_cast<float>(i) + 0.75f;
        fd[i] = static_cast<float>(i) + 1.0f;
    }

    void* da = nullptr;
    void* db = nullptr;
    void* dc = nullptr;
    void* dd = nullptr;
    void* o_bf = nullptr;
    void* o_f = nullptr;
    void* o_bf16 = nullptr;
    if (hipMalloc(&da, count * 2u) != hipSuccess || hipMalloc(&db, count * 2u) != hipSuccess ||
        hipMalloc(&dc, count * 2u) != hipSuccess || hipMalloc(&dd, count * 2u) != hipSuccess ||
        hipMalloc(&o_bf, count * 2u) != hipSuccess || hipMalloc(&o_f, count * 4u) != hipSuccess ||
        hipMalloc(&o_bf16, count * 2u) != hipSuccess) {
        fail("hipMalloc four input sum");
        return;
    }
    hipMemcpy(da, a.data(), count * 2u, hipMemcpyHostToDevice);
    hipMemcpy(db, b.data(), count * 2u, hipMemcpyHostToDevice);
    hipMemcpy(dc, c.data(), count * 2u, hipMemcpyHostToDevice);
    hipMemcpy(dd, d.data(), count * 2u, hipMemcpyHostToDevice);
    void* fa_dev = nullptr;
    void* fb_dev = nullptr;
    void* fc_dev = nullptr;
    void* fd_dev = nullptr;
    hipMalloc(&fa_dev, count * 4u);
    hipMalloc(&fb_dev, count * 4u);
    hipMalloc(&fc_dev, count * 4u);
    hipMalloc(&fd_dev, count * 4u);
    hipMemcpy(fa_dev, fa.data(), count * 4u, hipMemcpyHostToDevice);
    hipMemcpy(fb_dev, fb.data(), count * 4u, hipMemcpyHostToDevice);
    hipMemcpy(fc_dev, fc.data(), count * 4u, hipMemcpyHostToDevice);
    hipMemcpy(fd_dev, fd.data(), count * 4u, hipMemcpyHostToDevice);

    check(launch_tp_sum_bf16_4(static_cast<std::uint16_t*>(o_bf),
                               static_cast<const std::uint16_t*>(da),
                               static_cast<const std::uint16_t*>(db),
                               static_cast<const std::uint16_t*>(dc),
                               static_cast<const std::uint16_t*>(dd), count,
                               stream) == hipSuccess,
          "launch four BF16 sum");
    check(launch_tp_sum_f32_4(static_cast<float*>(o_f), static_cast<const float*>(fa_dev),
                              static_cast<const float*>(fb_dev),
                              static_cast<const float*>(fc_dev),
                              static_cast<const float*>(fd_dev), count,
                              stream) == hipSuccess,
          "launch four F32 sum");
    check(launch_tp_sum_f32_to_bf16_4(static_cast<std::uint16_t*>(o_bf16),
                                      static_cast<const float*>(fa_dev),
                                      static_cast<const float*>(fb_dev),
                                      static_cast<const float*>(fc_dev),
                                      static_cast<const float*>(fd_dev), count,
                                      stream) == hipSuccess,
          "launch four F32 to BF16 sum");

    hipMemcpy(out_bf.data(), o_bf, count * 2u, hipMemcpyDeviceToHost);
    hipMemcpy(out_f.data(), o_f, count * 4u, hipMemcpyDeviceToHost);
    std::vector<std::uint16_t> host_bf16(count);
    hipMemcpy(host_bf16.data(), o_bf16, count * 2u, hipMemcpyDeviceToHost);

    std::vector<std::uint16_t> want_bf(count);
    tp_sum_bf16_4_host(want_bf.data(), a.data(), b.data(), c.data(), d.data(), count);
    std::vector<float> want_f(count);
    tp_sum_f32_4_host(want_f.data(), fa.data(), fb.data(), fc.data(), fd.data(), count);
    std::vector<std::uint16_t> want_bf16(count);
    for (std::size_t i = 0u; i < count; ++i) {
        want_bf16[i] = shelf::f32_to_bf16_rne(want_f[i]);
    }
    for (std::size_t i = 0u; i < count; ++i) {
        check(out_bf[i] == want_bf[i], "device four BF16 sum matches host reference");
        check(out_f[i] == want_f[i], "device four F32 sum matches host reference");
        check(host_bf16[i] == want_bf16[i], "device four F32 to BF16 sum matches host reference");
    }
    for (void* ptr : {da, db, dc, dd, o_bf, o_f, o_bf16, fa_dev, fb_dev, fc_dev, fd_dev}) {
        shelf::gpu::discard_cleanup_result(hipFree(ptr));
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
}

static void test_two_shot_four_input_sum(int device) {
    const std::size_t bytes = 2u * 1024u * 1024u + 64u;
    const std::size_t count = bytes / sizeof(std::uint16_t);
    check(tp_reduce_four_input_shots(bytes) == 2u, "large reduce uses two shots");
    auto scope = shelf::gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        fail("scoped device for two shot sum");
        return;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        fail("hipStreamCreate two shot sum");
        return;
    }
    std::vector<std::uint16_t> a(count), b(count), c(count), d(count), out(count);
    for (std::size_t i = 0u; i < count; ++i) {
        a[i] = shelf::f32_to_bf16_rne(static_cast<float>(i % 17u));
        b[i] = shelf::f32_to_bf16_rne(static_cast<float>(i % 19u) + 1.0f);
        c[i] = shelf::f32_to_bf16_rne(static_cast<float>(i % 23u) + 2.0f);
        d[i] = shelf::f32_to_bf16_rne(static_cast<float>(i % 29u) + 3.0f);
    }
    void* da = nullptr;
    void* db = nullptr;
    void* dc = nullptr;
    void* dd = nullptr;
    void* dst = nullptr;
    if (hipMalloc(&da, bytes) != hipSuccess || hipMalloc(&db, bytes) != hipSuccess ||
        hipMalloc(&dc, bytes) != hipSuccess || hipMalloc(&dd, bytes) != hipSuccess ||
        hipMalloc(&dst, bytes) != hipSuccess) {
        fail("hipMalloc two shot sum");
        return;
    }
    hipMemcpy(da, a.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(db, b.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(dc, c.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(dd, d.data(), bytes, hipMemcpyHostToDevice);
    const std::size_t chunk = (count + 1u) / 2u;
    for (int shot = 0; shot < 2; ++shot) {
        const std::size_t begin = static_cast<std::size_t>(shot) * chunk;
        if (begin >= count) continue;
        const std::size_t length = std::min(chunk, count - begin);
        const hipError_t err = launch_tp_sum_bf16_4(
            static_cast<std::uint16_t*>(dst) + begin,
            static_cast<const std::uint16_t*>(da) + begin,
            static_cast<const std::uint16_t*>(db) + begin,
            static_cast<const std::uint16_t*>(dc) + begin,
            static_cast<const std::uint16_t*>(dd) + begin, length, stream);
        check(err == hipSuccess, "launch two shot chunk");
    }
    check(hipMemcpy(out.data(), dst, bytes, hipMemcpyDeviceToHost) == hipSuccess,
          "download two shot result");
    std::vector<std::uint16_t> want(count);
    tp_sum_bf16_4_host(want.data(), a.data(), b.data(), c.data(), d.data(), count);
    std::size_t mismatches = 0u;
    for (std::size_t i = 0u; i < count; ++i) {
        if (out[i] != want[i]) ++mismatches;
    }
    check(mismatches == 0u, "two shot reduce matches host reference");
    for (void* ptr : {da, db, dc, dd, dst}) shelf::gpu::discard_cleanup_result(hipFree(ptr));
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("SKIP: tp argmax kernels require a GPU\n");
        return 77;
    }
    test_kernel_argmax(0);
    test_kernel_four_input_sum(0);
    test_two_shot_four_input_sum(0);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
