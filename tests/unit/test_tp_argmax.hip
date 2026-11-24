#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/tp/tp_argmax.h>
#include <shelf/runtime/tp/tp_transports.h>
#include <shelf/runtime/tp/tp_barrier_group.h>

#include <thread>

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

static void test_kernel_argmax(int device, uint32_t vocab) {
    constexpr uint32_t rows = 5u;
    constexpr uint32_t ranks = 4u;
    const uint32_t local_cols = vocab / ranks;
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
    TpArgmaxPair* partials = nullptr;
    check(hipMalloc(&partials, rows * shelf::runtime::kTpArgmaxPartitions * sizeof(TpArgmaxPair)) == hipSuccess, "allocate argmax partials");
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
                                  vocab, local_cols, r * local_cols, stream, partials);
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
    check(got_ids[2] == static_cast<int32_t>(local_cols + 4u), "tie breaks toward the smaller global token id on device");

    for (void* ptr : logits_dev) shelf::gpu::discard_cleanup_result(hipFree(ptr));
    shelf::gpu::discard_cleanup_result(hipFree(pairs_dev));
    shelf::gpu::discard_cleanup_result(hipFree(ids_dev));
    shelf::gpu::discard_cleanup_result(hipFree(partials));
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
    check(hipMemcpy(da, a.data(), count * 2u, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(db, b.data(), count * 2u, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(dc, c.data(), count * 2u, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(dd, d.data(), count * 2u, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    void* fa_dev = nullptr;
    void* fb_dev = nullptr;
    void* fc_dev = nullptr;
    void* fd_dev = nullptr;
    check(hipMalloc(&fa_dev, count * 4u) == hipSuccess, "hipMalloc");
    check(hipMalloc(&fb_dev, count * 4u) == hipSuccess, "hipMalloc");
    check(hipMalloc(&fc_dev, count * 4u) == hipSuccess, "hipMalloc");
    check(hipMalloc(&fd_dev, count * 4u) == hipSuccess, "hipMalloc");
    check(hipMemcpy(fa_dev, fa.data(), count * 4u, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(fb_dev, fb.data(), count * 4u, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(fc_dev, fc.data(), count * 4u, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(fd_dev, fd.data(), count * 4u, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");

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

    check(hipMemcpy(out_bf.data(), o_bf, count * 2u, hipMemcpyDeviceToHost) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(out_f.data(), o_f, count * 4u, hipMemcpyDeviceToHost) == hipSuccess, "hipMemcpy");
    std::vector<std::uint16_t> host_bf16(count);
    check(hipMemcpy(host_bf16.data(), o_bf16, count * 2u, hipMemcpyDeviceToHost) == hipSuccess, "hipMemcpy");

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

static void test_large_four_input_sum(int device) {
    const std::size_t bytes = 2u * 1024u * 1024u + 64u;
    const std::size_t count = bytes / sizeof(std::uint16_t);
    auto scope = shelf::gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        fail("scoped device for large sum");
        return;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        fail("hipStreamCreate large sum");
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
        fail("hipMalloc large sum");
        return;
    }
    check(hipMemcpy(da, a.data(), bytes, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(db, b.data(), bytes, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(dc, c.data(), bytes, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    check(hipMemcpy(dd, d.data(), bytes, hipMemcpyHostToDevice) == hipSuccess, "hipMemcpy");
    const auto err = launch_tp_sum_bf16_4(
        static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(da), static_cast<const uint16_t*>(db),
        static_cast<const uint16_t*>(dc), static_cast<const uint16_t*>(dd), count, stream);
    check(err == hipSuccess, "launch large four-rank reduction");
    check(hipMemcpy(out.data(), dst, bytes, hipMemcpyDeviceToHost) == hipSuccess,
          "download large result");
    std::vector<std::uint16_t> want(count);
    tp_sum_bf16_4_host(want.data(), a.data(), b.data(), c.data(), d.data(), count);
    std::size_t mismatches = 0u;
    for (std::size_t i = 0u; i < count; ++i) {
        if (out[i] != want[i]) ++mismatches;
    }
    check(mismatches == 0u, "large reduce matches host reference");
    for (void* ptr : {da, db, dc, dd, dst}) shelf::gpu::discard_cleanup_result(hipFree(ptr));
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
}

static void test_staged_collective() {
    using namespace shelf::runtime;
    const std::vector<int> devices{0, 0};
    auto transport = HipPeerTpTransport::create(devices, TpCombineBackend::Event,
                                               TpPeerLinkMap(2, std::vector<uint8_t>(2, 0)));
    if (!transport.ok() || !transport.value()->verify().ok()) {
        fail("staged transport setup");
        return;
    }
    TpBarrierGroup group(2, devices, transport.value().get());
    if (!group.initialize().ok()) {
        fail("argmax barrier initialization");
        return;
    }
    auto run = [&](int rank, bool& passed) {
        hipStream_t stream{};
        float* logits{};
        int32_t* ids{};
        passed = hipSetDevice(0) == hipSuccess && hipStreamCreate(&stream) == hipSuccess &&
            hipMalloc(&logits, 12 * sizeof(float)) == hipSuccess &&
            hipMalloc(&ids, 32 * 3 * sizeof(int32_t)) == hipSuccess;
        const float data[2][12] = {{-2, -1, 0, 0, 9, 1, 0, 0, 4, 8, 0, 0},
                                   {0, 0, -3, 6, 0, 0, 9, 3, 0, 0, 7, 8}};
        if (passed) passed = hipMemcpy(logits, data[rank], sizeof(data[rank]), hipMemcpyHostToDevice) == hipSuccess;
        if (!passed) group.abort(shelf::Status::invalid_state("test setup failed", __FILE__, __LINE__));
        const int32_t want[3]{3, 0, 1};
        for (unsigned step = 0; passed && step < 32; ++step) {
            TpArgmaxInvocation in;
            in.local_offset = rank * 2u;
            in.local_logits = logits + in.local_offset;
            in.local_cols = 2;
            in.row_stride = in.vocab = 4;
            in.rows = step % 3 + 1;
            in.out_ids = ids + step * 3;
            in.stream = stream;
            passed = group.argmax_reduce(rank, in).ok();
            if (!passed) group.abort(shelf::Status::invalid_state("test argmax failed", __FILE__, __LINE__));
        }
        int32_t got[32 * 3]{};
        if (passed) passed = hipMemcpyAsync(got, ids, sizeof(got), hipMemcpyDeviceToHost, stream) == hipSuccess &&
            hipStreamSynchronize(stream) == hipSuccess;
        for (unsigned step = 0; passed && step < 32; ++step)
            passed = std::equal(got + 3 * step, got + 3 * step + step % 3 + 1, want);
        if (ids) shelf::gpu::discard_cleanup_result(hipFree(ids));
        if (logits) shelf::gpu::discard_cleanup_result(hipFree(logits));
        if (stream) shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    };
    bool first = false, second = false;
    std::thread a(run, 0, std::ref(first));
    std::thread b(run, 1, std::ref(second));
    a.join();
    b.join();
    check(first && second, "two staged ranks retain their inputs and agree on sampled IDs");
    check(group.shutdown().ok() && transport.value()->shutdown().ok() &&
          transport.value()->shutdown().ok(), "staged argmax shutdown is idempotent");
}

static void test_four_rank_order() {
    using namespace shelf::runtime;
    const std::vector<int> devices(4, 0);
    auto peer = HipPeerTpTransport::create(devices, TpCombineBackend::Event,
                                           TpPeerLinkMap(4, std::vector<uint8_t>(4, 0)));
    if (!peer.ok() || !peer.value()->verify().ok()) {
        fail("four staged ranks verify");
        return;
    }
    std::vector<TpSumTarget> targets(4);
    std::vector<hipStream_t> streams(4);
    std::vector<hipEvent_t> ready(4);
    const float initial[]{1.0e20f, -1.0e20f, 1.0f, 2.0f};
    bool valid = true;
    for (unsigned r = 0; r < 4 && valid; ++r) {
        auto& t = targets[r];
        t.rows = t.row_stride = t.feature_count = 1;
        t.dtype = ValueDType::F32;
        valid = hipStreamCreate(&streams[r]) == hipSuccess && hipEventCreate(&ready[r]) == hipSuccess &&
            hipMalloc(&t.ptr, sizeof(float)) == hipSuccess &&
            hipMemcpyAsync(t.ptr, initial + r, sizeof(float), hipMemcpyHostToDevice, streams[r]) == hipSuccess &&
            hipEventRecord(ready[r], streams[r]) == hipSuccess;
    }
    if (valid) valid = peer.value()->sum_hidden(TpSumInvocation{targets, streams, ready}).ok();
    for (unsigned r = 0; r < 4; ++r) {
        float result{};
        if (valid) valid = hipStreamSynchronize(streams[r]) == hipSuccess &&
            hipMemcpy(&result, targets[r].ptr, sizeof(result), hipMemcpyDeviceToHost) == hipSuccess &&
            result == 3.0f;
        if (targets[r].ptr) shelf::gpu::discard_cleanup_result(hipFree(targets[r].ptr));
        if (ready[r]) shelf::gpu::discard_cleanup_result(hipEventDestroy(ready[r]));
        if (streams[r]) shelf::gpu::discard_cleanup_result(hipStreamDestroy(streams[r]));
    }
    check(valid, "four ranks sum in the same order under cancellation");
    check(peer.value()->shutdown().ok(), "four-rank transport shutdown");
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("SKIP: tp argmax kernels require a GPU\n");
        return 77;
    }
    test_four_rank_order();
    test_staged_collective();
    test_kernel_argmax(0, 40u);
    test_kernel_argmax(0, 248320u);
    test_kernel_four_input_sum(0);
    test_large_four_input_sum(0);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
