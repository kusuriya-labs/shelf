#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/tp/tp_transports.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using ps::Status;
using ps::runtime::HostMediatedTpTransport;
using ps::runtime::TpSumInvocation;
using ps::runtime::TpSumTarget;
using ps::runtime::TpTransport;
using ps::runtime::ValueDType;
using ps::runtime::create_tp_transport;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static std::vector<std::uint8_t> make_reference(const std::vector<std::uint8_t>& a,
                                                const std::vector<std::uint8_t>& b,
                                                ValueDType dtype) {
    std::vector<std::uint8_t> out(a.size());
    if (dtype == ValueDType::F32) {
        auto* d = reinterpret_cast<float*>(out.data());
        const auto* pa = reinterpret_cast<const float*>(a.data());
        const auto* pb = reinterpret_cast<const float*>(b.data());
        for (std::size_t i = 0; i < a.size() / 4; ++i) d[i] = pa[i] + pb[i];
        return out;
    }
    auto* d = reinterpret_cast<std::uint16_t*>(out.data());
    const auto* pa = reinterpret_cast<const std::uint16_t*>(a.data());
    const auto* pb = reinterpret_cast<const std::uint16_t*>(b.data());
    for (std::size_t i = 0; i < a.size() / 2; ++i) {
        float fa;
        float fb;
        const std::uint32_t ua = static_cast<std::uint32_t>(pa[i]) << 16;
        const std::uint32_t ub = static_cast<std::uint32_t>(pb[i]) << 16;
        std::memcpy(&fa, &ua, sizeof(fa));
        std::memcpy(&fb, &ub, sizeof(fb));
        d[i] = ps::f32_to_bf16_rne(fa + fb);
    }
    return out;
}

static bool make_pair(int device, void*& buffer, hipStream_t& stream, hipEvent_t& ready,
                      std::size_t bytes) {
    auto scope = ps::gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        fail("device scope " + std::to_string(device));
        return false;
    }
    if (hipStreamCreate(&stream) != hipSuccess) {
        fail("hipStreamCreate");
        return false;
    }
    if (hipEventCreate(&ready) != hipSuccess) {
        fail("hipEventCreate");
        return false;
    }
    if (hipMalloc(&buffer, bytes) != hipSuccess) {
        fail("hipMalloc");
        return false;
    }
    return true;
}

static void destroy_pair(int device, void*& buffer, hipStream_t& stream,
                         hipEvent_t& ready) {
    auto scope = ps::gpu::ScopedDevice::create(device);
    if (!scope.ok()) return;
    if (buffer != nullptr) ps::gpu::discard_cleanup_result(hipFree(buffer));
    if (ready != nullptr) ps::gpu::discard_cleanup_result(hipEventDestroy(ready));
    if (stream != nullptr) ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    buffer = nullptr;
    ready = nullptr;
    stream = nullptr;
}

static void run_case(TpTransport* transport, void* buffer_a, hipStream_t stream_a,
                     hipEvent_t ready_a, void* buffer_b, hipStream_t stream_b,
                     hipEvent_t ready_b, std::uint64_t rows, std::uint32_t features,
                     ValueDType dtype, const std::string& tag, std::uint32_t row_stride = 0) {
    const std::size_t elem = dtype == ValueDType::F32 ? 4u : 2u;
    const std::size_t stride = row_stride == 0 ? features : row_stride;
    const std::size_t bytes = static_cast<std::size_t>(rows) * stride * elem;

    std::vector<std::uint8_t> host_a(bytes);
    std::vector<std::uint8_t> host_b(bytes);
    if (dtype == ValueDType::F32) {
        auto* fa = reinterpret_cast<float*>(host_a.data());
        auto* fb = reinterpret_cast<float*>(host_b.data());
        for (std::uint64_t i = 0; i < rows * stride; ++i) {
            fa[i] = static_cast<float>((i * 17u) % 251u) * 0.05f - 6.0f;
            fb[i] = static_cast<float>((i * 29u) % 241u) * 0.05f - 6.0f;
        }
    } else {
        auto* ba = reinterpret_cast<std::uint16_t*>(host_a.data());
        auto* bb = reinterpret_cast<std::uint16_t*>(host_b.data());
        for (std::uint64_t i = 0; i < rows * stride; ++i) {
            const float va =
                static_cast<float>((i * 37u + 11u) % 1009u) * (1.0f / 64.0f) - 8.0f;
            const float vb =
                static_cast<float>((i * 53u + 7u) % 997u) * (1.0f / 64.0f) - 8.0f;
            ba[i] = ps::f32_to_bf16_rne(va);
            bb[i] = ps::f32_to_bf16_rne(vb);
        }
    }

    {
        auto scope = ps::gpu::ScopedDevice::create(0);
        if (!scope.ok()) { fail("upload scope 0"); return; }
        if (hipMemcpy(buffer_a, host_a.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
            fail("upload a");
            return;
        }
        if (hipEventRecord(ready_a, stream_a) != hipSuccess) fail("record ready a");
    }
    {
        auto scope = ps::gpu::ScopedDevice::create(1);
        if (!scope.ok()) { fail("upload scope 1"); return; }
        if (hipMemcpy(buffer_b, host_b.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
            fail("upload b");
            return;
        }
        if (hipEventRecord(ready_b, stream_b) != hipSuccess) fail("record ready b");
    }

    TpSumTarget target_a;
    target_a.ptr = buffer_a;
    target_a.rows = rows;
    target_a.row_stride = stride;
    target_a.feature_count = features;
    target_a.dtype = dtype;
    TpSumTarget target_b = target_a;
    target_b.ptr = buffer_b;

    std::vector<TpSumTarget> targets{target_a, target_b};
    std::vector<hipStream_t> streams{stream_a, stream_b};
    std::vector<hipEvent_t> ready{ready_a, ready_b};
    TpSumInvocation invocation{targets, streams, ready};

    const std::string case_tag = tag + " rows=" + std::to_string(rows) +
                                 " features=" + std::to_string(features) +
                                 " stride=" + std::to_string(stride);
    Status st = transport->sum_hidden(invocation);
    check(st.ok(), case_tag + " sum_hidden: " + st.message());
    if (!st.ok()) return;

    {
        auto scope = ps::gpu::ScopedDevice::create(0);
        if (scope.ok() && hipStreamSynchronize(stream_a) != hipSuccess) fail("sync a");
    }
    {
        auto scope = ps::gpu::ScopedDevice::create(1);
        if (scope.ok() && hipStreamSynchronize(stream_b) != hipSuccess) fail("sync b");
    }

    const std::vector<std::uint8_t> reference = make_reference(host_a, host_b, dtype);
    std::vector<std::uint8_t> got_a(bytes);
    std::vector<std::uint8_t> got_b(bytes);
    {
        auto scope = ps::gpu::ScopedDevice::create(0);
        if (scope.ok() && hipMemcpy(got_a.data(), buffer_a, bytes,
                                    hipMemcpyDeviceToHost) != hipSuccess)
            fail("download a");
    }
    {
        auto scope = ps::gpu::ScopedDevice::create(1);
        if (scope.ok() && hipMemcpy(got_b.data(), buffer_b, bytes,
                                    hipMemcpyDeviceToHost) != hipSuccess)
            fail("download b");
    }
    check(got_a == reference, case_tag + " rank0 result matches CPU reference");
    check(got_b == reference, case_tag + " rank1 result matches CPU reference");
    check(got_a == got_b, case_tag + " ranks are identical after broadcast");
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: tp transport oracle requires 2 GPUs\n");
        return 77;
    }

    const std::vector<int> devices{0, 1};
    auto selection = create_tp_transport(devices);
    check(selection.ok(), "create_tp_transport");
    if (!selection.ok()) {
        std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    TpTransport* transport = selection.value().transport.get();
    std::printf("transport under test: %s (peer_access_available=%d)\n",
                transport->name(), selection.value().peer_access_available ? 1 : 0);

    constexpr std::uint64_t kMaxRows = 2048;
    constexpr std::uint32_t kFeatures = 5120;
    constexpr std::size_t kMaxBytes = kMaxRows * kFeatures * 4;

    void* buffer_a = nullptr;
    void* buffer_b = nullptr;
    hipStream_t stream_a = nullptr;
    hipStream_t stream_b = nullptr;
    hipEvent_t ready_a = nullptr;
    hipEvent_t ready_b = nullptr;
    if (!make_pair(0, buffer_a, stream_a, ready_a, kMaxBytes)) return 1;
    if (!make_pair(1, buffer_b, stream_b, ready_b, kMaxBytes)) return 1;

    const Status verify_status = transport->verify();
    const bool peer_usable = verify_status.ok();
    if (!peer_usable) {
        std::printf(
            "[tp] peer transport self-test failed (%s); using host-mediated "
            "reference transport\n",
            verify_status.message().c_str());
    }

    if (peer_usable) {
        for (std::uint64_t rows : {1ull, 2ull, 4ull, 8ull, 16ull, 32ull, 64ull, 128ull,
                                   256ull, 512ull, 1024ull, 2048ull}) {
            run_case(transport, buffer_a, stream_a, ready_a, buffer_b, stream_b, ready_b,
                     rows, kFeatures, ValueDType::BF16, transport->name());
        }
        run_case(transport, buffer_a, stream_a, ready_a, buffer_b, stream_b, ready_b, 1,
                 kFeatures, ValueDType::F32, transport->name());
        run_case(transport, buffer_a, stream_a, ready_a, buffer_b, stream_b, ready_b, 137,
                 kFeatures, ValueDType::F32, transport->name());
        run_case(transport, buffer_a, stream_a, ready_a, buffer_b, stream_b, ready_b, 64,
                 kFeatures, ValueDType::BF16, transport->name(), kFeatures + 16);
    }

    HostMediatedTpTransport host_transport(devices);
    for (std::uint64_t rows : {1ull, 7ull, 64ull, 513ull, 1024ull, 2048ull}) {
        run_case(&host_transport, buffer_a, stream_a, ready_a, buffer_b, stream_b,
                 ready_b, rows, kFeatures, ValueDType::BF16, "host-mediated");
    }
    run_case(&host_transport, buffer_a, stream_a, ready_a, buffer_b, stream_b, ready_b,
             9, kFeatures, ValueDType::F32, "host-mediated");
    run_case(&host_transport, buffer_a, stream_a, ready_a, buffer_b, stream_b, ready_b,
             17, kFeatures, ValueDType::BF16, "host-mediated padded", kFeatures + 64);

    destroy_pair(0, buffer_a, stream_a, ready_a);
    destroy_pair(1, buffer_b, stream_b, ready_b);

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
