#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/tp/tp_transports.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using shelf::Status;
using shelf::runtime::HostMediatedTpTransport;
using shelf::runtime::TpSumInvocation;
using shelf::runtime::TpSumTarget;
using shelf::runtime::TpTransport;
using shelf::runtime::ValueDType;
using shelf::runtime::create_tp_transport;

static int g_fail = 0;

static float bf16_to_f32_host(std::uint16_t v) {
    const std::uint32_t u = static_cast<std::uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

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
        d[i] = shelf::f32_to_bf16_rne(fa + fb);
    }
    return out;
}

static bool make_pair(int device, void*& buffer, hipStream_t& stream, hipEvent_t& ready,
                      std::size_t bytes) {
    auto scope = shelf::gpu::ScopedDevice::create(device);
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
    auto scope = shelf::gpu::ScopedDevice::create(device);
    if (!scope.ok()) return;
    if (buffer != nullptr) shelf::gpu::discard_cleanup_result(hipFree(buffer));
    if (ready != nullptr) shelf::gpu::discard_cleanup_result(hipEventDestroy(ready));
    if (stream != nullptr) shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
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
            ba[i] = shelf::f32_to_bf16_rne(va);
            bb[i] = shelf::f32_to_bf16_rne(vb);
        }
    }

    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (!scope.ok()) { fail("upload scope 0"); return; }
        if (hipMemcpy(buffer_a, host_a.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
            fail("upload a");
            return;
        }
        if (hipEventRecord(ready_a, stream_a) != hipSuccess) fail("record ready a");
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
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
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (scope.ok() && hipStreamSynchronize(stream_a) != hipSuccess) fail("sync a");
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (scope.ok() && hipStreamSynchronize(stream_b) != hipSuccess) fail("sync b");
    }

    const std::vector<std::uint8_t> reference = make_reference(host_a, host_b, dtype);
    std::vector<std::uint8_t> got_a(bytes);
    std::vector<std::uint8_t> got_b(bytes);
    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (scope.ok() && hipMemcpy(got_a.data(), buffer_a, bytes,
                                    hipMemcpyDeviceToHost) != hipSuccess)
            fail("download a");
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (scope.ok() && hipMemcpy(got_b.data(), buffer_b, bytes,
                                    hipMemcpyDeviceToHost) != hipSuccess)
            fail("download b");
    }
    check(got_a == reference, case_tag + " rank0 result matches CPU reference");
    check(got_b == reference, case_tag + " rank1 result matches CPU reference");
    check(got_a == got_b, case_tag + " ranks are identical after broadcast");
}

static void run_case_fused(TpTransport* transport, void* buffer_a, hipStream_t stream_a,
                           void* buffer_b, hipStream_t stream_b, std::uint64_t rows,
                           std::uint32_t features) {
    const std::size_t elems = static_cast<std::size_t>(rows) * features;
    const std::size_t bytes = elems * sizeof(std::uint16_t);
    std::vector<std::uint8_t> host_a(bytes);
    std::vector<std::uint8_t> host_b(bytes);
    auto* ba = reinterpret_cast<std::uint16_t*>(host_a.data());
    auto* bb = reinterpret_cast<std::uint16_t*>(host_b.data());
    for (std::size_t i = 0; i < elems; ++i) {
        const float va = static_cast<float>((i * 37u + 11u) % 1009u) * (1.0f / 64.0f) - 8.0f;
        const float vb = static_cast<float>((i * 53u + 7u) % 997u) * (1.0f / 64.0f) - 8.0f;
        ba[i] = shelf::f32_to_bf16_rne(va);
        bb[i] = shelf::f32_to_bf16_rne(vb);
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (!scope.ok()) { fail("fused upload scope 0"); return; }
        if (hipMemcpy(buffer_a, host_a.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
            fail("fused upload a");
            return;
        }
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (!scope.ok()) { fail("fused upload scope 1"); return; }
        if (hipMemcpy(buffer_b, host_b.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
            fail("fused upload b");
            return;
        }
    }

    TpSumTarget target_a;
    target_a.ptr = buffer_a;
    target_a.rows = rows;
    target_a.row_stride = features;
    target_a.feature_count = features;
    target_a.dtype = ValueDType::BF16;
    TpSumTarget target_b = target_a;
    target_b.ptr = buffer_b;

    const std::string tag = "fused rows=" + std::to_string(rows);
    Status st = transport->sum_hidden_fused(0, target_a, stream_a, false);
    check(st.ok(), tag + " rank0: " + st.message());
    if (!st.ok()) return;
    st = transport->sum_hidden_fused(1, target_b, stream_b, false);
    check(st.ok(), tag + " rank1: " + st.message());
    if (!st.ok()) return;
    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (scope.ok() && hipStreamSynchronize(stream_a) != hipSuccess) fail("fused sync a");
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (scope.ok() && hipStreamSynchronize(stream_b) != hipSuccess) fail("fused sync b");
    }

    std::vector<std::uint16_t> got_a(elems);
    std::vector<std::uint16_t> got_b(elems);
    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (scope.ok() && hipMemcpy(got_a.data(), buffer_a, bytes,
                                    hipMemcpyDeviceToHost) != hipSuccess)
            fail("fused download a");
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (scope.ok() && hipMemcpy(got_b.data(), buffer_b, bytes,
                                    hipMemcpyDeviceToHost) != hipSuccess)
            fail("fused download b");
    }
    for (std::size_t i = 0; i < elems; ++i) {
        const std::uint16_t want =
            shelf::f32_to_bf16_rne(bf16_to_f32_host(ba[i]) +
                                   bf16_to_f32_host(bb[i]));
        if (got_a[i] != want || got_b[i] != want) {
            fail(tag + " result mismatch at " + std::to_string(i));
            return;
        }
    }
}

static void run_case_fused_hyper(TpTransport* transport, void* own_a, void* own_b,
                                  hipStream_t stream_a, hipStream_t stream_b,
                                  hipEvent_t ready_a, hipEvent_t ready_b) {
    constexpr std::uint64_t kRows = 8;
    constexpr std::uint32_t kHidden = 5120;
    constexpr std::uint32_t kHc = 6;
    const std::size_t own_count = kRows * kHidden;
    const std::size_t wide_count = kRows * kHidden * kHc;
    const std::size_t inj_count = kRows * kHc;
    std::vector<std::uint16_t> host_a(own_count);
    std::vector<std::uint16_t> host_b(own_count);
    std::vector<std::uint16_t> raw_a(wide_count);
    std::vector<std::uint16_t> raw_b(wide_count);
    std::vector<std::uint16_t> inj_a(inj_count);
    std::vector<std::uint16_t> inj_b(inj_count);
    for (std::size_t i = 0; i < own_count; ++i) {
        host_a[i] = shelf::f32_to_bf16_rne(static_cast<float>((i * 37u) % 101u) * 0.01f - 0.5f);
        host_b[i] = shelf::f32_to_bf16_rne(static_cast<float>((i * 53u) % 97u) * 0.01f - 0.5f);
    }
    for (std::size_t i = 0; i < wide_count; ++i) {
        raw_a[i] = shelf::f32_to_bf16_rne(static_cast<float>((i * 11u) % 211u) * 0.001f - 0.1f);
        raw_b[i] = shelf::f32_to_bf16_rne(static_cast<float>((i * 17u) % 197u) * 0.001f - 0.1f);
    }
    for (std::size_t i = 0; i < inj_count; ++i) {
        inj_a[i] = shelf::f32_to_bf16_rne(static_cast<float>((i * 7u) % 29u) * 0.01f);
        inj_b[i] = shelf::f32_to_bf16_rne(static_cast<float>((i * 13u) % 31u) * 0.01f);
    }

    void* raw_dev_a = nullptr;
    void* raw_dev_b = nullptr;
    void* inj_dev_a = nullptr;
    void* inj_dev_b = nullptr;
    void* out_dev_a = nullptr;
    void* out_dev_b = nullptr;
    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (!scope.ok()) { fail("hyper scope 0"); return; }
        if (hipMalloc(&raw_dev_a, wide_count * 2) != hipSuccess ||
            hipMalloc(&inj_dev_a, inj_count * 2) != hipSuccess ||
            hipMalloc(&out_dev_a, wide_count * 2) != hipSuccess) {
            fail("hyper malloc 0");
            return;
        }
        if (hipMemcpy(own_a, host_a.data(), own_count * 2, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(raw_dev_a, raw_a.data(), wide_count * 2, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(inj_dev_a, inj_a.data(), inj_count * 2, hipMemcpyHostToDevice) != hipSuccess) {
            fail("hyper upload 0");
            return;
        }
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (!scope.ok()) { fail("hyper scope 1"); return; }
        if (hipMalloc(&raw_dev_b, wide_count * 2) != hipSuccess ||
            hipMalloc(&inj_dev_b, inj_count * 2) != hipSuccess ||
            hipMalloc(&out_dev_b, wide_count * 2) != hipSuccess) {
            fail("hyper malloc 1");
            return;
        }
        if (hipMemcpy(own_b, host_b.data(), own_count * 2, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(raw_dev_b, raw_b.data(), wide_count * 2, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(inj_dev_b, inj_b.data(), inj_count * 2, hipMemcpyHostToDevice) != hipSuccess) {
            fail("hyper upload 1");
            return;
        }
    }

    TpSumTarget target_a;
    target_a.ptr = own_a;
    target_a.rows = kRows;
    target_a.row_stride = kHidden;
    target_a.feature_count = kHidden;
    target_a.dtype = ValueDType::BF16;
    target_a.fusion = shelf::runtime::TpCombineFusion::HyperApply;
    target_a.residual = raw_dev_a;
    target_a.residual_row_stride = kHidden * kHc;
    target_a.injection = inj_dev_a;
    target_a.injection_row_stride = kHc;
    target_a.fusion_output = out_dev_a;
    target_a.output_row_stride = kHidden * kHc;
    target_a.hidden = kHidden;
    target_a.hc = kHc;
    TpSumTarget target_b = target_a;
    target_b.ptr = own_b;
    target_b.residual = raw_dev_b;
    target_b.injection = inj_dev_b;
    target_b.fusion_output = out_dev_b;

    Status st = transport->sum_hidden_fused(0, target_a, stream_a, false);
    check(st.ok(), "fused hyper rank0: " + st.message());
    if (!st.ok()) return;
    st = transport->sum_hidden_fused(1, target_b, stream_b, false);
    check(st.ok(), "fused hyper rank1: " + st.message());
    if (!st.ok()) return;
    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (scope.ok() && hipStreamSynchronize(stream_a) != hipSuccess) fail("hyper sync a");
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (scope.ok() && hipStreamSynchronize(stream_b) != hipSuccess) fail("hyper sync b");
    }

    std::vector<std::uint16_t> got_a(wide_count);
    std::vector<std::uint16_t> got_b(wide_count);
    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (scope.ok() && hipMemcpy(got_a.data(), out_dev_a, wide_count * 2,
                                    hipMemcpyDeviceToHost) != hipSuccess)
            fail("hyper download a");
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (scope.ok() && hipMemcpy(got_b.data(), out_dev_b, wide_count * 2,
                                    hipMemcpyDeviceToHost) != hipSuccess)
            fail("hyper download b");
    }

    for (std::uint64_t row = 0; row < kRows; ++row) {
        for (std::uint32_t h = 0; h < kHc; ++h) {
            const float gate = bf16_to_f32_host(inj_a[row * kHc + h]);
            const float gate_b = bf16_to_f32_host(inj_b[row * kHc + h]);
            for (std::uint32_t col = 0; col < kHidden; ++col) {
                const std::uint64_t own_off = row * kHidden + col;
                const std::uint64_t wide_off = row * std::uint64_t(kHidden) * kHc +
                                               std::uint64_t(h) * kHidden + col;
                const float sum = bf16_to_f32_host(host_a[own_off]) +
                                  bf16_to_f32_host(host_b[own_off]);
                const std::uint16_t want_a =
                    shelf::f32_to_bf16_rne(bf16_to_f32_host(raw_a[wide_off]) + sum * gate);
                const std::uint16_t want_b =
                    shelf::f32_to_bf16_rne(bf16_to_f32_host(raw_b[wide_off]) + sum * gate_b);
                if (got_a[wide_off] != want_a || got_b[wide_off] != want_b) {
                    std::printf("hyper mismatch idx=%llu got_a=%u want_a=%u got_b=%u "
                                "want_b=%u raw_a=%.6f raw_b=%.6f sum=%.6f gate=%.6f\n",
                                static_cast<unsigned long long>(wide_off), got_a[wide_off],
                                want_a, got_b[wide_off], want_b,
                                bf16_to_f32_host(raw_a[wide_off]),
                                bf16_to_f32_host(raw_b[wide_off]), sum, gate);
                    fail("fused hyper mismatch at " + std::to_string(wide_off));
                    goto done;
                }
            }
        }
    }
    {
        void* out_cpu_a = nullptr;
        void* out_cpu_b = nullptr;
        {
            auto scope = shelf::gpu::ScopedDevice::create(0);
            if (!scope.ok() || hipMalloc(&out_cpu_a, wide_count * 2) != hipSuccess) {
                fail("hyper cpu malloc 0");
                goto done;
            }
        }
        {
            auto scope = shelf::gpu::ScopedDevice::create(1);
            if (!scope.ok() || hipMalloc(&out_cpu_b, wide_count * 2) != hipSuccess) {
                fail("hyper cpu malloc 1");
                goto done;
            }
        }
        TpSumTarget cpu_a = target_a;
        cpu_a.fusion_output = out_cpu_a;
        TpSumTarget cpu_b = target_b;
        cpu_b.fusion_output = out_cpu_b;
        std::vector<TpSumTarget> targets{cpu_a, cpu_b};
        std::vector<hipStream_t> streams{stream_a, stream_b};
        std::vector<hipEvent_t> ready{ready_a, ready_b};
        TpSumInvocation invocation{targets, streams, ready};
        if (hipEventRecord(ready_a, stream_a) != hipSuccess ||
            hipEventRecord(ready_b, stream_b) != hipSuccess) {
            fail("hyper cpu record");
            goto done;
        }
        st = transport->sum_hidden(invocation);
        check(st.ok(), "hyper cpu sum: " + st.message());
        if (!st.ok()) goto done;
        {
            auto scope = shelf::gpu::ScopedDevice::create(0);
            if (scope.ok() && hipStreamSynchronize(stream_a) != hipSuccess)
                fail("hyper cpu sync a");
        }
        {
            auto scope = shelf::gpu::ScopedDevice::create(1);
            if (scope.ok() && hipStreamSynchronize(stream_b) != hipSuccess)
                fail("hyper cpu sync b");
        }
        std::vector<std::uint16_t> cpu_got_a(wide_count);
        std::vector<std::uint16_t> cpu_got_b(wide_count);
        {
            auto scope = shelf::gpu::ScopedDevice::create(0);
            if (scope.ok() && hipMemcpy(cpu_got_a.data(), out_cpu_a, wide_count * 2,
                                        hipMemcpyDeviceToHost) != hipSuccess)
                fail("hyper cpu download a");
        }
        {
            auto scope = shelf::gpu::ScopedDevice::create(1);
            if (scope.ok() && hipMemcpy(cpu_got_b.data(), out_cpu_b, wide_count * 2,
                                        hipMemcpyDeviceToHost) != hipSuccess)
                fail("hyper cpu download b");
        }
        for (std::size_t i = 0; i < wide_count; ++i) {
            if (cpu_got_a[i] != got_a[i] || cpu_got_b[i] != got_b[i]) {
                std::printf("hyper cpu/fused mismatch idx=%zu fused_a=%u cpu_a=%u "
                            "fused_b=%u cpu_b=%u\n",
                            i, got_a[i], cpu_got_a[i], got_b[i], cpu_got_b[i]);
                fail("fused hyper differs from cpu hyper");
                break;
            }
        }
        {
            auto scope = shelf::gpu::ScopedDevice::create(0);
            if (scope.ok()) shelf::gpu::discard_cleanup_result(hipFree(out_cpu_a));
        }
        {
            auto scope = shelf::gpu::ScopedDevice::create(1);
            if (scope.ok()) shelf::gpu::discard_cleanup_result(hipFree(out_cpu_b));
        }
    }

done:
    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (scope.ok()) {
            shelf::gpu::discard_cleanup_result(hipFree(raw_dev_a));
            shelf::gpu::discard_cleanup_result(hipFree(inj_dev_a));
            shelf::gpu::discard_cleanup_result(hipFree(out_dev_a));
        }
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (scope.ok()) {
            shelf::gpu::discard_cleanup_result(hipFree(raw_dev_b));
            shelf::gpu::discard_cleanup_result(hipFree(inj_dev_b));
            shelf::gpu::discard_cleanup_result(hipFree(out_dev_b));
        }
    }
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
    if (peer_usable) {
        const Status reserve_status = transport->reserve(kMaxBytes);
        check(reserve_status.ok(), "reserve tp inbox: " + reserve_status.message());
    }
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
        if (transport->fused_barrier_ready()) {
            for (std::uint64_t rows : {1ull, 64ull, 1024ull, 2048ull}) {
                run_case_fused(transport, buffer_a, stream_a, buffer_b, stream_b, rows,
                               kFeatures);
            }
            run_case_fused_hyper(transport, buffer_a, buffer_b, stream_a, stream_b,
                                 ready_a, ready_b);
        }
    }

    HostMediatedTpTransport host_transport(devices);
    check(host_transport.reserve(kMaxBytes).ok(), "reserve host transport staging");
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
