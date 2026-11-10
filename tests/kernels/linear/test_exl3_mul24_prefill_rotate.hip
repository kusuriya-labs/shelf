#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(x) do { const hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP error at %d: %s\n", __LINE__, hipGetErrorString(e)); return 1; } } while (0)

template <typename T>
struct Device {
    T* ptr = nullptr;
    explicit Device(size_t n) { if (hipMalloc(&ptr, n * sizeof(T)) != hipSuccess) std::abort(); }
    ~Device() { (void) hipFree(ptr); }
};

int main() {
    constexpr uint32_t rows = 256u, n = 384u, stride = 512u;
    const size_t entries = static_cast<size_t>(rows) * n;
    std::vector<int32_t> acc(entries), slots(rows);
    std::vector<float> rowscale(rows), flagged(entries);
    std::vector<uint16_t> svh(n);
    for (size_t i = 0; i < entries; ++i) {
        acc[i] = static_cast<int32_t>((i * 131u) % 2021u) - 1010;
        flagged[i] = static_cast<float>(static_cast<int32_t>((i * 37u) % 109u) - 54) * 0.03125f;
    }
    for (uint32_t i = 0; i < rows; ++i) {
        rowscale[i] = 0.001f + (i % 17u) * 0.0001f;
        slots[i] = i % 3u == 0u ? static_cast<int32_t>(i) : -1;
    }
    for (uint32_t i = 0; i < n; ++i) svh[i] = (i & 1u) ? 0x3c00u : 0xb800u;
    Device<int32_t> d_acc(entries), d_slots(rows);
    Device<float> d_rowscale(rows), d_flagged(entries);
    Device<uint16_t> d_svh(n);
    Device<uint8_t> d_base(static_cast<size_t>(rows) * stride * 4u), d_test(static_cast<size_t>(rows) * stride * 4u);
    HIP_OK(hipMemcpy(d_acc.ptr, acc.data(), entries * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_slots.ptr, slots.data(), rows * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_rowscale.ptr, rowscale.data(), rows * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_flagged.ptr, flagged.data(), entries * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_svh.ptr, svh.data(), n * sizeof(uint16_t), hipMemcpyHostToDevice));
    for (auto dtype : {shelf::kernel::Exl3OutputDType::BF16, shelf::kernel::Exl3OutputDType::F32}) {
        const size_t bytes = static_cast<size_t>(rows) * stride * (dtype == shelf::kernel::Exl3OutputDType::BF16 ? 2u : 4u);
        HIP_OK(hipMemset(d_base.ptr, 0xa5, bytes));
        HIP_OK(hipMemset(d_test.ptr, 0xa5, bytes));
        HIP_OK(shelf::kernel::launch_exl3_mul24_rotate_out_i32(d_acc.ptr, d_rowscale.ptr, d_slots.ptr, d_flagged.ptr,
            d_svh.ptr, d_base.ptr, dtype, rows, n, stride, nullptr));
        HIP_OK(shelf::kernel::launch_exl3_mul24_prefill_rotate_out_i32(d_acc.ptr, d_rowscale.ptr, d_slots.ptr, d_flagged.ptr,
            d_svh.ptr, d_test.ptr, dtype, rows, n, stride, nullptr));
        std::vector<uint8_t> base(bytes), test(bytes);
        HIP_OK(hipMemcpy(base.data(), d_base.ptr, bytes, hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(test.data(), d_test.ptr, bytes, hipMemcpyDeviceToHost));
        if (base != test) {
            std::fprintf(stderr, "prefill rotate differs for dtype %d\n", static_cast<int>(dtype));
            return 1;
        }
    }
    std::puts("prefill rotate bitwise comparison passed");
    return 0;
}
