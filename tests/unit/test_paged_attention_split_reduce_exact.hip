#include <shelf/kernels/optimized/attention/paged_attention.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* label) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", label);
    }
}

constexpr uint32_t kHeadLimit = shelf::kernel::kPagedAttentionHeadLimit;

uint32_t rng_state = 0x1234567u;

float next_float() {
    rng_state = rng_state * 1664525u + 1013904223u;
    const uint32_t u = rng_state >> 8;
    return (static_cast<float>(u) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
}

void set_qh(uint32_t qh) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%u", qh);
    (void)setenv("SHELF_PAGED_ATTENTION_SPLIT_REDUCE_QH", buf, 1);
}

struct Case {
    uint32_t rows;
    uint32_t kv_heads;
    uint32_t q_per_kv;
    uint32_t head_dim;
    uint32_t splits;
    bool extreme;
};

void run_case(const Case& c, std::vector<float>& partials_host, std::vector<uint32_t>& slots_host,
              float* partials_dev, uint32_t* slots_dev, uint32_t* positions_dev, float* out_a,
              float* out_b) {
    const uint32_t q_heads = c.kv_heads * c.q_per_kv;
    const uint32_t output_row_stride = q_heads * c.head_dim;
    const size_t per_row_kv = static_cast<size_t>(c.splits) * c.q_per_kv * (2u + kHeadLimit);
    const size_t partials_count = static_cast<size_t>(c.rows) * c.kv_heads * per_row_kv;
    partials_host.assign(partials_count, 0.0f);

    rng_state = 0x9e3779b9u + c.rows * 131u + c.splits * 17u + c.head_dim;
    for (uint32_t row = 0u; row < c.rows; ++row) {
        for (uint32_t kv = 0u; kv < c.kv_heads; ++kv) {
            const size_t base = (static_cast<size_t>(row) * c.kv_heads + kv) * per_row_kv;
            for (uint32_t s = 0u; s < c.splits; ++s) {
                const size_t sb = base + static_cast<size_t>(s) * c.q_per_kv * (2u + kHeadLimit);
                for (uint32_t qh = 0u; qh < c.q_per_kv; ++qh) {
                    const float lo = c.extreme ? -400.0f - static_cast<float>(s) * 64.0f : -30.0f;
                    const float hi = c.extreme ? -395.0f : 30.0f;
                    partials_host[sb + qh] = lo + next_float() * (hi - lo);
                    partials_host[sb + c.q_per_kv + qh] = 0.25f + next_float() * 0.5f;
                }
                for (uint32_t qh = 0u; qh < c.q_per_kv; ++qh) {
                    for (uint32_t d = 0u; d < kHeadLimit; ++d) {
                        partials_host[sb + 2u * c.q_per_kv + static_cast<size_t>(qh) * kHeadLimit + d] =
                            next_float() * 4.0f;
                    }
                }
            }
        }
    }
    slots_host.assign(c.rows, 0u);
    (void)hipMemcpy(partials_dev, partials_host.data(), partials_count * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(slots_dev, slots_host.data(), c.rows * sizeof(uint32_t), hipMemcpyHostToDevice);

    shelf::kernel::PagedAttentionCommonArgs args{};
    args.output_dtype = shelf::kernel::PagedAttentionOutputDType::F32;
    args.output_row_stride = output_row_stride;
    args.row_sequence_slots = slots_dev;
    args.row_positions = positions_dev;
    args.rows = c.rows;
    args.q_heads = q_heads;
    args.kv_heads = c.kv_heads;
    args.head_dim = c.head_dim;
    args.max_sequences = 1u;

    std::vector<float> host_a(static_cast<size_t>(c.rows) * output_row_stride, 0.0f);
    std::vector<float> host_b(host_a.size(), 0.0f);
    (void)hipMemcpy(out_a, host_a.data(), host_a.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(out_b, host_b.data(), host_b.size() * sizeof(float), hipMemcpyHostToDevice);

    char label[160];
    set_qh(c.q_per_kv);
    args.output = out_a;
    hipError_t err = shelf::kernel::launch_attention_paged_bf16_split_reduce(args, partials_dev,
                                                                        c.splits, nullptr);
    if (err != hipSuccess) {
        std::printf("FAIL: launch incumbent rows=%u splits=%u\n", c.rows, c.splits);
        ++failed;
        return;
    }
    std::snprintf(label, sizeof(label), "incumbent vs qh=1 rows=%u splits=%u hd=%u ext=%d",
                  c.rows, c.splits, c.head_dim, c.extreme ? 1 : 0);
    for (uint32_t qh = 1u; qh < c.q_per_kv; ++qh) {
        if (c.q_per_kv % qh != 0u) continue;
        set_qh(qh);
        args.output = out_b;
        err = shelf::kernel::launch_attention_paged_bf16_split_reduce(args, partials_dev, c.splits,
                                                                   nullptr);
        if (err != hipSuccess) {
            std::snprintf(label, sizeof(label), "launch qh=%u rows=%u splits=%u", qh, c.rows,
                          c.splits);
            check(false, label);
            continue;
        }
        (void)hipMemcpy(host_a.data(), out_a, host_a.size() * sizeof(float),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(host_b.data(), out_b, host_b.size() * sizeof(float),
                        hipMemcpyDeviceToHost);
        bool same = true;
        for (size_t i = 0; i < host_a.size(); ++i) {
            if (std::memcmp(&host_a[i], &host_b[i], sizeof(float)) != 0) {
                same = false;
                break;
            }
        }
        std::snprintf(label, sizeof(label), "incumbent vs qh=%u rows=%u splits=%u hd=%u ext=%d",
                      qh, c.rows, c.splits, c.head_dim, c.extreme ? 1 : 0);
        check(same, label);
    }
}

void run_zero_sum_case(uint32_t rows, uint32_t kv_heads, uint32_t q_per_kv, uint32_t head_dim,
                       uint32_t splits, std::vector<float>& partials_host,
                       std::vector<uint32_t>& slots_host, float* partials_dev,
                       uint32_t* slots_dev, uint32_t* positions_dev, float* out_a) {
    const uint32_t q_heads = kv_heads * q_per_kv;
    const size_t output_row_stride = static_cast<size_t>(q_heads) * head_dim;
    const size_t per_row_kv = static_cast<size_t>(splits) * q_per_kv * (2u + kHeadLimit);
    const size_t partials_count = static_cast<size_t>(rows) * kv_heads * per_row_kv;
    partials_host.assign(partials_count, 0.0f);
    for (uint32_t row = 0u; row < rows; ++row) {
        for (uint32_t kv = 0u; kv < kv_heads; ++kv) {
            const size_t base = (static_cast<size_t>(row) * kv_heads + kv) * per_row_kv;
            for (uint32_t s = 0u; s < splits; ++s) {
                const size_t sb = base + static_cast<size_t>(s) * q_per_kv * (2u + kHeadLimit);
                for (uint32_t qh = 0u; qh < q_per_kv; ++qh) {
                    partials_host[sb + qh] = -3.402823466e+38f;
                    partials_host[sb + q_per_kv + qh] = 0.0f;
                }
                for (uint32_t qh = 0u; qh < q_per_kv; ++qh) {
                    for (uint32_t d = 0u; d < kHeadLimit; ++d) {
                        partials_host[sb + 2u * q_per_kv + static_cast<size_t>(qh) * kHeadLimit + d] =
                            1.0f;
                    }
                }
            }
        }
    }
    slots_host.assign(rows, 0u);
    (void)hipMemcpy(partials_dev, partials_host.data(), partials_count * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(slots_dev, slots_host.data(), rows * sizeof(uint32_t), hipMemcpyHostToDevice);

    const float sentinel = 12345.0f;
    std::vector<float> host(static_cast<size_t>(rows) * output_row_stride, sentinel);
    (void)hipMemcpy(out_a, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice);

    shelf::kernel::PagedAttentionCommonArgs args{};
    args.output_dtype = shelf::kernel::PagedAttentionOutputDType::F32;
    args.output_row_stride = static_cast<uint32_t>(output_row_stride);
    args.output = out_a;
    args.row_sequence_slots = slots_dev;
    args.row_positions = positions_dev;
    args.rows = rows;
    args.q_heads = q_heads;
    args.kv_heads = kv_heads;
    args.head_dim = head_dim;
    args.max_sequences = 1u;

    set_qh(q_per_kv);
    hipError_t err = shelf::kernel::launch_attention_paged_bf16_split_reduce(args, partials_dev, splits,
                                                                        nullptr);
    char label[160];
    std::snprintf(label, sizeof(label), "zero-sum launch rows=%u splits=%u hd=%u", rows, splits,
                  head_dim);
    check(err == hipSuccess, label);
    if (err != hipSuccess) return;
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(host.data(), out_a, host.size() * sizeof(float), hipMemcpyDeviceToHost);
    bool untouched = true;
    bool finite = true;
    for (size_t i = 0; i < host.size(); ++i) {
        if (std::memcmp(&host[i], &sentinel, sizeof(float)) != 0) untouched = false;
        if (!std::isfinite(host[i])) finite = false;
    }
    std::snprintf(label, sizeof(label), "zero-sum keeps output rows=%u splits=%u hd=%u", rows,
                  splits, head_dim);
    check(untouched, label);
    std::snprintf(label, sizeof(label), "zero-sum finite rows=%u splits=%u hd=%u", rows, splits,
                  head_dim);
    check(finite, label);
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    const uint32_t rows_max = 8u;
    const uint32_t kv_heads = 4u;
    const uint32_t q_per_kv = 6u;
    const uint32_t splits_max = 16u;
    const uint32_t q_heads = kv_heads * q_per_kv;
    const size_t partials_max =
        static_cast<size_t>(rows_max) * kv_heads * splits_max * q_per_kv * (2u + kHeadLimit);
    const size_t out_max = static_cast<size_t>(rows_max) * q_heads * kHeadLimit;

    float* partials_dev = nullptr;
    uint32_t* slots_dev = nullptr;
    uint32_t* positions_dev = nullptr;
    float* out_a = nullptr;
    float* out_b = nullptr;
    if (hipMalloc(&partials_dev, partials_max * sizeof(float)) != hipSuccess ||
        hipMalloc(&slots_dev, rows_max * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&positions_dev, rows_max * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&out_a, out_max * sizeof(float)) != hipSuccess ||
        hipMalloc(&out_b, out_max * sizeof(float)) != hipSuccess) {
        std::printf("FAIL: hipMalloc\n");
        return 1;
    }

    std::vector<float> partials_host;
    std::vector<uint32_t> slots_host;
    const uint32_t rows_list[] = {1u, 2u, 4u, 8u};
    const uint32_t splits_list[] = {2u, 4u, 8u, 16u};
    const uint32_t head_list[] = {128u, 256u};
    for (uint32_t hd : head_list) {
        for (uint32_t rows : rows_list) {
            for (uint32_t splits : splits_list) {
                for (uint32_t ext = 0u; ext < 2u; ++ext) {
                    Case c{rows, kv_heads, q_per_kv, hd, splits, ext != 0u};
                    run_case(c, partials_host, slots_host, partials_dev, slots_dev, positions_dev,
                             out_a, out_b);
                }
            }
        }
    }

    for (uint32_t hd : head_list) {
        for (uint32_t rows : {1u, 4u}) {
            for (uint32_t splits : {2u, 16u}) {
                run_zero_sum_case(rows, kv_heads, q_per_kv, hd, splits, partials_host, slots_host,
                                  partials_dev, slots_dev, positions_dev, out_a);
            }
        }
    }

    (void)hipFree(partials_dev);
    (void)hipFree(slots_dev);
    (void)hipFree(positions_dev);
    (void)hipFree(out_a);
    (void)hipFree(out_b);
    std::printf("test_paged_attention_split_reduce_exact: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
