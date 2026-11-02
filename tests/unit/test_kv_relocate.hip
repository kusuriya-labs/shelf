#include <shelf/core/gpu/cleanup.h>
#include <shelf/kernels/optimized/kv_relocate.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kLayers = 2u;
constexpr uint32_t kKvHeads = 2u;
constexpr uint32_t kHeadDim = 128u;
constexpr uint32_t kPageTokens = 4u;
constexpr uint32_t kNumPages = 8u;
constexpr uint32_t kBtStride = 8u;
constexpr uint32_t kMaxSeq = 2u;
constexpr uint32_t kSlot = 0u;
constexpr uint32_t kEpt = kKvHeads * kHeadDim;
constexpr uint32_t kEpp = kPageTokens * kEpt;
constexpr uint32_t kEpl = kNumPages * kEpp;
constexpr uint32_t kScaleElems = kLayers * kNumPages * kPageTokens * kKvHeads;

int g_fail = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", name);
    }
}

const uint32_t kBlockTable[kBtStride] = {3, 1, 6, 0, 2, 7, 4, 5};
const uint32_t kSrcPositions[] = {10u, 12u, 14u};
const uint32_t kDstPositions[] = {10u, 11u, 12u};
constexpr uint32_t kRows = 3u;

size_t token_elem(uint32_t layer, uint32_t pos, uint32_t head) {
    const uint32_t page = kBlockTable[pos / kPageTokens];
    const uint32_t off = pos % kPageTokens;
    return static_cast<size_t>(layer) * kEpl + static_cast<size_t>(page) * kEpp +
           static_cast<size_t>(off) * kEpt + static_cast<size_t>(head) * kHeadDim;
}

size_t scale_elem(uint32_t layer, uint32_t pos, uint32_t head) {
    const uint32_t page = kBlockTable[pos / kPageTokens];
    const uint32_t off = pos % kPageTokens;
    return ((static_cast<size_t>(layer) * kNumPages + page) * kPageTokens + off) *
               kKvHeads +
           head;
}

void fill_pattern(std::vector<uint8_t>& buf, uint32_t salt) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<uint8_t>((i * 131u + salt + (i >> 8) * 17u) & 0xFFu);
    }
}

void run_case(uint32_t elem_bytes, bool fp8, const char* label) {
    const size_t pool_elems = static_cast<size_t>(kLayers) * kEpl;
    const size_t pool_bytes = pool_elems * elem_bytes;

    std::vector<uint8_t> before_k(pool_bytes);
    std::vector<uint8_t> before_v(pool_bytes);
    fill_pattern(before_k, 0x3Cu);
    fill_pattern(before_v, 0xC3u);

    std::vector<uint8_t> expected_k = before_k;
    for (uint32_t i = 0; i < kRows; ++i) {
        const size_t token_bytes = static_cast<size_t>(kHeadDim) * elem_bytes;
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            for (uint32_t head = 0; head < kKvHeads; ++head) {
                const size_t s = token_elem(layer, kSrcPositions[i], head) * elem_bytes;
                const size_t d = token_elem(layer, kDstPositions[i], head) * elem_bytes;
                std::memcpy(expected_k.data() + d, before_k.data() + s, token_bytes);
            }
        }
    }
    const std::vector<uint8_t> expected_v = [&] {
        std::vector<uint8_t> e = before_v;
        for (uint32_t i = 0; i < kRows; ++i) {
            const size_t token_bytes = static_cast<size_t>(kHeadDim) * elem_bytes;
            for (uint32_t layer = 0; layer < kLayers; ++layer) {
                for (uint32_t head = 0; head < kKvHeads; ++head) {
                    const size_t s = token_elem(layer, kSrcPositions[i], head) * elem_bytes;
                    const size_t d = token_elem(layer, kDstPositions[i], head) * elem_bytes;
                    std::memcpy(e.data() + d, before_v.data() + s, token_bytes);
                }
            }
        }
        return e;
    }();

    std::vector<uint8_t> before_ks(kScaleElems * 4u);
    std::vector<uint8_t> before_vs(kScaleElems * 4u);
    fill_pattern(before_ks, 0x11u);
    fill_pattern(before_vs, 0xEEu);
    std::vector<uint8_t> expected_ks = before_ks;
    std::vector<uint8_t> expected_vs = before_vs;
    for (uint32_t i = 0; i < kRows; ++i) {
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            for (uint32_t layer = 0; layer < kLayers; ++layer) {
                const size_t s = scale_elem(layer, kSrcPositions[i], head) * 4u;
                const size_t d = scale_elem(layer, kDstPositions[i], head) * 4u;
                std::memcpy(expected_ks.data() + d, before_ks.data() + s, 4u);
                std::memcpy(expected_vs.data() + d, before_vs.data() + s, 4u);
            }
        }
    }

    uint8_t* k_pool = nullptr;
    uint8_t* v_pool = nullptr;
    float* k_scale = nullptr;
    float* v_scale = nullptr;
    uint32_t* src_dev = nullptr;
    uint32_t* dst_dev = nullptr;
    uint32_t* bt_dev = nullptr;
    bool ok = true;
    ok = ok && hipMalloc(&k_pool, pool_bytes) == hipSuccess;
    ok = ok && hipMalloc(&v_pool, pool_bytes) == hipSuccess;
    ok = ok && hipMalloc(&k_scale, kScaleElems * 4u) == hipSuccess;
    ok = ok && hipMalloc(&v_scale, kScaleElems * 4u) == hipSuccess;
    ok = ok && hipMalloc(&src_dev, kRows * 4u) == hipSuccess;
    ok = ok && hipMalloc(&dst_dev, kRows * 4u) == hipSuccess;
    ok = ok && hipMalloc(&bt_dev, kBtStride * 4u) == hipSuccess;
    if (!ok) {
        std::printf("FAIL: %s alloc\n", label);
        ++g_fail;
        return;
    }
    ok = ok && hipMemcpy(k_pool, before_k.data(), pool_bytes, hipMemcpyHostToDevice) ==
                   hipSuccess;
    ok = ok && hipMemcpy(v_pool, before_v.data(), pool_bytes, hipMemcpyHostToDevice) ==
                   hipSuccess;
    ok = ok && hipMemcpy(k_scale, before_ks.data(), kScaleElems * 4u,
                         hipMemcpyHostToDevice) == hipSuccess;
    ok = ok && hipMemcpy(v_scale, before_vs.data(), kScaleElems * 4u,
                         hipMemcpyHostToDevice) == hipSuccess;
    ok = ok && hipMemcpy(src_dev, kSrcPositions, kRows * 4u, hipMemcpyHostToDevice) ==
                   hipSuccess;
    ok = ok && hipMemcpy(dst_dev, kDstPositions, kRows * 4u, hipMemcpyHostToDevice) ==
                   hipSuccess;
    ok = ok && hipMemcpy(bt_dev, kBlockTable, kBtStride * 4u, hipMemcpyHostToDevice) ==
                   hipSuccess;
    check(ok, (std::string(label) + " upload").c_str());
    if (!ok) return;

    ::ps::kernel::KvRelocateArgs common;
    common.src_positions = src_dev;
    common.dst_positions = dst_dev;
    common.rows = kRows;
    common.slot = kSlot;
    common.block_tables = bt_dev;
    common.block_table_stride = kBtStride;
    common.max_sequences = kMaxSeq;
    common.page_tokens = kPageTokens;
    common.num_pages = kNumPages;
    common.num_attention_layers = kLayers;
    common.kv_heads = kKvHeads;
    common.head_dim = kHeadDim;
    common.elems_per_token = kEpt;
    common.elems_per_page = kEpp;
    common.elems_per_layer = kEpl;

    hipError_t h = hipSuccess;
    if (fp8) {
        ::ps::kernel::KvRelocateFp8Args a;
        a.common = common;
        a.k_pool = reinterpret_cast<::ps::kernel::fp8e4m3_storage_t*>(k_pool);
        a.v_pool = reinterpret_cast<::ps::kernel::fp8e4m3_storage_t*>(v_pool);
        a.k_scale_pool = k_scale;
        a.v_scale_pool = v_scale;
        h = ::ps::kernel::launch_kv_relocate_fp8_e4m3(a, 0);
    } else {
        ::ps::kernel::KvRelocateBf16Args a;
        a.common = common;
        a.k_pool = reinterpret_cast<::ps::bf16_t*>(k_pool);
        a.v_pool = reinterpret_cast<::ps::bf16_t*>(v_pool);
        h = ::ps::kernel::launch_kv_relocate_bf16(a, 0);
    }
    ok = h == hipSuccess && hipStreamSynchronize(0) == hipSuccess;
    check(ok, (std::string(label) + " launch").c_str());
    if (!ok) {
        std::printf("  hip error %s\n", hipGetErrorString(h));
        return;
    }

    std::vector<uint8_t> got_k(pool_bytes);
    std::vector<uint8_t> got_v(pool_bytes);
    std::vector<uint8_t> got_ks(kScaleElems * 4u);
    std::vector<uint8_t> got_vs(kScaleElems * 4u);
    ok = hipMemcpy(got_k.data(), k_pool, pool_bytes, hipMemcpyDeviceToHost) == hipSuccess;
    ok = ok && hipMemcpy(got_v.data(), v_pool, pool_bytes,
                         hipMemcpyDeviceToHost) == hipSuccess;
    ok = ok && hipMemcpy(got_ks.data(), k_scale, kScaleElems * 4u,
                         hipMemcpyDeviceToHost) == hipSuccess;
    ok = ok && hipMemcpy(got_vs.data(), v_scale, kScaleElems * 4u,
                         hipMemcpyDeviceToHost) == hipSuccess;
    check(ok, (std::string(label) + " download").c_str());
    if (!ok) return;

    {
        const bool same =
            std::memcmp(got_k.data(), expected_k.data(), pool_bytes) == 0;
        if (!same) {
            for (size_t i = 0; i < pool_bytes; ++i) {
                if (got_k[i] != expected_k[i]) {
                    std::printf("  [%s/k] first bad byte=%zu got=%02x want=%02x\n",
                                label, i, got_k[i], expected_k[i]);
                    break;
                }
            }
        }
        check(same, (std::string(label) + " k tokens land at dst (alias-safe)").c_str());
    }
    {
        const bool same =
            std::memcmp(got_v.data(), expected_v.data(), pool_bytes) == 0;
        if (!same) {
            for (size_t i = 0; i < pool_bytes; ++i) {
                if (got_v[i] != expected_v[i]) {
                    std::printf("  [%s/v] first bad byte=%zu got=%02x want=%02x\n",
                                label, i, got_v[i], expected_v[i]);
                    break;
                }
            }
        }
        check(same, (std::string(label) + " v tokens land at dst (alias-safe)").c_str());
    }
    if (fp8) {
        check(std::memcmp(got_ks.data(), expected_ks.data(), kScaleElems * 4u) == 0,
              (std::string(label) + " k scales move with tokens").c_str());
        check(std::memcmp(got_vs.data(), expected_vs.data(), kScaleElems * 4u) == 0,
              (std::string(label) + " v scales move with tokens").c_str());
    }

    ps::gpu::discard_cleanup_result(hipFree(k_pool));
    ps::gpu::discard_cleanup_result(hipFree(v_pool));
    ps::gpu::discard_cleanup_result(hipFree(k_scale));
    ps::gpu::discard_cleanup_result(hipFree(v_scale));
    ps::gpu::discard_cleanup_result(hipFree(src_dev));
    ps::gpu::discard_cleanup_result(hipFree(dst_dev));
    ps::gpu::discard_cleanup_result(hipFree(bt_dev));
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    run_case(1u, true, "fp8");
    run_case(2u, false, "bf16");
    std::printf("test_kv_relocate: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
