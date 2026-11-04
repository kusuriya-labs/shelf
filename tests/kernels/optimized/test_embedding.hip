#include <shelf/kernels/optimized/embedding.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/program/program.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using V = ::shelf::runtime::DeviceValueBinding;
using P = ::shelf::runtime::DeviceProgramView;
using B = ::shelf::runtime::DispatchBinding;
using W = ::shelf::runtime::WeightSlot;

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

uint32_t h32(uint64_t x) {    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

struct Buf {
    void* dev = nullptr;
    size_t bytes = 0;
    ~Buf() {
        if (dev) shelf::gpu::discard_cleanup_result(hipFree(dev));
    }
    bool alloc(size_t b) {
        bytes = b;
        return hipMalloc(&dev, b) == hipSuccess;
    }
};

struct Case {
    const char* name;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t rows;
    uint32_t out_stride;
    std::vector<int32_t> tokens;
    bool invalid;
    bool check_weight;
};

size_t cmp_bits(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    size_t mm = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) ++mm;
    return (a.size() != b.size()) ? (mm + 1) : mm;
}

std::vector<uint16_t> read_u16(const Buf& b, size_t n) {
    std::vector<uint16_t> v(n);
    if (hipMemcpy(v.data(), b.dev, n * 2, hipMemcpyDeviceToHost) != hipSuccess)
        return {};
    return v;
}

std::vector<uint16_t> gather_u16(const Buf& b, uint32_t rows, uint32_t f, uint32_t os) {
    std::vector<uint16_t> t(b.bytes / 2);
    if (hipMemcpy(t.data(), b.dev, b.bytes, hipMemcpyDeviceToHost) != hipSuccess)
        return {};
    std::vector<uint16_t> out((size_t)rows * f);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t h = 0; h < f; ++h)
            out[(size_t)r * f + h] = t[(size_t)r * os + h];
    return out;
}

bool read_bytes(const Buf& b, size_t n, std::vector<uint8_t>& out) {
    out.assign(n, 0);
    return hipMemcpy(out.data(), b.dev, n, hipMemcpyDeviceToHost) == hipSuccess;
}

bool run_correctness(const W& ws, const Buf& toks, const std::vector<int32_t>& htok,
                     const Buf& out, uint32_t rows, uint32_t hidden, uint32_t out_stride,
                     P& p, V* vals_dev, Buf& vals_buf, ::shelf::runtime::DeviceBatchContext* bc_dev,
                     W* wslot_dev, float* scratch, uint32_t* err) {
    W w = ws;
    if (hipMemcpy(wslot_dev, &w, sizeof(W), hipMemcpyHostToDevice) != hipSuccess) return false;
    V vals[1];
    vals[0] = {};
    vals[0].logical_value = 1;
    vals[0].storage = ::shelf::runtime::ValueStorage::WORKSPACE;
    vals[0].dtype = ::shelf::runtime::ValueDType::BF16;
    vals[0].row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
    vals[0].slot = 0;
    vals[0].feature_count = hidden;
    vals[0].row_stride = out_stride;
    vals[0].offset = (uint64_t)(uintptr_t)out.dev;
    vals[0].bytes = 0;
    if (vals_buf.bytes < sizeof(V)) {
        if (!vals_buf.alloc(sizeof(V))) return false;
    }
    if (hipMemcpy(vals_dev, vals, sizeof(V), hipMemcpyHostToDevice) != hipSuccess) return false;
    p.values = (const V*)vals_dev;
    p.value_count = 1;
    p.workspace_base = 0;
    p.weight_table = wslot_dev;
    p.weight_slot_count = 1;
    B b{};
    b.kernel_id = ::shelf::runtime::KernelId::EMBEDDING_LOOKUP;
    b.input_slots[0] = 0;
    b.output_slots[0] = 1;
    b.input_count = 1;
    b.output_count = 1;
    b.weight_index = 0;
    (void)toks;
    (void)htok;
    shelf::gpu::discard_cleanup_result(hipMemsetAsync(err, 0, sizeof(uint32_t), nullptr));
    ::shelf::kernel::ModelDispatchStateView sv{};
    auto st = shelf::kernel::launch_model_dispatch_correctness(
        p, b, bc_dev, sv, scratch, err, nullptr);
    if (!st.ok()) return false;
    uint32_t e = 0;
    if (hipMemcpy(&e, err, 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
    return e == 0u;
}

bool build_batchctx(::shelf::runtime::DeviceBatchContext* bc_dev, uint32_t rows,
                    const void* tokens_dev) {
    ::shelf::runtime::DeviceBatchContext h{};
    h.actual_rows = rows;
    h.num_requests = 0;
    h.token_ids = (const int32_t*)tokens_dev;
    return hipMemcpy(bc_dev, &h, sizeof(h), hipMemcpyHostToDevice) == hipSuccess;
}

bool run_case(const Case& c, size_t* mm_out) {
    const uint32_t hidden = c.hidden;
    const uint32_t rows = c.rows;
    const uint32_t os = c.out_stride;
    const bool corr_valid = !c.invalid;

    Buf toks, table, out, bc, wslot, scratch, err, vals_buf;
    if (!toks.alloc((size_t)rows * 4) ||
        !out.alloc((size_t)rows * os * 2) ||
        !bc.alloc(sizeof(::shelf::runtime::DeviceBatchContext)) ||
        !wslot.alloc(sizeof(W)) ||
        !scratch.alloc((size_t)256 * 4096 * 4) ||
        !err.alloc(4) ||
        !vals_buf.alloc(sizeof(V))) {
        std::printf("  [%s] ALLOC FAIL\n", c.name);
        *mm_out = 1;
        return false;
    }
    if (hipMemcpy(toks.dev, c.tokens.data(), (size_t)rows * 4, hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("  [%s] COPY FAIL\n", c.name);
        *mm_out = 1;
        return false;
    }
    if (!build_batchctx((::shelf::runtime::DeviceBatchContext*)bc.dev, rows, toks.dev)) {
        std::printf("  [%s] COPY FAIL\n", c.name);
        *mm_out = 1;
        return false;
    }
    shelf::gpu::discard_cleanup_result(hipMemsetAsync(out.dev, 0xAB, out.bytes, nullptr));
    shelf::gpu::discard_cleanup_result(hipMemsetAsync(err.dev, 0, 4, nullptr));

    // ---- weight data + CPU reference (packed: rows*hidden) ----
    std::vector<uint16_t> ref((size_t)rows * hidden);
    W ws{};
    ws.rows = c.vocab;
    ws.cols = hidden;
    std::vector<uint8_t> table_bytes;
    table_bytes.assign((size_t)c.vocab * hidden * 2, 0);
    for (uint32_t t = 0; t < c.vocab; ++t)
        for (uint32_t h = 0; h < hidden; ++h) {
            float v =
                (static_cast<int32_t>(h32((uint64_t)t * 2654435761u ^ (uint64_t)h) % 1001) -
                 500) /
                500.0f;
            uint16_t bits = bf16_bits(v);
            table_bytes[(size_t)t * hidden * 2 + h * 2] = bits & 0xFFu;
            table_bytes[(size_t)t * hidden * 2 + h * 2 + 1] = (bits >> 8) & 0xFFu;
        }
    size_t tbytes = table_bytes.size();
    if (!table.alloc(tbytes) ||
        hipMemcpy(table.dev, table_bytes.data(), tbytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
        std::printf("  [%s] TBL FAIL\n", c.name);
        *mm_out = 1;
        return false;
    }
    for (uint32_t m = 0; m < rows; ++m)
        for (uint32_t h = 0; h < hidden; ++h)
            ref[(size_t)m * hidden + h] =
                (table_bytes[(size_t)c.tokens[m] * hidden * 2 + h * 2] |
                 (table_bytes[(size_t)c.tokens[m] * hidden * 2 + h * 2 + 1] << 8));
    ws.encoding = 0;
    ws.device_ptr = table.dev;

    // ---- correctness GPU (if applicable) ----
    std::vector<uint16_t> corr;
    if (corr_valid) {
        P p{};
        if (!run_correctness(ws, toks, c.tokens, out, rows, hidden, os, p,
                             (V*)vals_buf.dev, vals_buf,
                             (::shelf::runtime::DeviceBatchContext*)bc.dev, (W*)wslot.dev,
                             (float*)scratch.dev, (uint32_t*)err.dev)) {
            std::printf("  [%s] CORR RUN FAIL\n", c.name);
            *mm_out = 1;
            return false;
        }
        corr = gather_u16(out, rows, hidden, os);
        shelf::gpu::discard_cleanup_result(hipMemsetAsync(out.dev, 0xAB, out.bytes, nullptr));
    }

    // ---- optimized GPU ----
    bool ok = ::shelf::kernel::launch_embedding_bf16(
                  (::shelf::bf16_t*)table.dev, (const int32_t*)toks.dev,
                  (::shelf::bf16_t*)out.dev, rows, c.vocab, hidden, os,
                  (uint32_t*)err.dev, nullptr) == hipSuccess;
    if (!ok) {
        std::printf("  [%s] OPT LAUNCH FAIL\n", c.name);
        *mm_out = 1;
        return false;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        *mm_out = 1;
        return false;
    }
    std::vector<uint16_t> opt = gather_u16(out, rows, hidden, os);
    uint32_t ew = 0;
    if (hipMemcpy(&ew, err.dev, 4, hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("  [%s] READ FAIL\n", c.name);
        *mm_out = 1;
        return false;
    }

    size_t mm = 0;
    const bool expect_invalid = c.invalid;
    if (expect_invalid) {
        std::vector<uint8_t> ob;
        if (!read_bytes(out, (size_t)rows * os * 2, ob)) {
            std::printf("  [%s] READ FAIL\n", c.name);
            *mm_out = 1;
            return false;
        }
        for (uint32_t m = 0; m < rows; ++m) {
            const int32_t tk = c.tokens[m];
            if (tk >= 0 && (uint32_t)tk < c.vocab) continue;
            for (uint32_t h = 0; h < hidden; ++h)
                if (ob[(size_t)m * os * 2 + h * 2] != 0xABu ||
                    ob[(size_t)m * os * 2 + h * 2 + 1] != 0xABu)
                    ++mm;
        }
        if (ew != 3u) mm += 1000;
    } else {
        mm += cmp_bits(ref, opt);
        if (corr_valid) mm += cmp_bits(corr, opt);
    }

    // ---- weight read-only check ----
    if (c.check_weight) {
        std::vector<uint8_t> tb;
        if (!read_bytes(table, table_bytes.size(), tb)) {
            std::printf("  [%s] READ FAIL\n", c.name);
            *mm_out = 1;
            return false;
        }
        for (size_t i = 0; i < tb.size(); ++i)
            if (tb[i] != table_bytes[i]) mm += 5;
    }

    *mm_out = mm;
    const bool pass = (mm == 0);
    std::printf("  [%s] v=%u h=%u rows=%u os=%u ew=%u | mm=%zu %s\n", c.name,
                c.vocab, hidden, rows, os, ew, mm, pass ? "PASS" : "FAIL");
    return true;
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) return 1;
    const Case cases[] = {
        {"bf16-map", 64, 2560, 16, 2560, {7, 1, 9, 0, 5, 3, 8, 2, 11, 4, 13, 6, 15, 10, 12, 14}, false, false},
        {"bf16-edge", 32, 2560, 4, 2560, {0, 31, 1, 30}, false, false},
        {"bf16-dup", 32, 1024, 4, 1024, {3, 3, 3, 3}, false, false},
        {"bf16-tail33", 32, 33, 4, 33, {5, 2, 9, 0}, false, false},
        {"bf16-tail25", 32, 25, 4, 25, {5, 2, 9, 0}, false, false},
        {"bf16-tail9", 32, 9, 4, 9, {5, 2, 9, 0}, false, false},
        {"bf16-tail31", 32, 31, 4, 31, {5, 2, 9, 0}, false, false},
        {"bf16-pad", 32, 2560, 8, 2576, {7, 1, 9, 0, 5, 3, 8, 2}, false, false},
        {"bf16-hidden-2048", 64, 2048, 8, 2048, {7, 1, 9, 0, 5, 3, 8, 2}, false, false},
        {"bf16-hidden-2048-pad", 64, 2048, 4, 2064, {3, 17, 22, 41}, false, false},
        {"bf16-weight", 32, 512, 4, 512, {5, 2, 9, 0}, false, true},
        {"bf16-inval-neg", 32, 128, 4, 128, {0, -1, 2, 3}, true, false},
        {"bf16-inval-upper", 32, 128, 4, 128, {0, 32, 2, 3}, true, false},
        {"bf16-actual-4b", 248320, 2560, 8, 2560, {0, 1, 100, 248319, 248318, 512, 248320 - 1, 7}, false, false},
        {"bf16-actual-27b", 248320, 5120, 8, 5120, {0, 1, 100, 248319, 248318, 512, 248320 - 1, 7}, false, false},
    };
    int fail = 0;
    for (const auto& c : cases) {
        size_t mm = 0;
        run_case(c, &mm);
        if (mm != 0) ++fail;
    }
    std::printf("%s: %d failures\n", fail == 0 ? "PASS" : "FAIL", fail);
    return fail == 0 ? 0 : 1;
}
