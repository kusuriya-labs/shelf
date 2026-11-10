// Boundary checks for the EXL3 launch contracts. Every case returns before launching a
// kernel, so the test only needs the HIP runtime, not a working device workload.

#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

namespace {

char g_dummy[64];
const void* const kPtr = g_dummy;
void* const kMut = g_dummy;

int g_failures = 0;

void expect(const char* name, hipError_t got, hipError_t want) {
    if (got != want) {
        std::printf("FAIL %s: got %d want %d\n", name, static_cast<int>(got), static_cast<int>(want));
        ++g_failures;
    }
}

shelf::kernel::Exl3GemmArgs dense_args() {
    shelf::kernel::Exl3GemmArgs a;
    a.trellis = kPtr;
    a.suh = kPtr;
    a.svh = kPtr;
    a.input = kPtr;
    a.output = kMut;
    a.rows = 16u;
    a.out_features = 128u;
    a.k = 128u;
    a.input_row_stride = 128u;
    a.output_row_stride = 128u;
    a.ka = 4u;
    return a;
}

shelf::kernel::Exl3GroupedArgs grouped_args() {
    shelf::kernel::Exl3GroupedArgs a;
    a.trellis = kPtr;
    a.suh = kPtr;
    a.svh = kPtr;
    a.input = kPtr;
    a.output = kMut;
    a.out_features = 128u;
    a.k = 128u;
    a.input_row_stride = 128u;
    a.output_row_stride = 128u;
    a.ka = 4u;
    a.group_expert = reinterpret_cast<const uint32_t*>(kPtr);
    a.group_rows = reinterpret_cast<const uint32_t*>(kPtr);
    a.group_count = reinterpret_cast<const uint32_t*>(kPtr);
    a.group_capacity = 1u;
    return a;
}

}  // namespace

int main() {
    using namespace shelf::kernel;

    // dense mul1: pointers, shape, alignment, bit rate.
    {
        auto a = dense_args();
        a.trellis = nullptr;
        expect("dense null trellis", launch_gemm_exl3_w16a16(a, nullptr), hipErrorInvalidValue);
        a = dense_args();
        a.rows = 0u;
        expect("dense rows 0", launch_gemm_exl3_w16a16(a, nullptr), hipErrorInvalidValue);
        a = dense_args();
        a.out_features = 0u;
        expect("dense n 0", launch_gemm_exl3_w16a16(a, nullptr), hipErrorInvalidValue);
        a = dense_args();
        a.k = 0u;
        expect("dense k 0", launch_gemm_exl3_w16a16(a, nullptr), hipErrorInvalidValue);
        a = dense_args();
        a.out_features = 64u;
        expect("dense n align", launch_gemm_exl3_w16a16(a, nullptr), hipErrorInvalidValue);
        a = dense_args();
        a.k = 64u;
        expect("dense k align", launch_gemm_exl3_w16a16(a, nullptr), hipErrorInvalidValue);
        a = dense_args();
        a.ka = 0u;
        expect("dense ka 0", launch_gemm_exl3_w16a16(a, nullptr), hipErrorInvalidValue);
        a = dense_args();
        a.ka = 9u;
        expect("dense ka 9", launch_gemm_exl3_w16a16(a, nullptr), hipErrorInvalidValue);
        a = dense_args();
        a.half = 1u;
        a.ka = 4u;
        expect("dense half ka 4", launch_gemm_exl3_w16a16(a, nullptr), hipErrorInvalidValue);
    }

    // grouped mul1: null group plan, shape, stride, codebook / activation rules.
    {
        auto a = grouped_args();
        a.group_expert = nullptr;
        expect("grouped null expert", launch_gemm_exl3_w16a16_grouped(a, nullptr), hipErrorInvalidValue);
        a = grouped_args();
        a.out_features = 64u;
        expect("grouped n align", launch_gemm_exl3_w16a16_grouped(a, nullptr), hipErrorInvalidValue);
        a = grouped_args();
        a.input_row_stride = 64u;
        expect("grouped input stride", launch_gemm_exl3_w16a16_grouped(a, nullptr), hipErrorInvalidValue);
        a = grouped_args();
        a.group_capacity = 0u;
        expect("grouped cap 0", launch_gemm_exl3_w16a16_grouped(a, nullptr), hipErrorInvalidValue);
        a = grouped_args();
        a.mul24 = 1u;
        a.half = 1u;
        expect("grouped mul24 half", launch_gemm_exl3_w16a16_grouped(a, nullptr), hipErrorInvalidValue);
        a = grouped_args();
        a.int8_act = 1u;
        a.mul24 = 0u;
        expect("grouped int8 without mul24", launch_gemm_exl3_w16a16_grouped(a, nullptr), hipErrorInvalidValue);
    }

    // mul24 gemm: split range, n/k alignment, scale stride.
    {
        expect("gemm splits 0",
               launch_exl3_mul24_gemm(nullptr, nullptr, 0u, 0u, nullptr, 4u, nullptr, 16u, 128u, 128u, 0u, nullptr),
               hipErrorInvalidValue);
        expect("gemm splits too large",
               launch_exl3_mul24_gemm(nullptr, nullptr, 0u, 0u, nullptr, 4u, nullptr, 16u, 128u, 128u, 2u, nullptr),
               hipErrorInvalidValue);
        expect("gemm n align",
               launch_exl3_mul24_gemm(nullptr, nullptr, 0u, 0u, nullptr, 4u, nullptr, 16u, 128u + 32u, 128u, 1u, nullptr),
               hipErrorInvalidValue);
        expect("gemm k align",
               launch_exl3_mul24_gemm(nullptr, nullptr, 0u, 0u, nullptr, 4u, nullptr, 16u, 128u, 64u, 1u, nullptr),
               hipErrorInvalidValue);
        expect("gemm scale stride",
               launch_exl3_mul24_gemm(nullptr, nullptr, 0u, 2u, nullptr, 4u, nullptr, 16u, 128u, 128u, 1u, nullptr),
               hipErrorInvalidValue);
    }

    // mul24 rotations: n alignment.
    {
        expect("rotate_out n 0",
               launch_exl3_mul24_rotate_out(nullptr, nullptr, nullptr, Exl3OutputDType::BF16, 1u, 0u, 0u, 1u, nullptr),
               hipErrorInvalidValue);
        expect("rotate_out n align",
               launch_exl3_mul24_rotate_out(nullptr, nullptr, nullptr, Exl3OutputDType::BF16, 1u, 129u, 129u, 1u,
                                            nullptr),
               hipErrorInvalidValue);
        expect("rotate_out_i32 n align",
               launch_exl3_mul24_rotate_out_i32(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                Exl3OutputDType::BF16, 1u, 129u, 129u, nullptr),
               hipErrorInvalidValue);
    }

    // mul24 rotate_quantize / rowscale / recon: k and stride rules.
    {
        expect("rq k 0",
               launch_exl3_mul24_rotate_quantize(nullptr, 0u, nullptr, nullptr, nullptr, 1u, 0u, 0u, 0u, nullptr),
               hipErrorInvalidValue);
        expect("rq k align",
               launch_exl3_mul24_rotate_quantize(nullptr, 0u, nullptr, nullptr, nullptr, 1u, 64u, 64u, 0u, nullptr),
               hipErrorInvalidValue);
        expect("rq scale stride",
               launch_exl3_mul24_rotate_quantize(nullptr, 0u, nullptr, nullptr, nullptr, 1u, 128u, 128u, 2u, nullptr),
               hipErrorInvalidValue);
        expect("rowscale k align",
               launch_exl3_mul24_rowscale(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1u, 1u, 64u,
                                          64u, 0u, 0.5f, nullptr),
               hipErrorInvalidValue);
        expect("recon n align",
               launch_exl3_mul24_recon(nullptr, 4u, nullptr, 128u + 8u, 128u, nullptr),
               hipErrorInvalidValue);
        expect("recon k align",
               launch_exl3_mul24_recon(nullptr, 4u, nullptr, 128u, 64u, nullptr),
               hipErrorInvalidValue);
    }

    // mul24 grouped: half and n_tiles rules.
    {
        shelf::kernel::Exl3GemmArgs a = dense_args();
        a.half = 1u;
        expect("mul24 grouped half", launch_exl3_mul24_grouped(a, false, 1u, 1u, 1u, nullptr),
               hipErrorInvalidValue);
        a = dense_args();
        expect("mul24 grouped n_tiles", launch_exl3_mul24_grouped(a, false, 3u, 1u, 1u, nullptr),
               hipErrorInvalidValue);
    }

    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
