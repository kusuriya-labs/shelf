#include <shelf/quantization/fp8/fp8_frag.h>
#include <shelf/runtime/program/int8_activation_workspace.h>

#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const char* what) {
    if (ok) {
        ++g_pass;
        return;
    }
    ++g_fail;
    std::printf("FAIL: %s\n", what);
}

using namespace ps::quantization::fp8;

void test_block_is_permutation() {
    std::vector<int> seen(kFragBlockBytes, 0);
    for (uint32_t row = 0; row < 16u; ++row) {
        for (uint32_t k = 0; k < kFragBlockK; ++k) {
            const uint32_t off = fp8_frag_offset(row, k);
            if (off >= kFragBlockBytes) {
                check(false, "fragment offset inside block");
                return;
            }
            ++seen[off];
        }
    }
    bool bijective = true;
    for (int v : seen) {
        if (v != 1) bijective = false;
    }
    check(bijective, "fragment offsets cover each block byte exactly once");
}

void test_lane_slots_are_contiguous() {
    bool ok = true;
    for (uint32_t row = 0; row < 16u; ++row) {
        for (uint32_t kg = 0; kg < 2u; ++kg) {
            const uint32_t k0 = kg * 8u;
            const uint32_t base = fp8_frag_offset(row, k0);
            for (uint32_t f = 0; f < kFragBlockK / 16u; ++f) {
                const uint32_t frag_start = f * 16u + k0;
                if (fp8_frag_offset(row, frag_start) != base + f * 8u) ok = false;
                for (uint32_t e = 0; e < 8u; ++e) {
                    if (fp8_frag_offset(row, frag_start + e) != base + f * 8u + e) ok = false;
                }
            }
            if (base % kFragLaneBytes != 0u) ok = false;
            if (base / kFragLaneBytes != row + kg * 16u) ok = false;
        }
    }
    check(ok, "each lane holds 32 contiguous bytes for one row");
}

void test_fragment_rows_match_wmma_map() {
    bool ok = true;
    for (uint32_t row = 0; row < 16u; ++row) {
        for (uint32_t k = 0; k < kFragBlockK; ++k) {
            const uint32_t lane = fp8_frag_offset(row, k) / kFragLaneBytes;
            if ((lane & 15u) != row) ok = false;
            if ((lane >> 4u) != ((k & 15u) >> 3u)) ok = false;
        }
    }
    check(ok, "lane index encodes (row, k half) as the WMMA fragment map");
}

void test_block_indexing() {
    check(fp8_frag_blocks(4096u) == 64u, "block count for k=4096");
    check(fp8_frag_tile_bytes(4096u) == 4096u * 16u, "tile bytes for k=4096");
    check(fp8_frag_codes_bytes(3u, 4096u) == 3u * 4096u * 16u, "codes bytes for 3 tiles");
    check(fp8_frag_tile_bytes(4096u) == fp8_frag_blocks(4096u) * kFragBlockBytes,
          "tile bytes equal block bytes");
}

void test_activation_padding_alignment() {
    bool ok = true;
    const uint32_t ks[] = {1u, 31u, 32u, 33u, 64u, 100u, 5000u, 5120u, 17408u};
    for (uint32_t k : ks) {
        const uint32_t kp = ps::runtime::Int8ActivationWorkspaceLayout::k_padded_for(k);
        if ((kp % kFragBlockK) != 0u || kp < k) ok = false;
        const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(k, 8u);
        if ((layout.code_row_stride_bytes % kFragBlockK) != 0u) ok = false;
        if (layout.codes_bytes() % kFragBlockBytes != 0u) ok = false;
    }
    check(ok, "activation workspace is padded to whole fragment blocks");
}

}  // namespace

int main() {
    test_block_is_permutation();
    test_lane_slots_are_contiguous();
    test_fragment_rows_match_wmma_map();
    test_block_indexing();
    test_activation_padding_alignment();
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
