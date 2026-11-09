#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/kv_append.h>
#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using shelf::kernel::OctaveKMode;
using shelf::kernel::OctaveKVCacheFormat;

int failures = 0;

using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

__global__ void octave_wmma_layout_test_kernel(const _Float16* a,
                                                const _Float16* b,
                                                float* c) {
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;
    const uint32_t lo = lane & 15u;
    const uint32_t hi = lane >> 4u;
    f16x8_t af;
    f16x8_t bf;
    #pragma unroll
    for (uint32_t e = 0u; e < 4u; ++e) {
        af[e] = a[lo * 16u + hi * 4u + e];
        af[4u + e] = a[lo * 16u + hi * 4u + 8u + e];
        bf[e] = b[(hi * 4u + e) * 16u + lo];
        bf[4u + e] = b[(hi * 4u + 8u + e) * 16u + lo];
    }
    const f32x8_t out =
        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf, f32x8_t{});
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e)
        c[(hi * 8u + e) * 16u + lo] = out[e];
}

void check(bool condition, const char* name) {
    if (!condition) {
        ++failures;
        std::printf("FAIL: %s\n", name);
    }
}

void run_wmma_layout_test() {
    std::vector<_Float16> a(256u), b(256u);
    std::vector<float> expected(256u, 0.0f), actual(256u, 0.0f);
    for (uint32_t r = 0u; r < 16u; ++r) {
        for (uint32_t k = 0u; k < 16u; ++k) {
            a[r * 16u + k] = static_cast<_Float16>((static_cast<int>(r) - 7) * 0.125f +
                                                   static_cast<int>(k) * 0.03125f);
            b[k * 16u + r] = static_cast<_Float16>((static_cast<int>(k) - 5) * 0.0625f -
                                                   static_cast<int>(r) * 0.015625f);
        }
    }
    for (uint32_t r = 0u; r < 16u; ++r)
        for (uint32_t col = 0u; col < 16u; ++col)
            for (uint32_t k = 0u; k < 16u; ++k)
                expected[r * 16u + col] +=
                    static_cast<float>(a[r * 16u + k]) *
                    static_cast<float>(b[k * 16u + col]);

    _Float16* da = nullptr;
    _Float16* db = nullptr;
    float* dc = nullptr;
    bool ok = hipMalloc(&da, a.size() * sizeof(_Float16)) == hipSuccess &&
              hipMalloc(&db, b.size() * sizeof(_Float16)) == hipSuccess &&
              hipMalloc(&dc, actual.size() * sizeof(float)) == hipSuccess;
    if (ok) {
        ok = hipMemcpy(da, a.data(), a.size() * sizeof(_Float16), hipMemcpyHostToDevice) == hipSuccess &&
             hipMemcpy(db, b.data(), b.size() * sizeof(_Float16), hipMemcpyHostToDevice) == hipSuccess;
    }
    if (ok) {
        octave_wmma_layout_test_kernel<<<1u, 32u>>>(da, db, dc);
        ok = hipGetLastError() == hipSuccess && hipDeviceSynchronize() == hipSuccess &&
             hipMemcpy(actual.data(), dc, actual.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess;
    }
    float maximum = 0.0f;
    if (ok) {
        for (std::size_t i = 0u; i < actual.size(); ++i)
            maximum = std::max(maximum, std::fabs(actual[i] - expected[i]));
    }
    std::printf("gfx12 WMMA layout max_abs=%.7f\n", maximum);
    check(ok && maximum < 1.0e-5f, "gfx12 WMMA operand/result layout");
    if (da != nullptr) hipFree(da);
    if (db != nullptr) hipFree(db);
    if (dc != nullptr) hipFree(dc);
}

double phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * 3.14159265358979323846);
}

double Phi(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

std::vector<float> make_levels() {
    std::vector<float> table(7u * 64u, 0.0f);
    for (int bits = 3; bits <= 6; ++bits) {
        const int count = 1 << bits;
        std::vector<double> levels(count);
        for (int i = 0; i < count; ++i)
            levels[i] = -2.5 + 5.0 * (i + 0.5) / count;
        for (int iteration = 0; iteration < 4000; ++iteration) {
            std::vector<double> boundaries(count + 1);
            boundaries[0] = -12.0;
            boundaries[count] = 12.0;
            for (int i = 1; i < count; ++i)
                boundaries[i] = 0.5 * (levels[i - 1] + levels[i]);
            for (int i = 0; i < count; ++i) {
                const double mass = Phi(boundaries[i + 1]) - Phi(boundaries[i]);
                if (mass > 1e-300)
                    levels[i] = (phi(boundaries[i]) - phi(boundaries[i + 1])) / mass;
            }
        }
        for (int i = 0; i < count; ++i)
            table[static_cast<std::size_t>(bits) * 64u + i] = static_cast<float>(levels[i]);
    }
    return table;
}

float sign_of(uint32_t index, uint32_t salt) {
    uint32_t x = index * 2654435761u + salt * 40503u + 12345u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x & 1u) != 0u ? -1.0f : 1.0f;
}

void wht(std::vector<float>& values, uint32_t begin, uint32_t dimensions) {
    for (uint32_t half = 1u; half < dimensions; half <<= 1u) {
        const std::vector<float> prior(values.begin() + begin,
                                       values.begin() + begin + dimensions);
        for (uint32_t i = 0u; i < dimensions; ++i) {
            const float other = prior[i ^ half];
            values[begin + i] = (i & half) != 0u ? other - prior[i] : prior[i] + other;
        }
    }
    const float factor = 1.0f / std::sqrt(static_cast<float>(dimensions));
    for (uint32_t i = 0u; i < dimensions; ++i) values[begin + i] *= factor;
}

uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    return __half_as_ushort(h);
}

float half_value(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

struct Quantized {
    std::vector<uint8_t> codes;
    std::vector<uint16_t> scales;
    std::vector<float> reconstructed;
};

Quantized quantize(const std::vector<float>& input, uint32_t segment,
                   uint32_t first_bits, uint32_t remaining_bits,
                   uint32_t salt, const std::vector<float>& levels) {
    Quantized out;
    out.codes.resize(256u);
    out.scales.resize(256u / segment);
    out.reconstructed.resize(256u);
    std::vector<float> rotated(256u);
    for (uint32_t i = 0u; i < 256u; ++i) rotated[i] = input[i] * sign_of(i, salt);
    for (uint32_t begin = 0u; begin < 256u; begin += segment) {
        wht(rotated, begin, segment);
        float ss = 0.0f;
        for (uint32_t i = 0u; i < segment; ++i)
            ss += rotated[begin + i] * rotated[begin + i];
        const float rms = std::sqrt(ss / static_cast<float>(segment));
        const uint32_t bits = begin == 0u ? first_bits : remaining_bits;
        float dot = 0.0f;
        for (uint32_t i = 0u; i < segment; ++i) {
            uint32_t code = 0u;
            float best = 1.0e30f;
            if (rms > 0.0f) {
                const float y = rotated[begin + i] / rms;
                for (uint32_t candidate = 0u; candidate < (1u << bits); ++candidate) {
                    const float distance = std::fabs(y - levels[bits * 64u + candidate]);
                    if (distance < best) {
                        best = distance;
                        code = candidate;
                    }
                }
            }
            out.codes[begin + i] = static_cast<uint8_t>(code);
            dot += rotated[begin + i] * levels[bits * 64u + code];
        }
        const uint16_t scale_bits = half_bits(dot > 0.0f ? ss / dot : 0.0f);
        out.scales[begin / segment] = scale_bits;
        const float scale = half_value(scale_bits);
        for (uint32_t i = 0u; i < segment; ++i)
            out.reconstructed[begin + i] = levels[bits * 64u + out.codes[begin + i]] * scale;
        wht(out.reconstructed, begin, segment);
    }
    for (uint32_t i = 0u; i < 256u; ++i) out.reconstructed[i] *= sign_of(i, salt);
    return out;
}

void pack_plane(const std::vector<uint8_t>& codes, uint32_t begin,
                uint32_t dimensions, uint32_t shift, uint32_t plane_bits,
                std::vector<uint8_t>& output) {
    const uint32_t fields = 32u / plane_bits;
    const uint32_t mask = (1u << plane_bits) - 1u;
    for (uint32_t base = 0u; base < dimensions; base += fields) {
        uint32_t word = 0u;
        for (uint32_t field = 0u; field < fields; ++field)
            word |= ((codes[begin + base + field] >> shift) & mask) << (field * plane_bits);
        for (uint32_t byte = 0u; byte < 4u; ++byte)
            output.push_back(static_cast<uint8_t>(word >> (byte * 8u)));
    }
}

void pack_codes(const std::vector<uint8_t>& codes, uint32_t begin,
                uint32_t dimensions, uint32_t bits, std::vector<uint8_t>& output) {
    if (bits >= 4u) {
        pack_plane(codes, begin, dimensions, 0u, 4u, output);
        if (bits == 5u) pack_plane(codes, begin, dimensions, 4u, 1u, output);
        if (bits == 6u) pack_plane(codes, begin, dimensions, 4u, 2u, output);
    } else {
        pack_plane(codes, begin, dimensions, 0u, 2u, output);
        pack_plane(codes, begin, dimensions, 2u, 1u, output);
    }
}

std::vector<uint8_t> packed_reference(const Quantized& quantized,
                                      OctaveKMode mode,
                                      uint32_t first_bits,
                                      uint32_t remaining_bits) {
    std::vector<uint8_t> output;
    const uint32_t segment = mode == OctaveKMode::Compact ? 256u : 64u;
    for (uint32_t begin = 0u; begin < 256u; begin += segment) {
        const uint32_t bits = begin == 0u ? first_bits : remaining_bits;
        pack_codes(quantized.codes, begin, segment, bits, output);
    }
    for (uint16_t scale : quantized.scales) {
        output.push_back(static_cast<uint8_t>(scale));
        output.push_back(static_cast<uint8_t>(scale >> 8u));
    }
    return output;
}

bool hip_ok(hipError_t error, const char* operation) {
    if (error == hipSuccess) return true;
    std::printf("FAIL: %s: %s\n", operation, hipGetErrorString(error));
    ++failures;
    return false;
}

void run_gpu_case(OctaveKVCacheFormat format, const char* name,
                  const std::vector<float>& levels) {
    std::vector<shelf::bf16_t> k(256u), v(256u);
    std::vector<float> kf(256u), vf(256u);
    for (uint32_t i = 0u; i < 256u; ++i) {
        const float a = std::sin(static_cast<float>(i) * 0.173f) * 2.0f +
                        std::cos(static_cast<float>(i) * 0.037f) * 0.25f;
        const float b = std::cos(static_cast<float>(i) * 0.119f) * 1.7f -
                        std::sin(static_cast<float>(i) * 0.041f) * 0.4f;
        k[i].data = shelf::f32_to_bf16_rne(a);
        v[i].data = shelf::f32_to_bf16_rne(b);
        uint32_t ku = static_cast<uint32_t>(k[i].data) << 16u;
        uint32_t vu = static_cast<uint32_t>(v[i].data) << 16u;
        std::memcpy(&kf[i], &ku, sizeof(float));
        std::memcpy(&vf[i], &vu, sizeof(float));
    }
    const uint32_t segment = format.k_mode == OctaveKMode::Compact ? 256u : 64u;
    const Quantized kq = quantize(kf, segment, format.k_rope_bits,
                                  format.k_nope_bits, 11u, levels);
    const Quantized vq = quantize(vf, 256u, format.v_bits, format.v_bits, 29u, levels);
    const std::vector<uint8_t> expected_k = packed_reference(
        kq, format.k_mode, format.k_rope_bits, format.k_nope_bits);
    const std::vector<uint8_t> expected_v = packed_reference(
        vq, OctaveKMode::Compact, format.v_bits, format.v_bits);

    shelf::bf16_t* d_k = nullptr;
    shelf::bf16_t* d_v = nullptr;
    uint8_t* d_k_pool = nullptr;
    uint8_t* d_v_pool = nullptr;
    float* d_levels = nullptr;
    shelf::bf16_t* d_k_scratch = nullptr;
    shelf::bf16_t* d_v_scratch = nullptr;
    uint32_t* d_slot = nullptr;
    uint32_t* d_position = nullptr;
    uint32_t* d_table = nullptr;
    bool ok = hip_ok(hipMalloc(&d_k, k.size() * sizeof(shelf::bf16_t)), "alloc K") &&
              hip_ok(hipMalloc(&d_v, v.size() * sizeof(shelf::bf16_t)), "alloc V") &&
              hip_ok(hipMalloc(&d_k_pool, expected_k.size()), "alloc K pool") &&
              hip_ok(hipMalloc(&d_v_pool, expected_v.size()), "alloc V pool") &&
              hip_ok(hipMalloc(&d_levels, levels.size() * sizeof(float)), "alloc levels") &&
              hip_ok(hipMalloc(&d_k_scratch, 256u * sizeof(shelf::bf16_t)), "alloc K scratch") &&
              hip_ok(hipMalloc(&d_v_scratch, 256u * sizeof(shelf::bf16_t)), "alloc V scratch") &&
              hip_ok(hipMalloc(&d_slot, sizeof(uint32_t)), "alloc slot") &&
              hip_ok(hipMalloc(&d_position, sizeof(uint32_t)), "alloc position") &&
              hip_ok(hipMalloc(&d_table, sizeof(uint32_t)), "alloc table");
    const uint32_t zero = 0u;
    if (ok) {
        ok = hip_ok(hipMemcpy(d_k, k.data(), k.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice), "copy K") &&
             hip_ok(hipMemcpy(d_v, v.data(), v.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice), "copy V") &&
             hip_ok(hipMemcpy(d_levels, levels.data(), levels.size() * sizeof(float), hipMemcpyHostToDevice), "copy levels") &&
             hip_ok(hipMemcpy(d_slot, &zero, sizeof(zero), hipMemcpyHostToDevice), "copy slot") &&
             hip_ok(hipMemcpy(d_position, &zero, sizeof(zero), hipMemcpyHostToDevice), "copy position") &&
             hip_ok(hipMemcpy(d_table, &zero, sizeof(zero), hipMemcpyHostToDevice), "copy table");
    }
    if (ok) {
        shelf::kernel::KvAppendCommonArgs args{};
        args.k_input = d_k;
        args.v_input = d_v;
        args.k_row_stride = 256u;
        args.v_row_stride = 256u;
        args.row_sequence_slots = d_slot;
        args.row_positions = d_position;
        args.block_tables = d_table;
        args.block_table_stride = 1u;
        args.max_sequences = 1u;
        args.rows = 1u;
        args.page_tokens = 1u;
        args.num_pages = 1u;
        args.num_attention_layers = 1u;
        args.kv_heads = 1u;
        args.head_dim = 256u;
        args.elems_per_token = 256u;
        args.elems_per_page = 256u;
        args.elems_per_layer = 256u;
        ok = hip_ok(shelf::kernel::launch_kv_append_octave(
            args, d_k_pool, d_v_pool, d_levels, format, nullptr), name) &&
             hip_ok(hipDeviceSynchronize(), "synchronize Octave append");
    }
    std::vector<uint8_t> actual_k(expected_k.size()), actual_v(expected_v.size());
    if (ok) {
        ok = hip_ok(hipMemcpy(actual_k.data(), d_k_pool, actual_k.size(), hipMemcpyDeviceToHost), "read K pool") &&
             hip_ok(hipMemcpy(actual_v.data(), d_v_pool, actual_v.size(), hipMemcpyDeviceToHost), "read V pool");
    }
    if (ok) {
        std::size_t k_mismatch = 0u;
        std::size_t v_mismatch = 0u;
        for (std::size_t i = 0u; i + kq.scales.size() * 2u < actual_k.size(); ++i)
            k_mismatch += actual_k[i] != expected_k[i];
        for (std::size_t i = 0u; i + vq.scales.size() * 2u < actual_v.size(); ++i)
            v_mismatch += actual_v[i] != expected_v[i];
        check(k_mismatch == 0u, name);
        check(v_mismatch == 0u, name);
        check(actual_k.size() == shelf::kernel::octave_k_slot_bytes(format), name);
        check(actual_v.size() == shelf::kernel::octave_v_slot_bytes(format), name);
    }
    std::vector<shelf::bf16_t> actual_k_dequant(256u), actual_v_dequant(256u);
    if (ok) {
        ok = hip_ok(shelf::kernel::launch_octave_kv_dequant(
            d_k_pool, d_v_pool, d_levels, d_k_scratch, d_v_scratch, format,
            0u, 1u, 1u, 1u, nullptr), "Octave dequant") &&
             hip_ok(hipDeviceSynchronize(), "synchronize Octave dequant") &&
             hip_ok(hipMemcpy(actual_k_dequant.data(), d_k_scratch,
                              256u * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost),
                    "read K scratch") &&
             hip_ok(hipMemcpy(actual_v_dequant.data(), d_v_scratch,
                              256u * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost),
                    "read V scratch");
    }
    if (ok) {
        float max_difference = 0.0f;
        for (uint32_t i = 0u; i < 256u; ++i) {
            uint32_t ku = static_cast<uint32_t>(actual_k_dequant[i].data) << 16u;
            uint32_t vu = static_cast<uint32_t>(actual_v_dequant[i].data) << 16u;
            float ka = 0.0f;
            float va = 0.0f;
            std::memcpy(&ka, &ku, sizeof(float));
            std::memcpy(&va, &vu, sizeof(float));
            max_difference = std::max(max_difference, std::fabs(ka - kq.reconstructed[i]));
            max_difference = std::max(max_difference, std::fabs(va - vq.reconstructed[i]));
        }
        check(max_difference < 0.02f, name);
    }
    if (d_k != nullptr) hipFree(d_k);
    if (d_v != nullptr) hipFree(d_v);
    if (d_k_pool != nullptr) hipFree(d_k_pool);
    if (d_v_pool != nullptr) hipFree(d_v_pool);
    if (d_levels != nullptr) hipFree(d_levels);
    if (d_k_scratch != nullptr) hipFree(d_k_scratch);
    if (d_v_scratch != nullptr) hipFree(d_v_scratch);
    if (d_slot != nullptr) hipFree(d_slot);
    if (d_position != nullptr) hipFree(d_position);
    if (d_table != nullptr) hipFree(d_table);
}

double normalized_mse(uint32_t bits, const std::vector<float>& levels) {
    std::mt19937 rng(12345u + bits);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    double error = 0.0;
    double reference = 0.0;
    for (uint32_t sample = 0u; sample < 128u; ++sample) {
        std::vector<float> input(256u);
        for (float& value : input) value = normal(rng);
        const Quantized q = quantize(input, 256u, bits, bits, 29u, levels);
        for (uint32_t i = 0u; i < 256u; ++i) {
            const double difference = q.reconstructed[i] - input[i];
            error += difference * difference;
            reference += static_cast<double>(input[i]) * input[i];
        }
    }
    return error / reference;
}

void run_attention_parity(OctaveKVCacheFormat format, uint32_t query_rows,
                          bool tree_visibility, const char* name,
                          const std::vector<float>& levels,
                          bool prefill = false, uint32_t prefix = 0u,
                          uint32_t decode_tokens = 64u,
                          bool microbench = false) {
    const uint32_t tokens = prefill ? prefix + query_rows : decode_tokens;
    constexpr uint32_t kv_heads = 4u;
    constexpr uint32_t q_heads = 24u;
    constexpr uint32_t page_tokens = 16u;
    const uint32_t pages = (tokens + page_tokens - 1u) / page_tokens;
    constexpr uint32_t head_dim = 256u;
    constexpr uint32_t kv_features = kv_heads * head_dim;
    constexpr uint32_t q_features = q_heads * head_dim;
    std::vector<shelf::bf16_t> k(static_cast<std::size_t>(tokens) * kv_features);
    std::vector<shelf::bf16_t> v(k.size());
    for (std::size_t i = 0u; i < k.size(); ++i) {
        k[i].data = microbench ? 0u : shelf::f32_to_bf16_rne(
            std::sin(static_cast<float>(i) * 0.013f) * 1.5f);
        v[i].data = microbench ? 0u : shelf::f32_to_bf16_rne(
            std::cos(static_cast<float>(i) * 0.017f) * 1.3f);
    }
    std::vector<shelf::bf16_t> q(static_cast<std::size_t>(query_rows) * q_features);
    for (std::size_t i = 0u; i < q.size(); ++i)
        q[i].data = shelf::f32_to_bf16_rne(
            std::sin(static_cast<float>(i) * 0.021f) * 0.8f);
    std::vector<uint32_t> append_slots(tokens, 0u), append_positions(tokens);
    for (uint32_t i = 0u; i < tokens; ++i) append_positions[i] = i;
    std::vector<uint32_t> table(pages);
    for (uint32_t i = 0u; i < pages; ++i) table[i] = i;
    std::vector<uint32_t> query_slots(query_rows, 0u), query_positions(query_rows);
    for (uint32_t row = 0u; row < query_rows; ++row)
        query_positions[row] = prefill ? prefix + row : tokens - query_rows + row;
    std::vector<uint32_t> tree_offsets(query_rows + 1u, 0u);
    std::vector<uint32_t> tree_positions;
    if (tree_visibility) {
        tree_positions.reserve(query_rows * 2u);
        for (uint32_t row = 0u; row < query_rows; ++row) {
            tree_offsets[row] = static_cast<uint32_t>(tree_positions.size());
            tree_positions.push_back(16u + row % 16u);
            tree_positions.push_back(32u + (row * 3u) % 32u);
        }
        tree_offsets[query_rows] = static_cast<uint32_t>(tree_positions.size());
    }

    std::vector<void*> allocations;
    auto allocate = [&](void** pointer, std::size_t bytes) {
        if (!hip_ok(hipMalloc(pointer, bytes), name)) return false;
        allocations.push_back(*pointer);
        return true;
    };
    shelf::bf16_t* d_k = nullptr;
    shelf::bf16_t* d_v = nullptr;
    shelf::bf16_t* d_q = nullptr;
    uint8_t* d_k_pool = nullptr;
    uint8_t* d_v_pool = nullptr;
    shelf::bf16_t* d_k_scratch = nullptr;
    shelf::bf16_t* d_v_scratch = nullptr;
    float* d_levels = nullptr;
    uint32_t* d_append_slots = nullptr;
    uint32_t* d_append_positions = nullptr;
    uint32_t* d_query_slots = nullptr;
    uint32_t* d_query_positions = nullptr;
    uint32_t* d_table = nullptr;
    uint32_t* d_tree_offsets = nullptr;
    uint32_t* d_tree_positions = nullptr;
    float* d_reference = nullptr;
    float* d_fused = nullptr;
    float* d_legacy = nullptr;
    float* d_partials = nullptr;
    const std::size_t vectors = static_cast<std::size_t>(pages) * page_tokens * kv_heads;
    const std::size_t scratch_elems = vectors * head_dim;
    const uint32_t splits = tree_visibility ? 1u : 4u;
    const std::size_t partial_elems = static_cast<std::size_t>(query_rows) *
        kv_heads * splits * (q_heads / kv_heads) * 258u;
    bool ok = allocate(reinterpret_cast<void**>(&d_k), k.size() * sizeof(shelf::bf16_t)) &&
              allocate(reinterpret_cast<void**>(&d_v), v.size() * sizeof(shelf::bf16_t)) &&
              allocate(reinterpret_cast<void**>(&d_q), q.size() * sizeof(shelf::bf16_t)) &&
              allocate(reinterpret_cast<void**>(&d_k_pool), vectors * shelf::kernel::octave_k_slot_bytes(format)) &&
              allocate(reinterpret_cast<void**>(&d_v_pool), vectors * shelf::kernel::octave_v_slot_bytes(format)) &&
              allocate(reinterpret_cast<void**>(&d_k_scratch), scratch_elems * sizeof(shelf::bf16_t)) &&
              allocate(reinterpret_cast<void**>(&d_v_scratch), scratch_elems * sizeof(shelf::bf16_t)) &&
              allocate(reinterpret_cast<void**>(&d_levels), levels.size() * sizeof(float)) &&
              allocate(reinterpret_cast<void**>(&d_append_slots), tokens * sizeof(uint32_t)) &&
              allocate(reinterpret_cast<void**>(&d_append_positions), tokens * sizeof(uint32_t)) &&
              allocate(reinterpret_cast<void**>(&d_query_slots), query_rows * sizeof(uint32_t)) &&
              allocate(reinterpret_cast<void**>(&d_query_positions), query_rows * sizeof(uint32_t)) &&
              allocate(reinterpret_cast<void**>(&d_table), table.size() * sizeof(uint32_t)) &&
              (!tree_visibility ||
               (allocate(reinterpret_cast<void**>(&d_tree_offsets), tree_offsets.size() * sizeof(uint32_t)) &&
                allocate(reinterpret_cast<void**>(&d_tree_positions), tree_positions.size() * sizeof(uint32_t)))) &&
              allocate(reinterpret_cast<void**>(&d_reference), q.size() * sizeof(float)) &&
              allocate(reinterpret_cast<void**>(&d_fused), q.size() * sizeof(float)) &&
              (!prefill || allocate(reinterpret_cast<void**>(&d_legacy),
                                    q.size() * sizeof(float))) &&
              allocate(reinterpret_cast<void**>(&d_partials), partial_elems * sizeof(float));
    if (ok) {
        ok = hip_ok(hipMemcpy(d_k, k.data(), k.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice), name) &&
             hip_ok(hipMemcpy(d_v, v.data(), v.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice), name) &&
             hip_ok(hipMemcpy(d_q, q.data(), q.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice), name) &&
             hip_ok(hipMemcpy(d_levels, levels.data(), levels.size() * sizeof(float), hipMemcpyHostToDevice), name) &&
             hip_ok(hipMemcpy(d_append_slots, append_slots.data(), tokens * sizeof(uint32_t), hipMemcpyHostToDevice), name) &&
             hip_ok(hipMemcpy(d_append_positions, append_positions.data(), tokens * sizeof(uint32_t), hipMemcpyHostToDevice), name) &&
             hip_ok(hipMemcpy(d_query_slots, query_slots.data(), query_rows * sizeof(uint32_t), hipMemcpyHostToDevice), name) &&
             hip_ok(hipMemcpy(d_query_positions, query_positions.data(), query_rows * sizeof(uint32_t), hipMemcpyHostToDevice), name) &&
             hip_ok(hipMemcpy(d_table, table.data(), table.size() * sizeof(uint32_t), hipMemcpyHostToDevice), name);
        if (ok && tree_visibility) {
            ok = hip_ok(hipMemcpy(d_tree_offsets, tree_offsets.data(),
                                  tree_offsets.size() * sizeof(uint32_t),
                                  hipMemcpyHostToDevice), name) &&
                 hip_ok(hipMemcpy(d_tree_positions, tree_positions.data(),
                                  tree_positions.size() * sizeof(uint32_t),
                                  hipMemcpyHostToDevice), name);
        }
    }
    shelf::kernel::KvAppendCommonArgs append{};
    append.k_input = d_k;
    append.v_input = d_v;
    append.k_row_stride = kv_features;
    append.v_row_stride = kv_features;
    append.row_sequence_slots = d_append_slots;
    append.row_positions = d_append_positions;
    append.block_tables = d_table;
    append.block_table_stride = pages;
    append.max_sequences = 1u;
    append.rows = tokens;
    append.page_tokens = page_tokens;
    append.num_pages = pages;
    append.num_attention_layers = 1u;
    append.kv_heads = kv_heads;
    append.head_dim = head_dim;
    append.elems_per_token = kv_features;
    append.elems_per_page = page_tokens * kv_features;
    append.elems_per_layer = pages * append.elems_per_page;
    if (ok) {
        ok = hip_ok(shelf::kernel::launch_kv_append_octave(
                        append, d_k_pool, d_v_pool, d_levels, format, nullptr), name) &&
             hip_ok(shelf::kernel::launch_octave_kv_dequant(
                        d_k_pool, d_v_pool, d_levels, d_k_scratch, d_v_scratch,
                        format, 0u, pages, page_tokens, kv_heads, nullptr), name);
    }
    shelf::kernel::PagedAttentionCommonArgs attention{};
    attention.q = d_q;
    attention.output = d_reference;
    attention.output_dtype = shelf::kernel::PagedAttentionOutputDType::F32;
    attention.q_row_stride = q_features;
    attention.output_row_stride = q_features;
    attention.row_sequence_slots = d_query_slots;
    attention.row_positions = d_query_positions;
    attention.tree_anc_offsets = d_tree_offsets;
    attention.tree_anc_positions = d_tree_positions;
    attention.tree_committed_end = tree_visibility ? 16u : 0u;
    attention.block_tables = d_table;
    attention.block_table_stride = pages;
    attention.max_sequences = 1u;
    attention.rows = query_rows;
    attention.page_tokens = page_tokens;
    attention.num_pages = pages;
    attention.num_attention_layers = 1u;
    attention.q_heads = q_heads;
    attention.kv_heads = kv_heads;
    attention.head_dim = head_dim;
    attention.elems_per_token = kv_features;
    attention.elems_per_page = page_tokens * kv_features;
    attention.elems_per_layer = pages * attention.elems_per_page;
    attention.scale = 0.0625f;
    if (ok) {
        if (prefill) {
            ok = hip_ok(shelf::kernel::launch_attention_paged_prefill_bf16(
                            attention, d_k_scratch, d_v_scratch, nullptr), name);
            attention.output = d_fused;
            unsetenv("SHELF_OCTAVE_PREFILL_LEGACY");
            ok = ok && hip_ok(shelf::kernel::launch_attention_paged_prefill_octave(
                                  attention, d_k_pool, d_v_pool, d_levels, format,
                                  nullptr), name);
            attention.output = d_legacy;
            setenv("SHELF_OCTAVE_PREFILL_LEGACY", "1", 1);
            ok = ok && hip_ok(shelf::kernel::launch_attention_paged_prefill_octave(
                                  attention, d_k_pool, d_v_pool, d_levels, format,
                                  nullptr), name);
            unsetenv("SHELF_OCTAVE_PREFILL_LEGACY");
        } else {
            ok = hip_ok(shelf::kernel::launch_attention_paged_bf16(
                            attention, d_k_scratch, d_v_scratch, nullptr), name);
            attention.output = d_fused;
            ok = ok && hip_ok(shelf::kernel::launch_attention_paged_octave_split(
                                  attention, d_k_pool, d_v_pool, d_levels, format,
                                  d_partials, splits, nullptr), name) &&
                 hip_ok(shelf::kernel::launch_attention_paged_octave_reduce(
                            attention, d_partials, splits, nullptr), name);
        }
        ok = ok && hip_ok(hipDeviceSynchronize(), name);
    }
    if (ok && microbench) {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        ok = hip_ok(hipEventCreate(&start), name) &&
             hip_ok(hipEventCreate(&stop), name);
        std::vector<float> elapsed;
        for (uint32_t iteration = 0u; ok && iteration < 6u; ++iteration) {
            ok = hip_ok(hipEventRecord(start), name) &&
                 hip_ok(shelf::kernel::launch_attention_paged_octave_split(
                            attention, d_k_pool, d_v_pool, d_levels, format,
                            d_partials, splits, nullptr), name) &&
                 hip_ok(shelf::kernel::launch_attention_paged_octave_reduce(
                            attention, d_partials, splits, nullptr), name) &&
                 hip_ok(hipEventRecord(stop), name) &&
                 hip_ok(hipEventSynchronize(stop), name);
            float milliseconds = 0.0f;
            if (ok) ok = hip_ok(hipEventElapsedTime(&milliseconds, start, stop), name);
            if (ok && iteration != 0u) elapsed.push_back(milliseconds);
        }
        if (!elapsed.empty()) {
            std::sort(elapsed.begin(), elapsed.end());
            std::printf("%s gpu_ms_p50=%.4f rows=%u context=%u\n", name,
                        elapsed[elapsed.size() / 2u], query_rows, tokens);
        }
        if (start != nullptr) hipEventDestroy(start);
        if (stop != nullptr) hipEventDestroy(stop);
    }
    std::vector<float> reference(q.size()), fused(q.size()), legacy(q.size());
    if (ok) {
        ok = hip_ok(hipMemcpy(reference.data(), d_reference,
                              reference.size() * sizeof(float), hipMemcpyDeviceToHost), name) &&
             hip_ok(hipMemcpy(fused.data(), d_fused,
                              fused.size() * sizeof(float), hipMemcpyDeviceToHost), name);
        if (ok && prefill)
            ok = hip_ok(hipMemcpy(legacy.data(), d_legacy,
                                  legacy.size() * sizeof(float), hipMemcpyDeviceToHost), name);
    }
    if (ok) {
        float maximum = 0.0f;
        for (std::size_t i = 0u; i < reference.size(); ++i) {
            const float difference = std::fabs(reference[i] - fused[i]);
            maximum = std::max(maximum, difference);
        }
        std::printf("%s max_abs=%.7f\n", name, maximum);
        check(maximum < 0.025f, name);
        if (prefill) {
            float legacy_maximum = 0.0f;
            float new_legacy_maximum = 0.0f;
            for (std::size_t i = 0u; i < reference.size(); ++i) {
                legacy_maximum = std::max(
                    legacy_maximum, std::fabs(reference[i] - legacy[i]));
                new_legacy_maximum = std::max(
                    new_legacy_maximum, std::fabs(fused[i] - legacy[i]));
            }
            std::printf("%s legacy_max_abs=%.7f new_legacy_max_abs=%.7f\n",
                        name, legacy_maximum, new_legacy_maximum);
            check(legacy_maximum < 0.025f, name);
            check(new_legacy_maximum < 0.025f, name);
        }
    }
    for (void* pointer : allocations) hipFree(pointer);
}

}  // namespace

int main() {
    run_wmma_layout_test();
    const std::vector<float> levels = make_levels();
    run_gpu_case({OctaveKMode::Compact, 3u, 3u, 3u}, "Octave c3v3", levels);
    run_gpu_case({OctaveKMode::Compact, 4u, 4u, 4u}, "Octave c4v4", levels);
    run_gpu_case({OctaveKMode::Compact, 5u, 5u, 5u}, "Octave c5v5", levels);
    run_gpu_case({OctaveKMode::Compact, 6u, 6u, 6u}, "Octave c6v6", levels);
    run_gpu_case({OctaveKMode::Block, 4u, 3u, 3u}, "Octave b4n3v3", levels);
    run_gpu_case({OctaveKMode::Block, 4u, 3u, 4u}, "Octave b4n3v4", levels);
    run_attention_parity({OctaveKMode::Compact, 4u, 4u, 4u}, 1u, false,
                         "Octave decode parity", levels);
    run_attention_parity({OctaveKMode::Block, 4u, 3u, 3u}, 4u, false,
                         "Octave verify parity", levels);
    run_attention_parity({OctaveKMode::Compact, 5u, 5u, 5u}, 8u, false,
                         "Octave rows 8 parity", levels);
    run_attention_parity({OctaveKMode::Block, 4u, 3u, 3u}, 32u, false,
                         "Octave rows 32 parity", levels);
    run_attention_parity({OctaveKMode::Compact, 4u, 4u, 4u}, 8u, true,
                         "Octave tree visibility parity", levels);
    run_attention_parity({OctaveKMode::Compact, 4u, 4u, 4u}, 32u, false,
                         "Octave prefill causal prefix parity", levels,
                         true, 17u);
    run_attention_parity({OctaveKMode::Block, 4u, 3u, 3u}, 33u, false,
                         "Octave prefill chunk boundary parity", levels,
                         true, 16u);
    run_attention_parity({OctaveKMode::Compact, 4u, 4u, 4u}, 2048u, false,
                         "Octave prefill rows 2048 parity", levels,
                         true, 17u);
    run_attention_parity({OctaveKMode::Compact, 4u, 4u, 4u}, 8192u, false,
                         "Octave prefill rows 8192 parity", levels,
                         true, 16u);
    const char* long_bench = std::getenv("SHELF_OCTAVE_LONG_BENCH");
    if (long_bench != nullptr && long_bench[0] == '1' && long_bench[1] == '\0')
        run_attention_parity({OctaveKMode::Compact, 4u, 4u, 4u}, 4u, false,
                             "Octave decode rows 4 context 131072", levels,
                             false, 0u, 131072u, true);

    const double mse3 = normalized_mse(3u, levels);
    const double mse4 = normalized_mse(4u, levels);
    const double mse5 = normalized_mse(5u, levels);
    std::printf("NMSE 3=%.6f 4=%.6f 5=%.6f\n", mse3, mse4, mse5);
    check(mse3 > 0.030 && mse3 < 0.040, "3-bit normalized MSE");
    check(mse4 > 0.007 && mse4 < 0.012, "4-bit normalized MSE");
    check(mse5 > 0.0017 && mse5 < 0.0032, "5-bit normalized MSE");

    if (failures != 0) {
        std::printf("FAILED: %d\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
