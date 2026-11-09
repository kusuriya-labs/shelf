#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/kv_append.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using shelf::kernel::OctaveKMode;
using shelf::kernel::OctaveKVCacheFormat;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        ++failures;
        std::printf("FAIL: %s\n", name);
    }
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
    uint32_t* d_slot = nullptr;
    uint32_t* d_position = nullptr;
    uint32_t* d_table = nullptr;
    bool ok = hip_ok(hipMalloc(&d_k, k.size() * sizeof(shelf::bf16_t)), "alloc K") &&
              hip_ok(hipMalloc(&d_v, v.size() * sizeof(shelf::bf16_t)), "alloc V") &&
              hip_ok(hipMalloc(&d_k_pool, expected_k.size()), "alloc K pool") &&
              hip_ok(hipMalloc(&d_v_pool, expected_v.size()), "alloc V pool") &&
              hip_ok(hipMalloc(&d_levels, levels.size() * sizeof(float)), "alloc levels") &&
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
    if (d_k != nullptr) hipFree(d_k);
    if (d_v != nullptr) hipFree(d_v);
    if (d_k_pool != nullptr) hipFree(d_k_pool);
    if (d_v_pool != nullptr) hipFree(d_v_pool);
    if (d_levels != nullptr) hipFree(d_levels);
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

}  // namespace

int main() {
    const std::vector<float> levels = make_levels();
    run_gpu_case({OctaveKMode::Compact, 3u, 3u, 3u}, "Octave c3v3", levels);
    run_gpu_case({OctaveKMode::Compact, 4u, 4u, 4u}, "Octave c4v4", levels);
    run_gpu_case({OctaveKMode::Compact, 5u, 5u, 5u}, "Octave c5v5", levels);
    run_gpu_case({OctaveKMode::Compact, 6u, 6u, 6u}, "Octave c6v6", levels);
    run_gpu_case({OctaveKMode::Block, 4u, 3u, 3u}, "Octave b4n3v3", levels);
    run_gpu_case({OctaveKMode::Block, 4u, 3u, 4u}, "Octave b4n3v4", levels);

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
