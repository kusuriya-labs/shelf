#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

template <typename T>
bool allocate(shelf::gpu::GpuArena& arena, std::size_t count, shelf::gpu::Tensor& out) {
    auto view = arena.allocate_aligned(count * sizeof(T), 128);
    if (!view.ok()) return false;
    auto tensor = shelf::gpu::Tensor::create<T>(view.value(), {count}, {1u});
    if (!tensor.ok()) return false;
    out = tensor.release();
    return true;
}

int main() {
    constexpr uint32_t rows = 3, n = 256, k = 256, bits = 4, stride = n + 16;
    auto arena_result = shelf::gpu::GpuArena::create(0, 64u * 1024u * 1024u);
    if (!arena_result.ok()) return 1;
    auto arena = arena_result.release();
    std::mt19937 rng(193u);
    std::vector<uint16_t> source((k / 16u) * (n / 16u) * 16u * bits);
    for (auto& x : source) x = static_cast<uint16_t>(rng());
    std::vector<uint16_t> resident(source.size());
    std::vector<int8_t> decoded(n * k);
    if (!shelf::quantization::exl3::exl3_tiles_to_nmajor(source.data(), k / 16u, n / 16u, bits, resident.data(), 1u) ||
        !shelf::quantization::exl3::exl3_decode_int8(source.data(), k / 16u, n / 16u, bits, decoded.data(), 1u)) return 1;
    std::vector<uint16_t> scales(k, 0x3c00u);
    std::vector<shelf::bf16_t> input(rows * k);
    for (auto& x : input) x = shelf::bf16_t{shelf::f32_to_bf16_rne(static_cast<float>(static_cast<int>(rng() % 201u) - 100) / 100.0f)};
    shelf::weights::MatrixWeight weight;
    weight.encoding = shelf::weights::MatrixEncoding::Exl3;
    weight.quant_spec = shelf::quantization::QuantSpecId::EXL3_MUL24_INT8_V1;
    weight.rows = n; weight.cols = k; weight.k_padded = k; weight.exl3_ka = bits;
    shelf::qwen35::dflash2::DFlash2Config config;
    config.hidden_size = k; config.vocab_size = n;
    shelf::qwen35::dflash2::DFlash2Executor executor;
    executor.config = &config; executor.target_lm_head = &weight; executor.max_rows = rows;
    executor.draft_rows = rows; executor.initialized = true;
    auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(k, rows);
    shelf::gpu::Tensor device_input, device_output;
    if (!allocate<uint16_t>(arena, resident.size(), weight.codes) ||
        !allocate<uint16_t>(arena, k, weight.exl3_suh) || !allocate<uint16_t>(arena, n, weight.exl3_svh) ||
        !allocate<int8_t>(arena, layout.codes_bytes(), executor.activation_codes) ||
        !allocate<float>(arena, layout.scales_bytes() / sizeof(float), executor.activation_scales) ||
        !allocate<float>(arena, rows * n, executor.exl3_lm_head_accum) ||
        !allocate<shelf::bf16_t>(arena, rows * k, device_input) || !allocate<float>(arena, rows * stride, device_output)) return 1;
    if (hipMemcpy(weight.codes.data<void>(), resident.data(), resident.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(weight.exl3_suh.data<void>(), scales.data(), k * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(weight.exl3_svh.data<void>(), scales.data(), n * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(device_input.data<void>(), input.data(), input.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice) != hipSuccess) return 1;
    auto st = shelf::qwen35::dflash2::dflash2_compute_target_logits(executor, device_input.data<shelf::bf16_t>(), rows, device_output.data<float>(), stride, nullptr);
    if (!st.ok()) { std::printf("head: %s\n", st.message().c_str()); return 1; }
    std::vector<float> output(rows * stride);
    if (hipMemcpy(output.data(), device_output.data<void>(), output.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 1;
    double error = 0.0, norm = 0.0;
    for (uint32_t r = 0; r < rows; ++r) {
        double rotated[k];
        for (uint32_t i = 0; i < k; ++i) {
            double sum = 0.0;
            for (uint32_t j = 0; j < 128u; ++j) {
                uint32_t u = static_cast<uint32_t>(input[r * k + (i / 128u) * 128u + j].data) << 16;
                float value; std::memcpy(&value, &u, sizeof(value));
                sum += value * ((__builtin_popcount((i % 128u) & j) & 1) ? -1.0 : 1.0);
            }
            rotated[i] = sum / std::sqrt(128.0);
        }
        double y[n];
        for (uint32_t i = 0; i < n; ++i) {
            y[i] = 0.0;
            for (uint32_t j = 0; j < k; ++j) y[i] += decoded[i * k + j] * rotated[j];
            y[i] *= shelf::quantization::exl3::kExl3Int8GridStep;
        }
        for (uint32_t i = 0; i < n; ++i) {
            double ref = 0.0;
            for (uint32_t j = 0; j < 128u; ++j) ref += y[(i / 128u) * 128u + j] * ((__builtin_popcount((i % 128u) & j) & 1) ? -1.0 : 1.0);
            ref /= std::sqrt(128.0);
            double delta = output[r * stride + i] - ref;
            error += delta * delta; norm += ref * ref;
        }
    }
    const double rel = std::sqrt(error / norm);
    std::printf("dflash2 mul24 head relative L2: %.6f\n", rel);
    return std::isfinite(rel) && rel < 0.02 && arena.shutdown().ok() ? 0 : 1;
}
