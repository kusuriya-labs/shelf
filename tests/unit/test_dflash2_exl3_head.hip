#include <shelf/core/memory/arena.h>
#include <shelf/kernels/dflash2/coarse_head.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/dflash2/executor.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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

template<uint32_t bits, uint32_t n>
int check_head() {
    constexpr uint32_t rows = 3, k = 256, stride = n + 16;
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
    std::vector<uint16_t> scales(std::max(k, n), 0x3c00u);
    std::vector<shelf::bf16_t> input(rows * k);
    for (auto& x : input) x = shelf::bf16_t{shelf::f32_to_bf16_rne(static_cast<float>(static_cast<int>(rng() % 201u) - 100) / 100.0f)};
    shelf::weights::MatrixWeight weight;
    weight.encoding = shelf::weights::MatrixEncoding::Exl3;
    weight.quant_spec = shelf::quantization::QuantSpecId::EXL3_MUL24_INT8_V1;
    weight.rows = n; weight.cols = k; weight.k_padded = k; weight.exl3_ka = bits;
    ::shelf::runtime::DFlash2Config config;
    config.hidden_size = k; config.vocab_size = n;
    ::shelf::runtime::dflash2::DFlash2Executor executor;
    executor.config = &config; executor.target_lm_head = &weight; executor.max_rows = rows;
    executor.local_vocab = n;
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
    auto st = ::shelf::runtime::dflash2::dflash2_compute_target_logits(executor, device_input.data<shelf::bf16_t>(), rows, device_output.data<float>(), stride, nullptr);
    if (!st.ok()) { std::printf("head: %s\n", st.message().c_str()); return 1; }
    std::vector<float> output(rows * stride);
    if (hipMemcpy(output.data(), device_output.data<void>(), output.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 1;
    std::vector<float> coarse;
    if constexpr (bits == 6) {
        shelf::gpu::Tensor packed, peaks, mask;
        if (!allocate<uint32_t>(arena, n * k * 3 / 32, packed) ||
            !allocate<float>(arena, rows * n / 128, peaks) ||
            !allocate<int32_t>(arena, n / 128, mask)) return 1;
        shelf::kernel::DFlash2CoarseHeadArgs args{
            packed.data<uint32_t>(), weight.codes.data<uint16_t>(), weight.exl3_svh.data<uint16_t>(),
            executor.activation_codes.data<int8_t>(), executor.activation_scales.data<float>(),
            executor.exl3_lm_head_accum.data<float>(), device_output.data<float>(),
            peaks.data<float>(), mask.data<int32_t>(), rows, n, k, layout.code_row_stride_bytes,
            layout.scale_row_stride_bytes / 4, 0};
        if (shelf::kernel::build_dflash2_coarse_head(args.trellis, packed.data<uint32_t>(), n, k, nullptr) != hipSuccess ||
            shelf::kernel::launch_dflash2_coarse_head(args, nullptr) != hipSuccess) return 1;
        coarse.resize(rows * n);
        if (hipMemcpy(coarse.data(), args.logits, coarse.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 1;
        args.keep = 2;
        if (shelf::kernel::launch_dflash2_coarse_head(args, nullptr) != hipSuccess) return 1;
        std::vector<float> mixed(rows * n), ranked(rows * n);
        std::vector<int32_t> selected(n / 128);
        if (hipMemcpy(mixed.data(), args.logits, mixed.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(ranked.data(), args.accum, ranked.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(selected.data(), args.selected, selected.size() * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess) return 1;
        int blocks = 0;
        for (auto flag : selected) blocks += flag;
        if (blocks < 2 || blocks > 2 * int(rows)) return 1;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < n; ++col) {
                const auto i = row * n + col;
                const bool fine = selected[col / 128];
                const float ref = fine ? output[row * stride + col] : coarse[i];
                if (mixed[i] != ref || ranked[i] != (fine ? ref : -INFINITY)) {
                    std::fprintf(stderr, "row=%u col=%u selected=%d mixed=%g ranked=%g expected=%g\n",
                                 row, col, int(fine), mixed[i], ranked[i], ref);
                    return 1;
                }
            }
        }
    }
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
        double y[n], cy[n]{};
        for (uint32_t i = 0; i < n; ++i) {
            y[i] = 0.0;
            for (uint32_t j = 0; j < k; ++j) y[i] += decoded[i * k + j] * rotated[j];
            y[i] *= shelf::quantization::exl3::kExl3Int8GridStep;
            if constexpr (bits == 6) {
                for (uint32_t j = 0; j < k; ++j) {
                    const int q = decoded[i * k + j];
                    const int code = std::clamp(4 + int(std::floor(float(q) / 21.7f)), 0, 7);
                    const int level = 2 * code - 7;
                    cy[i] += level * rotated[j];
                }
                cy[i] *= 10.85 * shelf::quantization::exl3::kExl3Int8GridStep;
            }
        }
        for (uint32_t i = 0; i < n; ++i) {
            double ref = 0.0;
            for (uint32_t j = 0; j < 128u; ++j) ref += y[(i / 128u) * 128u + j] * ((__builtin_popcount((i % 128u) & j) & 1) ? -1.0 : 1.0);
            ref /= std::sqrt(128.0);
            double delta = output[r * stride + i] - ref;
            error += delta * delta; norm += ref * ref;
            if constexpr (bits == 6) {
                double expected = 0;
                for (uint32_t j = 0; j < 128; ++j) {
                    const double sign = __builtin_popcount((i % 128) & j) & 1 ? -1 : 1;
                    expected += sign * cy[(i / 128) * 128 + j];
                }
                expected /= std::sqrt(128.0);
                const double diff = coarse[r * n + i] - expected;
                error += diff * diff;
                norm += expected * expected;
            }
        }
    }
    const double rel = std::sqrt(error / norm);
    std::printf("dflash2 mul24 head relative L2: %.6f\n", rel);
    return std::isfinite(rel) && rel < 0.02 && arena.shutdown().ok() ? 0 : 1;
}

int main() {
    return check_head<4, 256>() || check_head<6, 4096>();
}
