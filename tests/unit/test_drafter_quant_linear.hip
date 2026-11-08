#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/inference/execution/selectors/linear_selector.h>
#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/models/common/drafter/linear.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

template <typename T>
bool alloc(shelf::gpu::GpuArena& arena, std::size_t count, shelf::gpu::Tensor& out) {
    auto view = arena.allocate_aligned(count * sizeof(T), 128);
    if (!view.ok()) return false;
    auto t = shelf::gpu::Tensor::create<T>(view.value(), {count}, {1u});
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

float as_float(shelf::bf16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v.data) << 16u;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

int main() {
    const char* dir = std::getenv("SHELF_DRAFTER_QUANT_DIR");
    const char* reference = std::getenv("SHELF_DRAFTER_DEQUANT_DIR");
    if (!dir || !reference) { std::puts("SKIP: set SHELF_DRAFTER_QUANT_DIR and SHELF_DRAFTER_DEQUANT_DIR"); return 77; }
    const std::string name = "layers.0.self_attn.q_proj.weight";
    auto source = shelf::quantization::bundle::synthesize_hf_quantized_source(dir, {name}, false);
    if (!source.ok()) { std::printf("source: %s\n", source.status().message().c_str()); return 1; }
    auto reader = shelf::quantization::bundle::QuantizedModelReader::open_hf_source(source.release(), false);
    auto collection = shelf::weights::SafetensorsCollection::open(reference);
    auto arena_result = shelf::gpu::GpuArena::create(0, 512ull << 20u);
    if (!reader.ok() || !collection.ok() || !arena_result.ok()) return 1;
    auto arena = arena_result.release();
    auto quant = shelf::weights::load_quantized_matrix(reader.value(), name, arena, nullptr, {});
    auto ref = shelf::weights::load_bf16_matrix(collection.value(), name, arena, nullptr, {});
    if (!quant.ok() || !ref.ok()) {
        std::printf("load: %s %s\n", quant.ok() ? "ok" : quant.status().message().c_str(),
                    ref.ok() ? "ok" : ref.status().message().c_str());
        return 1;
    }
    const auto& w = quant.value();
    if (w.rows != ref.value().rows || w.cols != ref.value().cols) return 1;
    constexpr uint32_t max_rows = 16u;
    const uint32_t k = w.cols, n = w.rows;
    const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(k, max_rows);
    shelf::drafter::LinearWorkspace ws;
    shelf::gpu::Tensor x, y, expected;
    if (!alloc<int8_t>(arena, layout.codes_bytes(), ws.codes) ||
        !alloc<float>(arena, layout.scales_bytes() / sizeof(float), ws.scales) ||
        !alloc<float>(arena, max_rows * n * 4u, ws.accum) ||
        !alloc<shelf::bf16_t>(arena, max_rows * k, x) ||
        !alloc<shelf::bf16_t>(arena, max_rows * n, y) ||
        !alloc<shelf::bf16_t>(arena, max_rows * n, expected)) return 1;
    std::vector<shelf::bf16_t> input(max_rows * k);
    uint32_t rng = 12345u;
    for (auto& v : input) {
        rng = rng * 1664525u + 1013904223u;
        v.data = shelf::f32_to_bf16_rne((static_cast<float>(rng >> 8u) / 8388608.f) - 1.f);
    }
    if (hipMemcpy(x.data<void>(), input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice) != hipSuccess) return 1;
    for (uint32_t rows : {1u, 8u, 16u}) {
        auto st = shelf::drafter::quantized_linear(w, x.data<shelf::bf16_t>(), k,
            y.data<shelf::bf16_t>(), n, rows, ws, nullptr);
        if (!st.ok()) { std::printf("quantized linear: %s\n", st.message().c_str()); return 1; }
        shelf::execution::Bf16GemmSelectorInput args{rows, n, k, k, true, true, true, true};
        auto cfg = shelf::execution::select_bf16_gemm_config(args);
        if (!cfg || shelf::kernel::launch_gemm_bf16(*cfg, ref.value().bf16.data<shelf::bf16_t>(),
            x.data<shelf::bf16_t>(), expected.data<shelf::bf16_t>(),
            shelf::kernel::Bf16GemmOutputDType::BF16, rows, n, k, k, n, nullptr) != hipSuccess) return 1;
        std::vector<shelf::bf16_t> a(rows * n), b(rows * n);
        if (hipMemcpy(a.data(), y.data<void>(), a.size() * 2u, hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(b.data(), expected.data<void>(), b.size() * 2u, hipMemcpyDeviceToHost) != hipSuccess) return 1;
        double e = 0.0, norm = 0.0, max_diff = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double target = as_float(b[i]);
            const double diff = as_float(a[i]) - target;
            e += diff * diff; norm += target * target;
            max_diff = std::max(max_diff, std::abs(diff));
        }
        const double rel = std::sqrt(e / norm);
        std::printf("rows=%u relative_l2=%.8f max_abs=%.8f\n", rows, rel, max_diff);
        if (!std::isfinite(rel) || rel > 0.08) return 1;
    }
    return arena.shutdown().ok() ? 0 : 1;
}
