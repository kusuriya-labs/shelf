#include <shelf/models/model_weights_factory.h>
#include <shelf/models/plamo3/model/config.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    const char* dir = std::getenv("SHELF_PLAMO3_MODEL_DIR");
    if (!dir) { std::puts("SKIP: SHELF_PLAMO3_MODEL_DIR is not set"); return 77; }
    if (hipSetDevice(0) != hipSuccess) return 1;
    auto arena = shelf::gpu::GpuArena::create(0, 3ull << 30);
    if (!arena.ok()) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto weights = shelf::models::load_model_weights(dir, arena.value(), stream);
    if (!weights.ok()) { std::fprintf(stderr, "%s\n", weights.status().message().c_str()); return 1; }
    auto* w = std::get_if<shelf::plamo3::ModelWeights>(&weights.value());
    if (!w || w->layers.size() != 16 || w->layers[0].qkv.rows != 2304 ||
        w->layers[0].qkv.cols != 1536 || w->layers[0].gate_up.rows != 12288) return 1;
    auto collection = shelf::weights::SafetensorsCollection::open(dir);
    if (!collection.ok()) return 1;
    std::size_t bytes = 0;
    auto expected = collection.value().tensor_data("model.layers.layers.0.mixer.qkv_proj.weight", bytes);
    if (!expected.ok() || bytes < 64) return 1;
    char actual[64];
    if (hipStreamSynchronize(stream) != hipSuccess ||
        hipMemcpy(actual, w->layers[0].qkv.bf16.data(), sizeof(actual), hipMemcpyDeviceToHost) != hipSuccess ||
        std::memcmp(actual, expected.value(), sizeof(actual)) != 0) return 1;
    if (hipStreamDestroy(stream) != hipSuccess) return 1;
    std::puts("PLaMo 0.8B BF16 weights loaded and QKV payload verified");
    return 0;
}
