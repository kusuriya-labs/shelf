#include <shelf/core/memory/arena.h>
#include <shelf/models/model_weights_factory.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace fs = std::filesystem;
namespace q35 = shelf::qwen35;
namespace moe = shelf::qwen35moe;
namespace models = shelf::models;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

std::uint64_t peak_rss_bytes() {
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) != 0)
        return 0;
#if defined(__GLIBC__)
    return static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ull;
#else
    return static_cast<std::uint64_t>(ru.ru_maxrss);
#endif
}

std::string model_dir_from_env() {
    for (const char* key : {"SHELF_MODEL_DIR_MOE", "SHELF_MODEL_DIR"}) {
        const char* value = std::getenv(key);
        if (value != nullptr && value[0] != '\0')
            return value;
    }
    return {};
}

}  // namespace

int main() {
    std::printf("=== qwen35moe real checkpoint weight load (GPU) ===\n");
    const std::string model_dir = model_dir_from_env();
    if (model_dir.empty()) {
        std::printf("SKIP: SHELF_MODEL_DIR_MOE / SHELF_MODEL_DIR is not set\n");
        return 77;
    }
    if (!fs::exists(fs::path(model_dir) / "config.json")) {
        std::printf("SKIP: no config.json in %s\n", model_dir.c_str());
        return 77;
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess)
        return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess)
        return 1;

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess)
        return 1;
    const std::uint64_t arena_bytes = free_bytes > (2ull << 20) ? free_bytes - (2ull << 20) : 0;
    auto arena = shelf::gpu::GpuArena::create(0, arena_bytes);
    check(arena.ok(), "arena create");
    if (!arena.ok())
        return 1;
    shelf::gpu::GpuArena gpu_arena = arena.release();
    std::printf("arena %llu GiB of %llu GiB free\n",
                static_cast<unsigned long long>(arena_bytes >> 30),
                static_cast<unsigned long long>(total_bytes >> 30));

    auto config = moe::read_qwen35_moe_text_config(model_dir);
    check(config.ok(), "read_qwen35_moe_text_config");
    if (!config.ok())
        return 1;
    std::printf("config hidden %zu moe_inter %zu shared_inter %zu experts %zu top_k %zu layers %zu\n",
                config.value().hidden_size, config.value().ffn.moe_intermediate_size,
                config.value().ffn.shared_expert_intermediate_size, config.value().ffn.num_experts,
                config.value().ffn.num_experts_per_tok, config.value().num_hidden_layers);

    auto probe = models::probe_model_dir(model_dir);
    check(probe.ok() && probe.value() == models::ModelFamily::Moe, "the factory picks Moe");

    const auto begin = std::chrono::steady_clock::now();
    auto loaded = models::load_model_weights(model_dir, gpu_arena, stream);
    const auto end = std::chrono::steady_clock::now();
    check(loaded.ok(), "load_model_weights on the real checkpoint");
    if (!loaded.ok()) {
        std::printf("load status: %s\n", loaded.status().message().c_str());
        gpu_arena.shutdown();
        hipStreamDestroy(stream);
        return 1;
    }
    const double seconds = std::chrono::duration<double>(end - begin).count();
    const models::LoadedModelWeights& w = loaded.value();

    check(q35::validate_qwen35_weights(w.body(), config.value()).ok(),
          "the MoE body passes the qwen35 model-level invariants");
    check(w.moe.body.layers.size() == config.value().num_hidden_layers, "layer count");
    check(w.moe.layers.size() == config.value().num_hidden_layers, "one MoE FFN per layer");

    std::uint64_t expert_bytes = 0;
    std::uint64_t fold_exact_layers = 0;
    std::uint64_t fold_exact_planes = 0;
    for (std::size_t l = 0; l < w.moe.layers.size(); ++l) {
        auto plan = moe::build_moe_expert_group_plan(config.value(), l);
        if (!plan.ok()) {
            check(false, "plan for layer " + std::to_string(l));
            break;
        }
        check(moe::moe_layer_weights_match_plan(w.moe.layers[l], plan.value()).ok(),
              "layer " + std::to_string(l) + " planes match the plan");
        expert_bytes += moe::moe_expert_group_bytes(plan.value());
        if (w.moe.layers[l].expert_w13.row_ref_exact)
            ++fold_exact_planes;
        if (w.moe.layers[l].expert_w2.row_ref_exact)
            ++fold_exact_planes;
        if (w.moe.layers[l].expert_w13.row_ref_exact && w.moe.layers[l].expert_w2.row_ref_exact)
            ++fold_exact_layers;
    }
    std::printf("load %.1f s, arena used %.3f GiB, expert planes %.3f GiB (plan sum), peak RSS %.2f GiB\n",
                seconds, static_cast<double>(gpu_arena.used()) / (1 << 30),
                static_cast<double>(expert_bytes) / (1 << 30),
                static_cast<double>(peak_rss_bytes()) / (1 << 30));
    std::printf("fold-exact planes: %llu / %llu (both planes in %llu layers)\n",
                static_cast<unsigned long long>(fold_exact_planes),
                static_cast<unsigned long long>(2ull * w.moe.layers.size()),
                static_cast<unsigned long long>(fold_exact_layers));
    check(gpu_arena.used() >= expert_bytes,
          "arena use covers the expert planes");

    gpu_arena.shutdown();
    if (hipStreamDestroy(stream) != hipSuccess)
        check(false, "stream destroy");

    std::printf("test_qwen35moe_model_load_real: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL",
                g_fail);
    return g_fail == 0 ? 0 : 1;
}
