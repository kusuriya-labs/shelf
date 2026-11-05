#include <shelf/core/memory/arena.h>
#include <shelf/io/safetensors_reader.h>
#include <shelf/models/model_weights_factory.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/weights/matrix_weight.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/resource.h>
#include <variant>
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

// One expert of a fused rank-3 plane, requantized here from the checkpoint's own bf16
// payload and preshuffled the way the dense per-expert path stores it. The resident
// plane must be byte-identical: the load's requantization is the same codec the
// checkpoint's other layers went through at export time.
bool fused_expert_matches(const shelf::weights::MatrixWeight& plane,
                          const std::uint16_t* expert_bf16, std::uint32_t rows,
                          std::uint32_t cols, std::uint32_t expert) {
    namespace qmx = shelf::quantization::mxfp4;

    std::vector<float> src(static_cast<std::size_t>(rows) * cols);
    for (std::size_t i = 0; i < src.size(); ++i) {
        const std::uint32_t word = static_cast<std::uint32_t>(expert_bf16[i]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &word, sizeof(value));
        src[i] = value;
    }
    const std::size_t codes_bytes = static_cast<std::size_t>(rows) * (plane.k_padded / 2u);
    const std::size_t scale_bytes = static_cast<std::size_t>(rows) * (plane.k_padded / 32u);
    std::vector<std::uint8_t> codes(codes_bytes), scales(scale_bytes);
    qmx::quantize_mxfp4_matrix(src.data(), cols, rows, cols, plane.k_padded, codes.data(),
                               scales.data());

    qmx::Mxfp4CanonicalView view;
    view.batch = 1u;
    view.n = rows;
    view.padded_k = plane.k_padded;
    view.codes = codes.data();
    view.codes_bytes = codes_bytes;
    view.scales = scales.data();
    view.scales_bytes = scale_bytes;
    if (!view.validate())
        return false;
    qmx::Mxfp4NativeHost native;
    if (!qmx::preshuffle_mxfp4_native(view, native))
        return false;

    std::vector<std::uint8_t> resident(native.codes.size());
    const std::uint8_t* at = plane.codes.data<std::uint8_t>() +
                             static_cast<std::size_t>(expert) * plane.codes_batch_stride_bytes;
    if (hipMemcpy(resident.data(), at, resident.size(), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (resident != native.codes)
        return false;
    resident.assign(native.scales.size(), 0u);
    if (hipMemcpy(resident.data(),
                  plane.scales.data<std::uint8_t>() +
                      static_cast<std::size_t>(expert) * plane.scales_batch_stride_bytes,
                  resident.size(), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (resident != native.scales)
        return false;
    resident.assign(native.row_ref.size(), 0u);
    if (hipMemcpy(resident.data(),
                  plane.row_ref.data<std::uint8_t>() + static_cast<std::size_t>(expert) * rows,
                  resident.size(), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    return resident == native.row_ref;
}

// Path of the shard that stores a logical tensor, empty when the checkpoint has none.
std::string shard_holding(const fs::path& dir, const std::string& name) {
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.path().extension() != ".safetensors")
            continue;
        auto reader = shelf::io::SafetensorsReader::open(entry.path().string());
        if (!reader.ok())
            continue;
        if (reader.value().tensor_spec(name).ok())
            return entry.path().string();
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
    q35::Qwen35LoadOptions options;
    const std::string mtp_shard =
        shard_holding(fs::path(model_dir), "mtp.layers.0.mlp.experts.gate_up_proj");
    options.load_mtp_layers = !mtp_shard.empty();
    std::printf("checkpoint MTP block: %s\n", mtp_shard.empty() ? "absent" : mtp_shard.c_str());
    auto loaded = models::load_model_weights(model_dir, gpu_arena, stream, options);
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
    check(models::model_family(w) == models::ModelFamily::Moe, "the factory reports MoE");
    const auto* m = std::get_if<moe::Qwen35MoeModelWeights>(&w);
    check(m != nullptr, "the factory holds MoE weights only");
    if (m == nullptr) {
        gpu_arena.shutdown();
        hipStreamDestroy(stream);
        return 1;
    }
    check(std::get_if<q35::Qwen35ModelWeights>(&w) == nullptr,
          "the factory holds no dense weights for a MoE checkpoint");
    check(moe::validate_qwen35moe_weights(*m, config.value()).ok(),
          "the MoE model passes its own invariants");
    check(m->layers.size() == config.value().num_hidden_layers,
          "one MoE layer per transformer layer");

    std::uint64_t expert_bytes = 0;
    std::uint64_t fold_exact_layers = 0;
    std::uint64_t fold_exact_planes = 0;
    for (std::size_t l = 0; l < m->layers.size(); ++l) {
        auto plan = moe::build_moe_expert_group_plan(config.value(), l);
        if (!plan.ok()) {
            check(false, "plan for layer " + std::to_string(l));
            break;
        }
        check(moe::moe_ffn_weights_match(m->layers[l].ffn, plan.value(), config.value().ffn).ok(),
              "layer " + std::to_string(l) + " FFN matches the plan and config");
        expert_bytes += moe::moe_expert_group_bytes(plan.value());
        if (m->layers[l].ffn.expert_w13.row_ref_exact)
            ++fold_exact_planes;
        if (m->layers[l].ffn.expert_w2.row_ref_exact)
            ++fold_exact_planes;
        if (m->layers[l].ffn.expert_w13.row_ref_exact && m->layers[l].ffn.expert_w2.row_ref_exact)
            ++fold_exact_layers;
    }
    std::uint64_t mtp_bytes = 0;
    if (mtp_shard.empty()) {
        check(!m->mtp.present, "a checkpoint without an MTP block leaves it absent");
    } else {
        check(m->mtp.present, "the real checkpoint's MTP block loads");
        auto mtp_plan = moe::build_moe_mtp_expert_group_plan(config.value());
        check(mtp_plan.ok(), "the MTP expert plan builds for the real config");
        if (mtp_plan.ok()) {
            const moe::MoeExpertGroupPlan& mp = mtp_plan.value();
            check(mp.experts == config.value().ffn.num_experts &&
                      mp.w13_rows == 2u * config.value().ffn.moe_intermediate_size &&
                      mp.w2_rows == static_cast<std::uint32_t>(config.value().hidden_size),
                  "the MTP plan follows the checkpoint's expert geometry");
            check(moe::validate_qwen35moe_mtp_weights(m->mtp, mp, config.value()).ok(),
                  "the MTP block passes its own invariants");
            mtp_bytes = moe::moe_expert_group_bytes(mp);

            auto shard = shelf::io::SafetensorsReader::open(mtp_shard);
            check(shard.ok(), "the MTP shard opens");
            if (shard.ok()) {
                std::size_t w13_len = 0;
                std::size_t w2_len = 0;
                auto fused_w13 = shard.value().tensor_data(mp.w13_name, w13_len);
                auto fused_w2 = shard.value().tensor_data(mp.w2_name, w2_len);
                check(fused_w13.ok() && fused_w2.ok(), "the fused MTP planes resolve");
                if (fused_w13.ok() && fused_w2.ok()) {
                    const auto* w13 = static_cast<const std::uint16_t*>(fused_w13.value());
                    const auto* w2 = static_cast<const std::uint16_t*>(fused_w2.value());
                    const std::uint32_t samples[] = {0u, 1u, mp.experts / 2u, mp.experts - 1u};
                    bool w13_ok = true;
                    bool w2_ok = true;
                    for (const std::uint32_t e : samples) {
                        if (!fused_expert_matches(m->mtp.layer.ffn.expert_w13,
                                                  w13 + static_cast<std::size_t>(e) * mp.w13_rows *
                                                           mp.w13_cols,
                                                  mp.w13_rows, mp.w13_cols, e))
                            w13_ok = false;
                        if (!fused_expert_matches(m->mtp.layer.ffn.expert_w2,
                                                  w2 + static_cast<std::size_t>(e) * mp.w2_rows *
                                                           mp.w2_cols,
                                                  mp.w2_rows, mp.w2_cols, e))
                            w2_ok = false;
                    }
                    check(w13_ok, "the MTP w13 plane is the checkpoint payload requantized");
                    check(w2_ok, "the MTP w2 plane is the checkpoint payload requantized");
                }
            }
        }
    }

    std::printf("load %.1f s, arena used %.3f GiB, expert planes %.3f GiB (plan sum), peak RSS %.2f GiB\n",
                seconds, static_cast<double>(gpu_arena.used()) / (1 << 30),
                static_cast<double>(expert_bytes) / (1 << 30),
                static_cast<double>(peak_rss_bytes()) / (1 << 30));
    std::printf("fold-exact planes: %llu / %llu (both planes in %llu layers)\n",
                static_cast<unsigned long long>(fold_exact_planes),
                static_cast<unsigned long long>(2ull * m->layers.size()),
                static_cast<unsigned long long>(fold_exact_layers));
    if (mtp_bytes != 0u)
        std::printf("MTP expert planes %.3f GiB (plan sum), fc %llu x %llu, "
                    "w13 row_ref_exact=%d w2 row_ref_exact=%d\n",
                    static_cast<double>(mtp_bytes) / (1 << 30),
                    static_cast<unsigned long long>(m->mtp.fc.rows),
                    static_cast<unsigned long long>(m->mtp.fc.cols),
                    m->mtp.layer.ffn.expert_w13.row_ref_exact ? 1 : 0,
                    m->mtp.layer.ffn.expert_w2.row_ref_exact ? 1 : 0);
    check(gpu_arena.used() >= expert_bytes + mtp_bytes,
          "arena use covers the expert planes");

    gpu_arena.shutdown();
    if (hipStreamDestroy(stream) != hipSuccess)
        check(false, "stream destroy");

    std::printf("test_qwen35moe_model_load_real: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL",
                g_fail);
    return g_fail == 0 ? 0 : 1;
}
