#include "support/resident_model_fixture.h"

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/resident/resident_client.h>
#include <phaseshift/weights/weight_loader.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool cond, const std::string& message) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", message.c_str());
    }
}

std::string model_dir_from_env() {
    const char* names[] = {"PHASESHIFT_MODEL_DIR",
                           "PHASESHIFT_MODEL_DIR_DFLASH2_TARGET"};
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            std::string dir(value);
            if (dir.size() > 1) {
                return dir;
            }
        }
    }
    return {};
}

std::uint64_t parse_stat(const std::string& text, const std::string& key) {
    const std::string needle = key + "=";
    std::size_t pos = text.find(needle);
    if (pos == std::string::npos) return 0;
    pos += needle.size();
    return std::strtoull(text.c_str() + pos, nullptr, 10);
}

bool contains_entry(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

}

int main() {
    if (!ps::resident::resident_socket_env_present()) {
        std::printf("SKIP: %s is not set\n", ps::resident::kSocketEnv);
        return 77;
    }
    const std::string model_dir = model_dir_from_env();
    if (model_dir.empty()) {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR\n");
        return 77;
    }
    if (!std::filesystem::exists(model_dir)) {
        std::printf("SKIP: model dir missing: %s\n", model_dir.c_str());
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto socket_result = ps::resident::resident_socket_path(0u);
    if (!socket_result.ok()) {
        std::printf("FAIL socket env: %s\n", socket_result.status().message().c_str());
        return 1;
    }
    const std::string& socket_path = socket_result.value();

    auto stats_before = ps::resident::resident_host_stats(socket_path);
    if (!stats_before.ok()) {
        std::printf("FAIL stats(before): %s\n", stats_before.status().message().c_str());
        return 1;
    }
    const std::uint64_t loads_before = parse_stat(stats_before.value(), "disk_load_count");
    const std::uint64_t attaches_before = parse_stat(stats_before.value(), "attach_count");

    ps::resident::ResidentTestRequest request;
    request.qwen_model_dir = model_dir;
    request.device = 0;
    auto fixture_result = ps::resident::ResidentModelFixture::acquire(request);
    if (!fixture_result.ok()) {
        std::printf("FAIL acquire: %s\n", fixture_result.status().message().c_str());
        return 1;
    }
    ps::resident::ResidentModelFixture fixture = fixture_result.release();
    check(fixture.enabled(), "resident fixture enabled");
    if (!fixture.enabled()) {
        std::printf("test_resident_model_ipc: failed=%d\n", g_fail);
        return 1;
    }
    check(fixture.resident_bytes() > 0, "resident bytes reported");

    auto stats_after = ps::resident::resident_host_stats(socket_path);
    check(stats_after.ok(), "stats(after) readable");
    if (stats_after.ok()) {
        const std::uint64_t loads_after = parse_stat(stats_after.value(), "disk_load_count");
        const std::uint64_t attaches_after = parse_stat(stats_after.value(), "attach_count");
        check(loads_after == loads_before,
              "disk load count unchanged by worker attach (" +
                  std::to_string(loads_before) + " -> " + std::to_string(loads_after) + ")");
        check(attaches_after == attaches_before + 1,
              "attach count incremented (" + std::to_string(attaches_before) + " -> " +
                  std::to_string(attaches_after) + ")");
        check(contains_entry(stats_after.value(), "resident_bytes="),
              "stats reports resident bytes");
    }

    auto model_result = fixture.take_qwen35();
    if (!model_result.ok()) {
        std::printf("FAIL take_qwen35: %s\n", model_result.status().message().c_str());
        std::printf("test_resident_model_ipc: failed=%d\n", g_fail);
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();

    auto config_result = ps::qwen35::read_qwen35_text_config(model_dir);
    check(config_result.ok(), "disk text config readable");
    if (config_result.ok()) {
        const auto& disk = config_result.value();
        const auto& resident = model.text_config();
        check(disk.vocab_size == resident.vocab_size, "vocab_size matches disk config");
        check(disk.hidden_size == resident.hidden_size, "hidden_size matches disk config");
        check(disk.num_hidden_layers == resident.num_hidden_layers,
              "num_hidden_layers matches disk config");
        check(disk.stop_tokens == resident.stop_tokens, "stop tokens match disk config");
        check(disk.layer_types == resident.layer_types, "layer types match disk config");
    }

    const auto& weights = model.weights();
    const auto* norm_base = static_cast<const std::byte*>(weights.final_norm_weight.allocation_base());
    check(weights.final_norm_weight.ndim() == 1, "final norm is 1D");
    check(norm_base != nullptr, "final norm allocation present");

    std::optional<ps::gpu::GpuArena> ephemeral;
    {
        auto arena_result = ps::gpu::GpuArena::create(0, 512ull * 1024ull * 1024ull);
        check(arena_result.ok(), "ephemeral arena created alongside resident block");
        if (arena_result.ok()) {
            ephemeral.emplace(arena_result.release());
            auto alloc = ephemeral->allocate_aligned(4ull * 1024ull * 1024ull, 256);
            check(alloc.ok(), "ephemeral allocation succeeded");
            if (alloc.ok()) {
                check(hipMemset(alloc.value().data(), 0x3c, 4ull * 1024ull * 1024ull) ==
                          hipSuccess,
                      "ephemeral memset succeeded");
            }
        }
    }

    std::vector<std::uint8_t> resident_bytes;
    const std::size_t norm_elems = weights.final_norm_weight.dim(0);
    const std::size_t norm_elem_size = weights.final_norm_weight.element_size();
    resident_bytes.resize(norm_elems * norm_elem_size);
    {
        std::size_t span = 0;
        const bool span_ok = weights.final_norm_weight.required_span_bytes(span).ok();
        check(span_ok && span == resident_bytes.size(), "resident norm span matches");
        hipError_t copy_error = hipMemcpy(resident_bytes.data(),
                                          weights.final_norm_weight.data<void>(), span,
                                          hipMemcpyDeviceToHost);
        check(copy_error == hipSuccess, "resident norm copied to host");
    }

    std::vector<std::uint8_t> reference_bytes;
    {
        hipStream_t stream = nullptr;
        check(hipStreamCreate(&stream) == hipSuccess, "reference stream");
        auto arena_result = ps::gpu::GpuArena::create(0, 64ull * 1024ull * 1024ull);
        check(arena_result.ok(), "reference arena created");
        if (arena_result.ok() && stream != nullptr) {
            ps::gpu::GpuArena arena = arena_result.release();
            ps::Result<ps::gpu::Tensor> loaded = [&]() -> ps::Result<ps::gpu::Tensor> {
                if (ps::weights::is_quantized_model_dir(model_dir)) {
                    auto reader = ps::quantization::bundle::QuantizedModelReader::open(model_dir, false);
                    if (!reader.ok()) return reader.status();
                    return ps::weights::load_quantized_small(
                        reader.value(), "model.language_model.norm.weight", arena, stream);
                }
                auto collection = ps::weights::SafetensorsCollection::open(model_dir);
                if (!collection.ok()) return collection.status();
                return ps::weights::load_bf16_tensor(
                    collection.value(), "model.language_model.norm.weight", arena, stream);
            }();
            check(loaded.ok(), "reference norm loaded from disk: " +
                                   (loaded.ok() ? "" : loaded.status().message()));
            if (loaded.ok()) {
                ps::gpu::Tensor ref = loaded.release();
                check(ref.dim(0) == norm_elems, "reference norm element count matches");
                check(ref.element_size() == norm_elem_size,
                      "reference norm element size matches");
                reference_bytes.resize(norm_elems * norm_elem_size);
                hipError_t copy_error = hipMemcpy(reference_bytes.data(),
                                                  ref.data<void>(), reference_bytes.size(),
                                                  hipMemcpyDeviceToHost);
                check(copy_error == hipSuccess, "reference norm copied to host");
            }
        }
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
        }
    }

    if (!reference_bytes.empty() && reference_bytes.size() == resident_bytes.size()) {
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < reference_bytes.size(); ++i) {
            if (reference_bytes[i] != resident_bytes[i]) ++mismatch;
        }
        check(mismatch == 0, "resident weight bytes identical to disk load (" +
                                 std::to_string(mismatch) + " of " +
                                 std::to_string(reference_bytes.size()) + " differ)");
    } else if (!reference_bytes.empty()) {
        check(false, "resident / reference norm sizes differ");
    }

    if (ephemeral.has_value()) {
        auto st = ephemeral->shutdown();
        check(st.ok(), "ephemeral arena shutdown");
    }

    std::printf("resident_bytes=%zu requested_arena=%zu\n", fixture.resident_bytes(),
                fixture.requested_arena_bytes());
    std::printf("test_resident_model_ipc: passed=%d failed=%d\n", 0, g_fail);
    return g_fail == 0 ? 0 : 1;
}
