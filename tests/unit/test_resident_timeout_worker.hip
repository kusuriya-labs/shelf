#include "support/resident_model_fixture.h"

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/resident/resident_client.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>

namespace {

std::string model_dir_from_env() {
    const char* names[] = {"PHASESHIFT_MODEL_DIR",
                           "PHASESHIFT_MODEL_DIR_DFLASH2_TARGET"};
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return std::string(value);
        }
    }
    return {};
}

}

int main() {
    if (!ps::resident::resident_socket_env_present()) {
        std::printf("SKIP: %s is not set\n", ps::resident::kSocketEnv);
        return 77;
    }
    const std::string model_dir = model_dir_from_env();
    if (model_dir.empty() || !std::filesystem::exists(model_dir)) {
        std::printf("SKIP: model dir missing\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    const char* stats_env = std::getenv("PHASESHIFT_TEST_STATS_SOCKET");
    std::uint64_t loads_before = 0;
    if (stats_env != nullptr && stats_env[0] != '\0') {
        auto stats = ps::resident::resident_host_stats(stats_env);
        if (stats.ok()) {
            const std::string needle = "disk_load_count=";
            const std::size_t pos = stats.value().find(needle);
            if (pos != std::string::npos) {
                loads_before = std::strtoull(stats.value().c_str() + pos + needle.size(),
                                             nullptr, 10);
            }
        }
    }

    ps::resident::ResidentTestRequest request;
    request.qwen_model_dir = model_dir;
    request.device = 0;
    auto fixture_result = ps::resident::ResidentModelFixture::acquire(request);
    if (!fixture_result.ok()) {
        std::printf("FAIL resident acquire: %s\n",
                    fixture_result.status().message().c_str());
        return 1;
    }
    ps::resident::ResidentModelFixture fixture = fixture_result.release();
    if (!fixture.enabled()) {
        std::printf("FAIL resident fixture disabled\n");
        return 1;
    }

    std::optional<ps::gpu::GpuArena> ephemeral;
    auto arena_result = ps::gpu::GpuArena::create(0, 512ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("FAIL ephemeral arena: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    ephemeral.emplace(arena_result.release());
    auto alloc = ephemeral->allocate_aligned(8ull * 1024ull * 1024ull, 256);
    if (!alloc.ok()) {
        std::printf("FAIL ephemeral allocation: %s\n", alloc.status().message().c_str());
        return 1;
    }
    if (hipMemset(alloc.value().data(), 0x11, 8ull * 1024ull * 1024ull) != hipSuccess) {
        std::printf("FAIL ephemeral memset\n");
        return 1;
    }

    std::uint64_t loads_after = loads_before;
    if (stats_env != nullptr && stats_env[0] != '\0') {
        auto stats = ps::resident::resident_host_stats(stats_env);
        if (stats.ok()) {
            const std::string needle = "disk_load_count=";
            const std::size_t pos = stats.value().find(needle);
            if (pos != std::string::npos) {
                loads_after = std::strtoull(stats.value().c_str() + pos + needle.size(),
                                            nullptr, 10);
            }
        }
    }

    auto model_result = fixture.take_qwen35();
    if (!model_result.ok()) {
        std::printf("FAIL take_qwen35: %s\n", model_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();

    std::printf("RESIDENT_WORKER_ATTACHED resident_bytes=%zu disk_load_count=%llu "
                "layers=%zu\n",
                fixture.resident_bytes(),
                static_cast<unsigned long long>(loads_after),
                model.text_config().num_hidden_layers);
    std::fflush(stdout);

    const char* sleep_env = std::getenv("PHASESHIFT_TEST_SLEEP_SEC");
    const long seconds =
        sleep_env != nullptr && sleep_env[0] != '\0' ? std::strtol(sleep_env, nullptr, 10)
                                                     : 3600;
    std::printf("RESIDENT_WORKER_SLEEP %ld\n", seconds);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(seconds > 0 ? seconds : 1));

    if (ephemeral.has_value()) {
        auto st = ephemeral->shutdown();
        if (!st.ok()) {
            std::printf("FAIL ephemeral shutdown: %s\n", st.message().c_str());
            return 1;
        }
    }
    std::printf("test_resident_timeout_worker: completed without timeout\n");
    return 0;
}
