#include <shelf/core/memory/arena.h>
#include <shelf/inference/execution/weight_slot_validation.h>
#include <shelf/io/safetensors_reader.h>
#include <shelf/models/common/qwen_hybrid/weight_slot.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/weights/ple_table.h>
#include <shelf/models/qwen4exp/weights/qwen4exp_weights.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::fprintf(stderr, "ok: %s\n", what);
    }
}

float from_bf16_bits(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

std::string shard_for_tensor(const std::string& model_dir, const std::string& tensor) {
    std::ifstream file(model_dir + "/model.safetensors.index.json");
    if (!file.is_open()) return {};
    nlohmann::json index;
    try {
        file >> index;
    } catch (const nlohmann::json::exception&) {
        return {};
    }
    if (!index.contains("weight_map") || !index["weight_map"].contains(tensor)) return {};
    return index["weight_map"][tensor].get<std::string>();
}

bool compare_plane_prefix(const std::string& model_dir, const std::string& tensor,
                          const std::uint8_t* device_base, std::size_t bytes,
                          hipStream_t stream) {
    const std::string shard = shard_for_tensor(model_dir, tensor);
    if (shard.empty()) {
        std::fprintf(stderr, "missing index entry: %s\n", tensor.c_str());
        return false;
    }
    auto reader = shelf::io::SafetensorsReader::open(model_dir + "/" + shard);
    if (!reader.ok()) {
        std::fprintf(stderr, "cannot open shard %s\n", shard.c_str());
        return false;
    }
    std::size_t source_bytes = 0;
    auto source = reader.value().tensor_data(tensor, source_bytes);
    if (!source.ok() || source_bytes < bytes) {
        std::fprintf(stderr, "cannot read source tensor %s\n", tensor.c_str());
        return false;
    }
    std::vector<std::uint8_t> host(bytes);
    const hipError_t error = hipMemcpy(host.data(), device_base, bytes, hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        std::fprintf(stderr, "device copy failed: %s\n", hipGetErrorString(error));
        return false;
    }
    return std::memcmp(host.data(), source.value(), bytes) == 0;
}

bool compare_host_plane_prefix(const std::string& model_dir, const std::string& tensor,
                               const std::uint8_t* host_base, std::size_t bytes) {
    const std::string shard = shard_for_tensor(model_dir, tensor);
    if (shard.empty()) {
        std::fprintf(stderr, "missing index entry: %s\n", tensor.c_str());
        return false;
    }
    auto reader = shelf::io::SafetensorsReader::open(model_dir + "/" + shard);
    if (!reader.ok()) return false;
    std::size_t source_bytes = 0;
    auto source = reader.value().tensor_data(tensor, source_bytes);
    if (!source.ok() || source_bytes < bytes) return false;
    return std::memcmp(host_base, source.value(), bytes) == 0;
}

void verify_expert_sample(const std::string& model_dir, const shelf::qwen4exp::Qwen4ExpRankWeights& w,
                          std::uint32_t rank, hipStream_t stream) {
    const auto& layer0 = w.layers[0];
    const auto& gate = layer0.mlp.gate_experts;
    const auto& gate_host = layer0.mlp.gate_experts_host;
    const auto& range = layer0.mlp.experts;
    check(gate.experts + gate_host.experts == range.local_experts,
          "MOE expert split covers the local expert range");
    if (gate.encoding != shelf::weights::MatrixEncoding::Exl3) {
        if (gate_host.experts != 0u) {
            check(gate_host.host_resident && gate_host.host_codes != nullptr &&
                      gate_host.host_scales != nullptr &&
                      gate_host.host_row_ref != nullptr &&
                      w.expert_slot_map.ndim() != 0u,
                  "MOE host expert planes are resident");
        }
        return;
    }
    if (gate_host.experts != 0u) {
        check(gate_host.host_resident && gate_host.host_codes != nullptr &&
                  gate_host.host_scales != nullptr && gate_host.host_row_ref != nullptr &&
                  w.expert_slot_map.ndim() != 0u,
              "MOE EXL3 host expert planes are resident");
    }
    const std::uint64_t sample_bytes =
        std::min<std::uint64_t>(4096u, gate.codes_batch_stride_bytes);
    const std::uint32_t first_expert = range.expert_begin;
    const std::uint32_t last_vram_expert = range.expert_begin + gate.experts - 1u;
    const std::uint32_t last_expert = range.expert_begin + range.local_experts - 1;
    const std::string first_name =
        "model.language_model.layers.0.mlp.experts." + std::to_string(first_expert) +
        ".gate_proj.trellis";
    const std::string last_vram_name =
        "model.language_model.layers.0.mlp.experts." + std::to_string(last_vram_expert) +
        ".gate_proj.trellis";
    const std::string last_host_name =
        "model.language_model.layers.0.mlp.experts." + std::to_string(last_expert) +
        ".gate_proj.trellis";
    char label[128];
    std::snprintf(label, sizeof(label), "rank%u first expert trellis matches source", rank);
    check(compare_plane_prefix(model_dir, first_name, gate.codes.data<std::uint8_t>(),
                               sample_bytes, stream),
          label);
    std::snprintf(label, sizeof(label), "rank%u last VRAM expert trellis matches source", rank);
    check(compare_plane_prefix(model_dir, last_vram_name,
                               gate.codes.data<std::uint8_t>() +
                                   static_cast<std::uint64_t>(gate.experts - 1u) *
                                       gate.codes_batch_stride_bytes,
                               sample_bytes, stream),
          label);
    if (gate_host.experts != 0u) {
        std::snprintf(label, sizeof(label), "rank%u last host expert trellis matches source",
                      rank);
        check(compare_host_plane_prefix(
                  model_dir, last_host_name,
                  static_cast<const std::uint8_t*>(gate_host.host_codes) +
                      static_cast<std::uint64_t>(gate_host.experts - 1u) *
                          gate_host.codes_batch_stride_bytes,
                  sample_bytes),
              label);
    }
}

void verify_int6_source_rows(const std::string& model_dir,
                             const shelf::qwen4exp::PleTable& table,
                             const shelf::qwen4exp::Qwen4ExpTextConfig& config) {
    const std::uint64_t rows_to_check[] = {
        0u, 1u, 123456u, 2500011u, 2500012u, table.rows() - 1u};
    const std::vector<std::size_t> ple_layers = config.ple_layers();
    if (ple_layers.size() != 1u) return;
    for (const std::uint64_t row : rows_to_check) {
        const std::uint64_t shard = row / table.shard_rows();
        const std::string name =
            "model.language_model.layers." + std::to_string(ple_layers[0]) +
            ".ple.ple_embedding.ngram_embedding.shard_" + std::to_string(shard) +
            ".weight";
        const std::string file = shard_for_tensor(model_dir, name);
        if (file.empty()) {
            std::fprintf(stderr, "missing PLE shard index entry: %s\n", name.c_str());
            ++failures;
            continue;
        }
        auto reader = shelf::io::SafetensorsReader::open(model_dir + "/" + file);
        if (!reader.ok()) {
            std::fprintf(stderr, "cannot open PLE shard %s\n", name.c_str());
            ++failures;
            continue;
        }
        std::size_t bytes = 0;
        auto data = reader.value().tensor_data(name, bytes);
        if (!data.ok() ||
            bytes != static_cast<std::size_t>(table.shard_rows()) * 160u * 2u) {
            std::fprintf(stderr, "cannot read PLE shard %s\n", name.c_str());
            ++failures;
            continue;
        }
        const auto* source = static_cast<const shelf::bf16_t*>(data.value()) +
                             (row % table.shard_rows()) * 160u;
        auto decoded = table.decode_row(row, false);
        if (!decoded.ok() || decoded.value().size() != 160u) {
            std::fprintf(stderr, "PLE INT6 decode failed for row %llu\n",
                         static_cast<unsigned long long>(row));
            ++failures;
            continue;
        }
        bool bounded = true;
        for (std::uint32_t group = 0; group < 5u; ++group) {
            float amax = 0.0f;
            for (std::uint32_t i = 0; i < 32u; ++i) {
                amax = std::max(amax, std::fabs(from_bf16_bits(
                    source[group * 32u + i].data)));
            }
            for (std::uint32_t i = 0; i < 32u; ++i) {
                const float expected =
                    from_bf16_bits(source[group * 32u + i].data);
                const float error =
                    std::fabs(decoded.value()[group * 32u + i] - expected);
                bounded = bounded && error <= amax / 62.0f + amax / 1000.0f + 1e-6f;
            }
        }
        check(bounded, "PLE INT6 decode stays within the W32 error bound");
    }
}

void verify_ple(const std::string& model_dir, const shelf::qwen4exp::PleTable& table,
                const shelf::qwen4exp::Qwen4ExpTextConfig& config) {
    check(table.rows() == 320001536u, "PLE row count is 320001536");
    check(table.row_width() == 160u, "PLE row width is 160");
    check(table.head_count() == 16u, "PLE head count is 16");
    const bool int6 = table.int6_w32_fp16();
    if (int6) {
        check(table.row_bytes() == 130u, "PLE INT6 W32 row stride is 130 bytes");
        check(table.shard_count() == 128u, "PLE shard count is 128");
        verify_int6_source_rows(model_dir, table, config);
    }

    auto row = table.decode_row(0u);
    check(row.ok() && row.value().size() == 160u, "PLE row decodes to 160 values");
    if (row.ok()) {
        bool finite = true;
        float rms = 0.0f;
        for (const float value : row.value()) {
            finite = finite && std::isfinite(value);
            rms += value * value;
        }
        rms = std::sqrt(rms / static_cast<float>(row.value().size()));
        std::fprintf(stderr, "PLE row 0 rms=%g\n", static_cast<double>(rms));
        check(finite && rms > 0.0f, "PLE row 0 is finite and non-zero");
    }

    std::vector<std::int32_t> history = {
        config.ple.eos_token_id, config.ple.eos_token_id, 100, 200, 300};
    const std::vector<std::uint64_t> expected_rows = {
        6129193u,  20484628u, 57667260u, 77728366u, 86357215u, 109234597u,
        123551585u, 149309416u, 172664702u, 184433127u, 201501905u, 227445817u,
        243341740u, 279238932u, 284538191u, 311036336u};
    auto rows = table.ngram_rows(history, 1u);
    check(rows.ok() && rows.value() == expected_rows,
          "PLE n-gram hash rows match the reference");

    auto gathered = table.gather(history, 3u);
    check(gathered.ok() && gathered.value().size() == 3u * 16u * 160u,
          "PLE 16-row gather has 2560 values per token");
    if (gathered.ok()) {
        bool finite = true;
        for (const float value : gathered.value()) finite = finite && std::isfinite(value);
        check(finite, "PLE gather is finite");
    }
}

}  // namespace

int main(int argc, char** argv) {
#ifdef SHELF_PLE_TIERED_TEST
    ::setenv("SHELF_PLE_HOT_CACHE_GIB", "1", 1);
#endif
#ifdef SHELF_REPO_DIR
    std::string model_dir = std::string(SHELF_REPO_DIR) +
        "/models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw";
#else
    std::string model_dir = "models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw";
#endif
    if (const char* env = std::getenv("SHELF_QWEN4EXP_MODEL_DIR")) {
        model_dir = env;
    } else if (const char* home = std::getenv("HOME")) {
        const std::string candidate =
            std::string(home) + "/work/shelf/models/Qwen3.8-Flash-Next-exl3";
        if (::access(candidate.c_str(), R_OK) == 0) model_dir = candidate;
    }
    bool skip_ple = false;
#ifdef SHELF_PLE_TIERED_TEST
    bool skip_gpu = true;
#else
    bool skip_gpu = false;
#endif
    int single_rank = -1;
    int single_device = -1;
    std::uint32_t arena_gib = 29;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--skip-ple") == 0) {
            skip_ple = true;
        } else if (std::strcmp(argv[i], "--skip-gpu") == 0) {
            skip_gpu = true;
        } else if (std::strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
            single_rank = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            single_device = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--arena-gib") == 0 && i + 1 < argc) {
            arena_gib = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
    }

    auto config_result = shelf::qwen4exp::read_qwen4exp_text_config(model_dir);
    if (!config_result.ok()) {
        std::fprintf(stderr, "config: %s\n", config_result.status().message().c_str());
        return 1;
    }
    const shelf::qwen4exp::Qwen4ExpTextConfig& config = config_result.value();
    std::fprintf(stderr,
                 "model: layers=%zu hidden=%zu experts=%zu topk=%zu hc=%zu PLE layers=%zu\n",
                 config.num_hidden_layers, config.hidden_size, config.ffn.num_experts,
                 config.ffn.num_experts_per_tok, config.hc_count, config.ple.layer_ids.size());

    shelf::qwen4exp::PleTable ple;
    if (!skip_ple) {
        auto ple_result = shelf::qwen4exp::PleTable::load(model_dir, config, true);
        if (!ple_result.ok()) {
            std::fprintf(stderr, "PLE load: %s\n", ple_result.status().message().c_str());
            return 1;
        }
        ple = ple_result.release();
        std::fprintf(stderr, "PLE resident: %.3f GiB\n",
                     static_cast<double>(ple.payload_bytes()) / 1073741824.0);
        verify_ple(model_dir, ple, config);
#ifdef SHELF_PLE_TIERED_TEST
        for (std::uint64_t index = 0; index < 20000u; ++index) {
            const std::uint64_t row = (index * 12347u) % ple.rows();
            auto decoded = ple.decode_row(row, false);
            if (!decoded.ok()) {
                std::fprintf(stderr, "tiered decode failed at row %llu\n",
                             static_cast<unsigned long long>(row));
                return 1;
            }
        }
        check(ple.cached_blocks() != 0u, "PLE tiered hot cache holds blocks");
        check(ple.host_bytes() < ple.payload_bytes(), "PLE tiered host bytes are bounded");
#endif
    }

    if (skip_gpu) {
        std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
        return failures == 0 ? 0 : 1;
    }

    const int rank_count = single_rank >= 0 ? 1 : 2;
    int ranks[2] = {0, 1};
    int devices[2] = {0, 1};
    if (single_rank >= 0) {
        ranks[0] = single_rank;
        devices[0] = single_device >= 0 ? single_device : single_rank;
    }

    std::optional<shelf::gpu::GpuArena> arenas[2];
    hipStream_t streams[2] = {nullptr, nullptr};
    for (int index = 0; index < rank_count; ++index) {
        hipError_t error = hipSetDevice(devices[index]);
        check(error == hipSuccess, "hipSetDevice");
        error = hipStreamCreate(&streams[index]);
        check(error == hipSuccess, "hipStreamCreate");
        const std::size_t capacity =
            static_cast<std::size_t>(arena_gib) * 1024ull * 1024ull * 1024ull;
        auto arena = shelf::gpu::GpuArena::create(devices[index], capacity);
        if (!arena.ok()) {
            std::fprintf(stderr, "arena: %s\n", arena.status().message().c_str());
            return 1;
        }
        arenas[index].emplace(arena.release());
    }

    for (int index = 0; index < rank_count; ++index) {
        const std::uint32_t rank = static_cast<std::uint32_t>(ranks[index]);
        shelf::qwen4exp::Qwen4ExpLoadOptions options;
        options.tp_size = 2;
        options.tp_rank = rank;
        if (const char* host_experts = std::getenv("SHELF_MOE_HOST_EXPERTS")) {
            options.moe_host_experts =
                static_cast<std::uint32_t>(std::strtoul(host_experts, nullptr, 10));
        }
        auto loaded = shelf::qwen4exp::load_qwen4exp_rank(model_dir, *arenas[index],
                                                          streams[index], options);
        if (!loaded.ok()) {
            std::fprintf(stderr, "rank%u load: %s\n", rank,
                         loaded.status().message().c_str());
            return 1;
        }
        const auto& weights = loaded.value();
        std::fprintf(stderr, "rank%u resident weights: %.3f GiB (arena used %.3f GiB)\n",
                     rank, static_cast<double>(weights.resident_bytes()) / 1073741824.0,
                     static_cast<double>(arenas[index]->used()) / 1073741824.0);
        verify_expert_sample(model_dir, weights, rank, streams[index]);
        auto validation = shelf::qwen4exp::validate_qwen4exp_rank(weights, config, 2, rank);
        check(validation.ok(), "rank weight invariants");
        const auto expert_slot = shelf::qwen_hybrid::make_weight_slot(
            weights.layers[0].mlp.gate_experts);
        check(shelf::execution::validate_host_weight_slot(expert_slot, 0u).ok(),
              "EXL3 expert plane forms a rank-3 weight slot");
        if (weights.layers[0].mlp.gate_experts_host.experts != 0u) {
            const auto host_expert_slot = shelf::qwen_hybrid::make_weight_slot(
                weights.layers[0].mlp.gate_experts_host);
            const auto host_validation =
                shelf::execution::validate_host_weight_slot(host_expert_slot, 0u);
            if (!host_validation.ok())
                std::fprintf(stderr, "host slot validation: %s\n",
                             host_validation.message().c_str());
            check(host_validation.ok(),
                  "EXL3 host expert plane forms a rank-3 weight slot");
        }
        if (const hipError_t error = hipStreamSynchronize(streams[index]);
            error != hipSuccess) {
            std::fprintf(stderr, "stream sync: %s\n", hipGetErrorString(error));
            ++failures;
        }
    }

    for (int index = 0; index < rank_count; ++index) {
        if (streams[index] != nullptr) {
            (void)hipStreamDestroy(streams[index]);
        }
        const auto shutdown = arenas[index]->shutdown();
        check(shutdown.ok(), "arena shutdown");
    }

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
