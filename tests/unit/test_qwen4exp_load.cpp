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

void verify_expert_sample(const std::string& model_dir, const shelf::qwen4exp::Qwen4ExpRankWeights& w,
                          std::uint32_t rank, hipStream_t stream) {
    const auto& layer0 = w.layers[0];
    const auto& gate = layer0.mlp.gate_experts;
    const std::uint64_t sample_bytes =
        std::min<std::uint64_t>(4096u, gate.trellis_bytes_per_expert);
    const std::uint32_t first_expert = gate.expert_begin;
    const std::uint32_t last_expert = gate.expert_begin + gate.local_experts - 1;
    const std::string first_name =
        "model.language_model.layers.0.mlp.experts." + std::to_string(first_expert) +
        ".gate_proj.trellis";
    const std::string last_name =
        "model.language_model.layers.0.mlp.experts." + std::to_string(last_expert) +
        ".gate_proj.trellis";
    char label[128];
    std::snprintf(label, sizeof(label), "rank%u first expert trellis matches source", rank);
    check(compare_plane_prefix(model_dir, first_name, gate.codes.data<std::uint8_t>(),
                               sample_bytes, stream),
          label);
    std::snprintf(label, sizeof(label), "rank%u last expert trellis matches source", rank);
    check(compare_plane_prefix(model_dir, last_name,
                               gate.codes.data<std::uint8_t>() +
                                   static_cast<std::uint64_t>(gate.local_experts - 1u) *
                                       gate.trellis_bytes_per_expert,
                               sample_bytes, stream),
          label);
}

void verify_ple(const std::string& model_dir, const shelf::qwen4exp::PleTable& table,
                const shelf::qwen4exp::Qwen4ExpTextConfig& config) {
    check(table.rows() == 320001536u, "PLE row count is 320001536");
    check(table.row_width() == 160u, "PLE row width is 160");
    check(table.k_bits() == 8u, "PLE trellis K is 8");
    check(table.row_bytes() == 162u, "PLE row stride is 162 bytes");
    check(table.shard_count() == 128u, "PLE shard count is 128");
    check(table.head_count() == 16u, "PLE head count is 16");

    const std::uint64_t rows_to_check[] = {
        0u, 1u, 123456u, 2500011u, 2500012u, 123456789u, table.rows() - 1u};
    for (const std::uint64_t row : rows_to_check) {
        auto source = table.read_source_row(model_dir, row);
        if (!source.ok()) {
            std::fprintf(stderr, "PLE source row %llu failed\n",
                         static_cast<unsigned long long>(row));
            ++failures;
            continue;
        }
        const std::uint8_t* resident = table.row_ptr(row);
        check(resident != nullptr && std::memcmp(resident, source.value().data(),
                                                 table.row_bytes()) == 0,
              "PLE packaged row matches source");
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
    std::string model_dir = "/mnt/Qwen3.8-Flash-Next-exl3";
    bool skip_ple = false;
    bool skip_gpu = false;
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
        const auto expert_matrix =
            shelf::qwen4exp::make_expert_matrix(weights.layers[0].mlp.gate_experts);
        const auto expert_slot = shelf::qwen_hybrid::make_weight_slot(expert_matrix);
        check(shelf::execution::validate_host_weight_slot(expert_slot, 0u).ok(),
              "EXL3 expert plane forms a rank-3 weight slot");
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
