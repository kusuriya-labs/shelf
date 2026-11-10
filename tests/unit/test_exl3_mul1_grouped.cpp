#include <shelf/io/safetensors_reader.h>
#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/quantization/exl3/exl3.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <fstream>
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

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

std::string shard_for(const std::string& model_dir, const std::string& tensor) {
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

struct HostTensor {
    std::vector<std::uint8_t> bytes;
    std::vector<std::size_t> shape;
    std::string dtype;
};

bool read_tensor(const std::string& model_dir, const std::string& name, HostTensor& out) {
    const std::string shard = shard_for(model_dir, name);
    if (shard.empty()) {
        std::fprintf(stderr, "missing index entry %s\n", name.c_str());
        return false;
    }
    auto reader = shelf::io::SafetensorsReader::open(model_dir + "/" + shard);
    if (!reader.ok()) return false;
    auto spec = reader.value().tensor_spec(name);
    if (!spec.ok()) return false;
    std::size_t bytes = 0;
    auto data = reader.value().tensor_data(name, bytes);
    if (!data.ok()) return false;
    out.shape.clear();
    for (const auto dim : spec.value().shape) out.shape.push_back(dim);
    out.dtype = shelf::io::stype_name(spec.value().dtype);
    out.bytes.assign(static_cast<const std::uint8_t*>(data.value()),
                     static_cast<const std::uint8_t*>(data.value()) + bytes);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "/mnt/Qwen3.8-Flash-Next-exl3";
    if (const char* env = std::getenv("SHELF_QWEN4EXP_MODEL_DIR")) {
        model_dir = env;
    } else if (const char* home = std::getenv("HOME")) {
        const std::string candidate =
            std::string(home) + "/work/shelf/models/Qwen3.8-Flash-Next-exl3";
        if (::access(candidate.c_str(), R_OK) == 0) model_dir = candidate;
    }
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        }
    }

    HostTensor trellis[2];
    HostTensor suh[2];
    HostTensor svh[2];
    for (int expert = 0; expert < 2; ++expert) {
        const std::string base = "model.language_model.layers.0.mlp.experts." +
                                 std::to_string(expert) + ".gate_proj.";
        check(read_tensor(model_dir, base + "trellis", trellis[expert]),
              "read expert trellis");
        check(read_tensor(model_dir, base + "suh", suh[expert]), "read expert suh");
        check(read_tensor(model_dir, base + "svh", svh[expert]), "read expert svh");
    }
    if (failures != 0) return 1;

    for (int expert = 0; expert < 2; ++expert) {
        check(trellis[expert].dtype == "I16" && trellis[expert].shape.size() == 3,
              "trellis is rank-3 I16");
        check(suh[expert].dtype == "F16" && suh[expert].shape.size() == 1,
              "suh is rank-1 F16");
        check(svh[expert].dtype == "F16" && svh[expert].shape.size() == 1,
              "svh is rank-1 F16");
        check(trellis[expert].shape == trellis[0].shape, "experts share trellis shape");
        check(suh[expert].bytes.size() == suh[0].bytes.size(), "experts share suh bytes");
        check(svh[expert].bytes.size() == svh[0].bytes.size(), "experts share svh bytes");
    }
    if (failures != 0) return 1;

    const std::uint32_t words = static_cast<std::uint32_t>(trellis[0].shape[2]);
    const std::uint32_t k = static_cast<std::uint32_t>(trellis[0].shape[0]) * 16u;
    const std::uint32_t n = static_cast<std::uint32_t>(trellis[0].shape[1]) * 16u;
    std::uint32_t ka = 0;
    bool half = false;
    if (words % 16u == 0u) {
        ka = words / 16u;
    } else if (words % 16u == 8u) {
        ka = (words - 8u) / 16u;
        half = true;
    } else {
        std::fprintf(stderr, "invalid EXL3 words=%u\n", words);
        return 1;
    }
    std::fprintf(stderr, "expert gate_proj: k=%u n=%u ka=%u half=%d\n", k, n, ka, half);

    const std::uint32_t row_tile = shelf::kernel::kExl3W16A16RowTile;
    const std::uint32_t group_count = 2u;
    std::vector<std::uint16_t> input(
        static_cast<std::size_t>(group_count) * row_tile * k, 0u);
    input[0] = 0x3F80u;
    input[static_cast<std::size_t>(row_tile) * k + 5u] = 0x3F80u;
    std::vector<std::uint16_t> output(
        static_cast<std::size_t>(group_count) * row_tile * n, 0u);

    void* device_trellis = nullptr;
    void* device_suh = nullptr;
    void* device_svh = nullptr;
    void* device_input = nullptr;
    void* device_output = nullptr;
    std::uint32_t* device_group_expert = nullptr;
    std::uint32_t* device_group_rows = nullptr;
    std::uint32_t* device_group_count = nullptr;
    std::int32_t* device_row_token = nullptr;
    hipError_t error = hipMalloc(&device_trellis, 2u * trellis[0].bytes.size());
    check(error == hipSuccess, "hipMalloc trellis");
    error = hipMalloc(&device_suh, 2u * suh[0].bytes.size());
    check(error == hipSuccess, "hipMalloc suh");
    error = hipMalloc(&device_svh, 2u * svh[0].bytes.size());
    check(error == hipSuccess, "hipMalloc svh");
    error = hipMalloc(&device_input, input.size() * sizeof(std::uint16_t));
    check(error == hipSuccess, "hipMalloc input");
    error = hipMalloc(&device_output, output.size() * sizeof(std::uint16_t));
    check(error == hipSuccess, "hipMalloc output");
    error = hipMalloc(&device_group_expert, 2u * sizeof(std::uint32_t));
    check(error == hipSuccess, "hipMalloc group expert");
    error = hipMalloc(&device_group_rows, 2u * sizeof(std::uint32_t));
    check(error == hipSuccess, "hipMalloc group rows");
    error = hipMalloc(&device_group_count, sizeof(std::uint32_t));
    check(error == hipSuccess, "hipMalloc group count");
    error = hipMalloc(&device_row_token, static_cast<std::size_t>(group_count) * row_tile * sizeof(std::int32_t));
    check(error == hipSuccess, "hipMalloc row token");
    if (failures != 0) return 1;

    for (int expert = 0; expert < 2; ++expert) {
        const std::size_t toff = static_cast<std::size_t>(expert) * trellis[0].bytes.size();
        const std::size_t soff = static_cast<std::size_t>(expert) * suh[0].bytes.size();
        const std::size_t voff = static_cast<std::size_t>(expert) * svh[0].bytes.size();
        error = hipMemcpy(static_cast<std::uint8_t*>(device_trellis) + toff,
                          trellis[expert].bytes.data(), trellis[expert].bytes.size(),
                          hipMemcpyHostToDevice);
        check(error == hipSuccess, "upload trellis");
        error = hipMemcpy(static_cast<std::uint8_t*>(device_suh) + soff,
                          suh[expert].bytes.data(), suh[expert].bytes.size(),
                          hipMemcpyHostToDevice);
        check(error == hipSuccess, "upload suh");
        error = hipMemcpy(static_cast<std::uint8_t*>(device_svh) + voff,
                          svh[expert].bytes.data(), svh[expert].bytes.size(),
                          hipMemcpyHostToDevice);
        check(error == hipSuccess, "upload svh");
    }
    error = hipMemcpy(device_input, input.data(), input.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice);
    check(error == hipSuccess, "upload input");
    error = hipMemcpy(device_output, output.data(), output.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice);
    check(error == hipSuccess, "upload output");
    const std::uint32_t host_group_expert[2] = {0u, 1u};
    const std::uint32_t host_group_rows[2] = {1u, 2u};
    const std::uint32_t host_group_count = group_count;
    error = hipMemcpy(device_group_expert, host_group_expert, sizeof(host_group_expert),
                      hipMemcpyHostToDevice);
    check(error == hipSuccess, "upload group expert");
    error = hipMemcpy(device_group_rows, host_group_rows, sizeof(host_group_rows),
                      hipMemcpyHostToDevice);
    check(error == hipSuccess, "upload group rows");
    error = hipMemcpy(device_group_count, &host_group_count, sizeof(host_group_count),
                      hipMemcpyHostToDevice);
    check(error == hipSuccess, "upload group count");

    shelf::kernel::Exl3GroupedArgs args;
    args.trellis = device_trellis;
    args.suh = device_suh;
    args.svh = device_svh;
    args.input = device_input;
    args.output = device_output;
    args.out_features = n;
    args.k = k;
    args.input_row_stride = k;
    args.output_row_stride = n;
    args.output_dtype = 0u;
    args.ka = ka;
    args.half = half ? 1u : 0u;
    args.trellis_batch_stride = trellis[0].bytes.size();
    args.suh_batch_stride = suh[0].bytes.size();
    args.svh_batch_stride = svh[0].bytes.size();
    args.group_expert = device_group_expert;
    args.group_rows = device_group_rows;
    args.group_count = device_group_count;
    args.group_capacity = group_count;

    error = shelf::kernel::launch_gemm_exl3_w16a16_grouped(args, nullptr);
    check(error == hipSuccess, "grouped EXL3 launch");
    if (error != hipSuccess) {
        std::fprintf(stderr, "launch: %s\n", hipGetErrorString(error));
        return 1;
    }
    error = hipDeviceSynchronize();
    check(error == hipSuccess, "grouped EXL3 sync");
    error = hipMemcpy(output.data(), device_output, output.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost);
    check(error == hipSuccess, "download output");

    std::vector<float> reference[2];
    for (int expert = 0; expert < 2; ++expert) {
        reference[expert].assign(static_cast<std::size_t>(k) * n, 0.0f);
        const bool decoded = shelf::quantization::exl3::dequant(
            reinterpret_cast<const std::uint16_t*>(trellis[expert].bytes.data()),
            reinterpret_cast<const std::uint16_t*>(suh[expert].bytes.data()),
            reinterpret_cast<const std::uint16_t*>(svh[expert].bytes.data()), k, n, ka, half,
            reference[expert].data(), false);
        check(decoded, "CPU EXL3 reference decode");
    }

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::uint32_t col = 0; col < n; ++col) {
        for (int group = 0; group < 2; ++group) {
            const std::uint32_t input_row = group == 0 ? 0u : 5u;
            const std::uint32_t output_row = group * row_tile;
            const float expected =
                reference[group][static_cast<std::size_t>(input_row) * n + col];
            const float actual =
                bf16_to_float(output[static_cast<std::size_t>(output_row) * n + col]);
            const double abs_error = std::fabs(actual - expected);
            const double rel_error = abs_error / std::max<double>(1e-6, std::fabs(expected));
            max_abs = std::max(max_abs, abs_error);
            max_rel = std::max(max_rel, rel_error);
        }
    }
    std::fprintf(stderr, "grouped EXL3 vs CPU: max_abs=%g max_rel=%g\n", max_abs, max_rel);
    check(max_abs <= 2.0e-2 || max_rel <= 2.0e-2,
          "grouped EXL3 output matches the CPU reference");

    // Token-row input: the plan's row_token gathers each group row from a token row.
    std::vector<std::uint16_t> token_input(static_cast<std::size_t>(4u) * k, 0u);
    token_input[static_cast<std::size_t>(1u) * k + 7u] = 0x3F80u;
    token_input[static_cast<std::size_t>(3u) * k + 9u] = 0x3F80u;
    std::vector<std::int32_t> host_row_token(
        static_cast<std::size_t>(group_count) * row_tile, -1);
    host_row_token[0] = 1;
    host_row_token[row_tile] = 3;
    error = hipMemcpy(device_input, token_input.data(),
                      token_input.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice);
    check(error == hipSuccess, "upload token-row input");
    error = hipMemcpy(device_row_token, host_row_token.data(),
                      host_row_token.size() * sizeof(std::int32_t), hipMemcpyHostToDevice);
    check(error == hipSuccess, "upload row token");
    error = hipMemset(device_output, 0, output.size() * sizeof(std::uint16_t));
    check(error == hipSuccess, "clear grouped output");

    const std::uint32_t token_group_rows[2] = {1u, 1u};
    error = hipMemcpy(device_group_rows, token_group_rows, sizeof(token_group_rows),
                      hipMemcpyHostToDevice);
    check(error == hipSuccess, "upload token group rows");
    args.input = device_input;
    args.row_token = device_row_token;
    error = shelf::kernel::launch_gemm_exl3_w16a16_grouped(args, nullptr);
    check(error == hipSuccess, "token-row grouped EXL3 launch");
    if (error != hipSuccess) {
        std::fprintf(stderr, "launch: %s\n", hipGetErrorString(error));
        return 1;
    }
    error = hipDeviceSynchronize();
    check(error == hipSuccess, "token-row grouped EXL3 sync");
    error = hipMemcpy(output.data(), device_output,
                      output.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
    check(error == hipSuccess, "download token-row output");

    double token_max_abs = 0.0;
    for (std::uint32_t col = 0; col < n; ++col) {
        const float expected0 = reference[0][static_cast<std::size_t>(7u) * n + col];
        const float expected1 = reference[1][static_cast<std::size_t>(9u) * n + col];
        const float actual0 = bf16_to_float(output[col]);
        const float actual1 = bf16_to_float(
            output[static_cast<std::size_t>(row_tile) * n + col]);
        token_max_abs = std::max<double>(
            token_max_abs, std::fabs(static_cast<double>(actual0 - expected0)));
        token_max_abs = std::max<double>(
            token_max_abs, std::fabs(static_cast<double>(actual1 - expected1)));
    }
    std::fprintf(stderr, "token-row grouped EXL3 max_abs=%g\n", token_max_abs);
    check(token_max_abs <= 2.0e-2, "token-row gather matches the CPU reference");

    // Fused grouped: one expert owns 5 groups, so the first block fuses 4 groups (64 rows)
    // and the next block takes the remaining group. Every group sees the same expert plane.
    {
        const std::uint32_t fused_groups = 5u;
        std::vector<std::uint16_t> fused_input(
            static_cast<std::size_t>(fused_groups) * row_tile * k, 0u);
        for (std::uint32_t g = 0; g < fused_groups; ++g) {
            fused_input[static_cast<std::size_t>(g) * row_tile * k] = 0x3F80u;
        }
        std::vector<std::uint16_t> fused_output(
            static_cast<std::size_t>(fused_groups) * row_tile * n, 0u);
        void* d_fused_input = nullptr;
        void* d_fused_output = nullptr;
        std::uint32_t* d_fused_expert = nullptr;
        std::uint32_t* d_fused_rows = nullptr;
        std::uint32_t* d_fused_count = nullptr;
        std::uint32_t* d_fused_begin = nullptr;
        check(hipMalloc(&d_fused_input, fused_input.size() * 2u) == hipSuccess,
              "malloc fused input");
        check(hipMalloc(&d_fused_output, fused_output.size() * 2u) == hipSuccess,
              "malloc fused output");
        check(hipMalloc(&d_fused_expert, fused_groups * 4u) == hipSuccess,
              "malloc fused expert");
        check(hipMalloc(&d_fused_rows, fused_groups * 4u) == hipSuccess, "malloc fused rows");
        check(hipMalloc(&d_fused_count, 4u) == hipSuccess, "malloc fused count");
        check(hipMalloc(&d_fused_begin, 2u * 4u) == hipSuccess, "malloc fused begin");
        const std::uint32_t fused_expert[5] = {0u, 0u, 0u, 0u, 0u};
        const std::uint32_t fused_rows[5] = {16u, 16u, 16u, 16u, 16u};
        const std::uint32_t fused_count = fused_groups;
        const std::uint32_t fused_begin[2] = {0u, 5u};
        check(hipMemcpy(d_fused_input, fused_input.data(), fused_input.size() * 2u,
                        hipMemcpyHostToDevice) == hipSuccess,
              "copy fused input");
        check(hipMemcpy(d_fused_expert, fused_expert, sizeof(fused_expert),
                        hipMemcpyHostToDevice) == hipSuccess,
              "copy fused expert");
        check(hipMemcpy(d_fused_rows, fused_rows, sizeof(fused_rows),
                        hipMemcpyHostToDevice) == hipSuccess,
              "copy fused rows");
        check(hipMemcpy(d_fused_count, &fused_count, 4u, hipMemcpyHostToDevice) ==
                  hipSuccess,
              "copy fused count");
        check(hipMemcpy(d_fused_begin, fused_begin, sizeof(fused_begin),
                        hipMemcpyHostToDevice) == hipSuccess,
              "copy fused begin");

        shelf::kernel::Exl3GroupedArgs fused_args = args;
        fused_args.input = d_fused_input;
        fused_args.output = d_fused_output;
        fused_args.row_token = nullptr;
        fused_args.group_expert = d_fused_expert;
        fused_args.group_rows = d_fused_rows;
        fused_args.group_count = d_fused_count;
        fused_args.expert_group_begin = d_fused_begin;
        fused_args.group_capacity = fused_groups;
        check(shelf::kernel::launch_gemm_exl3_w16a16_grouped(fused_args, nullptr) ==
                  hipSuccess,
              "fused launch");
        check(hipDeviceSynchronize() == hipSuccess, "fused sync");
        check(hipMemcpy(fused_output.data(), d_fused_output, fused_output.size() * 2u,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "copy fused output");

        double fused_max_abs = 0.0;
        for (std::uint32_t g = 0; g < fused_groups; ++g) {
            for (std::uint32_t col = 0; col < n; ++col) {
                const float expected = reference[0][col];
                const float actual = bf16_to_float(
                    fused_output[static_cast<std::size_t>(g) * row_tile * n + col]);
                fused_max_abs = std::max<double>(
                    fused_max_abs, std::fabs(static_cast<double>(actual - expected)));
            }
        }
        std::fprintf(stderr, "fused grouped EXL3 max_abs=%g\n", fused_max_abs);
        check(fused_max_abs <= 2.0e-2, "fused grouped matches the CPU reference");
        (void)hipFree(d_fused_input);
        (void)hipFree(d_fused_output);
        (void)hipFree(d_fused_expert);
        (void)hipFree(d_fused_rows);
        (void)hipFree(d_fused_count);
        (void)hipFree(d_fused_begin);
    }

    (void)hipFree(device_row_token);
    (void)hipFree(device_trellis);
    (void)hipFree(device_suh);
    (void)hipFree(device_svh);
    (void)hipFree(device_input);
    (void)hipFree(device_output);
    (void)hipFree(device_group_expert);
    (void)hipFree(device_group_rows);
    (void)hipFree(device_group_count);

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
