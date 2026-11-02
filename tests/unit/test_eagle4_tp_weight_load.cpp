#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/eagle4/tensor_parallel_plan.h>
#include <phaseshift/models/qwen35/eagle4/weights.h>

#include "support/eagle4_checkpoint.h"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

int g_fail = 0;

void check(bool ok, const std::string& tag) {
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s\n", tag.c_str());
}

using ps::weights::TensorPartitionDesc;
using ps::weights::TensorPartitionRange;

void check_partition(const TensorPartitionDesc& d, std::uint32_t axis,
                     std::uint32_t index, std::vector<std::int64_t> global_shape,
                     std::vector<TensorPartitionRange> ranges,
                     const std::string& tag) {
    check(d.axis == static_cast<std::int32_t>(axis), tag + " axis");
    check(d.index == index, tag + " index");
    check(d.count == 2, tag + " count");
    check(d.global_shape == global_shape, tag + " global_shape");
    check(d.ranges == ranges, tag + " ranges");
}

bool check_matrix_values(const ps::weights::MatrixWeight& w, const eagle4_ckpt::HostTensor& host,
                         std::uint32_t row_begin, std::uint32_t rows, const std::string& tag) {
    if (w.rows != rows || w.cols != host.cols) {
        check(false, tag + " shape for value check");
        return false;
    }
    const std::size_t count = static_cast<std::size_t>(rows) * w.cols;
    std::vector<ps::bf16_t> got(count);
    const hipError_t err = hipMemcpy(got.data(), w.bf16.data<ps::bf16_t>(),
                                     count * sizeof(ps::bf16_t), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        check(false, tag + " d2h");
        return false;
    }
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::size_t host_base =
            static_cast<std::size_t>(row_begin + r) * host.cols;
        const std::size_t local_base = static_cast<std::size_t>(r) * w.cols;
        for (std::uint32_t c = 0; c < w.cols; ++c) {
            if (got[local_base + c].data != host.data[host_base + c].data) {
                check(false, tag + " value mismatch at row " + std::to_string(r) +
                                 " col " + std::to_string(c));
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL hipSetDevice\n");
        return 1;
    }

    std::error_code ec;
    const fs::path dir =
        fs::temp_directory_path(ec) / ("ps_eagle4_tp_weight_" +
                                       std::to_string(static_cast<long>(::getpid())));
    fs::remove_all(dir, ec);
    if (!fs::create_directories(dir, ec)) {
        std::printf("FAIL create dir\n");
        return 1;
    }

    if (!eagle4_ckpt::write_eagle4_config_json(dir.string(), 256u, 64u, {1, 2, 3})) {
        std::printf("FAIL config json\n");
        fs::remove_all(dir, ec);
        return 1;
    }
    auto config_result = ps::qwen35::eagle4::read_eagle4_config(dir.string());
    check(config_result.ok(), "config reads");
    if (!config_result.ok()) {
        std::printf("config: %s\n", config_result.status().message().c_str());
        fs::remove_all(dir, ec);
        std::printf("FAIL (%d)\n", g_fail);
        return 1;
    }
    const ps::qwen35::eagle4::Eagle4Config config = config_result.release();
    const eagle4_ckpt::HostModel host = eagle4_ckpt::make_host_model();
    check(eagle4_ckpt::write_shard(dir.string(), config, host), "checkpoint writes");
    if (g_fail != 0) {
        fs::remove_all(dir, ec);
        std::printf("FAIL (%d)\n", g_fail);
        return 1;
    }

    auto plan0_result =
        ps::qwen35::eagle4::build_eagle4_tensor_partition_plan(config, 2u, 0u);
    auto plan1_result =
        ps::qwen35::eagle4::build_eagle4_tensor_partition_plan(config, 2u, 1u);
    check(plan0_result.ok(), "plan rank0");
    check(plan1_result.ok(), "plan rank1");
    if (plan0_result.ok() && plan1_result.ok()) {
        const ps::weights::TensorPartitionPlan& p0 = plan0_result.value();
        const ps::weights::TensorPartitionPlan& p1 = plan1_result.value();
        check(p0.tp_size == 2 && p0.tp_rank == 0, "plan0 rank fields");
        check(p0.tensors.count("fc.weight") == 0, "fc stays replicated");
        check(p0.tensors.count("efficient_lm_head_down_proj.weight") == 0,
              "efficient down stays replicated");
        auto it_q0 = p0.tensors.find("layers.0.self_attn.q_proj.weight");
        auto it_q1 = p1.tensors.find("layers.0.self_attn.q_proj.weight");
        check(it_q0 != p0.tensors.end() && it_q1 != p1.tensors.end(), "q partition present");
        if (it_q0 != p0.tensors.end() && it_q1 != p1.tensors.end()) {
            check_partition(it_q0->second, 0, 0, {256, 512}, {{0, 128}}, "q rank0");
            check_partition(it_q1->second, 0, 1, {256, 512}, {{128, 128}}, "q rank1");
        }
        auto it_down0 = p0.tensors.find("layers.0.mlp.down_proj.weight");
        check(it_down0 != p0.tensors.end(), "down partition present");
        if (it_down0 != p0.tensors.end()) {
            check_partition(it_down0->second, 1, 0, {256, 64}, {{0, 32}}, "down rank0");
        }
        auto it_embed0 = p0.tensors.find("embed_tokens.weight");
        check(it_embed0 != p0.tensors.end(), "embed partition present");
        if (it_embed0 != p0.tensors.end()) {
            check_partition(it_embed0->second, 0, 0, {64, 256}, {{0, 32}}, "embed rank0");
        }
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL stream\n");
        fs::remove_all(dir, ec);
        return 1;
    }
    auto arena_result = ps::gpu::GpuArena::create(0, 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("FAIL arena: %s\n", arena_result.status().message().c_str());
        fs::remove_all(dir, ec);
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    ps::qwen35::eagle4::Eagle4WeightLoadOptions options;
    options.preshuffle = false;
    options.tp_size = 2;

    options.tp_rank = 0;
    auto rank0_result =
        ps::qwen35::eagle4::load_eagle4_weights(dir.string(), config, arena, stream, options);
    check(rank0_result.ok(), "rank0 load");
    if (!rank0_result.ok()) {
        std::printf("rank0 load: %s\n", rank0_result.status().message().c_str());
    }

    options.tp_rank = 1;
    auto rank1_result =
        ps::qwen35::eagle4::load_eagle4_weights(dir.string(), config, arena, stream, options);
    check(rank1_result.ok(), "rank1 load");
    if (!rank1_result.ok()) {
        std::printf("rank1 load: %s\n", rank1_result.status().message().c_str());
    }

    if (rank0_result.ok() && rank1_result.ok()) {
        const ps::qwen35::eagle4::Eagle4Weights& w0 = rank0_result.value();
        const ps::qwen35::eagle4::Eagle4Weights& w1 = rank1_result.value();

        check(w0.present && w1.present, "weights present");
        check(w0.embed_tokens.rows == 32 && w1.embed_tokens.rows == 32,
              "embed rows halved");
        check(w0.candidate.lm_head.rows == 32 && w1.candidate.lm_head.rows == 32, "lm_head rows halved");
        check(w0.candidate.efficient_lm_head.rows == 32,
              "efficient head rows halved");
        check(w0.embed_tokens.cols == 256, "embed cols untouched");
        check(!w0.fc.partition.has_value() && !w1.fc.partition.has_value(),
              "fc replicated without partition");
        check(!w0.candidate.efficient_lm_head_down_proj.partition.has_value(),
              "efficient down replicated without partition");

        check(w0.layers.size() == 1 && w1.layers.size() == 1, "layer count");
        if (w0.layers.size() == 1 && w1.layers.size() == 1) {
            const auto& l0 = w0.layers[0];
            const auto& l1 = w1.layers[0];
            check(l0.attn_q_proj.rows == 128 && l1.attn_q_proj.rows == 128,
                  "q rows halved");
            check(l0.attn_q_proj.cols == 512, "q cols untouched");
            check(l0.attn_o_proj.cols == 128 && l1.attn_o_proj.cols == 128,
                  "o cols halved");
            check(l0.attn_o_proj.rows == 256, "o rows untouched");
            check(l0.mlp_gate_proj.rows == 32 && l1.mlp_gate_proj.rows == 32,
                  "gate rows halved");
            check(l0.mlp_down_proj.cols == 32 && l1.mlp_down_proj.cols == 32,
                  "down cols halved");
            check(l0.input_layernorm_weight.dim(0) == 256, "norm replicated");

            auto it_q = l0.attn_q_proj.partition;
            check(l0.attn_q_proj.partition.has_value(), "q has partition");
            if (l0.attn_q_proj.partition.has_value()) {
                check_partition(*l0.attn_q_proj.partition, 0, 0, {256, 512},
                                {{0, 128}}, "loaded q rank0");
            }
            if (l1.attn_q_proj.partition.has_value()) {
                check_partition(*l1.attn_q_proj.partition, 0, 1, {256, 512},
                                {{128, 128}}, "loaded q rank1");
            }
        }

        check_matrix_values(w0.embed_tokens, host.embed, 0, 32, "embed rank0 rows");
        check_matrix_values(w1.embed_tokens, host.embed, 32, 32, "embed rank1 rows");
        if (w0.layers.size() == 1) {
            check_matrix_values(w0.layers[0].mlp_gate_proj, host.gate, 0, 32,
                                "gate rank0 rows");
            check_matrix_values(w1.layers[0].mlp_gate_proj, host.gate, 32, 32,
                                "gate rank1 rows");
        }
    }

    (void)arena.shutdown();
    fs::remove_all(dir, ec);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
