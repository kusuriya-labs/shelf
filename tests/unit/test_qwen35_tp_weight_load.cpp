#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/weights/model_weights.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "support/qwen35_tp_fixture.h"

namespace fs = std::filesystem;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static void check_partition(const shelf::weights::MatrixWeight& w, std::uint32_t axis,
                            std::uint32_t index, std::vector<int64_t> global_shape,
                            std::vector<shelf::weights::TensorPartitionRange> ranges,
                            const std::string& tag) {
    if (!w.partition.has_value()) {
        fail(tag + " missing partition");
        return;
    }
    const shelf::weights::TensorPartitionDesc& d = *w.partition;
    check(d.axis == static_cast<int32_t>(axis), tag + " axis");
    check(d.index == index, tag + " index");
    check(d.count == 2, tag + " count");
    check(d.global_shape == global_shape, tag + " global_shape");
    check(d.ranges == ranges, tag + " ranges");
}

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
        fs::temp_directory_path(ec) / ("qwen35_tp_weight_load_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    shelf::Status fixture = tp_fixture::build_fixture(dir);
    if (!fixture.ok()) {
        fail("fixture: " + fixture.message());
        fs::remove_all(dir, ec);
        std::printf("FAIL (%d)\n", g_fail);
        return 1;
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        fail("stream create");
        return 1;
    }
    auto arena_result = shelf::gpu::GpuArena::create(0, 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        fail("arena");
        return 1;
    }
    shelf::gpu::GpuArena arena = arena_result.release();

    shelf::qwen35::Qwen35LoadOptions options;
    options.verify_quantized_payload_crc = true;

    auto rank0_result = shelf::qwen35::load_qwen35_weights_tensor_parallel_rank(
        dir.string(), 2, 0, arena, stream, options);
    if (!rank0_result.ok()) {
        fail("rank0 load: " + rank0_result.status().message());
    }
    auto rank1_result = shelf::qwen35::load_qwen35_weights_tensor_parallel_rank(
        dir.string(), 2, 1, arena, stream, options);
    if (!rank1_result.ok()) {
        fail("rank1 load: " + rank1_result.status().message());
    }

    if (rank0_result.ok() && rank1_result.ok()) {
        const shelf::qwen35::Qwen35ModelWeights& r0 = rank0_result.value();
        const shelf::qwen35::Qwen35ModelWeights& r1 = rank1_result.value();
        check(r0.layers.size() == 4 && r1.layers.size() == 4, "layer count");

        check(r0.layers[0].ffn.gate.rows == 64, "rank0 gate rows local");
        check(r1.layers[0].ffn.gate.rows == 64, "rank1 gate rows local");
        check_partition(r0.layers[0].ffn.gate, 0, 0, {128, 64}, {{0, 64}},
                        "rank0 gate");
        check_partition(r1.layers[0].ffn.gate, 0, 1, {128, 64}, {{64, 64}},
                        "rank1 gate");
        check(r0.layers[0].ffn.down.cols == 64, "rank0 down cols local");
        check_partition(r0.layers[0].ffn.down, 1, 0, {64, 128}, {{0, 64}},
                        "rank0 down");
        check_partition(r1.layers[0].ffn.down, 1, 1, {64, 128}, {{64, 64}},
                        "rank1 down");

        check(r0.layers[1].attn_q_proj.rows == 64, "rank0 q rows local");
        check(r1.layers[1].attn_q_proj.rows == 64, "rank1 q rows local");
        check_partition(r0.layers[1].attn_q_proj, 0, 0, {128, 64}, {{0, 64}}, "rank0 q");
        check_partition(r1.layers[1].attn_q_proj, 0, 1, {128, 64}, {{64, 64}}, "rank1 q");
        check(r0.layers[1].attn_k_proj.rows == 16, "rank0 kv rows local");
        check_partition(r0.layers[1].attn_k_proj, 0, 0, {32, 64}, {{0, 16}}, "rank0 k");
        check_partition(r1.layers[1].attn_k_proj, 0, 1, {32, 64}, {{16, 16}}, "rank1 k");
        check(r0.layers[1].attn_o_proj.cols == 32, "rank0 o cols local");
        check_partition(r0.layers[1].attn_o_proj, 1, 0, {64, 64}, {{0, 32}}, "rank0 o");

        check(r0.layers[0].attn_in_proj_qkv.rows == 1024, "rank0 gdn qkv rows local");
        check(r0.layers[0].attn_in_proj_qkv.partition.has_value() &&
                  r0.layers[0].attn_in_proj_qkv.partition->ranges.size() == 3,
              "rank0 gdn qkv segmented ranges");
        check(r1.layers[0].attn_in_proj_qkv.partition.has_value() &&
                  r1.layers[0].attn_in_proj_qkv.partition->ranges.size() == 3,
              "rank1 gdn qkv segmented ranges");
        check(r0.layers[0].attn_in_proj_qkv.partition !=
                  r1.layers[0].attn_in_proj_qkv.partition,
              "rank0 and rank1 gdn qkv partitions differ");
        check(r0.layers[0].attn_conv1d_weight.dim(0) == 1024,
              "rank0 conv1d channels local");
        check(r1.layers[0].attn_conv1d_weight.dim(0) == 1024,
              "rank1 conv1d channels local");
        check(r0.layers[0].attn_dt_bias.dim(0) == 4, "rank0 dt_bias local");
        check(r1.layers[0].attn_dt_bias.dim(0) == 4, "rank1 dt_bias local");
        check(r0.layers[0].attn_norm_weight.dim(0) == 128, "gdn norm replicated");
        check(r0.embed_tokens.rows == 128 && r0.embed_tokens.cols == 64,
              "embedding replicated");
        check(!r0.embed_tokens.partition.has_value(), "embedding has no partition");
        check(r0.layers[0].input_layernorm_weight.dim(0) == 64, "input norm replicated");

        // The model invariants are stated for the model, not for one rank: a rank holds an
        // equal slice of a sharded axis, and validate has to accept that and reject a slice
        // that does not add back up.
        auto config_result = shelf::qwen35::read_qwen35_text_config(dir.string());
        check(config_result.ok(), "dense config for the TP fixture");
        if (config_result.ok()) {
            const shelf::qwen35::Qwen35TextConfig& cfg = config_result.value();
            check(shelf::qwen35::validate_qwen35_weights(r0, cfg).ok(),
                  "rank0 weights pass the model invariants (partition aware)");
            check(shelf::qwen35::validate_qwen35_weights(r1, cfg).ok(),
                  "rank1 weights pass the model invariants (partition aware)");

            shelf::qwen35::Qwen35ModelWeights wrong_ffn = r0;
            wrong_ffn.layers[0].ffn.gate.rows = 33;
            check(!shelf::qwen35::validate_qwen35_weights(wrong_ffn, cfg).ok(),
                  "an FFN slice that does not add back up to intermediate_size is rejected");

            shelf::qwen35::Qwen35ModelWeights missing_attn = r0;
            missing_attn.layers[1].attn_q_proj = {};
            check(!shelf::qwen35::validate_qwen35_weights(missing_attn, cfg).ok(),
                  "an attention layer that lost q_proj is rejected");

            shelf::qwen35::Qwen35ModelWeights both_attns = r0;
            both_attns.layers[1].attn_in_proj_a = r0.layers[0].attn_in_proj_a;
            check(!shelf::qwen35::validate_qwen35_weights(both_attns, cfg).ok(),
                  "a layer carrying both attentions is rejected");
        }
    }

    {
        shelf::qwen35::Qwen35LoadOptions bad;
        bad.weights.partition_plan = new shelf::weights::TensorPartitionPlan();
        auto rejected = shelf::qwen35::load_qwen35_weights_tensor_parallel_rank(
            dir.string(), 2, 0, arena, stream, bad);
        delete bad.weights.partition_plan;
        check(!rejected.ok(), "explicit partition plan is rejected");
    }

    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    arena.shutdown();
    fs::remove_all(dir, ec);

    if (tp_fixture::g_fail != 0) g_fail += tp_fixture::g_fail;
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
