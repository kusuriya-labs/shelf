#include <phaseshift/models/qwen4exp/model/lower_to_primitives.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_config.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_weights.h>
#include <phaseshift/runtime/execution/row_bucket.h>
#include <phaseshift/runtime/program/program.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace q4e = ps::qwen4exp;
namespace rt = ps::runtime;

int g_fail = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

struct StageRange {
    std::uint32_t begin;
    std::uint32_t end;
};

}

int main(int argc, char** argv) {
    const char* model_dir = argc > 1
        ? argv[1]
        : "/opt/zen/work/models/Qwen3.8-Flash-Next-Quark-AWQ-PSQ";

    std::printf("test_qwen4exp_pp_stage\n");
    std::printf("model: %s\n", model_dir);

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 4) {
        std::printf("SKIP: needs 4 GPUs\n");
        return 77;
    }

    auto config_result = q4e::load_qwen4exp_config(model_dir);
    if (!config_result.ok()) {
        std::printf("FAIL: load config: %s\n", config_result.status().message().c_str());
        return 1;
    }
    const q4e::Qwen4ExpTextConfig cfg = config_result.value();
    check(cfg.num_hidden_layers == 48, "48 layers");

    const std::uint32_t total_layers = static_cast<std::uint32_t>(cfg.num_hidden_layers);
    const std::uint32_t per_stage = total_layers / 4u;
    std::vector<StageRange> stages;
    for (std::uint32_t s = 0; s < 4u; ++s) {
        stages.push_back(StageRange{s * per_stage, (s + 1u) * per_stage});
    }

    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto arena_result = ps::gpu::GpuArena::create(0, 24ull << 30);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    q4e::Qwen4ExpLoadOptions load_opts;
    load_opts.load_expert_weights = false;
    auto weights_result = q4e::load_qwen4exp_weights_quantized(model_dir, arena, stream,
                                                               load_opts);
    if (!weights_result.ok()) {
        std::printf("FAIL: load weights: %s\n", weights_result.status().message().c_str());
        return 1;
    }
    q4e::Qwen4ExpModelWeights weights = weights_result.release();

    {
        auto st0 = ps::gpu::GpuArena::create(0, 8ull << 30);
        if (st0.ok()) {
            ps::gpu::GpuArena st_arena = st0.release();
            q4e::Qwen4ExpLoadOptions stage_opts;
            stage_opts.load_expert_weights = false;
            stage_opts.layer_begin = 24;
            stage_opts.layer_end = 36;
            auto staged = q4e::load_qwen4exp_weights_quantized(model_dir, st_arena, stream,
                                                               stage_opts);
            if (!staged.ok()) {
                std::printf("FAIL: stage range load: %s\n",
                            staged.status().message().c_str());
                ++g_fail;
            } else {
                q4e::Qwen4ExpModelWeights sw = staged.release();
                check(sw.layers.size() == total_layers, "layer array keeps full size");
                check(sw.layers[0].gdn.qkv.rows == 0, "layer before range not loaded");
                check(sw.layers[24].gdn.qkv.rows > 0, "range start layer loaded");
                check(sw.layers[30].gdn.qkv.rows > 0, "mid range layer loaded");
                check(sw.layers[40].gdn.qkv.rows == 0, "layer after range not loaded");
                std::printf("stage-range load used %.3f GiB\n",
                            static_cast<double>(st_arena.used()) / (1ull << 30));
            }
            st_arena.shutdown();
        }
    }

    std::uint32_t prompt_tokens = 32;
    auto bucket_result = rt::resolve_row_bucket(prompt_tokens);
    if (!bucket_result.ok()) {
        std::printf("FAIL: resolve_row_bucket: %s\n",
                    bucket_result.status().message().c_str());
        return 1;
    }

    std::vector<q4e::Qwen4ExpLoweredPrimitives> lowered_stages;
    for (std::size_t si = 0; si < stages.size(); ++si) {
        const StageRange& r = stages[si];
        q4e::Qwen4ExpLowerOptions lower_opts;
        lower_opts.connect_moe = false;
        lower_opts.connect_ple = false;
        lower_opts.stage_begin = r.begin;
        lower_opts.stage_end = r.end;
        auto lowered = q4e::lower_qwen4exp_to_primitives(cfg, weights, lower_opts);
        if (!lowered.ok()) {
            std::printf("FAIL: stage %zu lowering: %s\n", si,
                        lowered.status().message().c_str());
            ++g_fail;
            continue;
        }
        q4e::Qwen4ExpLoweredPrimitives lp = lowered.release();

        const bool first = r.begin == 0;
        const bool last = r.end == total_layers;
        check(lp.graph.external_inputs.size() == 1, "stage external input count");
        if (last) {
            check(lp.graph.external_outputs.size() >= 3, "final stage emits logits");
        } else {
            check(lp.graph.external_outputs.size() == 1, "stage hidden output count");
        }
        check(lp.graph.nodes.size() > 100, "stage graph has nodes");
        check(!lp.weights.empty(), "stage weight slots registered");
        std::printf("stage %zu [%u,%u): in=%zu out=%zu nodes=%zu weights=%zu\n", si,
                    r.begin, r.end, lp.graph.external_inputs.size(),
                    lp.graph.external_outputs.size(), lp.graph.nodes.size(),
                    lp.weights.size());

        rt::ProgramBuildOptions build_opts;
        build_opts.validate_workspace = true;
        build_opts.no_workspace_reuse = true;
        build_opts.max_token_rows = prompt_tokens;
        build_opts.max_output_rows = 1;
        const rt::WeightTableView weight_view{lp.weights.data(), lp.weights.size()};
        auto built = rt::build_program(lp.graph, weight_view, bucket_result.value(),
                                       rt::ExecutionClass::PREFILL, build_opts);
        if (!built.ok()) {
            std::printf("FAIL: stage %zu build_program: %s\n", si,
                        built.status().message().c_str());
            ++g_fail;
            continue;
        }
        rt::Program plan = built.release();
        check(plan.dispatches.size() > 0, "stage program has dispatches");
        std::printf("  build ok: dispatches=%zu values=%zu workspace=%zu\n",
                    plan.dispatches.size(), plan.values.size(), plan.workspace.total_bytes);
    }

    {
        q4e::Qwen4ExpLowerOptions bad;
        bad.stage_begin = 24;
        bad.stage_end = 12;
        auto r = q4e::lower_qwen4exp_to_primitives(cfg, weights, bad);
        check(!r.ok(), "empty stage rejected");

        q4e::Qwen4ExpLowerOptions out_of_range;
        out_of_range.stage_begin = 40;
        out_of_range.stage_end = 60;
        auto r2 = q4e::lower_qwen4exp_to_primitives(cfg, weights, out_of_range);
        check(!r2.ok(), "stage beyond layer count rejected");

        q4e::Qwen4ExpLowerOptions ple_stage;
        ple_stage.connect_ple = true;
        ple_stage.stage_begin = 12;
        ple_stage.stage_end = 24;
        auto r3 = q4e::lower_qwen4exp_to_primitives(cfg, weights, ple_stage);
        check(!r3.ok(), "connect_ple outside stage 0 rejected");
    }

    arena.shutdown();
    (void)hipStreamDestroy(stream);

    if (g_fail != 0) {
        std::printf("=== test_qwen4exp_pp_stage: %d FAIL ===\n", g_fail);
        return 1;
    }
    std::printf("=== test_qwen4exp_pp_stage: PASS ===\n");
    return 0;
}
