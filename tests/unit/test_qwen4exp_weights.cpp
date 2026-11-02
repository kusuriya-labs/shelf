#include <phaseshift/core/memory/arena.h>
#include <phaseshift/io/safetensors_writer.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_weights.h>
#include <phaseshift/weights/weight_loader.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
namespace q4e = ps::qwen4exp;
namespace fw = ps::weights;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        ++passed;
        printf("PASS: %s\n", msg);
    } else {
        ++failed;
        printf("FAIL: %s\n", msg);
    }
}

namespace {

constexpr std::size_t kVocab = 32;
constexpr std::size_t kHidden = 64;
constexpr std::size_t kLayers = 4;
constexpr std::size_t kHcCount = 4;
constexpr std::size_t kHcLowrank = 8;
constexpr std::size_t kHcHidden = kHcCount * kHidden;
constexpr std::size_t kExperts = 4;
constexpr std::size_t kMoeInter = 16;
constexpr std::size_t kSharedInter = 16;
constexpr std::size_t kKeyHeads = 4;
constexpr std::size_t kValHeads = 8;
constexpr std::size_t kHeadDim = 8;
constexpr std::size_t kValueDim = kValHeads * kHeadDim;
constexpr std::size_t kQkDim = kKeyHeads * kHeadDim;
constexpr std::size_t kConvDim = 2 * kQkDim + kValueDim;
constexpr std::size_t kQsaHeads = 4;
constexpr std::size_t kQsaKv = 2;
constexpr std::size_t kQsaHeadDim = 16;
constexpr std::size_t kIndexerHeads = 2;
constexpr std::size_t kIndexerHeadDim = 8;

std::string g_dir;
std::mt19937_64 g_rng(0xC0FFEEu);

template <typename T>
std::vector<T> values(std::size_t n) {
    std::vector<T> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<T>((g_rng() % 97u) + 1u);
    }
    return out;
}

const char* kLmPrefix = "model.language_model.";

void build_fixture() {
    g_dir = "/tmp/ps_q4e_weights_" + std::to_string(getpid());
    fs::remove_all(g_dir);
    fs::create_directories(g_dir);

    {
        std::ofstream f(g_dir + "/config.json");
        f << R"({"model_type":"qwen4_exp","text_config":{)"
          << R"("vocab_size":)" << kVocab << R"(,"hidden_size":)" << kHidden
          << R"(,"num_hidden_layers":)" << kLayers
          << R"(,"num_attention_heads":)" << kQsaHeads
          << R"(,"num_key_value_heads":)" << kQsaKv
          << R"(,"head_dim":)" << kQsaHeadDim
          << R"(,"full_attention_interval":4,)"
          << R"("linear_num_key_heads":)" << kKeyHeads
          << R"(,"linear_num_value_heads":)" << kValHeads
          << R"(,"linear_key_head_dim":)" << kHeadDim
          << R"(,"linear_value_head_dim":)" << kHeadDim
          << R"(,"linear_conv_kernel_dim":4,)"
          << R"("num_experts":)" << kExperts
          << R"(,"num_experts_per_tok":2,"moe_intermediate_size":)" << kMoeInter
          << R"(,"shared_expert_intermediate_size":)" << kSharedInter
          << R"(,"hc_count":)" << kHcCount << R"(,"hc_lowrank":)" << kHcLowrank
          << R"(,"ngram_size":3,"heads_per_ngram":8,)"
          << R"("ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,)"
          << R"("split_ngram_parts":128,"ple_layer_ids":[2],)"
          << R"("ple_embed_dim":)" << kHidden << R"(,"ple_conv_kernel_size":4,)"
          << R"("indexer_n_heads":)" << kIndexerHeads
          << R"(,"indexer_kv_heads":1,"indexer_head_dim":)" << kIndexerHeadDim
          << R"(,"indexer_budget":16,"indexer_compress_ratio":4,)"
          << R"("rms_norm_eps":1e-6,"output_gate_type":"sigmoid",)"
          << R"("rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25},)"
          << R"("eos_token_id":7,)"
          << R"("layer_types":["linear_attention","linear_attention",)"
          << R"("linear_attention","full_attention"]}})";
    }
    {
        std::ofstream f(g_dir + "/model.safetensors.index.json");
        f << R"({"weight_map":{)";
        bool first = true;
        const auto emit = [&](const std::string& name, const std::string& shard) {
            if (!first) f << ",";
            first = false;
            f << "\"" << name << "\":\"" << shard << "\"";
        };
        emit("model.language_model.embed_tokens.weight", "model-1.safetensors");
        emit("lm_head.weight", "model-2.safetensors");
        emit("model.language_model.hyper_connection_mixer.hc_norm.weight", "model-2.safetensors");
        emit("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
             "model-2.safetensors");
        emit("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
             "model-2.safetensors");
        for (std::size_t l = 0; l < kLayers; ++l) {
            const std::string p = std::string("model.language_model.layers.") +
                                  std::to_string(l) + ".";
            emit(p + "attn_hyper_connection.hc_norm.weight", "model-2.safetensors");
            emit(p + "attn_hyper_connection.input_mix_weight_down.weight", "model-2.safetensors");
            emit(p + "attn_hyper_connection.input_mix_weight_up.weight", "model-2.safetensors");
            emit(p + "attn_hyper_connection.block_inject_weight.weight", "model-2.safetensors");
            emit(p + "mlp_hyper_connection.hc_norm.weight", "model-3.safetensors");
            emit(p + "mlp_hyper_connection.input_mix_weight_down.weight", "model-3.safetensors");
            emit(p + "mlp_hyper_connection.input_mix_weight_up.weight", "model-3.safetensors");
            emit(p + "mlp_hyper_connection.block_inject_weight.weight", "model-3.safetensors");
            if (l != 3) {
                emit(p + "linear_attn.in_proj_qkv.weight", "model-4.safetensors");
                emit(p + "linear_attn.in_proj_z.weight", "model-4.safetensors");
                emit(p + "linear_attn.in_proj_b.weight", "model-4.safetensors");
                emit(p + "linear_attn.in_proj_a.weight", "model-4.safetensors");
                emit(p + "linear_attn.out_proj.weight", "model-4.safetensors");
                emit(p + "linear_attn.conv1d.weight", "model-4.safetensors");
                emit(p + "linear_attn.norm.weight", "model-4.safetensors");
                emit(p + "linear_attn.dt_bias", "model-4.safetensors");
                emit(p + "linear_attn.A_log", "model-4.safetensors");
            } else {
                emit(p + "self_attn.q_proj.weight", "model-5.safetensors");
                emit(p + "self_attn.k_proj.weight", "model-5.safetensors");
                emit(p + "self_attn.v_proj.weight", "model-5.safetensors");
                emit(p + "self_attn.o_proj.weight", "model-5.safetensors");
                emit(p + "self_attn.q_norm.weight", "model-5.safetensors");
                emit(p + "self_attn.k_norm.weight", "model-5.safetensors");
                emit(p + "self_attn.indexer.index_qk_proj.weight", "model-5.safetensors");
                emit(p + "self_attn.indexer.q_layernorm.weight", "model-5.safetensors");
                emit(p + "self_attn.indexer.k_layernorm.weight", "model-5.safetensors");
            }
            emit(p + "mlp.gate.weight", "model-6.safetensors");
            emit(p + "mlp.shared_expert.gate_proj.weight", "model-6.safetensors");
            emit(p + "mlp.shared_expert.up_proj.weight", "model-6.safetensors");
            emit(p + "mlp.shared_expert.down_proj.weight", "model-6.safetensors");
            emit(p + "mlp.shared_expert_gate.weight", "model-6.safetensors");
            emit(p + "mlp.experts.gate_up_proj", "model-7.safetensors");
            emit(p + "mlp.experts.down_proj", "model-7.safetensors");
        }
        const std::string pp = "model.language_model.layers.1.ple.";
        emit(pp + "key_proj.weight", "model-8.safetensors");
        emit(pp + "value_proj.weight", "model-8.safetensors");
        emit(pp + "conv1d.weight", "model-8.safetensors");
        emit(pp + "norm_key.weight", "model-8.safetensors");
        emit(pp + "norm_query.weight", "model-8.safetensors");
        emit(pp + "norm_conv.weight", "model-8.safetensors");
        emit(pp + "ple_embedding.layer_multipliers", "model-8.safetensors");
        emit(pp + "ple_embedding.ngram_heads_vocab_sizes", "model-8.safetensors");
        emit(pp + "ple_embedding.ngram_heads_offsets", "model-8.safetensors");
        f << "}}";
    }
}

}  // namespace

using ShapeSpec = std::pair<std::string, std::vector<std::size_t>>;

static void write_shard(const std::string& file,
                        const std::vector<ShapeSpec>& bf16,
                        const std::vector<ShapeSpec>& i64) {
    auto w = ps::io::SafetensorsWriter::create(g_dir + "/" + file);
    if (!w.ok()) {
        printf("FAIL: create %s\n", file.c_str());
        ++failed;
        return;
    }
    const auto count = [](const std::vector<std::size_t>& shape) {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        return n;
    };
    for (const auto& [name, shape] : bf16) {
        if (!w.value().plan_tensor(name, ps::io::SType::BF16, shape).ok()) {
            printf("FAIL: plan %s\n", name.c_str());
            ++failed;
        }
    }
    for (const auto& [name, shape] : i64) {
        if (!w.value().plan_tensor(name, ps::io::SType::I64, shape).ok()) {
            printf("FAIL: plan i64 %s\n", name.c_str());
            ++failed;
        }
    }
    if (!w.value().write_header().ok()) {
        printf("FAIL: header %s\n", file.c_str());
        ++failed;
        return;
    }
    for (const auto& [name, shape] : bf16) {
        auto v = values<std::uint16_t>(count(shape));
        if (!w.value().write_tensor(name, v.data(), v.size() * 2).ok()) {
            printf("FAIL: write %s\n", name.c_str());
            ++failed;
        }
    }
    for (const auto& [name, shape] : i64) {
        auto v = values<std::int64_t>(count(shape));
        if (!w.value().write_tensor(name, v.data(), v.size() * 8).ok()) {
            printf("FAIL: write i64 %s\n", name.c_str());
            ++failed;
        }
    }
    if (!w.value().finish().ok()) {
        printf("FAIL: finish %s\n", file.c_str());
        ++failed;
    }
}

static void build_shards() {
    const std::string lp = "model.language_model.layers.";
    const auto p = [&](std::size_t l) { return lp + std::to_string(l) + "."; };
    const auto two = [](std::size_t a, std::size_t b) {
        return std::vector<std::size_t>{a, b};
    };

    write_shard("model-1.safetensors",
                {{"model.language_model.embed_tokens.weight", two(kVocab, kHidden)}}, {});

    std::vector<ShapeSpec> head = {
        {"lm_head.weight", two(kVocab, kHidden)},
        {"model.language_model.hyper_connection_mixer.hc_norm.weight", {kHcHidden}},
        {"model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
         two(kHcLowrank, kHcHidden)},
        {"model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
         two(kHcHidden, kHcLowrank)}};
    for (std::size_t l = 0; l < kLayers; ++l) {
        head.push_back({p(l) + "attn_hyper_connection.hc_norm.weight", {kHcHidden}});
        head.push_back({p(l) + "attn_hyper_connection.input_mix_weight_down.weight",
                        two(kHcLowrank, kHcHidden)});
        head.push_back({p(l) + "attn_hyper_connection.input_mix_weight_up.weight",
                        two(kHcHidden, kHcLowrank)});
        head.push_back({p(l) + "attn_hyper_connection.block_inject_weight.weight",
                        two(kHcCount, kHcHidden)});
    }
    write_shard("model-2.safetensors", head, {});

    std::vector<ShapeSpec> hyper3;
    for (std::size_t l = 0; l < kLayers; ++l) {
        hyper3.push_back({p(l) + "mlp_hyper_connection.hc_norm.weight", {kHcHidden}});
        hyper3.push_back({p(l) + "mlp_hyper_connection.input_mix_weight_down.weight",
                          two(kHcLowrank, kHcHidden)});
        hyper3.push_back({p(l) + "mlp_hyper_connection.input_mix_weight_up.weight",
                          two(kHcHidden, kHcLowrank)});
        hyper3.push_back({p(l) + "mlp_hyper_connection.block_inject_weight.weight",
                          two(kHcCount, kHcHidden)});
    }
    write_shard("model-3.safetensors", hyper3, {});

    std::vector<ShapeSpec> gdn;
    for (std::size_t l = 0; l + 1 < kLayers; ++l) {
        gdn.push_back({p(l) + "linear_attn.in_proj_qkv.weight", two(kConvDim, kHidden)});
        gdn.push_back({p(l) + "linear_attn.in_proj_z.weight", two(kValueDim, kHidden)});
        gdn.push_back({p(l) + "linear_attn.in_proj_b.weight", two(kValHeads, kHidden)});
        gdn.push_back({p(l) + "linear_attn.in_proj_a.weight", two(kValHeads, kHidden)});
        gdn.push_back({p(l) + "linear_attn.out_proj.weight", two(kHidden, kValueDim)});
        gdn.push_back({p(l) + "linear_attn.conv1d.weight", {kConvDim, 1, 4}});
        gdn.push_back({p(l) + "linear_attn.norm.weight", {kHeadDim}});
        gdn.push_back({p(l) + "linear_attn.dt_bias", {kValHeads}});
        gdn.push_back({p(l) + "linear_attn.A_log", {kValHeads}});
    }
    write_shard("model-4.safetensors", gdn, {});

    const std::size_t ql = kLayers - 1;
    write_shard("model-5.safetensors",
                {{p(ql) + "self_attn.q_proj.weight",
                  two(2 * kQsaHeads * kQsaHeadDim, kHidden)},
                 {p(ql) + "self_attn.k_proj.weight", two(kQsaKv * kQsaHeadDim, kHidden)},
                 {p(ql) + "self_attn.v_proj.weight", two(kQsaKv * kQsaHeadDim, kHidden)},
                 {p(ql) + "self_attn.o_proj.weight",
                  two(kHidden, kQsaHeads * kQsaHeadDim)},
                 {p(ql) + "self_attn.q_norm.weight", {kQsaHeadDim}},
                 {p(ql) + "self_attn.k_norm.weight", {kQsaHeadDim}},
                 {p(ql) + "self_attn.indexer.index_qk_proj.weight",
                  two(kIndexerHeads * kIndexerHeadDim + kIndexerHeadDim, kHidden)},
                 {p(ql) + "self_attn.indexer.q_layernorm.weight", {kIndexerHeadDim}},
                 {p(ql) + "self_attn.indexer.k_layernorm.weight", {kIndexerHeadDim}}},
                {});

    std::vector<ShapeSpec> moe;
    for (std::size_t l = 0; l < kLayers; ++l) {
        moe.push_back({p(l) + "mlp.gate.weight", two(kExperts, kHidden)});
        moe.push_back({p(l) + "mlp.shared_expert.gate_proj.weight",
                       two(kSharedInter, kHidden)});
        moe.push_back({p(l) + "mlp.shared_expert.up_proj.weight", two(kSharedInter, kHidden)});
        moe.push_back({p(l) + "mlp.shared_expert.down_proj.weight",
                       two(kHidden, kSharedInter)});
        moe.push_back({p(l) + "mlp.shared_expert_gate.weight", {1, kHidden}});
    }
    write_shard("model-6.safetensors", moe, {});

    std::vector<ShapeSpec> experts;
    for (std::size_t l = 0; l < kLayers; ++l) {
        experts.push_back({p(l) + "mlp.experts.gate_up_proj",
                           {kExperts, 2 * kMoeInter, kHidden}});
        experts.push_back({p(l) + "mlp.experts.down_proj", {kExperts, kHidden, kMoeInter}});
    }
    write_shard("model-7.safetensors", experts, {});

    const std::string pp = "model.language_model.layers.1.ple.";
    write_shard("model-8.safetensors",
                {{pp + "key_proj.weight", two(kHcHidden, kHidden)},
                 {pp + "value_proj.weight", two(kHidden, kHidden)},
                 {pp + "conv1d.weight", {kHcHidden, 1, 4}},
                 {pp + "norm_key.weight", {kHcHidden}},
                 {pp + "norm_query.weight", {kHcHidden}},
                 {pp + "norm_conv.weight", {kHcHidden}}},
                {{pp + "ple_embedding.layer_multipliers", {3}},
                 {pp + "ple_embedding.ngram_heads_vocab_sizes", {16}},
                 {pp + "ple_embedding.ngram_heads_offsets", {16}}});
}

static void test_load() {
    build_fixture();
    build_shards();

    auto cfg = q4e::read_qwen4exp_text_config(g_dir);
    check(cfg.ok(), "weights: config parses");
    if (!cfg.ok()) return;

    if (hipSetDevice(0) != hipSuccess) {
        check(false, "weights: gpu set");
        return;
    }
    auto arena_result = ps::gpu::GpuArena::create(0, 1ull << 30);
    check(arena_result.ok(), "weights: arena create");
    if (!arena_result.ok()) return;
    ps::gpu::GpuArena arena = arena_result.release();

    auto w = q4e::load_qwen4exp_weights_bf16(g_dir, arena, nullptr);
    check(w.ok(), "weights: load bf16");
    if (!w.ok()) {
        printf("  reason: %s\n", w.status().message().c_str());
        return;
    }
    auto& mw = w.value();

    check(mw.embed_tokens.rows == kVocab && mw.embed_tokens.cols == kHidden,
          "weights: embed_tokens shape");
    check(mw.lm_head.rows == kVocab && mw.lm_head.cols == kHidden, "weights: lm_head shape");
    check(!mw.lm_head_tied, "weights: lm_head not tied");
    check(mw.final_hyper.hc_norm.ndim() == 1 &&
              mw.final_hyper.hc_norm.dim(0) == kHcHidden,
          "weights: final hc_norm");
    check(mw.final_hyper.mix_down.rows == kHcLowrank &&
              mw.final_hyper.mix_down.cols == kHcHidden,
          "weights: final mix down");
    check(!mw.final_hyper.has_block_inject, "weights: final mixer has no block_inject");
    check(mw.layers.size() == kLayers, "weights: layer count");

    for (std::size_t l = 0; l < kLayers; ++l) {
        const auto& lw = mw.layers[l];
        const bool expect_gdn = (l != 3);
        check(lw.is_gdn == expect_gdn, "weights: layer topology matches config");
        check(lw.attn_hyper.has_block_inject, "weights: attn hyper has block_inject");
        check(lw.mlp_hyper.has_block_inject, "weights: mlp hyper has block_inject");
        check(lw.mlp_hyper.mix_down.rows == kHcLowrank, "weights: mlp hyper lowrank");

        if (expect_gdn) {
            check(lw.gdn.qkv.rows == kConvDim && lw.gdn.qkv.cols == kHidden,
                  "weights: gdn qkv shape");
            check(lw.gdn.z.rows == kValueDim, "weights: gdn z shape");
            check(lw.gdn.b.rows == kValHeads, "weights: gdn b shape");
            check(lw.gdn.out.cols == kValueDim, "weights: gdn out shape");
            check(lw.gdn.conv1d.ndim() == 3 && lw.gdn.conv1d.dim(0) == kConvDim &&
                      lw.gdn.conv1d.dim(2) == 4,
                  "weights: gdn conv1d rank-3");
            check(lw.gdn.norm.dim(0) == kHeadDim, "weights: gdn norm");
            check(lw.gdn.dt_bias.dim(0) == kValHeads, "weights: gdn dt_bias");
            check(lw.gdn.a_log.dim(0) == kValHeads, "weights: gdn A_log");
            check(!lw.qsa.q.bf16.data<void>(), "weights: gdn layer has no qsa");
        } else {
            check(lw.qsa.q.rows == 2 * kQsaHeads * kQsaHeadDim &&
                      lw.qsa.q.cols == kHidden,
                  "weights: qsa q_proj shape");
            check(lw.qsa.k.rows == kQsaKv * kQsaHeadDim, "weights: qsa k_proj shape");
            check(lw.qsa.o.cols == kQsaHeads * kQsaHeadDim, "weights: qsa o_proj shape");
            check(lw.qsa.q_norm.dim(0) == kQsaHeadDim, "weights: qsa q_norm");
            check(lw.qsa.indexer.present, "weights: indexer present on QSA layer");
            check(lw.qsa.indexer.index_qk.rows ==
                      kIndexerHeads * kIndexerHeadDim + kIndexerHeadDim,
                  "weights: index_qk shape");
            check(lw.qsa.indexer.q_layernorm.dim(0) == kIndexerHeadDim,
                  "weights: indexer q layernorm");
            check(!lw.gdn.qkv.bf16.data<void>(), "weights: qsa layer has no gdn");
        }

        check(lw.moe.router.rows == kExperts && lw.moe.router.cols == kHidden,
              "weights: router shape");
        check(lw.moe.expert_count == kExperts, "weights: expert count from rank-3");
        check(lw.moe.shared.gate.rows == kSharedInter, "weights: shared expert gate");
        check(lw.moe.shared.down.cols == kSharedInter, "weights: shared expert down");
        check(lw.moe.shared.gate_scale.ndim() == 2 &&
                  lw.moe.shared.gate_scale.dim(0) == 1,
              "weights: shared gate scale");

        if (l == 1) {
            check(lw.ple.present, "weights: PLE present on layer 1");
            check(lw.ple.key_proj.rows == kHcHidden, "weights: ple key_proj");
            check(lw.ple.value_proj.rows == kHidden, "weights: ple value_proj");
            check(lw.ple.conv1d.ndim() == 3 && lw.ple.conv1d.dim(2) == 4,
                  "weights: ple conv1d rank-3");
            check(lw.ple.norm_key.dim(0) == kHcHidden, "weights: ple norm_key");
            check(lw.ple.layer_multipliers.size() == 3, "weights: ple multipliers i64");
            check(lw.ple.vocab_sizes.size() == 16, "weights: ple vocab sizes i64");
            check(lw.ple.offsets.size() == 16, "weights: ple offsets i64");
        } else {
            check(!lw.ple.present, "weights: PLE absent outside layer 1");
        }
    }
}

static void test_missing_tensor() {
    const std::string dir = "/tmp/ps_q4e_weights_missing_" + std::to_string(getpid());
    fs::remove_all(dir);
    fs::create_directories(dir);
    {
        std::ofstream f(dir + "/config.json");
        f << R"({"text_config":{"vocab_size":8,"hidden_size":16,"num_hidden_layers":1,)"
             R"("num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8,)"
             R"("hc_count":4,"hc_lowrank":8,"full_attention_interval":4}})";
    }
    auto ar = ps::gpu::GpuArena::create(0, 1ull << 30);
    check(ar.ok(), "weights: arena for missing test");
    if (ar.ok()) {
        ps::gpu::GpuArena arena = ar.release();
        auto w = q4e::load_qwen4exp_weights_bf16(dir, arena, nullptr);
        check(!w.ok(), "weights: absent tensors rejected");
        if (!w.ok()) printf("  reason: %s\n", w.status().message().c_str());
    }
    fs::remove_all(dir);
}

int main() {
    printf("=== Qwen4Exp weights load (synthetic fixture) ===\n");
    test_load();
    test_missing_tensor();
    fs::remove_all(g_dir);
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
