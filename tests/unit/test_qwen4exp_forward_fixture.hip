#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/inference/state/gdn_state_pool.h>
#include <phaseshift/inference/state/paged_kv_pool.h>
#include <phaseshift/inference/state/paged_sequence_state.h>
#include <phaseshift/inference/state/sequence_slot_pool.h>
#include <phaseshift/io/safetensors_writer.h>
#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/models/qwen4exp/hyper/gated_residual.h>
#include <phaseshift/models/qwen4exp/model/lower_to_primitives.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_config.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_weights.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/device_program.h>
#include <phaseshift/runtime/program/program.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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
namespace rt = ::ps::runtime;
namespace gpu = ::ps::gpu;
namespace q4e = ::ps::qwen4exp;
namespace hy = ::ps::qwen4exp::hyper;
namespace state = ::ps::state;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

float bits_bf16(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Metrics {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rmse = 0.0;
    bool non_finite = false;
};

Metrics compare(const float* got, const float* ref, size_t count) {
    Metrics m;
    double sum2 = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double g = got[i];
        if (!std::isfinite(g)) {
            m.non_finite = true;
            return m;
        }
        const double r = ref[i];
        const double da = std::fabs(g - r);
        const double dr = da / std::max(1e-6, std::fabs(r));
        m.max_abs = std::max(m.max_abs, da);
        m.max_rel = std::max(m.max_rel, dr);
        sum2 += da * da;
    }
    m.rmse = count != 0 ? std::sqrt(sum2 / static_cast<double>(count)) : 0.0;
    return m;
}

void print_metrics(const char* tag, const Metrics& m) {
    std::printf("METRIC %s max_abs=%.9g max_rel=%.9g RMSE=%.9g%s\n", tag, m.max_abs,
                m.max_rel, m.rmse, m.non_finite ? " NON-FINITE" : "");
}

void check_metrics(const char* tag, const Metrics& m, double abs_tol, double rel_tol,
                   double rmse_tol) {
    print_metrics(tag, m);
    const bool ok = !m.non_finite && m.max_abs <= abs_tol && m.max_rel <= rel_tol &&
                    m.rmse <= rmse_tol;
    check(ok, std::string(tag) + " within tolerance");
}

struct Buf {
    void* dev = nullptr;
    size_t bytes = 0;
    ~Buf() { free_dev(); }
    void free_dev() {
        if (dev) {
            (void)hipFree(dev);
            dev = nullptr;
        }
    }
    bool alloc(size_t b) {
        free_dev();
        bytes = b;
        return hipMalloc(&dev, b) == hipSuccess;
    }
    bool fill(const void* src, size_t b) {
        return hipMemcpy(dev, src, b, hipMemcpyHostToDevice) == hipSuccess;
    }
    bool read(void* dst, size_t b) const {
        return hipMemcpy(dst, dev, b, hipMemcpyDeviceToHost) == hipSuccess;
    }
};

constexpr std::uint32_t kVocab = 32;
constexpr std::uint32_t kHidden = 64;
constexpr std::uint32_t kLayers = 4;
constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kLowrank = 8;
constexpr std::uint32_t kHcHidden = kStreams * kHidden;
constexpr std::uint32_t kKeyHeads = 4;
constexpr std::uint32_t kValHeads = 8;
constexpr std::uint32_t kHeadDim = 8;
constexpr std::uint32_t kQk = kKeyHeads * kHeadDim;
constexpr std::uint32_t kVal = kValHeads * kHeadDim;
constexpr std::uint32_t kConvF = 2 * kQk + kVal;
constexpr std::uint32_t kConvKernel = 4;
constexpr std::uint32_t kConvHistory = 3;
constexpr std::uint32_t kQsaHeads = 4;
constexpr std::uint32_t kQsaKv = 2;
constexpr std::uint32_t kQsaHeadDim = 16;
constexpr std::uint32_t kQsaRotary = 4;
constexpr std::uint32_t kIndexerHeads = 2;
constexpr std::uint32_t kIndexerHeadDim = 8;
constexpr std::uint32_t kExperts = 4;
constexpr std::uint32_t kTopK = 2;
constexpr std::uint32_t kMoeInter = 16;
constexpr std::uint32_t kSharedInter = 16;
constexpr std::uint32_t kGdnLayers = 3;
constexpr std::uint32_t kQsaLayers = 1;
constexpr std::uint32_t kMaxTokens = 5;
constexpr std::uint32_t kPageTokens = 16;
constexpr std::uint32_t kNumPages = 8;
constexpr std::uint32_t kMaxSequences = 4;
constexpr float kEps = 1.0e-6f;
constexpr float kTheta = 10000000.0f;

std::string g_dir;
std::mt19937_64 g_rng(0xC0FFEEu);

float uniform11() {
    return static_cast<float>(static_cast<int32_t>(g_rng() % 2001u) - 1000) / 1000.0f;
}

using ShapeSpec = std::pair<std::string, std::vector<std::size_t>>;

void write_fixture_shard() {
    std::vector<ShapeSpec> bf16;
    std::vector<ShapeSpec> i64;
    std::vector<std::vector<float>> values;
    const auto count_of = [](const std::vector<std::size_t>& shape) {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        return n;
    };
    const auto add = [&](const std::string& name, const std::vector<std::size_t>& shape,
                         float scale) {
        std::vector<float> v(count_of(shape));
        for (size_t i = 0; i < v.size(); ++i) v[i] = scale * uniform11();
        bf16.emplace_back(name, shape);
        values.push_back(std::move(v));
    };
    const auto add_with = [&](const std::string& name,
                              const std::vector<std::size_t>& shape,
                              const std::vector<float>& fixed) {
        bf16.emplace_back(name, shape);
        values.push_back(fixed);
    };
    const auto inv_sqrt = [](float n) { return 1.0f / std::sqrt(n); };
    const auto one_plus = [&](size_t n) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = 0.1f * uniform11();
        return v;
    };
    const auto near_one = [&](size_t n) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = 1.0f + 0.05f * uniform11();
        return v;
    };

    std::vector<float> embed(static_cast<std::size_t>(kVocab) * kHidden);
    for (size_t i = 0; i < embed.size(); ++i) embed[i] = uniform11();
    add_with("model.language_model.embed_tokens.weight", {kVocab, kHidden}, embed);
    add("lm_head.weight", {kVocab, kHidden}, inv_sqrt(static_cast<float>(kHidden)));
    add("model.language_model.hyper_connection_mixer.hc_norm.weight", {kHcHidden}, 0.0f);
    {
        std::vector<float> v(kHcHidden);
        for (size_t i = 0; i < v.size(); ++i) v[i] = 0.1f * uniform11();
        values.back() = v;
    }
    add("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        {kLowrank, kHcHidden}, inv_sqrt(static_cast<float>(kHcHidden)));
    add("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        {kHcHidden, kLowrank}, inv_sqrt(static_cast<float>(kLowrank)));

    const std::string lp = "model.language_model.layers.";
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        const std::string p = lp + std::to_string(l) + ".";
        for (const std::string& side : {"attn_hyper_connection.",
                                        "mlp_hyper_connection."}) {
            const std::string h = p + side;
            add(h + "hc_norm.weight", {kHcHidden}, 0.0f);
            values.back() = one_plus(kHcHidden);
            add(h + "input_mix_weight_down.weight", {kLowrank, kHcHidden},
                inv_sqrt(static_cast<float>(kHcHidden)));
            add(h + "input_mix_weight_up.weight", {kHcHidden, kLowrank},
                inv_sqrt(static_cast<float>(kLowrank)));
            add(h + "block_inject_weight.weight", {kStreams, kHcHidden},
                inv_sqrt(static_cast<float>(kHcHidden)));
        }
        if (l + 1 < kLayers) {
            const std::string g = p + "linear_attn.";
            add(g + "in_proj_qkv.weight", {kConvF, kHidden},
                inv_sqrt(static_cast<float>(kHidden)));
            add(g + "in_proj_z.weight", {kVal, kHidden},
                inv_sqrt(static_cast<float>(kHidden)));
            add(g + "in_proj_b.weight", {kValHeads, kHidden},
                inv_sqrt(static_cast<float>(kHidden)));
            add(g + "in_proj_a.weight", {kValHeads, kHidden},
                inv_sqrt(static_cast<float>(kHidden)));
            add(g + "out_proj.weight", {kHidden, kVal},
                inv_sqrt(static_cast<float>(kVal)));
            add(g + "conv1d.weight", {kConvF, 1, kConvKernel}, 0.5f);
            add_with(g + "norm.weight", {kHeadDim}, near_one(kHeadDim));
            add(g + "dt_bias", {kValHeads}, 0.1f);
            add(g + "A_log", {kValHeads}, 0.5f);
        } else {
            const std::string a = p + "self_attn.";
            add(a + "q_proj.weight", {2 * kQsaHeads * kQsaHeadDim, kHidden},
                inv_sqrt(static_cast<float>(kHidden)));
            add(a + "k_proj.weight", {kQsaKv * kQsaHeadDim, kHidden},
                inv_sqrt(static_cast<float>(kHidden)));
            add(a + "v_proj.weight", {kQsaKv * kQsaHeadDim, kHidden},
                inv_sqrt(static_cast<float>(kHidden)));
            add(a + "o_proj.weight", {kHidden, kQsaHeads * kQsaHeadDim},
                inv_sqrt(static_cast<float>(kQsaHeads * kQsaHeadDim)));
            add_with(a + "q_norm.weight", {kQsaHeadDim}, one_plus(kQsaHeadDim));
            add_with(a + "k_norm.weight", {kQsaHeadDim}, one_plus(kQsaHeadDim));
            add(a + "indexer.index_qk_proj.weight",
                {kIndexerHeads * kIndexerHeadDim + kIndexerHeadDim, kHidden},
                inv_sqrt(static_cast<float>(kHidden)));
            add_with(a + "indexer.q_layernorm.weight", {kIndexerHeadDim},
                     one_plus(kIndexerHeadDim));
            add_with(a + "indexer.k_layernorm.weight", {kIndexerHeadDim},
                     one_plus(kIndexerHeadDim));
        }
        const std::string m = p + "mlp.";
        add(m + "gate.weight", {kExperts, kHidden}, inv_sqrt(static_cast<float>(kHidden)));
        add(m + "shared_expert.gate_proj.weight", {kSharedInter, kHidden},
            inv_sqrt(static_cast<float>(kHidden)));
        add(m + "shared_expert.up_proj.weight", {kSharedInter, kHidden},
            inv_sqrt(static_cast<float>(kHidden)));
        add(m + "shared_expert.down_proj.weight", {kHidden, kSharedInter},
            inv_sqrt(static_cast<float>(kSharedInter)));
        add(m + "shared_expert_gate.weight", {1, kHidden},
            inv_sqrt(static_cast<float>(kHidden)));
        add(m + "experts.gate_up_proj", {kExperts, 2 * kMoeInter, kHidden},
            inv_sqrt(static_cast<float>(kHidden)));
        add(m + "experts.down_proj", {kExperts, kHidden, kMoeInter},
            inv_sqrt(static_cast<float>(kMoeInter)));
    }

    const std::string pp = lp + "1.ple.";
    add(pp + "key_proj.weight", {kHcHidden, kHidden}, inv_sqrt(static_cast<float>(kHidden)));
    add(pp + "value_proj.weight", {kHidden, kHidden}, inv_sqrt(static_cast<float>(kHidden)));
    add(pp + "conv1d.weight", {kHcHidden, 1, kConvKernel}, 0.5f);
    add_with(pp + "norm_key.weight", {kHcHidden}, near_one(kHcHidden));
    add_with(pp + "norm_query.weight", {kHcHidden}, near_one(kHcHidden));
    add_with(pp + "norm_conv.weight", {kHcHidden}, near_one(kHcHidden));
    i64.emplace_back(pp + "ple_embedding.layer_multipliers", std::vector<std::size_t>{3});
    i64.emplace_back(pp + "ple_embedding.ngram_heads_vocab_sizes",
                     std::vector<std::size_t>{16});
    i64.emplace_back(pp + "ple_embedding.ngram_heads_offsets",
                     std::vector<std::size_t>{16});

    auto w = ps::io::SafetensorsWriter::create(g_dir + "/model-1.safetensors");
    if (!w.ok()) {
        std::printf("FAIL: writer create\n");
        ++g_fail;
        return;
    }
    for (const auto& [name, shape] : bf16) {
        if (!w.value().plan_tensor(name, ps::io::SType::BF16, shape).ok()) {
            std::printf("FAIL: plan %s\n", name.c_str());
            ++g_fail;
        }
    }
    for (const auto& [name, shape] : i64) {
        if (!w.value().plan_tensor(name, ps::io::SType::I64, shape).ok()) {
            std::printf("FAIL: plan i64 %s\n", name.c_str());
            ++g_fail;
        }
    }
    if (!w.value().write_header().ok()) {
        std::printf("FAIL: write header\n");
        ++g_fail;
        return;
    }
    for (size_t i = 0; i < bf16.size(); ++i) {
        std::vector<std::uint16_t> bits(values[i].size());
        for (size_t j = 0; j < bits.size(); ++j) bits[j] = bf16_bits(values[i][j]);
        if (!w.value()
                 .write_tensor(bf16[i].first, bits.data(), bits.size() * 2)
                 .ok()) {
            std::printf("FAIL: write %s\n", bf16[i].first.c_str());
            ++g_fail;
        }
    }
    for (const auto& [name, shape] : i64) {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        std::vector<std::int64_t> meta(n);
        for (std::size_t j = 0; j < n; ++j) meta[j] = static_cast<std::int64_t>(j) + 1;
        if (!w.value().write_tensor(name, meta.data(), meta.size() * 8).ok()) {
            std::printf("FAIL: write i64 %s\n", name.c_str());
            ++g_fail;
        }
    }
    if (!w.value().finish().ok()) {
        std::printf("FAIL: finish\n");
        ++g_fail;
    }
}

void build_fixture() {
    g_dir = "/tmp/ps_q4e_forward_" + std::to_string(getpid());
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
          << R"(,"linear_conv_kernel_dim":)" << kConvKernel
          << R"(,"num_experts":)" << kExperts
          << R"(,"num_experts_per_tok":)" << kTopK
          << R"(,"moe_intermediate_size":)" << kMoeInter
          << R"(,"shared_expert_intermediate_size":)" << kSharedInter
          << R"(,"hc_count":)" << kStreams << R"(,"hc_lowrank":)" << kLowrank
          << R"(,"ngram_size":3,"heads_per_ngram":8,)"
          << R"("ngram_vocab_size_base":20000000,)"
          << R"("make_ngram_vocab_size_divisible_by":128,)"
          << R"("split_ngram_parts":128,"ple_layer_ids":[2],)"
          << R"("ple_embed_dim":)" << kHidden
          << R"(,"ple_conv_kernel_size":4,)"
          << R"("indexer_n_heads":)" << kIndexerHeads
          << R"(,"indexer_kv_heads":1,"indexer_head_dim":)" << kIndexerHeadDim
          << R"(,"indexer_budget":16,"indexer_compress_ratio":4,)"
          << R"("rms_norm_eps":1e-6,"output_gate_type":"sigmoid",)"
          << R"("rope_parameters":{"rope_theta":10000000,)"
          << R"("partial_rotary_factor":0.25},)"
          << R"("eos_token_id":7,)"
          << R"("layer_types":["linear_attention","linear_attention",)"
          << R"("linear_attention","full_attention"]}})";
    }
    {
        std::ofstream f(g_dir + "/model.safetensors.index.json");
        f << R"({"weight_map":{)";
        bool first = true;
        const auto emit = [&](const std::string& name) {
            if (!first) f << ",";
            first = false;
            f << "\"" << name << "\":\"model-1.safetensors\"";
        };
        emit("model.language_model.embed_tokens.weight");
        emit("lm_head.weight");
        emit("model.language_model.hyper_connection_mixer.hc_norm.weight");
        emit("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight");
        emit("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight");
        const std::string lp = "model.language_model.layers.";
        for (std::uint32_t l = 0; l < kLayers; ++l) {
            const std::string p = lp + std::to_string(l) + ".";
            for (const std::string& side : {"attn_hyper_connection.",
                                            "mlp_hyper_connection."}) {
                const std::string h = p + side;
                emit(h + "hc_norm.weight");
                emit(h + "input_mix_weight_down.weight");
                emit(h + "input_mix_weight_up.weight");
                emit(h + "block_inject_weight.weight");
            }
            if (l + 1 < kLayers) {
                for (const std::string& field :
                     {"in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
                      "in_proj_a.weight", "out_proj.weight", "conv1d.weight",
                      "norm.weight", "dt_bias", "A_log"}) {
                    emit(p + "linear_attn." + field);
                }
            } else {
                for (const std::string& field :
                     {"q_proj.weight", "k_proj.weight", "v_proj.weight",
                      "o_proj.weight", "q_norm.weight", "k_norm.weight",
                      "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight",
                      "indexer.k_layernorm.weight"}) {
                    emit(p + "self_attn." + field);
                }
            }
            for (const std::string& field :
                 {"gate.weight", "shared_expert.gate_proj.weight",
                  "shared_expert.up_proj.weight", "shared_expert.down_proj.weight",
                  "shared_expert_gate.weight", "experts.gate_up_proj",
                  "experts.down_proj"}) {
                emit(p + "mlp." + field);
            }
        }
        const std::string pp = lp + "1.ple.";
        for (const std::string& field :
             {"key_proj.weight", "value_proj.weight", "conv1d.weight",
              "norm_key.weight", "norm_query.weight", "norm_conv.weight",
              "ple_embedding.layer_multipliers", "ple_embedding.ngram_heads_vocab_sizes",
              "ple_embedding.ngram_heads_offsets"}) {
            emit(pp + field);
        }
        f << "}}";
    }
    write_fixture_shard();
}

bool read_f32_device(const void* dev, std::size_t count, std::vector<float>& out) {
    out.resize(count);
    return hipMemcpy(out.data(), dev, count * sizeof(float),
                     hipMemcpyDeviceToHost) == hipSuccess;
}

bool copy_bf16_to_f32(const void* dev, std::size_t count, std::vector<float>& out) {
    std::vector<std::uint16_t> bits(count);
    if (hipMemcpy(bits.data(), dev, count * sizeof(std::uint16_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }
    out.resize(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = bits_bf16(bits[i]);
    return true;
}

struct DeviceCtx {
    rt::DeviceBatchContext h{};
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* err = nullptr;
    void* values_dev = nullptr;
    void* weights_dev = nullptr;
    void* params_dev = nullptr;
    void* workspace = nullptr;
    void* states_dev = nullptr;
    ~DeviceCtx() {
        void* bufs[] = {ctx_dev, scratch, err, values_dev, weights_dev, params_dev,
                        workspace, states_dev};
        for (void* q : bufs)
            if (q) (void)hipFree(q);
    }
};

bool run_program(const rt::Program& plan,
                 const std::vector<rt::WeightSlot>& host_weights,
                 const std::vector<rt::StaticParameterSlot>& host_params,
                 DeviceCtx& dc, const rt::DeviceBatchContext& ctx,
                 const ::ps::kernel::ModelDispatchStateView& state_view,
                 rt::ValueId tokens_id, hipStream_t stream) {
    dc.h = ctx;
    if (dc.ctx_dev == nullptr) {
        if (hipMalloc(&dc.ctx_dev, sizeof(dc.h)) != hipSuccess) return false;
        if (hipMalloc(&dc.scratch, (size_t)256 * 4096 * sizeof(float)) != hipSuccess)
            return false;
        if (hipMalloc(&dc.err, sizeof(uint32_t)) != hipSuccess) return false;

        std::vector<rt::DeviceValueBinding> vals(plan.values.size());
        for (size_t i = 0; i < plan.values.size(); ++i) {
            const auto& s = plan.values[i];
            auto& d = vals[i];
            d.logical_value = s.logical_value.id;
            d.storage = s.storage;
            d.dtype = s.dtype;
            d.row_domain = s.row_domain;
            d.slot = s.slot;
            d.feature_count = s.feature_count;
            d.row_stride = s.row_stride;
            d.offset = s.offset;
            d.bytes = s.bytes;
        }
        for (size_t i = 0; i < plan.values.size(); ++i) {
            const auto& s = plan.values[i];
            if ((s.storage == rt::ValueStorage::EXTERNAL_INPUT ||
                 s.storage == rt::ValueStorage::EXTERNAL_OUTPUT) &&
                s.offset == 0) {
                std::printf("FAIL: external value %u lacks device pointer\n",
                            s.logical_value.id);
                return false;
            }
            (void)tokens_id;
        }
        if (hipMalloc(&dc.values_dev, vals.size() * sizeof(rt::DeviceValueBinding)) !=
            hipSuccess)
            return false;
        if (hipMemcpy(dc.values_dev, vals.data(),
                      vals.size() * sizeof(rt::DeviceValueBinding),
                      hipMemcpyHostToDevice) != hipSuccess) {
            return false;
        }
        if (!host_weights.empty()) {
            if (hipMalloc(&dc.weights_dev,
                          host_weights.size() * sizeof(rt::WeightSlot)) != hipSuccess)
                return false;
            if (hipMemcpy(dc.weights_dev, host_weights.data(),
                          host_weights.size() * sizeof(rt::WeightSlot),
                          hipMemcpyHostToDevice) != hipSuccess)
                return false;
        }
        if (!host_params.empty()) {
            if (hipMalloc(&dc.params_dev,
                          host_params.size() * sizeof(rt::StaticParameterSlot)) !=
                hipSuccess)
                return false;
            if (hipMemcpy(dc.params_dev, host_params.data(),
                          host_params.size() * sizeof(rt::StaticParameterSlot),
                          hipMemcpyHostToDevice) != hipSuccess)
                return false;
        }
        if (plan.workspace.total_bytes != 0) {
            if (hipMalloc(&dc.workspace, plan.workspace.total_bytes) != hipSuccess)
                return false;
        }
        if (!plan.states.empty()) {
            std::vector<rt::DeviceStateBinding> hs(plan.states.size());
            for (size_t i = 0; i < plan.states.size(); ++i) {
                hs[i].slot = plan.states[i].slot;
                hs[i].kind = static_cast<uint8_t>(plan.states[i].kind);
                hs[i].state_index = plan.states[i].state_index;
            }
            if (hipMalloc(&dc.states_dev,
                          hs.size() * sizeof(rt::DeviceStateBinding)) != hipSuccess)
                return false;
            if (hipMemcpy(dc.states_dev, hs.data(),
                          hs.size() * sizeof(rt::DeviceStateBinding),
                          hipMemcpyHostToDevice) != hipSuccess)
                return false;
        }
    }
    if (hipMemcpy(dc.ctx_dev, &dc.h, sizeof(dc.h), hipMemcpyHostToDevice) !=
        hipSuccess)
        return false;

    rt::DeviceProgramView view{};
    view.values = static_cast<const rt::DeviceValueBinding*>(dc.values_dev);
    view.value_count = static_cast<uint32_t>(plan.values.size());
    view.weight_table = dc.weights_dev;
    view.weight_slot_count = static_cast<uint32_t>(host_weights.size());
    view.parameter_table = dc.params_dev;
    view.parameter_slot_count = static_cast<uint32_t>(host_params.size());
    view.workspace_base = reinterpret_cast<uint64_t>(dc.workspace);
    view.states = static_cast<const rt::DeviceStateBinding*>(dc.states_dev);
    view.state_count = static_cast<uint32_t>(plan.states.size());

    for (size_t di = 0; di < plan.dispatches.size(); ++di) {
        ps::gpu::discard_cleanup_result(
            hipMemsetAsync(dc.err, 0, sizeof(uint32_t), stream));
        const ps::Status st = ::ps::kernel::launch_model_dispatch_correctness(
            view, plan.dispatches[di],
            reinterpret_cast<rt::DeviceBatchContext*>(dc.ctx_dev), state_view,
            static_cast<float*>(dc.scratch), static_cast<uint32_t*>(dc.err), stream);
        if (!st.ok()) {
            std::printf("FAIL: dispatch %zu (%s) launch: %s\n", di,
                        rt::kernel_to_name(plan.dispatches[di].kernel_id),
                        st.message().c_str());
            return false;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) return false;
        uint32_t e = 0;
        if (hipMemcpy(&e, dc.err, 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
        if (e != 0u) {
            std::printf("FAIL: dispatch %zu (%s) status=%u\n", di,
                        rt::kernel_to_name(plan.dispatches[di].kernel_id), e);
            return false;
        }
    }
    return true;
}

void patch_external(rt::Program& plan, rt::ValueId id, void* dev_ptr) {
    for (auto& v : plan.values) {
        if (v.logical_value.id == id.id) {
            v.offset = reinterpret_cast<uint64_t>(dev_ptr);
        }
    }
}

rt::ValueId node_output_by_debug_name(const rt::PrimitiveGraph& graph,
                                       const char* name) {
    for (const auto& gn : graph.nodes) {
        if (gn.debug_name == name && !gn.outputs.empty()) return gn.outputs[0];
    }
    return rt::ValueId{UINT32_MAX};
}

bool read_value_rows(const rt::Program& plan, const void* workspace, rt::ValueId id,
                     uint32_t rows, std::vector<float>& out) {
    const rt::ValueBinding* b = plan.find_value(id);
    if (b == nullptr) return false;
    const uint32_t stride = b->row_stride != 0 ? b->row_stride : b->feature_count;
    if (b->dtype != rt::ValueDType::F32 && b->dtype != rt::ValueDType::BF16)
        return false;
    const size_t elems = static_cast<size_t>(rows) * stride;
    const size_t base = b->storage == rt::ValueStorage::WORKSPACE
                            ? reinterpret_cast<size_t>(workspace) + b->offset
                            : static_cast<size_t>(b->offset);
    if (b->storage == rt::ValueStorage::WORKSPACE && workspace == nullptr)
        return false;
    if (b->dtype == rt::ValueDType::F32) {
        std::vector<float> strided(elems);
        if (hipMemcpy(strided.data(), reinterpret_cast<const void*>(base),
                      elems * 4, hipMemcpyDeviceToHost) != hipSuccess)
            return false;
        out.resize(static_cast<size_t>(rows) * b->feature_count);
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t c = 0; c < b->feature_count; ++c)
                out[static_cast<size_t>(r) * b->feature_count + c] =
                    strided[static_cast<size_t>(r) * stride + c];
        return true;
    }
    return copy_bf16_to_f32(reinterpret_cast<const void*>(base), elems, out);
}

struct StateSnap {
    std::vector<uint8_t> conv;
    std::vector<uint8_t> recurrent;
    std::vector<uint8_t> kv;
};

bool capture_state(const ::ps::kernel::ModelDispatchStateView& sv, uint32_t slot,
                   const state::PagedSequenceState& seq, StateSnap& out) {
    const auto& g = sv.gdn;
    const std::size_t conv_bytes =
        static_cast<std::size_t>(g.conv_slot_stride) * sizeof(ps::bf16_t);
    out.conv.resize(conv_bytes);
    const auto* conv_base = reinterpret_cast<const uint8_t*>(g.conv_base) +
                            static_cast<std::size_t>(slot) * g.conv_slot_stride *
                                sizeof(ps::bf16_t);
    if (hipMemcpy(out.conv.data(), conv_base, conv_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    const std::size_t rec_bytes =
        static_cast<std::size_t>(g.recurrent_slot_stride) * sizeof(ps::fp16_t);
    out.recurrent.resize(rec_bytes);
    const auto* rec_base = reinterpret_cast<const uint8_t*>(g.recurrent_base) +
                           static_cast<std::size_t>(slot) * g.recurrent_slot_stride *
                               sizeof(ps::fp16_t);
    if (hipMemcpy(out.recurrent.data(), rec_base, rec_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    const auto& kv = sv.kv_bf16;
    const std::size_t page_bytes = static_cast<std::size_t>(kv.elems_per_page) *
                                   sizeof(ps::bf16_t);
    out.kv.assign(seq.block_table.size() * page_bytes, 0);
    for (std::size_t i = 0; i < seq.block_table.size(); ++i) {
        const auto* page_base = reinterpret_cast<const uint8_t*>(kv.k_pool) +
                                static_cast<std::size_t>(seq.block_table[i]) *
                                    page_bytes;
        if (hipMemcpy(out.kv.data() + i * page_bytes, page_base, page_bytes,
                      hipMemcpyDeviceToHost) != hipSuccess)
            return false;
    }
    return true;
}

bool differs(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) return true;
    return std::memcmp(a.data(), b.data(), a.size()) != 0;
}

}  // namespace

int main() {
    std::printf("test_qwen4exp_forward_fixture\n");
    std::printf(
        "scope: MoE (mlp sub-layer) 未接続 — GR + GDN/QSA + collapse + lm_head の"
        "層間入出力と state 更新のみを検証\n");

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    build_fixture();
    if (g_fail != 0) {
        fs::remove_all(g_dir);
        return 1;
    }

    auto arena_result = gpu::GpuArena::create(0, 1ull << 30);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        fs::remove_all(g_dir);
        return 77;
    }
    gpu::GpuArena arena = arena_result.release();

    auto cfg_result = q4e::read_qwen4exp_text_config(g_dir);
    check(cfg_result.ok(), std::string("config parses") +
                               (cfg_result.ok() ? ""
                                                : " (" + cfg_result.status().message() +
                                                      ")"));
    if (!cfg_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    const q4e::Qwen4ExpTextConfig cfg = cfg_result.release();

    q4e::Qwen4ExpLoadOptions load_options;
    load_options.load_expert_weights = true;
    auto weights_result =
        q4e::load_qwen4exp_weights_bf16(g_dir, arena, stream, load_options);
    check(weights_result.ok(),
          std::string("weights load with expert slices") +
              (weights_result.ok()
                   ? ""
                   : " (" + weights_result.status().message() + ")"));
    if (!weights_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    q4e::Qwen4ExpModelWeights weights = weights_result.release();

    bool expert_slices_ok = true;
    for (const auto& lw : weights.layers) {
        if (lw.moe.expert_gate_up.size() != kExperts ||
            lw.moe.expert_down.size() != kExperts) {
            expert_slices_ok = false;
            break;
        }
        if (lw.moe.expert_gate_up[0].rows != 2 * kMoeInter ||
            lw.moe.expert_gate_up[0].cols != kHidden ||
            lw.moe.expert_down[0].rows != kHidden ||
            lw.moe.expert_down[0].cols != kMoeInter) {
            expert_slices_ok = false;
            break;
        }
    }
    check(expert_slices_ok, "expert rank-2 slices loaded for every layer");

    const std::vector<std::uint32_t> tap_layers{0, 1, 2, 3};
    q4e::Qwen4ExpLowerOptions lower_options;
    lower_options.connect_moe = false;
    lower_options.hidden_taps = tap_layers;
    auto lower_result = q4e::lower_qwen4exp_to_primitives(cfg, weights, lower_options);
    check(lower_result.ok(),
          std::string("lowering (MoE 未接続, hidden taps)") +
              (lower_result.ok() ? ""
                                 : " (" + lower_result.status().message() + ")"));
    if (!lower_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    q4e::Qwen4ExpLoweredPrimitives lowered = lower_result.release();

    rt::ProgramBuildOptions build_options;
    build_options.validate_workspace = true;
    build_options.no_workspace_reuse = true;
    build_options.max_token_rows = kMaxTokens;
    build_options.max_output_rows = 1;
    const rt::WeightTableView weight_view{lowered.weights.data(),
                                           lowered.weights.size()};
    auto built = rt::build_program(lowered.graph, weight_view, rt::RowBucket::R16,
                                   rt::ExecutionClass::PREFILL, build_options);
    check(built.ok(), std::string("build_program") +
                          (built.ok() ? ""
                                      : " (" + built.status().message() + ")"));
    if (!built.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    rt::Program plan = built.release();
    plan.parameter_slot_count = static_cast<uint32_t>(lowered.parameters.size());

    auto seq_pool_result = state::SequenceSlotPool::create(arena, kMaxSequences, 4);
    const state::GdnStatePoolLayout gdn_layout = state::GdnStatePoolLayout::make(
        kGdnLayers, kConvF, kConvHistory, kValHeads, kHeadDim, kHeadDim);
    auto gdn_result = state::GdnStatePool::create(arena, kMaxSequences, gdn_layout);
    auto kv_result = state::PagedKVPool::create(arena, kNumPages, kPageTokens,
                                                kQsaLayers, kQsaKv, kQsaHeadDim,
                                                ps::kernel::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_result.ok() && kv_result.ok(), "pool create");
    if (!seq_pool_result.ok() || !gdn_result.ok() || !kv_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    state::SequenceSlotPool seq_pool = seq_pool_result.release();
    state::GdnStatePool gdn_pool = gdn_result.release();
    state::PagedKVPool kv_pool = kv_result.release();

    ::ps::kernel::ModelDispatchStateView state_view{};
    state_view.block_table = seq_pool.device_view();
    state_view.gdn = gdn_pool.device_view();
    state_view.kv_bf16 = state::make_paged_kv_bf16_view(kv_pool);
    state_view.kv_dtype = 0;

    const rt::ValueId tokens_id = lowered.graph.external_inputs[0];
    const rt::ValueId hidden_id = lowered.graph.external_outputs[0];
    const rt::ValueId logits_id = lowered.graph.external_outputs[1];
    const rt::ValueId sampled_id = lowered.graph.external_outputs[2];
    const rt::ValueId final_hidden_id = lowered.graph.external_outputs[3];
    std::vector<rt::ValueId> tap_ids;
    for (std::size_t t = 0; t < tap_layers.size(); ++t) {
        tap_ids.push_back(lowered.graph.external_outputs[4 + t]);
    }
    const rt::ValueId mixed_id =
        node_output_by_debug_name(lowered.graph, "L0.attn.hyper_mixed");
    const rt::ValueId inject_id =
        node_output_by_debug_name(lowered.graph, "L0.attn.hyper_inject");
    const rt::ValueId sub_out_id =
        node_output_by_debug_name(lowered.graph, "L0.attn.gdn_out");
    check(mixed_id.id != UINT32_MAX && inject_id.id != UINT32_MAX &&
              sub_out_id.id != UINT32_MAX,
          "L0 attention-side GR intermediate nodes present");

    Buf tok_dev, hidden_dev, logits_dev, sampled_dev, final_dev;
    std::vector<Buf> tap_dev(kMaxTokens > 0 ? tap_ids.size() : 0);
    if (!tok_dev.alloc(kMaxTokens * sizeof(int32_t)) ||
        !hidden_dev.alloc(kHidden * sizeof(float)) ||
        !logits_dev.alloc(kVocab * sizeof(float)) ||
        !sampled_dev.alloc(sizeof(int32_t)) ||
        !final_dev.alloc(static_cast<size_t>(kMaxTokens) * kHcHidden * sizeof(float))) {
        std::printf("FAIL: output buffer allocation\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    for (std::size_t t = 0; t < tap_ids.size(); ++t) {
        if (!tap_dev[t].alloc(static_cast<size_t>(kMaxTokens) * kHcHidden *
                              sizeof(float))) {
            std::printf("FAIL: tap buffer allocation\n");
            arena.shutdown();
            fs::remove_all(g_dir);
            return 1;
        }
        patch_external(plan, tap_ids[t], tap_dev[t].dev);
    }
    patch_external(plan, tokens_id, tok_dev.dev);
    patch_external(plan, hidden_id, hidden_dev.dev);
    patch_external(plan, logits_id, logits_dev.dev);
    patch_external(plan, sampled_id, sampled_dev.dev);
    patch_external(plan, final_hidden_id, final_dev.dev);

    Buf req_dev, slots_dev, pos_dev, out_rows_dev, samp_dev;
    if (!req_dev.alloc(sizeof(rt::DeviceRequestDescriptor)) ||
        !slots_dev.alloc(kMaxTokens * sizeof(uint32_t)) ||
        !pos_dev.alloc(kMaxTokens * sizeof(uint32_t)) ||
        !out_rows_dev.alloc(sizeof(uint32_t)) ||
        !samp_dev.alloc(sizeof(rt::DeviceSamplingParams))) {
        std::printf("FAIL: context buffer allocation\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    rt::DeviceSamplingParams sampling{};
    sampling.temperature = 1.0f;
    sampling.top_p = 1.0f;
    sampling.top_k = 0;
    sampling.mode = 1;
    if (!samp_dev.fill(&sampling, sizeof(sampling))) {
        std::printf("FAIL: sampling params upload\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    DeviceCtx dc;

    struct RunResult {
        std::vector<float> hidden;
        std::vector<float> logits;
        std::vector<float> final_hidden;
        std::vector<std::vector<float>> taps;
        int32_t sampled = -1;
        bool ran = false;
    };
    const auto run = [&](state::PagedSequenceState& seq, const int32_t* tokens,
                         uint32_t count, uint32_t start_position, bool prefill,
                         int64_t output_row_override = -1) -> RunResult {
        RunResult rr;
        if (!tok_dev.fill(tokens, count * sizeof(int32_t))) return rr;
        std::vector<uint32_t> slots(count);
        std::vector<uint32_t> positions(count);
        for (uint32_t i = 0; i < count; ++i) {
            slots[i] = seq.slot;
            positions[i] = start_position + i;
        }
        const uint32_t output_row = output_row_override >= 0
                                        ? static_cast<uint32_t>(output_row_override)
                                        : count - 1u;
        if (!slots_dev.fill(slots.data(), count * sizeof(uint32_t)) ||
            !pos_dev.fill(positions.data(), count * sizeof(uint32_t)) ||
            !out_rows_dev.fill(&output_row, sizeof(uint32_t))) {
            return rr;
        }
        rt::DeviceRequestDescriptor req{};
        std::memset(&req, 0, sizeof(req));
        req.request_handle = seq.request_handle();
        req.row_begin = 0;
        req.row_count = count;
        req.prefix_length = start_position;
        req.sequence_length = start_position + count;
        req.execution_class =
            prefill ? rt::ExecutionClass::PREFILL : rt::ExecutionClass::DECODE;
        req.compute_logits = 1;
        req.output_count = 1;
        if (!req_dev.fill(&req, sizeof(req))) return rr;

        rt::DeviceBatchContext ctx{};
        ctx.actual_rows = count;
        ctx.num_requests = 1;
        ctx.num_outputs = 1;
        ctx.num_prefill_requests = prefill ? 1u : 0u;
        ctx.num_decode_requests = prefill ? 0u : 1u;
        ctx.token_ids = reinterpret_cast<const int32_t*>(tok_dev.dev);
        ctx.requests = reinterpret_cast<const rt::DeviceRequestDescriptor*>(req_dev.dev);
        ctx.row_sequence_slots = reinterpret_cast<uint32_t*>(slots_dev.dev);
        ctx.row_positions = reinterpret_cast<uint32_t*>(pos_dev.dev);
        ctx.output_rows = reinterpret_cast<uint32_t*>(out_rows_dev.dev);
        ctx.output_sampling_params =
            reinterpret_cast<rt::DeviceSamplingParams*>(samp_dev.dev);

        if (!run_program(plan, lowered.weights, lowered.parameters, dc, ctx,
                         state_view, tokens_id, stream)) {
            return rr;
        }
        rr.ran = true;
        if (!read_f32_device(hidden_dev.dev, kHidden, rr.hidden)) rr.ran = false;
        if (!read_f32_device(logits_dev.dev, kVocab, rr.logits)) rr.ran = false;
        if (!read_f32_device(final_dev.dev,
                             static_cast<size_t>(count) * kHcHidden,
                             rr.final_hidden)) {
            rr.ran = false;
        }
        if (hipMemcpy(&rr.sampled, sampled_dev.dev, sizeof(int32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            rr.ran = false;
        }
        rr.taps.resize(tap_ids.size());
        for (std::size_t t = 0; t < tap_ids.size(); ++t) {
            if (!read_f32_device(
                    tap_dev[t].dev, static_cast<size_t>(count) * kHcHidden,
                    rr.taps[t])) {
                rr.ran = false;
            }
        }
        return rr;
    };

    const std::vector<int32_t> tokens4{3, 11, 7, 19};
    const std::vector<int32_t> tokens_head{3, 11};
    const std::vector<int32_t> tokens_tail{7, 19};
    const std::vector<int32_t> tokens5{3, 11, 7, 19, 23};
    const std::vector<int32_t> decode_token{23};

    auto seq_a_result = state::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, 16u, stream);
    auto seq_b_result = state::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, 16u, stream);
    auto seq_c_result = state::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, 16u, stream);
    auto seq_d_result = state::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, 16u, stream);
    check(seq_a_result.ok() && seq_b_result.ok() && seq_c_result.ok() &&
              seq_d_result.ok(),
          "sequence create");
    if (!seq_a_result.ok() || !seq_b_result.ok() || !seq_c_result.ok() ||
        !seq_d_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    state::PagedSequenceState seq_a = seq_a_result.release();
    state::PagedSequenceState seq_b = seq_b_result.release();
    state::PagedSequenceState seq_c = seq_c_result.release();
    state::PagedSequenceState seq_d = seq_d_result.release();

    check(state::reserve_sequence_append(seq_a, 4u, stream).ok(),
          "reserve sequence A");
    check(state::reserve_sequence_append(seq_b, 4u, stream).ok(),
          "reserve sequence B");
    check(state::reserve_sequence_append(seq_c, 5u, stream).ok(),
          "reserve sequence C");
    check(state::reserve_sequence_append(seq_d, 5u, stream).ok(),
          "reserve sequence D");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: stream sync after reserve\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    StateSnap snap_b_before;
    StateSnap snap_a_before;
    check(capture_state(state_view, seq_b.slot, seq_b, snap_b_before),
          "capture state before run B");
    check(capture_state(state_view, seq_a.slot, seq_a, snap_a_before),
          "capture state before run A");

    const RunResult run_a = run(seq_a, tokens4.data(), 4u, 0u, true);
    check(run_a.ran, "run A (prefill 4 tokens, single chunk) executes");
    std::vector<float> ws_mixed;
    std::vector<float> ws_inject;
    std::vector<float> ws_sub_out;
    const bool ws_reads =
        read_value_rows(plan, dc.workspace, mixed_id, 4u, ws_mixed) &&
        read_value_rows(plan, dc.workspace, inject_id, 4u, ws_inject) &&
        read_value_rows(plan, dc.workspace, sub_out_id, 4u, ws_sub_out);
    check(ws_reads, "GR intermediate workspace values readable after run A");
    const rt::ValueId x4_id =
        node_output_by_debug_name(lowered.graph, "hc_expand_2");
    check(x4_id.id != UINT32_MAX, "embedding repeat output node present");
    const RunResult run_b1 = run(seq_b, tokens_head.data(), 2u, 0u, true);
    check(run_b1.ran, "run B1 (prefill first 2 tokens) executes");
    StateSnap snap_b_mid;
    check(capture_state(state_view, seq_b.slot, seq_b, snap_b_mid),
          "capture state after run B1");
    const RunResult run_b2 = run(seq_b, tokens_tail.data(), 2u, 2u, true);
    check(run_b2.ran, "run B2 (prefill next 2 tokens, continued state) executes");

    if (!run_a.ran || !run_b1.ran || !run_b2.ran) {
        (void)state::release_paged_sequence_state(seq_a, stream);
        (void)state::release_paged_sequence_state(seq_b, stream);
        (void)state::release_paged_sequence_state(seq_c, stream);
    (void)state::release_paged_sequence_state(seq_d, stream);
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    {
        const Metrics m =
            compare(run_b2.logits.data(), run_a.logits.data(), kVocab);
        check_metrics("split-equivalence logits (B2 vs A)", m, 1e-4, 1e-4, 1e-4);
    }
    {
        const Metrics m = compare(run_b2.hidden.data(), run_a.hidden.data(), kHidden);
        check_metrics("split-equivalence selected hidden (B2 vs A)", m, 1e-4, 1e-4,
                      1e-4);
    }
    for (std::size_t t = 0; t < tap_ids.size(); ++t) {
        std::vector<float> a_row(kHcHidden);
        std::vector<float> b_row(kHcHidden);
        for (std::size_t i = 0; i < kHcHidden; ++i) {
            a_row[i] = run_a.taps[t][3u * kHcHidden + i];
            b_row[i] = run_b2.taps[t][1u * kHcHidden + i];
        }
        const Metrics m = compare(b_row.data(), a_row.data(), kHcHidden);
        const std::string tag =
            "split-equivalence layer output tap" + std::to_string(t) + " (B2 vs A)";
        check_metrics(tag.c_str(), m, 1e-3, 1e-4, 1e-4);
    }

    StateSnap snap_a_after;
    StateSnap snap_b_after;
    check(capture_state(state_view, seq_a.slot, seq_a, snap_a_after),
          "capture state after run A");
    check(capture_state(state_view, seq_b.slot, seq_b, snap_b_after),
          "capture state after run B2");
    check(differs(snap_a_before.conv, snap_a_after.conv),
          "GDN conv state updated by prefill");
    check(differs(snap_a_before.recurrent, snap_a_after.recurrent),
          "GDN recurrent state updated by prefill");
    check(differs(snap_a_before.kv, snap_a_after.kv),
          "QSA KV cache appended by prefill");
    check(differs(snap_b_before.conv, snap_b_mid.conv),
          "GDN conv state updated by split prefill chunk 1");
    check(differs(snap_b_mid.conv, snap_b_after.conv),
          "GDN conv state updated again by split prefill chunk 2");
    check(differs(snap_b_before.recurrent, snap_b_mid.recurrent),
          "GDN recurrent state updated by split prefill chunk 1");
    check(differs(snap_b_mid.recurrent, snap_b_after.recurrent),
          "GDN recurrent state updated again by split prefill chunk 2");
    check(differs(snap_b_before.kv, snap_b_after.kv),
          "QSA KV cache appended across split prefill");

    const RunResult run_decode = run(seq_b, decode_token.data(), 1u, 4u, false);
    check(run_decode.ran, "run decode (1 token, continued state) executes");
    const RunResult run_c = run(seq_c, tokens5.data(), 5u, 0u, true);
    check(run_c.ran, "run C (prefill 5 tokens) executes");

    if (run_decode.ran && run_c.ran) {
        const Metrics m = compare(run_decode.logits.data(), run_c.logits.data(), kVocab);
        check_metrics("decode-continuation logits (decode after split vs prefill5)", m,
                      1e-4, 1e-4, 1e-4);
    }
    const RunResult run_d = run(seq_d, tokens5.data(), 5u, 0u, true, 3);
    check(run_d.ran, "run D (prefill 5 tokens, output row 3) executes");
    if (run_d.ran) {
        const Metrics m = compare(run_d.logits.data(), run_a.logits.data(), kVocab);
        check_metrics("causal prefix logits (prefill5 pos3 vs prefill4 pos3)", m,
                      1e-4, 1e-4, 1e-4);
    }
    if (run_c.ran) {
        std::vector<float> c_row3(kHcHidden);
        std::vector<float> a_row3(kHcHidden);
        for (std::size_t i = 0; i < kHcHidden; ++i) {
            c_row3[i] = run_c.taps[3][3u * kHcHidden + i];
            a_row3[i] = run_a.taps[3][3u * kHcHidden + i];
        }
        const Metrics m = compare(c_row3.data(), a_row3.data(), kHcHidden);
        check_metrics("causal prefix layer output tap3 (prefill5 pos3 vs prefill4)",
                      m, 1e-4, 1e-4, 1e-4);
    }

    hy::GatedResidualDims dims;
    dims.streams = kStreams;
    dims.hidden = kHidden;
    dims.lowrank = kLowrank;
    dims.eps = kEps;

    std::vector<float> embed_host;
    std::vector<float> hc_norm_host;
    std::vector<float> mix_down_host;
    std::vector<float> mix_up_host;
    std::vector<float> inject_w_host;
    std::vector<float> final_hc_host;
    std::vector<float> final_down_host;
    std::vector<float> final_up_host;
    std::vector<float> lm_host;
    const bool copies_ok =
        copy_bf16_to_f32(weights.embed_tokens.bf16.data<void>(),
                         static_cast<std::size_t>(kVocab) * kHidden, embed_host) &&
        copy_bf16_to_f32(weights.layers[0].attn_hyper.hc_norm.data<void>(), kHcHidden,
                         hc_norm_host) &&
        copy_bf16_to_f32(weights.layers[0].attn_hyper.mix_down.bf16.data<void>(),
                         static_cast<std::size_t>(kLowrank) * kHcHidden,
                         mix_down_host) &&
        copy_bf16_to_f32(weights.layers[0].attn_hyper.mix_up.bf16.data<void>(),
                         static_cast<std::size_t>(kHcHidden) * kLowrank,
                         mix_up_host) &&
        copy_bf16_to_f32(weights.layers[0].attn_hyper.block_inject.bf16.data<void>(),
                         static_cast<std::size_t>(kStreams) * kHcHidden,
                         inject_w_host) &&
        copy_bf16_to_f32(weights.final_hyper.hc_norm.data<void>(), kHcHidden,
                         final_hc_host) &&
        copy_bf16_to_f32(weights.final_hyper.mix_down.bf16.data<void>(),
                         static_cast<std::size_t>(kLowrank) * kHcHidden,
                         final_down_host) &&
        copy_bf16_to_f32(weights.final_hyper.mix_up.bf16.data<void>(),
                         static_cast<std::size_t>(kHcHidden) * kLowrank,
                         final_up_host) &&
        copy_bf16_to_f32(weights.lm_head.bf16.data<void>(),
                         static_cast<std::size_t>(kVocab) * kHidden, lm_host);
    check(copies_ok, "host copies of weights for references");

    if (copies_ok) {
        std::vector<float> x4_0(static_cast<size_t>(4) * kHcHidden);
        for (uint32_t t = 0; t < 4u; ++t) {
            for (uint32_t s = 0; s < kStreams; ++s) {
                for (uint32_t c = 0; c < kHidden; ++c) {
                    x4_0[static_cast<size_t>(t) * kHcHidden + s * kHidden + c] =
                        embed_host[static_cast<size_t>(tokens4[t]) * kHidden + c];
                }
            }
        }

        std::vector<float> graph_x4;
        if (read_value_rows(plan, dc.workspace, x4_id, 4u, graph_x4)) {
            const Metrics mx = compare(graph_x4.data(), x4_0.data(),
                                       static_cast<size_t>(4) * kHcHidden);
            check_metrics("graph x4 (embedding repeat) vs host x4", mx, 1e-6, 1e-6,
                          1e-6);
        } else {
            check(false, "graph x4 readable");
        }
        hy::GatedResidualWeights gw;
        gw.hc_norm = hc_norm_host.data();
        gw.mix_down = mix_down_host.data();
        gw.mix_up = mix_up_host.data();
        gw.block_inject = inject_w_host.data();
        std::vector<float> ref_mixed(static_cast<size_t>(4) * kHidden);
        std::vector<float> ref_inject(static_cast<size_t>(4) * kStreams);
        const ps::Status rst =
            hy::gated_residual_read(dims, gw, x4_0.data(), 4u, ref_mixed.data(),
                                    ref_inject.data(), nullptr);
        check(rst.ok(), "host reference: gated_residual_read runs");

        if (ws_reads) {
            check_metrics("GR read mixed (L0 attn) vs host",
                          compare(ws_mixed.data(), ref_mixed.data(),
                                  ws_mixed.size()),
                          1e-5, 1e-4, 1e-6);
            check_metrics("GR read inject (L0 attn) vs host",
                          compare(ws_inject.data(), ref_inject.data(),
                                  ws_inject.size()),
                          1e-5, 1e-4, 1e-6);

            std::vector<float> ref_write(static_cast<size_t>(4) * kHcHidden);
            const ps::Status wst = hy::gated_residual_write(
                dims, x4_0.data(), ws_sub_out.data(), ws_inject.data(), 4u,
                ref_write.data());
            check(wst.ok(), "host reference: gated_residual_write runs");
            const Metrics m = compare(run_a.taps[0].data(), ref_write.data(),
                                      static_cast<size_t>(4) * kHcHidden);
            check_metrics("GR write (L0 attn) layer output vs host", m, 1e-5, 1e-4,
                          1e-6);
        } else {
            check(false, "GR intermediates readable after run A");
        }

        std::vector<float> ref_collapse(static_cast<size_t>(4) * kHidden);
        hy::GatedResidualCollapseWeights cw;
        cw.hc_norm = final_hc_host.data();
        cw.mix_down = final_down_host.data();
        cw.mix_up = final_up_host.data();
        const ps::Status cst = hy::gated_residual_collapse(
            dims, cw, run_a.taps[3].data(), 4u, ref_collapse.data());
        check(cst.ok(), "host reference: gated_residual_collapse runs");
        check_metrics(
            "final collapse vs host",
            compare(run_a.final_hidden.data(), ref_collapse.data(),
                    static_cast<size_t>(4) * kHidden),
            2e-2, 5e-2, 5e-3);

        std::vector<float> ref_logits(kVocab);
        const float* h_row = run_a.final_hidden.data() + 3u * kHidden;
        for (uint32_t v = 0; v < kVocab; ++v) {
            float acc = 0.0f;
            for (uint32_t c = 0; c < kHidden; ++c) {
                acc += lm_host[static_cast<size_t>(v) * kHidden + c] * h_row[c];
            }
            ref_logits[v] = acc;
        }
        check_metrics("lm_head logits (final norm absent) vs host",
                      compare(run_a.logits.data(), ref_logits.data(), kVocab), 1e-6,
                      1e-5, 1e-6);

        int32_t argmax = 0;
        for (uint32_t v = 1; v < kVocab; ++v) {
            if (ref_logits[v] > ref_logits[argmax]) argmax = static_cast<int32_t>(v);
        }
        check(run_a.sampled == argmax, "sampled token equals greedy argmax");
    }

    (void)state::release_paged_sequence_state(seq_a, stream);
    (void)state::release_paged_sequence_state(seq_b, stream);
    (void)state::release_paged_sequence_state(seq_c, stream);
    (void)state::release_paged_sequence_state(seq_d, stream);

    std::printf("test_qwen4exp_forward_fixture: passed=%d failed=%d\n", g_pass, g_fail);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    if (!shutdown.ok()) {
        std::printf("FAIL: arena shutdown\n");
        ++g_fail;
    }
    fs::remove_all(g_dir);
    return g_fail == 0 ? 0 : 1;
}
