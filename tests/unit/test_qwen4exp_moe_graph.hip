#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/models/common/qwen_hybrid/moe_math.h>
#include <shelf/models/qwen4exp/model/lower_to_primitives.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/model/qwen4exp_weights.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/program/program.h>
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
namespace qh = ::ps::qwen_hybrid;
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
    g_dir = "/tmp/ps_q4e_moe_graph_" + std::to_string(getpid());
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
                                      const std::string& name,
                                      std::size_t out_index = 0) {
    for (const auto& gn : graph.nodes) {
        if (gn.debug_name == name && gn.outputs.size() > out_index)
            return gn.outputs[out_index];
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

bool read_i32_rows(const rt::Program& plan, const void* workspace, rt::ValueId id,
                   uint32_t rows, std::vector<std::int32_t>& out) {
    const rt::ValueBinding* b = plan.find_value(id);
    if (b == nullptr || b->dtype != rt::ValueDType::I32) return false;
    if (b->storage != rt::ValueStorage::WORKSPACE || workspace == nullptr)
        return false;
    const uint32_t stride = b->row_stride != 0 ? b->row_stride : b->feature_count;
    const size_t elems = static_cast<size_t>(rows) * stride;
    std::vector<std::int32_t> strided(elems);
    const size_t base = reinterpret_cast<size_t>(workspace) + b->offset;
    if (hipMemcpy(strided.data(), reinterpret_cast<const void*>(base), elems * 4,
                  hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    out.resize(static_cast<size_t>(rows) * b->feature_count);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < b->feature_count; ++c)
            out[static_cast<size_t>(r) * b->feature_count + c] =
                strided[static_cast<size_t>(r) * stride + c];
    return true;
}

struct MoeTapIds {
    rt::ValueId mixed;
    rt::ValueId top_idx;
    rt::ValueId top_w;
    rt::ValueId routed;
    rt::ValueId shared;
    rt::ValueId composed;
    bool present = false;
};

std::vector<MoeTapIds> collect_moe_taps(const rt::PrimitiveGraph& graph) {
    std::vector<MoeTapIds> taps(kLayers);
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        const std::string prefix = "L" + std::to_string(l) + ".mlp.";
        MoeTapIds& t = taps[l];
        t.mixed = node_output_by_debug_name(graph, prefix + "hyper_mixed");
        t.top_idx = node_output_by_debug_name(graph, prefix + "moe_router", 0);
        t.top_w = node_output_by_debug_name(graph, prefix + "moe_router", 1);
        t.routed = node_output_by_debug_name(graph, prefix + "moe_experts");
        t.shared = node_output_by_debug_name(graph, prefix + "shared_gate_mul");
        t.composed = node_output_by_debug_name(graph, prefix + "moe_add");
        t.present = t.mixed.id != UINT32_MAX && t.top_idx.id != UINT32_MAX &&
                    t.top_w.id != UINT32_MAX && t.routed.id != UINT32_MAX &&
                    t.shared.id != UINT32_MAX && t.composed.id != UINT32_MAX;
    }
    return taps;
}

struct MoeGraphValues {
    std::vector<float> mixed;
    std::vector<std::int32_t> top_idx;
    std::vector<float> top_w;
    std::vector<float> routed;
    std::vector<float> shared;
    std::vector<float> composed;
};

bool read_moe_values(const rt::Program& plan, const void* workspace,
                     const std::vector<MoeTapIds>& taps, uint32_t rows,
                     std::vector<MoeGraphValues>& out) {
    out.resize(taps.size());
    for (std::size_t l = 0; l < taps.size(); ++l) {
        if (!taps[l].present) return false;
        MoeGraphValues& v = out[l];
        if (!read_value_rows(plan, workspace, taps[l].mixed, rows, v.mixed) ||
            !read_i32_rows(plan, workspace, taps[l].top_idx, rows, v.top_idx) ||
            !read_value_rows(plan, workspace, taps[l].top_w, rows, v.top_w) ||
            !read_value_rows(plan, workspace, taps[l].routed, rows, v.routed) ||
            !read_value_rows(plan, workspace, taps[l].shared, rows, v.shared) ||
            !read_value_rows(plan, workspace, taps[l].composed, rows, v.composed)) {
            return false;
        }
    }
    return true;
}

struct LayerMoeHost {
    std::vector<float> router;
    std::vector<float> gate_up;
    std::vector<float> down;
    std::vector<float> shared_gate;
    std::vector<float> shared_up;
    std::vector<float> shared_down;
    std::vector<float> gate_scale;
};

bool copy_layer_moe_host(const q4e::Qwen4ExpLayerWeights& lw, LayerMoeHost& out) {
    out.router.assign(static_cast<size_t>(kExperts) * kHidden, 0.0f);
    out.gate_up.assign(static_cast<size_t>(kExperts) * 2u * kMoeInter * kHidden, 0.0f);
    out.down.assign(static_cast<size_t>(kExperts) * kHidden * kMoeInter, 0.0f);
    out.shared_gate.assign(static_cast<size_t>(kSharedInter) * kHidden, 0.0f);
    out.shared_up.assign(static_cast<size_t>(kSharedInter) * kHidden, 0.0f);
    out.shared_down.assign(static_cast<size_t>(kHidden) * kSharedInter, 0.0f);
    out.gate_scale.assign(kHidden, 0.0f);
    std::vector<float> tmp;
    if (!copy_bf16_to_f32(lw.moe.router.bf16.data<void>(), out.router.size(),
                          out.router))
        return false;
    for (std::uint32_t e = 0; e < kExperts; ++e) {
        if (!copy_bf16_to_f32(lw.moe.expert_gate_up[e].bf16.data<void>(),
                              static_cast<size_t>(2u) * kMoeInter * kHidden, tmp))
            return false;
        std::memcpy(out.gate_up.data() + static_cast<size_t>(e) * 2u * kMoeInter * kHidden,
                    tmp.data(), tmp.size() * sizeof(float));
        if (!copy_bf16_to_f32(lw.moe.expert_down[e].bf16.data<void>(),
                              static_cast<size_t>(kHidden) * kMoeInter, tmp))
            return false;
        std::memcpy(out.down.data() + static_cast<size_t>(e) * kHidden * kMoeInter,
                    tmp.data(), tmp.size() * sizeof(float));
    }
    return copy_bf16_to_f32(lw.moe.shared.gate.bf16.data<void>(), out.shared_gate.size(),
                            out.shared_gate) &&
           copy_bf16_to_f32(lw.moe.shared.up.bf16.data<void>(), out.shared_up.size(),
                            out.shared_up) &&
           copy_bf16_to_f32(lw.moe.shared.down.bf16.data<void>(), out.shared_down.size(),
                            out.shared_down) &&
           copy_bf16_to_f32(lw.moe.shared.gate_scale.data<void>(), out.gate_scale.size(),
                            out.gate_scale);
}

void oracle_moe_sublayer(const std::vector<float>& mixed, const LayerMoeHost& w,
                         std::uint32_t tokens, std::vector<std::uint32_t>& top_idx,
                         std::vector<float>& top_w, std::vector<float>& routed,
                         std::vector<float>& shared, std::vector<float>& composed,
                         std::vector<float>& shared_gates) {
    const std::size_t te = static_cast<std::size_t>(tokens) * kExperts;
    const std::size_t tk = static_cast<std::size_t>(tokens) * kTopK;
    const std::size_t th = static_cast<std::size_t>(tokens) * kHidden;
    std::vector<float> probs(te, 0.0f);
    qh::moe_router_logits_f32(mixed.data(), w.router.data(), tokens, kHidden, kExperts,
                              probs.data());
    qh::moe_softmax_f32(probs.data(), tokens, kExperts);
    top_idx.assign(tk, 0u);
    top_w.assign(tk, 0.0f);
    qh::moe_topk_stable_f32(probs.data(), tokens, kExperts, kTopK, top_idx.data(),
                            top_w.data());
    qh::moe_normalize_topk_f32(top_w.data(), tokens, kTopK);
    routed.assign(th, 0.0f);
    std::vector<float> xrow(kHidden);
    std::vector<float> out_e(kHidden);
    for (std::uint32_t t = 0; t < tokens; ++t) {
        std::memcpy(xrow.data(), mixed.data() + static_cast<std::size_t>(t) * kHidden,
                    kHidden * sizeof(float));
        float* acc = routed.data() + static_cast<std::size_t>(t) * kHidden;
        for (std::uint32_t s = 0; s < kTopK; ++s) {
            const std::uint32_t e = top_idx[static_cast<std::size_t>(t) * kTopK + s];
            const float wgt = top_w[static_cast<std::size_t>(t) * kTopK + s];
            const float* gu =
                w.gate_up.data() + static_cast<std::size_t>(e) * 2u * kMoeInter * kHidden;
            const float* dn =
                w.down.data() + static_cast<std::size_t>(e) * kHidden * kMoeInter;
            qh::moe_expert_forward_f32(xrow.data(), gu, dn, kMoeInter, kHidden,
                                       out_e.data());
            for (std::uint32_t h = 0; h < kHidden; ++h) acc[h] += wgt * out_e[h];
        }
    }
    shared.assign(th, 0.0f);
    shared_gates.assign(tokens, 0.0f);
    qh::moe_shared_expert_f32(mixed.data(), tokens, kHidden, kSharedInter,
                              w.shared_gate.data(), w.shared_up.data(),
                              w.shared_down.data(), w.gate_scale.data(), shared.data(),
                              shared_gates.data());
    composed.assign(th, 0.0f);
    qh::moe_compose_f32(routed.data(), shared.data(), th, composed.data());
}

void check_layer_moe(std::uint32_t layer, std::uint32_t tokens,
                     const MoeGraphValues& got, const LayerMoeHost& host) {
    const std::string p = "L" + std::to_string(layer) + " (T=" +
                          std::to_string(tokens) + ") ";
    std::vector<std::uint32_t> ref_idx;
    std::vector<float> ref_w;
    std::vector<float> ref_routed;
    std::vector<float> ref_shared;
    std::vector<float> ref_composed;
    std::vector<float> ref_gates;
    oracle_moe_sublayer(got.mixed, host, tokens, ref_idx, ref_w, ref_routed, ref_shared,
                        ref_composed, ref_gates);

    bool idx_ok = got.top_idx.size() == ref_idx.size();
    bool range_ok = true;
    for (std::size_t i = 0; idx_ok && i < ref_idx.size(); ++i) {
        const std::int32_t e = got.top_idx[i];
        range_ok &= e >= 0 && static_cast<std::uint32_t>(e) < kExperts;
        idx_ok = static_cast<std::uint32_t>(e) == ref_idx[i];
    }
    if (!idx_ok) {
        std::printf("  got top_idx:");
        for (auto v : got.top_idx) std::printf(" %d", v);
        std::printf("\n  ref top_idx:");
        for (auto v : ref_idx) std::printf(" %u", v);
        std::printf("\n");
    }
    check(idx_ok, p + "router top_idx bitwise vs moe_math oracle");
    check(range_ok, p + "router top_idx within expert range");

    check_metrics((p + "router top_w").c_str(),
                  compare(got.top_w.data(), ref_w.data(), got.top_w.size()), 1.0e-5,
                  1.0e-4, 1.0e-6);
    check_metrics((p + "routed experts").c_str(),
                  compare(got.routed.data(), ref_routed.data(), got.routed.size()),
                  5.0e-3, 1.0e-2, 5.0e-4);
    check_metrics((p + "shared expert").c_str(),
                  compare(got.shared.data(), ref_shared.data(), got.shared.size()),
                  1.0e-4, 1.0e-3, 1.0e-5);
    check_metrics((p + "composed routed+shared vs oracle").c_str(),
                  compare(got.composed.data(), ref_composed.data(),
                          got.composed.size()),
                  5.0e-3, 1.0e-2, 5.0e-4);

    bool wsum_ok = true;
    for (std::uint32_t t = 0; t < tokens; ++t) {
        float s = 0.0f;
        for (std::uint32_t j = 0; j < kTopK; ++j)
            s += got.top_w[static_cast<std::size_t>(t) * kTopK + j];
        wsum_ok &= std::fabs(s - 1.0f) <= 1.0e-5f;
    }
    check(wsum_ok, p + "renormalized top_w sums to 1 per token");

    float routed_max = 0.0f;
    float composed_max = 0.0f;
    for (float v : got.routed) routed_max = std::max(routed_max, std::fabs(v));
    for (float v : got.composed) composed_max = std::max(composed_max, std::fabs(v));
    check(routed_max > 0.0f, p + "routed expert output nonzero (experts active)");
    check(composed_max > 0.0f, p + "composed MoE output nonzero");

    std::vector<float> graph_sum(got.composed.size());
    for (std::size_t i = 0; i < graph_sum.size(); ++i)
        graph_sum[i] = got.routed[i] + got.shared[i];
    check_metrics((p + "graph composed == routed+shared").c_str(),
                  compare(got.composed.data(), graph_sum.data(), graph_sum.size()), 0.0,
                  0.0, 0.0);
}

struct OutputBufs {
    Buf hidden;
    Buf logits;
    Buf sampled;
    Buf final_hidden;
    std::vector<Buf> taps;

    bool alloc(std::size_t tap_count) {
        taps.resize(tap_count);
        if (!hidden.alloc(kHidden * sizeof(float)) ||
            !logits.alloc(kVocab * sizeof(float)) ||
            !sampled.alloc(sizeof(std::int32_t)) ||
            !final_hidden.alloc(static_cast<size_t>(kMaxTokens) * kHcHidden *
                                sizeof(float))) {
            return false;
        }
        for (Buf& b : taps) {
            if (!b.alloc(static_cast<size_t>(kMaxTokens) * kHcHidden * sizeof(float)))
                return false;
        }
        return true;
    }
};

struct RunInputs {
    Buf tok;
    Buf req;
    Buf slots;
    Buf pos;
    Buf out_rows;
    Buf samp;

    bool alloc() {
        return tok.alloc(kMaxTokens * sizeof(std::int32_t)) &&
               req.alloc(sizeof(rt::DeviceRequestDescriptor)) &&
               slots.alloc(kMaxTokens * sizeof(std::uint32_t)) &&
               pos.alloc(kMaxTokens * sizeof(std::uint32_t)) &&
               out_rows.alloc(sizeof(std::uint32_t)) &&
               samp.alloc(sizeof(rt::DeviceSamplingParams));
    }
};

struct RunResult {
    std::vector<float> hidden;
    std::vector<float> logits;
    std::vector<float> final_hidden;
    std::vector<std::vector<float>> taps;
    std::int32_t sampled = -1;
    bool ran = false;
};

RunResult run_forward(const rt::Program& plan,
                      const std::vector<rt::WeightSlot>& host_weights,
                      const std::vector<rt::StaticParameterSlot>& host_params,
                      DeviceCtx& dc, OutputBufs& ob, RunInputs& in,
                      state::PagedSequenceState& seq, const std::int32_t* tokens,
                      std::uint32_t count, std::uint32_t start_position, bool prefill,
                      hipStream_t stream,
                      const ::ps::kernel::ModelDispatchStateView& state_view,
                      rt::ValueId tokens_id, std::size_t tap_count,
                      std::int64_t output_row_override = -1) {
    RunResult rr;
    if (!in.tok.fill(tokens, count * sizeof(std::int32_t))) return rr;
    std::vector<std::uint32_t> slots(count);
    std::vector<std::uint32_t> positions(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        slots[i] = seq.slot;
        positions[i] = start_position + i;
    }
    const std::uint32_t output_row = output_row_override >= 0
                                         ? static_cast<std::uint32_t>(output_row_override)
                                         : count - 1u;
    if (!in.slots.fill(slots.data(), count * sizeof(std::uint32_t)) ||
        !in.pos.fill(positions.data(), count * sizeof(std::uint32_t)) ||
        !in.out_rows.fill(&output_row, sizeof(std::uint32_t))) {
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
    if (!in.req.fill(&req, sizeof(req))) return rr;

    rt::DeviceBatchContext ctx{};
    ctx.actual_rows = count;
    ctx.num_requests = 1;
    ctx.num_outputs = 1;
    ctx.num_prefill_requests = prefill ? 1u : 0u;
    ctx.num_decode_requests = prefill ? 0u : 1u;
    ctx.token_ids = reinterpret_cast<const std::int32_t*>(in.tok.dev);
    ctx.requests = reinterpret_cast<const rt::DeviceRequestDescriptor*>(in.req.dev);
    ctx.row_sequence_slots = reinterpret_cast<std::uint32_t*>(in.slots.dev);
    ctx.row_positions = reinterpret_cast<std::uint32_t*>(in.pos.dev);
    ctx.output_rows = reinterpret_cast<std::uint32_t*>(in.out_rows.dev);
    ctx.output_sampling_params =
        reinterpret_cast<rt::DeviceSamplingParams*>(in.samp.dev);

    if (!run_program(plan, host_weights, host_params, dc, ctx, state_view, tokens_id,
                     stream)) {
        return rr;
    }
    rr.ran = true;
    if (!read_f32_device(ob.hidden.dev, kHidden, rr.hidden)) rr.ran = false;
    if (!read_f32_device(ob.logits.dev, kVocab, rr.logits)) rr.ran = false;
    if (!read_f32_device(ob.final_hidden.dev,
                         static_cast<size_t>(count) * kHcHidden, rr.final_hidden)) {
        rr.ran = false;
    }
    if (hipMemcpy(&rr.sampled, ob.sampled.dev, sizeof(std::int32_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        rr.ran = false;
    }
    rr.taps.resize(tap_count);
    for (std::size_t t = 0; t < tap_count; ++t) {
        if (!read_f32_device(ob.taps[t].dev,
                             static_cast<size_t>(count) * kHcHidden, rr.taps[t])) {
            rr.ran = false;
        }
    }
    return rr;
}

std::int32_t greedy_argmax(const std::vector<float>& logits) {
    std::int32_t best = 0;
    for (std::size_t v = 1; v < logits.size(); ++v) {
        if (logits[v] > logits[static_cast<std::size_t>(best)])
            best = static_cast<std::int32_t>(v);
    }
    return best;
}

void print_logits_line(const char* tag, const std::vector<float>& logits) {
    std::printf("%s:", tag);
    for (float v : logits) std::printf(" %.5f", v);
    std::printf("\n");
}

}  // namespace

int main() {
    std::printf("test_qwen4exp_moe_graph\n");
    std::printf(
        "scope: connect_moe=true の full-graph 実行 — router / selected experts / "
        "shared expert / 合算を G4b moe_math.h（pure F32）oracle と比較し、"
        "connect_moe=false の結果と併記する\n");

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
    check(cfg.norm_topk_prob, "config norm_topk_prob is true (router renormalizes)");

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
    lower_options.connect_moe = true;
    lower_options.hidden_taps = tap_layers;
    auto lower_result = q4e::lower_qwen4exp_to_primitives(cfg, weights, lower_options);
    check(lower_result.ok(),
          std::string("lowering (connect_moe=true, hidden taps)") +
              (lower_result.ok() ? ""
                                 : " (" + lower_result.status().message() + ")"));
    if (!lower_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    q4e::Qwen4ExpLoweredPrimitives lowered = lower_result.release();

    const std::vector<MoeTapIds> moe_taps = collect_moe_taps(lowered.graph);
    bool moe_nodes_ok = true;
    for (const MoeTapIds& t : moe_taps) moe_nodes_ok &= t.present;
    check(moe_nodes_ok, "per-layer MoE sub-layer debug nodes present");

    rt::ProgramBuildOptions build_options;
    build_options.validate_workspace = true;
    build_options.no_workspace_reuse = true;
    build_options.max_token_rows = kMaxTokens;
    build_options.max_output_rows = 1;
    const rt::WeightTableView weight_view{lowered.weights.data(),
                                           lowered.weights.size()};
    auto built = rt::build_program(lowered.graph, weight_view, rt::RowBucket::R16,
                                   rt::ExecutionClass::PREFILL, build_options);
    check(built.ok(), std::string("build_program (connect_moe=true)") +
                          (built.ok() ? ""
                                      : " (" + built.status().message() + ")"));
    if (!built.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    rt::Program plan_moe = built.release();
    plan_moe.parameter_slot_count = static_cast<uint32_t>(lowered.parameters.size());

    {
        std::size_t router_dispatch = 0;
        std::size_t experts_dispatch = 0;
        for (const auto& d : plan_moe.dispatches) {
            if (d.kernel_id == rt::KernelId::MOE_ROUTER) ++router_dispatch;
            if (d.kernel_id == rt::KernelId::MOE_EXPERTS) ++experts_dispatch;
        }
        check(router_dispatch == kLayers && experts_dispatch == kLayers,
              "MOE_ROUTER / MOE_EXPERTS dispatch emitted once per layer");
    }

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

    OutputBufs ob_moe;
    if (!ob_moe.alloc(tap_ids.size())) {
        std::printf("FAIL: MoE output buffer allocation\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    OutputBufs ob_off;
    if (!ob_off.alloc(tap_ids.size())) {
        std::printf("FAIL: no-MoE output buffer allocation\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    RunInputs in;
    if (!in.alloc()) {
        std::printf("FAIL: run input buffer allocation\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    rt::DeviceSamplingParams sampling{};
    sampling.temperature = 1.0f;
    sampling.top_p = 1.0f;
    sampling.top_k = 0;
    sampling.mode = 1;
    if (!in.samp.fill(&sampling, sizeof(sampling))) {
        std::printf("FAIL: sampling params upload\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    patch_external(plan_moe, tokens_id, in.tok.dev);
    patch_external(plan_moe, hidden_id, ob_moe.hidden.dev);
    patch_external(plan_moe, logits_id, ob_moe.logits.dev);
    patch_external(plan_moe, sampled_id, ob_moe.sampled.dev);
    patch_external(plan_moe, final_hidden_id, ob_moe.final_hidden.dev);
    for (std::size_t t = 0; t < tap_ids.size(); ++t)
        patch_external(plan_moe, tap_ids[t], ob_moe.taps[t].dev);

    DeviceCtx dc_moe;

    auto seq_a_result = state::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool,
                                                           16u, stream);
    auto seq_b_result = state::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool,
                                                           16u, stream);
    auto seq_c_result = state::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool,
                                                           16u, stream);
    auto seq_e_result = state::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool,
                                                           16u, stream);
    check(seq_a_result.ok() && seq_b_result.ok() && seq_c_result.ok() &&
              seq_e_result.ok(),
          "sequence create");
    if (!seq_a_result.ok() || !seq_b_result.ok() || !seq_c_result.ok() ||
        !seq_e_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    state::PagedSequenceState seq_a = seq_a_result.release();
    state::PagedSequenceState seq_b = seq_b_result.release();
    state::PagedSequenceState seq_c = seq_c_result.release();
    state::PagedSequenceState seq_e = seq_e_result.release();

    check(state::reserve_sequence_append(seq_a, 4u, stream).ok(),
          "reserve sequence A");
    check(state::reserve_sequence_append(seq_b, 4u, stream).ok(),
          "reserve sequence B");
    check(state::reserve_sequence_append(seq_c, 5u, stream).ok(),
          "reserve sequence C");
    check(state::reserve_sequence_append(seq_e, 4u, stream).ok(),
          "reserve sequence E");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: stream sync after reserve\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    const std::vector<std::int32_t> tokens4{3, 11, 7, 19};
    const std::vector<std::int32_t> tokens_head{3, 11};
    const std::vector<std::int32_t> tokens_tail{7, 19};
    const std::vector<std::int32_t> tokens5{3, 11, 7, 19, 23};
    const std::vector<std::int32_t> decode_token{23};

    const RunResult run_a = run_forward(plan_moe, lowered.weights, lowered.parameters,
                                        dc_moe, ob_moe, in, seq_a, tokens4.data(), 4u,
                                        0u, true, stream, state_view, tokens_id,
                                        tap_ids.size());
    check(run_a.ran, "run A (connect_moe=true, prefill 4 tokens) executes");
    if (!run_a.ran) {
        (void)state::release_paged_sequence_state(seq_a, stream);
        (void)state::release_paged_sequence_state(seq_b, stream);
        (void)state::release_paged_sequence_state(seq_c, stream);
        (void)state::release_paged_sequence_state(seq_e, stream);
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    std::vector<LayerMoeHost> host_moe(kLayers);
    bool host_moe_ok = true;
    for (std::uint32_t l = 0; l < kLayers; ++l)
        host_moe_ok &= copy_layer_moe_host(weights.layers[l], host_moe[l]);
    check(host_moe_ok, "host copies of per-layer MoE weights");

    std::vector<MoeGraphValues> moe_vals;
    const bool moe_reads = host_moe_ok && read_moe_values(plan_moe, dc_moe.workspace,
                                                          moe_taps, 4u, moe_vals);
    check(moe_reads, "MoE sub-layer intermediates readable (no_workspace_reuse)");
    if (moe_reads) {
        for (std::uint32_t l = 0; l < kLayers; ++l)
            check_layer_moe(l, 4u, moe_vals[l], host_moe[l]);
    }

    const RunResult run_b1 = run_forward(plan_moe, lowered.weights, lowered.parameters,
                                         dc_moe, ob_moe, in, seq_b, tokens_head.data(),
                                         2u, 0u, true, stream, state_view, tokens_id,
                                         tap_ids.size());
    check(run_b1.ran, "run B1 (prefill first 2 tokens) executes");
    const RunResult run_b2 = run_forward(plan_moe, lowered.weights, lowered.parameters,
                                         dc_moe, ob_moe, in, seq_b, tokens_tail.data(),
                                         2u, 2u, true, stream, state_view, tokens_id,
                                         tap_ids.size());
    check(run_b2.ran, "run B2 (prefill next 2 tokens, continued state) executes");
    if (run_b1.ran && run_b2.ran) {
        const Metrics m =
            compare(run_b2.logits.data(), run_a.logits.data(), kVocab);
        check_metrics("split-equivalence logits (B2 vs A, connect_moe=true)", m, 1e-4,
                      1e-4, 1e-4);
        const Metrics mh = compare(run_b2.hidden.data(), run_a.hidden.data(), kHidden);
        check_metrics("split-equivalence selected hidden (B2 vs A)", mh, 1e-4, 1e-4,
                      1e-4);
        for (std::size_t t = 0; t < tap_ids.size(); ++t) {
            std::vector<float> a_row(kHcHidden);
            std::vector<float> b_row(kHcHidden);
            for (std::size_t i = 0; i < kHcHidden; ++i) {
                a_row[i] = run_a.taps[t][3u * kHcHidden + i];
                b_row[i] = run_b2.taps[t][1u * kHcHidden + i];
            }
            const Metrics mt = compare(b_row.data(), a_row.data(), kHcHidden);
            check_metrics(("split-equivalence layer output tap" + std::to_string(t) +
                           " (B2 vs A)")
                              .c_str(),
                          mt, 1e-3, 1e-4, 1e-4);
        }
    }

    const RunResult run_decode = run_forward(plan_moe, lowered.weights,
                                             lowered.parameters, dc_moe, ob_moe, in,
                                             seq_b, decode_token.data(), 1u, 4u, false,
                                             stream, state_view, tokens_id,
                                             tap_ids.size());
    check(run_decode.ran, "run decode (1 token, continued state) executes");
    std::vector<MoeGraphValues> decode_vals;
    bool decode_reads = false;
    if (run_decode.ran) {
        decode_reads = read_moe_values(plan_moe, dc_moe.workspace, moe_taps, 1u,
                                       decode_vals);
    }
    check(decode_reads, "MoE sub-layer intermediates readable for decode run");
    if (decode_reads) {
        for (std::uint32_t l = 0; l < kLayers; ++l)
            check_layer_moe(l, 1u, decode_vals[l], host_moe[l]);
    }

    const RunResult run_c = run_forward(plan_moe, lowered.weights, lowered.parameters,
                                        dc_moe, ob_moe, in, seq_c, tokens5.data(), 5u,
                                        0u, true, stream, state_view, tokens_id,
                                        tap_ids.size());
    check(run_c.ran, "run C (prefill 5 tokens) executes");
    if (run_decode.ran && run_c.ran) {
        const Metrics m =
            compare(run_decode.logits.data(), run_c.logits.data(), kVocab);
        check_metrics("decode-continuation logits (decode after split vs prefill5)", m,
                      1e-4, 1e-4, 1e-4);
    }

    check(run_a.sampled == greedy_argmax(run_a.logits),
          "connect_moe=true sampled token equals greedy argmax");

    q4e::Qwen4ExpLowerOptions no_moe_options;
    no_moe_options.connect_moe = false;
    no_moe_options.hidden_taps = tap_layers;
    auto lower_off = q4e::lower_qwen4exp_to_primitives(cfg, weights, no_moe_options);
    check(lower_off.ok(),
          std::string("lowering (connect_moe=false) succeeds") +
              (lower_off.ok() ? ""
                              : " (" + lower_off.status().message() + ")"));
    if (lower_off.ok()) {
        q4e::Qwen4ExpLoweredPrimitives lowered_off = lower_off.release();
        const rt::WeightTableView wv_off{lowered_off.weights.data(),
                                          lowered_off.weights.size()};
        auto built_off = rt::build_program(lowered_off.graph, wv_off,
                                           rt::RowBucket::R16,
                                           rt::ExecutionClass::PREFILL, build_options);
        check(built_off.ok(), std::string("build_program (connect_moe=false)") +
                                  (built_off.ok()
                                       ? ""
                                       : " (" + built_off.status().message() + ")"));
        if (built_off.ok()) {
            rt::Program plan_off = built_off.release();
            plan_off.parameter_slot_count =
                static_cast<uint32_t>(lowered_off.parameters.size());
            bool moe_dispatch_absent = true;
            for (const auto& d : plan_off.dispatches) {
                if (d.kernel_id == rt::KernelId::MOE_ROUTER ||
                    d.kernel_id == rt::KernelId::MOE_EXPERTS) {
                    moe_dispatch_absent = false;
                }
            }
            check(moe_dispatch_absent, "connect_moe=false plan has no MoE dispatch");
            const rt::ValueId tokens_off = lowered_off.graph.external_inputs[0];
            patch_external(plan_off, tokens_off, in.tok.dev);
            patch_external(plan_off, lowered_off.graph.external_outputs[0],
                           ob_off.hidden.dev);
            patch_external(plan_off, lowered_off.graph.external_outputs[1],
                           ob_off.logits.dev);
            patch_external(plan_off, lowered_off.graph.external_outputs[2],
                           ob_off.sampled.dev);
            patch_external(plan_off, lowered_off.graph.external_outputs[3],
                           ob_off.final_hidden.dev);
            for (std::size_t t = 0; t < tap_ids.size(); ++t)
                patch_external(plan_off, lowered_off.graph.external_outputs[4 + t],
                               ob_off.taps[t].dev);
            DeviceCtx dc_off;
            const RunResult run_off =
                run_forward(plan_off, lowered_off.weights, lowered_off.parameters,
                            dc_off, ob_off, in, seq_e, tokens4.data(), 4u, 0u, true,
                            stream, state_view, tokens_off, tap_ids.size());
            check(run_off.ran, "run E (connect_moe=false, prefill 4 tokens) executes");
            if (run_off.ran) {
                check(run_off.sampled == greedy_argmax(run_off.logits),
                      "connect_moe=false sampled token equals greedy argmax");
                print_logits_line("logits connect_moe=true ", run_a.logits);
                print_logits_line("logits connect_moe=false", run_off.logits);
                std::printf("sampled connect_moe=true: %d, connect_moe=false: %d\n",
                            run_a.sampled, run_off.sampled);
                const Metrics m_logits =
                    compare(run_off.logits.data(), run_a.logits.data(), kVocab);
                print_metrics("moe-on vs off logits delta", m_logits);
                check(!m_logits.non_finite && m_logits.max_abs > 1.0e-3,
                      "MoE connection changes lm_head logits (full-graph effect)");
                const Metrics m_final =
                    compare(run_off.final_hidden.data(), run_a.final_hidden.data(),
                            static_cast<size_t>(4) * kHcHidden);
                print_metrics("moe-on vs off final collapse delta", m_final);
                check(!m_final.non_finite && m_final.max_abs > 1.0e-3,
                      "MoE connection changes final collapse output");
                for (std::size_t t = 0; t < tap_ids.size(); ++t) {
                    const Metrics m_tap =
                        compare(run_off.taps[t].data(), run_a.taps[t].data(),
                                static_cast<size_t>(4) * kHcHidden);
                    print_metrics(("moe-on vs off layer output tap" +
                                   std::to_string(t) + " delta")
                                      .c_str(),
                                  m_tap);
                    check(!m_tap.non_finite && m_tap.max_abs > 1.0e-3,
                          "MoE connection changes layer output tap" +
                              std::to_string(t));
                }
            }
        }
    }

    (void)state::release_paged_sequence_state(seq_a, stream);
    (void)state::release_paged_sequence_state(seq_b, stream);
    (void)state::release_paged_sequence_state(seq_c, stream);
    (void)state::release_paged_sequence_state(seq_e, stream);

    std::printf("test_qwen4exp_moe_graph: passed=%d failed=%d\n", g_pass, g_fail);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    if (!shutdown.ok()) {
        std::printf("FAIL: arena shutdown\n");
        ++g_fail;
    }
    fs::remove_all(g_dir);
    return g_fail == 0 ? 0 : 1;
}
