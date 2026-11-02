#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/inference/state/gdn_state_pool.h>
#include <phaseshift/inference/state/paged_kv_pool.h>
#include <phaseshift/inference/state/paged_sequence_state.h>
#include <phaseshift/inference/state/sequence_slot_pool.h>
#include <phaseshift/io/safetensors_writer.h>
#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/models/qwen4exp/model/lower_to_primitives.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_config.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_weights.h>
#include <phaseshift/models/qwen4exp/ple/ple_forward.h>
#include <phaseshift/models/qwen4exp/runtime/qwen4exp_state.h>
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
#include <cstdlib>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
namespace rt = ::ps::runtime;
namespace gpu = ::ps::gpu;
namespace q4e = ::ps::qwen4exp;
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
constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kLowrank = 8;
constexpr std::uint32_t kHcHidden = kStreams * kHidden;
constexpr std::uint32_t kLayers = 2;
constexpr std::uint32_t kPleLayer = 1;
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
constexpr std::uint32_t kExperts = 4;
constexpr std::uint32_t kTopK = 2;
constexpr std::uint32_t kMoeInter = 16;
constexpr std::uint32_t kSharedInter = 16;
constexpr std::uint32_t kGdnLayers = 1;
constexpr std::uint32_t kQsaLayers = 1;
constexpr std::uint32_t kMaxTokens = 8;
constexpr std::uint32_t kPageTokens = 16;
constexpr std::uint32_t kNumPages = 8;
constexpr std::uint32_t kMaxSequences = 6;
constexpr std::uint32_t kEosToken = 7;
constexpr std::uint32_t kHeadRows = 97;
constexpr std::uint32_t kTableRows = 16 * kHeadRows;
constexpr std::uint32_t kRowCols = kHidden / 16;
constexpr float kEps = 1.0e-6f;

std::string g_dir;
std::mt19937_64 g_rng(0xBEEF17u);

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
    add("model.language_model.hyper_connection_mixer.hc_norm.weight", {kStreams * kHidden},
        0.0f);
    values.back() = one_plus(kStreams * kHidden);
    add("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        {kLowrank, kStreams * kHidden},
        inv_sqrt(static_cast<float>(kStreams * kHidden)));
    add("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        {kStreams * kHidden, kLowrank}, inv_sqrt(static_cast<float>(kLowrank)));

    const std::string lp = "model.language_model.layers.";
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        const std::string p = lp + std::to_string(l) + ".";
        for (const std::string& side : {"attn_hyper_connection.",
                                        "mlp_hyper_connection."}) {
            const std::string h = p + side;
            add(h + "hc_norm.weight", {kStreams * kHidden}, 0.0f);
            values.back() = one_plus(kStreams * kHidden);
            add(h + "input_mix_weight_down.weight", {kLowrank, kStreams * kHidden},
                inv_sqrt(static_cast<float>(kStreams * kHidden)));
            add(h + "input_mix_weight_up.weight", {kStreams * kHidden, kLowrank},
                inv_sqrt(static_cast<float>(kLowrank)));
            add(h + "block_inject_weight.weight", {kStreams, kStreams * kHidden},
                inv_sqrt(static_cast<float>(kStreams * kHidden)));
        }
        if (l == 0) {
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

    const std::string pp = lp + std::to_string(kPleLayer) + ".ple.";
    add(pp + "key_proj.weight", {kHcHidden, kHidden}, inv_sqrt(static_cast<float>(kHidden)));
    add(pp + "value_proj.weight", {kHidden, kHidden},
        inv_sqrt(static_cast<float>(kHidden)));
    add(pp + "conv1d.weight", {kHcHidden, 1, kConvKernel}, 0.5f);
    add_with(pp + "norm_key.weight", {kHcHidden}, near_one(kHcHidden));
    add_with(pp + "norm_query.weight", {kHcHidden}, near_one(kHcHidden));
    add_with(pp + "norm_conv.weight", {kHcHidden}, near_one(kHcHidden));
    {
        std::vector<std::int64_t> mult{11, 17, 23};
        std::vector<std::int64_t> sizes(16, static_cast<std::int64_t>(kHeadRows));
        std::vector<std::int64_t> offsets(16);
        for (std::size_t h = 0; h < offsets.size(); ++h) {
            offsets[h] = static_cast<std::int64_t>(h) * kHeadRows;
        }
        i64.emplace_back(pp + "ple_embedding.layer_multipliers",
                         std::vector<std::size_t>{3});
        values.push_back(std::vector<float>{});
        i64.emplace_back(pp + "ple_embedding.ngram_heads_vocab_sizes",
                         std::vector<std::size_t>{16});
        values.push_back(std::vector<float>{});
        i64.emplace_back(pp + "ple_embedding.ngram_heads_offsets",
                         std::vector<std::size_t>{16});
        values.push_back(std::vector<float>{});

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
        const std::vector<std::int64_t>* i64_values[3] = {&mult, &sizes, &offsets};
        for (size_t i = 0; i < i64.size(); ++i) {
            const std::vector<std::int64_t>& v = *i64_values[i];
            if (!w.value()
                     .write_tensor(i64[i].first, v.data(), v.size() * 8)
                     .ok()) {
                std::printf("FAIL: write i64 %s\n", i64[i].first.c_str());
                ++g_fail;
            }
        }
        if (!w.value().finish().ok()) {
            std::printf("FAIL: finish\n");
            ++g_fail;
        }
    }
}

void build_fixture() {
    g_dir = "/tmp/ps_q4e_ple_" + std::to_string(getpid());
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
          << R"("rms_norm_eps":1e-6,"output_gate_type":"sigmoid",)"
          << R"("rope_parameters":{"rope_theta":10000000,)"
          << R"("partial_rotary_factor":0.25},)"
          << R"("eos_token_id":)" << kEosToken << R"(,)"
          << R"("layer_types":["linear_attention","full_attention"]}})";
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
            if (l == 0) {
                for (const std::string& field :
                     {"in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
                      "in_proj_a.weight", "out_proj.weight", "conv1d.weight",
                      "norm.weight", "dt_bias", "A_log"}) {
                    emit(p + "linear_attn." + field);
                }
            } else {
                for (const std::string& field :
                     {"q_proj.weight", "k_proj.weight", "v_proj.weight",
                      "o_proj.weight", "q_norm.weight", "k_norm.weight"}) {
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
        const std::string pp = lp + std::to_string(kPleLayer) + ".ple.";
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
                 hipStream_t stream) {
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

std::vector<float> grouped_norm(const std::vector<float>& x, const float* weight,
                                std::uint32_t groups, std::uint32_t group) {
    std::vector<float> y(x.size());
    for (std::uint32_t g = 0; g < groups; ++g) {
        const std::size_t base = static_cast<std::size_t>(g) * group;
        float ss = 0.0f;
        for (std::uint32_t i = 0; i < group; ++i) ss += x[base + i] * x[base + i];
        const float r = 1.0f / std::sqrt(ss / static_cast<float>(group) + kEps);
        for (std::uint32_t i = 0; i < group; ++i) {
            y[base + i] = x[base + i] * r * (1.0f + weight[i]);
        }
    }
    return y;
}

std::vector<float> gemv(const std::vector<float>& w, const std::vector<float>& x,
                        std::uint32_t rows, std::uint32_t cols) {
    std::vector<float> y(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (std::uint32_t c = 0; c < cols; ++c) {
            acc += w[static_cast<std::size_t>(r) * cols + c] * x[c];
        }
        y[r] = acc;
    }
    return y;
}

struct OracleState {
    std::vector<std::uint32_t> context;
    std::vector<std::vector<float>> conv_history;

    OracleState(std::uint32_t eos, std::uint32_t conv_dim)
        : context(2, eos), conv_history(9, std::vector<float>(conv_dim, 0.0f)) {}
};

std::vector<std::uint64_t> oracle_ngram_ids(OracleState& st,
                                            const std::vector<std::uint32_t>& tokens) {
    std::vector<std::uint32_t> hist;
    hist.insert(hist.end(), st.context.begin(), st.context.end());
    hist.insert(hist.end(), tokens.begin(), tokens.end());
    std::vector<std::uint32_t> prefix(hist.size() + 1, 0u);
    for (std::size_t i = 0; i < hist.size(); ++i) {
        prefix[i + 1] = prefix[i] + (hist[i] == kEosToken ? 1u : 0u);
    }
    const std::uint64_t mult[3] = {11, 17, 23};
    std::vector<std::uint64_t> ids(tokens.size() * 16);
    for (std::size_t p = 0; p < tokens.size(); ++p) {
        const std::size_t pos = 2 + p;
        std::uint64_t shifted[3];
        for (std::uint32_t k = 0; k < 3; ++k) {
            if (pos < k) {
                shifted[k] = kEosToken;
                continue;
            }
            const std::size_t src = pos - k;
            shifted[k] = (prefix[pos] > prefix[src]) ? kEosToken : hist[src];
        }
        for (std::uint32_t degree = 2; degree <= 3; ++degree) {
            std::uint64_t h = shifted[0] * mult[0];
            for (std::uint32_t k = 1; k < degree; ++k) h ^= shifted[k] * mult[k];
            for (std::uint32_t j = 0; j < 8; ++j) {
                const std::uint32_t head = (degree - 2u) * 8u + j;
                ids[p * 16 + head] = (h % kHeadRows) + head * kHeadRows;
            }
        }
    }
    st.context[0] = hist[hist.size() - 2];
    st.context[1] = hist[hist.size() - 1];
    return ids;
}

std::vector<float> oracle_ple(OracleState& st, const std::vector<std::uint32_t>& tokens,
                              const std::vector<float>& hidden_states,
                              const std::vector<float>& table,
                              const q4e::ple::PleBranch& branch, std::uint32_t count) {
    const std::vector<std::uint64_t> ids = oracle_ngram_ids(st, tokens);
    const std::uint32_t hc_hidden = kStreams * kHidden;
    std::vector<float> embed(static_cast<std::size_t>(count) * kHidden);
    for (std::uint32_t t = 0; t < count; ++t) {
        for (std::uint32_t h = 0; h < 16; ++h) {
            const std::uint64_t row = ids[static_cast<std::size_t>(t) * 16 + h];
            for (std::uint32_t c = 0; c < kRowCols; ++c) {
                embed[static_cast<std::size_t>(t) * kHidden + h * kRowCols + c] =
                    table[static_cast<std::size_t>(row) * kRowCols + c];
            }
        }
    }

    std::vector<float> key(static_cast<std::size_t>(count) * hc_hidden);
    std::vector<float> value(static_cast<std::size_t>(count) * kHidden);
    std::vector<float> query(static_cast<std::size_t>(count) * hc_hidden);
    const std::vector<float> key_w(branch.key_proj_host(),
                                   branch.key_proj_host() + hc_hidden * kHidden);
    const std::vector<float> value_w(branch.value_proj_host(),
                                     branch.value_proj_host() + kHidden * kHidden);
    for (std::uint32_t t = 0; t < count; ++t) {
        const std::vector<float> row(embed.begin() + static_cast<std::size_t>(t) * kHidden,
                                     embed.begin() +
                                         static_cast<std::size_t>(t + 1) * kHidden);
        const std::vector<float> k_row = gemv(key_w, row, hc_hidden, kHidden);
        const std::vector<float> v_row = gemv(value_w, row, kHidden, kHidden);
        std::copy(k_row.begin(), k_row.end(),
                  key.begin() + static_cast<std::size_t>(t) * hc_hidden);
        std::copy(v_row.begin(), v_row.end(),
                  value.begin() + static_cast<std::size_t>(t) * kHidden);
        std::copy(hidden_states.begin() + static_cast<std::size_t>(t) * hc_hidden,
                  hidden_states.begin() + static_cast<std::size_t>(t + 1) * hc_hidden,
                  query.begin() + static_cast<std::size_t>(t) * hc_hidden);
    }
    const std::vector<float> key_n =
        grouped_norm(key, branch.norm_key_host(), count * kStreams, kHidden);
    const std::vector<float> query_n =
        grouped_norm(query, branch.norm_query_host(), count * kStreams, kHidden);

    std::vector<float> gated(static_cast<std::size_t>(count) * hc_hidden);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(kHidden));
    for (std::uint32_t t = 0; t < count; ++t) {
        for (std::uint32_t s = 0; s < kStreams; ++s) {
            float acc = 0.0f;
            for (std::uint32_t f = 0; f < kHidden; ++f) {
                const std::size_t i = static_cast<std::size_t>(t) * hc_hidden +
                                      static_cast<std::size_t>(s) * kHidden + f;
                acc += key_n[i] * query_n[i];
            }
            const float raw = acc * inv_sqrt;
            const float shaped = std::sqrt(std::max(std::fabs(raw), 1.0e-6f)) *
                                 (raw < 0.0f ? -1.0f : 1.0f);
            const float sig = 1.0f / (1.0f + std::exp(-shaped));
            for (std::uint32_t f = 0; f < kHidden; ++f) {
                gated[static_cast<std::size_t>(t) * hc_hidden +
                      static_cast<std::size_t>(s) * kHidden + f] =
                    sig * value[static_cast<std::size_t>(t) * kHidden + f];
            }
        }
    }
    const std::vector<float> normed =
        grouped_norm(gated, branch.norm_conv_host(), count * kStreams, kHidden);
    const float* conv_w = branch.conv_weight_host();

    std::vector<float> out(gated);
    for (std::uint32_t t = 0; t < count; ++t) {
        for (std::uint32_t c = 0; c < hc_hidden; ++c) {
            const float* w = conv_w + static_cast<std::size_t>(c) * kConvKernel;
            const float x = normed[static_cast<std::size_t>(t) * hc_hidden + c];
            const float acc = w[0] * st.conv_history[0][c] +
                              w[1] * st.conv_history[3][c] +
                              w[2] * st.conv_history[6][c] + w[3] * x;
            out[static_cast<std::size_t>(t) * hc_hidden + c] +=
                acc / (1.0f + std::exp(-acc));
        }
        for (std::uint32_t c = 0; c < hc_hidden; ++c) {
            for (std::uint32_t k = 0; k + 1u < 9u; ++k) {
                st.conv_history[k][c] = st.conv_history[k + 1u][c];
            }
            st.conv_history[8][c] = normed[static_cast<std::size_t>(t) * hc_hidden + c];
        }
    }
    return out;
}

std::vector<std::uint32_t> to_tokens(const std::vector<int32_t>& tokens) {
    std::vector<std::uint32_t> out(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        out[i] = static_cast<std::uint32_t>(tokens[i]);
    }
    return out;
}

std::vector<std::uint32_t> context_of(const q4e::runtime::PleNgramContext& ngram,
                                      state::SequenceSlotId slot) {
    const std::uint32_t* c = ngram.context(slot);
    return std::vector<std::uint32_t>(c, c + 2);
}

}  // namespace

int main() {
    std::printf("test_qwen4exp_ple_graph\n");

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
                               (cfg_result.ok()
                                    ? ""
                                    : " (" + cfg_result.status().message() + ")"));
    if (!cfg_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    const q4e::Qwen4ExpTextConfig cfg = cfg_result.release();
    check(cfg.ple_layer_index == kPleLayer, "PLE layer index is 1");

    q4e::Qwen4ExpLoadOptions load_options;
    auto weights_result =
        q4e::load_qwen4exp_weights_bf16(g_dir, arena, stream, load_options);
    check(weights_result.ok(),
          std::string("weights load") +
              (weights_result.ok() ? ""
                                   : " (" + weights_result.status().message() + ")"));
    if (!weights_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    q4e::Qwen4ExpModelWeights weights = weights_result.release();
    check(weights.layers[kPleLayer].ple.present, "PLE weights loaded");

    std::vector<float> table_host(static_cast<std::size_t>(kTableRows) * kRowCols);
    std::vector<std::uint16_t> table_bits(table_host.size());
    for (size_t i = 0; i < table_host.size(); ++i) {
        table_host[i] = uniform11();
        table_bits[i] = bf16_bits(table_host[i]);
        table_host[i] = bits_bf16(table_bits[i]);
    }
    Buf table_dev;
    if (!table_dev.alloc(table_bits.size() * 2) ||
        !table_dev.fill(table_bits.data(), table_bits.size() * 2)) {
        std::printf("FAIL: table allocation\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    q4e::ple::TableLayout layout;
    layout.shard_count = 1;
    layout.rows_per_shard = kTableRows;
    layout.row_cols = kRowCols;
    layout.row_bytes = kRowCols * 2;
    layout.tensor_prefix = "fixture.shard_";
    layout.tensor_suffix = ".weight";

    q4e::ple::PleRowSource source;
    source.kind = q4e::ple::PleRowSource::Kind::vram;
    source.layout = layout;
    source.device_table = table_dev.dev;
    source.table_rows = kTableRows;

    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: weight load sync\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    auto branch_result = q4e::ple::PleBranch::create(
        cfg, weights.layers[kPleLayer].ple, kMaxTokens, source, stream);
    check(branch_result.ok(),
          std::string("PleBranch create") +
              (branch_result.ok() ? ""
                                  : " (" + branch_result.status().message() + ")"));
    if (!branch_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    q4e::ple::PleBranch branch = branch_result.release();

    auto ngram_result =
        q4e::runtime::PleNgramContext::create(arena, kMaxSequences, kEosToken);
    auto conv_pool_result =
        q4e::runtime::PleConvStatePool::create(kMaxSequences, kHcHidden);
    check(ngram_result.ok() && conv_pool_result.ok(), "PLE state pool create");
    if (!ngram_result.ok() || !conv_pool_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    q4e::runtime::PleNgramContext ngram = ngram_result.release();
    q4e::runtime::PleConvStatePool conv_pool = conv_pool_result.release();

    q4e::Qwen4ExpLowerOptions lower_options;
    lower_options.connect_moe = false;
    lower_options.connect_ple = true;
    auto lower_result = q4e::lower_qwen4exp_to_primitives(cfg, weights, lower_options);
    check(lower_result.ok(),
          std::string("lowering (connect_ple=true)") +
              (lower_result.ok() ? ""
                                 : " (" + lower_result.status().message() + ")"));
    if (!lower_result.ok()) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    q4e::Qwen4ExpLoweredPrimitives lowered = lower_result.release();
    const rt::PrimitiveGraph& pre_g = lowered.ple_prefix_graph;
    const rt::PrimitiveGraph& main_g = lowered.graph;

    check(lowered.has_ple_prefix, "prefix graph present");
    check(pre_g.external_inputs.size() == 1, "prefix graph has one external input");
    check(pre_g.external_outputs.size() == 1, "prefix graph has one external output");
    if (!pre_g.external_outputs.empty()) {
        const rt::PrimitiveValueSpec* x4_spec = pre_g.value_spec(pre_g.external_outputs[0]);
        check(x4_spec != nullptr && x4_spec->dtype == rt::ValueDType::F32 &&
                  x4_spec->features == kHcHidden,
              "prefix graph exposes x4 as F32 hc_hidden");
    }
    {
        bool ple_in_prefix = false;
        for (const auto& n : pre_g.nodes) {
            if (n.debug_name.find(".ple.") != std::string::npos) ple_in_prefix = true;
        }
        check(!ple_in_prefix, "prefix graph has no PLE nodes");
        const std::vector<rt::PrimitiveKind> kinds = pre_g.kind_sequence();
        check(!kinds.empty() && kinds.front() == rt::PrimitiveKind::EMBEDDING_LOOKUP,
              "prefix graph starts with the embedding lookup");
    }

    check(main_g.external_inputs.size() == 2, "main graph has two external inputs");
    if (main_g.external_inputs.size() == 2) {
        const rt::PrimitiveValueSpec* x4_spec =
            main_g.value_spec(main_g.external_inputs[0]);
        const rt::PrimitiveValueSpec* ple_spec =
            main_g.value_spec(main_g.external_inputs[1]);
        check(x4_spec != nullptr && x4_spec->dtype == rt::ValueDType::F32 &&
                  x4_spec->features == kHcHidden,
              "main graph takes x4 as F32 hc_hidden");
        check(ple_spec != nullptr && ple_spec->dtype == rt::ValueDType::F32 &&
                  ple_spec->features == kHcHidden,
              "main graph takes the PLE output as F32 hc_hidden");
    }
    {
        const rt::PrimitiveGraphNode* ple_add = nullptr;
        for (const auto& n : main_g.nodes) {
            if (n.debug_name == "L1.ple.add") ple_add = &n;
        }
        check(ple_add != nullptr, "L1.ple.add node present");
        if (ple_add != nullptr) {
            check(rt::primitive_kind_of(ple_add->node) == rt::PrimitiveKind::RESIDUAL_ADD,
                  "L1.ple.add is a residual add");
            const bool uses_inputs =
                main_g.external_inputs.size() == 2 && ple_add->inputs.size() == 2 &&
                ple_add->inputs[0] == main_g.external_inputs[0] &&
                ple_add->inputs[1] == main_g.external_inputs[1];
            check(uses_inputs, "L1.ple.add consumes both external inputs");
        }
        bool embedding_in_main = false;
        for (const auto& n : main_g.nodes) {
            if (rt::primitive_kind_of(n.node) == rt::PrimitiveKind::EMBEDDING_LOOKUP)
                embedding_in_main = true;
        }
        check(!embedding_in_main, "main graph has no embedding lookup");
    }
    check(pre_g.states.size() == 2, "prefix graph owns the GDN conv and recurrence states");
    check(main_g.states.size() == 1, "main graph owns the QSA KV state");

    rt::ProgramBuildOptions build_options;
    build_options.validate_workspace = true;
    build_options.no_workspace_reuse = true;
    build_options.max_token_rows = kMaxTokens;
    build_options.max_output_rows = 1;

    const auto build = [&](const rt::PrimitiveGraph& g,
                           const std::vector<rt::WeightSlot>& w,
                           const std::vector<rt::StaticParameterSlot>& p, rt::Program& out,
                           const char* tag) {
        const rt::WeightTableView view{w.data(), w.size()};
        auto built = rt::build_program(g, view, rt::RowBucket::R16,
                                       rt::ExecutionClass::PREFILL, build_options);
        check(built.ok(), std::string("build_program ") + tag +
                              (built.ok() ? ""
                                          : " (" + built.status().message() + ")"));
        if (!built.ok()) return false;
        out = built.release();
        out.parameter_slot_count = static_cast<uint32_t>(p.size());
        return true;
    };

    rt::Program pre_plan;
    rt::Program main_plan;
    if (!build(pre_g, lowered.ple_prefix_weights, lowered.ple_prefix_parameters, pre_plan,
               "prefix") ||
        !build(main_g, lowered.weights, lowered.parameters, main_plan, "main")) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    auto seq_pool_result = state::SequenceSlotPool::create(arena, kMaxSequences, 4);
    const state::GdnStatePoolLayout gdn_layout = state::GdnStatePoolLayout::make(
        kGdnLayers, kConvF, kConvHistory, kValHeads, kHeadDim, kHeadDim);
    auto gdn_result = state::GdnStatePool::create(arena, kMaxSequences, gdn_layout);
    auto kv_result = state::PagedKVPool::create(arena, kNumPages, kPageTokens, kQsaLayers,
                                                kQsaKv, kQsaHeadDim,
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

    const rt::ValueId tokens_id = pre_g.external_inputs[0];
    const rt::ValueId x4_out_id = pre_g.external_outputs[0];
    const rt::ValueId x4_in_id = main_g.external_inputs[0];
    const rt::ValueId ple_in_id = main_g.external_inputs[1];
    const rt::ValueId hidden_id = main_g.external_outputs[0];
    const rt::ValueId logits_id = main_g.external_outputs[1];
    const rt::ValueId sampled_id = main_g.external_outputs[2];
    const rt::ValueId final_hidden_id = main_g.external_outputs[3];
    const rt::ValueId ple_add_id = node_output_by_debug_name(main_g, "L1.ple.add");
    check(ple_add_id.id != UINT32_MAX, "L1.ple.add output readable");

    Buf tok_dev, x4_dev, ple_dev, hidden_dev, logits_dev, sampled_dev, final_dev;
    if (!tok_dev.alloc(kMaxTokens * sizeof(int32_t)) ||
        !x4_dev.alloc(static_cast<size_t>(kMaxTokens) * kHcHidden * sizeof(float)) ||
        !ple_dev.alloc(static_cast<size_t>(kMaxTokens) * kHcHidden * sizeof(float)) ||
        !hidden_dev.alloc(kHidden * sizeof(float)) ||
        !logits_dev.alloc(kVocab * sizeof(float)) ||
        !sampled_dev.alloc(sizeof(int32_t)) ||
        !final_dev.alloc(static_cast<size_t>(kMaxTokens) * kHcHidden * sizeof(float))) {
        std::printf("FAIL: buffer allocation\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    patch_external(pre_plan, tokens_id, tok_dev.dev);
    patch_external(pre_plan, x4_out_id, x4_dev.dev);
    patch_external(main_plan, x4_in_id, x4_dev.dev);
    patch_external(main_plan, ple_in_id, ple_dev.dev);
    patch_external(main_plan, hidden_id, hidden_dev.dev);
    patch_external(main_plan, logits_id, logits_dev.dev);
    patch_external(main_plan, sampled_id, sampled_dev.dev);
    patch_external(main_plan, final_hidden_id, final_dev.dev);

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

    DeviceCtx pre_dc;
    DeviceCtx main_dc;

    struct RunResult {
        std::vector<float> logits;
        std::vector<float> hidden;
        std::vector<float> x4;
        std::vector<float> ple;
        std::vector<float> ple_add;
        int32_t sampled = -1;
        bool ran = false;
    };

    const auto run = [&](state::PagedSequenceState& seq,
                         const std::vector<int32_t>& tokens, uint32_t start_position,
                         bool prefill, RunResult& rr) {
        const uint32_t count = static_cast<uint32_t>(tokens.size());
        if (!tok_dev.fill(tokens.data(), count * sizeof(int32_t))) return;
        std::vector<uint32_t> slots(count);
        std::vector<uint32_t> positions(count);
        for (uint32_t i = 0; i < count; ++i) {
            slots[i] = seq.slot;
            positions[i] = start_position + i;
        }
        const uint32_t output_row = count - 1u;
        if (!slots_dev.fill(slots.data(), count * sizeof(uint32_t)) ||
            !pos_dev.fill(positions.data(), count * sizeof(uint32_t)) ||
            !out_rows_dev.fill(&output_row, sizeof(uint32_t))) {
            return;
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
        if (!req_dev.fill(&req, sizeof(req))) return;

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

        if (!run_program(pre_plan, lowered.ple_prefix_weights,
                         lowered.ple_prefix_parameters, pre_dc, ctx, state_view, stream)) {
            return;
        }
        if (!read_f32_device(x4_dev.dev, static_cast<size_t>(count) * kHcHidden, rr.x4)) {
            return;
        }

        const std::vector<std::uint32_t> tokens_u = to_tokens(tokens);
        rr.ple.resize(static_cast<size_t>(count) * kHcHidden);
        const ps::Status ple_st = branch.run(
            tokens_u.data(), count, rr.x4.data(), ngram, conv_pool, seq.slot,
            rr.ple.data(), stream);
        if (!ple_st.ok()) {
            std::printf("FAIL: PLE branch: %s\n", ple_st.message().c_str());
            return;
        }
        if (!ple_dev.fill(rr.ple.data(), rr.ple.size() * sizeof(float))) return;

        const bool main_ok = run_program(main_plan, lowered.weights, lowered.parameters,
                                         main_dc, ctx, state_view, stream);
        if (!main_ok) {
            return;
        }
        rr.ran = true;
        if (!read_f32_device(hidden_dev.dev, kHidden, rr.hidden)) rr.ran = false;
        if (!read_f32_device(logits_dev.dev, kVocab, rr.logits)) rr.ran = false;
        if (hipMemcpy(&rr.sampled, sampled_dev.dev, sizeof(int32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            rr.ran = false;
        }
        if (!read_value_rows(main_plan, main_dc.workspace, ple_add_id, count, rr.ple_add))
            rr.ran = false;
    };

    const auto check_ple_oracle = [&](const char* tag, OracleState& oracle,
                                      const std::vector<int32_t>& tokens,
                                      const RunResult& rr) {
        const std::vector<std::uint32_t> t = to_tokens(tokens);
        const std::vector<float> ref =
            oracle_ple(oracle, t, rr.x4, table_host, branch,
                       static_cast<std::uint32_t>(tokens.size()));
        const Metrics m = compare(rr.ple.data(), ref.data(), rr.ple.size());
        check_metrics(tag, m, 1e-5, 1e-5, 1e-6);
    };

    const auto check_add = [&](const char* tag, const RunResult& rr, uint32_t rows,
                               uint32_t row_offset) {
        std::vector<float> ref(static_cast<size_t>(rows) * kHcHidden);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::uint32_t c = 0; c < kHcHidden; ++c) {
                const std::size_t src =
                    (static_cast<std::size_t>(row_offset) + r) * kHcHidden + c;
                ref[r * kHcHidden + c] = rr.x4[src] + rr.ple[src];
            }
        }
        const Metrics m = compare(rr.ple_add.data() +
                                      static_cast<size_t>(row_offset) * kHcHidden,
                                  ref.data(), static_cast<size_t>(rows) * kHcHidden);
        check_metrics(tag, m, 0.0, 0.0, 0.0);
    };

    const auto new_sequence = [&](std::optional<state::PagedSequenceState>& out) {
        out.reset();
        auto r = state::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, 16u,
                                                     stream);
        if (!r.ok()) {
            std::printf("FAIL: sequence create: %s\n", r.status().message().c_str());
            return false;
        }
        out.emplace(r.release());
        if (!state::reserve_sequence_append(*out, 8u, stream).ok()) {
            std::printf("FAIL: reserve sequence\n");
            return false;
        }
        if (!ngram.initialize(out->slot).ok()) {
            std::printf("FAIL: ngram slot initialize\n");
            return false;
        }
        if (!conv_pool.initialize(out->slot).ok()) {
            std::printf("FAIL: conv slot initialize\n");
            return false;
        }
        return true;
    };

    std::optional<state::PagedSequenceState> seq_a;
    std::optional<state::PagedSequenceState> seq_b;
    std::optional<state::PagedSequenceState> seq_c;
    std::optional<state::PagedSequenceState> seq_d;
    std::optional<state::PagedSequenceState> seq_e;
    std::optional<state::PagedSequenceState> seq_f;
    if (!new_sequence(seq_a) || !new_sequence(seq_b) || !new_sequence(seq_c) ||
        !new_sequence(seq_d) || !new_sequence(seq_e) || !new_sequence(seq_f)) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: stream sync after reserve\n");
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }

    const std::vector<int32_t> tokens4{3, 11, 19, 23};
    const std::vector<int32_t> tokens_head{3, 11};
    const std::vector<int32_t> tokens_tail{19, 23};
    const std::vector<int32_t> decode_token{29};
    const std::vector<int32_t> tokens5{3, 11, 19, 23, 29};
    const std::vector<int32_t> eos_head{3, 7};
    const std::vector<int32_t> eos_tail{11};
    const std::vector<int32_t> eos_all{3, 7, 11};

    const std::vector<std::uint32_t> ctx_initial = context_of(ngram, seq_a->slot);
    check(ctx_initial[0] == kEosToken && ctx_initial[1] == kEosToken,
          "fresh n-gram context is eos, eos");
    std::vector<float> conv_fresh;
    {
        const q4e::ple::PleConvState* st = conv_pool.state(seq_a->slot);
        check(st != nullptr, "fresh conv state available");
        conv_fresh = st->buffer;
    }

    OracleState oracle_a(kEosToken, kHcHidden);
    RunResult run_a;
    run(*seq_a, tokens4, 0u, true, run_a);
    check(run_a.ran, "run A (prefill 4 tokens) executes");
    if (!run_a.ran) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    check_ple_oracle("PLE branch vs spec oracle (prefill 4)", oracle_a, tokens4, run_a);
    check_add("graph ple.add vs host x4 + PLE (bitwise)", run_a, 4u, 0u);
    check(context_of(ngram, seq_a->slot) ==
              std::vector<std::uint32_t>({19u, 23u}),
          "n-gram context advanced to the last two tokens");
    {
        const q4e::ple::PleConvState* st = conv_pool.state(seq_a->slot);
        check(st != nullptr &&
                  std::memcmp(conv_fresh.data(), st->buffer.data(),
                              conv_fresh.size() * sizeof(float)) != 0,
              "PLE conv state changed by prefill");
    }

    OracleState oracle_b(kEosToken, kHcHidden);
    RunResult run_b1;
    RunResult run_b2;
    run(*seq_b, tokens_head, 0u, true, run_b1);
    check(run_b1.ran, "run B1 (prefill first 2 tokens) executes");
    check_ple_oracle("PLE branch vs spec oracle (chunk 2 tokens)", oracle_b, tokens_head,
                     run_b1);
    run(*seq_b, tokens_tail, 2u, true, run_b2);
    check(run_b2.ran, "run B2 (prefill next 2 tokens, continued PLE state) executes");
    check_ple_oracle("PLE branch vs spec oracle (chunk boundary)", oracle_b, tokens_tail,
                     run_b2);
    if (run_b2.ran) {
        const Metrics m = compare(run_b2.logits.data(), run_a.logits.data(), kVocab);
        check_metrics("split-equivalence logits (B2 vs A)", m, 1e-4, 1e-4, 1e-4);
        const Metrics mh = compare(run_b2.hidden.data(), run_a.hidden.data(), kHidden);
        check_metrics("split-equivalence selected hidden (B2 vs A)", mh, 1e-4, 1e-4, 1e-4);
        check_add("split-equivalence graph ple.add (bitwise)", run_b2, 2u, 0u);
        const Metrics mp = compare(run_b2.ple_add.data(),
                                   run_a.ple_add.data() + 2u * kHcHidden,
                                   static_cast<size_t>(2) * kHcHidden);
        check_metrics("split-equivalence ple.add rows vs single chunk (bitwise)", mp, 0.0,
                      0.0, 0.0);
    }

    OracleState oracle_c(kEosToken, kHcHidden);
    RunResult run_c1;
    RunResult run_c2;
    RunResult run_c_decode;
    run(*seq_c, tokens_head, 0u, true, run_c1);
    check_ple_oracle("PLE branch vs spec oracle (chunk 2 tokens)", oracle_c, tokens_head,
                     run_c1);
    run(*seq_c, tokens_tail, 2u, true, run_c2);
    check_ple_oracle("PLE branch vs spec oracle (chunk boundary)", oracle_c, tokens_tail,
                     run_c2);
    run(*seq_c, decode_token, 4u, false, run_c_decode);
    check(run_c_decode.ran, "run decode (1 token after split prefill) executes");
    check_ple_oracle("PLE branch vs spec oracle (decode 1 token)", oracle_c, decode_token,
                     run_c_decode);

    OracleState oracle_d(kEosToken, kHcHidden);
    RunResult run_d;
    run(*seq_d, tokens5, 0u, true, run_d);
    check(run_d.ran, "run D (prefill 5 tokens) executes");
    if (run_c_decode.ran && run_d.ran) {
        const Metrics m = compare(run_c_decode.logits.data(), run_d.logits.data(), kVocab);
        check_metrics("decode-continuation logits (decode after split vs prefill5)", m,
                      1e-4, 1e-4, 1e-4);
        const Metrics mp =
            compare(run_c_decode.ple_add.data(), run_d.ple_add.data() + 4u * kHcHidden,
                    static_cast<size_t>(1) * kHcHidden);
        check_metrics("decode-continuation graph ple.add (bitwise)", mp, 0.0, 0.0, 0.0);
    }

    OracleState oracle_e(kEosToken, kHcHidden);
    RunResult run_e1;
    RunResult run_e2;
    run(*seq_e, eos_head, 0u, true, run_e1);
    check_ple_oracle("PLE branch vs spec oracle (eos chunk)", oracle_e, eos_head, run_e1);
    run(*seq_e, eos_tail, 2u, true, run_e2);
    check(run_e2.ran, "run EOS split (chunk boundary across the eos token) executes");
    check_ple_oracle("PLE branch vs spec oracle (eos chunk)", oracle_e, eos_tail, run_e2);
    check(context_of(ngram, seq_e->slot) == std::vector<std::uint32_t>({7u, 11u}),
          "n-gram context keeps the eos token");

    OracleState oracle_f(kEosToken, kHcHidden);
    RunResult run_f;
    run(*seq_f, eos_all, 0u, true, run_f);
    check(run_f.ran, "run EOS one-shot (3 tokens) executes");
    if (run_f.ran && run_e2.ran) {
        const Metrics m = compare(run_e2.logits.data(), run_f.logits.data(), kVocab);
        check_metrics("eos split-equivalence logits (chunk across eos vs one-shot)", m,
                      1e-4, 1e-4, 1e-4);
    }

    const state::SequenceSlotId reuse_slot = seq_b->slot;
    check(ngram.reset(reuse_slot).ok(), "cancel resets the n-gram context");
    check(conv_pool.reset(reuse_slot).ok(), "cancel resets the PLE conv state");
    check(context_of(ngram, reuse_slot) ==
              std::vector<std::uint32_t>({kEosToken, kEosToken}),
          "cancel restores the eos context");
    {
        const q4e::ple::PleConvState* st = conv_pool.state(reuse_slot);
        bool all_zero = st != nullptr;
        if (st != nullptr) {
            for (float v : st->buffer) {
                if (v != 0.0f) all_zero = false;
            }
        }
        check(all_zero, "cancel zeroes the PLE conv state");
    }

    OracleState oracle_cancel(kEosToken, kHcHidden);
    RunResult run_cancel;
    run(*seq_b, tokens4, 4u, true, run_cancel);
    check(run_cancel.ran, "run after cancel executes");
    check_ple_oracle("PLE branch vs spec oracle (fresh state after cancel)", oracle_cancel,
                     tokens4, run_cancel);

    check(state::release_paged_sequence_state(*seq_b, stream).ok(),
          "release the sequence state");
    check(ngram.release(reuse_slot).ok(), "ngram slot release before reuse");
    check(conv_pool.release(reuse_slot).ok(), "conv slot release before reuse");
    if (!new_sequence(seq_b)) {
        arena.shutdown();
        fs::remove_all(g_dir);
        return 1;
    }
    check(seq_b->slot == reuse_slot, "request reuse returns the same sequence slot");
    OracleState oracle_reuse(kEosToken, kHcHidden);
    RunResult run_reuse;
    run(*seq_b, tokens4, 0u, true, run_reuse);
    check(run_reuse.ran, "run reuse (same slot after release) executes");
    if (run_reuse.ran) {
        const Metrics m = compare(run_reuse.logits.data(), run_a.logits.data(), kVocab);
        check_metrics("request-reuse logits vs fresh slot (bitwise)", m, 0.0, 0.0, 0.0);
        const Metrics mp =
            compare(run_reuse.ple.data(), run_a.ple.data(), run_a.ple.size());
        check_metrics("request-reuse PLE output vs fresh slot (bitwise)", mp, 0.0, 0.0,
                      0.0);
        check_ple_oracle("PLE branch vs spec oracle (reuse)", oracle_reuse, tokens4,
                         run_reuse);
    }

    for (std::optional<state::PagedSequenceState>* s :
         {&seq_a, &seq_b, &seq_c, &seq_d, &seq_e, &seq_f}) {
        check(ngram.release((*s)->slot).ok(), "ngram slot release");
        check(conv_pool.release((*s)->slot).ok(), "conv slot release");
        (void)state::release_paged_sequence_state(**s, stream);
    }
    check(branch.shutdown().ok(), "PLE branch shutdown");
    check(ngram.shutdown().ok(), "ngram shutdown");
    check(conv_pool.shutdown().ok(), "conv pool shutdown");

    std::printf("test_qwen4exp_ple_graph: passed=%d failed=%d\n", g_pass, g_fail);
    arena.shutdown();
    fs::remove_all(g_dir);
    return g_fail == 0 ? 0 : 1;
}
