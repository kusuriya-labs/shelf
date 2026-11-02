#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/gfx_arch.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace shelf;

static int g_fail = 0;

static void fail(const std::string& msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static qwen35::Qwen35LoadOptions mtp_load_options() {
    qwen35::Qwen35LoadOptions o;
    o.load_mtp_layers = true;
    return o;
}

static void check_eq_u64(uint64_t got, uint64_t want, const std::string& msg) {
    if (got != want) fail(msg + " got=" + std::to_string(got) + " want=" + std::to_string(want));
}

static void check_contains(const std::string& haystack, const std::string& needle,
                           const std::string& tag) {
    if (haystack.find(needle) == std::string::npos) {
        fail(tag + ": message missing '" + needle + "' in: " + haystack);
    }
}

static std::vector<uint8_t> d2h(const void* dev, std::size_t bytes) {
    std::vector<uint8_t> host(bytes);
    if (bytes == 0) return host;
    const hipError_t e = hipMemcpy(host.data(), dev, bytes, hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        fail(std::string("d2h ") + hipGetErrorString(e));
        host.assign(bytes, 0);
    }
    return host;
}

static std::size_t tensor_span_bytes(const gpu::Tensor& t) {
    std::size_t span = 0;
    if (!t.required_span_bytes(span).ok()) {
        fail("required_span_bytes");
        return 0;
    }
    return span;
}

static std::size_t tensor_element_count(const gpu::Tensor& t) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < t.ndim(); ++i) n *= t.dim(i);
    return n;
}

static std::vector<uint8_t> bf16_bytes(uint64_t n, uint32_t seed) {
    std::vector<uint8_t> out(n * 2u);
    auto* u = reinterpret_cast<uint16_t*>(out.data());
    for (uint64_t i = 0; i < n; ++i) {
        u[i] = static_cast<uint16_t>(0x3F80u + ((seed + static_cast<uint32_t>(i)) % 64u));
    }
    return out;
}

static void write_text(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << text;
}

struct Planned {
    std::string name;
    io::SType dtype;
    std::vector<std::size_t> shape;
    std::vector<uint8_t> bytes;
};

static uint64_t numel(const std::vector<std::size_t>& shape) {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

static bool write_shard(const std::string& path, std::vector<Planned>& tensors) {
    auto w = io::SafetensorsWriter::create(path);
    if (!w.ok()) { fail("writer create"); return false; }
    io::SafetensorsWriter writer = w.release();
    for (auto& t : tensors) {
        if (!writer.plan_tensor(t.name, t.dtype, t.shape).ok()) { fail("plan " + t.name); return false; }
    }
    auto hdr = writer.write_header();
    if (!hdr.ok()) { fail("write_header"); return false; }
    for (auto& t : tensors) {
        if (!writer.write_tensor(t.name, t.bytes.data(), t.bytes.size()).ok()) { fail("write " + t.name); return false; }
    }
    if (!writer.finish().ok()) { fail("finish"); return false; }
    return true;
}

static fs::path g_root;

static fs::path make_case_dir(const std::string& name) {
    const fs::path dir = g_root / name;
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir;
}

static void write_index(const std::string& dir, const std::string& shard,
                        const std::vector<Planned>& tensors) {
    std::string j = "{\"weight_map\":{";
    bool first = true;
    for (auto& t : tensors) {
        if (!first) j += ",";
        first = false;
        j += "\"" + t.name + "\":\"" + shard + "\"";
    }
    j += "},\"metadata\":{\"total_size\":0}}";
    write_text(dir + "/model.safetensors.index.json", j);
}

// Contract constants (H=32 model):
//   H=32 I=48 V=40
//   q_heads=2 kv_heads=1 head_dim=16 -> Q=32 KV=16 q_proj_out=2Q=64
//   MTP layer == target body full-attention decoder layer contract
static const std::string kConfigJson =
    "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,\"text_config\":{"
    "\"model_type\":\"qwen3_5_text\",\"vocab_size\":40,\"hidden_size\":32,\"intermediate_size\":48,"
    "\"num_hidden_layers\":1,\"full_attention_interval\":1,\"layer_types\":[\"full_attention\"],"
    "\"num_attention_heads\":2,\"num_key_value_heads\":1,\"head_dim\":16,"
    "\"rms_norm_eps\":1e-06,"
    "\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25}"
    "}}";

struct MtpFixtureOptions {
    bool include_mtp = true;
    bool include_second_mtp_layer = false;
    std::string drop_tensor;
    std::vector<std::size_t> mtp_fc_shape = {32, 64};
    std::vector<std::size_t> pre_fc_norm_hidden_shape = {32};
    std::vector<std::size_t> mtp_gate_shape = {48, 32};
    std::vector<std::size_t> mtp_q_proj_shape = {64, 32};
    std::vector<std::size_t> mtp_k_proj_shape = {16, 32};
};

struct MtpFixture {
    fs::path dir;
    std::map<std::string, std::vector<uint8_t>> mtp_source;
};

static Planned make_tensor(const std::string& name, std::vector<std::size_t> shape, uint32_t seed) {
    Planned p;
    p.name = name;
    p.dtype = io::SType::BF16;
    p.shape = std::move(shape);
    p.bytes = bf16_bytes(numel(p.shape), seed);
    return p;
}

static MtpFixture build_mtp_model(const std::string& name, const MtpFixtureOptions& opts) {
    const fs::path dir = make_case_dir(name);
    std::vector<Planned> t;
    uint32_t seed = 1000;
    const auto add = [&](const std::string& nm, std::vector<std::size_t> shape) {
        if (nm == opts.drop_tensor) return;
        t.push_back(make_tensor(nm, std::move(shape), seed++));
    };

    add("model.language_model.embed_tokens.weight", {40, 32});
    add("model.language_model.lm_head.weight", {40, 32});
    add("model.language_model.norm.weight", {32});

    const std::string L = "model.language_model.layers.0.";
    add(L + "input_layernorm.weight", {32});
    add(L + "post_attention_layernorm.weight", {32});
    add(L + "mlp.gate_proj.weight", {48, 32});
    add(L + "mlp.up_proj.weight", {48, 32});
    add(L + "mlp.down_proj.weight", {32, 48});
    add(L + "self_attn.q_proj.weight", {64, 32});
    add(L + "self_attn.k_proj.weight", {16, 32});
    add(L + "self_attn.v_proj.weight", {16, 32});
    add(L + "self_attn.o_proj.weight", {32, 32});
    add(L + "self_attn.q_norm.weight", {16});
    add(L + "self_attn.k_norm.weight", {16});

    MtpFixture fixture;
    fixture.dir = dir;

    if (opts.include_mtp) {
        auto add_mtp = [&](const std::string& nm, std::vector<std::size_t> shape) {
            if (nm == opts.drop_tensor) return;
            Planned p = make_tensor(nm, std::move(shape), seed++);
            fixture.mtp_source[p.name] = p.bytes;
            t.push_back(std::move(p));
        };
        add_mtp("mtp.pre_fc_norm_embedding.weight", {32});
        add_mtp("mtp.pre_fc_norm_hidden.weight", opts.pre_fc_norm_hidden_shape);
        add_mtp("mtp.norm.weight", {32});
        add_mtp("mtp.fc.weight", opts.mtp_fc_shape);
        add_mtp("mtp.layers.0.input_layernorm.weight", {32});
        add_mtp("mtp.layers.0.post_attention_layernorm.weight", {32});
        add_mtp("mtp.layers.0.mlp.gate_proj.weight", opts.mtp_gate_shape);
        add_mtp("mtp.layers.0.mlp.up_proj.weight", {48, 32});
        add_mtp("mtp.layers.0.mlp.down_proj.weight", {32, 48});
        add_mtp("mtp.layers.0.self_attn.q_proj.weight", opts.mtp_q_proj_shape);
        add_mtp("mtp.layers.0.self_attn.k_proj.weight", opts.mtp_k_proj_shape);
        add_mtp("mtp.layers.0.self_attn.v_proj.weight", {16, 32});
        add_mtp("mtp.layers.0.self_attn.o_proj.weight", {32, 32});
        add_mtp("mtp.layers.0.self_attn.q_norm.weight", {16});
        add_mtp("mtp.layers.0.self_attn.k_norm.weight", {16});
        if (opts.include_second_mtp_layer) {
            add_mtp("mtp.layers.1.input_layernorm.weight", {32});
            add_mtp("mtp.layers.1.post_attention_layernorm.weight", {32});
            add_mtp("mtp.layers.1.mlp.gate_proj.weight", {48, 32});
            add_mtp("mtp.layers.1.mlp.up_proj.weight", {48, 32});
            add_mtp("mtp.layers.1.mlp.down_proj.weight", {32, 48});
            add_mtp("mtp.layers.1.self_attn.q_proj.weight", {64, 32});
            add_mtp("mtp.layers.1.self_attn.k_proj.weight", {16, 32});
            add_mtp("mtp.layers.1.self_attn.v_proj.weight", {16, 32});
            add_mtp("mtp.layers.1.self_attn.o_proj.weight", {32, 32});
            add_mtp("mtp.layers.1.self_attn.q_norm.weight", {16});
            add_mtp("mtp.layers.1.self_attn.k_norm.weight", {16});
        }
    }

    if (!write_shard((dir / "model.safetensors-00001-of-00001.safetensors").string(), t)) {
        return fixture;
    }
    write_index(dir.string(), "model.safetensors-00001-of-00001.safetensors", t);
    write_text((dir / "config.json").string(), kConfigJson);
    return fixture;
}

static const char* kMtpMatrixNames[] = {
    "mtp.fc.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
};

static const char* kMtpVectorNames[] = {
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
};

static void check_matrix_payload(const weights::MatrixWeight& w,
                                 const std::vector<uint8_t>& source,
                                 const std::string& tag) {
    if (w.encoding != weights::MatrixEncoding::Bf16) {
        fail(tag + ": encoding not bf16");
        return;
    }
    check_eq_u64(source.size(), tensor_span_bytes(w.bf16), tag + " span");
    const auto got = d2h(w.bf16.data<void>(), source.size());
    check(got == source, tag + " raw payload");
}

// Case A + B: correct MTP BF16 checkpoint.
static void test_mtp_bf16_contract(gpu::GpuArena& arena, hipStream_t stream) {
    MtpFixture fixture = build_mtp_model("mtp_correct", MtpFixtureOptions{});

    qwen35::Qwen35LoadOptions options;
    options.load_mtp_layers = true;
    auto wr = qwen35::load_qwen35_weights_from_safetensors(fixture.dir.string(), arena, stream, options);
    if (!wr.ok()) { fail("mtp load: " + wr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = wr.release();

    check(w.mtp.present, "mtp present");
    check(!w.mtp.layer.is_gdn, "mtp layer is full attention");

    check_eq_u64(w.mtp.fc.rows, 32, "case A mtp.fc.rows == H");
    check_eq_u64(w.mtp.fc.cols, 64, "case A mtp.fc.cols == 2H");
    check(w.mtp.fc.encoding == weights::MatrixEncoding::Bf16, "case A mtp.fc encoding");

    check_eq_u64(w.mtp.layer.mlp_gate_proj.rows, 48, "mtp gate rows == I");
    check_eq_u64(w.mtp.layer.mlp_gate_proj.cols, 32, "mtp gate cols == H");
    check_eq_u64(w.mtp.layer.mlp_down_proj.rows, 32, "mtp down rows == H");
    check_eq_u64(w.mtp.layer.mlp_down_proj.cols, 48, "mtp down cols == I");
    check_eq_u64(w.mtp.layer.attn_q_proj.rows, 64, "mtp q_proj rows == 2Q");
    check_eq_u64(w.mtp.layer.attn_q_proj.cols, 32, "mtp q_proj cols == H");
    check_eq_u64(w.mtp.layer.attn_k_proj.rows, 16, "mtp k_proj rows == KV");
    check_eq_u64(w.mtp.layer.attn_v_proj.rows, 16, "mtp v_proj rows == KV");
    check_eq_u64(w.mtp.layer.attn_o_proj.rows, 32, "mtp o_proj rows == H");
    check_eq_u64(w.mtp.layer.attn_o_proj.cols, 32, "mtp o_proj cols == Q");

    check_eq_u64(tensor_element_count(w.mtp.pre_fc_norm_embedding_weight), 32, "pre_fc_norm_embedding [H]");
    check_eq_u64(tensor_element_count(w.mtp.pre_fc_norm_hidden_weight), 32, "pre_fc_norm_hidden [H]");
    check_eq_u64(tensor_element_count(w.mtp.final_norm_weight), 32, "mtp.norm [H]");
    check_eq_u64(tensor_element_count(w.mtp.layer.input_layernorm_weight), 32, "mtp input_layernorm [H]");
    check_eq_u64(tensor_element_count(w.mtp.layer.post_attention_layernorm_weight), 32, "mtp post_attention_layernorm [H]");
    check_eq_u64(tensor_element_count(w.mtp.layer.attn_q_norm_weight), 16, "mtp q_norm [head_dim]");
    check_eq_u64(tensor_element_count(w.mtp.layer.attn_k_norm_weight), 16, "mtp k_norm [head_dim]");

    const weights::MatrixWeight* matrices[8] = {
        &w.mtp.fc,
        &w.mtp.layer.mlp_gate_proj,
        &w.mtp.layer.mlp_up_proj,
        &w.mtp.layer.mlp_down_proj,
        &w.mtp.layer.attn_q_proj,
        &w.mtp.layer.attn_k_proj,
        &w.mtp.layer.attn_v_proj,
        &w.mtp.layer.attn_o_proj,
    };
    const gpu::Tensor* vectors[7] = {
        &w.mtp.pre_fc_norm_embedding_weight,
        &w.mtp.pre_fc_norm_hidden_weight,
        &w.mtp.final_norm_weight,
        &w.mtp.layer.input_layernorm_weight,
        &w.mtp.layer.post_attention_layernorm_weight,
        &w.mtp.layer.attn_q_norm_weight,
        &w.mtp.layer.attn_k_norm_weight,
    };
    for (int i = 0; i < 8; ++i) {
        const auto it = fixture.mtp_source.find(kMtpMatrixNames[i]);
        check(it != fixture.mtp_source.end(), std::string("case B source tracked: ") + kMtpMatrixNames[i]);
        if (it == fixture.mtp_source.end()) return;
        check_matrix_payload(*matrices[i], it->second, std::string("case B payload ") + kMtpMatrixNames[i]);
    }
    for (int i = 0; i < 7; ++i) {
        const auto it = fixture.mtp_source.find(kMtpVectorNames[i]);
        check(it != fixture.mtp_source.end(), std::string("case B source tracked: ") + kMtpVectorNames[i]);
        if (it == fixture.mtp_source.end()) return;
        check_eq_u64(it->second.size(), tensor_span_bytes(*vectors[i]),
                     std::string("case B span ") + kMtpVectorNames[i]);
        const auto got = d2h(vectors[i]->data<void>(), it->second.size());
        check(got == it->second, std::string("case B payload ") + kMtpVectorNames[i]);
    }

    auto config_res = qwen35::read_qwen35_text_config(fixture.dir.string());
    check(config_res.ok(), "config read");
    if (!config_res.ok()) return;
    qwen35::Qwen35TextConfig config = config_res.release();

    auto lower = qwen35::lower_qwen35_mtp_to_primitives(config, w);
    check(lower.ok(), std::string("case A lower mtp: ") + (lower.ok() ? "" : lower.status().message().c_str()));
    auto body_lower = qwen35::lower_qwen35_to_primitives(config, w);
    check(body_lower.ok(), std::string("case A lower body: ") +
                               (body_lower.ok() ? "" : body_lower.status().message().c_str()));

    qwen35::Qwen35LoadOptions no_pre;
    no_pre.load_mtp_layers = true;
    no_pre.weights.preshuffle = false;
    auto wr2 = qwen35::load_qwen35_weights_from_safetensors(fixture.dir.string(), arena, stream, no_pre);
    if (!wr2.ok()) { fail("mtp load skip_preshuffle: " + wr2.status().message()); return; }
    qwen35::Qwen35ModelWeights w2 = wr2.release();
    check(!w2.mtp.fc.preshuffled, "skip_preshuffle mtp.fc raw");
    const auto got = d2h(w2.mtp.fc.bf16.data<void>(), 32 * 64 * 2);
    const auto it = fixture.mtp_source.find("mtp.fc.weight");
    check(got == it->second, "case A skip_preshuffle mtp.fc raw payload");
}

// Case C: checkpoint without MTP tensors.
static void test_mtp_absent(gpu::GpuArena& arena, hipStream_t stream) {
    MtpFixtureOptions opts;
    opts.include_mtp = false;
    MtpFixture fixture = build_mtp_model("mtp_absent", opts);

    auto wr = qwen35::load_qwen35_weights_from_safetensors(fixture.dir.string(), arena, stream,
                                                           mtp_load_options());
    if (!wr.ok()) { fail("mtp absent load: " + wr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = wr.release();
    check(!w.mtp.present, "case C mtp.present == false");

    auto config_res = qwen35::read_qwen35_text_config(fixture.dir.string());
    if (!config_res.ok()) { fail("mtp absent config"); return; }
    qwen35::Qwen35TextConfig config = config_res.release();
    auto lower = qwen35::lower_qwen35_mtp_to_primitives(config, w);
    check(!lower.ok(), "case C lower mtp must fail without MTP weights");
    check_contains(lower.status().message(), "model has no MTP weights", "case C lower");
}

// Case D: mtp.fc transposed to [64,32].
static void test_mtp_fc_transposed(gpu::GpuArena& arena, hipStream_t stream) {
    MtpFixtureOptions opts;
    opts.mtp_fc_shape = {64, 32};
    MtpFixture fixture = build_mtp_model("mtp_fc_t", opts);

    auto wr = qwen35::load_qwen35_weights_from_safetensors(fixture.dir.string(), arena, stream,
                                                           mtp_load_options());
    if (!wr.ok()) { fail("case D load: " + wr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = wr.release();
    check(w.mtp.present, "case D mtp present");
    check_eq_u64(w.mtp.fc.rows, 64, "case D fc rows as loaded");

    auto config_res = qwen35::read_qwen35_text_config(fixture.dir.string());
    if (!config_res.ok()) { fail("case D config"); return; }
    qwen35::Qwen35TextConfig config = config_res.release();
    auto lower = qwen35::lower_qwen35_mtp_to_primitives(config, w);
    check(!lower.ok(), "case D lower must fail on transposed fc");
    check_contains(lower.status().message(), "mtp.fc.weight", "case D message name");
    check_contains(lower.status().message(), "expected [32,64]", "case D message expected");
    check_contains(lower.status().message(), "got [64,32]", "case D message got");
}

// Case E: mtp.pre_fc_norm_hidden.weight missing.
static void test_mtp_missing_norm(gpu::GpuArena& arena, hipStream_t stream) {
    MtpFixtureOptions opts;
    opts.drop_tensor = "mtp.pre_fc_norm_hidden.weight";
    MtpFixture fixture = build_mtp_model("mtp_missing_norm", opts);

    auto wr = qwen35::load_qwen35_weights_from_safetensors(fixture.dir.string(), arena, stream,
                                                           mtp_load_options());
    check(!wr.ok(), "case E load must fail on missing norm");
    if (!wr.ok()) {
        check_contains(wr.status().message(), "mtp.pre_fc_norm_hidden.weight", "case E message");
    }
}

// Case F: mtp.pre_fc_norm_hidden.weight has 31 elements.
static void test_mtp_wrong_norm_size(gpu::GpuArena& arena, hipStream_t stream) {
    MtpFixtureOptions opts;
    opts.pre_fc_norm_hidden_shape = {31};
    MtpFixture fixture = build_mtp_model("mtp_norm31", opts);

    auto wr = qwen35::load_qwen35_weights_from_safetensors(fixture.dir.string(), arena, stream,
                                                           mtp_load_options());
    if (!wr.ok()) { fail("case F load: " + wr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = wr.release();

    auto config_res = qwen35::read_qwen35_text_config(fixture.dir.string());
    if (!config_res.ok()) { fail("case F config"); return; }
    qwen35::Qwen35TextConfig config = config_res.release();
    auto lower = qwen35::lower_qwen35_mtp_to_primitives(config, w);
    check(!lower.ok(), "case F lower must fail on wrong norm size");
    check_contains(lower.status().message(), "mtp.pre_fc_norm_hidden.weight", "case F message name");
    check_contains(lower.status().message(), "got [31]", "case F message got");
}

// Case G: mtp gate_proj transposed to [32,48].
static void test_mtp_mlp_transposed(gpu::GpuArena& arena, hipStream_t stream) {
    MtpFixtureOptions opts;
    opts.mtp_gate_shape = {32, 48};
    MtpFixture fixture = build_mtp_model("mtp_gate_t", opts);

    auto wr = qwen35::load_qwen35_weights_from_safetensors(fixture.dir.string(), arena, stream,
                                                           mtp_load_options());
    if (!wr.ok()) { fail("case G load: " + wr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = wr.release();

    auto config_res = qwen35::read_qwen35_text_config(fixture.dir.string());
    if (!config_res.ok()) { fail("case G config"); return; }
    qwen35::Qwen35TextConfig config = config_res.release();
    auto lower = qwen35::lower_qwen35_mtp_to_primitives(config, w);
    check(!lower.ok(), "case G lower must fail on transposed gate");
    check_contains(lower.status().message(), "mtp.layers.0.mlp.gate_proj.weight", "case G message name");
    check_contains(lower.status().message(), "expected [48,32]", "case G message expected");
    check_contains(lower.status().message(), "got [32,48]", "case G message got");
}

// Case H: mtp q_proj transposed to [32,]; k_proj with wrong feature count [17,32].
static void test_mtp_attn_wrong_geometry(gpu::GpuArena& arena, hipStream_t stream) {
    MtpFixtureOptions opts;
    opts.mtp_q_proj_shape = {32, 64};
    MtpFixture fixture = build_mtp_model("mtp_q_t", opts);

    auto wr = qwen35::load_qwen35_weights_from_safetensors(fixture.dir.string(), arena, stream,
                                                           mtp_load_options());
    if (!wr.ok()) { fail("case H load: " + wr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = wr.release();

    auto config_res = qwen35::read_qwen35_text_config(fixture.dir.string());
    if (!config_res.ok()) { fail("case H config"); return; }
    qwen35::Qwen35TextConfig config = config_res.release();
    auto lower = qwen35::lower_qwen35_mtp_to_primitives(config, w);
    check(!lower.ok(), "case H lower must fail on transposed q_proj");
    check_contains(lower.status().message(), "mtp.layers.0.self_attn.q_proj.weight", "case H message name");
    check_contains(lower.status().message(), "expected [64,32]", "case H message expected");
    check_contains(lower.status().message(), "got [32,64]", "case H message got");

    MtpFixtureOptions opts2;
    opts2.mtp_k_proj_shape = {17, 32};
    MtpFixture fixture2 = build_mtp_model("mtp_k_bad", opts2);
    auto wr2 = qwen35::load_qwen35_weights_from_safetensors(fixture2.dir.string(), arena, stream,
                                                            mtp_load_options());
    if (!wr2.ok()) { fail("case H2 load: " + wr2.status().message()); return; }
    qwen35::Qwen35ModelWeights w2 = wr2.release();
    auto lower2 = qwen35::lower_qwen35_mtp_to_primitives(config, w2);
    check(!lower2.ok(), "case H2 lower must fail on wrong k_proj features");
    check_contains(lower2.status().message(), "mtp.layers.0.self_attn.k_proj.weight", "case H2 message name");
    check_contains(lower2.status().message(), "expected [16,32]", "case H2 message expected");
    check_contains(lower2.status().message(), "got [17,32]", "case H2 message got");
}

// Case I: checkpoint with two MTP layers must be rejected, not silently trimmed.
static void test_mtp_multi_layer_rejected(gpu::GpuArena& arena, hipStream_t stream) {
    MtpFixtureOptions opts;
    opts.include_second_mtp_layer = true;
    MtpFixture fixture = build_mtp_model("mtp_2layers", opts);

    auto wr = qwen35::load_qwen35_weights_from_safetensors(fixture.dir.string(), arena, stream,
                                                           mtp_load_options());
    check(!wr.ok(), "case I load must fail with 2 MTP layers");
    if (!wr.ok()) {
        check_contains(wr.status().message(), "checkpoint contains 2 MTP layers", "case I message");
        check_contains(wr.status().message(), "runtime currently supports 1", "case I message supports");
    }
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto ar = gpu::GpuArena::create(0, 256ull * 1024ull * 1024ull);
    if (!ar.ok()) { fail("arena"); return 1; }
    gpu::GpuArena arena = ar.release();

    std::error_code ec;
    g_root = fs::temp_directory_path(ec) / ("shelf_mtp_weight_load_" + std::to_string(::getpid()));
    fs::create_directories(g_root, ec);

    test_mtp_bf16_contract(arena, stream);
    test_mtp_absent(arena, stream);
    test_mtp_fc_transposed(arena, stream);
    test_mtp_missing_norm(arena, stream);
    test_mtp_wrong_norm_size(arena, stream);
    test_mtp_mlp_transposed(arena, stream);
    test_mtp_attn_wrong_geometry(arena, stream);
    test_mtp_multi_layer_rejected(arena, stream);

    fs::remove_all(g_root, ec);
    arena.shutdown();
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    std::printf("test_qwen35_mtp_weight_load: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
