#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/quantization/manifest/crc32.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/quantization/fp8/fp8.h>
#include <shelf/quantization/fp8/fp8_native.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/models/common/qwen_hybrid/weight_slot.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <span>
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

static void check_eq_u64(uint64_t got, uint64_t want, const std::string& msg) {
    if (got != want) fail(msg + " got=" + std::to_string(got) + " want=" + std::to_string(want));
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

static std::string crc_hex(const std::vector<uint8_t>& bytes) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08x",
                  quantization::manifest::crc32_iso_hdlc(bytes.data(), bytes.size()));
    return buf;
}

static void write_text(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << text;
}

static std::vector<uint8_t> bf16_bytes(uint64_t n, uint32_t seed) {
    std::vector<uint8_t> out(n * 2u);
    auto* u = reinterpret_cast<uint16_t*>(out.data());
    for (uint64_t i = 0; i < n; ++i) {
        u[i] = static_cast<uint16_t>(0x3F80u + ((seed + static_cast<uint32_t>(i)) % 64u));
    }
    return out;
}

static std::vector<uint8_t> f32_bytes(uint64_t n, uint32_t seed) {
    std::vector<uint8_t> out(n * 4u);
    auto* f = reinterpret_cast<float*>(out.data());
    for (uint64_t i = 0; i < n; ++i) {
        f[i] = 0.25f * static_cast<float>((seed + static_cast<uint32_t>(i)) % 251u) - 31.0f;
    }
    return out;
}

static bool bf16_convert_matches(const std::vector<uint8_t>& got, const std::vector<uint8_t>& src) {
    const auto* sf = reinterpret_cast<const float*>(src.data());
    const uint64_t n = src.size() / 4u;
    for (uint64_t i = 0; i < n; ++i) {
        const uint16_t want = f32_to_bf16_rne(sf[i]);
        if (std::memcmp(got.data() + i * 2, &want, 2) != 0) return false;
    }
    return true;
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

static bool write_shard(const std::string& path, std::vector<Planned>& tensors,
                        const std::vector<std::pair<std::string, std::string>>& metadata) {
    auto w = io::SafetensorsWriter::create(path);
    if (!w.ok()) { fail("writer create"); return false; }
    io::SafetensorsWriter writer = w.release();
    for (auto& t : tensors) {
        if (!writer.plan_tensor(t.name, t.dtype, t.shape).ok()) { fail("plan " + t.name); return false; }
    }
    for (auto& kv : metadata) {
        if (!writer.set_metadata(kv.first, kv.second).ok()) { fail("metadata " + kv.first); return false; }
    }
    auto hdr = writer.write_header();
    if (!hdr.ok()) { fail("write_header"); return false; }
    for (auto& t : tensors) {
        if (!writer.write_tensor(t.name, t.bytes.data(), t.bytes.size()).ok()) { fail("write " + t.name); return false; }
    }
    if (!writer.finish().ok()) { fail("finish"); return false; }
    return true;
}

static std::string config_json(uint32_t layers, const std::string& layer_types) {
    return std::string("{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,\"text_config\":{") +
        "\"model_type\":\"qwen3_5_text\",\"vocab_size\":40,\"hidden_size\":32,\"intermediate_size\":32,\"" +
        "num_hidden_layers\":" + std::to_string(layers) + ",\"full_attention_interval\":2,\"layer_types\":[" +
        layer_types + "]}}";
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

struct QTensor {
    std::string physical;
    io::SType dtype;
    std::vector<std::size_t> shape;
    std::vector<uint8_t> bytes;
};

struct QMatrixPlan {
    quantization::manifest::QuantizedTensorMetadata meta;
    std::vector<QTensor> physical;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> metadata1;
};

static std::vector<uint8_t> as_bytes(std::span<const float> v) {
    std::vector<uint8_t> out(v.size_bytes());
    if (!out.empty()) std::memcpy(out.data(), v.data(), out.size());
    return out;
}

static QMatrixPlan make_matrix(quantization::manifest::QuantizedEncoding enc, uint64_t rows,
                               uint64_t k, const std::string& logical) {
    using namespace quantization;
    using manifest::QuantizedEncoding;
    QMatrixPlan plan;
    plan.meta.encoding = enc;
    plan.meta.role = "weight";
    plan.meta.logical_shape = {static_cast<int64_t>(rows), static_cast<int64_t>(k)};

    if (enc == manifest::QuantizedEncoding::Bf16) {
        plan.meta.k_padded = k;
        QTensor data;
        data.physical = logical + ".__shelf_data";
        data.dtype = io::SType::BF16;
        data.shape = {static_cast<size_t>(rows), static_cast<size_t>(k)};
        data.bytes = bf16_bytes(rows * k, static_cast<uint32_t>(rows * 7 + k));
        manifest::QuantizedTensorRef ref;
        ref.tensor = data.physical;
        ref.dtype = "BF16";
        ref.crc32 = crc_hex(data.bytes);
        plan.meta.data = ref;
        plan.codes = data.bytes;
        plan.physical.push_back(std::move(data));
        return plan;
    }

    if (enc == manifest::QuantizedEncoding::Fp8) {
        const uint64_t fkp = (k + 127u) / 128u * 128u;
        plan.meta.k_padded = fkp;
        fp8::Fp8BlockCanonicalStore fstore;
        fstore.init(1, rows, k, fkp);
        std::mt19937 rng(static_cast<uint32_t>(rows * 131 + k + 5));
        std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
        std::vector<float> src(rows * k);
        for (auto& v : src) v = dist(rng);
        if (!fstore.quantize_matrix(0, src.data(), k)) fail("fp8 quantize_matrix " + logical);
        if (!fstore.view().validate()) fail("fp8 store invalid " + logical);
        plan.codes = fstore.codes;
        plan.metadata1 = as_bytes(fstore.scales);

        QTensor ct;
        ct.physical = logical + ".__shelf_codes";
        ct.dtype = io::SType::U8;
        ct.shape = {static_cast<std::size_t>(rows), static_cast<std::size_t>(fkp)};
        ct.bytes = fstore.codes;
        manifest::QuantizedTensorRef cref;
        cref.tensor = ct.physical;
        cref.dtype = "U8";
        cref.crc32 = crc_hex(ct.bytes);
        plan.meta.codes = cref;
        plan.physical.push_back(std::move(ct));

        QTensor st;
        st.physical = logical + ".__shelf_metadata1";
        st.dtype = io::SType::F32;
        st.shape = {static_cast<std::size_t>((rows + 127u) / 128u),
                    static_cast<std::size_t>(fkp / 128u)};
        st.bytes = plan.metadata1;
        manifest::QuantizedTensorRef sref;
        sref.tensor = st.physical;
        sref.dtype = "F32";
        sref.crc32 = crc_hex(st.bytes);
        plan.meta.metadata1 = sref;
        plan.physical.push_back(std::move(st));
        return plan;
    }

    fail("unsupported matrix encoding " + logical);
    return plan;
}

static quantization::fp8::Fp8NativeHost expect_native_fp8(
    const std::vector<uint8_t>& codes, const std::vector<uint8_t>& scales_bytes,
    uint64_t rows, uint64_t kp) {
    quantization::fp8::Fp8BlockCanonicalView cv;
    cv.batch = 1;
    cv.n = rows;
    cv.padded_k = kp;
    cv.codes = codes.data();
    cv.codes_bytes = codes.size();
    cv.scales = reinterpret_cast<const float*>(scales_bytes.data());
    cv.scales_bytes = scales_bytes.size();
    cv.scale_n = (rows + 127u) / 128u;
    cv.scale_k = kp / 128u;
    quantization::fp8::Fp8NativeHost native;
    if (!quantization::fp8::preshuffle_fp8_native(cv, native)) fail("expect_native_fp8");
    return native;
}

static QTensor make_small(uint64_t n, const std::string& logical, std::vector<std::size_t> shape) {
    QTensor t;
    t.physical = logical + ".__shelf_data";
    t.dtype = io::SType::BF16;
    t.shape = std::move(shape);
    t.bytes = bf16_bytes(n, static_cast<uint32_t>(n * 3 + 11));
    return t;
}

static quantization::manifest::QuantizedTensorMetadata small_meta(const QTensor& t, const std::string& role) {
    using namespace quantization::manifest;
    QuantizedTensorMetadata m;
    m.role = role;
    m.encoding = QuantizedEncoding::Bf16;
    for (auto d : t.shape) m.logical_shape.push_back(static_cast<int64_t>(d));
    m.k_padded = static_cast<uint64_t>(t.shape.back());
    QuantizedTensorRef ref;
    ref.tensor = t.physical;
    ref.dtype = "BF16";
    ref.crc32 = crc_hex(t.bytes);
    m.data = ref;
    return m;
}

static std::size_t tensor_span_bytes(const gpu::Tensor& t) {
    std::size_t span = 0;
    if (!t.required_span_bytes(span).ok()) {
        fail("required_span_bytes");
        return 0;
    }
    return span;
}

static void check_matrix_bytes(const weights::MatrixWeight& w, std::span<const uint8_t> want_codes,
                               std::span<const uint8_t> want_scales, const std::string& tag) {
    check_eq_u64(want_codes.size(), tensor_span_bytes(w.codes), tag + " codes size");
    if (!want_codes.empty()) {
        const auto got = d2h(w.codes.data<void>(), want_codes.size());
        check(std::equal(got.begin(), got.end(), want_codes.begin(), want_codes.end()),
              tag + " codes bytes");
    }
    check_eq_u64(want_scales.size(), tensor_span_bytes(w.scales), tag + " scale size");
    if (!want_scales.empty()) {
        const auto got = d2h(w.scales.data<void>(), want_scales.size());
        check(std::equal(got.begin(), got.end(), want_scales.begin(), want_scales.end()),
              tag + " scales bytes");
    }
}

static void test_bf16_matrix_load(gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("bf16_single");
    std::vector<Planned> t;
    Planned mat;
    mat.name = "x.weight";
    mat.dtype = io::SType::BF16;
    mat.shape = {32, 32};
    mat.bytes = bf16_bytes(32 * 32, 55);
    t.push_back(std::move(mat));
    write_index(dir.string(), "model.safetensors-00001-of-00001.safetensors", t);
    if (!write_shard((dir / "model.safetensors-00001-of-00001.safetensors").string(), t, {})) return;

    auto cr = weights::SafetensorsCollection::open(dir.string());
    if (!cr.ok()) { fail("collection open"); return; }
    weights::SafetensorsCollection coll = cr.release();

    check(!weights::is_quantized_model_dir(dir.string()), "bf16 dir not quantized");
    check(!coll.tensor_spec("missing.weight").ok(), "missing spec");

    weights::WeightLoadOptions raw_options;
    raw_options.preshuffle = false;
    auto raw = weights::load_bf16_matrix(coll, "x.weight", arena, stream, raw_options);
    if (!raw.ok()) { fail("raw load"); return; }
    weights::MatrixWeight m = raw.release();
    check_eq_u64(m.rows, 32, "raw rows");
    check_eq_u64(m.cols, 32, "raw cols");
    check_eq_u64(m.k_padded, 32, "raw k_padded");
    check(!m.preshuffled, "raw not preshuffled");
    check(m.quant_spec == quantization::QuantSpecId::BF16_V1, "raw quant_spec");
    check(m.compute_spec == quantization::ComputeSpecId::BF16_F32_BF16, "raw compute_spec");
    {
        const auto got = d2h(m.bf16.data<void>(), t[0].bytes.size());
        check(got == t[0].bytes, "raw bytes");
    }

    auto pre = weights::load_bf16_matrix(coll, "x.weight", arena, stream,
                                         weights::WeightLoadOptions{});
    if (!pre.ok()) { fail("preshuffle load"); return; }
    weights::MatrixWeight p = pre.release();
    check(!p.preshuffled, "bf16 never preshuffled");
    const auto got = d2h(p.bf16.data<void>(), t[0].bytes.size());
    check(got == t[0].bytes, "bf16 bytes");
}

static bool message_has(const Status& st, const std::string& needle) {
    return std::string(st.message()).find(needle) != std::string::npos;
}

// A checkpoint whose payload is not bf16 has to fail at load. The bf16 path reads the
// payload as 2-byte elements, so an F32 matrix would write twice the requested bytes past
// the tensor and only fail later, inside a kernel.
static void test_bf16_payload_dtype_guard(gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("bf16_payload_dtype");
    std::vector<Planned> t;
    const auto add = [&](const std::string& name, io::SType dt, std::vector<std::size_t> shape,
                         uint32_t seed) {
        Planned p;
        p.name = name;
        p.dtype = dt;
        p.shape = std::move(shape);
        const uint64_t n = numel(p.shape);
        p.bytes = (dt == io::SType::F32) ? f32_bytes(n, seed) : bf16_bytes(n, seed);
        t.push_back(std::move(p));
    };
    add("bf16.weight", io::SType::BF16, {16, 16}, 1);
    add("f32.weight", io::SType::F32, {16, 16}, 2);
    add("f16.weight", io::SType::F16, {16}, 3);
    add("f32_vec.weight", io::SType::F32, {16}, 4);
    if (!write_shard((dir / "model.safetensors").string(), t, {})) return;
    write_index(dir.string(), "model.safetensors", t);

    auto cr = weights::SafetensorsCollection::open(dir.string());
    if (!cr.ok()) {
        fail("payload dtype guard: collection open");
        return;
    }
    weights::SafetensorsCollection coll = cr.release();

    auto matrix = weights::load_bf16_matrix(coll, "f32.weight", arena, stream,
                                           weights::WeightLoadOptions{});
    check(!matrix.ok(), "an F32 matrix is not read by the bf16 matrix path");
    if (!matrix.ok()) {
        check(message_has(matrix.status(), "must be BF16"),
              "the matrix guard states the payload rule");
        check(message_has(matrix.status(), "f32.weight is F32"),
              "the matrix guard names the tensor and its dtype");
    }

    auto f16_norm = weights::load_bf16_or_f32_tensor(coll, "f16.weight", arena, stream);
    check(!f16_norm.ok() && message_has(f16_norm.status(), "f16.weight is F16"),
          "an F16 vector is rejected instead of read as bf16");

    auto f16_vector = weights::load_bf16_tensor(coll, "f16.weight", arena, stream);
    check(!f16_vector.ok() && message_has(f16_vector.status(), "f16.weight is F16"),
          "the vector path guards the same way");

    auto f32_vector = weights::load_bf16_or_f32_tensor(coll, "f32_vec.weight", arena, stream);
    check(f32_vector.ok(), "an F32 vector keeps being converted, which the path supports");

    auto bf16 = weights::load_bf16_matrix(coll, "bf16.weight", arena, stream,
                                         weights::WeightLoadOptions{});
    check(bf16.ok(), "a bf16 payload keeps loading");
}

// lm_head_name lets a case place the output head the way a HF checkpoint does: the
// transformer tensors carry the model.language_model. prefix, lm_head does not.
static bool build_bf16_model(const fs::path& dir,
                             const char* lm_head_name = "model.language_model.lm_head.weight") {
    std::vector<Planned> t;
    const auto add = [&](const std::string& name, io::SType dt, std::vector<std::size_t> shape, uint32_t seed) {
        Planned p;
        p.name = name;
        p.dtype = dt;
        p.shape = std::move(shape);
        const uint64_t n = numel(p.shape);
        p.bytes = (dt == io::SType::F32) ? f32_bytes(n, seed) : bf16_bytes(n, seed);
        t.push_back(std::move(p));
    };

    add("model.language_model.embed_tokens.weight", io::SType::BF16, {40, 32}, 1);
    add(lm_head_name, io::SType::BF16, {40, 32}, 2);
    add("model.language_model.norm.weight", io::SType::BF16, {32}, 3);

    add("model.language_model.layers.0.input_layernorm.weight", io::SType::F32, {32}, 4);
    add("model.language_model.layers.0.post_attention_layernorm.weight", io::SType::BF16, {32}, 5);
    add("model.language_model.layers.0.mlp.gate_proj.weight", io::SType::BF16, {32, 32}, 6);
    add("model.language_model.layers.0.mlp.up_proj.weight", io::SType::BF16, {32, 32}, 7);
    add("model.language_model.layers.0.mlp.down_proj.weight", io::SType::BF16, {32, 32}, 8);
    add("model.language_model.layers.0.linear_attn.in_proj_a.weight", io::SType::BF16, {8, 32}, 9);
    add("model.language_model.layers.0.linear_attn.in_proj_b.weight", io::SType::BF16, {8, 32}, 10);
    add("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", io::SType::BF16, {24, 32}, 11);
    add("model.language_model.layers.0.linear_attn.in_proj_z.weight", io::SType::BF16, {8, 32}, 12);
    add("model.language_model.layers.0.linear_attn.conv1d.weight", io::SType::F32, {8, 1, 4}, 13);
    add("model.language_model.layers.0.linear_attn.out_proj.weight", io::SType::BF16, {32, 32}, 14);
    add("model.language_model.layers.0.linear_attn.norm.weight", io::SType::F32, {32}, 15);
    add("model.language_model.layers.0.linear_attn.dt_bias", io::SType::BF16, {8}, 16);
    add("model.language_model.layers.0.linear_attn.A_log", io::SType::F32, {8}, 17);

    add("model.language_model.layers.1.input_layernorm.weight", io::SType::BF16, {32}, 18);
    add("model.language_model.layers.1.post_attention_layernorm.weight", io::SType::BF16, {32}, 19);
    add("model.language_model.layers.1.mlp.gate_proj.weight", io::SType::BF16, {32, 32}, 20);
    add("model.language_model.layers.1.mlp.up_proj.weight", io::SType::BF16, {32, 32}, 21);
    add("model.language_model.layers.1.mlp.down_proj.weight", io::SType::BF16, {32, 32}, 22);
    add("model.language_model.layers.1.self_attn.q_proj.weight", io::SType::BF16, {32, 32}, 23);
    add("model.language_model.layers.1.self_attn.k_proj.weight", io::SType::BF16, {8, 24}, 24);
    add("model.language_model.layers.1.self_attn.v_proj.weight", io::SType::BF16, {8, 24}, 25);
    add("model.language_model.layers.1.self_attn.o_proj.weight", io::SType::BF16, {32, 32}, 26);
    add("model.language_model.layers.1.self_attn.q_norm.weight", io::SType::BF16, {32}, 27);
    add("model.language_model.layers.1.self_attn.k_norm.weight", io::SType::BF16, {32}, 28);

    add("mtp.pre_fc_norm_embedding.weight", io::SType::BF16, {32}, 29);
    add("mtp.pre_fc_norm_hidden.weight", io::SType::BF16, {32}, 30);
    add("mtp.norm.weight", io::SType::BF16, {32}, 31);
    add("mtp.fc.weight", io::SType::BF16, {32, 64}, 32);
    add("mtp.layers.0.input_layernorm.weight", io::SType::BF16, {32}, 33);
    add("mtp.layers.0.post_attention_layernorm.weight", io::SType::BF16, {32}, 34);
    add("mtp.layers.0.mlp.gate_proj.weight", io::SType::BF16, {32, 32}, 35);
    add("mtp.layers.0.mlp.up_proj.weight", io::SType::BF16, {32, 32}, 36);
    add("mtp.layers.0.mlp.down_proj.weight", io::SType::BF16, {32, 32}, 37);
    add("mtp.layers.0.self_attn.q_proj.weight", io::SType::BF16, {32, 32}, 38);
    add("mtp.layers.0.self_attn.k_proj.weight", io::SType::BF16, {8, 32}, 39);
    add("mtp.layers.0.self_attn.v_proj.weight", io::SType::BF16, {8, 32}, 40);
    add("mtp.layers.0.self_attn.o_proj.weight", io::SType::BF16, {32, 32}, 41);
    add("mtp.layers.0.self_attn.q_norm.weight", io::SType::BF16, {32}, 42);
    add("mtp.layers.0.self_attn.k_norm.weight", io::SType::BF16, {32}, 43);

    if (!write_shard((dir / "model.safetensors-00001-of-00001.safetensors").string(), t, {})) {
        return false;
    }
    write_index(dir.string(), "model.safetensors-00001-of-00001.safetensors", t);
    write_text((dir / "config.json").string(), config_json(2, "\"linear_attention\",\"full_attention\""));
    return true;
}

static void test_bf16_model_load(gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("bf16_model");
    if (!build_bf16_model(dir)) return;

    qwen35::Qwen35LoadOptions options;
    options.load_mtp_layers = true;
    auto wr = qwen35::load_qwen35_weights_from_safetensors(dir.string(), arena, stream, options);
    if (!wr.ok()) { fail("bf16 model load: " + wr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = wr.release();

    check_eq_u64(w.layers.size(), 2, "model layers");
    check(w.layers[0].is_gdn, "layer0 gdn");
    check(!w.layers[1].is_gdn, "layer1 attn");
    check(w.mtp.present, "mtp present");
    check(!w.lm_head_tied, "lm_head untied");

    check_eq_u64(w.embed_tokens.rows, 40, "embed rows");
    check_eq_u64(w.embed_tokens.cols, 32, "embed cols");
    check_eq_u64(w.embed_tokens.k_padded, 32, "embed k_padded");
    check(!w.embed_tokens.preshuffled, "embed not preshuffled");

    check_eq_u64(w.layers[1].attn_q_proj.rows, 32, "q rows");
    check_eq_u64(w.layers[1].attn_k_proj.cols, 24, "k cols");
    check_eq_u64(w.layers[1].attn_k_proj.k_padded, 24, "k k_padded fallback");
    check(!w.layers[1].attn_k_proj.preshuffled, "k not preshuffled");

    {
        const auto got = d2h(w.layers[0].attn_A_log.data<void>(), 8 * 2);
        check(bf16_convert_matches(got, f32_bytes(8, 17)), "A_log f32->bf16 conversion");
    }
    {
        const auto got = d2h(w.layers[0].input_layernorm_weight.data<void>(), 32 * 4);
        check(got == f32_bytes(32, 4), "input_layernorm f32 preserved");
    }
    {
        const auto got = d2h(w.layers[0].attn_norm_weight.data<void>(), 32 * 2);
        check(bf16_convert_matches(got, f32_bytes(32, 15)), "gdn norm f32->bf16 conversion");
    }
    {
        const auto got = d2h(w.layers[0].attn_conv1d_weight.data<void>(), 32 * 2);
        check(bf16_convert_matches(got, f32_bytes(32, 13)), "conv1d f32->bf16 conversion");
        check_eq_u64(w.layers[0].attn_conv1d_weight.ndim(), 3, "conv1d ndim");
    }
    {
        const auto got = d2h(w.final_norm_weight.data<void>(), 64);
        check(got == bf16_bytes(32, 3), "final_norm bytes");
    }
    check(w.mtp.fc.encoding == weights::MatrixEncoding::Bf16, "mtp fc encoding");
    check_eq_u64(w.mtp.fc.rows, 32, "mtp fc rows == H");
    check_eq_u64(w.mtp.fc.cols, 64, "mtp fc cols == 2H");
    check(!w.mtp.layer.is_gdn, "mtp layer attn");

    qwen35::Qwen35LoadOptions no_pre;
    no_pre.load_mtp_layers = true;
    no_pre.weights.preshuffle = false;
    auto wr2 = qwen35::load_qwen35_weights_from_safetensors(dir.string(), arena, stream, no_pre);
    if (!wr2.ok()) { fail("bf16 model load skip_preshuffle"); return; }
    qwen35::Qwen35ModelWeights w2 = wr2.release();
    check(!w2.layers[1].attn_q_proj.preshuffled, "skip_preshuffle q raw");
}

static void test_bf16_model_load_bare_lm_head(gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("bf16_model_bare_head");
    if (!build_bf16_model(dir, "lm_head.weight")) return;

    auto wr = qwen35::load_qwen35_weights_from_safetensors(dir.string(), arena, stream,
                                                           qwen35::Qwen35LoadOptions{});
    if (!wr.ok()) { fail("bf16 model load with bare lm_head.weight: " + wr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = wr.release();

    check_eq_u64(w.lm_head.rows, 40, "bare lm_head rows");
    check_eq_u64(w.lm_head.cols, 32, "bare lm_head cols");
    check(!w.lm_head_tied, "bare lm_head is untied");
    check(w.lm_head.bf16.data() != w.embed_tokens.bf16.data(),
          "bare lm_head is its own storage");

    auto config_res = qwen35::read_qwen35_text_config(dir.string());
    check(config_res.ok(), "bare lm_head config read");
    if (config_res.ok()) {
        auto st = qwen35::validate_qwen35_weights(w, config_res.value());
        check(st.ok(), std::string("bare lm_head model invariants: ") + (st.ok() ? "" : st.message()));
    }
}

static void test_standalone_quantized_reader(gpu::GpuArena& arena, hipStream_t stream) {
    using namespace quantization;
    using manifest::QuantizedEncoding;
    using bundle::QuantizedModelReader;

    const fs::path dir = make_case_dir("quantized_standalone");

    QMatrixPlan m_fp8 = make_matrix(QuantizedEncoding::Fp8, 130, 200, "fp8");
    QMatrixPlan m_bf = make_matrix(QuantizedEncoding::Bf16, 8, 24, "bf");
    QTensor small = make_small(32, "vec", {32});

    std::vector<Planned> shard;
    manifest::QuantizedManifest manifest;
    manifest.architecture = "qwen35_dense";
    manifest.preset = "fp8";
    manifest.source_dtype = "BF16";
    const auto add_matrix = [&](const std::string& logical, QMatrixPlan& plan) {
        for (auto& p : plan.physical) {
            Planned pl;
            pl.name = p.physical;
            pl.dtype = p.dtype;
            pl.shape = p.shape;
            pl.bytes = p.bytes;
            shard.push_back(std::move(pl));
        }
        manifest.tensors[logical] = plan.meta;
    };
    add_matrix("fp8", m_fp8);
    add_matrix("bf", m_bf);
    {
        Planned pl;
        pl.name = small.physical;
        pl.dtype = small.dtype;
        pl.shape = small.shape;
        pl.bytes = small.bytes;
        shard.push_back(std::move(pl));
        manifest.tensors["vec"] = small_meta(small, "norm");
    }
    manifest.aliases["vec_alias"] = "vec";
    manifest.aliases["dangling"] = "nope";

    auto ms = manifest::serialize_quantized_manifest(manifest);
    if (!ms.ok()) { fail("serialize manifest"); return; }
    if (!write_shard((dir / "model.safetensors").string(), shard,
                     {{"shelf.format", manifest::kQuantizedSafetensorsFormat},
                      {"shelf.format_version", std::to_string(manifest::kQuantizedSafetensorsFormatVersion)},
                      {"shelf.quantization_metadata", manifest::kQuantizedMetadataFile}})) {
        return;
    }
    write_text((dir / manifest::kQuantizedMetadataFile).string(), ms.value());

    check(weights::is_quantized_model_dir(dir.string()), "is_quantized_model_dir(standalone)");

    auto rr = QuantizedModelReader::open(dir.string(), true);
    if (!rr.ok()) { fail("reader open: " + rr.status().message()); return; }
    QuantizedModelReader reader = rr.release();

    check(!weights::validate_quantized_model(reader).ok(), "dangling alias must fail validation");
    manifest.aliases.erase("dangling");
    auto ms2 = manifest::serialize_quantized_manifest(manifest);
    if (!ms2.ok()) { fail("serialize manifest 2"); return; }
    write_text((dir / manifest::kQuantizedMetadataFile).string(), ms2.value());
    auto rr2 = QuantizedModelReader::open(dir.string(), true);
    if (!rr2.ok()) { fail("reader reopen: " + rr2.status().message()); return; }
    QuantizedModelReader good = rr2.release();
    check(weights::validate_quantized_model(good).ok(), "validate_quantized_model");

    {
        auto w = weights::load_quantized_matrix(good, "fp8", arena, stream, weights::WeightLoadOptions{});
        if (!w.ok()) { fail("load fp8: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check_eq_u64(static_cast<uint32_t>(m.encoding),
                         static_cast<uint32_t>(weights::MatrixEncoding::Fp8Finegrained), "fp8 encoding");
            check_eq_u64(m.rows, 130, "fp8 rows");
            check_eq_u64(m.cols, 200, "fp8 cols");
            check_eq_u64(m.k_padded, 256, "fp8 k_padded");
            check_eq_u64(m.weight_scale_group, 128, "fp8 wsg");
            check(m.preshuffled, "fp8 preshuffled");
            check(m.quant_spec == QuantSpecId::FP8_FINEGRAINED_F32_V1, "fp8 quant_spec");
            check(m.compute_spec == ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16, "fp8 compute_spec");
            const auto n = expect_native_fp8(m_fp8.codes, m_fp8.metadata1, 130, 256);
            check_matrix_bytes(m, n.codes_span(), as_bytes(n.scales_span()), "fp8");
        }
    }
    {
        weights::WeightLoadOptions no_pre;
        no_pre.preshuffle = false;
        auto w = weights::load_quantized_matrix(good, "fp8", arena, stream, no_pre);
        if (!w.ok()) { fail("load fp8 canonical: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check(!m.preshuffled, "fp8 canonical not preshuffled");
            check_eq_u64(m.codes_row_stride_bytes, 256, "fp8 canonical codes stride");
            check_eq_u64(m.storage_scale_stride_bytes, 8, "fp8 canonical scale stride");
            check_matrix_bytes(m, m_fp8.codes, m_fp8.metadata1, "fp8 canonical");
        }
    }
    {
        auto w = weights::load_quantized_matrix(good, "bf", arena, stream, weights::WeightLoadOptions{});
        if (!w.ok()) { fail("load bf: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check_eq_u64(static_cast<uint32_t>(m.encoding), static_cast<uint32_t>(weights::MatrixEncoding::Bf16), "bf encoding");
            check_eq_u64(m.rows, 8, "bf rows");
            check_eq_u64(m.cols, 24, "bf cols");
            check_eq_u64(m.k_padded, 24, "bf k_padded");
            check(!m.preshuffled, "bf not preshuffled");
            check(m.quant_spec == QuantSpecId::BF16_V1, "bf quant_spec");
            const auto got = d2h(m.bf16.data<void>(), m_bf.codes.size());
            check(got == m_bf.codes, "bf bytes");
        }
    }
    {
        auto tv = weights::load_quantized_small(good, "vec", arena, stream);
        if (!tv.ok()) { fail("load small vec: " + tv.status().message()); }
        else {
            const gpu::Tensor v = tv.release();
            check_eq_u64(v.ndim(), 1, "vec ndim");
            const auto got = d2h(v.data<void>(), small.bytes.size());
            check(got == small.bytes, "vec bytes");
        }
        auto ta = weights::load_quantized_small(good, "vec_alias", arena, stream);
        check(ta.ok(), "vec alias resolve");
        auto miss = weights::load_quantized_small(good, "nope", arena, stream);
        check(!miss.ok(), "missing tensor must fail");
    }
}

static void test_qwen35_quantized_model_load(gpu::GpuArena& arena, hipStream_t stream) {
    using namespace quantization;
    using manifest::QuantizedEncoding;
    const fs::path dir = make_case_dir("quantized_model");

    std::vector<Planned> shard;
    manifest::QuantizedManifest manifest;
    manifest.architecture = "qwen35_dense";
    manifest.preset = "fp8";
    manifest.source_dtype = "BF16";

    const auto add_matrix = [&](const std::string& logical, QMatrixPlan plan) {
        for (auto& p : plan.physical) {
            Planned pl;
            pl.name = p.physical;
            pl.dtype = p.dtype;
            pl.shape = p.shape;
            pl.bytes = p.bytes;
            shard.push_back(std::move(pl));
        }
        manifest.tensors[logical] = plan.meta;
    };
    const auto add_small = [&](const std::string& logical, std::vector<std::size_t> shape) {
        const uint64_t n = numel(shape);
        QTensor t = make_small(n, logical, shape);
        Planned pl;
        pl.name = t.physical;
        pl.dtype = t.dtype;
        pl.shape = t.shape;
        pl.bytes = t.bytes;
        shard.push_back(std::move(pl));
        manifest.tensors[logical] = small_meta(t, "norm");
    };

    const auto L = [](const char* s) { return std::string("model.language_model.") + s; };

    add_small(L("layers.0.input_layernorm.weight"), {32});
    add_small(L("layers.0.post_attention_layernorm.weight"), {32});
    add_matrix(L("layers.0.mlp.gate_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 32, 32, "L0_g"));
    add_matrix(L("layers.0.mlp.up_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 32, 32, "L0_u"));
    add_matrix(L("layers.0.mlp.down_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 32, 32, "L0_d"));
    add_matrix(L("layers.0.linear_attn.in_proj_a.weight"), make_matrix(QuantizedEncoding::Bf16, 8, 32, "L0_ia"));
    add_matrix(L("layers.0.linear_attn.in_proj_b.weight"), make_matrix(QuantizedEncoding::Bf16, 8, 32, "L0_ib"));
    add_matrix(L("layers.0.linear_attn.in_proj_qkv.weight"), make_matrix(QuantizedEncoding::Fp8, 24, 32, "L0_iqkv"));
    add_matrix(L("layers.0.linear_attn.in_proj_z.weight"), make_matrix(QuantizedEncoding::Bf16, 8, 32, "L0_iz"));
    add_small(L("layers.0.linear_attn.conv1d.weight"), {8, 1, 4});
    add_matrix(L("layers.0.linear_attn.out_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 32, 32, "L0_op"));
    add_small(L("layers.0.linear_attn.norm.weight"), {32});
    add_small(L("layers.0.linear_attn.dt_bias"), {8});
    add_small(L("layers.0.linear_attn.A_log"), {8});

    add_small(L("layers.1.input_layernorm.weight"), {32});
    add_small(L("layers.1.post_attention_layernorm.weight"), {32});
    add_matrix(L("layers.1.mlp.gate_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 32, 32, "L1_g"));
    add_matrix(L("layers.1.mlp.up_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 32, 32, "L1_u"));
    add_matrix(L("layers.1.mlp.down_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 32, 32, "L1_d"));
    add_matrix(L("layers.1.self_attn.q_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 32, 32, "L1_q"));
    add_matrix(L("layers.1.self_attn.k_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 8, 32, "L1_k"));
    add_matrix(L("layers.1.self_attn.v_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 8, 32, "L1_v"));
    add_matrix(L("layers.1.self_attn.o_proj.weight"), make_matrix(QuantizedEncoding::Fp8, 32, 32, "L1_o"));
    add_small(L("layers.1.self_attn.q_norm.weight"), {32});
    add_small(L("layers.1.self_attn.k_norm.weight"), {32});

    add_matrix(L("embed_tokens.weight"), make_matrix(QuantizedEncoding::Bf16, 40, 32, "embed"));
    add_matrix(L("lm_head.weight"), make_matrix(QuantizedEncoding::Bf16, 40, 32, "lmhead"));
    add_small(L("norm.weight"), {32});

    add_small("mtp.pre_fc_norm_embedding.weight", {32});
    add_small("mtp.pre_fc_norm_hidden.weight", {32});
    add_small("mtp.norm.weight", {32});
    add_matrix("mtp.fc.weight", make_matrix(QuantizedEncoding::Bf16, 32, 64, "mtp_fc"));
    add_small("mtp.layers.0.input_layernorm.weight", {32});
    add_small("mtp.layers.0.post_attention_layernorm.weight", {32});
    add_matrix("mtp.layers.0.mlp.gate_proj.weight", make_matrix(QuantizedEncoding::Fp8, 32, 32, "mtp_e_g"));
    add_matrix("mtp.layers.0.mlp.up_proj.weight", make_matrix(QuantizedEncoding::Fp8, 32, 32, "mtp_e_u"));
    add_matrix("mtp.layers.0.mlp.down_proj.weight", make_matrix(QuantizedEncoding::Fp8, 32, 32, "mtp_e_d"));
    add_matrix("mtp.layers.0.self_attn.q_proj.weight", make_matrix(QuantizedEncoding::Fp8, 32, 32, "mtp_e_q"));
    add_matrix("mtp.layers.0.self_attn.k_proj.weight", make_matrix(QuantizedEncoding::Fp8, 8, 32, "mtp_e_k"));
    add_matrix("mtp.layers.0.self_attn.v_proj.weight", make_matrix(QuantizedEncoding::Fp8, 8, 32, "mtp_e_v"));
    add_matrix("mtp.layers.0.self_attn.o_proj.weight", make_matrix(QuantizedEncoding::Fp8, 32, 32, "mtp_e_o"));
    add_small("mtp.layers.0.self_attn.q_norm.weight", {32});
    add_small("mtp.layers.0.self_attn.k_norm.weight", {32});

    auto ms = manifest::serialize_quantized_manifest(manifest);
    if (!ms.ok()) { fail("serialize model manifest"); return; }
    if (!write_shard((dir / "model.safetensors").string(), shard,
                     {{"shelf.format", manifest::kQuantizedSafetensorsFormat},
                      {"shelf.format_version", std::to_string(manifest::kQuantizedSafetensorsFormatVersion)},
                      {"shelf.quantization_metadata", manifest::kQuantizedMetadataFile}})) {
        return;
    }
    write_text((dir / manifest::kQuantizedMetadataFile).string(), ms.value());
    write_text((dir / "config.json").string(), config_json(2, "\"linear_attention\",\"full_attention\""));

    check(weights::is_quantized_model_dir(dir.string()), "is_quantized_model_dir(model)");

    qwen35::Qwen35LoadOptions options;
    options.load_mtp_layers = true;
    options.verify_quantized_payload_crc = true;
    auto rr = qwen35::load_qwen35_weights_from_safetensors(dir.string(), arena, stream, options);
    if (!rr.ok()) { fail("qwen35 quantized load: " + rr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = rr.release();

    check_eq_u64(w.layers.size(), 2, "quant model layers");
    check(w.layers[0].is_gdn, "quant layer0 gdn");
    check(!w.layers[1].is_gdn, "quant layer1 attn");
    check(w.mtp.present, "quant mtp");
    check(!w.lm_head_tied, "quant lm_head untied");

    check_eq_u64(w.embed_tokens.rows, 40, "quant embed rows");
    check_eq_u64(w.embed_tokens.cols, 32, "quant embed cols");
    check(w.embed_tokens.encoding == weights::MatrixEncoding::Bf16, "quant embed encoding");

    check_eq_u64(static_cast<uint32_t>(w.layers[0].ffn.gate.encoding),
                 static_cast<uint32_t>(weights::MatrixEncoding::Fp8Finegrained), "quant gate encoding");
    check(w.layers[0].ffn.gate.preshuffled, "quant gate preshuffled");
    check(w.layers[0].ffn.down.compute_spec ==
              quantization::ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16,
          "quant down compute_spec");
    check(w.layers[0].attn_in_proj_a.encoding == weights::MatrixEncoding::Bf16, "in_proj_a bf16 matrix");
    check_eq_u64(static_cast<uint32_t>(w.layers[1].attn_q_proj.encoding),
                 static_cast<uint32_t>(weights::MatrixEncoding::Fp8Finegrained), "quant q encoding");
    check(w.mtp.fc.encoding == weights::MatrixEncoding::Bf16, "mtp fc encoding bf16 (MTP not quantized)");
    check_eq_u64(w.mtp.fc.rows, 32, "mtp fc rows == H");
    check_eq_u64(w.mtp.fc.cols, 64, "mtp fc cols == 2H");

    {
        weights::TensorPartitionPlan guard_plan;
        guard_plan.tp_size = 2;
        guard_plan.tp_rank = 0;
        qwen35::Qwen35LoadOptions guard_options;
        guard_options.weights.partition_plan = &guard_plan;
        auto blocked = qwen35::load_qwen35_weights_from_safetensors(
            dir.string(), arena, stream, guard_options);
        check(!blocked.ok() && blocked.status().code() == Status::Code::unsupported,
              "qwen35 load path rejects a tensor partition plan");
    }

    {
        auto rdr = bundle::QuantizedModelReader::open(dir.string(), false);
        if (!rdr.ok()) { fail("reopen reader"); return; }
        auto view = rdr.value().resolve("model.language_model.layers.1.self_attn.q_proj.weight");
        if (!view.ok()) { fail("resolve q"); return; }
        const auto& qv = view.value();
        const std::vector<uint8_t> codes_src(static_cast<const uint8_t*>(qv.codes.data),
                                            static_cast<const uint8_t*>(qv.codes.data) + qv.codes.size);
        const std::vector<uint8_t> metadata1_src(static_cast<const uint8_t*>(qv.metadata1.data),
                                          static_cast<const uint8_t*>(qv.metadata1.data) + qv.metadata1.size);
        const auto n = expect_native_fp8(codes_src, metadata1_src, 32, 128);
        check_matrix_bytes(w.layers[1].attn_q_proj, n.codes_span(), as_bytes(n.scales_span()),
                           "quant layer1 q bytes");

        auto norm_view = rdr.value().resolve(L("norm.weight"));
        if (!norm_view.ok()) { fail("resolve final norm"); return; }
        const auto& nv = norm_view.value();
        const std::vector<uint8_t> norm_src(static_cast<const uint8_t*>(nv.data.data),
                                            static_cast<const uint8_t*>(nv.data.data) + nv.data.size);
        const auto got = d2h(w.final_norm_weight.data<void>(), norm_src.size());
        check(got == norm_src, "final norm bytes");
    }
}

struct HfFxpFixture {
    std::map<std::string, std::vector<uint8_t>> codes;
    std::map<std::string, std::vector<float>> scales;
    std::map<std::string, std::vector<uint8_t>> bf16;
};

static std::vector<uint8_t> bf16_bytes_from_f32(const std::vector<float>& values) {
    std::vector<uint8_t> out(values.size() * 2u);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const uint16_t h = shelf::f32_to_bf16_rne(values[i]);
        std::memcpy(out.data() + i * 2, &h, 2);
    }
    return out;
}

static float bf16_bits_to_f32(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static std::vector<float> bf16_rounded_scales(uint64_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.01f, 0.5f);
    std::vector<float> out(count);
    for (uint64_t i = 0; i < count; ++i) {
        out[i] = bf16_bits_to_f32(shelf::f32_to_bf16_rne(dist(rng)));
    }
    return out;
}

static void hf_add_fp8_matrix(HfFxpFixture& f, std::vector<Planned>& shard,
                              const std::string& logical, uint64_t rows, uint64_t k,
                              uint32_t seed) {
    const uint64_t kp = (k + 127u) / 128u * 128u;
    std::vector<uint8_t> codes(rows * kp);
    std::mt19937 rng(seed);
    for (uint8_t& b : codes) b = static_cast<uint8_t>(rng() & 0x7Fu);
    const uint64_t scale_n = (rows + 127u) / 128u;
    const uint64_t scale_k = kp / 128u;
    std::vector<float> scales = bf16_rounded_scales(scale_n * scale_k, seed + 1u);

    Planned cw;
    cw.name = logical;
    cw.dtype = io::SType::F8_E4M3;
    cw.shape = {static_cast<std::size_t>(rows), static_cast<std::size_t>(kp)};
    cw.bytes = codes;
    shard.push_back(std::move(cw));
    Planned sw;
    sw.name = logical + "_scale_inv";
    sw.dtype = io::SType::BF16;
    sw.shape = {static_cast<std::size_t>(scale_n), static_cast<std::size_t>(scale_k)};
    sw.bytes = bf16_bytes_from_f32(scales);
    shard.push_back(std::move(sw));

    f.codes[logical] = std::move(codes);
    f.scales[logical] = std::move(scales);
}

static void hf_add_bf16(HfFxpFixture& f, std::vector<Planned>& shard,
                        const std::string& logical, std::vector<std::size_t> shape,
                        uint32_t seed) {
    std::vector<uint8_t> bytes;
    if (!shape.empty()) {
        uint64_t n = 1;
        for (auto d : shape) n *= d;
        bytes = bf16_bytes(n, seed);
    }
    Planned p;
    p.name = logical;
    p.dtype = io::SType::BF16;
    p.shape = shape;
    p.bytes = bytes;
    shard.push_back(std::move(p));
    f.bf16[logical] = std::move(bytes);
}

static void write_index_multi(const std::string& dir,
                              const std::vector<Planned>& a, const std::string& shard_a,
                              const std::vector<Planned>& b, const std::string& shard_b) {
    std::string j = "{\"weight_map\":{";
    bool first = true;
    const auto append = [&](const std::vector<Planned>& tensors, const std::string& shard) {
        for (const auto& t : tensors) {
            if (!first) j += ",";
            first = false;
            j += "\"" + t.name + "\":\"" + shard + "\"";
        }
    };
    append(a, shard_a);
    append(b, shard_b);
    j += "},\"metadata\":{\"total_size\":0}}";
    write_text(dir + "/model.safetensors.index.json", j);
}

static std::string hf_config_json(const std::string& quant_config) {
    return std::string("{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,") +
        "\"text_config\":{\"model_type\":\"qwen3_5_text\",\"vocab_size\":40,\"hidden_size\":128," +
        "\"intermediate_size\":128,\"num_hidden_layers\":2,\"full_attention_interval\":2," +
        "\"layer_types\":[\"linear_attention\",\"full_attention\"]}," +
        "\"quantization_config\":" + quant_config + "}";
}

static std::string hf_fp8_quant_config() {
    return std::string("{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"activation_scheme\":\"dynamic\",") +
           "\"weight_block_size\":[128,128]}";
}

static const std::string& hf_layer_prefix(std::size_t layer) {
    static const std::vector<std::string> prefixes = {
        "model.language_model.layers.0.",
        "model.language_model.layers.1.",
    };
    return prefixes[layer];
}

static void hf_add_gdn_layer(HfFxpFixture& f,
                             std::vector<Planned>& shard, std::size_t layer) {
    const std::string p = hf_layer_prefix(layer);
    hf_add_bf16(f, shard, p + "input_layernorm.weight", {128}, static_cast<uint32_t>(layer * 7 + 1));
    hf_add_bf16(f, shard, p + "post_attention_layernorm.weight", {128}, static_cast<uint32_t>(layer * 7 + 2));
    const auto add_matrix = [&](const std::string& name, uint64_t rows, uint64_t k, uint32_t seed) {
        hf_add_fp8_matrix(f, shard, p + name, rows, k, seed);
    };
    add_matrix("mlp.gate_proj.weight", 128, 128, static_cast<uint32_t>(layer * 101 + 11));
    add_matrix("mlp.up_proj.weight", 128, 128, static_cast<uint32_t>(layer * 101 + 13));
    add_matrix("mlp.down_proj.weight", 128, 128, static_cast<uint32_t>(layer * 101 + 17));
    hf_add_bf16(f, shard, p + "linear_attn.in_proj_a.weight", {48, 128}, static_cast<uint32_t>(layer * 7 + 3));
    hf_add_bf16(f, shard, p + "linear_attn.in_proj_b.weight", {48, 128}, static_cast<uint32_t>(layer * 7 + 4));
    add_matrix("linear_attn.in_proj_qkv.weight", 24, 128, static_cast<uint32_t>(layer * 101 + 19));
    add_matrix("linear_attn.in_proj_z.weight", 8, 128, static_cast<uint32_t>(layer * 101 + 23));
    hf_add_bf16(f, shard, p + "linear_attn.conv1d.weight", {8, 1, 4}, static_cast<uint32_t>(layer * 7 + 5));
    add_matrix("linear_attn.out_proj.weight", 128, 128, static_cast<uint32_t>(layer * 101 + 29));
    hf_add_bf16(f, shard, p + "linear_attn.norm.weight", {128}, static_cast<uint32_t>(layer * 7 + 6));
    hf_add_bf16(f, shard, p + "linear_attn.dt_bias", {8}, static_cast<uint32_t>(layer * 7 + 7));
    hf_add_bf16(f, shard, p + "linear_attn.A_log", {8}, static_cast<uint32_t>(layer * 7 + 8));
}

static void hf_add_attn_layer(HfFxpFixture& f,
                              std::vector<Planned>& shard, std::size_t layer) {
    const std::string p = hf_layer_prefix(layer);
    hf_add_bf16(f, shard, p + "input_layernorm.weight", {128}, static_cast<uint32_t>(layer * 7 + 1));
    hf_add_bf16(f, shard, p + "post_attention_layernorm.weight", {128}, static_cast<uint32_t>(layer * 7 + 2));
    const auto add_matrix = [&](const std::string& name, uint64_t rows, uint64_t k, uint32_t seed) {
        hf_add_fp8_matrix(f, shard, p + name, rows, k, seed);
    };
    add_matrix("mlp.gate_proj.weight", 128, 128, static_cast<uint32_t>(layer * 101 + 31));
    add_matrix("mlp.up_proj.weight", 128, 128, static_cast<uint32_t>(layer * 101 + 37));
    add_matrix("mlp.down_proj.weight", 128, 128, static_cast<uint32_t>(layer * 101 + 41));
    add_matrix("self_attn.q_proj.weight", 128, 128, static_cast<uint32_t>(layer * 101 + 43));
    add_matrix("self_attn.k_proj.weight", 32, 128, static_cast<uint32_t>(layer * 101 + 47));
    add_matrix("self_attn.v_proj.weight", 32, 128, static_cast<uint32_t>(layer * 101 + 53));
    add_matrix("self_attn.o_proj.weight", 128, 128, static_cast<uint32_t>(layer * 101 + 59));
    hf_add_bf16(f, shard, p + "self_attn.q_norm.weight", {32}, static_cast<uint32_t>(layer * 7 + 9));
    hf_add_bf16(f, shard, p + "self_attn.k_norm.weight", {32}, static_cast<uint32_t>(layer * 7 + 10));
}

static void hf_add_mtp(HfFxpFixture& f, std::vector<Planned>& shard) {
    hf_add_bf16(f, shard, "mtp.pre_fc_norm_embedding.weight", {128}, 71);
    hf_add_bf16(f, shard, "mtp.pre_fc_norm_hidden.weight", {128}, 72);
    hf_add_bf16(f, shard, "mtp.norm.weight", {128}, 73);
    hf_add_bf16(f, shard, "mtp.fc.weight", {128, 256}, 74);
    hf_add_bf16(f, shard, "mtp.layers.0.input_layernorm.weight", {128}, 75);
    hf_add_bf16(f, shard, "mtp.layers.0.post_attention_layernorm.weight", {128}, 76);
    const auto add_matrix = [&](const std::string& name, uint64_t rows, uint64_t k, uint32_t seed) {
        hf_add_fp8_matrix(f, shard, name, rows, k, seed);
    };
    add_matrix("mtp.layers.0.mlp.gate_proj.weight", 128, 128, 77);
    add_matrix("mtp.layers.0.mlp.up_proj.weight", 128, 128, 78);
    add_matrix("mtp.layers.0.mlp.down_proj.weight", 128, 128, 79);
    add_matrix("mtp.layers.0.self_attn.q_proj.weight", 128, 128, 80);
    add_matrix("mtp.layers.0.self_attn.k_proj.weight", 32, 128, 81);
    add_matrix("mtp.layers.0.self_attn.v_proj.weight", 32, 128, 82);
    add_matrix("mtp.layers.0.self_attn.o_proj.weight", 128, 128, 83);
    hf_add_bf16(f, shard, "mtp.layers.0.self_attn.q_norm.weight", {32}, 84);
    hf_add_bf16(f, shard, "mtp.layers.0.self_attn.k_norm.weight", {32}, 85);
}

static bool build_hf_fp8_model(const fs::path& dir, HfFxpFixture& f,
                               bool invalid_mtp = false) {
    std::vector<Planned> a;
    std::vector<Planned> b;
    hf_add_bf16(f, a, "model.language_model.embed_tokens.weight", {40, 128}, 5);
    hf_add_bf16(f, a, "model.language_model.norm.weight", {128}, 6);
    hf_add_bf16(f, b, "lm_head.weight", {40, 128}, 7);
    hf_add_gdn_layer(f, a, 0);
    hf_add_attn_layer(f, b, 1);
    if (invalid_mtp) {
        Planned p;
        p.name = "mtp.fc.weight";
        p.dtype = io::SType::F32;
        p.shape = {128, 256};
        p.bytes.assign(128u * 256u * 4u, 0u);
        b.push_back(std::move(p));
    } else {
        hf_add_mtp(f, b);
    }

    const std::string shard_a = "model-00001-of-00002.safetensors";
    const std::string shard_b = "model-00002-of-00002.safetensors";
    if (!write_shard((dir / shard_a).string(), a, {})) return false;
    if (!write_shard((dir / shard_b).string(), b, {})) return false;
    write_index_multi(dir.string(), a, shard_a, b, shard_b);
    write_text((dir / "config.json").string(), hf_config_json(hf_fp8_quant_config()));
    return true;
}

static void check_fp8_hf_matrix(const weights::MatrixWeight& m, const HfFxpFixture& f,
                                const std::string& logical, uint64_t rows, uint64_t k) {
    const uint64_t kp = (k + 127u) / 128u * 128u;
    check(m.encoding == weights::MatrixEncoding::Fp8Finegrained, logical + " fp8 encoding");
    check(m.preshuffled, logical + " preshuffled");
    const auto n = expect_native_fp8(f.codes.at(logical), as_bytes(f.scales.at(logical)), rows, kp);
    check_matrix_bytes(m, n.codes_span(), as_bytes(n.scales_span()), logical + " native");
}

static void test_qwen35_hf_fp8_model_load(gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("hf_fp8_model");
    HfFxpFixture f;
    if (!build_hf_fp8_model(dir, f)) return;

    check(weights::is_quantized_model_dir(dir.string()), "hf fp8 dir is quantized");

    qwen35::Qwen35LoadOptions options;
    options.load_mtp_layers = true;
    options.verify_quantized_payload_crc = true;
    auto rr = qwen35::load_qwen35_weights_from_safetensors(dir.string(), arena, stream, options);
    if (!rr.ok()) { fail("hf fp8 model load: " + rr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = rr.release();

    check_eq_u64(w.layers.size(), 2, "hf fp8 layers");
    check(w.layers[0].is_gdn, "hf fp8 layer0 gdn");
    check(!w.layers[1].is_gdn, "hf fp8 layer1 attn");
    check(w.mtp.present, "hf fp8 mtp present");
    check(!w.lm_head_tied, "hf fp8 lm_head untied");

    check(w.embed_tokens.encoding == weights::MatrixEncoding::Bf16, "hf fp8 embed bf16");
    {
        const auto got = d2h(w.embed_tokens.bf16.data<void>(), f.bf16.at("model.language_model.embed_tokens.weight").size());
        check(got == f.bf16.at("model.language_model.embed_tokens.weight"), "hf fp8 embed bytes");
    }
    {
        const auto got = d2h(w.lm_head.bf16.data<void>(), f.bf16.at("lm_head.weight").size());
        check(got == f.bf16.at("lm_head.weight"), "hf fp8 lm_head bytes via bare name");
    }

    check_fp8_hf_matrix(w.layers[0].ffn.gate, f, hf_layer_prefix(0) + "mlp.gate_proj.weight", 128, 128);
    check_fp8_hf_matrix(w.layers[0].attn_in_proj_qkv, f, hf_layer_prefix(0) + "linear_attn.in_proj_qkv.weight", 24, 128);
    check_fp8_hf_matrix(w.layers[1].attn_q_proj, f, hf_layer_prefix(1) + "self_attn.q_proj.weight", 128, 128);
    check_fp8_hf_matrix(w.layers[1].attn_k_proj, f, hf_layer_prefix(1) + "self_attn.k_proj.weight", 32, 128);
    check_fp8_hf_matrix(w.layers[1].ffn.down, f, hf_layer_prefix(1) + "mlp.down_proj.weight", 128, 128);

    check(w.layers[0].attn_in_proj_a.encoding == weights::MatrixEncoding::Bf16, "hf fp8 in_proj_a bf16");
    check(w.layers[0].attn_conv1d_weight.ndim() == 3, "hf fp8 conv1d ndim");
    check(w.mtp.fc.encoding == weights::MatrixEncoding::Bf16, "hf fp8 mtp fc bf16");
    check(w.mtp.layer.ffn.gate.encoding == weights::MatrixEncoding::Fp8Finegrained,
          "hf fp8 mtp layer fp8");
}

static std::string hf_quark_mxfp4_quant_config() {
    return std::string("{\"quant_method\":\"quark\",\"global_quant_config\":{\"weight\":{") +
        "\"dtype\":\"fp4\",\"qscheme\":\"per_group\",\"group_size\":32,\"is_dynamic\":false," +
        "\"scale_format\":\"e8m0\"},\"input_tensors\":{\"dtype\":\"fp4\",\"qscheme\":\"per_group\"," +
        "\"group_size\":32,\"is_dynamic\":true,\"scale_format\":\"e8m0\"}}," +
        "\"export\":{\"weight_format\":\"real_quantized\"}}";
}

struct MxMatrix {
    uint64_t rows = 0;
    uint64_t k = 0;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
};

static MxMatrix make_mxfp4_matrix(uint64_t rows, uint64_t k, uint32_t seed,
                                  uint8_t forced_scale_byte = 0u) {
    MxMatrix m;
    m.rows = rows;
    m.k = k;
    std::vector<float> w(static_cast<std::size_t>(rows) * k, 0.0f);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (float& value : w) value = dist(rng);
    m.codes.assign(static_cast<std::size_t>(rows) * (k / 2u), 0u);
    m.scales.assign(static_cast<std::size_t>(rows) * (k / 32u), 0u);
    quantization::mxfp4::quantize_mxfp4_matrix(
        w.data(), k, rows, k, k, m.codes.data(), m.scales.data());
    if (forced_scale_byte != 0u) m.scales[0] = forced_scale_byte;
    return m;
}

static void test_hf_mxfp4_matrix_load(gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("hf_mxfp4");
    const MxMatrix good = make_mxfp4_matrix(32u, 128u, 91u);
    const MxMatrix nan_scale = make_mxfp4_matrix(32u, 128u, 93u, 255u);
    const MxMatrix narrow_rows = make_mxfp4_matrix(24u, 128u, 95u);

    std::vector<Planned> shard;
    const auto add = [&](const std::string& base, const MxMatrix& m) {
        Planned c;
        c.name = base + ".weight";
        c.dtype = io::SType::U8;
        c.shape = {static_cast<std::size_t>(m.rows), static_cast<std::size_t>(m.k / 2u)};
        c.bytes = m.codes;
        shard.push_back(std::move(c));
        Planned s;
        s.name = base + ".weight_scale";
        s.dtype = io::SType::U8;
        s.shape = {static_cast<std::size_t>(m.rows), static_cast<std::size_t>(m.k / 32u)};
        s.bytes = m.scales;
        shard.push_back(std::move(s));
    };
    add("gate", good);
    add("badscale", nan_scale);
    add("narrow", narrow_rows);
    if (!write_shard((dir / "model.safetensors").string(), shard, {})) return;
    write_text((dir / "config.json").string(),
               std::string("{\"model_type\":\"qwen3_5\",\"quantization_config\":") +
                   hf_quark_mxfp4_quant_config() + "}");

    check(weights::is_quantized_model_dir(dir.string()), "hf quark mxfp4 dir is quantized");
    auto source = quantization::bundle::synthesize_hf_quantized_source(
        dir.string(), {"gate.weight", "badscale.weight", "narrow.weight"}, true);
    if (!source.ok()) {
        fail("mxfp4 hf source: " + source.status().message());
        return;
    }
    auto reader_result = quantization::bundle::QuantizedModelReader::open_hf_source(
        source.release(), true);
    if (!reader_result.ok()) {
        fail("mxfp4 reader: " + reader_result.status().message());
        return;
    }
    quantization::bundle::QuantizedModelReader reader = reader_result.release();

    auto loaded = weights::load_quantized_matrix(
        reader, "gate.weight", arena, stream, weights::WeightLoadOptions{});
    if (!loaded.ok()) {
        fail("load mxfp4 matrix: " + loaded.status().message());
        return;
    }
    weights::MatrixWeight m = loaded.release();
    check(m.encoding == weights::MatrixEncoding::Mxfp4, "mxfp4 matrix encoding");
    check_eq_u64(m.rows, 32, "mxfp4 matrix rows");
    check_eq_u64(m.cols, 128, "mxfp4 matrix cols");
    check_eq_u64(m.k_padded, 128, "mxfp4 matrix k_padded");
    check_eq_u64(m.weight_scale_group, 32, "mxfp4 matrix weight scale group");
    check_eq_u64(m.codes_row_stride_bytes, 1024, "mxfp4 codes row stride is the native tile stride");
    check_eq_u64(m.scale_row_stride_bytes, 4, "mxfp4 scale row stride is group count");
    check_eq_u64(m.storage_scale_stride_bytes, 4, "mxfp4 storage scale stride");
    check(m.preshuffled, "mxfp4 matrix is native layout");
    check(m.quant_spec == quantization::QuantSpecId::MXFP4_E8M0_G32_V1, "mxfp4 quant spec");
    check(m.compute_spec == quantization::ComputeSpecId::MXFP4_W4A8_F32_BF16, "mxfp4 compute spec");
    quantization::mxfp4::Mxfp4CanonicalView cv;
    cv.batch = 1;
    cv.n = good.rows;
    cv.padded_k = good.k;
    cv.codes = good.codes.data();
    cv.codes_bytes = good.codes.size();
    cv.scales = good.scales.data();
    cv.scales_bytes = good.scales.size();
    quantization::mxfp4::Mxfp4NativeHost native;
    check(quantization::mxfp4::preshuffle_mxfp4_native(cv, native),
          "mxfp4 native preshuffle of the loaded payload");
    check(d2h(m.codes.data<void>(), native.codes.size()) == native.codes,
          "mxfp4 codes reach the GPU in the native fragment order");
    check(d2h(m.scales.data<void>(), good.scales.size()) == good.scales,
          "mxfp4 scale bytes reach the GPU unchanged");
    check(m.row_ref.ndim() != 0 && m.row_ref.data<void>() != nullptr,
          "mxfp4 native payload carries the row reference plane");
    check(d2h(m.row_ref.data<void>(), native.row_ref.size()) == native.row_ref,
          "mxfp4 row reference bytes reach the GPU");
    check(m.row_ref_exact == native.row_ref_exact,
          "mxfp4 fold exactness flag travels with the payload");

    weights::WeightLoadOptions canonical;
    canonical.preshuffle = false;
    auto canonical_load = weights::load_quantized_matrix(
        reader, "gate.weight", arena, stream, canonical);
    if (!canonical_load.ok()) {
        fail("load mxfp4 canonical: " + canonical_load.status().message());
    } else {
        const weights::MatrixWeight& cm = canonical_load.value();
        check(!cm.preshuffled, "mxfp4 canonical load keeps preshuffled false");
        check(d2h(cm.codes.data<void>(), good.codes.size()) == good.codes,
              "mxfp4 canonical codes bytes");
        check(d2h(cm.scales.data<void>(), good.scales.size()) == good.scales,
              "mxfp4 canonical scale bytes");
        check(cm.row_ref.ndim() == 0, "mxfp4 canonical payload has no row reference plane");
    }

    auto bad = weights::load_quantized_matrix(
        reader, "badscale.weight", arena, stream, weights::WeightLoadOptions{});
    check(!bad.ok(), "mxfp4 E8M0 NaN scale byte rejected at load");
    if (!bad.ok())
        check(std::string(bad.status().message()).find("E8M0") != std::string::npos,
              "mxfp4 NaN scale rejection message names E8M0");

    auto narrow = weights::load_quantized_matrix(
        reader, "narrow.weight", arena, stream, weights::WeightLoadOptions{});
    check(!narrow.ok(), "mxfp4 rows that are not a 16-row tile rejected at load");
}

static void test_qwen35_hf_fp8_model_load_without_mtp(
    gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("hf_fp8_model_no_mtp");
    HfFxpFixture f;
    if (!build_hf_fp8_model(dir, f, true)) return;

    qwen35::Qwen35LoadOptions options;
    options.verify_quantized_payload_crc = true;
    auto rr = qwen35::load_qwen35_weights_from_safetensors(dir.string(), arena, stream, options);
    if (!rr.ok()) {
        fail("hf fp8 model load without mtp: " + rr.status().message());
        return;
    }
    qwen35::Qwen35ModelWeights w = rr.release();

    check_eq_u64(w.layers.size(), 2, "hf fp8 no mtp layers");
    check(!w.mtp.present, "hf fp8 no mtp present");
    check_eq_u64(w.mtp.fc.rows, 0, "hf fp8 no mtp fc rows");
    check_eq_u64(w.mtp.layer.ffn.gate.rows, 0, "hf fp8 no mtp layer rows");
}

static std::vector<uint8_t> exl3_pattern(std::size_t bytes, uint32_t seed) {
    std::vector<uint8_t> out(bytes);
    uint32_t state = seed;
    for (uint8_t& b : out) {
        state = state * 1664525u + 1013904223u;
        b = static_cast<uint8_t>(state >> 24);
    }
    return out;
}

static void test_hf_exl3_matrix_load(gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("hf_exl3");
    constexpr uint32_t k = 128u;
    constexpr uint32_t n = 128u;
    constexpr uint32_t words = 64u;

    const std::vector<uint8_t> trellis = exl3_pattern(8u * 8u * words * 2u, 71u);
    const std::vector<uint8_t> suh = exl3_pattern(k * 2u, 73u);
    const std::vector<uint8_t> svh = exl3_pattern(n * 2u, 79u);
    const std::vector<uint8_t> marker = {0x2Du, 0xD1u, 0xDCu, 0x83u};

    std::vector<Planned> shard;
    shard.push_back({"layer.trellis", io::SType::I16, {8, 8, words}, trellis});
    shard.push_back({"layer.suh", io::SType::F16, {k}, suh});
    shard.push_back({"layer.svh", io::SType::F16, {n}, svh});
    shard.push_back({"layer.mul1", io::SType::I32, {}, marker});
    if (!write_shard((dir / "model.safetensors").string(), shard, {})) return;
    write_index(dir.string(), "model.safetensors", shard);
    write_text((dir / "config.json").string(),
               std::string("{\"model_type\":\"qwen3_5\",\"quantization_config\":")
                   + "{\"quant_method\":\"exl3\",\"codebook\":\"mul1\"}}");

    auto source = quantization::bundle::synthesize_hf_quantized_source(
        dir.string(), {"layer.weight"}, true);
    if (!source.ok()) {
        fail("exl3 hf source: " + source.status().message());
        return;
    }
    auto reader_result = quantization::bundle::QuantizedModelReader::open_hf_source(
        source.release(), true);
    if (!reader_result.ok()) {
        fail("exl3 reader: " + reader_result.status().message());
        return;
    }
    quantization::bundle::QuantizedModelReader reader = reader_result.release();

    auto loaded = weights::load_quantized_matrix(
        reader, "layer.weight", arena, stream, weights::WeightLoadOptions{});
    if (!loaded.ok()) {
        fail("load exl3 matrix: " + loaded.status().message());
        return;
    }
    weights::MatrixWeight m = loaded.release();
    check(m.encoding == weights::MatrixEncoding::Exl3, "exl3 matrix encoding");
    check_eq_u64(m.rows, n, "exl3 matrix rows");
    check_eq_u64(m.cols, k, "exl3 matrix cols");
    check_eq_u64(m.k_padded, k, "exl3 matrix k_padded");
    check_eq_u64(m.exl3_ka, 4, "exl3 matrix bit rate");
    check(!m.exl3_half, "exl3 matrix integer rate");
    check(!m.preshuffled, "exl3 matrix is not preshuffled");
    check_eq_u64(m.codes_row_stride_bytes, 8u * words * 2u, "exl3 trellis row stride");
    check(m.codes.ndim() == 3, "exl3 trellis tensor is rank 3");
    check(m.exl3_suh.ndim() == 1, "exl3 suh tensor is rank 1");
    check(m.exl3_svh.ndim() == 1, "exl3 svh tensor is rank 1");
    check(m.exl3_marker.ndim() == 1, "exl3 marker tensor is rank 1");
    check(d2h(m.codes.data<void>(), trellis.size()) == trellis, "exl3 trellis reaches the GPU");
    check(d2h(m.exl3_suh.data<void>(), suh.size()) == suh, "exl3 suh reaches the GPU");
    check(d2h(m.exl3_svh.data<void>(), svh.size()) == svh, "exl3 svh reaches the GPU");
    check(d2h(m.exl3_marker.data<void>(), marker.size()) == marker,
          "exl3 marker reaches the GPU");

    const runtime::WeightSlot slot = shelf::qwen_hybrid::make_weight_slot(m);
    check(slot.encoding == static_cast<uint8_t>(weights::MatrixEncoding::Exl3),
          "exl3 weight slot encoding");
    check(slot.codes == m.codes.data<void>() && slot.exl3_suh == m.exl3_suh.data<void>() &&
              slot.exl3_svh == m.exl3_svh.data<void>() &&
              slot.exl3_marker == m.exl3_marker.data<void>(),
          "exl3 weight slot carries the four payload planes");
    check_eq_u64(slot.exl3_ka, 4, "exl3 weight slot bit rate");
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

    auto ar = gpu::GpuArena::create(0, 512ull * 1024ull * 1024ull);
    if (!ar.ok()) { fail("arena"); return 1; }
    gpu::GpuArena arena = ar.release();

    std::error_code ec;
    g_root = fs::temp_directory_path(ec) / ("shelf_weight_load_" + std::to_string(::getpid()));
    fs::create_directories(g_root, ec);

    test_bf16_matrix_load(arena, stream);
    test_bf16_payload_dtype_guard(arena, stream);
    test_bf16_model_load(arena, stream);
    test_bf16_model_load_bare_lm_head(arena, stream);
    test_standalone_quantized_reader(arena, stream);
    test_qwen35_quantized_model_load(arena, stream);
    test_qwen35_hf_fp8_model_load(arena, stream);
    test_qwen35_hf_fp8_model_load_without_mtp(arena, stream);
    test_hf_mxfp4_matrix_load(arena, stream);
    test_hf_exl3_matrix_load(arena, stream);

    fs::remove_all(g_root, ec);
    arena.shutdown();
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    std::printf("test_weight_load: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
