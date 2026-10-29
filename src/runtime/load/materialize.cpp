// shelf runtime/load — 検証と実体化の実装（D3）。
#include <shelf/runtime/load/materialize.hpp>

#include <cstring>
#include <exception>
#include <utility>

#include <nlohmann/json.hpp>

namespace shelf::runtime::load {
namespace {

using namespace shelf::quant::fp8;

std::size_t numel_of(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

}  // namespace

WeightKind classify(const TensorSource& src, const std::string& name) {
    if (!src.has(name)) return WeightKind::Missing;
    auto v = src.read(name);
    if (!v.ok()) return WeightKind::Missing;
    switch (v.value().dtype) {
    case DataType::bf16:
        return WeightKind::Bf16;
    case DataType::f32:
        return WeightKind::F32;
    case DataType::fp8_e4m3:
        return WeightKind::Fp8W8A8;
    default:
        return WeightKind::Missing;
    }
}

Result<std::vector<bf16_t>> read_bf16(const TensorSource& src, const std::string& name) {
    auto v = src.read(name);
    if (!v.ok()) return v.status();
    if (v.value().dtype != DataType::bf16)
        return Status::unsupported("bf16 ではない: " + name, __FILE__, __LINE__);
    const auto& b = v.value().bytes;
    const std::size_t n = b.size() / sizeof(bf16_t);
    std::vector<bf16_t> out(n);
    if (!b.empty()) std::memcpy(out.data(), b.data(), b.size());
    return out;
}

Result<std::vector<std::uint8_t>> read_u8(const TensorSource& src, const std::string& name) {
    auto v = src.read(name);
    if (!v.ok()) return v.status();
    if (v.value().dtype != DataType::fp8_e4m3 && v.value().dtype != DataType::u8)
        return Status::unsupported("8 ビットのペイロードではない: " + name, __FILE__, __LINE__);
    return std::move(v).value().bytes;
}

Result<std::vector<float>> read_f32(const TensorSource& src, const std::string& name) {
    auto v = src.read(name);
    if (!v.ok()) return v.status();
    const auto dtype = v.value().dtype;
    if (dtype == DataType::f32) {
        std::vector<float> out(v.value().bytes.size() / sizeof(float));
        if (!v.value().bytes.empty())
            std::memcpy(out.data(), v.value().bytes.data(), v.value().bytes.size());
        return out;
    }
    if (dtype == DataType::bf16) {
        const std::size_t n = v.value().bytes.size() / sizeof(bf16_t);
        std::vector<float> out(n);
        const auto* p = reinterpret_cast<const bf16_t*>(v.value().bytes.data());
        for (std::size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
        return out;
    }
    if (dtype == DataType::fp8_e4m3)
        return Status::unsupported(
            "read_f32: FP8 の重みは f32 ゴールデン（CPU 参照）では読まない。"
            "GPU 経路を使うか、正本の bf16 checkpoint を使う: " + name,
            __FILE__, __LINE__);
    return Status::unsupported("read_f32: 未対応 dtype: " + name, __FILE__, __LINE__);
}

Result<Fp8ModelFormat> read_fp8_format(const std::string& config_json_text) {
    using shelf::quant::fp8::WeightScaleGranularity;
    Fp8ModelFormat f;
    if (config_json_text.empty()) return f;      // config 自体が読めない場合は非量子化扱い
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(config_json_text);
    } catch (const std::exception& e) {
        return Status::invalid_argument(std::string("config.json parse: ") + e.what(),
                                        __FILE__, __LINE__);
    }
    if (!root.contains("quantization_config") ||
        !root.at("quantization_config").is_object())
        return f;                                 // 量子化されていない checkpoint（正本など）
    const auto& q = root.at("quantization_config");
    const std::string method = q.value("quant_method", std::string());

    auto bad = [&](const std::string& msg) {
        return Status::unsupported("quantization_config: " + msg, __FILE__, __LINE__);
    };

    if (q.contains("kv_cache_scheme") && !q.at("kv_cache_scheme").is_null())
        return bad("KV キャッシュの量子化は未対応（kv_cache_scheme が non-null）");

    if (method == "compressed-tensors") {
        f.quantized = true;
        f.scale_suffix = "_scale";
        if (!q.contains("config_groups") || !q.at("config_groups").is_object() ||
            q.at("config_groups").empty())
            return bad("compressed-tensors に config_groups がない");
        const auto& groups = q.at("config_groups");
        const auto git = groups.begin();
        const auto& g = *git;
        if (!g.contains("weights") || !g.at("weights").is_object())
            return bad("weights が無い");
        const auto& wj = g.at("weights");
        if (wj.value("type", std::string()) != "float" || wj.value("num_bits", 0) != 8)
            return bad("重みは float / 8 bit 以外未対応");
        const std::string wstrat = wj.value("strategy", std::string());
        if (wstrat == "channel") {
            f.weight = WeightScaleGranularity::PerChannel;
        } else if (wstrat == "block") {
            if (!wj.contains("block_structure") || !wj.at("block_structure").is_array() ||
                wj.at("block_structure").size() != 2 ||
                wj.at("block_structure")[0].get<int>() != 128 ||
                wj.at("block_structure")[1].get<int>() != 128)
                return bad("重み block は 128x128 以外未対応");
            f.weight = WeightScaleGranularity::Block128;
        } else {
            return bad("重み strategy='" + wstrat + "' 未対応");
        }

        std::string astrat = "tensor";
        if (g.contains("input_activations") && g.at("input_activations").is_object()) {
            const auto& aj = g.at("input_activations");
            if (!aj.value("dynamic", false))
                return bad("活性化の静的スケール（input_scale テンソル）は未対応");
            astrat = aj.value("strategy", std::string());
            if (astrat == "token" || astrat == "tensor") {
                f.act_group = 0u;                    // 1 行（= トークン）で 1 スケール
            } else if (astrat == "group" || astrat == "block") {
                std::size_t gs = 0;
                if (astrat == "group") gs = aj.value("group_size", 0);
                else if (aj.contains("block_structure") && aj.at("block_structure").is_array() &&
                          !aj.at("block_structure").empty())
                    gs = aj.at("block_structure").back().get<std::size_t>();
                if (gs != 128u)
                    return bad("活性化 group/block=" + std::to_string(gs) + " は未対応（128 のみ）");
                f.act_group = 128u;
            } else {
                return bad("活性化 strategy='" + astrat + "' 未対応");
            }
        }
        f.declared = "compressed-tensors weight=" + wstrat + " act=" + astrat +
                     (f.act_group ? "(1x128)" : "(per-token)");
        return f;
    }

    if (method == "fp8") {
        // HF blockwise 形。スケールは `<base>.weight_scale_inv` で、格納値は
        // **実乗数**（dequant = code × inv）。活性化は 1x128 の動的スケーリング。
        f.quantized = true;
        f.scale_suffix = "_scale_inv";
        if (q.value("fmt", std::string("e4m3")) != "e4m3")
            return bad("fmt=" + q.value("fmt", std::string()) + " は未対応（e4m3 のみ）");
        if (q.value("activation_scheme", std::string("dynamic")) != "dynamic")
            return bad("activation_scheme が dynamic 以外（静的スケール）は未対応");
        if (!q.contains("weight_block_size") || !q.at("weight_block_size").is_array() ||
            q.at("weight_block_size").size() != 2 ||
            q.at("weight_block_size")[0].get<int>() != 128 ||
            q.at("weight_block_size")[1].get<int>() != 128)
            return bad("weight_block_size が [128,128] 以外は未対応");
        f.weight = WeightScaleGranularity::Block128;
        f.act_group = 128u;
        f.declared = "fp8 blockwise 128x128 (weight_scale_inv) + dynamic 1x128";
        return f;
    }

    return bad("quant_method='" + method + "' 未対応");
}

Result<MaterializedWeight> materialize(const TensorSource& src, const WeightRequest& req,
                                       const Fp8ModelFormat& fmt) {
    MaterializedWeight out;
    out.kind = classify(src, req.checkpoint_name);
    if (out.kind == WeightKind::Missing)
        return Status::not_found("weight not in source: " + req.checkpoint_name, __FILE__,
                                 __LINE__);
    if (req.shape.empty())
        return Status::invalid_argument("shape が空: " + req.checkpoint_name, __FILE__, __LINE__);

    const std::size_t want = numel_of(req.shape);
    if (out.kind == WeightKind::Bf16) {
        auto t = read_bf16(src, req.checkpoint_name);
        if (!t.ok()) return t.status();
        if (t.value().size() != want)
            return Status::invalid_state("要素数が shape と不一致: " + req.checkpoint_name,
                                         __FILE__, __LINE__);
        out.bf16 = std::move(t).value();
        return out;
    }
    if (out.kind == WeightKind::F32) {
        auto t = read_f32(src, req.checkpoint_name);
        if (!t.ok()) return t.status();
        if (t.value().size() != want)
            return Status::invalid_state("要素数が shape と不一致: " + req.checkpoint_name,
                                         __FILE__, __LINE__);
        out.f32 = std::move(t).value();
        return out;
    }

    // ---- FP8 W8A8（E4M3 コード + スケール）----
    // FP8 の重みは Linear の 2 次元のみ（スケールの粒度が [out, K] 前提）。
    if (req.shape.size() != 2u)
        return Status::unsupported("fp8 重みが 2 次元でない: " + req.checkpoint_name, __FILE__,
                                   __LINE__);
    if (req.shape[0] > 0xFFFFFFFFu || req.shape[1] > 0xFFFFFFFFu)
        return Status::invalid_argument("重みが大きすぎる: " + req.checkpoint_name, __FILE__,
                                        __LINE__);
    const std::uint32_t features = static_cast<std::uint32_t>(req.shape[0]);
    const std::uint32_t cols = static_cast<std::uint32_t>(req.shape[1]);
    const std::uint32_t kp = pad_k(cols);
    if (kp != cols)
        return Status::unsupported("fp8 重みで K が 32 の倍数でない（断片順に置けない）: " +
                                       req.checkpoint_name,
                                   __FILE__, __LINE__);
    auto codes = read_u8(src, req.checkpoint_name);
    if (!codes.ok()) return codes.status();
    if (codes.value().size() != want)
        return Status::invalid_state("fp8 コード数が shape と不一致: " + req.checkpoint_name,
                                     __FILE__, __LINE__);

    // 名前の規約: `<base>.weight` に対してスケールは `<base>.weight<suffix>`
    // （`weight` を差し替える。連結ではない）。
    std::string sname;
    const std::string suffix = ".weight";
    if (req.checkpoint_name.size() > suffix.size() &&
        req.checkpoint_name.compare(req.checkpoint_name.size() - suffix.size(), suffix.size(),
                                    suffix) == 0)
        sname = req.checkpoint_name.substr(0, req.checkpoint_name.size() - suffix.size()) +
                ".weight" + fmt.scale_suffix;
    else
        sname = req.checkpoint_name + fmt.scale_suffix;
    if (!src.has(sname))
        return Status::unsupported("宣言されたスケールテンソルが見つからない: " + sname + "（" +
                                       fmt.declared + "）",
                                   __FILE__, __LINE__);
    auto sv = src.read(sname);
    if (!sv.ok()) return sv.status();
    const std::size_t have = sv.value().bytes.size() / element_size(sv.value().dtype);
    const std::size_t need = weight_scale_count(fmt.weight, features, kp);
    if (have != need)
        return Status::invalid_state(
            "weight_scale の個数が宣言と不一致: " + sname + " = " + std::to_string(have) +
                "（宣言 " + std::to_string(need) + " / " + fmt.declared + "）",
            __FILE__, __LINE__);
    auto sc = read_f32(src, sname);   // bf16 / f32 のどちらでも f32 へ（値域は狭く無損）
    if (!sc.ok()) return sc.status();

    out.features = features;
    out.k_padded = kp;
    out.weight_block128 = (fmt.weight == WeightScaleGranularity::Block128);
    out.scales = std::move(sc).value();
    out.codes.assign(tiled_bytes(features, kp), 0u);
    SHELF_RETURN_IF_ERROR(preshuffle_codes(codes.value().data(), features, cols, kp,
                                           out.codes.data()));
    return out;
}

}  // namespace shelf::runtime::load
