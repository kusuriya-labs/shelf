// shelf apps/common — 読取ヘルパの実装（shelf-run / shelf-bench 共有）。
#include "app_support.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include <shelf/quant/fp8_w8a8.hpp>

namespace shelf::app {

std::string read_file_str(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::size_t numel(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

Status open_shards(const std::string& model_dir, ShardSet* out) {
    if (out == nullptr)
        return Status::invalid_argument("open_shards: null out", __FILE__, __LINE__);
    const std::string index_path = model_dir + "/model.safetensors.index.json";
    const std::string text = read_file_str(index_path);
    if (text.empty())
        return Status::not_found("cannot read " + index_path, __FILE__, __LINE__);

    nlohmann::json index;
    try {
        index = nlohmann::json::parse(text);
    } catch (const std::exception& e) {
        return Status::invalid_argument(std::string("parse index.json: ") + e.what(),
                                        __FILE__, __LINE__);
    }
    if (!index.contains("weight_map") || !index.at("weight_map").is_object())
        return Status::invalid_argument("index.json: missing weight_map", __FILE__, __LINE__);

    std::vector<std::string> shard_files;
    for (const auto& f : index.at("weight_map").items()) {
        if (!f.value().is_string()) continue;
        const std::string file = f.value().get<std::string>();
        if (std::find(shard_files.begin(), shard_files.end(), file) == shard_files.end())
            shard_files.push_back(file);
    }
    if (shard_files.empty())
        return Status::invalid_argument("index.json: empty weight_map", __FILE__, __LINE__);

    out->readers.clear();
    out->where.clear();
    out->readers.reserve(shard_files.size());
    for (const auto& file : shard_files) {
        auto r = artifact::safetensors::Reader::open(model_dir + "/" + file);
        if (!r.ok())
            return Status::invalid_argument("open " + file + ": " + r.status().message(),
                                            __FILE__, __LINE__);
        out->readers.push_back(
            std::make_unique<artifact::safetensors::Reader>(r.release()));
    }
    // reader の実体は unique_ptr で固定（vector のムーブでポインタが死なない）。
    for (auto& rp : out->readers)
        for (const auto& name : rp->list_tensors())
            if (!out->where.contains(name)) out->where.emplace(name, rp.get());
    return Status::make_ok();
}

Result<std::vector<bf16_t>> read_bf16(
    const std::map<std::string, artifact::safetensors::Reader*>& where,
    const std::string& name) {
    auto it = where.find(name);
    if (it == where.end())
        return Status::not_found("tensor not in shards: " + name, __FILE__, __LINE__);
    const auto* reader = it->second;
    const auto* entry = reader->catalog().find(name);
    if (!entry)
        return Status::not_found("catalog miss: " + name, __FILE__, __LINE__);
    if (entry->dtype != DataType::bf16)
        return Status::unsupported("not bf16 on disk: " + name, __FILE__, __LINE__);
    const std::size_t bytes = entry->end - entry->begin;
    std::vector<bf16_t> out(bytes / 2);
    SHELF_RETURN_IF_ERROR(reader->read_tensor(name, out.data(), bytes));
    return out;
}

Result<std::vector<float>> read_f32(
    const std::map<std::string, artifact::safetensors::Reader*>& where,
    const std::string& name) {
    auto it = where.find(name);
    if (it == where.end())
        return Status::not_found("tensor not in shards: " + name, __FILE__, __LINE__);
    const auto* reader = it->second;
    const auto* entry = reader->catalog().find(name);
    if (!entry)
        return Status::not_found("catalog miss: " + name, __FILE__, __LINE__);
    const std::size_t bytes = entry->end - entry->begin;
    if (entry->dtype == DataType::f32) {
        std::vector<float> out(bytes / sizeof(float));
        SHELF_RETURN_IF_ERROR(reader->read_tensor(name, out.data(), bytes));
        return out;
    }
    if (entry->dtype == DataType::bf16) {
        std::vector<bf16_t> raw(bytes / 2);
        SHELF_RETURN_IF_ERROR(reader->read_tensor(name, raw.data(), bytes));
        std::vector<float> out(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) out[i] = bf16_to_f32(raw[i]);
        return out;
    }
    if (entry->dtype == DataType::fp8_e4m3)
        return Status::unsupported(
            "read_f32: FP8 の重みは f32 ゴールデン（CPU 参照）では読まない。"
            "GPU 経路を使うか、正本の bf16 checkpoint を使う: " + name,
            __FILE__, __LINE__);
    return Status::unsupported("read_f32: dtype for " + name, __FILE__, __LINE__);
}

Result<std::vector<std::uint8_t>> read_u8(
    const std::map<std::string, artifact::safetensors::Reader*>& where,
    const std::string& name) {
    auto it = where.find(name);
    if (it == where.end())
        return Status::not_found("tensor not in shards: " + name, __FILE__, __LINE__);
    const auto* reader = it->second;
    const auto* entry = reader->catalog().find(name);
    if (!entry)
        return Status::not_found("catalog miss: " + name, __FILE__, __LINE__);
    if (entry->dtype != DataType::fp8_e4m3 && entry->dtype != DataType::u8)
        return Status::unsupported("not an 8-bit payload on disk: " + name, __FILE__, __LINE__);
    const std::size_t bytes = entry->end - entry->begin;
    std::vector<std::uint8_t> out(bytes);
    SHELF_RETURN_IF_ERROR(reader->read_tensor(name, out.data(), bytes));
    return out;
}

WeightKind classify(const std::map<std::string, artifact::safetensors::Reader*>& where,
                    const std::string& name) {
    auto it = where.find(name);
    if (it == where.end()) return WeightKind::Missing;
    const auto* entry = it->second->catalog().find(name);
    if (!entry) return WeightKind::Missing;
    switch (entry->dtype) {
        case DataType::bf16:     return WeightKind::Bf16;
        case DataType::f32:      return WeightKind::F32;
        case DataType::fp8_e4m3: return WeightKind::Fp8W8A8;
        default:                 return WeightKind::Missing;
    }
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

Result<Fp8WeightPayload> build_fp8_weight(
    const std::map<std::string, artifact::safetensors::Reader*>& where, const std::string& name,
    const std::vector<std::size_t>& shape, const Fp8ModelFormat& fmt) {
    using namespace shelf::quant::fp8;
    if (shape.size() != 2u)
        return Status::invalid_argument("fp8 重みは 2 次元のみ: " + name, __FILE__, __LINE__);
    if (shape[0] > 0xFFFFFFFFu || shape[1] > 0xFFFFFFFFu)
        return Status::invalid_argument("fp8 重みが大きすぎる: " + name, __FILE__, __LINE__);
    const std::uint32_t features = static_cast<std::uint32_t>(shape[0]);
    const std::uint32_t cols = static_cast<std::uint32_t>(shape[1]);
    const std::uint32_t kp = pad_k(cols);
    if (kp != cols)
        return Status::unsupported("fp8 重みで K が 32 の倍数でない（断片順に置けない）: " + name,
                                   __FILE__, __LINE__);

    auto codes = read_u8(where, name);
    if (!codes.ok()) return codes.status();
    if (codes.value().size() != static_cast<std::size_t>(features) * cols)
        return Status::invalid_state("fp8 コード数が shape と不一致: " + name,
                                     __FILE__, __LINE__);

    // 名前の規約: `<base>.weight` に対してスケールは `<base>.weight<suffix>`
    // （`weight` を差し替える。連結ではない）。suffix は config の宣言から来る
    // （`_scale` / `_scale_inv`）。
    std::string sname;
    const std::string suffix = ".weight";
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
        sname = name.substr(0, name.size() - suffix.size()) + ".weight" + fmt.scale_suffix;
    else
        sname = name + fmt.scale_suffix;

    auto it = where.find(sname);
    if (it == where.end())
        return Status::unsupported("宣言されたスケールテンソルが見つからない: " + sname +
                                       "（" + fmt.declared + "）",
                                   __FILE__, __LINE__);
    const auto* sentry = it->second->catalog().find(sname);
    if (sentry == nullptr)
        return Status::not_found("catalog miss: " + sname, __FILE__, __LINE__);

    // 個数は宣言（config）から決める。実データと合わなければ**推測せず**失敗させる。
    const std::size_t want =
        weight_scale_count(fmt.weight, features, kp);
    const std::size_t have = static_cast<std::size_t>(sentry->end - sentry->begin) /
                             element_size(sentry->dtype);
    if (have != want)
        return Status::invalid_state(
            "weight_scale の個数が宣言と不一致: " + sname + " = " + std::to_string(have) +
            "（宣言 " + std::to_string(want) + " / " + fmt.declared + "）",
            __FILE__, __LINE__);

    std::vector<float> scales;
    if (sentry->dtype == DataType::bf16) {
        auto sc = read_bf16(where, sname);
        if (!sc.ok()) return sc.status();
        scales.resize(sc.value().size());
        for (std::size_t i = 0; i < sc.value().size(); ++i)
            scales[i] = bf16_to_f32(sc.value()[i]);      // 値域は狭いので無損
    } else if (sentry->dtype == DataType::f32) {
        auto sc = read_f32(where, sname);
        if (!sc.ok()) return sc.status();
        scales = sc.value();
    } else {
        return Status::unsupported("weight_scale の dtype 未対応: " + sname, __FILE__, __LINE__);
    }

    Fp8WeightPayload out;
    out.features = features;
    out.k_padded = kp;
    out.weight_block128 = (fmt.weight == WeightScaleGranularity::Block128);
    out.scales = std::move(scales);
    out.codes_tiled.assign(tiled_bytes(features, kp), 0u);
    SHELF_RETURN_IF_ERROR(preshuffle_codes(codes.value().data(), features, cols, kp,
                                           out.codes_tiled.data()));
    return out;
}

}  // namespace shelf::app
