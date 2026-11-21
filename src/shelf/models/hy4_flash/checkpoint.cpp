#include <shelf/models/hy4_flash/checkpoint.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <unistd.h>
#include <sys/resource.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace shelf::hy4 {
namespace {
using Json = nlohmann::json;
Json read_json(const Path &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("ファイルを開けません: " + path.string());
    return Json::parse(in);
}
void require(bool valid, const std::string &msg) {
    if (!valid)
        throw std::runtime_error(msg);
}
template <class T> T take(Result<T> value) {
    if (!value.ok())
        throw std::runtime_error(value.status().message());
    return value.release();
}
}
Owners read_owners(const Path &file) {
    const auto data = read_json(file);
    Owners owners{};
    require(data.size() == 41, "配置表の層数が違います");
    for (size_t i = 0; i < 41; ++i) {
        require(data[i].size() == 512, "配置表の expert 数が違います");
        for (size_t e = 0; e < 512; ++e) {
            require(data[i][e].is_number_integer(), "配置表の rank が不正です");
            int r = data[i][e].get<int>();
            require(r >= 0 && r < 4, "配置表の rank が範囲外です");
            owners[i][e] = r;
        }
    }
    return owners;
}
const Weight &Checkpoint::at(const std::string &key) const {
    const auto it = weights.find(key);
    if (it == weights.end())
        throw std::runtime_error("重みがありません: " + key);
    return it->second;
}
void Checkpoint::populate_engram(bool lock) const {
    auto begin = std::chrono::steady_clock::now();
    if (lock) {
        rlimit limit{};
        if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0)
            throw std::runtime_error("memlock 上限を取得できません");
        long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) throw std::runtime_error("page size の取得に失敗しました");
        size_t bytes = 0;
        for (auto index : engram_files)
            bytes += (files.at(index).file_size() + page - 1) / page * page;
        if (limit.rlim_cur != RLIM_INFINITY && bytes > limit.rlim_cur)
            throw std::runtime_error("Engram の固定に必要な memlock 上限が不足しています");
    }
    for (auto index : engram_files) {
        auto status = lock ? files.at(index).lock_pages() : files.at(index).populate_read();
        if (!status.ok())
            throw std::runtime_error("Engram の populate に失敗しました: " + status.message());
    }
    fprintf(stderr, "engram_populate_s=%.3f locked=%d\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(), lock);
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
        throw std::runtime_error("page size の取得に失敗しました");
    bool complete = true;
    for (size_t g = 0; g < engram_files.size(); ++g) {
        const auto &file = files.at(engram_files[g]);
        auto count = file.resident_pages();
        if (!count.ok())
            throw std::runtime_error("Engram の常駐確認に失敗しました: " + count.status().message());
        size_t total = (file.file_size() + page - 1) / page;
        fprintf(stderr, "engram_populate layer=%zu pages=%zu total=%zu\n", 1 + 10 * g, count.value(), total);
        complete &= count.value() == total;
    }
    if (!complete) {
        if (lock) throw std::runtime_error("固定した Engram の常駐確認に失敗しました");
        fprintf(stderr, "engram_populate_incomplete=1\n");
    }
}
void Checkpoint::report_engram_residency(const char *phase) const {
    size_t resident = 0, total = 0;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
        throw std::runtime_error("page size の取得に失敗しました");
    for (auto index : engram_files) {
        const auto &file = files.at(index);
        auto count = file.resident_pages();
        if (!count.ok())
            throw std::runtime_error("Engram の常駐確認に失敗しました: " + count.status().message());
        resident += count.value();
        total += (file.file_size() + page - 1) / page;
    }
    fprintf(stderr, "engram_resident phase=%s pages=%zu total=%zu page_size=%ld\n",
            phase, resident, total, page);
}
Checkpoint::Checkpoint(const Path &model, const Path &engram, const Path &draft, const Owners &layout)
    : owners(layout), has_draft(!draft.empty()) {
    const auto cfg = read_json(model / "config.json");
    require(cfg.value("model_type", std::string{}) == "hy_v4_flash" && cfg.value("exl3_text_only", false) &&
                cfg.at("num_hidden_layers") == 42 && cfg.at("hc_mult") == 4 &&
                cfg.at("n_routed_experts") == 512 && cfg.at("num_experts_per_tok") == 16 &&
                cfg.at("hidden_size") == 3072 && cfg.at("routed_expert_hidden_size") == 1536 &&
                cfg.at("kv_lora_rank") == 512 && cfg.at("vocab_size") == 120832,
            "Hy4-flash の text 専用 checkpoint が必要です");
    const auto &quant = cfg.at("quantization_config");
    mixed = quant.at("bits") == 3.5;
    require(quant.at("mul1_multiplier") == quantization::exl3::kExl3Mul24Multiplier &&
                (mixed || quant.at("bits") == 4) && quant.at("head_bits") == 6,
            "量子化形式が不正です");
    require(std::filesystem::is_regular_file(model / "conversion_report.json"), "本体の組立が未完了です");
    for (int i = 0; i < 42; ++i) {
        const bool sparse = i == 0 || i % 4 == 1;
        require(cfg.at("layer_types").at(i) == (sparse ? "deepseek_sparse_attention" : "sliding_attention"),
                "attention 層の順序が異なります");
    }
    for (const auto &[name, meta] : quant.at("linear_storage").items())
        shape[name.substr(0, name.size() - 7)] = meta.at("original_shape").get<std::array<int, 2>>();
    load(model, false);
    for (int g = 0; g < 4; ++g) {
        int layer = 1 + 10 * g;
        const auto &orders = cfg.at("engram_layer_head_vocab_sizes").at(std::to_string(layer + 1));
        size_t rows = 0;
        for (int h = 0; h < 16; ++h) {
            int size = orders.at(h / 8).at(h % 8).get<int>();
            require(size > 0, "Engram の語彙数が不正です");
            rows += size;
            moduli[g][h] = size;
        }
        auto key =
            "model.layers." + std::to_string(layer) + ".engram_layer.multi_head_embedding.embedding.weight";
        auto reader = take(io::SafetensorsReader::open((engram / (key + ".fp8.safetensors")).string()));
        for (bool scale : {false, true}) {
            auto name = scale ? key + ".scale" : key;
            auto spec = take(reader.tensor_spec(name));
            auto expected = scale ? io::SType::BF16 : io::SType::F8_E4M3;
            require(spec.dtype == expected && spec.shape == std::vector<size_t>{rows, scale ? 1u : 96u},
                    "Engram 表の型または形状が不正です");
            size_t length;
            auto data = take(reader.tensor_data(name, length));
            weights.emplace(name, Weight{spec, data, length, -1, 0});
        }
        engram_files[g] = files.size();
        files.push_back(std::move(reader));
    }
    if (has_draft) {
        auto q = read_json(draft / "conversion_report.json");
        require(q.at("method") == "dfly" && q.at("bits") == 4, "DFly の変換結果が不正です");
        const auto report = read_json(model / "conversion_report.json");
        const auto &signature = report.at("assembly_signature");
        require(q.at("body_assembly_signature") ==
                        (mixed ? report.at("dfly_calibration_signature") : signature) &&
                    (!mixed || q.at("body_report_sha256") == signature.at("body_report_sha256")),
                "DFly と本体の校正元が一致しません");
        for (const auto &[name, dims] : q.at("linear_shapes").items())
            shape[name] = dims.get<std::array<int, 2>>();
        load(draft, true);
    }
    for (const auto &[key, w] : weights) {
        if (!key.ends_with(".trellis"))
            continue;
        const auto name = key.substr(0, key.size() - 8);
        const bool expert = name.find(".mlp.experts.") != std::string::npos;
        auto logical = expert ? std::array<int, 2>{1536, 1536} : shape.at(name);
        int k = logical[0], n = logical[1], bits = name == "lm_head" ? 6 : 4;
        if (mixed && expert && w.spec.shape.size() == 3 && w.spec.shape[2] == 48)
            bits = 3;
        require(k > 0 && n > 0 && k <= 120832 && n <= 120832, "線形の幅が不正です: " + name);
        size_t pk = (k + 127) / 128 * 128, pn = (n + 127) / 128 * 128;
        auto matrix = [&](const std::string &suffix, io::SType dtype, std::vector<size_t> dims) {
            const auto &part = at(name + suffix);
            require(part.spec.dtype == dtype && part.spec.shape == dims,
                    "量子化 tensor が不正です: " + name + suffix);
        };
        matrix(".suh", io::SType::F16, {pk});
        matrix(".svh", io::SType::F16, {pn});
        matrix(".trellis", io::SType::I16, {pk / 16, pn / 16, size_t(16 * bits)});
        matrix(".mul1", io::SType::I32, {});
        int32_t marker;
        std::memcpy(&marker, at(name + ".mul1").host, sizeof(marker));
        require(marker == quantization::exl3::kExl3Mul24Multiplier, "mul24 乗数が不正です");
    }
    auto raw = [&](const std::string &name, io::SType dtype, std::vector<size_t> dims) {
        const auto &w = at(name);
        require(w.spec.dtype == dtype && w.spec.shape == dims, "未量子化 tensor の型・形状が不正です: " + name);
    };
    auto linear = [&](const std::string &name, int k, int n, bool allow_raw = false) {
        auto it = shape.find(name);
        if (it == shape.end() && allow_raw) {
            const auto &w = at(name + ".weight");
            require(w.spec.shape == std::vector<size_t>{size_t(n), size_t(k)} &&
                        (w.spec.dtype == io::SType::BF16 || w.spec.dtype == io::SType::F32),
                    "indexer 線形の形状・型が不正です: " + name);
        } else {
            require(it != shape.end() && it->second == std::array<int, 2>{k, n},
                    "射影の形状が不正です: " + name);
        }
    };
    auto norm = [&](const std::string &name, size_t n) {
        const auto &w = at(name);
        require(w.spec.shape == std::vector<size_t>{n} &&
                    (w.spec.dtype == io::SType::BF16 || w.spec.dtype == io::SType::F32),
                "正規化係数の型・形状が不正です: " + name);
    };
    auto mlp = [&](const std::string &p, int k, int n, int out) {
        linear(p + ".gate_proj", k, n);
        linear(p + ".up_proj", k, n);
        linear(p + ".down_proj", n, out);
    };
    auto attention = [&](const std::string &p, int heads) {
        linear(p + "q_a_proj", 3072, 1024);
        linear(p + "q_b_proj", 1024, heads * 256);
        linear(p + "kv_a_proj_with_mqa", 3072, 576);
        linear(p + "kv_b_proj", 512, heads * 448);
        linear(p + "linear_gate", 3072, heads * 256);
        linear(p + "o_proj", heads * 256, 3072);
        norm(p + "q_a_layernorm.weight", 1024);
        norm(p + "kv_a_layernorm.weight", 512);
        norm(p + "learnable_sink_param", heads);
        norm(p + "post_proj_norm.weight", 3072);
    };
    raw("model.embed_tokens.weight", io::SType::BF16, {120832, 3072});
    raw("model.hc_head.hc_head_fn", io::SType::F32, {4, 12288});
    raw("model.hc_head.hc_head_base", io::SType::F32, {4});
    norm("model.norm.weight", 3072);
    require(at("model.hc_head.hc_head_scale").bytes == 4 &&
                at("model.hc_head.hc_head_scale").spec.dtype == io::SType::F32,
            "head の scale が不正です");
    linear("lm_head", 3072, 120832);
    for (int i = 0; i < 42; ++i) {
        auto p = "model.layers." + std::to_string(i) + ".";
        bool sparse = i == 0 || i % 4 == 1;
        attention(p + "self_attn.", sparse ? 32 : 64);
        norm(p + "input_layernorm.weight", 3072);
        norm(p + "post_attention_layernorm.weight", 3072);
        for (auto sub : {"hc_attn_layer.", "hc_mlp_layer."}) {
            auto h = p + sub + "hc_pre.hc_";
            raw(h + "fn", io::SType::F32, {8, 12288});
            raw(h + "scale", io::SType::F32, {2});
            raw(h + "base", io::SType::F32, {8});
        }
        if (sparse) {
            auto a = p + "self_attn.compressed_indexer.";
            linear(a + "token_key_projection", 3072, 128, true);
            linear(a + "head_weight_projection", 3072, 16, true);
            linear(a + "query_projection", 1024, 2048, true);
            for (auto key : {"key_value_projection.weight", "key_gate_projection.weight"})
                raw(a + key, io::SType::F32, {128, 128});
            raw(a + "intra_group_ape", io::SType::F32, {4, 128});
            norm(a + "key_norm.weight", 128);
            norm(a + "token_key_norm.weight", 128);
        }
        if (i == 0) {
            mlp(p + "mlp", 3072, 13824, 3072);
            norm(p + "mlp.post_fc2_norm.weight", 3072);
        } else {
            raw(p + "mlp.gate.weight", io::SType::BF16, {512, 3072});
            raw(p + "mlp.gate.e_score_correction_bias", io::SType::F32, {512});
            linear(p + "mlp.routed_expert_down_proj", 3072, 1536);
            linear(p + "mlp.routed_expert_up_proj", 1536, 3072);
            mlp(p + "mlp.shared_experts", 3072, 1536, 3072);
            norm(p + "mlp.combine_output_norm.weight", 3072);
        }
        if (i % 10 == 1 && i <= 31) {
            auto e = p + "engram_layer.";
            raw(e + "lookup_table", io::SType::I32, {120818});
            auto lut = static_cast<const int32_t *>(at(e + "lookup_table").host);
            auto bounds = std::minmax_element(lut, lut + 120818);
            require(*bounds.first >= 0 && *bounds.second == 92619, "Engram の正規化語彙が異なります");
            linear(e + "value_proj", 1536, 3072);
            for (auto name : {"value_proj.bias", "unified_pre_gamma"})
                raw(e + name, io::SType::BF16, {3072});
            for (int j = 0; j < 16; ++j)
                raw(e + "lookup_per_head_norms." + std::to_string(j) + ".weight", io::SType::BF16, {96});
            for (int j = 0; j < 4; ++j) {
                auto num = std::to_string(j);
                linear(e + "key_projs." + num, 1536, 3072);
                raw(e + "key_projs." + num + ".bias", io::SType::BF16, {3072});
                raw(e + "norm_key." + num + ".weight", io::SType::BF16, {3072});
                raw(e + "norm_query." + num + ".weight", io::SType::BF16, {3072});
                raw(e + "unified_per_hc_scales." + num, io::SType::BF16, {3072});
            }
        }
    }
    if (has_draft) {
        std::string p = "model.dfly_layers.";
        for (int i = 0; i < 5; ++i) {
            auto l = p + std::to_string(i) + ".";
            attention(l + "self_attn.", 32);
            norm(l + "input_layernorm.weight", 3072);
            norm(l + "post_attention_layernorm.weight", 3072);
            mlp(l + "mlp", 3072, 9216, 3072);
        }
        for (auto name : {"mtp_fusion.enorm", "mtp_fusion.hnorm", "hidden_correction.embed_norm",
                          "hidden_correction.hidden_norm", "final_norm"})
            norm(p + name + ".weight", 3072);
        raw(p + "mtp_fusion.gate", io::SType::BF16, {3072});
        linear(p + "mtp_fusion.eh_proj", 6144, 3072);
        mlp(p + "hidden_correction", 6144, 3072, 3072);
    }
    for (const auto &[name, dims] : shape)
        at(name + ".trellis");
    for (int i = 1; i < 42; ++i)
        for (int e = 0; e < 512; ++e) {
            auto stem = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(e) + ".";
            int bits = at(stem + "gate_proj.trellis").spec.shape[2] / 16;
            for (auto name : {"up_proj", "down_proj"})
                require(at(stem + name + ".trellis").spec.shape[2] == size_t(bits * 16),
                        "expert の 3 射影の bpw が一致しません");
            expert_bits[i - 1][e] = bits;
        }
    if (mixed) {
        const auto plan = read_json(model / "mixed_plan.json");
        const auto report = read_json(model / "conversion_report.json");
        size_t count = 0;
        for (int i = 0; i < 41; ++i) {
            std::array<uint8_t, 512> expected;
            expected.fill(4);
            for (const auto &id : plan.at(std::to_string(i + 1))) {
                int e = id.get<int>();
                require(e >= 0 && e < 512 && expected[e] == 4, "mixed 計画に不正な expert ID があります");
                expected[e] = 3;
                ++count;
            }
            require(expert_bits[i] == expected, "mixed 計画と実 expert のビット幅が一致しません");
        }
        const auto elements = report.at("decoder_quantized_weights").get<uint64_t>();
        require(elements > 0 && count == report.at("mixed_3bit_experts").get<size_t>(),
                "mixed 変換の件数が report と一致しません");
        const double bpw = 4.0 - double(count * 3ull * 1536 * 1536) / double(elements);
        require(std::abs(bpw - 3.5) <= 0.0001 &&
                    std::abs(bpw - report.at("decoder_code_bpw").get<double>()) < 1e-8,
                "実重みに基づく平均 bpw が一致しません");
    }
}
void Checkpoint::load(const Path &directory, bool draft) {
    auto index = read_json(directory / "model.safetensors.index.json").at("weight_map");
    std::map<std::string, std::vector<std::string>> shards;
    for (const auto &[key, file] : index.items()) {
        require(key.starts_with("model.dfly_layers.") == draft, "本体・DFly の index が混在しています");
        if (key.ends_with(".multi_head_embedding.embedding.weight") ||
            key.ends_with(".multi_head_embedding.embedding.weight.scale"))
            continue;
        shards[file.get<std::string>()].push_back(key);
    }
    for (const auto &[file, keys] : shards) {
        require(Path(file).filename() == file, "shard ファイル名が不正です");
        auto reader = take(io::SafetensorsReader::open((directory / file).string()));
        size_t first = upload_order.size();
        for (const auto &key : keys) {
            int rank = 0;
            if (key.find(".mlp.experts.") != std::string::npos) {
                int layer = 0, e = 0;
                require(sscanf(key.c_str(), "model.layers.%d.mlp.experts.%d.", &layer, &e) == 2 && layer >= 1 &&
                            layer <= 41 && e >= 0 && e < 512,
                        "expert 名が不正です");
                rank = owners[layer - 1][e];
            }
            if (key.ends_with(".mul1") || key.ends_with(".lookup_table"))
                rank = -1;
            size_t length;
            auto ptr = take(reader.tensor_data(key, length));
            auto spec = take(reader.tensor_spec(key));
            auto [entry, inserted] = weights.emplace(key, Weight{std::move(spec), ptr, length, rank, 0});
            require(inserted, "tensor 名の重複");
            if (rank >= 0)
                upload_order.push_back(&entry->second);
        }
        std::sort(upload_order.begin() + first, upload_order.end(),
                  [](const Weight *a, const Weight *b) { return a->spec.data_begin < b->spec.data_begin; });
        for (size_t i = first; i < upload_order.size(); ++i) {
            auto &w = *upload_order[i];
            w.offset = (bytes[w.rank] + 15) & ~size_t(15);
            bytes[w.rank] = w.offset + w.bytes;
        }
        files.push_back(std::move(reader));
    }
}
void Checkpoint::relocate(const Owners &layout) {
    bytes.fill(0);
    for (auto &[key, w] : weights) {
        if (w.rank < 0)
            continue;
        if (key.find(".mlp.experts.") != std::string::npos) {
            int layer = 0, e = 0;
            if (sscanf(key.c_str(), "model.layers.%d.mlp.experts.%d.", &layer, &e) != 2)
                throw std::runtime_error("expert 名の解析に失敗しました");
            w.rank = layout.at(layer - 1).at(e);
        }
    }
    for (auto *w : upload_order) {
        w->offset = (bytes[w->rank] + 15) & ~size_t(15);
        bytes[w->rank] = w->offset + w->bytes;
    }
    owners = layout;
}

}
