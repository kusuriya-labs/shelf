#include <shelf/models/qwen4exp/model/qwen4exp_weights.h>
#include <shelf/quantization/bundle/psq5_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ps::qwen4exp {
namespace {

namespace fw = ps::weights;

std::string layer_prefix(std::size_t layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

using QuantReader = ps::quantization::bundle::QuantizedModelReader;

Status load_matrix(const fw::SafetensorsCollection& collection, const QuantReader* reader,
                   const std::string& name, gpu::GpuArena& arena, hipStream_t stream,
                   const fw::WeightLoadOptions& options, fw::MatrixWeight& dst) {
    if (reader != nullptr) {
        auto m = fw::load_quantized_matrix(*reader, name, arena, stream, options);
        if (!m.ok()) return m.status();
        dst = m.release();
        return Status::make_ok();
    }
    auto m = fw::load_bf16_matrix(collection, name, arena, stream, options);
    if (!m.ok()) return m.status();
    dst = m.release();
    return Status::make_ok();
}

Status load_tensor(const fw::SafetensorsCollection& collection, const QuantReader* reader,
                   const std::string& name, gpu::GpuArena& arena, hipStream_t stream,
                   gpu::Tensor& dst) {
    if (reader != nullptr) {
        auto t = fw::load_quantized_small(*reader, name, arena, stream);
        if (!t.ok()) return t.status();
        dst = t.release();
        return Status::make_ok();
    }
    auto t = fw::load_bf16_tensor(collection, name, arena, stream);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

Status load_i64_vector(const fw::SafetensorsCollection& collection, const std::string& name,
                       std::vector<std::int64_t>& dst) {
    auto spec = collection.tensor_spec(name);
    if (!spec.ok()) return spec.status();
    if (spec.value().dtype != io::SType::I64) {
        return Status::invalid_argument(
            ("PLE metadata must be I64: " + name).c_str(), __FILE__, __LINE__);
    }
    std::size_t bytes = 0;
    auto data = collection.tensor_data(name, bytes);
    if (!data.ok()) return data.status();
    const std::size_t count = bytes / sizeof(std::int64_t);
    if (count * sizeof(std::int64_t) != bytes) {
        return Status::invalid_argument(
            ("PLE metadata byte count mismatch: " + name).c_str(), __FILE__, __LINE__);
    }
    dst.resize(count);
    std::memcpy(dst.data(), data.value(), bytes);
    return Status::make_ok();
}

Status load_hyper(const fw::SafetensorsCollection& collection, const QuantReader* reader,
                  const std::string& prefix, gpu::GpuArena& arena, hipStream_t stream,
                  const fw::WeightLoadOptions& options, bool with_inject,
                  Qwen4ExpHyperWeights& dst) {
    Status st = load_tensor(collection, reader, prefix + "hc_norm.weight", arena, stream, dst.hc_norm);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "input_mix_weight_down.weight", arena, stream,
                     options, dst.mix_down);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "input_mix_weight_up.weight", arena, stream,
                     options, dst.mix_up);
    if (!st.ok()) return st;
    if (with_inject) {
        st = load_matrix(collection, reader, prefix + "block_inject_weight.weight", arena, stream,
                         options, dst.block_inject);
        if (!st.ok()) return st;
        dst.has_block_inject = true;
    }
    return Status::make_ok();
}

Status load_gdn(const fw::SafetensorsCollection& collection, const QuantReader* reader,
                const std::string& prefix, gpu::GpuArena& arena, hipStream_t stream,
                const fw::WeightLoadOptions& options, Qwen4ExpGdnWeights& dst) {
    Status st = load_matrix(collection, reader, prefix + "in_proj_qkv.weight", arena, stream, options,
                            dst.qkv);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "in_proj_z.weight", arena, stream, options, dst.z);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "in_proj_b.weight", arena, stream, options, dst.b);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "in_proj_a.weight", arena, stream, options, dst.a);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "out_proj.weight", arena, stream, options, dst.out);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "conv1d.weight", arena, stream, dst.conv1d);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "norm.weight", arena, stream, dst.norm);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "dt_bias", arena, stream, dst.dt_bias);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "A_log", arena, stream, dst.a_log);
    if (!st.ok()) return st;
    return Status::make_ok();
}

Status load_qsa(const fw::SafetensorsCollection& collection, const QuantReader* reader,
                const std::string& prefix, gpu::GpuArena& arena, hipStream_t stream,
                const fw::WeightLoadOptions& options, Qwen4ExpQsaWeights& dst) {
    Status st = load_matrix(collection, reader, prefix + "q_proj.weight", arena, stream, options, dst.q);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "k_proj.weight", arena, stream, options, dst.k);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "v_proj.weight", arena, stream, options, dst.v);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "o_proj.weight", arena, stream, options, dst.o);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "q_norm.weight", arena, stream, dst.q_norm);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "k_norm.weight", arena, stream, dst.k_norm);
    if (!st.ok()) return st;

    const std::string ip = prefix + "indexer.";
    if (collection.tensor_spec(ip + "index_qk_proj.weight").ok()) {
        st = load_matrix(collection, reader, ip + "index_qk_proj.weight", arena, stream, options,
                         dst.indexer.index_qk);
        if (!st.ok()) return st;
        st = load_tensor(collection, reader, ip + "q_layernorm.weight", arena, stream,
                         dst.indexer.q_layernorm);
        if (!st.ok()) return st;
        st = load_tensor(collection, reader, ip + "k_layernorm.weight", arena, stream,
                         dst.indexer.k_layernorm);
        if (!st.ok()) return st;
        dst.indexer.present = true;
    }
    return Status::make_ok();
}

Status load_moe_quantized(
    const ps::quantization::bundle::QuantizedModelReader& reader, const std::string& prefix,
    std::uint32_t expert_count, gpu::GpuArena& arena, hipStream_t stream,
    const fw::WeightLoadOptions& options, bool load_experts, Qwen4ExpMoeWeights& dst) {
    dst.expert_gate_up_tensor = prefix + "experts.gate_up_proj";
    dst.expert_down_tensor = prefix + "experts.down_proj";
    dst.expert_count = expert_count;
    if (!load_experts || expert_count == 0u) return Status::make_ok();

    const std::string names[2] = {dst.expert_gate_up_tensor, dst.expert_down_tensor};
    std::uint64_t rows_each[2] = {0, 0};
    std::uint64_t cols_each[2] = {0, 0};
    for (int i = 0; i < 2; ++i) {
        auto v = reader.resolve(names[i]);
        if (!v.ok()) return v.status();
        if (v.value().logical_shape.size() != 2) {
            return Status::invalid_argument(
                ("fused expert tensor must be rank-2 logical: " + names[i]).c_str(),
                __FILE__, __LINE__);
        }
        rows_each[i] = static_cast<std::uint64_t>(v.value().logical_shape[0]);
        cols_each[i] = static_cast<std::uint64_t>(v.value().logical_shape[1]);
        if (rows_each[i] % expert_count != 0u) {
            return Status::invalid_argument(
                ("fused expert rows not divisible by expert count: " + names[i]).c_str(),
                __FILE__, __LINE__);
        }
    }

    const std::uint64_t gate_rows = rows_each[0] / expert_count;
    const std::uint64_t down_rows = rows_each[1] / expert_count;
    dst.expert_gate_up.resize(expert_count);
    dst.expert_down.resize(expert_count);
    for (std::uint32_t e = 0; e < expert_count; ++e) {
        auto g = fw::load_quantized_psq5_matrix_axis_slice(
            reader, names[0], static_cast<std::uint64_t>(e) * gate_rows, gate_rows, arena,
            stream, options);
        if (!g.ok()) return g.status();
        dst.expert_gate_up[e] = g.release();
        auto d = fw::load_quantized_psq5_matrix_axis_slice(
            reader, names[1], static_cast<std::uint64_t>(e) * down_rows, down_rows, arena,
            stream, options);
        if (!d.ok()) return d.status();
        dst.expert_down[e] = d.release();
    }
    return Status::make_ok();
}

Status load_moe(const fw::SafetensorsCollection& collection, const QuantReader* reader,
                const std::string& prefix, gpu::GpuArena& arena, hipStream_t stream,
                const fw::WeightLoadOptions& options, bool load_experts,
                Qwen4ExpMoeWeights& dst, std::uint32_t expert_count = 0u) {
    Status st = load_matrix(collection, reader, prefix + "gate.weight", arena, stream, options,
                            dst.router);
    if (!st.ok()) return st;
    const std::string sp = prefix + "shared_expert.";
    st = load_matrix(collection, reader, sp + "gate_proj.weight", arena, stream, options,
                     dst.shared.gate);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, sp + "up_proj.weight", arena, stream, options, dst.shared.up);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, sp + "down_proj.weight", arena, stream, options,
                     dst.shared.down);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "shared_expert_gate.weight", arena, stream,
                     dst.shared.gate_scale);
    if (!st.ok()) return st;

    if (reader != nullptr) {
        return load_moe_quantized(*reader, prefix, expert_count, arena, stream, options,
                                  load_experts, dst);
    }
    dst.expert_gate_up_tensor = prefix + "experts.gate_up_proj";
    dst.expert_down_tensor = prefix + "experts.down_proj";
    auto spec = collection.tensor_spec(dst.expert_gate_up_tensor);
    if (!spec.ok()) return spec.status();
    if (spec.value().shape.size() != 3) {
        return Status::invalid_argument(
            ("fused expert tensor must be rank-3: " + dst.expert_gate_up_tensor).c_str(),
            __FILE__, __LINE__);
    }
    dst.expert_count = static_cast<std::uint32_t>(spec.value().shape[0]);

    if (load_experts && dst.expert_count > 0) {
        auto down_spec = collection.tensor_spec(dst.expert_down_tensor);
        if (!down_spec.ok()) return down_spec.status();
        if (down_spec.value().shape.size() != 3 ||
            down_spec.value().shape[0] != spec.value().shape[0]) {
            return Status::invalid_argument(
                ("fused expert down tensor must match gate_up: " +
                 dst.expert_down_tensor)
                    .c_str(),
                __FILE__, __LINE__);
        }
        auto gate_up = fw::load_bf16_matrix_axis_slice(
            collection, dst.expert_gate_up_tensor, 0, dst.expert_count, arena, stream);
        if (!gate_up.ok()) return gate_up.status();
        auto down = fw::load_bf16_matrix_axis_slice(
            collection, dst.expert_down_tensor, 0, dst.expert_count, arena, stream);
        if (!down.ok()) return down.status();

        const std::size_t gate_up_rows = spec.value().shape[1];
        const std::size_t gate_up_cols = spec.value().shape[2];
        const std::size_t down_rows = down_spec.value().shape[1];
        const std::size_t down_cols = down_spec.value().shape[2];
        dst.expert_gate_up.resize(dst.expert_count);
        dst.expert_down.resize(dst.expert_count);
        for (std::uint32_t e = 0; e < dst.expert_count; ++e) {
            auto gate_slice = gate_up.value().bf16.slice(
                0, static_cast<std::size_t>(e) * gate_up_rows, gate_up_rows);
            if (!gate_slice.ok()) return gate_slice.status();
            fw::MatrixWeight gm;
            gm.bf16 = gate_slice.release();
            gm.rows = static_cast<std::uint32_t>(gate_up_rows);
            gm.cols = static_cast<std::uint32_t>(gate_up_cols);
            gm.k_padded = gm.cols;
            fw::apply_default_compute_spec(gm);
            dst.expert_gate_up[e] = std::move(gm);

            auto down_slice = down.value().bf16.slice(
                0, static_cast<std::size_t>(e) * down_rows, down_rows);
            if (!down_slice.ok()) return down_slice.status();
            fw::MatrixWeight dm;
            dm.bf16 = down_slice.release();
            dm.rows = static_cast<std::uint32_t>(down_rows);
            dm.cols = static_cast<std::uint32_t>(down_cols);
            dm.k_padded = dm.cols;
            fw::apply_default_compute_spec(dm);
            dst.expert_down[e] = std::move(dm);
        }
    }
    return Status::make_ok();
}

Status load_ple(const fw::SafetensorsCollection& collection, const QuantReader* reader,
                const std::string& prefix, gpu::GpuArena& arena, hipStream_t stream,
                const fw::WeightLoadOptions& options, Qwen4ExpPleWeights& dst) {
    Status st = load_matrix(collection, reader, prefix + "key_proj.weight", arena, stream, options,
                            dst.key_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, reader, prefix + "value_proj.weight", arena, stream, options,
                     dst.value_proj);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "conv1d.weight", arena, stream, dst.conv1d);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "norm_key.weight", arena, stream, dst.norm_key);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "norm_query.weight", arena, stream, dst.norm_query);
    if (!st.ok()) return st;
    st = load_tensor(collection, reader, prefix + "norm_conv.weight", arena, stream, dst.norm_conv);
    if (!st.ok()) return st;

    const std::string ep = prefix + "ple_embedding.";
    st = load_i64_vector(collection, ep + "layer_multipliers", dst.layer_multipliers);
    if (!st.ok()) return st;
    st = load_i64_vector(collection, ep + "ngram_heads_vocab_sizes", dst.vocab_sizes);
    if (!st.ok()) return st;
    st = load_i64_vector(collection, ep + "ngram_heads_offsets", dst.offsets);
    if (!st.ok()) return st;

    dst.present = true;
    return Status::make_ok();
}

std::vector<std::string> qwen4exp_quantized_logical_names(const Qwen4ExpTextConfig& config,
                                                           std::uint32_t layer_begin,
                                                           std::uint32_t layer_end) {
    std::vector<std::string> names;
    names.reserve(1024);
    names.push_back("model.language_model.embed_tokens.weight");
    names.push_back("lm_head.weight");
    const std::string mixer = "model.language_model.hyper_connection_mixer.";
    names.push_back(mixer + "hc_norm.weight");
    names.push_back(mixer + "input_mix_weight_down.weight");
    names.push_back(mixer + "input_mix_weight_up.weight");

    const std::size_t begin = std::min<std::size_t>(layer_begin, config.num_hidden_layers);
    const std::size_t end = layer_end == 0
                                ? config.num_hidden_layers
                                : std::min<std::size_t>(layer_end, config.num_hidden_layers);
    for (std::size_t l = begin; l < end; ++l) {
        const std::string p = layer_prefix(l);
        for (const std::string& hp : {"attn_hyper_connection.", "mlp_hyper_connection."}) {
            names.push_back(p + hp + "hc_norm.weight");
            names.push_back(p + hp + "input_mix_weight_down.weight");
            names.push_back(p + hp + "input_mix_weight_up.weight");
            names.push_back(p + hp + "block_inject_weight.weight");
        }
        if (config.layer_is_qsa(l)) {
            for (const std::string& q : {"q_proj.weight", "k_proj.weight", "v_proj.weight",
                                         "o_proj.weight", "q_norm.weight", "k_norm.weight"}) {
                names.push_back(p + "self_attn." + q);
            }
            const std::string ip = p + "self_attn.indexer.";
            names.push_back(ip + "index_qk_proj.weight");
            names.push_back(ip + "q_layernorm.weight");
            names.push_back(ip + "k_layernorm.weight");
        } else {
            for (const std::string& g : {"in_proj_qkv.weight", "in_proj_z.weight",
                                         "in_proj_b.weight", "in_proj_a.weight",
                                         "out_proj.weight", "conv1d.weight", "norm.weight",
                                         "dt_bias", "A_log"}) {
                names.push_back(p + "linear_attn." + g);
            }
        }
        names.push_back(p + "mlp.gate.weight");
        names.push_back(p + "mlp.shared_expert.gate_proj.weight");
        names.push_back(p + "mlp.shared_expert.up_proj.weight");
        names.push_back(p + "mlp.shared_expert.down_proj.weight");
        names.push_back(p + "mlp.shared_expert_gate.weight");
        names.push_back(p + "mlp.experts.gate_up_proj");
        names.push_back(p + "mlp.experts.down_proj");
        if (l == config.ple_layer_index) {
            names.push_back(p + "ple.key_proj.weight");
            names.push_back(p + "ple.value_proj.weight");
            names.push_back(p + "ple.conv1d.weight");
            names.push_back(p + "ple.norm_key.weight");
            names.push_back(p + "ple.norm_query.weight");
            names.push_back(p + "ple.norm_conv.weight");
        }
    }
    return names;
}

}  // namespace

Result<Qwen4ExpModelWeights> load_qwen4exp_weights_quantized(
    const std::string& model_dir, gpu::GpuArena& arena, hipStream_t stream,
    const Qwen4ExpLoadOptions& options) {
    auto config_result = read_qwen4exp_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    const Qwen4ExpTextConfig config = config_result.value();

    auto collection_result = fw::SafetensorsCollection::open(model_dir);
    if (!collection_result.ok()) return collection_result.status();
    const fw::SafetensorsCollection& collection = collection_result.value();

    const bool verify = options.verify_quantized_payload_crc;
    const std::size_t load_begin =
        std::min<std::size_t>(options.layer_begin, config.num_hidden_layers);
    const std::size_t load_end = options.layer_end == 0
                                     ? config.num_hidden_layers
                                     : std::min<std::size_t>(options.layer_end,
                                                             config.num_hidden_layers);
    std::vector<std::string> names = qwen4exp_quantized_logical_names(
        config, options.layer_begin, options.layer_end);
    auto source = ps::quantization::bundle::synthesize_psq5_quantized_source(
        model_dir, names, verify, "qwen4exp");
    if (!source.ok()) return source.status();
    auto reader = ps::quantization::bundle::QuantizedModelReader::open_hf_source(
        source.release(), verify);
    if (!reader.ok()) return reader.status();
    QuantReader rd = reader.release();

    Qwen4ExpModelWeights w;
    Status st = load_matrix(collection, &rd, "model.language_model.embed_tokens.weight",
                            arena, stream, options.weights, w.embed_tokens);
    if (!st.ok()) return st;
    st = load_matrix(collection, &rd, "lm_head.weight", arena, stream, options.weights,
                     w.lm_head);
    if (!st.ok()) return st;
    w.lm_head_tied = config.tie_word_embeddings &&
                     w.embed_tokens.rows == w.lm_head.rows;

    st = load_hyper(collection, &rd, "model.language_model.hyper_connection_mixer.", arena,
                    stream, options.weights, false, w.final_hyper);
    if (!st.ok()) return st;

    w.layers.resize(config.num_hidden_layers);
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        Qwen4ExpLayerWeights& lw = w.layers[layer];
        if (layer < load_begin || layer >= load_end) {
            lw.is_gdn = !config.layer_is_qsa(layer);
            continue;
        }
        const std::string prefix = layer_prefix(layer);
        lw.is_gdn = !config.layer_is_qsa(layer);

        st = load_hyper(collection, &rd, prefix + "attn_hyper_connection.", arena, stream,
                        options.weights, true, lw.attn_hyper);
        if (!st.ok()) return st;
        st = load_hyper(collection, &rd, prefix + "mlp_hyper_connection.", arena, stream,
                        options.weights, true, lw.mlp_hyper);
        if (!st.ok()) return st;

        if (lw.is_gdn) {
            st = load_gdn(collection, &rd, prefix + "linear_attn.", arena, stream,
                          options.weights, lw.gdn);
            if (!st.ok()) return st;
        } else {
            st = load_qsa(collection, &rd, prefix + "self_attn.", arena, stream,
                          options.weights, lw.qsa);
            if (!st.ok()) return st;
        }

        st = load_moe(collection, &rd, prefix + "mlp.", arena, stream, options.weights,
                      options.load_expert_weights, lw.moe,
                      static_cast<std::uint32_t>(config.num_experts));
        if (!st.ok()) return st;

        if (layer == config.ple_layer_index) {
            st = load_ple(collection, &rd, prefix + "ple.", arena, stream, options.weights,
                          lw.ple);
            if (!st.ok()) return st;
        }
    }

    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error(
            "hipStreamSynchronize after quantized weight load",
            hipGetErrorString(sync_err), __FILE__, __LINE__);
    }
    return w;
}

Result<Qwen4ExpTextConfig> load_qwen4exp_config(const std::string& model_dir) {
    return read_qwen4exp_text_config(model_dir);
}

Result<Qwen4ExpModelWeights> load_qwen4exp_weights_bf16(
    const std::string& model_dir, gpu::GpuArena& arena, hipStream_t stream,
    const Qwen4ExpLoadOptions& options) {
    auto config_result = read_qwen4exp_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    const Qwen4ExpTextConfig config = config_result.value();

    auto collection_result = fw::SafetensorsCollection::open(model_dir);
    if (!collection_result.ok()) return collection_result.status();
    const fw::SafetensorsCollection& collection = collection_result.value();

    Qwen4ExpModelWeights w;
    Status st = load_matrix(collection, nullptr, "model.language_model.embed_tokens.weight", arena,
                            stream, options.weights, w.embed_tokens);
    if (!st.ok()) return st;
    st = load_matrix(collection, nullptr, "lm_head.weight", arena, stream, options.weights, w.lm_head);
    if (!st.ok()) return st;
    w.lm_head_tied = config.tie_word_embeddings &&
                     w.embed_tokens.rows == w.lm_head.rows;

    st = load_hyper(collection, nullptr, "model.language_model.hyper_connection_mixer.", arena,
                    stream, options.weights, false, w.final_hyper);
    if (!st.ok()) return st;

    w.layers.resize(config.num_hidden_layers);
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        Qwen4ExpLayerWeights& lw = w.layers[layer];
        const std::string prefix = layer_prefix(layer);
        lw.is_gdn = !config.layer_is_qsa(layer);

        st = load_hyper(collection, nullptr, prefix + "attn_hyper_connection.", arena, stream,
                        options.weights, true, lw.attn_hyper);
        if (!st.ok()) return st;
        st = load_hyper(collection, nullptr, prefix + "mlp_hyper_connection.", arena, stream,
                        options.weights, true, lw.mlp_hyper);
        if (!st.ok()) return st;

        if (lw.is_gdn) {
            st = load_gdn(collection, nullptr, prefix + "linear_attn.", arena, stream,
                          options.weights, lw.gdn);
            if (!st.ok()) return st;
        } else {
            st = load_qsa(collection, nullptr, prefix + "self_attn.", arena, stream, options.weights,
                          lw.qsa);
            if (!st.ok()) return st;
        }

        st = load_moe(collection, nullptr, prefix + "mlp.", arena, stream, options.weights,
                      options.load_expert_weights, lw.moe);
        if (!st.ok()) return st;

        if (layer == config.ple_layer_index) {
            st = load_ple(collection, nullptr, prefix + "ple.", arena, stream, options.weights,
                          lw.ple);
            if (!st.ok()) return st;
        }
    }

    return w;
}

}  // namespace ps::qwen4exp
