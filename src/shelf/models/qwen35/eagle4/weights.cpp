#include <shelf/models/qwen35/eagle4/weights.h>
#include <shelf/models/qwen35/eagle4/tensor_parallel_plan.h>
#include <shelf/io/safetensors_reader.h>

#include <set>
#include <utility>

namespace ps::qwen35::eagle4 {

namespace {

std::string shape_to_string(const std::vector<std::size_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

void push_tensor(
    std::vector<Eagle4ExpectedTensor>& out,
    std::string name,
    std::vector<std::size_t> shape) {
    out.push_back(Eagle4ExpectedTensor{std::move(name), std::move(shape)});
}

Status load_matrix(
    const ps::weights::SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const ps::weights::WeightLoadOptions& options,
    ps::weights::MatrixWeight& dst) {
    auto m = ps::weights::load_bf16_matrix(collection, name, arena, stream, options);
    if (!m.ok()) return m.status();
    dst = m.release();
    return Status::make_ok();
}

Status load_vector(
    const ps::weights::SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    gpu::Tensor& dst) {
    auto t = ps::weights::load_bf16_tensor(collection, name, arena, stream);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

Status load_eagle4_weights_from(
    const ps::weights::SafetensorsCollection& collection,
    const Eagle4Config& config,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Eagle4WeightLoadOptions& options,
    const ps::weights::TensorPartitionPlan* partition_plan,
    Eagle4Weights& weights) {
    weights.layers.resize(config.num_hidden_layers);
    ps::weights::WeightLoadOptions matrix_options;
    matrix_options.preshuffle = options.preshuffle;
    matrix_options.partition_plan = partition_plan;

    Status st = load_matrix(collection, "fc.weight", arena, stream, matrix_options, weights.fc);
    if (!st.ok()) return st;
    for (std::size_t i = 0; i < kEagle4AuxLayers; ++i) {
        st = load_vector(collection, "fc_norm." + std::to_string(i) + ".weight", arena,
                         stream, weights.fc_norm_weight[i]);
        if (!st.ok()) return st;
    }
    st = load_vector(collection, "norm.weight", arena, stream, weights.final_norm_weight);
    if (!st.ok()) return st;
    if (options.load_embed_tokens) {
        st = load_matrix(collection, "embed_tokens.weight", arena, stream, matrix_options,
                         weights.embed_tokens);
        if (!st.ok()) return st;
    }
    if (options.load_lm_head) {
        st = load_matrix(collection, "lm_head.weight", arena, stream, matrix_options,
                         weights.candidate.lm_head);
        if (!st.ok()) return st;
    }
    st = load_matrix(collection, "efficient_lm_head.weight", arena, stream, matrix_options,
                     weights.candidate.efficient_lm_head);
    if (!st.ok()) return st;
    st = load_matrix(collection, "efficient_lm_head_down_proj.weight", arena, stream,
                     matrix_options, weights.candidate.efficient_lm_head_down_proj);
    if (!st.ok()) return st;

    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string pre = "layers." + std::to_string(layer) + ".";
        Eagle4LayerWeights& lw = weights.layers[layer];

        st = load_vector(collection, pre + "input_layernorm.weight", arena, stream,
                         lw.input_layernorm_weight);
        if (!st.ok()) return st;
        st = load_vector(collection, pre + "hidden_norm.weight", arena, stream,
                         lw.hidden_norm_weight);
        if (!st.ok()) return st;
        st = load_vector(collection, pre + "post_attention_layernorm.weight", arena, stream,
                         lw.post_attention_layernorm_weight);
        if (!st.ok()) return st;

        st = load_matrix(collection, pre + "self_attn.q_proj.weight", arena, stream, matrix_options,
                         lw.attn_q_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, pre + "self_attn.k_proj.weight", arena, stream, matrix_options,
                         lw.attn_k_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, pre + "self_attn.v_proj.weight", arena, stream, matrix_options,
                         lw.attn_v_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, pre + "self_attn.o_proj.weight", arena, stream, matrix_options,
                         lw.attn_o_proj);
        if (!st.ok()) return st;

        st = load_matrix(collection, pre + "mlp.gate_proj.weight", arena, stream, matrix_options,
                         lw.mlp_gate_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, pre + "mlp.up_proj.weight", arena, stream, matrix_options,
                         lw.mlp_up_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, pre + "mlp.down_proj.weight", arena, stream, matrix_options,
                         lw.mlp_down_proj);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

Status validate_loaded(const Eagle4Weights& weights) {
    auto check = [](const ps::weights::MatrixWeight& w, const std::string& name) -> Status {
        if (w.encoding != ps::weights::MatrixEncoding::Bf16) {
            return Status::invalid_argument(
                ("EAGLE-4 matrix is not BF16: " + name).c_str(), __FILE__, __LINE__);
        }
        if (w.bf16.ndim() == 0) {
            return Status::invalid_argument(
                ("EAGLE-4 matrix payload is missing: " + name).c_str(), __FILE__, __LINE__);
        }
        return Status::make_ok();
    };
    Status st = check(weights.fc, "fc.weight");
    if (!st.ok()) return st;
    if (weights.embed_tokens.rows != 0u) {
        st = check(weights.embed_tokens, "embed_tokens.weight");
        if (!st.ok()) return st;
    }
    if (weights.candidate.lm_head.rows != 0u) {
        st = check(weights.candidate.lm_head, "lm_head.weight");
        if (!st.ok()) return st;
    }
    st = check(weights.candidate.efficient_lm_head, "efficient_lm_head.weight");
    if (!st.ok()) return st;
    st = check(weights.candidate.efficient_lm_head_down_proj,
               "efficient_lm_head_down_proj.weight");
    if (!st.ok()) return st;
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const Eagle4LayerWeights& lw = weights.layers[layer];
        const ps::weights::MatrixWeight* matrices[] = {
            &lw.attn_q_proj, &lw.attn_k_proj, &lw.attn_v_proj, &lw.attn_o_proj,
            &lw.mlp_gate_proj, &lw.mlp_up_proj, &lw.mlp_down_proj};
        const char* names[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                               "mlp.down_proj"};
        for (std::size_t i = 0; i < 7; ++i) {
            st = check(*matrices[i], std::to_string(layer) + "." + names[i]);
            if (!st.ok()) return st;
        }
    }
    return Status::make_ok();
}

}  // namespace

std::vector<Eagle4ExpectedTensor> eagle4_expected_tensors(const Eagle4Config& config) {
    std::vector<Eagle4ExpectedTensor> out;
    out.reserve(16 + 10 * config.num_hidden_layers);

    const std::size_t h = config.hidden_size;
    const std::size_t th = config.target_hidden_size;
    const std::size_t i = config.intermediate_size;
    const std::size_t r = config.candidate_rank;
    const std::size_t v = config.vocab_size;
    const std::size_t two_h = 2 * h;

    push_tensor(out, "fc.weight", {h, config.tap_feature_size()});
    for (std::size_t n = 0; n < kEagle4AuxLayers; ++n) {
        push_tensor(out, "fc_norm." + std::to_string(n) + ".weight", {th});
    }
    push_tensor(out, "norm.weight", {h});
    push_tensor(out, "embed_tokens.weight", {v, h});
    push_tensor(out, "lm_head.weight", {v, h});
    push_tensor(out, "efficient_lm_head.weight", {v, r});
    push_tensor(out, "efficient_lm_head_down_proj.weight", {r, h});

    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string pre = "layers." + std::to_string(layer) + ".";
        push_tensor(out, pre + "input_layernorm.weight", {h});
        push_tensor(out, pre + "hidden_norm.weight", {h});
        push_tensor(out, pre + "post_attention_layernorm.weight", {h});
        push_tensor(out, pre + "self_attn.q_proj.weight", {h, two_h});
        push_tensor(out, pre + "self_attn.k_proj.weight", {h, two_h});
        push_tensor(out, pre + "self_attn.v_proj.weight", {h, two_h});
        push_tensor(out, pre + "self_attn.o_proj.weight", {h, h});
        push_tensor(out, pre + "mlp.gate_proj.weight", {i, h});
        push_tensor(out, pre + "mlp.up_proj.weight", {i, h});
        push_tensor(out, pre + "mlp.down_proj.weight", {h, i});
    }

    return out;
}

Status validate_eagle4_tensor_contract(
    const std::string& model_dir,
    const Eagle4Config& config) {
    auto collection_result = ps::weights::SafetensorsCollection::open(model_dir);
    if (!collection_result.ok()) return collection_result.status();
    ps::weights::SafetensorsCollection collection = collection_result.release();

    auto names_result = collection.tensor_names();
    if (!names_result.ok()) return names_result.status();
    const std::vector<std::string>& names = names_result.value();

    const std::vector<Eagle4ExpectedTensor> expected = eagle4_expected_tensors(config);
    std::set<std::string> expected_names;
    for (const Eagle4ExpectedTensor& e : expected) expected_names.insert(e.name);
    const std::set<std::string> actual_names(names.begin(), names.end());

    std::string missing;
    for (const Eagle4ExpectedTensor& e : expected) {
        if (actual_names.find(e.name) == actual_names.end()) {
            if (!missing.empty()) missing += ", ";
            missing += e.name;
        }
    }
    if (!missing.empty()) {
        return Status::invalid_argument(
            ("EAGLE-4 checkpoint is missing tensors: " + missing).c_str(), __FILE__,
            __LINE__);
    }

    std::string unexpected;
    for (const std::string& n : names) {
        if (expected_names.find(n) == expected_names.end()) {
            if (!unexpected.empty()) unexpected += ", ";
            unexpected += n;
        }
    }
    if (!unexpected.empty()) {
        return Status::invalid_argument(
            ("EAGLE-4 checkpoint has unexpected tensors: " + unexpected).c_str(), __FILE__,
            __LINE__);
    }

    for (const Eagle4ExpectedTensor& e : expected) {
        auto spec_result = collection.tensor_spec(e.name);
        if (!spec_result.ok()) return spec_result.status();
        const ps::io::StTensorSpec& spec = spec_result.value();
        if (spec.dtype != ps::io::SType::BF16) {
            return Status::invalid_argument(
                ("EAGLE-4 tensor '" + e.name + "' must be BF16").c_str(), __FILE__,
                __LINE__);
        }
        if (spec.shape != e.shape) {
            return Status::invalid_argument(
                ("EAGLE-4 tensor '" + e.name + "' has shape " + shape_to_string(spec.shape) +
                 ", expected " + shape_to_string(e.shape))
                    .c_str(),
                __FILE__, __LINE__);
        }
    }

    return Status::make_ok();
}

Result<Eagle4Weights> load_eagle4_weights(
    const std::string& model_dir,
    const Eagle4Config& config,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Eagle4WeightLoadOptions& options) {
    Eagle4Weights weights;

    Status contract = validate_eagle4_tensor_contract(model_dir, config);
    if (!contract.ok()) return contract;

    auto collection_result = ps::weights::SafetensorsCollection::open(model_dir);
    if (!collection_result.ok()) return collection_result.status();
    ps::weights::SafetensorsCollection collection = collection_result.release();

    ps::weights::TensorPartitionPlan partition_plan;
    const ps::weights::TensorPartitionPlan* partition = nullptr;
    if (options.tp_size > 1) {
        auto plan_result =
            build_eagle4_tensor_partition_plan(config, options.tp_size, options.tp_rank);
        if (!plan_result.ok()) return plan_result.status();
        partition_plan = plan_result.release();
        partition = &partition_plan;
    }

    Status st = load_eagle4_weights_from(collection, config, arena, stream, options,
                                         partition, weights);
    if (!st.ok()) return st;
    st = validate_loaded(weights);
    if (!st.ok()) return st;

    const hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error("hipStreamSynchronize after EAGLE-4 weight load",
                                 hipGetErrorString(sync_err), __FILE__, __LINE__);
    }

    weights.present = true;
    return weights;
}

}  // namespace ps::qwen35::eagle4
