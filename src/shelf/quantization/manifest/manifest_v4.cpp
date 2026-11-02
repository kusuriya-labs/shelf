#include <shelf/quantization/manifest/manifest_v4.h>
#include <shelf/quantization/payload_descriptor.h>
#include <shelf/quantization/quant_contract.h>
#include <shelf/quantization/quant_format.h>
#include <shelf/quantization/quant_spec.h>
#include <nlohmann/json.hpp>

namespace shelf::quantization::manifest {

namespace {

using shelf::quantization::QuantDomain;
using shelf::quantization::QuantPayloadRef;
using shelf::quantization::QuantPayloadRole;
using shelf::quantization::QuantScalarType;
using shelf::quantization::QuantizedTensorDescriptor;
using shelf::quantization::QuantizedTensorLayout;
using shelf::quantization::ScaleApplication;
using shelf::quantization::ScaleGranularity;
using shelf::quantization::ScaleGrid;
using shelf::quantization::TensorStorageOrder;
using shelf::quantization::WeightBlockShape;
using shelf::quantization::WeightQuantSpec;
using shelf::quantization::WeightScaleSpec;
using shelf::quantization::canonical_layout_name;
using shelf::quantization::default_weight_block;
using shelf::quantization::default_weight_scale_spec;
using shelf::quantization::derive_scale_grid;
using shelf::quantization::quant_padded_k;

const char* scalar_type_name(QuantScalarType dtype) {
    switch (dtype) {
        case QuantScalarType::U8: return "U8";
        case QuantScalarType::I8: return "I8";
        case QuantScalarType::BF16: return "BF16";
        case QuantScalarType::F32: return "F32";
        case QuantScalarType::FP8E4M3: return "FP8E4M3";
        case QuantScalarType::E8M0: return "E8M0";
    }
    return "U8";
}

Result<QuantScalarType> parse_scalar_type(const std::string& name) {
    if (name == "U8") return QuantScalarType::U8;
    if (name == "I8") return QuantScalarType::I8;
    if (name == "BF16") return QuantScalarType::BF16;
    if (name == "F32") return QuantScalarType::F32;
    if (name == "FP8E4M3") return QuantScalarType::FP8E4M3;
    if (name == "E8M0") return QuantScalarType::E8M0;
    return Status::invalid_argument("unsupported quant scalar type", __FILE__, __LINE__);
}

const char* granularity_name(ScaleGranularity granularity) {
    switch (granularity) {
        case ScaleGranularity::PerTensor: return "per_tensor";
        case ScaleGranularity::PerRow: return "per_row";
        case ScaleGranularity::MatrixBlock: return "matrix_block";
        case ScaleGranularity::PerTokenHead: return "per_token_head";
        case ScaleGranularity::PerHeadBlock: return "per_head_block";
    }
    return "matrix_block";
}

Result<ScaleGranularity> parse_granularity(const std::string& name) {
    if (name == "per_tensor") return ScaleGranularity::PerTensor;
    if (name == "per_row") return ScaleGranularity::PerRow;
    if (name == "matrix_block") return ScaleGranularity::MatrixBlock;
    if (name == "per_token_head") return ScaleGranularity::PerTokenHead;
    if (name == "per_head_block") return ScaleGranularity::PerHeadBlock;
    return Status::invalid_argument("unsupported scale granularity", __FILE__, __LINE__);
}

const char* application_name(ScaleApplication application) {
    return application == ScaleApplication::Divide ? "divide" : "multiply";
}

Result<ScaleApplication> parse_application(const std::string& name) {
    if (name == "multiply") return ScaleApplication::Multiply;
    if (name == "divide") return ScaleApplication::Divide;
    return Status::invalid_argument("unsupported scale application", __FILE__, __LINE__);
}

const char* role_name(QuantPayloadRole role) {
    switch (role) {
        case QuantPayloadRole::Codes: return "codes";
        case QuantPayloadRole::Scale: return "scale";
        case QuantPayloadRole::ZeroPoint: return "zero_point";
        case QuantPayloadRole::Codebook: return "codebook";
        case QuantPayloadRole::Auxiliary: return "auxiliary";
        case QuantPayloadRole::Data: return "data";
    }
    return "codes";
}

Result<QuantPayloadRole> parse_role(const std::string& name) {
    if (name == "codes") return QuantPayloadRole::Codes;
    if (name == "scale") return QuantPayloadRole::Scale;
    if (name == "zero_point") return QuantPayloadRole::ZeroPoint;
    if (name == "codebook") return QuantPayloadRole::Codebook;
    if (name == "auxiliary") return QuantPayloadRole::Auxiliary;
    if (name == "data") return QuantPayloadRole::Data;
    return Status::invalid_argument("unsupported quant payload role", __FILE__, __LINE__);
}

const char* storage_order_name(TensorStorageOrder order) {
    switch (order) {
        case TensorStorageOrder::RowMajor: return "row_major";
        case TensorStorageOrder::ColumnMajor: return "column_major";
        case TensorStorageOrder::Packed: return "packed";
    }
    return "row_major";
}

Result<TensorStorageOrder> parse_storage_order(const std::string& name) {
    if (name == "row_major") return TensorStorageOrder::RowMajor;
    if (name == "column_major") return TensorStorageOrder::ColumnMajor;
    if (name == "packed") return TensorStorageOrder::Packed;
    return Status::invalid_argument("unsupported tensor storage order", __FILE__, __LINE__);
}

std::string internal_scale_tag(QuantScalarType dtype) {
    return dtype == QuantScalarType::F32 ? "F32" : "U8";
}

QuantScalarType scale_semantic_dtype(QuantizedEncoding enc) {
    return default_weight_scale_spec(enc)->dtype;
}

QuantScalarType scale_semantic_dtype_from_tag(QuantizedEncoding enc, const std::string& tag) {
    (void)enc;
    if (tag == "F32") return QuantScalarType::F32;
    return QuantScalarType::BF16;
}

std::vector<uint64_t> codes_shape_of(
    const std::vector<int64_t>& logical_shape, uint64_t k_padded, QuantizedEncoding enc) {
    const QuantFormatDesc* format = quant_format_desc(quant_format_id(enc));
    std::vector<uint64_t> shape(logical_shape.begin(), logical_shape.end());
    shape.back() = quant_codes_bytes(1, k_padded, *format);
    return shape;
}

std::vector<uint64_t> scale_shape_of(
    const std::vector<int64_t>& logical_shape, uint64_t k_padded, WeightBlockShape block) {
    const uint64_t n = static_cast<uint64_t>(logical_shape[logical_shape.size() - 2]);
    const ScaleGrid grid = derive_scale_grid(n, k_padded, block);
    std::vector<uint64_t> shape(logical_shape.begin(), logical_shape.end() - 2);
    shape.push_back(grid.n);
    shape.push_back(grid.k);
    return shape;
}

Result<QuantizedTensorDescriptor> descriptor_from_v4(const nlohmann::json& tj) {
    QuantizedTensorDescriptor desc;
    if (!tj.is_object()) {
        return Status::invalid_argument("bad quantized tensor metadata", __FILE__, __LINE__);
    }
    const std::string domain = tj.value("domain", std::string());
    if (domain != "weight") {
        return Status::unsupported("only weight domain tensors are supported", __FILE__, __LINE__);
    }
    desc.domain = QuantDomain::Weight;
    if (!tj.contains("logical_shape") || !tj["logical_shape"].is_array()) {
        return Status::invalid_argument("quantized tensor missing logical_shape", __FILE__, __LINE__);
    }
    for (const auto& d : tj["logical_shape"]) {
        desc.logical_shape.push_back(d.get<int64_t>());
    }
    if (!tj.contains("quant") || !tj["quant"].is_object()) {
        return Status::invalid_argument("quantized tensor missing quant spec", __FILE__, __LINE__);
    }
    const nlohmann::json& qj = tj["quant"];
    const std::string enc_name = qj.value("encoding", std::string());
    Result<QuantizedEncoding> enc = parse_v4_encoding(enc_name);
    if (!enc.ok()) return enc.status();
    desc.quant.encoding = enc.value();
    if (qj.contains("scale")) {
        if (!qj["scale"].is_object()) {
            return Status::invalid_argument("quantized scale spec is not an object", __FILE__, __LINE__);
        }
        const nlohmann::json& sj = qj["scale"];
        WeightScaleSpec scale;
        Result<ScaleGranularity> granularity = parse_granularity(sj.value("granularity", std::string()));
        if (!granularity.ok()) return granularity.status();
        scale.granularity = granularity.value();
        if (!sj.contains("block_shape") || !sj["block_shape"].is_object()) {
            return Status::invalid_argument("quantized scale spec missing block_shape", __FILE__, __LINE__);
        }
        scale.block.n = sj["block_shape"].value("n", 0u);
        scale.block.k = sj["block_shape"].value("k", 0u);
        Result<QuantScalarType> dtype = parse_scalar_type(sj.value("dtype", std::string()));
        if (!dtype.ok()) return dtype.status();
        scale.dtype = dtype.value();
        Result<ScaleApplication> application = parse_application(sj.value("application", std::string("multiply")));
        if (!application.ok()) return application.status();
        scale.application = application.value();
        desc.quant.scale = scale;
    }
    const int64_t logical_k = desc.logical_shape.empty() ? 0 : desc.logical_shape.back();
    if (tj.contains("layout") && tj["layout"].is_object()) {
        const nlohmann::json& lj = tj["layout"];
        desc.layout.layout_id = lj.value("layout_id", std::string());
        Result<TensorStorageOrder> order = parse_storage_order(lj.value("storage_order", std::string("row_major")));
        if (!order.ok()) return order.status();
        desc.layout.storage_order = order.value();
        if (lj.contains("physical_shape") && lj["physical_shape"].is_array()) {
            for (const auto& d : lj["physical_shape"]) {
                desc.layout.physical_shape.push_back(d.get<uint64_t>());
            }
        }
        desc.layout.k_padded = lj.value("k_padded", uint64_t{0});
    }
    if (desc.layout.k_padded == 0) {
        desc.layout.k_padded = logical_k <= 0
            ? uint64_t{0}
            : quant_padded_k(static_cast<uint64_t>(logical_k), quant_format_id(desc.quant.encoding));
    }
    if (!tj.contains("payloads") || !tj["payloads"].is_array()) {
        return Status::invalid_argument("quantized tensor missing payloads", __FILE__, __LINE__);
    }
    for (const auto& pj : tj["payloads"]) {
        if (!pj.is_object()) {
            return Status::invalid_argument("bad quant payload ref", __FILE__, __LINE__);
        }
        QuantPayloadRef ref;
        Result<QuantPayloadRole> role = parse_role(pj.value("role", std::string()));
        if (!role.ok()) return role.status();
        ref.role = role.value();
        ref.tensor = pj.value("tensor", std::string());
        Result<QuantScalarType> dtype = parse_scalar_type(pj.value("dtype", std::string()));
        if (!dtype.ok()) return dtype.status();
        ref.dtype = dtype.value();
        if (pj.contains("shape") && pj["shape"].is_array()) {
            for (const auto& d : pj["shape"]) {
                ref.shape.push_back(d.get<uint64_t>());
            }
        }
        ref.crc32 = pj.value("crc32", std::string());
        desc.payloads.push_back(std::move(ref));
    }
    return desc;
}

Status normalize_descriptor(
    const QuantizedTensorDescriptor& desc, QuantizedTensorMetadata& out) {
    out.role = "weight";
    out.encoding = desc.quant.encoding;
    out.layout = desc.layout.layout_id;
    out.logical_shape = desc.logical_shape;
    out.k_padded = desc.layout.k_padded;
    if (desc.quant.scale.has_value()) {
        out.scale_block_n = desc.quant.scale->block.n;
        out.scale_block_k = desc.quant.scale->block.k;
    }
    if (desc.quant.encoding == QuantizedEncoding::Bf16) {
        const QuantPayloadRef* data = shelf::quantization::find_payload(desc, QuantPayloadRole::Data);
        out.data = QuantizedTensorRef{data->tensor, "BF16", data->crc32};
        return Status::make_ok();
    }
    const QuantPayloadRef* codes = shelf::quantization::find_payload(desc, QuantPayloadRole::Codes);
    const QuantPayloadRef* scale = shelf::quantization::find_payload(desc, QuantPayloadRole::Scale);
    out.codes = QuantizedTensorRef{codes->tensor, "U8", codes->crc32};
    out.metadata1 = QuantizedTensorRef{
        scale->tensor, internal_scale_tag(scale_semantic_dtype(desc.quant.encoding)), scale->crc32};
    return Status::make_ok();
}

nlohmann::json tensor_to_v4(const QuantizedTensorMetadata& t) {
    const QuantizedEncoding enc = t.encoding;
    nlohmann::json tj;
    tj["domain"] = "weight";
    tj["logical_shape"] = t.logical_shape;
    nlohmann::json qj;
    qj["encoding"] = v4_encoding_name(enc);
    if (enc != QuantizedEncoding::Bf16) {
        const WeightBlockShape block = WeightBlockShape{
            t.scale_block_n != 0 ? t.scale_block_n : default_weight_block(enc).n,
            t.scale_block_k != 0 ? t.scale_block_k : default_weight_block(enc).k};
        nlohmann::json sj;
        sj["granularity"] = granularity_name(ScaleGranularity::MatrixBlock);
        sj["block_shape"]["n"] = block.n;
        sj["block_shape"]["k"] = block.k;
        sj["dtype"] = scalar_type_name(scale_semantic_dtype(enc));
        sj["application"] = application_name(ScaleApplication::Multiply);
        qj["scale"] = std::move(sj);
    }
    tj["quant"] = std::move(qj);
    nlohmann::json lj;
    lj["layout_id"] = t.layout.empty() ? canonical_layout_name(enc) : t.layout;
    lj["storage_order"] = storage_order_name(TensorStorageOrder::RowMajor);
    if (enc == QuantizedEncoding::Bf16) {
        nlohmann::json physical = nlohmann::json::array();
        for (const int64_t dim : t.logical_shape) physical.push_back(dim);
        lj["physical_shape"] = std::move(physical);
    } else {
        nlohmann::json physical = nlohmann::json::array();
        for (const uint64_t dim : codes_shape_of(t.logical_shape, t.k_padded, enc)) {
            physical.push_back(dim);
        }
        lj["physical_shape"] = std::move(physical);
    }
    lj["k_padded"] = t.k_padded;
    tj["layout"] = std::move(lj);
    nlohmann::json payloads = nlohmann::json::array();
    if (enc == QuantizedEncoding::Bf16 && t.data) {
        nlohmann::json pj;
        pj["role"] = role_name(QuantPayloadRole::Data);
        pj["tensor"] = t.data->tensor;
        pj["dtype"] = "BF16";
        nlohmann::json shape = nlohmann::json::array();
        for (const int64_t dim : t.logical_shape) shape.push_back(dim);
        pj["shape"] = std::move(shape);
        pj["crc32"] = t.data->crc32;
        payloads.push_back(std::move(pj));
    }
    if (t.codes) {
        nlohmann::json pj;
        pj["role"] = role_name(QuantPayloadRole::Codes);
        pj["tensor"] = t.codes->tensor;
        pj["dtype"] = "U8";
        nlohmann::json shape = nlohmann::json::array();
        for (const uint64_t dim : codes_shape_of(t.logical_shape, t.k_padded, enc)) shape.push_back(dim);
        pj["shape"] = std::move(shape);
        pj["crc32"] = t.codes->crc32;
        payloads.push_back(std::move(pj));
    }
    if (t.metadata1) {
        nlohmann::json pj;
        pj["role"] = role_name(QuantPayloadRole::Scale);
        pj["tensor"] = t.metadata1->tensor;
        pj["dtype"] = scalar_type_name(scale_semantic_dtype(enc));
        nlohmann::json shape = nlohmann::json::array();
        const WeightBlockShape scale_block{
            t.scale_block_n != 0 ? t.scale_block_n : default_weight_block(enc).n,
            t.scale_block_k != 0 ? t.scale_block_k : default_weight_block(enc).k};
        for (const uint64_t dim : scale_shape_of(t.logical_shape, t.k_padded, scale_block)) {
            shape.push_back(dim);
        }
        pj["shape"] = std::move(shape);
        pj["crc32"] = t.metadata1->crc32;
        payloads.push_back(std::move(pj));
    }
    tj["payloads"] = std::move(payloads);
    return tj;
}

}

const char* v4_encoding_name(QuantizedEncoding enc) {
    switch (enc) {
        case QuantizedEncoding::Bf16: return "bf16";
        case QuantizedEncoding::Fp8: return kFp8V4EncodingName;
    }
    return "unknown";
}

Result<QuantizedEncoding> parse_v4_encoding(const std::string& name) {
    if (name == "bf16") return QuantizedEncoding::Bf16;
    if (name == kFp8V4EncodingName) return QuantizedEncoding::Fp8;
    return Status::invalid_argument("unsupported quantized tensor encoding", __FILE__, __LINE__);
}

Result<QuantizedManifest> parse_quantized_manifest_v4(const std::string& json) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json);
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("bad quantized manifest: " + std::string(e.what())).c_str(), __FILE__, __LINE__);
    }
    if (!j.is_object()) {
        return Status::invalid_argument("bad quantized manifest: not an object", __FILE__, __LINE__);
    }
    QuantizedManifest m;
    m.format = j.value("format", std::string());
    if (m.format != kQuantizedSafetensorsFormat) {
        return Status::invalid_argument("unsupported quantized format", __FILE__, __LINE__);
    }
    m.format_version = j.value("format_version", 0u);
    if (m.format_version != kQuantizedSafetensorsFormatVersion) {
        return Status::invalid_argument("unsupported quantized format_version", __FILE__, __LINE__);
    }
    m.architecture = j.value("architecture", std::string());
    m.preset = j.value("preset", std::string());
    m.scope = j.value("scope", std::string());
    m.source_dtype = j.value("source_dtype", std::string());
    m.source_model_fingerprint = j.value("source_model_fingerprint", std::string());
    m.logical_parameter_count = j.value("logical_parameter_count", 0u);
    m.payload_bytes = j.value("payload_bytes", 0u);
    m.padding_bytes = j.value("padding_bytes", 0u);
    m.effective_payload_bpw = j.value("effective_payload_bpw", 0.0);
    if (j.contains("aliases") && j["aliases"].is_object()) {
        for (auto it = j["aliases"].begin(); it != j["aliases"].end(); ++it) {
            m.aliases[it.key()] = it.value().get<std::string>();
        }
    }
    if (j.contains("imatrix") && j["imatrix"].is_object()) {
        QuantizedImatrixMetadata im;
        const nlohmann::json& ij = j["imatrix"];
        im.enabled = ij.value("enabled", false);
        im.policy = ij.value("policy", std::string());
        im.sha256 = ij.value("sha256", std::string());
        im.model_fingerprint = ij.value("model_fingerprint", std::string());
        im.corpus_sha256 = ij.value("corpus_sha256", std::string());
        if (ij.contains("coverage") && ij["coverage"].is_object()) {
            im.required = ij["coverage"].value("required", 0u);
            im.matched = ij["coverage"].value("matched", 0u);
        }
        m.imatrix = im;
    }
    if (!j.contains("tensors") || !j["tensors"].is_object()) {
        return Status::invalid_argument("quantized manifest missing tensors", __FILE__, __LINE__);
    }
    for (auto it = j["tensors"].begin(); it != j["tensors"].end(); ++it) {
        Result<QuantizedTensorDescriptor> desc = descriptor_from_v4(it.value());
        if (!desc.ok()) return desc.status();
        Status contract = shelf::quantization::validate_quantized_tensor(desc.value());
        if (!contract.ok()) return contract;
        QuantizedTensorMetadata meta;
        Status normalized = normalize_descriptor(desc.value(), meta);
        if (!normalized.ok()) return normalized;
        m.tensors[it.key()] = std::move(meta);
    }
    return m;
}

Result<std::string> serialize_quantized_manifest_v4(const QuantizedManifest& m) {
    nlohmann::json j;
    j["format"] = m.format;
    j["format_version"] = kQuantizedSafetensorsFormatVersion;
    j["architecture"] = m.architecture;
    j["preset"] = m.preset;
    j["scope"] = m.scope;
    j["source_dtype"] = m.source_dtype;
    j["source_model_fingerprint"] = m.source_model_fingerprint;
    j["logical_parameter_count"] = m.logical_parameter_count;
    j["payload_bytes"] = m.payload_bytes;
    j["padding_bytes"] = m.padding_bytes;
    j["effective_payload_bpw"] = m.effective_payload_bpw;
    if (!m.aliases.empty()) {
        nlohmann::json al = nlohmann::json::object();
        for (const auto& kv : m.aliases) al[kv.first] = kv.second;
        j["aliases"] = std::move(al);
    }
    if (m.imatrix) {
        nlohmann::json im;
        im["enabled"] = m.imatrix->enabled;
        im["policy"] = m.imatrix->policy;
        im["sha256"] = m.imatrix->sha256;
        im["model_fingerprint"] = m.imatrix->model_fingerprint;
        im["corpus_sha256"] = m.imatrix->corpus_sha256;
        im["coverage"]["required"] = m.imatrix->required;
        im["coverage"]["matched"] = m.imatrix->matched;
        j["imatrix"] = std::move(im);
    }
    nlohmann::json tensors = nlohmann::json::object();
    for (const auto& kv : m.tensors) {
        tensors[kv.first] = tensor_to_v4(kv.second);
    }
    j["tensors"] = std::move(tensors);
    return j.dump(2) + "\n";
}

}
