#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/quantization/manifest/manifest_v4.h>
#include <nlohmann/json.hpp>

namespace ps::quantization::manifest {

std::string quantized_encoding_name(QuantizedEncoding enc) {
    switch (enc) {
        case QuantizedEncoding::Bf16: return "bf16";
        case QuantizedEncoding::Fp8: return kFp8EncodingName;
    }
    return "unknown";
}

Result<QuantizedEncoding> parse_quantized_encoding(const std::string& name) {
    if (name == "bf16") return QuantizedEncoding::Bf16;
    if (name == kFp8EncodingName) return QuantizedEncoding::Fp8;
    return Status::invalid_argument("unsupported quantized tensor encoding", __FILE__, __LINE__);
}

ps::quantization::QuantFormatId quant_format_id(QuantizedEncoding enc) {
    switch (enc) {
        case QuantizedEncoding::Bf16: return ps::quantization::QuantFormatId::Bf16;
        case QuantizedEncoding::Fp8: return ps::quantization::QuantFormatId::Fp8Finegrained;
    }
    return ps::quantization::QuantFormatId::None;
}

namespace {

Status tensor_ref_from_json(const nlohmann::json& j, QuantizedTensorRef& r) {
    if (!j.is_object() || !j.contains("tensor") || !j.contains("dtype") ||
        !j.contains("crc32")) {
        return Status::invalid_argument("bad tensor ref", __FILE__, __LINE__);
    }
    r.tensor = j["tensor"].get<std::string>();
    r.dtype = j["dtype"].get<std::string>();
    r.crc32 = j["crc32"].get<std::string>();
    return Status::make_ok();
}

}

Result<std::string> serialize_quantized_manifest(const QuantizedManifest& m) {
    return serialize_quantized_manifest_v4(m);
}

namespace {

Result<QuantizedManifest> parse_quantized_manifest_v3(const std::string& json) {
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
    if (m.format_version != kQuantizedSafetensorsMinFormatVersion) {
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
        const nlohmann::json& tj = it.value();
        if (!tj.is_object()) {
            return Status::invalid_argument("bad tensor metadata", __FILE__, __LINE__);
        }
        QuantizedTensorMetadata t;
        t.role = tj.value("role", std::string());
        const std::string enc = tj.value("encoding", std::string());
        auto enc_result = parse_quantized_encoding(enc);
        if (!enc_result.ok()) {
            return enc_result.status();
        }
        t.encoding = enc_result.release();
        t.codebook = tj.value("codebook", std::string());
        t.layout = tj.value("layout", std::string());
        if (tj.contains("logical_shape") && tj["logical_shape"].is_array()) {
            for (const auto& d : tj["logical_shape"]) {
                t.logical_shape.push_back(d.get<int64_t>());
            }
        }
        t.k_padded = tj.value("k_padded", 0u);
        if (tj.contains("codes")) {
            QuantizedTensorRef r;
            if (!tensor_ref_from_json(tj["codes"], r).ok()) {
                return Status::invalid_argument("bad codes ref", __FILE__, __LINE__);
            }
            t.codes = r;
        }
        if (tj.contains("metadata1")) {
            QuantizedTensorRef r;
            if (!tensor_ref_from_json(tj["metadata1"], r).ok()) {
                return Status::invalid_argument("bad metadata1 ref", __FILE__, __LINE__);
            }
            t.metadata1 = r;
        }
        if (tj.contains("metadata2")) {
            QuantizedTensorRef r;
            if (!tensor_ref_from_json(tj["metadata2"], r).ok()) return Status::invalid_argument("bad metadata2 ref", __FILE__, __LINE__);
            t.metadata2 = r;
        }
        if (tj.contains("metadata3")) {
            QuantizedTensorRef r;
            if (!tensor_ref_from_json(tj["metadata3"], r).ok()) return Status::invalid_argument("bad metadata3 ref", __FILE__, __LINE__);
            t.metadata3 = r;
        }
        if (tj.contains("metadata4")) {
            QuantizedTensorRef r;
            if (!tensor_ref_from_json(tj["metadata4"], r).ok()) return Status::invalid_argument("bad metadata4 ref", __FILE__, __LINE__);
            t.metadata4 = r;
        }
        if (tj.contains("data")) {
            QuantizedTensorRef r;
            if (!tensor_ref_from_json(tj["data"], r).ok()) {
                return Status::invalid_argument("bad data ref", __FILE__, __LINE__);
            }
            t.data = r;
        }
        m.tensors[it.key()] = std::move(t);
    }

    return m;
}

}

Result<QuantizedManifest> parse_quantized_manifest(const std::string& json) {
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
    const uint32_t version = j.value("format_version", 0u);
    if (version == kQuantizedSafetensorsFormatVersion) {
        return parse_quantized_manifest_v4(json);
    }
    if (version == kQuantizedSafetensorsMinFormatVersion) {
        return parse_quantized_manifest_v3(json);
    }
    return Status::invalid_argument("unsupported quantized format_version", __FILE__, __LINE__);
}

}
