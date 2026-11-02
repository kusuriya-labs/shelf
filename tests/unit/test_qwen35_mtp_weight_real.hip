#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/gfx_arch.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/quantization/fp8/fp8_native.h>
#include <shelf/quantization/manifest/crc32.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace shelf;

namespace {

struct Failure {
    std::string tensor;
    std::string detail;
};

struct MtpSource {
    std::string format;
    std::vector<std::size_t> shape;
    std::vector<uint8_t> data;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> metadata1;
    std::vector<uint8_t> metadata2;
    std::string source_crc32;
    std::uint64_t k_padded = 0;

    std::vector<uint8_t> concat_payload() const {
        std::vector<uint8_t> out;
        for (const auto* stream : {&data, &codes, &metadata1, &metadata2}) {
            out.insert(out.end(), stream->begin(), stream->end());
        }
        return out;
    }
};

const char* kMatrixNames[] = {
    "mtp.fc.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
};

const char* kVectorNames[] = {
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
};

std::string crc32_hex(const void* data, std::size_t size) {
    char hex[11];
    std::snprintf(
        hex, sizeof(hex), "0x%08x",
        quantization::manifest::crc32_iso_hdlc(data, size));
    return hex;
}

std::vector<uint8_t> d2h(const void* dev, std::size_t bytes) {
    std::vector<uint8_t> host(bytes);
    if (bytes == 0) return host;
    if (hipMemcpy(host.data(), dev, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        host.assign(bytes, 0);
    }
    return host;
}

std::size_t span_bytes(const gpu::Tensor& t) {
    std::size_t span = 0;
    (void)t.required_span_bytes(span);
    return span;
}

bool expected_matrix_payload(
    const weights::MatrixWeight& w,
    const MtpSource& source,
    std::vector<uint8_t>& out_codes,
    std::vector<uint8_t>& out_scales,
    std::string& why) {

    if (w.encoding == weights::MatrixEncoding::Bf16) {
        out_codes = source.data;
        return true;
    }

    if (w.encoding == weights::MatrixEncoding::Fp8Finegrained) {
        if (!w.preshuffled) {
            out_codes = source.codes;
            out_scales = source.metadata1;
            return true;
        }
        quantization::fp8::Fp8BlockCanonicalView cv;
        cv.batch = 1;
        cv.n = w.rows;
        cv.padded_k = w.k_padded;
        cv.codes = source.codes.data();
        cv.codes_bytes = source.codes.size();
        cv.scales = reinterpret_cast<const float*>(source.metadata1.data());
        cv.scales_bytes = source.metadata1.size();
        cv.scale_n = (w.rows + 127u) / 128u;
        cv.scale_k = w.k_padded / 128u;
        quantization::fp8::Fp8NativeHost native;
        if (!cv.validate() ||
            !quantization::fp8::preshuffle_fp8_native(cv, native)) {
            why = "fp8 preshuffle reference failed";
            return false;
        }
        out_codes.assign(native.codes.get(), native.codes.get() + native.codes_bytes);
        out_scales.resize(native.scale_count * sizeof(float));
        if (!out_scales.empty()) {
            std::memcpy(out_scales.data(), native.scales.get(), out_scales.size());
        }
        return true;
    }

    why = "unsupported matrix encoding";
    return false;
}

}  // namespace

int main() {
    const char* model_dir_env = std::getenv("SHELF_MODEL_DIR_MTP");
    if (model_dir_env == nullptr || model_dir_env[0] == '\0') {
        model_dir_env = std::getenv("SHELF_MODEL_DIR");
    }
    if (model_dir_env == nullptr || model_dir_env[0] == '\0') {
        std::printf("SKIP: set SHELF_MODEL_DIR_MTP or SHELF_MODEL_DIR to a "
                    "Qwen3.5 model directory\n");
        return 77;
    }
    const std::string model_dir = model_dir_env;
    if (!fs::is_directory(model_dir)) {
        std::printf("FAIL: model dir not found: %s\n", model_dir.c_str());
        std::printf("MTP_WEIGHT_CONTRACT: FAIL\n");
        return 1;
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::vector<Failure> failures;
    const auto fail_tensor = [&](const std::string& t, const std::string& d) {
        failures.push_back({t, d});
    };

    auto ar = gpu::GpuArena::create(0, 24ull * 1024ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        std::printf("FAIL: arena: %s\n", ar.status().message().c_str());
        std::printf("MTP_WEIGHT_CONTRACT: FAIL\n");
        shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    gpu::GpuArena arena = ar.release();

    const bool quantized = weights::is_quantized_model_dir(model_dir);
    std::printf("=== MTP weight contract: real checkpoint ===\n");
    std::printf("model_dir: %s\n", model_dir.c_str());
    std::printf("format: %s\n", quantized ? "quantized (shelf)" : "plain safetensors (bf16)");

    qwen35::Qwen35LoadOptions options;
    options.load_mtp_layers = true;
    options.verify_quantized_payload_crc = false;
    auto wr = qwen35::load_qwen35_weights_from_safetensors(model_dir, arena, stream, options);
    if (!wr.ok()) {
        fail_tensor("(model load)", wr.status().message());
    } else {
        qwen35::Qwen35ModelWeights w = wr.release();

        auto config_res = qwen35::read_qwen35_text_config(model_dir);
        if (!config_res.ok()) {
            fail_tensor("(config)", config_res.status().message());
        } else {
            qwen35::Qwen35TextConfig config = config_res.release();
            const std::uint32_t H = static_cast<std::uint32_t>(config.hidden_size);
            std::printf("H=%zu I=%zu mtp.present=%d\n",
                        config.hidden_size, config.intermediate_size, w.mtp.present ? 1 : 0);

            if (!w.mtp.present) {
                fail_tensor("(mtp)", "checkpoint has no MTP weights");
            } else {
                auto lower = qwen35::lower_qwen35_mtp_to_primitives(config, w);
                if (!lower.ok()) {
                    fail_tensor("(mtp geometry)", lower.status().message());
                } else {
                    std::printf("mtp geometry validation: PASS (fc=[%u,%u] expected=[%u,%u])\n",
                                w.mtp.fc.rows, w.mtp.fc.cols, H, 2u * H);
                }

                std::printf(
                    "%-44s %-10s %-18s %-64s %s\n",
                    "tensor", "src fmt", "src shape", "source_crc32", "device payload");

                if (quantized) {
                    auto rr = quantization::bundle::QuantizedModelReader::open(model_dir, false);
                    if (!rr.ok()) {
                        fail_tensor("(reader)", rr.status().message());
                    } else {
                        quantization::bundle::QuantizedModelReader reader = rr.release();
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

                        const auto fetch_source = [&](const std::string& name, MtpSource& src,
                                                      std::string& detail) -> bool {
                            auto view = reader.resolve(name);
                            if (!view.ok()) {
                                detail = "resolve failed: " + view.status().message();
                                return false;
                            }
                            const auto& v = view.value();
                            src.k_padded = static_cast<std::uint64_t>(v.k_padded);
                            for (auto d : v.logical_shape) src.shape.push_back(d);
                            if (v.encoding == quantization::manifest::QuantizedEncoding::Bf16) {
                                src.format = "bf16";
                                src.data.assign(static_cast<const uint8_t*>(v.data.data),
                                                static_cast<const uint8_t*>(v.data.data) + v.data.size);
                            } else {
                                src.format =
                                    v.encoding ==
                                            quantization::manifest::QuantizedEncoding::Fp8
                                        ? "fp8"
                                        : "unknown";
                                if (v.codes.data != nullptr) {
                                    const auto* p = static_cast<const uint8_t*>(v.codes.data);
                                    src.codes.assign(p, p + v.codes.size);
                                }
                                if (v.metadata1.data != nullptr) {
                                    const auto* p = static_cast<const uint8_t*>(v.metadata1.data);
                                    src.metadata1.assign(p, p + v.metadata1.size);
                                }
                                if (v.metadata2.data != nullptr) {
                                    const auto* p = static_cast<const uint8_t*>(v.metadata2.data);
                                    src.metadata2.assign(p, p + v.metadata2.size);
                                }
                            }
                            const auto payload = src.concat_payload();
                            src.source_crc32 = crc32_hex(payload.data(), payload.size());
                            return true;
                        };

                        for (int i = 0; i < 8; ++i) {
                            const std::string name = kMatrixNames[i];
                            const weights::MatrixWeight& mw = *matrices[i];
                            MtpSource src;
                            std::string detail;
                            if (!fetch_source(name, src, detail)) {
                                fail_tensor(name, detail);
                                std::printf("%-44s SRC FAIL: %s\n", name.c_str(), detail.c_str());
                                continue;
                            }
                            std::string shape_str;
                            for (auto d : src.shape) shape_str += std::to_string(d) + " ";
                            std::vector<uint8_t> want_codes;
                            std::vector<uint8_t> want_scales;
                            std::string why;
                            bool ok = expected_matrix_payload(mw, src, want_codes, want_scales, why);
                            bool match = false;
                            if (ok) {
                                const gpu::Tensor& codes_dest =
                                    mw.encoding == weights::MatrixEncoding::Bf16 ? mw.bf16 : mw.codes;
                                const std::size_t codes_cap =
                                    codes_dest.ndim() != 0 ? codes_dest.allocation_bytes() : 0;
                                if (codes_cap < want_codes.size()) {
                                    detail = "codes capacity got=" + std::to_string(codes_cap) +
                                             " want=" + std::to_string(want_codes.size());
                                    match = false;
                                } else {
                                    const auto got = d2h(codes_dest.data<void>(), want_codes.size());
                                    match = (got == want_codes);
                                    if (!match) detail = "codes bytes mismatch";
                                }
                                if (match && !want_scales.empty()) {
                                    const std::size_t scales_cap =
                                        mw.scales.ndim() != 0 ? mw.scales.allocation_bytes() : 0;
                                    if (scales_cap < want_scales.size()) {
                                        detail = "scales capacity got=" + std::to_string(scales_cap) +
                                                 " want=" + std::to_string(want_scales.size());
                                        match = false;
                                    } else {
                                        const auto got = d2h(mw.scales.data<void>(), want_scales.size());
                                        match = (got == want_scales);
                                        if (!match) detail = "scales bytes mismatch";
                                    }
                                }
                            } else {
                                match = false;
                                detail = why;
                            }
                            if (!match) fail_tensor(name, detail);
                            std::printf("%-44s %-10s %-18s %-64s %s\n",
                                        name.c_str(), src.format.c_str(), shape_str.c_str(),
                                        src.source_crc32.c_str(), match ? "MATCH" : "MISMATCH");
                        }

                        for (int i = 0; i < 7; ++i) {
                            const std::string name = kVectorNames[i];
                            const gpu::Tensor& tv = *vectors[i];
                            MtpSource src;
                            std::string detail;
                            if (!fetch_source(name, src, detail)) {
                                fail_tensor(name, detail);
                                std::printf("%-44s SRC FAIL: %s\n", name.c_str(), detail.c_str());
                                continue;
                            }
                            bool match = false;
                            if (tv.ndim() != 0 && span_bytes(tv) == src.data.size()) {
                                const auto got = d2h(tv.data<void>(), src.data.size());
                                match = (got == src.data);
                                if (!match) detail = "bytes mismatch";
                            } else {
                                detail = "size got=" + std::to_string(tv.ndim() != 0 ? span_bytes(tv) : 0) +
                                         " want=" + std::to_string(src.data.size());
                            }
                            if (!match) fail_tensor(name, detail);
                            std::string shape_str;
                            for (auto d : src.shape) shape_str += std::to_string(d) + " ";
                            std::printf("%-44s %-10s %-18s %-64s %s\n",
                                        name.c_str(), src.format.c_str(), shape_str.c_str(),
                                        src.source_crc32.c_str(), match ? "MATCH" : "MISMATCH");
                        }
                    }
                } else {
                    auto cr = weights::SafetensorsCollection::open(model_dir);
                    if (!cr.ok()) {
                        fail_tensor("(collection)", cr.status().message());
                    } else {
                        weights::SafetensorsCollection collection = cr.release();
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

                        const auto fetch_source = [&](const std::string& name, MtpSource& src,
                                                      std::string& detail) -> bool {
                            auto spec = collection.tensor_spec(name);
                            if (!spec.ok()) {
                                detail = "spec failed: " + spec.status().message();
                                return false;
                            }
                            src.format = "BF16";
                            src.shape = spec.value().shape;
                            std::size_t bytes = 0;
                            auto data = collection.tensor_data(name, bytes);
                            if (!data.ok()) {
                                detail = "data failed: " + data.status().message();
                                return false;
                            }
                            src.data.assign(static_cast<const uint8_t*>(data.value()),
                                            static_cast<const uint8_t*>(data.value()) + bytes);
                            src.source_crc32 = crc32_hex(src.data.data(), src.data.size());
                            return true;
                        };

                        for (int i = 0; i < 8; ++i) {
                            const std::string name = kMatrixNames[i];
                            const weights::MatrixWeight& mw = *matrices[i];
                            MtpSource src;
                            std::string detail;
                            if (!fetch_source(name, src, detail)) {
                                fail_tensor(name, detail);
                                std::printf("%-44s SRC FAIL: %s\n", name.c_str(), detail.c_str());
                                continue;
                            }
                            std::string shape_str;
                            for (auto d : src.shape) shape_str += std::to_string(d) + " ";
                            std::vector<uint8_t> want_codes;
                            std::vector<uint8_t> want_scales;
                            std::string why;
                            bool ok = expected_matrix_payload(mw, src, want_codes, want_scales, why);
                            bool match = false;
                            if (!ok) {
                                detail = why;
                            } else {
                                const gpu::Tensor& codes_dest =
                                    mw.encoding == weights::MatrixEncoding::Bf16 ? mw.bf16 : mw.codes;
                                const std::size_t codes_cap =
                                    codes_dest.ndim() != 0 ? codes_dest.allocation_bytes() : 0;
                                if (codes_cap < want_codes.size()) {
                                    detail = "codes capacity got=" + std::to_string(codes_cap) +
                                             " want=" + std::to_string(want_codes.size());
                                } else {
                                    const auto got = d2h(codes_dest.data<void>(), want_codes.size());
                                    match = (got == want_codes);
                                    if (!match) detail = "codes bytes mismatch";
                                }
                            }
                            if (!match) fail_tensor(name, detail);
                            std::printf("%-44s %-10s %-18s %-64s %s\n",
                                        name.c_str(), src.format.c_str(), shape_str.c_str(),
                                        src.source_crc32.c_str(), match ? "MATCH" : "MISMATCH");
                        }

                        for (int i = 0; i < 7; ++i) {
                            const std::string name = kVectorNames[i];
                            const gpu::Tensor& tv = *vectors[i];
                            MtpSource src;
                            std::string detail;
                            if (!fetch_source(name, src, detail)) {
                                fail_tensor(name, detail);
                                std::printf("%-44s SRC FAIL: %s\n", name.c_str(), detail.c_str());
                                continue;
                            }
                            bool match = false;
                            if (tv.ndim() != 0 && span_bytes(tv) == src.data.size()) {
                                const auto got = d2h(tv.data<void>(), src.data.size());
                                match = (got == src.data);
                                if (!match) detail = "bytes mismatch";
                            } else {
                                detail = "size got=" + std::to_string(tv.ndim() != 0 ? span_bytes(tv) : 0) +
                                         " want=" + std::to_string(src.data.size());
                            }
                            if (!match) fail_tensor(name, detail);
                            std::string shape_str;
                            for (auto d : src.shape) shape_str += std::to_string(d) + " ";
                            std::printf("%-44s %-10s %-18s %-64s %s\n",
                                        name.c_str(), src.format.c_str(), shape_str.c_str(),
                                        src.source_crc32.c_str(), match ? "MATCH" : "MISMATCH");
                        }
                    }
                }
            }
        }
    }

    if (!failures.empty()) {
        std::printf("\n=== failures (%zu) ===\n", failures.size());
        for (const auto& f : failures) {
            std::printf("FAIL %s: %s\n", f.tensor.c_str(), f.detail.c_str());
        }
    }
    std::printf("\nMTP_WEIGHT_CONTRACT: %s\n", failures.empty() ? "PASS" : "FAIL");

    arena.shutdown();
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return failures.empty() ? 0 : 1;
}
