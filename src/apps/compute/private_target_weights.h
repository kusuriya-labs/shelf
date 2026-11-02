#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/weights/weight_loader.h>

#include <hip/hip_runtime.h>

#include <string>
#include <vector>

namespace shelf::app {

inline Status load_private_target_weights(const std::string& model_dir,
                                          gpu::GpuArena& arena,
                                          hipStream_t stream,
                                          shelf::weights::MatrixWeight& lm_head,
                                          shelf::weights::MatrixWeight& embed_tokens,
                                          bool load_lm_head,
                                          bool load_embed_tokens) {
    std::vector<std::string> names;
    if (load_lm_head) names.push_back("model.language_model.lm_head.weight");
    if (load_embed_tokens) names.push_back("model.language_model.embed_tokens.weight");
    if (names.empty()) return Status::make_ok();
    auto hf_source = shelf::quantization::bundle::synthesize_hf_quantized_source(
        model_dir,
        names,
        false);
    if (!hf_source.ok()) return hf_source.status();
    auto reader_result = shelf::quantization::bundle::QuantizedModelReader::open_hf_source(
        hf_source.release(), false);
    if (!reader_result.ok()) return reader_result.status();
    shelf::quantization::bundle::QuantizedModelReader reader = reader_result.release();
    shelf::weights::WeightLoadOptions options;
    if (load_lm_head) {
        auto lm = shelf::weights::load_quantized_matrix(
            reader, "model.language_model.lm_head.weight", arena, stream, options);
        if (!lm.ok()) return lm.status();
        lm_head = lm.release();
    }
    if (load_embed_tokens) {
        auto embed = shelf::weights::load_quantized_matrix(
            reader, "model.language_model.embed_tokens.weight", arena, stream, options);
        if (!embed.ok()) return embed.status();
        embed_tokens = embed.release();
    }
    return Status::make_ok();
}

}  // namespace shelf::app
