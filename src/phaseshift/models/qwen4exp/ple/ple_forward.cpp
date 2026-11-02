#include <phaseshift/models/qwen4exp/ple/ple_forward.h>

#include <phaseshift/models/qwen4exp/ple/ple_conv.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ps::qwen4exp::ple {
namespace {

Status hip_fail(const char* what, hipError_t e) {
    return Status::hip_error(what, hipGetErrorString(e), __FILE__, __LINE__);
}

std::size_t element_count(const gpu::Tensor& t) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < t.ndim(); ++i) n *= t.dim(i);
    return n;
}

Status copy_device_bf16(const gpu::Tensor& t, std::vector<float>& out,
                        hipStream_t stream) {
    const std::size_t n = element_count(t);
    if (t.element_size() != sizeof(std::uint16_t) || n == 0) {
        return Status::invalid_argument("PLE weight must be non-empty BF16", __FILE__,
                                        __LINE__);
    }
    std::vector<std::uint16_t> bits(n);
    hipError_t e = hipMemcpyAsync(bits.data(), t.data<void>(),
                                  n * sizeof(std::uint16_t),
                                  hipMemcpyDeviceToHost, stream);
    if (e != hipSuccess) return hip_fail("PLE weight upload copy", e);
    e = hipStreamSynchronize(stream);
    if (e != hipSuccess) return hip_fail("PLE weight upload sync", e);
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t u = static_cast<std::uint32_t>(bits[i]) << 16;
        float f = 0.0f;
        std::memcpy(&f, &u, sizeof(f));
        out[i] = f;
    }
    return Status::make_ok();
}

Status copy_device_bf16_matrix(const weights::MatrixWeight& w, std::uint32_t rows,
                               std::uint32_t cols, std::vector<float>& out,
                               hipStream_t stream) {
    if (w.rows != rows || w.cols != cols) {
        return Status::invalid_argument("PLE projection geometry mismatch", __FILE__,
                                        __LINE__);
    }
    return copy_device_bf16(w.bf16, out, stream);
}

void grouped_rms_norm_one_plus(const float* x, const float* weight,
                               std::uint32_t groups, std::uint32_t group, float eps,
                               float* y) {
    const float inv_n = 1.0f / static_cast<float>(group);
    for (std::uint32_t g = 0; g < groups; ++g) {
        const float* row = x + static_cast<std::size_t>(g) * group;
        float* dst = y + static_cast<std::size_t>(g) * group;
        float ss = 0.0f;
        for (std::uint32_t i = 0; i < group; ++i) ss += row[i] * row[i];
        const float r = 1.0f / std::sqrt(ss * inv_n + eps);
        for (std::uint32_t i = 0; i < group; ++i) {
            dst[i] = row[i] * r * (1.0f + weight[i]);
        }
    }
}

void gemv(const float* w, const float* x, std::uint32_t rows, std::uint32_t cols,
          float* y) {
    for (std::uint32_t r = 0; r < rows; w += cols, ++r) {
        float acc = 0.0f;
        for (std::uint32_t c = 0; c < cols; ++c) acc += w[c] * x[c];
        y[r] = acc;
    }
}

Status validate_source(const PleRowSource& source, const NgramIdTables& tables) {
    std::int64_t max_row = 0;
    for (std::uint32_t h = 0; h < kNgramHeads; ++h) {
        max_row = std::max(max_row, tables.offsets[h] + tables.vocab_sizes[h] - 1);
    }
    switch (source.kind) {
        case PleRowSource::Kind::vram:
            if (source.device_table == nullptr || source.table_rows == 0) {
                return Status::invalid_argument("VRAM PLE source needs a table", __FILE__,
                                                __LINE__);
            }
            if (static_cast<std::uint64_t>(max_row) >= source.table_rows) {
                return Status::out_of_range("PLE row id outside the VRAM table", __FILE__,
                                            __LINE__);
            }
            break;
        case PleRowSource::Kind::ram:
        case PleRowSource::Kind::ssd:
            if (source.collection == nullptr) {
                return Status::invalid_argument("file PLE source needs a collection",
                                                __FILE__, __LINE__);
            }
            break;
    }
    if (source.layout.row_cols == 0 || source.layout.row_bytes == 0) {
        return Status::invalid_argument("PLE row layout is empty", __FILE__, __LINE__);
    }
    if (source.scale_layout.shard_count != 0) {
        if (source.collection == nullptr) {
            return Status::invalid_argument("FP8 PLE source needs a collection", __FILE__,
                                            __LINE__);
        }
        if (source.scale_layout.row_bytes != 2 ||
            source.scale_layout.shard_count != source.layout.shard_count ||
            source.scale_layout.rows_per_shard != source.layout.rows_per_shard) {
            return Status::invalid_argument("PLE scale layout mismatch", __FILE__,
                                            __LINE__);
        }
    }
    return Status::make_ok();
}

float e4m3_to_f32(std::uint8_t x) {
    const float sign = (x & 0x80u) != 0u ? -1.0f : 1.0f;
    const std::uint32_t exponent = (x >> 3) & 0x0Fu;
    const std::uint32_t mantissa = x & 0x07u;
    if (exponent == 0u) return sign * static_cast<float>(mantissa) * (1.0f / 512.0f);
    if (exponent == 0x0Fu && mantissa == 0x07u) return 0.0f;
    return sign * (1.0f + static_cast<float>(mantissa) * (1.0f / 8.0f)) *
           std::ldexp(1.0f, static_cast<int>(exponent) - 7);
}

Status load_row_scale(const TableLayout& layout,
                      const weights::SafetensorsCollection& collection,
                      std::uint64_t global_row, float& out) {
    const std::uint64_t shard = global_row / layout.rows_per_shard;
    const std::uint64_t row = global_row % layout.rows_per_shard;
    const std::string name =
        layout.tensor_prefix + std::to_string(shard) + layout.tensor_suffix;
    std::size_t bytes = 0;
    auto data = collection.tensor_data(name, bytes);
    if (!data.ok()) return data.status();
    if ((row + 1u) * layout.row_bytes > bytes) {
        return Status::out_of_range("PLE scale row outside tensor payload", __FILE__,
                                    __LINE__);
    }
    std::uint16_t bits = 0;
    std::memcpy(&bits, static_cast<const char*>(data.value()) + row * layout.row_bytes,
                sizeof(bits));
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    out = f;
    return Status::make_ok();
}

}  // namespace

Result<PleBranch> PleBranch::create(const Qwen4ExpTextConfig& config,
                                    const Qwen4ExpPleWeights& weights,
                                    std::uint32_t max_tokens,
                                    const PleRowSource& source, hipStream_t stream) {
    if (!weights.present) {
        return Status::invalid_state("PLE weights are not loaded", __FILE__, __LINE__);
    }
    if (max_tokens == 0) {
        return Status::invalid_argument("max_tokens must be positive", __FILE__, __LINE__);
    }
    if (config.hc_count == 0 || config.hidden_size == 0 || config.ple_embed_dim == 0) {
        return Status::invalid_argument("invalid PLE geometry", __FILE__, __LINE__);
    }
    if (config.ple_embed_dim != config.hidden_size) {
        return Status::invalid_argument("PLE embed dim must equal hidden size", __FILE__,
                                        __LINE__);
    }
    if (config.ple_conv_kernel_size != 0 &&
        config.ple_conv_kernel_size != kPleConvKernel) {
        return Status::invalid_argument("PLE convolution kernel mismatch", __FILE__,
                                        __LINE__);
    }
    const std::int32_t eos = qwen35::primary_stop_token(config.stop_tokens);
    if (eos < 0) {
        return Status::invalid_argument("config has no eos token", __FILE__, __LINE__);
    }

    PleBranch b;
    b.embed_dim_ = static_cast<std::uint32_t>(config.ple_embed_dim);
    b.hidden_ = static_cast<std::uint32_t>(config.hidden_size);
    b.streams_ = static_cast<std::uint32_t>(config.hc_count);
    b.max_tokens_ = max_tokens;
    b.eos_token_ = static_cast<std::uint32_t>(eos);
    b.eps_ = config.rms_norm_eps != 0.0f ? config.rms_norm_eps : 1.0e-6f;
    b.source_ = source;

    const std::uint32_t hc_hidden = b.streams_ * b.hidden_;
    if (source.layout.row_cols != b.embed_dim_ / kNgramHeads) {
        return Status::invalid_argument("PLE row layout does not match embed dim",
                                        __FILE__, __LINE__);
    }

    Status st = copy_device_bf16_matrix(weights.key_proj, hc_hidden, b.embed_dim_,
                                        b.key_proj_, stream);
    if (!st.ok()) return st;
    st = copy_device_bf16_matrix(weights.value_proj, b.embed_dim_, b.embed_dim_,
                                 b.value_proj_, stream);
    if (!st.ok()) return st;
    st = copy_device_bf16(weights.conv1d, b.conv_weight_, stream);
    if (!st.ok()) return st;
    if (b.conv_weight_.size() !=
        static_cast<std::size_t>(hc_hidden) * kPleConvKernel) {
        return Status::invalid_argument("PLE convolution weight size mismatch", __FILE__,
                                        __LINE__);
    }
    st = copy_device_bf16(weights.norm_key, b.norm_key_, stream);
    if (!st.ok()) return st;
    st = copy_device_bf16(weights.norm_query, b.norm_query_, stream);
    if (!st.ok()) return st;
    st = copy_device_bf16(weights.norm_conv, b.norm_conv_, stream);
    if (!st.ok()) return st;
    for (const std::vector<float>* v : {&b.norm_key_, &b.norm_query_, &b.norm_conv_}) {
        if (v->size() != hc_hidden) {
            return Status::invalid_argument("PLE norm weight size mismatch", __FILE__,
                                            __LINE__);
        }
    }

    if (weights.vocab_sizes.size() == kNgramHeads &&
        weights.offsets.size() == kNgramHeads &&
        weights.layer_multipliers.size() == kNgramSize) {
        for (std::uint32_t h = 0; h < kNgramHeads; ++h) {
            b.tables_.vocab_sizes[h] = weights.vocab_sizes[h];
            b.tables_.offsets[h] = weights.offsets[h];
        }
        for (std::uint32_t k = 0; k < kNgramSize; ++k) {
            b.tables_.multipliers[k] = weights.layer_multipliers[k];
        }
    } else {
        const NgramHashParams params{
            .seed = config.ngram_seed,
            .unigram_vocab_size = config.vocab_size,
            .ple_layer_slot = static_cast<std::uint32_t>(config.ple_slot),
        };
        st = build_ngram_id_tables(params, b.tables_);
        if (!st.ok()) return st;
    }
    st = validate_source(source, b.tables_);
    if (!st.ok()) return st;

    const std::size_t row_ids_bytes =
        static_cast<std::size_t>(max_tokens) * kNgramHeads * sizeof(std::uint64_t);
    const std::size_t rows_bytes = static_cast<std::size_t>(max_tokens) *
                                   b.embed_dim_ * sizeof(std::uint16_t);
    void* raw = nullptr;
    hipError_t e = hipMalloc(&raw, row_ids_bytes);
    if (e != hipSuccess) return hip_fail("PLE row id buffer", e);
    b.row_ids_dev_.reset(raw);
    e = hipMalloc(&raw, rows_bytes);
    if (e != hipSuccess) {
        (void)b.release_device();
        return hip_fail("PLE row buffer", e);
    }
    b.rows_dev_.reset(raw);

    return b;
}

Status PleBranch::release_device() {
    Status st = Status::make_ok();
    if (row_ids_dev_) {
        const hipError_t e = hipFree(row_ids_dev_.get());
        row_ids_dev_.reset();
        if (e != hipSuccess) st = hip_fail("PLE row id buffer release", e);
    }
    if (rows_dev_) {
        const hipError_t e = hipFree(rows_dev_.get());
        rows_dev_.reset();
        if (e != hipSuccess) st = hip_fail("PLE row buffer release", e);
    }
    return st;
}

Status PleBranch::shutdown() {
    if (shutdown_) return Status::make_ok();
    shutdown_ = true;
    return release_device();
}

Status PleBranch::run(const std::uint32_t* input_ids, std::uint32_t count,
                      const float* hidden_states,
                      ::ps::qwen4exp::runtime::PleNgramContext& ngram,
                      ::ps::qwen4exp::runtime::PleConvStatePool& conv,
                      ::ps::state::SequenceSlotId slot, float* out_host,
                      hipStream_t stream) {
    if (shutdown_) return Status::invalid_state("PLE branch is shut down", __FILE__, __LINE__);
    if (count == 0 || count > max_tokens_) {
        return Status::out_of_range("PLE token count outside the prepared window",
                                    __FILE__, __LINE__);
    }
    if (input_ids == nullptr || hidden_states == nullptr || out_host == nullptr) {
        return Status::invalid_argument("PLE run buffers must not be null", __FILE__,
                                        __LINE__);
    }
    if (!ngram.is_initialized(slot) || !conv.is_initialized(slot)) {
        return Status::invalid_state("PLE slot state is not initialized", __FILE__,
                                     __LINE__);
    }

    const std::uint32_t hc_hidden = streams_ * hidden_;
    std::vector<std::uint64_t> row_ids(static_cast<std::size_t>(count) * kNgramHeads);
    Status st = compute_ngram_ids(tables_, ngram.context(slot), kNgramContextTokens,
                                  input_ids, count, eos_token_, row_ids.data());
    if (!st.ok()) return st;

    const std::size_t ids_bytes = row_ids.size() * sizeof(std::uint64_t);
    hipError_t e = hipMemcpy(row_ids_dev_.get(), row_ids.data(), ids_bytes,
                             hipMemcpyHostToDevice);
    if (e != hipSuccess) return hip_fail("PLE row id upload", e);

    const std::uint32_t gathered = count * kNgramHeads;
    switch (source_.kind) {
        case PleRowSource::Kind::vram:
            st = gather_rows_vram(source_.layout, source_.device_table,
                                  source_.table_rows,
                                  static_cast<const std::uint64_t*>(row_ids_dev_.get()),
                                  gathered, rows_dev_.get(), stream);
            break;
        case PleRowSource::Kind::ram:
            st = gather_rows_ram(source_.layout, *source_.collection, row_ids.data(),
                                 gathered, rows_dev_.get(), stream);
            break;
        case PleRowSource::Kind::ssd: {
            SsdGatherStats stats{};
            st = gather_rows_ssd(source_.layout, *source_.collection, row_ids.data(),
                                 gathered, rows_dev_.get(), stream, stats,
                                 source_.allow_hipfile);
            break;
        }
    }
    if (!st.ok()) return st;
    e = hipStreamSynchronize(stream);
    if (e != hipSuccess) return hip_fail("PLE row gather", e);

    std::vector<float> embed(static_cast<std::size_t>(count) * embed_dim_);
    if (source_.scale_layout.shard_count != 0) {
        const std::size_t gathered_bytes =
            static_cast<std::size_t>(gathered) * source_.layout.row_bytes;
        std::vector<std::uint8_t> raw(gathered_bytes);
        e = hipMemcpy(raw.data(), rows_dev_.get(), raw.size(), hipMemcpyDeviceToHost);
        if (e != hipSuccess) return hip_fail("PLE row download", e);
        for (std::uint32_t r = 0; r < gathered; ++r) {
            float scale = 1.0f;
            st = load_row_scale(source_.scale_layout, *source_.collection, row_ids[r],
                                scale);
            if (!st.ok()) return st;
            const std::size_t base = static_cast<std::size_t>(r) * source_.layout.row_bytes;
            for (std::uint64_t c = 0; c < source_.layout.row_cols; ++c) {
                embed[static_cast<std::size_t>(r) * source_.layout.row_cols + c] =
                    e4m3_to_f32(raw[base + c]) * scale;
            }
        }
    } else {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(count) * embed_dim_);
        e = hipMemcpy(bits.data(), rows_dev_.get(), bits.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost);
        if (e != hipSuccess) return hip_fail("PLE row download", e);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            const std::uint32_t u = static_cast<std::uint32_t>(bits[i]) << 16;
            float f = 0.0f;
            std::memcpy(&f, &u, sizeof(f));
            embed[i] = f;
        }
    }

    std::vector<float> key(static_cast<std::size_t>(count) * hc_hidden);
    std::vector<float> value(static_cast<std::size_t>(count) * embed_dim_);
    std::vector<float> query(static_cast<std::size_t>(count) * hc_hidden);
    for (std::uint32_t t = 0; t < count; ++t) {
        gemv(key_proj_.data(), embed.data() + static_cast<std::size_t>(t) * embed_dim_,
             hc_hidden, embed_dim_, key.data() + static_cast<std::size_t>(t) * hc_hidden);
        gemv(value_proj_.data(),
             embed.data() + static_cast<std::size_t>(t) * embed_dim_, embed_dim_,
             embed_dim_, value.data() + static_cast<std::size_t>(t) * embed_dim_);
        std::memcpy(query.data() + static_cast<std::size_t>(t) * hc_hidden,
                    hidden_states + static_cast<std::size_t>(t) * hc_hidden,
                    static_cast<std::size_t>(hc_hidden) * sizeof(float));
    }
    grouped_rms_norm_one_plus(key.data(), norm_key_.data(), count * streams_, hidden_,
                              eps_, key.data());
    grouped_rms_norm_one_plus(query.data(), norm_query_.data(), count * streams_,
                              hidden_, eps_, query.data());

    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hidden_));
    std::vector<float> gated(static_cast<std::size_t>(count) * hc_hidden);
    std::vector<float> normed(static_cast<std::size_t>(count) * hc_hidden);
    std::vector<float> conv_out(static_cast<std::size_t>(count) * hc_hidden);
    for (std::uint32_t t = 0; t < count; ++t) {
        const float* key_row = key.data() + static_cast<std::size_t>(t) * hc_hidden;
        const float* query_row = query.data() + static_cast<std::size_t>(t) * hc_hidden;
        const float* value_row = value.data() + static_cast<std::size_t>(t) * embed_dim_;
        float* gated_row = gated.data() + static_cast<std::size_t>(t) * hc_hidden;
        for (std::uint32_t s = 0; s < streams_; ++s) {
            float acc = 0.0f;
            for (std::uint32_t f = 0; f < hidden_; ++f) {
                const std::size_t i = static_cast<std::size_t>(s) * hidden_ + f;
                acc += key_row[i] * query_row[i];
            }
            const float raw = acc * inv_sqrt;
            const float shaped =
                std::sqrt(std::max(std::fabs(raw), 1.0e-6f)) * (raw < 0.0f ? -1.0f : 1.0f);
            const float sig = 1.0f / (1.0f + std::exp(-shaped));
            for (std::uint32_t f = 0; f < hidden_; ++f) {
                gated_row[static_cast<std::size_t>(s) * hidden_ + f] =
                    sig * value_row[f];
            }
        }
    }
    grouped_rms_norm_one_plus(gated.data(), norm_conv_.data(), count * streams_,
                              hidden_, eps_, normed.data());

    st = conv.convolve(slot, conv_weight_.data(), normed.data(), count, conv_out.data());
    if (!st.ok()) return st;
    for (std::size_t i = 0; i < conv_out.size(); ++i) {
        const float x = conv_out[i];
        conv_out[i] = x / (1.0f + std::exp(-x));
    }
    for (std::size_t i = 0; i < gated.size(); ++i) out_host[i] = gated[i] + conv_out[i];

    return ngram.advance(slot, input_ids, count);
}

}  // namespace ps::qwen4exp::ple
