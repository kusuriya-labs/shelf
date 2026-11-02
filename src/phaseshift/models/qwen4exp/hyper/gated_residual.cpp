#include <phaseshift/models/qwen4exp/hyper/gated_residual.h>

#include <cmath>
#include <cstdint>

namespace ps::qwen4exp::hyper {
namespace {

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float silu(float x) {
    return x * sigmoid(x);
}

Status validate_dims(const GatedResidualDims& dims) {
    if (dims.streams == 0u || dims.hidden == 0u || dims.lowrank == 0u) {
        return Status::invalid_argument("dims must be positive", __FILE__, __LINE__);
    }
    if (!(dims.eps > 0.0f)) {
        return Status::invalid_argument("eps must be positive", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status validate_read(const GatedResidualDims& dims, const GatedResidualWeights& weights,
                     const float* x4, std::uint32_t tokens, const float* mixed,
                     const float* inject) {
    const Status d = validate_dims(dims);
    if (!d.ok()) return d;
    if (weights.hc_norm == nullptr || weights.mix_down == nullptr ||
        weights.mix_up == nullptr || weights.block_inject == nullptr) {
        return Status::invalid_argument("weights must not be null", __FILE__, __LINE__);
    }
    if (tokens > 0u &&
        (x4 == nullptr || mixed == nullptr || inject == nullptr)) {
        return Status::invalid_argument("buffers must not be null", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

void grouped_rmsnorm_one_plus(const GatedResidualDims& dims, const float* weight,
                              const float* x, float* out) {
    const std::uint32_t hidden = dims.hidden;
    const std::uint32_t streams = dims.streams;
    for (std::uint32_t s = 0; s < streams; ++s) {
        const float* xs = x + static_cast<std::uint64_t>(s) * hidden;
        float* os = out + static_cast<std::uint64_t>(s) * hidden;
        const float* ws = weight + static_cast<std::uint64_t>(s) * hidden;
        float acc = 0.0f;
        for (std::uint32_t h = 0; h < hidden; ++h) {
            acc += xs[h] * xs[h];
        }
        const float rsqrt =
            1.0f / std::sqrt(acc / static_cast<float>(hidden) + dims.eps);
        for (std::uint32_t h = 0; h < hidden; ++h) {
            os[h] = (1.0f + ws[h]) * (xs[h] * rsqrt);
        }
    }
}

void lowrank_down(const GatedResidualDims& dims, const float* weight, const float* n,
                  float* out) {
    const std::uint32_t hc_hidden = dims.streams * dims.hidden;
    const std::uint32_t lowrank = dims.lowrank;
    for (std::uint32_t j = 0; j < lowrank; ++j) {
        const float* wrow = weight + static_cast<std::uint64_t>(j) * hc_hidden;
        float acc = 0.0f;
        for (std::uint32_t h = 0; h < hc_hidden; ++h) {
            acc += n[h] * wrow[h];
        }
        out[j] = acc;
    }
}

void lowrank_up(const GatedResidualDims& dims, const float* weight, const float* d,
                float* out) {
    const std::uint32_t hc_hidden = dims.streams * dims.hidden;
    const std::uint32_t lowrank = dims.lowrank;
    const float scale = 1.0f / static_cast<float>(dims.streams);
    for (std::uint32_t i = 0; i < hc_hidden; ++i) {
        const float* wrow = weight + static_cast<std::uint64_t>(i) * lowrank;
        float acc = 0.0f;
        for (std::uint32_t j = 0; j < lowrank; ++j) {
            acc += silu(d[j] * scale) * wrow[j];
        }
        out[i] = acc;
    }
}

}  // namespace

Status gated_residual_read(const GatedResidualDims& dims,
                           const GatedResidualWeights& weights,
                           const float* x4, std::uint32_t tokens,
                           float* mixed, float* inject,
                           GatedResidualReadTrace* trace) {
    const Status v = validate_read(dims, weights, x4, tokens, mixed, inject);
    if (!v.ok()) return v;

    const std::uint32_t hidden = dims.hidden;
    const std::uint32_t streams = dims.streams;
    const std::uint32_t hc_hidden = hidden * streams;
    const float stream_scale = 1.0f / static_cast<float>(streams);

    for (std::uint32_t t = 0; t < tokens; ++t) {
        const float* x = x4 + static_cast<std::uint64_t>(t) * hc_hidden;
        float* m = mixed + static_cast<std::uint64_t>(t) * hidden;
        float* inj = inject + static_cast<std::uint64_t>(t) * streams;

        std::vector<float> n(hc_hidden);
        grouped_rmsnorm_one_plus(dims, weights.hc_norm, x, n.data());

        std::vector<float> d(dims.lowrank);
        lowrank_down(dims, weights.mix_down, n.data(), d.data());

        std::vector<float> u(hc_hidden);
        lowrank_up(dims, weights.mix_up, d.data(), u.data());

        for (std::uint32_t h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (std::uint32_t s = 0; s < streams; ++s) {
                const std::uint32_t idx = s * hidden + h;
                acc += sigmoid(u[idx]) * n[idx];
            }
            m[h] = acc * stream_scale;
        }
        for (std::uint32_t s = 0; s < streams; ++s) {
            const float* wrow =
                weights.block_inject + static_cast<std::uint64_t>(s) * hc_hidden;
            float acc = 0.0f;
            for (std::uint32_t h = 0; h < hc_hidden; ++h) {
                acc += wrow[h] * n[h];
            }
            inj[s] = 2.0f * sigmoid(acc * stream_scale);
        }

        if (trace != nullptr) {
            trace->normed.insert(trace->normed.end(), n.begin(), n.end());
            trace->down.insert(trace->down.end(), d.begin(), d.end());
            trace->up.insert(trace->up.end(), u.begin(), u.end());
            for (std::uint32_t i = 0; i < hc_hidden; ++i) {
                trace->gate.push_back(sigmoid(u[i]));
            }
            trace->mixed.insert(trace->mixed.end(), m, m + hidden);
            trace->inject.insert(trace->inject.end(), inj, inj + streams);
        }
    }
    return Status::make_ok();
}

Status gated_residual_write(const GatedResidualDims& dims,
                            const float* x4, const float* sub_out,
                            const float* inject, std::uint32_t tokens,
                            float* out_x4) {
    const Status d = validate_dims(dims);
    if (!d.ok()) return d;
    if (tokens > 0u &&
        (x4 == nullptr || sub_out == nullptr || inject == nullptr || out_x4 == nullptr)) {
        return Status::invalid_argument("buffers must not be null", __FILE__, __LINE__);
    }

    const std::uint32_t hidden = dims.hidden;
    const std::uint32_t streams = dims.streams;
    const std::uint32_t hc_hidden = hidden * streams;
    for (std::uint32_t t = 0; t < tokens; ++t) {
        const float* x = x4 + static_cast<std::uint64_t>(t) * hc_hidden;
        const float* sub = sub_out + static_cast<std::uint64_t>(t) * hidden;
        const float* inj = inject + static_cast<std::uint64_t>(t) * streams;
        float* o = out_x4 + static_cast<std::uint64_t>(t) * hc_hidden;
        for (std::uint32_t s = 0; s < streams; ++s) {
            for (std::uint32_t h = 0; h < hidden; ++h) {
                const std::uint32_t idx = s * hidden + h;
                o[idx] = x[idx] + sub[h] * inj[s];
            }
        }
    }
    return Status::make_ok();
}

Status gated_residual_collapse(const GatedResidualDims& dims,
                               const GatedResidualCollapseWeights& weights,
                               const float* x4, std::uint32_t tokens,
                               float* out) {
    const Status d = validate_dims(dims);
    if (!d.ok()) return d;
    if (weights.hc_norm == nullptr || weights.mix_down == nullptr ||
        weights.mix_up == nullptr) {
        return Status::invalid_argument("weights must not be null", __FILE__, __LINE__);
    }
    if (tokens > 0u && (x4 == nullptr || out == nullptr)) {
        return Status::invalid_argument("buffers must not be null", __FILE__, __LINE__);
    }

    const std::uint32_t hidden = dims.hidden;
    const std::uint32_t streams = dims.streams;
    const std::uint32_t hc_hidden = hidden * streams;
    const float stream_scale = 1.0f / static_cast<float>(streams);

    for (std::uint32_t t = 0; t < tokens; ++t) {
        const float* x = x4 + static_cast<std::uint64_t>(t) * hc_hidden;
        float* o = out + static_cast<std::uint64_t>(t) * hidden;

        std::vector<float> n(hc_hidden);
        grouped_rmsnorm_one_plus(dims, weights.hc_norm, x, n.data());

        std::vector<float> low(dims.lowrank);
        lowrank_down(dims, weights.mix_down, n.data(), low.data());

        std::vector<float> u(hc_hidden);
        lowrank_up(dims, weights.mix_up, low.data(), u.data());

        for (std::uint32_t h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (std::uint32_t s = 0; s < streams; ++s) {
                const std::uint32_t idx = s * hidden + h;
                acc += sigmoid(u[idx]) * n[idx];
            }
            o[h] = acc * stream_scale;
        }
    }
    return Status::make_ok();
}

}  // namespace ps::qwen4exp::hyper
