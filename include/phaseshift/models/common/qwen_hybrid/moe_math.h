#pragma once

#include <phaseshift/core/memory/types.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ps {
namespace qwen_hybrid {

inline void moe_router_logits_f32(const float* x, const float* w,
                                  std::uint32_t tokens, std::uint32_t hidden,
                                  std::uint32_t experts, float* logits) {
    for (std::uint32_t t = 0; t < tokens; ++t) {
        const float* xr = x + static_cast<std::size_t>(t) * hidden;
        float* lr = logits + static_cast<std::size_t>(t) * experts;
        for (std::uint32_t e = 0; e < experts; ++e) {
            const float* wr = w + static_cast<std::size_t>(e) * hidden;
            float acc = 0.0f;
            for (std::uint32_t h = 0; h < hidden; ++h) {
                acc += xr[h] * wr[h];
            }
            lr[e] = acc;
        }
    }
}

inline void moe_softmax_f32(float* probs, std::uint32_t rows, std::uint32_t cols) {
    for (std::uint32_t r = 0; r < rows; ++r) {
        float* p = probs + static_cast<std::size_t>(r) * cols;
        float mx = p[0];
        for (std::uint32_t c = 1; c < cols; ++c) {
            if (p[c] > mx) mx = p[c];
        }
        float sum = 0.0f;
        for (std::uint32_t c = 0; c < cols; ++c) {
            p[c] = expf(p[c] - mx);
            sum += p[c];
        }
        for (std::uint32_t c = 0; c < cols; ++c) {
            p[c] /= sum;
        }
    }
}

inline void moe_topk_stable_f32(const float* probs, std::uint32_t rows,
                                std::uint32_t cols, std::uint32_t k,
                                std::uint32_t* indices, float* weights) {
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* p = probs + static_cast<std::size_t>(r) * cols;
        std::uint32_t* idx = indices + static_cast<std::size_t>(r) * k;
        float* w = weights + static_cast<std::size_t>(r) * k;
        std::uint32_t order[512];
        for (std::uint32_t c = 0; c < cols; ++c) order[c] = c;
        for (std::uint32_t a = 1; a < cols; ++a) {
            const std::uint32_t key = order[a];
            std::uint32_t b = a;
            while (b > 0) {
                const std::uint32_t prev = order[b - 1];
                const bool less = p[prev] < p[key] ||
                                  (p[prev] == p[key] && prev > key);
                if (!less) break;
                order[b] = prev;
                --b;
            }
            order[b] = key;
        }
        for (std::uint32_t s = 0; s < k; ++s) {
            idx[s] = order[s];
            w[s] = p[order[s]];
        }
    }
}

inline void moe_normalize_topk_f32(float* weights, std::uint32_t rows,
                                   std::uint32_t k) {
    for (std::uint32_t r = 0; r < rows; ++r) {
        float* w = weights + static_cast<std::size_t>(r) * k;
        float sum = 0.0f;
        for (std::uint32_t s = 0; s < k; ++s) sum += w[s];
        for (std::uint32_t s = 0; s < k; ++s) w[s] /= sum;
    }
}

inline void moe_f32_to_bf16(const float* in, std::uint16_t* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = f32_to_bf16_rne(in[i]);
}

inline void moe_expert_forward_f32(const float* x, const float* gate_up_e,
                                   const float* down_e, std::uint32_t inter,
                                   std::uint32_t hidden, float* out_e) {
    float gate[1024];
    float up[1024];
    float hidden_vec[1024];
    for (std::uint32_t r = 0; r < inter; ++r) {
        const float* row = gate_up_e + static_cast<std::size_t>(r) * hidden;
        float a = 0.0f;
        for (std::uint32_t h = 0; h < hidden; ++h) a += x[h] * row[h];
        gate[r] = a;
    }
    for (std::uint32_t r = 0; r < inter; ++r) {
        const float* row = gate_up_e + static_cast<std::size_t>(inter + r) * hidden;
        float a = 0.0f;
        for (std::uint32_t h = 0; h < hidden; ++h) a += x[h] * row[h];
        up[r] = a;
    }
    for (std::uint32_t i = 0; i < inter; ++i) {
        const float g = gate[i];
        hidden_vec[i] = g * (1.0f / (1.0f + expf(-g))) * up[i];
    }
    for (std::uint32_t h = 0; h < hidden; ++h) {
        const float* row = down_e + static_cast<std::size_t>(h) * inter;
        float a = 0.0f;
        for (std::uint32_t i = 0; i < inter; ++i) {
            a += hidden_vec[i] * row[i];
        }
        out_e[h] = a;
    }
}

inline void moe_shared_expert_f32(const float* x, std::uint32_t tokens,
                                  std::uint32_t hidden, std::uint32_t inter,
                                  const float* sg, const float* su,
                                  const float* sd, const float* scalar_w,
                                  float* shared_out, float* gate_out) {
    for (std::uint32_t t = 0; t < tokens; ++t) {
        const float* xr = x + static_cast<std::size_t>(t) * hidden;
        float gatev[1024];
        float upv[1024];
        for (std::uint32_t r = 0; r < inter; ++r) {
            const float* row = sg + static_cast<std::size_t>(r) * hidden;
            float a = 0.0f;
            for (std::uint32_t h = 0; h < hidden; ++h) a += xr[h] * row[h];
            gatev[r] = a;
        }
        for (std::uint32_t r = 0; r < inter; ++r) {
            const float* row = su + static_cast<std::size_t>(r) * hidden;
            float a = 0.0f;
            for (std::uint32_t h = 0; h < hidden; ++h) a += xr[h] * row[h];
            upv[r] = a;
        }
        float dot = 0.0f;
        for (std::uint32_t h = 0; h < hidden; ++h) dot += xr[h] * scalar_w[h];
        const float sgate = 1.0f / (1.0f + expf(-dot));
        if (gate_out != nullptr) gate_out[t] = sgate;
        float* sr = shared_out + static_cast<std::size_t>(t) * hidden;
        for (std::uint32_t h = 0; h < hidden; ++h) {
            const float* row = sd + static_cast<std::size_t>(h) * inter;
            float a = 0.0f;
            for (std::uint32_t i = 0; i < inter; ++i) {
                const float g = gatev[i];
                a += (g * (1.0f / (1.0f + expf(-g)))) * upv[i] * row[i];
            }
            sr[h] = sgate * a;
        }
    }
}

inline void moe_compose_f32(const float* routed, const float* shared,
                            std::size_t n, float* out) {
    for (std::size_t i = 0; i < n; ++i) out[i] = routed[i] + shared[i];
}

}
}
