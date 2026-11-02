#pragma once

#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/kernels/optimized/sampling_limits.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::kernel {

inline constexpr uint32_t kMaxSamplingAttempts = 4096u;
inline constexpr uint32_t kSamplingNoneMode = 0u;
inline constexpr uint32_t kSamplingGreedyMode = 1u;
inline constexpr uint32_t kSamplingStochasticMode = 2u;

__host__ __device__ __forceinline__ uint64_t sampling_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__host__ __device__ __forceinline__ uint64_t sampling_rng_bits(
    uint64_t seed,
    uint64_t sample_index,
    uint64_t attempt,
    uint64_t token_id) {
    uint64_t h = sampling_splitmix64(seed + 0x243F6A8885A308D3ull);
    h = sampling_splitmix64(h ^ (sample_index * 0x9E3779B97F4A7C15ull));
    h = sampling_splitmix64(h ^ (attempt * 0xC2B2AE3D27D4EB4Full));
    h = sampling_splitmix64(h ^ (token_id + 0x165667B19E3779F9ull));
    return h;
}

__host__ __device__ __forceinline__ float sampling_uniform(
    uint64_t seed,
    uint64_t sample_index,
    uint64_t attempt,
    uint64_t token_id) {
    const uint32_t bits =
        static_cast<uint32_t>(sampling_rng_bits(seed, sample_index, attempt, token_id));
    return (static_cast<float>(bits) + 0.5f) * (1.0f / 4294967296.0f);
}

__host__ __device__ __forceinline__ float sampling_gumbel(float u) {
    return -::logf(-::logf(u));
}

struct SamplingRowParams {
    uint32_t mode = kSamplingNoneMode;
    float temperature = 0.0f;
    float top_p = 1.0f;
    float min_p = 0.0f;
    uint32_t top_k = 0u;
    uint64_t seed = 0u;
    uint64_t sample_index = 0u;
};

struct SamplingTopKView {
    const int32_t* ids = nullptr;
    const float* logits = nullptr;
    uint32_t count = 0;
};

__host__ __device__ __forceinline__ SamplingRowParams sampling_row_params(
    const ::shelf::runtime::DeviceSamplingParams& p) {
    SamplingRowParams r;
    r.mode = p.mode;
    r.temperature = p.temperature;
    r.top_p = p.top_p;
    r.min_p = p.min_p;
    r.top_k = p.top_k;
    r.seed = p.seed;
    r.sample_index = p.sample_index;
    return r;
}

__host__ __device__ __forceinline__ bool sampling_topk_path_enabled(
    const ::shelf::runtime::DeviceSamplingParams& p) {
    return (p.reserved[0] & kSamplingTopKPathFlag) != 0u;
}

__device__ __forceinline__ void sampling_report_error(uint32_t* error_word) {
    if (error_word == nullptr) return;
    atomicCAS(error_word, 0u,
              static_cast<uint32_t>(::shelf::runtime::ProgramStatus::DISPATCH_FAILED));
}

namespace sampling_detail {

constexpr uint32_t kNoToken = 0xFFFFFFFFu;
constexpr float kNegInf = -3.4e38f;

__device__ __forceinline__ bool argmax_better(
    float cv, uint32_t ct, float bv, uint32_t bt) {
    return cv > bv || (cv == bv && ct < bt);
}

__device__ __forceinline__ bool min_p_membership(float weight, float min_p) {
    return min_p <= 0.0f || weight >= min_p;
}

__device__ __forceinline__ void block_argmax(
    float& value, uint32_t& token, float* sv, uint32_t* st) {
    const uint32_t n = blockDim.x;
    sv[threadIdx.x] = value;
    st[threadIdx.x] = token;
    __syncthreads();
    for (uint32_t off = n >> 1u; off > 0u; off >>= 1u) {
        if (threadIdx.x < off) {
            const float ov = sv[threadIdx.x + off];
            const uint32_t ot = st[threadIdx.x + off];
            if (argmax_better(ov, ot, sv[threadIdx.x], st[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                st[threadIdx.x] = ot;
            }
        }
        __syncthreads();
    }
    value = sv[0];
    token = st[0];
}

__device__ __forceinline__ float block_max(float value, float* sv) {
    const uint32_t n = blockDim.x;
    sv[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t off = n >> 1u; off > 0u; off >>= 1u) {
        if (threadIdx.x < off) {
            const float ov = sv[threadIdx.x + off];
            if (ov > sv[threadIdx.x]) sv[threadIdx.x] = ov;
        }
        __syncthreads();
    }
    return sv[0];
}

__device__ __forceinline__ double block_sum_f64(double value, double* sd) {
    const uint32_t n = blockDim.x;
    sd[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t off = n >> 1u; off > 0u; off >>= 1u) {
        if (threadIdx.x < off) sd[threadIdx.x] += sd[threadIdx.x + off];
        __syncthreads();
    }
    return sd[0];
}

__device__ __forceinline__ uint32_t block_sum_u32(uint32_t value, uint32_t* su) {
    const uint32_t n = blockDim.x;
    su[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t off = n >> 1u; off > 0u; off >>= 1u) {
        if (threadIdx.x < off) su[threadIdx.x] += su[threadIdx.x + off];
        __syncthreads();
    }
    return su[0];
}

}  // namespace sampling_detail

__device__ __forceinline__ void sampling_sample_row(
    const float* __restrict__ row,
    uint32_t vocab_size,
    const SamplingRowParams& params,
    int32_t* __restrict__ out_token,
    uint32_t* error_word,
    uint32_t* attempt_count = nullptr) {
    using namespace sampling_detail;

    __shared__ float s_val[1024];
    __shared__ uint32_t s_tok[1024];
    __shared__ double s_acc[1024];

    if (attempt_count != nullptr && threadIdx.x == 0u) *attempt_count = 0u;

    if (params.mode == kSamplingNoneMode) {
        if (threadIdx.x == 0u) *out_token = -1;
        __syncthreads();
        return;
    }

    if (params.mode == kSamplingGreedyMode) {
        float best_value = kNegInf;
        uint32_t best_token = kNoToken;
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            const float val = row[v];
            if (argmax_better(val, v, best_value, best_token)) {
                best_value = val;
                best_token = v;
            }
        }
        block_argmax(best_value, best_token, s_val, s_tok);
        if (threadIdx.x == 0u) {
            *out_token = best_token == kNoToken ? -1 : static_cast<int32_t>(best_token);
        }
        __syncthreads();
        if (best_token == kNoToken) {
            sampling_report_error(error_word);
        }
        return;
    }

    if (vocab_size == 0u || !(params.temperature > 0.0f)) {
        sampling_report_error(error_word);
        if (threadIdx.x == 0u) *out_token = -1;
        __syncthreads();
        return;
    }

    const float temperature = params.temperature;
    const float* logits = row;

    float lmax = kNegInf;
    for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float val = logits[v];
        if (val > lmax) lmax = val;
    }
    lmax = block_max(lmax, s_val);

    bool usable = isfinite(lmax) != 0;
    double z = 0.0;
    if (usable) {
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            z += static_cast<double>(::expf((logits[v] - lmax) / temperature));
        }
        z = block_sum_f64(z, s_acc);
        usable = (isfinite(z) != 0) && z > 0.0;
    }

    if (!usable) {
        sampling_report_error(error_word);
        if (threadIdx.x == 0u) *out_token = -1;
        __syncthreads();
        return;
    }

    const double top_p_mass = static_cast<double>(params.top_p) * z;

    int32_t result = -1;
    uint32_t attempt = 0u;
    while (attempt < kMaxSamplingAttempts) {
        float best_score = kNegInf;
        uint32_t best_token = kNoToken;
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            const float u =
                sampling_uniform(params.seed, params.sample_index, attempt, v);
            const float score =
                (logits[v] - lmax) / temperature + sampling_gumbel(u);
            if (argmax_better(score, v, best_score, best_token)) {
                best_score = score;
                best_token = v;
            }
        }
        block_argmax(best_score, best_token, s_val, s_tok);
        const uint32_t candidate = s_tok[0];
        if (candidate == kNoToken) {
            ++attempt;
            continue;
        }
        const float candidate_logit = logits[candidate];
        const float candidate_weight =
            ::expf((candidate_logit - lmax) / temperature);

        uint32_t better_count = 0u;
        double better_mass = 0.0;
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            const float val = logits[v];
            const bool better =
                (val > candidate_logit) || (val == candidate_logit && v < candidate);
            if (better) {
                ++better_count;
                better_mass +=
                    static_cast<double>(::expf((val - lmax) / temperature));
            }
        }
        better_count = block_sum_u32(better_count, s_tok);
        better_mass = block_sum_f64(better_mass, s_acc);

        const bool top_k_ok = (params.top_k == 0u) || (better_count < params.top_k);
        const bool top_p_ok =
            (params.top_p >= 1.0f) || (better_mass < top_p_mass);
        const bool min_p_ok = min_p_membership(candidate_weight, params.min_p);
        if (top_k_ok && top_p_ok && min_p_ok) {
            result = static_cast<int32_t>(candidate);
            break;
        }
        ++attempt;
    }

    if (result < 0) sampling_report_error(error_word);
    if (threadIdx.x == 0u) {
        *out_token = result;
        if (attempt_count != nullptr) {
            *attempt_count = (result < 0) ? kMaxSamplingAttempts : (attempt + 1u);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void sampling_topk_active_row(
    const float* __restrict__ row,
    uint32_t vocab_size,
    const SamplingRowParams& params,
    const SamplingTopKView& topk,
    int32_t* __restrict__ out_token,
    uint32_t* error_word,
    uint32_t* attempt_count = nullptr,
    uint32_t* active_count_out = nullptr) {
    using namespace sampling_detail;

    __shared__ float s_val[1024];
    __shared__ uint32_t s_tok[1024];
    __shared__ double s_acc[1024];
    __shared__ uint32_t s_active[kSamplingMaxTopK];

    if (attempt_count != nullptr && threadIdx.x == 0u) *attempt_count = 0u;

    if (params.mode == kSamplingNoneMode) {
        if (threadIdx.x == 0u) *out_token = -1;
        __syncthreads();
        return;
    }

    if (params.mode == kSamplingGreedyMode) {
        float best_value = kNegInf;
        uint32_t best_token = kNoToken;
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            const float val = row[v];
            if (argmax_better(val, v, best_value, best_token)) {
                best_value = val;
                best_token = v;
            }
        }
        block_argmax(best_value, best_token, s_val, s_tok);
        if (threadIdx.x == 0u) {
            *out_token = best_token == kNoToken ? -1 : static_cast<int32_t>(best_token);
        }
        __syncthreads();
        if (best_token == kNoToken) {
            sampling_report_error(error_word);
        }
        return;
    }

    const uint32_t count = topk.count;
    if (vocab_size == 0u || !(params.temperature > 0.0f) || topk.ids == nullptr ||
        topk.logits == nullptr || count == 0u || count > kSamplingMaxTopK) {
        sampling_report_error(error_word);
        if (threadIdx.x == 0u) *out_token = -1;
        __syncthreads();
        return;
    }

    const float temperature = params.temperature;

    float lmax = kNegInf;
    uint32_t nan_count = 0u;
    for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float val = row[v];
        if (val != val) nan_count = 1u;
        if (val > lmax) lmax = val;
    }
    lmax = block_max(lmax, s_val);
    nan_count = block_sum_u32(nan_count, s_tok);

    bool usable = (isfinite(lmax) != 0) && lmax > kNegInf && nan_count == 0u;
    double z = 0.0;
    if (usable && params.top_p < 1.0f) {
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            z += static_cast<double>(::expf((row[v] - lmax) / temperature));
        }
        z = block_sum_f64(z, s_acc);
        usable = (isfinite(z) != 0) && z > 0.0;
    }

    if (!usable) {
        sampling_report_error(error_word);
        if (threadIdx.x == 0u) *out_token = -1;
        __syncthreads();
        return;
    }

    const bool top_p_active = params.top_p < 1.0f;
    const bool min_p_active = params.min_p > 0.0f;
    const double top_p_mass = top_p_active ? static_cast<double>(params.top_p) * z : 0.0;
    if (threadIdx.x == 0u) {
        double prefix = 0.0;
        uint32_t active = 0u;
        for (uint32_t i = 0u; i < count; ++i) {
            const float weight = ::expf((topk.logits[i] - lmax) / temperature);
            const bool take =
                (!top_p_active || prefix < top_p_mass) &&
                (!min_p_active || min_p_membership(weight, params.min_p));
            s_active[i] = take ? 1u : 0u;
            if (take) ++active;
            prefix += static_cast<double>(weight);
        }
        if (active_count_out != nullptr) *active_count_out = active;
    }
    __syncthreads();

    float best_score = kNegInf;
    uint32_t best_token = kNoToken;
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        if (s_active[i] == 0u) continue;
        const uint32_t token = static_cast<uint32_t>(topk.ids[i]);
        const float u = sampling_uniform(params.seed, params.sample_index, 0u, token);
        const float score =
            (topk.logits[i] - lmax) / temperature + sampling_gumbel(u);
        if (argmax_better(score, token, best_score, best_token)) {
            best_score = score;
            best_token = token;
        }
    }
    block_argmax(best_score, best_token, s_val, s_tok);

    const int32_t result =
        best_token == kNoToken ? -1 : static_cast<int32_t>(best_token);
    if (result < 0) sampling_report_error(error_word);
    if (threadIdx.x == 0u) {
        *out_token = result;
        if (attempt_count != nullptr) {
            *attempt_count = (result < 0) ? kMaxSamplingAttempts : 1u;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void sampling_topk_select_reference(
    const float* __restrict__ row,
    uint32_t vocab_size,
    uint32_t count,
    int32_t* __restrict__ ids,
    float* __restrict__ values) {
    using namespace sampling_detail;

    __shared__ float s_val[1024];
    __shared__ uint32_t s_tok[1024];
    __shared__ uint32_t s_taken[kSamplingMaxTopK];

    if (threadIdx.x == 0u) {
        for (uint32_t i = 0u; i < count; ++i) s_taken[i] = 0xFFFFFFFFu;
    }
    __syncthreads();

    for (uint32_t it = 0u; it < count; ++it) {
        float best_value = kNegInf;
        uint32_t best_token = kNoToken;
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            bool taken = false;
            for (uint32_t t = 0u; t < it; ++t) {
                if (s_taken[t] == v) {
                    taken = true;
                    break;
                }
            }
            if (taken) continue;
            const float val = row[v];
            if (argmax_better(val, v, best_value, best_token)) {
                best_value = val;
                best_token = v;
            }
        }
        block_argmax(best_value, best_token, s_val, s_tok);
        if (threadIdx.x == 0u) {
            const uint32_t token = best_token == kNoToken ? 0u : best_token;
            s_taken[it] = token;
            ids[it] = static_cast<int32_t>(token);
            values[it] = best_token == kNoToken ? kNegInf : row[token];
        }
        __syncthreads();
    }
}

}  // namespace shelf::kernel
