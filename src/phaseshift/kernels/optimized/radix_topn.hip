#include <phaseshift/kernels/optimized/radix_topn.h>

#include "detail/coarse_topn_device.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace ps::kernel {

namespace {

constexpr uint32_t kThreads = kDflash2RadixTopnThreads;
constexpr uint32_t kSortThreads = kDflash2RadixSortThreads;
constexpr uint32_t kBins = kDflash2RadixTopnBins;
constexpr uint32_t kPasses = kDflash2RadixTopnPasses;

constexpr uint32_t kStThreshold = 0u;
constexpr uint32_t kStNeed = 1u;
constexpr uint32_t kStFlags = 2u;
constexpr uint32_t kStIdThr = 3u;
constexpr uint32_t kStCount = 4u;
constexpr uint32_t kStWords = 5u;

constexpr uint32_t kFlagDone = 1u;
constexpr uint32_t kFlagTie = 2u;

static_assert(kBins == kThreads, "bin per thread");

__device__ __forceinline__ uint32_t radix_ordered(float v) {
    const uint32_t bits = __float_as_uint(v);
    if (v != v)
        return 0u;
    if (v == 0.0f)
        return 0x80000000u;
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__global__ void __launch_bounds__(kThreads) radix_hist_kernel(
    const float* __restrict__ logits,
    uint32_t vocab,
    uint32_t row_stride,
    uint32_t parts,
    uint32_t part_size,
    uint32_t pass,
    const uint32_t* __restrict__ state,
    uint32_t* __restrict__ hist) {
    const uint32_t row = blockIdx.x / parts;
    const uint32_t part = blockIdx.x - row * parts;
    uint32_t prefix = 0u;
    if (pass != 0u) {
        if ((state[static_cast<std::size_t>(row) * kStWords + kStFlags] & kFlagDone) != 0u)
            return;
        prefix = state[static_cast<std::size_t>(row) * kStWords + kStThreshold];
    }
    const uint32_t shift = 24u - 8u * pass;
    const uint32_t mask = pass == 0u ? 0u : (0xFFFFFFFFu << (32u - 8u * pass));
    const float* rowp = logits + static_cast<std::size_t>(row) * row_stride;
    uint32_t beg = part * part_size;
    uint32_t end = beg + part_size;
    if (end > vocab)
        end = vocab;
    const uint32_t tid = threadIdx.x;
    __shared__ uint32_t local[kBins];
    for (uint32_t b = tid; b < kBins; b += kThreads)
        local[b] = 0u;
    __syncthreads();
    for (uint32_t v = beg + tid; v < end; v += kThreads) {
        const float value = rowp[v];
        if (value == -INFINITY)
            continue;
        const uint32_t ordered = radix_ordered(value);
        if (mask != 0u && (ordered & mask) != prefix)
            continue;
        atomicAdd(&local[(ordered >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    uint32_t* dst = hist + (static_cast<std::size_t>(row) * parts + part) * kBins;
    for (uint32_t b = tid; b < kBins; b += kThreads)
        dst[b] = local[b];
}

__device__ __forceinline__ void radix_pick_desc(const uint32_t* bins, uint32_t need,
                                                 uint32_t* bucket, uint32_t* rest) {
    uint32_t acc = 0u;
    uint32_t b = kBins;
    while (b-- > 0u) {
        if (acc + bins[b] >= need) {
            *bucket = b;
            *rest = need - acc;
            return;
        }
        acc += bins[b];
    }
    *bucket = 0u;
    *rest = need;
}

__global__ void __launch_bounds__(kThreads) radix_select_kernel(
    const float* __restrict__ logits,
    uint32_t vocab,
    uint32_t row_stride,
    uint32_t parts,
    uint32_t pass,
    uint32_t pool,
    uint32_t* __restrict__ state,
    const uint32_t* __restrict__ hist,
    uint32_t* __restrict__ bitmap) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    uint32_t* stp = state + static_cast<std::size_t>(row) * kStWords;
    if (pass + 1u == kPasses && tid == 0u)
        stp[kStCount] = 0u;
    if (pass != 0u && (stp[kStFlags] & kFlagDone) != 0u)
        return;

    __shared__ uint32_t sbin[kBins];
    const std::size_t base = static_cast<std::size_t>(row) * parts * kBins;
    uint32_t lane[8] = {};
    uint32_t p = 0u;
    for (; p + 8u <= parts; p += 8u) {
        #pragma unroll
        for (uint32_t a = 0u; a < 8u; ++a)
            lane[a] += hist[base + static_cast<std::size_t>(p + a) * kBins + tid];
    }
    uint32_t sum = 0u;
    #pragma unroll
    for (uint32_t a = 0u; a < 8u; ++a)
        sum += lane[a];
    for (; p < parts; ++p)
        sum += hist[base + static_cast<std::size_t>(p) * kBins + tid];
    sbin[tid] = sum;
    __syncthreads();

    const uint32_t need = pass == 0u ? pool : stp[kStNeed];
    const uint32_t prefix = pass == 0u ? 0u : stp[kStThreshold];
    if (tid == 0u) {
        uint32_t total = 0u;
        for (uint32_t b = 0; b < kBins; ++b)
            total += sbin[b];
        if (total < need) {
            stp[kStThreshold] = 0u;
            stp[kStNeed] = 0u;
            stp[kStFlags] = kFlagDone;
        } else {
            uint32_t bucket = 0u;
            uint32_t rest = 0u;
            radix_pick_desc(sbin, need, &bucket, &rest);
            stp[kStThreshold] = prefix | (bucket << (24u - 8u * pass));
            stp[kStNeed] = rest;
            stp[kStFlags] = 0u;
            if (pass + 1u == kPasses && rest < sbin[bucket])
                stp[kStFlags] = kFlagTie;
        }
    }
    __syncthreads();
    if (pass + 1u != kPasses || (stp[kStFlags] & kFlagTie) == 0u)
        return;

    __shared__ uint32_t pick[2];
    const uint32_t word_count = (vocab + 31u) / 32u;
    const uint32_t chunk = (word_count + kThreads - 1u) / kThreads;
    uint32_t* row_bits = bitmap + static_cast<std::size_t>(row) * word_count;
    const uint32_t target = stp[kStThreshold];
    const float* rowp = logits + static_cast<std::size_t>(row) * row_stride;
    for (uint32_t w = tid; w < word_count; w += kThreads) {
        const uint32_t v0 = w * 32u;
        uint32_t mask = 0u;
        for (uint32_t j = 0u; j < 32u; ++j) {
            const uint32_t v = v0 + j;
            if (v >= vocab)
                break;
            const float value = rowp[v];
            if (value == -INFINITY)
                continue;
            if (radix_ordered(value) == target)
                mask |= 1u << j;
        }
        row_bits[w] = mask;
    }
    __syncthreads();

    const uint32_t w0 = tid * chunk;
    const uint32_t w1 = w0 + chunk < word_count ? w0 + chunk : word_count;
    uint32_t local = 0u;
    for (uint32_t w = w0; w < w1; ++w)
        local += __builtin_popcount(row_bits[w]);
    sbin[tid] = local;
    __syncthreads();

    if (tid == 0u) {
        const uint32_t want = stp[kStNeed];
        uint32_t run = 0u;
        pick[0] = 0xFFFFFFFFu;
        pick[1] = 0u;
        for (uint32_t t = 0; t < kThreads; ++t) {
            if (run + sbin[t] >= want) {
                pick[0] = t;
                pick[1] = want - run;
                break;
            }
            run += sbin[t];
        }
    }
    __syncthreads();
    if (pick[0] == 0xFFFFFFFFu)
        return;

    if (tid == pick[0]) {
        const uint32_t want = pick[1];
        uint32_t run = 0u;
        uint32_t id_thr = 0u;
        for (uint32_t w = w0; w < w1; ++w) {
            const uint32_t word = row_bits[w];
            const uint32_t hits = __builtin_popcount(word);
            if (run + hits >= want) {
                uint32_t left = want - run;
                uint32_t b = 0u;
                for (; b < 32u; ++b) {
                    if ((word >> b) & 1u) {
                        --left;
                        if (left == 0u)
                            break;
                    }
                }
                id_thr = w * 32u + b;
                break;
            }
            run += hits;
        }
        stp[kStIdThr] = id_thr;
    }
}

__global__ void __launch_bounds__(kThreads) radix_compact_kernel(
    const float* __restrict__ logits,
    uint32_t vocab,
    uint32_t row_stride,
    uint32_t parts,
    uint32_t part_size,
    uint32_t pool,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_logits,
    uint32_t* __restrict__ state) {
    const uint32_t row = blockIdx.x / parts;
    const uint32_t part = blockIdx.x - row * parts;
    const uint32_t tid = threadIdx.x;
    const std::size_t st = static_cast<std::size_t>(row) * kStWords;
    const uint32_t flags = state[st + kStFlags];
    const uint32_t threshold = state[st + kStThreshold];
    const int32_t id_thr = static_cast<int32_t>(state[st + kStIdThr]);
    const bool take_all = (flags & kFlagDone) != 0u;
    const bool tie = (flags & kFlagTie) != 0u;
    const float* rowp = logits + static_cast<std::size_t>(row) * row_stride;
    uint32_t beg = part * part_size;
    uint32_t end = beg + part_size;
    if (end > vocab)
        end = vocab;
    for (uint32_t v = beg + tid; v < end; v += kThreads) {
        const float value = rowp[v];
        if (value == -INFINITY)
            continue;
        const uint32_t ordered = radix_ordered(value);
        bool keep = take_all || ordered > threshold;
        if (!keep && ordered == threshold)
            keep = !tie || static_cast<int32_t>(v) <= id_thr;
        if (!keep)
            continue;
        const uint32_t pos = atomicAdd(&state[st + kStCount], 1u);
        if (pos >= pool)
            continue;
        out_ids[static_cast<std::size_t>(row) * pool + pos] = static_cast<int32_t>(v);
        out_logits[static_cast<std::size_t>(row) * pool + pos] = value;
    }
}

__global__ void __launch_bounds__(kSortThreads) radix_sort_kernel(
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_logits,
    uint32_t pool,
    uint32_t sort_size,
    const uint32_t* __restrict__ state) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t count = state[static_cast<std::size_t>(row) * kStWords + kStCount];
    __shared__ uint64_t skey[kSortThreads];
    __shared__ float sval[kSortThreads];
    __shared__ int32_t sid[kSortThreads];

    uint64_t key = 0ull;
    float value = -INFINITY;
    int32_t id = 0;
    if (tid < pool && tid < count) {
        const std::size_t off = static_cast<std::size_t>(row) * pool + tid;
        id = out_ids[off];
        value = out_logits[off];
        key = detail::topn_key(value, id);
    }
    skey[tid] = key;
    sval[tid] = value;
    sid[tid] = id;
    __syncthreads();

    for (uint32_t k = 2u; k <= sort_size; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            if (tid < sort_size) {
                const uint32_t l = tid ^ j;
                if (l > tid && l < sort_size) {
                    const bool up = (tid & k) == 0u;
                    const uint64_t ki = skey[tid];
                    const uint64_t kl = skey[l];
                    const bool sw = up ? (ki < kl) : (ki > kl);
                    if (sw) {
                        skey[tid] = kl;
                        skey[l] = ki;
                        const float vi = sval[tid];
                        const float vl = sval[l];
                        const int32_t ii = sid[tid];
                        const int32_t il = sid[l];
                        sval[tid] = vl;
                        sval[l] = vi;
                        sid[tid] = il;
                        sid[l] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid < pool) {
        const std::size_t off = static_cast<std::size_t>(row) * pool + tid;
        out_ids[off] = sid[tid];
        out_logits[off] = sval[tid];
    }
}

}  // namespace

namespace {

std::size_t radix_hist_bytes(uint32_t rows, uint32_t partitions) {
    return static_cast<std::size_t>(rows) * partitions * kBins * sizeof(uint32_t);
}

std::size_t radix_align128(std::size_t bytes) { return (bytes + 127u) & ~std::size_t{127u}; }

std::size_t radix_state_offset(uint32_t rows, uint32_t partitions) {
    return radix_align128(radix_hist_bytes(rows, partitions));
}

std::size_t radix_bitmap_offset(uint32_t rows, uint32_t partitions) {
    return radix_align128(radix_state_offset(rows, partitions) +
                          static_cast<std::size_t>(rows) * kStWords * sizeof(uint32_t));
}

}  // namespace

bool dflash2_radix_topn_preferred(uint32_t pool) {
    static const int mode = [] {
        const char* env = std::getenv("PHASESHIFT_DFLASH2_RADIX_TOPN");
        if (env == nullptr || env[0] == '\0')
            return 1;
        return (env[0] == '0' && env[1] == '\0') ? 0 : 2;
    }();
    if (mode == 0)
        return false;
    if (mode == 2)
        return true;
    return pool >= kDflash2RadixTopnCrossoverPool;
}

std::size_t dflash2_radix_topn_scratch_bytes(uint32_t rows, uint32_t partitions,
                                             uint32_t vocab) {
    if (rows == 0u || partitions == 0u || vocab == 0u)
        return 0u;
    const std::size_t bitmap_words = (static_cast<std::size_t>(vocab) + 31u) / 32u;
    return radix_bitmap_offset(rows, partitions) +
           static_cast<std::size_t>(rows) * bitmap_words * sizeof(uint32_t);
}

hipError_t launch_dflash2_radix_topn(
    const float* logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t row_stride,
    uint32_t pool,
    int32_t* out_ids,
    float* out_logits,
    uint32_t partitions,
    void* scratch,
    std::size_t scratch_bytes,
    hipStream_t stream) {
    if (logits == nullptr || out_ids == nullptr || out_logits == nullptr || scratch == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u)
        return hipSuccess;
    if (pool == 0u || pool > kDflash2RadixTopnMaxPool)
        return hipErrorInvalidValue;
    if (vocab == 0u || row_stride < vocab)
        return hipErrorInvalidValue;
    if (partitions == 0u || partitions > kDflash2RadixMaxPartitions)
        return hipErrorInvalidValue;
    if (scratch_bytes < dflash2_radix_topn_scratch_bytes(rows, partitions, vocab))
        return hipErrorInvalidValue;

    auto* base = reinterpret_cast<std::byte*>(scratch);
    uint32_t* hist = reinterpret_cast<uint32_t*>(base);
    uint32_t* state = reinterpret_cast<uint32_t*>(base + radix_state_offset(rows, partitions));
    uint32_t* bitmap = reinterpret_cast<uint32_t*>(base + radix_bitmap_offset(rows, partitions));
    const uint32_t part_size = (vocab + partitions - 1u) / partitions;
    const uint32_t blocks = rows * partitions;
    uint32_t sort_size = 1u;
    while (sort_size < pool)
        sort_size <<= 1u;

    for (uint32_t pass = 0u; pass < kPasses; ++pass) {
        radix_hist_kernel<<<blocks, kThreads, 0, stream>>>(
            logits, vocab, row_stride, partitions, part_size, pass, state, hist);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
            return err;
        radix_select_kernel<<<rows, kThreads, 0, stream>>>(
            logits, vocab, row_stride, partitions, pass, pool, state, hist, bitmap);
        err = hipGetLastError();
        if (err != hipSuccess)
            return err;
    }

    radix_compact_kernel<<<blocks, kThreads, 0, stream>>>(
        logits, vocab, row_stride, partitions, part_size, pool, out_ids, out_logits, state);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        return err;
    radix_sort_kernel<<<rows, kSortThreads, 0, stream>>>(out_ids, out_logits, pool, sort_size,
                                                         state);
    return hipGetLastError();
}

}  // namespace ps::kernel
