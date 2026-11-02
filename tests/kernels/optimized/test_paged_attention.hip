#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/core/gpu/gfx_arch.h>
#include <shelf/kernels/kv_types.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/quantization/codec/e4m3_detail.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool g_wmma12 = false;

static hipError_t launch_paged_bf16(
    const ps::kernel::PagedAttentionCommonArgs& c, const ps::bf16_t* k_pool,
    const ps::bf16_t* v_pool, hipStream_t stream) {
    if (g_wmma12)
        return ps::kernel::launch_attention_paged_bf16(c, k_pool, v_pool, stream);
    return ps::kernel::launch_attention_paged_bf16(c, k_pool, v_pool, stream);
}

namespace {

using bf16 = __hip_bfloat16;
using fp8 = ::ps::kernel::fp8e4m3_storage_t;
using bf16t = ps::bf16_t;

float e4m3_decode_host(uint8_t x) {
    const int sign = (x & 0x80u) ? -1 : 1;
    const int exp = (x >> 3) & 0x0Fu;
    const int man = x & 0x07u;
    const float val = (exp == 0)
        ? (man * 0.125f) * std::ldexp(1.0f, -6)
        : (1.0f + man * 0.125f) * std::ldexp(1.0f, exp - 7);
    return sign * val;
}

int passed = 0;
int failed = 0;
int skipped_cases = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

uint32_t h32(uint64_t x) {
    uint32_t h = static_cast<uint32_t>(x * 2654435761u + 12345u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

uint16_t bf16_bits(double x) {
    return ps::f32_to_bf16_rne(static_cast<float>(x));
}

int32_t rng1k(uint64_t x) {
    return static_cast<int32_t>(h32(x) % 2001u) - 1000;
}

float f32_from_bf16(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

constexpr float kFp8E4M3Max = 448.0f;

struct Config {
    uint32_t q_heads = 8;
    uint32_t kv_heads = 4;
    uint32_t head_dim = 64;
    uint32_t page_tokens = 4;
    uint32_t num_pages = 8;
    uint32_t num_layers = 2;
    uint32_t layer = 1;
    uint32_t block_table_stride = 8;
    uint32_t max_sequences = 4;
    uint32_t visible_override = 0;
    uint32_t rows = 8;
    bool fp8 = false;
    bool prefill = false;
    bool zero_k = false;
    bool zero_v = false;
    float scale = 0.125f;
    uint32_t seed = 1;
    bool tree = false;
    uint32_t tree_committed_end = 0;
};

struct Dev {
    void* q = nullptr;
    void* out = nullptr;
    void* row_seq = nullptr;
    void* row_pos = nullptr;
    void* anc_off = nullptr;
    void* anc_pos = nullptr;
    void* block_tables = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* k_scale = nullptr;
    void* v_scale = nullptr;
};

void free_dev(Dev& d) {
    if (d.q) ps::gpu::discard_cleanup_result(hipFree(d.q));
    if (d.out) ps::gpu::discard_cleanup_result(hipFree(d.out));
    if (d.row_seq) ps::gpu::discard_cleanup_result(hipFree(d.row_seq));
    if (d.row_pos) ps::gpu::discard_cleanup_result(hipFree(d.row_pos));
    if (d.anc_off) ps::gpu::discard_cleanup_result(hipFree(d.anc_off));
    if (d.anc_pos) ps::gpu::discard_cleanup_result(hipFree(d.anc_pos));
    if (d.block_tables) ps::gpu::discard_cleanup_result(hipFree(d.block_tables));
    if (d.k_pool) ps::gpu::discard_cleanup_result(hipFree(d.k_pool));
    if (d.v_pool) ps::gpu::discard_cleanup_result(hipFree(d.v_pool));
    if (d.k_scale) ps::gpu::discard_cleanup_result(hipFree(d.k_scale));
    if (d.v_scale) ps::gpu::discard_cleanup_result(hipFree(d.v_scale));
    d = Dev{};
}

bool run_case(const Config& cfg, const char* label) {
    const uint32_t q_features = cfg.q_heads * cfg.head_dim;
    const uint32_t ept = cfg.kv_heads * cfg.head_dim;
    const uint32_t epp = cfg.page_tokens * ept;
    const uint32_t epl = cfg.num_pages * epp;
    const uint64_t pool_elems = static_cast<uint64_t>(cfg.num_layers) * epl;
    const uint32_t scale_elems = cfg.num_layers * cfg.num_pages * cfg.page_tokens * cfg.kv_heads;
    const size_t q_elems = static_cast<size_t>(cfg.rows) * q_features;
    const size_t bt_elems = static_cast<size_t>(cfg.max_sequences) * cfg.block_table_stride;
    const uint32_t q_per_kv = cfg.q_heads / cfg.kv_heads;

    std::vector<uint32_t> row_seq(cfg.rows), row_pos(cfg.rows), block_tables(bt_elems, 0);
    for (uint32_t r = 0; r < cfg.rows; ++r) {
        row_seq[r] = h32(cfg.seed * 1000u + r) % cfg.max_sequences;
        const uint32_t lb = r / cfg.page_tokens;
        uint32_t pos = lb * cfg.page_tokens;
        if (cfg.visible_override != 0u) {
            const uint32_t cap = cfg.page_tokens * cfg.num_pages;
            pos = cfg.visible_override > 1u ? cfg.visible_override - 1u : 0u;
            if (pos > cap - 1u) pos = cap - 1u;
        }
        row_pos[r] = pos;
    }
    for (uint32_t slot = 0; slot < cfg.max_sequences; ++slot)
        for (uint32_t lb = 0; lb < cfg.block_table_stride; ++lb)
            block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb] = lb % cfg.num_pages;

    std::vector<uint16_t> q(q_elems);
    for (size_t i = 0; i < q_elems; ++i)
        q[i] = bf16_bits((rng1k(cfg.seed * 9000u + i)) / 128.0);

    std::vector<uint8_t> k_pool, v_pool;
    std::vector<float> k_scale, v_scale;
    const uint8_t init_byte = cfg.fp8 ? 0xEE : 0x3F;
    if (cfg.fp8) {
        k_pool.assign(pool_elems, init_byte);
        v_pool.assign(pool_elems, init_byte);
        k_scale.assign(scale_elems, 0.0f);
        v_scale.assign(scale_elems, 0.0f);
        for (uint32_t page = 0; page < cfg.num_pages; ++page) {
            for (uint32_t off = 0; off < cfg.page_tokens; ++off) {
                for (uint32_t h = 0; h < cfg.kv_heads; ++h) {
                    const uint64_t hb = static_cast<uint64_t>(cfg.layer) * epl
                        + static_cast<uint64_t>(page) * epp
                        + static_cast<uint64_t>(off) * ept
                        + static_cast<uint64_t>(h) * cfg.head_dim;
                    const size_t si = (static_cast<size_t>(cfg.layer) * cfg.num_pages + page)
                        * cfg.page_tokens + off;
                    const size_t sih = si * cfg.kv_heads + h;
                    float k_amax = 0.f, v_amax = 0.f;
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        k_amax = std::fmax(k_amax, std::fabs(f32_from_bf16(
                            bf16_bits((rng1k(cfg.seed * 2000u + hb + d)) / 128.0))));
                        v_amax = std::fmax(v_amax, std::fabs(f32_from_bf16(
                            bf16_bits((rng1k(cfg.seed * 3000u + hb + d)) / 128.0))));
                    }
                    const float ks = k_amax == 0.f ? 1.f : k_amax / kFp8E4M3Max;
                    const float vs = v_amax == 0.f ? 1.f : v_amax / kFp8E4M3Max;
                    k_scale[sih] = ks;
                    v_scale[sih] = vs;
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        const float kf = f32_from_bf16(bf16_bits((rng1k(cfg.seed * 2000u + hb + d)) / 128.0));
                        const float vf = f32_from_bf16(bf16_bits((rng1k(cfg.seed * 3000u + hb + d)) / 128.0));
                        k_pool[hb + d] = cfg.zero_k ? 0u
                            : static_cast<uint8_t>(__hip_cvt_float_to_fp8(kf / ks, __HIP_SATFINITE, __HIP_E4M3));
                        v_pool[hb + d] = cfg.zero_v ? 0u
                            : static_cast<uint8_t>(__hip_cvt_float_to_fp8(vf / vs, __HIP_SATFINITE, __HIP_E4M3));
                    }
                }
            }
        }
    } else {
        k_pool.assign(pool_elems * 2, init_byte);
        v_pool.assign(pool_elems * 2, init_byte);
        for (uint32_t page = 0; page < cfg.num_pages; ++page) {
            for (uint32_t off = 0; off < cfg.page_tokens; ++off) {
                for (uint32_t h = 0; h < cfg.kv_heads; ++h) {
                    const uint64_t hb = static_cast<uint64_t>(cfg.layer) * epl
                        + static_cast<uint64_t>(page) * epp
                        + static_cast<uint64_t>(off) * ept
                        + static_cast<uint64_t>(h) * cfg.head_dim;
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        const uint16_t kb = cfg.zero_k ? 0u
                            : bf16_bits((rng1k(cfg.seed * 2000u + hb + d)) / 128.0);
                        const uint16_t vb = cfg.zero_v ? 0u
                            : bf16_bits((rng1k(cfg.seed * 3000u + hb + d)) / 128.0);
                        k_pool[hb * 2 + d * 2] = kb & 0xFF;
                        k_pool[hb * 2 + d * 2 + 1] = kb >> 8;
                        v_pool[hb * 2 + d * 2] = vb & 0xFF;
                        v_pool[hb * 2 + d * 2 + 1] = vb >> 8;
                    }
                }
            }
        }
    }

    std::vector<uint32_t> anc_offsets;
    std::vector<uint32_t> anc_positions;
    if (cfg.tree) {
        anc_offsets.resize(cfg.rows + 1u, 0u);
        for (uint32_t r = 0; r < cfg.rows; ++r) {
            const uint32_t depth = r + 1u;
            anc_offsets[r + 1u] = anc_offsets[r] + depth;
            for (uint32_t i = 0; i < depth; ++i)
                anc_positions.push_back(cfg.tree_committed_end + i);
        }
        const uint32_t garbage_base = cfg.tree_committed_end + cfg.rows + 2u;
        for (uint32_t gp = garbage_base; gp < garbage_base + 2u; ++gp) {
            const uint32_t lb = gp / cfg.page_tokens;
            const uint32_t off = gp % cfg.page_tokens;
            if (lb >= cfg.block_table_stride) continue;
            for (uint32_t slot = 0; slot < cfg.max_sequences; ++slot) {
                const uint32_t page = block_tables[static_cast<size_t>(slot) *
                                                   cfg.block_table_stride + lb];
                for (uint32_t h = 0; h < cfg.kv_heads; ++h) {
                    const uint64_t hb = static_cast<uint64_t>(cfg.layer) * epl
                        + static_cast<uint64_t>(page) * epp
                        + static_cast<uint64_t>(off) * ept
                        + static_cast<uint64_t>(h) * cfg.head_dim;
                    if (cfg.fp8) {
                        const size_t si = (static_cast<size_t>(cfg.layer) * cfg.num_pages + page)
                            * cfg.page_tokens + off;
                        const size_t sih = si * cfg.kv_heads + h;
                        k_scale[sih] = 1.0f;
                        v_scale[sih] = 1.0f;
                        for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                            k_pool[hb + d] = 0x7Eu;
                            v_pool[hb + d] = 0x7Eu;
                        }
                    } else {
                        const uint16_t kb = bf16_bits(400.0);
                        const uint16_t vb = bf16_bits(400.0);
                        for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                            k_pool[hb * 2 + d * 2] = kb & 0xFF;
                            k_pool[hb * 2 + d * 2 + 1] = kb >> 8;
                            v_pool[hb * 2 + d * 2] = vb & 0xFF;
                            v_pool[hb * 2 + d * 2 + 1] = vb >> 8;
                        }
                    }
                }
            }
        }
    }

    std::vector<float> ref(static_cast<size_t>(cfg.rows) * q_features, 0.0f);
    for (uint32_t r = 0; r < cfg.rows; ++r) {
        const uint32_t slot = row_seq[r];
        std::vector<uint32_t> keys;
        if (cfg.tree) {
            keys.reserve(cfg.tree_committed_end + (r + 1u));
            for (uint32_t p = 0; p < cfg.tree_committed_end; ++p) keys.push_back(p);
            for (uint32_t i = anc_offsets[r]; i < anc_offsets[r + 1u]; ++i)
                keys.push_back(anc_positions[i]);
        } else {
            const uint32_t visible = row_pos[r] + 1;
            keys.reserve(visible);
            for (uint32_t p = 0; p < visible; ++p) keys.push_back(p);
        }
        const uint32_t visible = static_cast<uint32_t>(keys.size());
        for (uint32_t qh = 0; qh < cfg.q_heads; ++qh) {
            const uint32_t kv_head = qh / q_per_kv;
            std::vector<float> qv(cfg.head_dim);
            for (uint32_t d = 0; d < cfg.head_dim; ++d)
                qv[d] = f32_from_bf16(q[static_cast<size_t>(r) * q_features + qh * cfg.head_dim + d]);
            if (cfg.fp8 && cfg.prefill) {
                float q_amax = 0.0f;
                for (uint32_t d = 0; d < cfg.head_dim; ++d)
                    q_amax = std::fmax(q_amax, std::fabs(qv[d]));
                const float q_scale = q_amax > 0.0f ? q_amax / kFp8E4M3Max : 1.0f;
                for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                    const uint8_t code = static_cast<uint8_t>(
                        __hip_cvt_float_to_fp8(qv[d] / q_scale, __HIP_SATFINITE, __HIP_E4M3));
                    qv[d] = e4m3_decode_host(code) * q_scale;
                }
            }
            float max_score = -3.402823466e+38f;
            std::vector<float> scores(visible);
            for (uint32_t ki = 0; ki < visible; ++ki) {
                const uint32_t p = keys[ki];
                const uint32_t lb = p / cfg.page_tokens;
                const uint32_t off = p % cfg.page_tokens;
                const uint32_t page = block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb];
                const uint64_t hb = static_cast<uint64_t>(cfg.layer) * epl
                    + static_cast<uint64_t>(page) * epp
                    + static_cast<uint64_t>(off) * ept
                    + static_cast<uint64_t>(kv_head) * cfg.head_dim;
                float dot = 0.f;
                if (cfg.fp8) {
                    const size_t si = (static_cast<size_t>(cfg.layer) * cfg.num_pages + page) * cfg.page_tokens + off;
                    const size_t sih = si * cfg.kv_heads + kv_head;
                    const float ks = k_scale[sih];
                    for (uint32_t d = 0; d < cfg.head_dim; ++d)
                        dot += qv[d] * e4m3_decode_host(k_pool[hb + d]);
                    scores[ki] = dot * ks * cfg.scale;
                } else {
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        const uint16_t kb = k_pool[hb * 2 + d * 2] | (k_pool[hb * 2 + d * 2 + 1] << 8);
                        dot += qv[d] * f32_from_bf16(kb);
                    }
                    scores[ki] = dot * cfg.scale;
                }
                max_score = std::fmax(max_score, scores[ki]);
            }
            float sum_exp = 0.f;
            std::vector<float> acc(cfg.head_dim, 0.f);
            for (uint32_t ki = 0; ki < visible; ++ki) {
                const float pv = std::exp(scores[ki] - max_score);
                sum_exp += pv;
                const uint32_t p = keys[ki];
                const uint32_t lb = p / cfg.page_tokens;
                const uint32_t off = p % cfg.page_tokens;
                const uint32_t page = block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb];
                const uint64_t hb = static_cast<uint64_t>(cfg.layer) * epl
                    + static_cast<uint64_t>(page) * epp
                    + static_cast<uint64_t>(off) * ept
                    + static_cast<uint64_t>(kv_head) * cfg.head_dim;
                if (cfg.fp8) {
                    const size_t si = (static_cast<size_t>(cfg.layer) * cfg.num_pages + page) * cfg.page_tokens + off;
                    const size_t sih = si * cfg.kv_heads + kv_head;
                    const float w = pv * v_scale[sih];
                    for (uint32_t d = 0; d < cfg.head_dim; ++d)
                        acc[d] += w * e4m3_decode_host(v_pool[hb + d]);
                } else {
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        const uint16_t vb = v_pool[hb * 2 + d * 2] | (v_pool[hb * 2 + d * 2 + 1] << 8);
                        acc[d] += pv * f32_from_bf16(vb);
                    }
                }
            }
            for (uint32_t d = 0; d < cfg.head_dim; ++d)
                ref[static_cast<size_t>(r) * q_features + qh * cfg.head_dim + d] = acc[d] / sum_exp;
        }
    }

    Dev d;
    bool ok = true;
    const size_t pool_bytes = cfg.fp8 ? pool_elems : pool_elems * 2;
    if (hipMalloc(&d.q, q_elems * 2) != hipSuccess) ok = false;
    if (hipMalloc(&d.out, q_elems * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.row_seq, cfg.rows * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.row_pos, cfg.rows * 4) != hipSuccess) ok = false;
    if (cfg.tree) {
        if (hipMalloc(&d.anc_off, (cfg.rows + 1u) * 4) != hipSuccess) ok = false;
        if (hipMalloc(&d.anc_pos, anc_positions.size() * 4) != hipSuccess) ok = false;
    }
    if (hipMalloc(&d.block_tables, bt_elems * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.k_pool, pool_bytes) != hipSuccess) ok = false;
    if (hipMalloc(&d.v_pool, pool_bytes) != hipSuccess) ok = false;
    if (cfg.fp8) {
        if (hipMalloc(&d.k_scale, scale_elems * 4) != hipSuccess) ok = false;
        if (hipMalloc(&d.v_scale, scale_elems * 4) != hipSuccess) ok = false;
    }
    if (!ok) { std::printf("FAIL: %s alloc\n", label); free_dev(d); check(false, label); return false; }

    if (hipMemcpy(d.q, q.data(), q_elems * 2, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemset(d.out, 0, q_elems * 4) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_seq, row_seq.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_pos, row_pos.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (cfg.tree) {
        if (hipMemcpy(d.anc_off, anc_offsets.data(), anc_offsets.size() * 4,
                      hipMemcpyHostToDevice) != hipSuccess) ok = false;
        if (hipMemcpy(d.anc_pos, anc_positions.data(), anc_positions.size() * 4,
                      hipMemcpyHostToDevice) != hipSuccess) ok = false;
    }
    if (hipMemcpy(d.block_tables, block_tables.data(), bt_elems * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.k_pool, k_pool.data(), pool_bytes, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.v_pool, v_pool.data(), pool_bytes, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (cfg.fp8) {
        if (hipMemcpy(d.k_scale, k_scale.data(), scale_elems * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
        if (hipMemcpy(d.v_scale, v_scale.data(), scale_elems * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    }
    if (!ok) { std::printf("FAIL: %s upload\n", label); free_dev(d); check(false, label); return false; }

    ::ps::kernel::PagedAttentionCommonArgs c;
    c.q = static_cast<const bf16t*>(d.q);
    c.output = d.out;
    c.output_dtype = ::ps::kernel::PagedAttentionOutputDType::F32;
    c.q_row_stride = q_features;
    c.output_row_stride = q_features;
    c.row_sequence_slots = static_cast<const uint32_t*>(d.row_seq);
    c.row_positions = static_cast<const uint32_t*>(d.row_pos);
    if (cfg.tree) {
        c.tree_anc_offsets = static_cast<const uint32_t*>(d.anc_off);
        c.tree_anc_positions = static_cast<const uint32_t*>(d.anc_pos);
        c.tree_committed_end = cfg.tree_committed_end;
    }
    c.block_tables = static_cast<const uint32_t*>(d.block_tables);
    c.block_table_stride = cfg.block_table_stride;
    c.max_sequences = cfg.max_sequences;
    c.rows = cfg.rows;
    c.layer = cfg.layer;
    c.page_tokens = cfg.page_tokens;
    c.num_pages = cfg.num_pages;
    c.num_attention_layers = cfg.num_layers;
    c.q_heads = cfg.q_heads;
    c.kv_heads = cfg.kv_heads;
    c.head_dim = cfg.head_dim;
    c.elems_per_token = ept;
    c.elems_per_page = epp;
    c.elems_per_layer = epl;
    c.scale = cfg.scale;

    hipError_t h = hipSuccess;
    if (cfg.prefill && cfg.fp8) {
        if (!g_wmma12) {
            std::printf("SKIP: %s (fp8 attention requires gfx12xx)\n", label);
            ++skipped_cases;
            return true;
        }
        h = ::ps::kernel::launch_attention_paged_prefill_fp8_e4m3(
            c, static_cast<const fp8*>(d.k_pool), static_cast<const fp8*>(d.v_pool),
            static_cast<const float*>(d.k_scale), static_cast<const float*>(d.v_scale), 0);
    } else if (cfg.prefill) {
        h = ::ps::kernel::launch_attention_paged_prefill_bf16(
            c, static_cast<const bf16t*>(d.k_pool), static_cast<const bf16t*>(d.v_pool), 0);
    } else if (cfg.fp8) {
        if (!g_wmma12) {
            std::printf("SKIP: %s (fp8 attention requires gfx12xx)\n", label);
            ++skipped_cases;
            return true;
        }
        h = ::ps::kernel::launch_attention_paged_fp8_e4m3(
            c, static_cast<const fp8*>(d.k_pool), static_cast<const fp8*>(d.v_pool),
            static_cast<const float*>(d.k_scale), static_cast<const float*>(d.v_scale), 0);
    } else {
        h = launch_paged_bf16(
            c, static_cast<const bf16t*>(d.k_pool), static_cast<const bf16t*>(d.v_pool), 0);
    }
    if (h == hipErrorNotSupported) {
        std::printf("SKIP: %s (wmma kernel not yet implemented for this arch)\n", label);
        free_dev(d);
        ++skipped_cases;
        return true;
    }
    ok = (h == hipSuccess) && (hipStreamSynchronize(0) == hipSuccess);
    if (!ok) {
        std::printf("FAIL: %s launch/sync (%s)\n", label, hipGetErrorString(h));
        free_dev(d);
        check(false, label);
        return false;
    }

    std::vector<float> got(q_elems);
    if (hipMemcpy(got.data(), d.out, q_elems * 4, hipMemcpyDeviceToHost) != hipSuccess) ok = false;
    free_dev(d);
    if (!ok) { check(false, label); return false; }

    size_t nonfinite = 0;
    for (size_t i = 0; i < q_elems; ++i) {
        if (!std::isfinite(static_cast<double>(got[i]))) ++nonfinite;
    }
    if (nonfinite != 0) ok = false;
    if (cfg.prefill && cfg.fp8) {
        double row_rel_sum = 0.0;
        double row_rel_max = 0.0;
        size_t row_count = 0;
        for (uint32_t r2 = 0; r2 < cfg.rows; ++r2) {
            for (uint32_t qh2 = 0; qh2 < cfg.q_heads; ++qh2) {
                double num = 0.0;
                double den = 0.0;
                for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                    const size_t idx = static_cast<size_t>(r2) * q_features
                        + static_cast<size_t>(qh2) * cfg.head_dim + d;
                    const double diff = static_cast<double>(got[idx]) - ref[idx];
                    num += diff * diff;
                    den += static_cast<double>(ref[idx]) * ref[idx];
                }
                const double rel = std::sqrt(num) / std::fmax(1.0, std::sqrt(den));
                row_rel_sum += rel;
                row_rel_max = std::fmax(row_rel_max, rel);
                ++row_count;
            }
        }
        const double row_rel_mean = row_count == 0u ? 0.0 : row_rel_sum / static_cast<double>(row_count);
        if (nonfinite != 0 || row_rel_mean > 1e-2 || row_rel_max > 0.6) ok = false;
        if (!ok) std::printf("  (nonfinite=%zu row_rel_mean=%.3e row_rel_max=%.3e)\n",
                             nonfinite, row_rel_mean, row_rel_max);
        check(ok, label);
        return ok;
    }
    double max_rel = 0.0;
    size_t mm = 0;
    const double tol = cfg.prefill ? 1e-2 : 2e-3;
    for (size_t i = 0; i < q_elems; ++i) {
        const double denom = std::fmax(1.0, std::fabs(static_cast<double>(ref[i])));
        const double rel = std::fabs(got[i] - ref[i]) / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > tol) ++mm;
    }
    if (mm != 0) ok = false;
    if (!ok) std::printf("  (max_rel=%.3e mm=%zu nonfinite=%zu)\n", max_rel, mm, nonfinite);
    check(ok, label);
    return ok;
}

}  // namespace

int main() {
    g_wmma12 = true;
    Config c;
    c.rows = 8;
    run_case(c, "bf16 GQA qh=8 kvh=4 hd=64 rows=8 causal");
    c = Config{};
    c.rows = 8;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.tree = true;
    c.tree_committed_end = 32;
    run_case(c, "bf16 tree visibility committed=32 rows=8 garbage excluded");
    c.fp8 = true;
    run_case(c, "fp8 tree visibility committed=32 rows=8 garbage excluded");
    c = Config{};
    c.rows = 1;
    c.q_heads = 4;
    c.kv_heads = 4;
    c.head_dim = 128;
    run_case(c, "bf16 MHA qh=kvh=4 hd=128 decode rows=1");
    c = Config{};
    c.rows = 16;
    c.q_heads = 8;
    c.kv_heads = 2;
    c.head_dim = 32;
    c.page_tokens = 8;
    run_case(c, "bf16 GQA qh=8 kvh=2 hd=32 page=8 rows=16");
    c = Config{};
    c.rows = 8;
    c.fp8 = true;
    run_case(c, "fp8 GQA qh=8 kvh=4 hd=64 rows=8 causal");
    c = Config{};
    c.rows = 1;
    c.fp8 = true;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 128;
    run_case(c, "fp8 GQA qh=4 kvh=2 hd=128 decode rows=1");
    c = Config{};
    c.rows = 24;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 64;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    run_case(c, "bf16 real-ish qh=16 kvh=4 hd=64 page=16 rows=24");
    c = Config{};
    c.rows = 4;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    run_case(c, "bf16 PREFILL qh=4 kvh=2 hd=256 page=16 rows=4");
    c = Config{};
    c.rows = 64;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    run_case(c, "bf16 PREFILL qh=4 kvh=2 hd=256 page=16 rows=64");
    c = Config{};
    c.rows = 4;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    c.fp8 = true;
    run_case(c, "fp8 PREFILL qh=4 kvh=2 hd=256 page=16 rows=4");
    c = Config{};
    c.rows = 64;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    c.fp8 = true;
    run_case(c, "fp8 PREFILL qh=4 kvh=2 hd=256 page=16 rows=64");
    c = Config{};
    c.rows = 64;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    c.zero_k = true;
    run_case(c, "bf16 PREFILL uniform K rows=64");
    c.zero_k = false;
    c.zero_v = true;
    run_case(c, "bf16 PREFILL zero V rows=64");
    c = Config{};
    c.rows = 64;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    c.fp8 = true;
    c.zero_k = true;
    run_case(c, "fp8 PREFILL uniform K rows=64");
    c.zero_k = false;
    c.zero_v = true;
    run_case(c, "fp8 PREFILL zero V rows=64");
    c = Config{};
    c.rows = 1;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 128;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 128;
    c.max_sequences = 1;
    c.visible_override = 8192;
    c.zero_k = true;
    run_case(c, "bf16 DECODE uniform K visible=8192");
    c.zero_k = false;
    c.zero_v = true;
    run_case(c, "bf16 DECODE zero V visible=8192");
    c = Config{};
    c.rows = 1;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 128;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 128;
    c.max_sequences = 1;
    c.visible_override = 8192;
    c.fp8 = true;
    c.zero_k = true;
    run_case(c, "fp8 DECODE uniform K visible=8192");
    c.zero_k = false;
    c.zero_v = true;
    run_case(c, "fp8 DECODE zero V visible=8192");
    c = Config{};
    c.rows = 1;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 128;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 128;
    c.max_sequences = 1;
    c.visible_override = 8192;
    run_case(c, "bf16 DECODE long visible=8192 hd=256 rows=1");
    c = Config{};
    c.rows = 1;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 128;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 128;
    c.max_sequences = 1;
    c.visible_override = 4097;
    run_case(c, "bf16 DECODE long visible=4097 hd=256 rows=1");
    c = Config{};
    c.rows = 1;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 512;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 512;
    c.max_sequences = 1;
    c.visible_override = 32768;
    run_case(c, "bf16 DECODE long visible=32768 hd=256 rows=1");

    // Qwen3.8-27B TP=2 local geometry: q_heads=12, kv_heads=2, head_dim=256, page=16.
    c = Config{};
    c.rows = 1;
    c.q_heads = 12;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 16;
    c.max_sequences = 1;
    c.visible_override = 64;
    run_case(c, "bf16 TP2 DECODE qh=12 kvh=2 hd=256 page=16 vis=64 rows=1");
    c = Config{};
    c.rows = 64;
    c.q_heads = 12;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    run_case(c, "bf16 TP2 PREFILL qh=12 kvh=2 hd=256 page=16 rows=64");
    c = Config{};
    c.rows = 1;
    c.q_heads = 12;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 64;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 64;
    c.max_sequences = 1;
    c.visible_override = 2048;
    run_case(c, "bf16 TP2 DECODE qh=12 kvh=2 hd=256 page=64 vis=2048 rows=1");

    std::printf("\nResults: %d passed, %d failed, %d skipped\n", passed, failed, skipped_cases);
    if (failed == 0 && passed == 0)
        return 77;
    return failed > 0 ? 1 : 0;
}
