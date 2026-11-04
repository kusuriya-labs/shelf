#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/kernels/kv_types.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/quantization/codec/e4m3_detail.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/program/program.h>
#include <shelf/models/qwen35/runtime/selector_types.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

using bf16 = __hip_bfloat16;
using fp8 = ::shelf::kernel::fp8e4m3_storage_t;
constexpr float kFp8E4M3Max = 448.0f;

struct Profile {
    bool fp8 = false;
    bool f32_output = true;
    uint32_t q_heads = 16;
    uint32_t kv_heads = 4;
    uint32_t head_dim = 256;
    uint32_t page_tokens = 16;
    uint32_t num_pages = 256;
    uint32_t num_layers = 1;
    uint32_t layer = 0;
    uint32_t block_table_stride = 256;
    uint32_t max_sequences = 8;
    float scale = 0.0625f;
};

struct Options {
    std::string variant = "optimized";
    std::vector<uint32_t> rows = {1, 4, 16, 64, 256, 1024, 2048};
    std::vector<uint32_t> visible = {1, 16, 64, 256, 1024, 4096};
    bool fp8 = false;
    bool f32_output = true;
    uint32_t q_heads = 16;
    uint32_t kv_heads = 4;
    uint32_t head_dim = 256;
    uint32_t page_tokens = 16;
    uint32_t splits = 1;
    std::string dump_out;
    int warmup = 50;
    int samples = 30;
    int launches = 10;
    uint32_t seed = 1;
    int device = 0;
    bool check = false;
    std::string out;
    std::string model_dir;
    bool actual_qwen_profiles = false;
    double arena_gib = 12.0;
};

void usage(FILE* f) {
    std::fprintf(f,
                  "usage: shelf-bench paged-attention [options]\n"
                  "  --variant correctness|optimized|prefill|auto\n"
                  "  --splits N (optimized only, bf16 decode: KV split + reduce, N in 2/4/8/16)\n"
                  "  --rows 1,4,...\n"
                  "  --visible 1,16,64,...\n"
                  "  --kv-dtype bf16|fp8\n"
                  "  --out-dtype bf16|f32\n"
                  "  --q-heads N\n"
                  "  --kv-heads N\n"
                  "  --head-dim N\n"
                  "  --page-tokens N\n"
                  "  --warmup N\n"
                  "  --samples N\n"
                  "  --launches N\n"
                  "  --seed N\n"
                  "  --device N\n"
                  "  --check\n"
                  "  --out PATH\n"
                  "  --model-dir PATH\n"
                  "  --actual-qwen-profiles\n"
                  "  --arena-gib N\n");
}

bool parse_list(const std::string& s, std::vector<uint32_t>& out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        size_t j = i;
        while (j < s.size() && std::isdigit((unsigned char)s[j])) ++j;
        if (j == i) return false;
        out.push_back((uint32_t)std::stoul(s.substr(i, j - i)));
        i = j;
        if (i < s.size() && s[i] == ',') ++i;
    }
    return !out.empty();
}

uint32_t h32(uint64_t x) {
    uint32_t h = (uint32_t)(x * 2654435761u + 12345u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

int32_t rng1k(uint64_t x) {
    return (int32_t)(h32(x) % 2001u) - 1000;
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    const double idx = p * (double)(v.size() - 1);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min((size_t)std::ceil(idx), (size_t)v.size() - 1);
    const double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

float f32_from_bf16(uint16_t bits) {
    uint32_t ui = (uint32_t)bits << 16;
    float r;
    std::memcpy(&r, &ui, 4);
    return r;
}

uint16_t bf16_bits(double x) {
    return shelf::f32_to_bf16_rne((float)x);
}

float e4m3_decode_host(uint8_t x) {
    const int sign = (x & 0x80u) ? -1 : 1;
    const int exp = (x >> 3) & 0x0Fu;
    const int man = x & 0x07u;
    const float val = (exp == 0)
        ? (man * 0.125f) * std::ldexp(1.0f, -6)
        : (1.0f + man * 0.125f) * std::ldexp(1.0f, exp - 7);
    return sign * val;
}

struct Frag {
    void* values_dev = nullptr;
    void* states_dev = nullptr;
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* error_word = nullptr;
    ::shelf::runtime::DeviceProgramView prog{};
    ::shelf::runtime::DispatchBinding binding{};
    ~Frag() { teardown(); }
    void teardown() {
        if (values_dev) shelf::gpu::discard_cleanup_result(hipFree(values_dev));
        if (states_dev) shelf::gpu::discard_cleanup_result(hipFree(states_dev));
        if (ctx_dev) shelf::gpu::discard_cleanup_result(hipFree(ctx_dev));
        if (scratch) shelf::gpu::discard_cleanup_result(hipFree(scratch));
        if (error_word) shelf::gpu::discard_cleanup_result(hipFree(error_word));
        values_dev = states_dev = ctx_dev = scratch = error_word = nullptr;
        prog = {};
    }
};

struct Env {
    Profile p;
    uint32_t rows = 0;
    uint32_t visible = 0;
    uint32_t q_features = 0;
    uint32_t ept = 0;
    void* q = nullptr;
    void* out_corr = nullptr;
    void* out_opt = nullptr;
    void* row_seq = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* k_scale = nullptr;
    void* v_scale = nullptr;
    Frag frag;
    ::shelf::kernel::ModelDispatchStateView state{};
    ::shelf::kernel::PagedAttentionCommonArgs common{};
    std::vector<uint16_t> h_q;
    std::vector<uint8_t> h_kpool, h_vpool;
    std::vector<float> h_kscale, h_vscale;
    std::vector<uint32_t> h_row_seq, h_row_pos, h_bt;
    uint32_t splits = 1;
    float* partials = nullptr;
    ~Env() { teardown(); }
    void teardown() {
        void** bufs[10] = {&q, &out_corr, &out_opt, &row_seq, &row_pos, &block_tables,
                           &k_pool, &v_pool, &k_scale, &v_scale};
        for (auto b : bufs)
            if (*b) shelf::gpu::discard_cleanup_result(hipFree(*b)), *b = nullptr;
        if (partials) shelf::gpu::discard_cleanup_result(hipFree(partials)), partials = nullptr;
    }
};

bool build_state(Env& e, uint32_t seed) {
    const uint32_t pt = e.p.page_tokens;
    const uint32_t num_pages = (e.visible + pt - 1) / pt + 1;
    e.p.num_pages = num_pages;
    e.p.block_table_stride = num_pages;
    const uint32_t ept = e.ept;
    const uint32_t epp = pt * ept;
    const uint32_t epl = num_pages * epp;
    const uint64_t pool_elems = (uint64_t)e.p.num_layers * epl;
    const size_t pool_bytes = e.p.fp8 ? (size_t)pool_elems : (size_t)pool_elems * 2;
    const uint32_t scale_elems = e.p.num_layers * num_pages * pt * e.p.kv_heads;
    const size_t q_elems = (size_t)e.rows * e.q_features;
    const size_t out_elems = (size_t)e.rows * e.q_features;
    const size_t bt_elems = (size_t)e.p.max_sequences * num_pages;
    const size_t out_bytes = e.p.f32_output ? out_elems * 4 : out_elems * 2;

    size_t k_pool_sz = 0, v_pool_sz = 0, k_scale_sz = 0, v_scale_sz = 0;
    if (e.p.fp8) {
        k_pool_sz = v_pool_sz = pool_bytes;
        k_scale_sz = v_scale_sz = (size_t)scale_elems * 4;
    } else {
        k_pool_sz = v_pool_sz = pool_bytes;
    }

    void** buf[10] = {&e.q, &e.out_corr, &e.out_opt, &e.row_seq, &e.row_pos, &e.block_tables,
                      &e.k_pool, &e.v_pool, &e.k_scale, &e.v_scale};
    size_t sz[10] = {q_elems * 2, out_bytes, out_bytes, (size_t)e.rows * 4, (size_t)e.rows * 4,
                     bt_elems * 4, k_pool_sz, v_pool_sz, k_scale_sz, v_scale_sz};
    for (int i = 0; i < 10; ++i) {
        if (sz[i] == 0) continue;
        if (hipMalloc(buf[i], sz[i]) != hipSuccess) return false;
    }

    e.h_q.resize(q_elems);
    for (size_t i = 0; i < q_elems; ++i)
        e.h_q[i] = bf16_bits(rng1k((size_t)i * 7919ull + seed) / 128.0);
    e.h_row_seq.resize(e.rows);
    e.h_row_pos.resize(e.rows);
    for (uint32_t r = 0; r < e.rows; ++r) {
        e.h_row_seq[r] = r % e.p.max_sequences;
        e.h_row_pos[r] = e.visible - 1;
    }
    e.h_bt.resize(bt_elems);
    for (uint32_t slot = 0; slot < e.p.max_sequences; ++slot)
        for (uint32_t lb = 0; lb < num_pages; ++lb)
            e.h_bt[(size_t)slot * num_pages + lb] = lb;

    if (e.p.fp8) {
        e.h_kpool.assign(pool_elems, 0);
        e.h_vpool.assign(pool_elems, 0);
        e.h_kscale.assign(scale_elems, 0.0f);
        e.h_vscale.assign(scale_elems, 0.0f);
        for (uint32_t page = 0; page < num_pages; ++page) {
            for (uint32_t off = 0; off < pt; ++off) {
                for (uint32_t h = 0; h < e.p.kv_heads; ++h) {
                    const uint64_t hb = (uint64_t)e.p.layer * epl + (uint64_t)page * epp
                        + (uint64_t)off * ept + (uint64_t)h * e.p.head_dim;
                    const size_t sih = ((size_t)e.p.layer * num_pages + page) * pt + off;
                    const size_t sih2 = sih * e.p.kv_heads + h;
                    float k_amax = 0.f, v_amax = 0.f;
                    for (uint32_t d = 0; d < e.p.head_dim; ++d) {
                        k_amax = std::fmax(k_amax, std::fabs(rng1k(hb + d + 11u) / 128.0f));
                        v_amax = std::fmax(v_amax, std::fabs(rng1k(hb + d + 29u) / 128.0f));
                    }
                    const float ks = k_amax == 0.f ? 1.f : k_amax / kFp8E4M3Max;
                    const float vs = v_amax == 0.f ? 1.f : v_amax / kFp8E4M3Max;
                    e.h_kscale[sih2] = ks;
                    e.h_vscale[sih2] = vs;
                    for (uint32_t d = 0; d < e.p.head_dim; ++d) {
                        const float kf = rng1k(hb + d + 11u) / 128.0f;
                        const float vf = rng1k(hb + d + 29u) / 128.0f;
                        e.h_kpool[hb + d] = (uint8_t)__hip_cvt_float_to_fp8(kf / ks, __HIP_SATFINITE, __HIP_E4M3);
                        e.h_vpool[hb + d] = (uint8_t)__hip_cvt_float_to_fp8(vf / vs, __HIP_SATFINITE, __HIP_E4M3);
                    }
                }
            }
        }
    } else {
        e.h_kpool.assign(pool_bytes, 0);
        e.h_vpool.assign(pool_bytes, 0);
        for (uint32_t page = 0; page < num_pages; ++page) {
            for (uint32_t off = 0; off < pt; ++off) {
                for (uint32_t h = 0; h < e.p.kv_heads; ++h) {
                    const uint64_t hb = (uint64_t)e.p.layer * epl + (uint64_t)page * epp
                        + (uint64_t)off * ept + (uint64_t)h * e.p.head_dim;
                    for (uint32_t d = 0; d < e.p.head_dim; ++d) {
                        const uint16_t kb = bf16_bits(rng1k(hb + d + 11u) / 128.0);
                        const uint16_t vb = bf16_bits(rng1k(hb + d + 29u) / 128.0);
                        e.h_kpool[hb * 2 + d * 2] = kb & 0xFF;
                        e.h_kpool[hb * 2 + d * 2 + 1] = kb >> 8;
                        e.h_vpool[hb * 2 + d * 2] = vb & 0xFF;
                        e.h_vpool[hb * 2 + d * 2 + 1] = vb >> 8;
                    }
                }
            }
        }
    }

    if (hipMemcpy(e.q, e.h_q.data(), q_elems * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemset(e.out_corr, 0, out_bytes) != hipSuccess) return false;
    if (hipMemset(e.out_opt, 0, out_bytes) != hipSuccess) return false;
    if (hipMemcpy(e.row_seq, e.h_row_seq.data(), e.rows * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(e.row_pos, e.h_row_pos.data(), e.rows * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(e.block_tables, e.h_bt.data(), bt_elems * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (k_pool_sz != 0 && hipMemcpy(e.k_pool, e.h_kpool.data(), k_pool_sz, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (v_pool_sz != 0 && hipMemcpy(e.v_pool, e.h_vpool.data(), v_pool_sz, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (e.p.fp8) {
        if (hipMemcpy(e.k_scale, e.h_kscale.data(), (size_t)scale_elems * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
        if (hipMemcpy(e.v_scale, e.h_vscale.data(), (size_t)scale_elems * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
    }

    e.state = {};
    e.state.kv_dtype = e.p.fp8 ? 1u : 0u;
    e.state.block_table.block_tables = (uint32_t*)e.block_tables;
    e.state.block_table.max_sequences = e.p.max_sequences;
    e.state.block_table.block_table_stride = num_pages;
    if (e.p.fp8) {
        e.state.kv_fp8.k_pool = (fp8*)e.k_pool;
        e.state.kv_fp8.v_pool = (fp8*)e.v_pool;
        e.state.kv_fp8.k_scale_pool = (float*)e.k_scale;
        e.state.kv_fp8.v_scale_pool = (float*)e.v_scale;
        e.state.kv_fp8.page_tokens = pt;
        e.state.kv_fp8.num_pages = num_pages;
        e.state.kv_fp8.num_attention_layers = e.p.num_layers;
        e.state.kv_fp8.kv_heads = e.p.kv_heads;
        e.state.kv_fp8.head_dim = e.p.head_dim;
        e.state.kv_fp8.elems_per_token = ept;
        e.state.kv_fp8.elems_per_page = epp;
        e.state.kv_fp8.elems_per_layer = epl;
    } else {
        e.state.kv_bf16.k_pool = (shelf::bf16_t*)e.k_pool;
        e.state.kv_bf16.v_pool = (shelf::bf16_t*)e.v_pool;
        e.state.kv_bf16.page_tokens = pt;
        e.state.kv_bf16.num_pages = num_pages;
        e.state.kv_bf16.num_attention_layers = e.p.num_layers;
        e.state.kv_bf16.kv_heads = e.p.kv_heads;
        e.state.kv_bf16.head_dim = e.p.head_dim;
        e.state.kv_bf16.elems_per_token = ept;
        e.state.kv_bf16.elems_per_page = epp;
        e.state.kv_bf16.elems_per_layer = epl;
    }

    e.common = {};
    e.common.q = (const shelf::bf16_t*)e.q;
    e.common.output = e.out_opt;
    e.common.output_dtype = e.p.f32_output ? ::shelf::kernel::PagedAttentionOutputDType::F32
                                           : ::shelf::kernel::PagedAttentionOutputDType::BF16;
    e.common.q_row_stride = e.q_features;
    e.common.output_row_stride = e.q_features;
    e.common.row_sequence_slots = (const uint32_t*)e.row_seq;
    e.common.row_positions = (const uint32_t*)e.row_pos;
    e.common.block_tables = (const uint32_t*)e.block_tables;
    e.common.block_table_stride = num_pages;
    e.common.max_sequences = e.p.max_sequences;
    e.common.rows = e.rows;
    e.common.layer = e.p.layer;
    e.common.page_tokens = pt;
    e.common.num_pages = num_pages;
    e.common.num_attention_layers = e.p.num_layers;
    e.common.q_heads = e.p.q_heads;
    e.common.kv_heads = e.p.kv_heads;
    e.common.head_dim = e.p.head_dim;
    e.common.elems_per_token = ept;
    e.common.elems_per_page = epp;
    e.common.elems_per_layer = epl;
    e.common.scale = e.p.scale;
    return true;
}

bool build_correctness(Env& e) {
    const uint32_t ept = e.ept;
    const uint32_t epp = e.p.page_tokens * ept;
    const uint32_t epl = e.p.num_pages * epp;
    (void)epp; (void)epl;
    const auto out_dt = e.p.f32_output ? ::shelf::runtime::ValueDType::F32 : ::shelf::runtime::ValueDType::BF16;
    const uint64_t out_bytes = e.p.f32_output ? (uint64_t)e.rows * e.q_features * 4
                                              : (uint64_t)e.rows * e.q_features * 2;
    ::shelf::runtime::DeviceValueBinding vals[2];
    vals[0].logical_value = 0;
    vals[0].storage = ::shelf::runtime::ValueStorage::EXTERNAL_INPUT;
    vals[0].dtype = ::shelf::runtime::ValueDType::BF16;
    vals[0].row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
    vals[0].slot = 0;
    vals[0].feature_count = e.q_features;
    vals[0].row_stride = e.q_features;
    vals[0].offset = (uint64_t)(uintptr_t)e.q;
    vals[0].bytes = (uint64_t)e.rows * e.q_features * 2;
    vals[1] = vals[0];
    vals[1].logical_value = 1;
    vals[1].storage = ::shelf::runtime::ValueStorage::WORKSPACE;
    vals[1].dtype = out_dt;
    vals[1].offset = (uint64_t)(uintptr_t)e.out_corr;
    vals[1].bytes = out_bytes;
    if (hipMalloc(&e.frag.values_dev, 2 * sizeof(::shelf::runtime::DeviceValueBinding)) != hipSuccess)
        return false;
    std::memcpy(e.frag.values_dev, vals, 2 * sizeof(::shelf::runtime::DeviceValueBinding));

    ::shelf::runtime::DeviceStateBinding st;
    st.slot = 0;
    st.kind = (uint8_t)::shelf::runtime::PrimitiveStateKind::KV_CACHE;
    st.state_index = e.p.layer;
    if (hipMalloc(&e.frag.states_dev, sizeof(st)) != hipSuccess) return false;
    std::memcpy(e.frag.states_dev, &st, sizeof(st));

    ::shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = e.rows;
    ctx.row_sequence_slots = (uint32_t*)e.row_seq;
    ctx.row_positions = (uint32_t*)e.row_pos;
    if (hipMalloc(&e.frag.ctx_dev, sizeof(ctx)) != hipSuccess) return false;
    if (hipMemcpy(e.frag.ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMalloc(&e.frag.scratch, (size_t)64 * 2048 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&e.frag.error_word, sizeof(uint32_t)) != hipSuccess) return false;

    e.frag.prog = {};
    e.frag.prog.values = (const ::shelf::runtime::DeviceValueBinding*)e.frag.values_dev;
    e.frag.prog.value_count = 2;
    e.frag.prog.states = (const ::shelf::runtime::DeviceStateBinding*)e.frag.states_dev;
    e.frag.prog.state_count = 1;
    e.frag.prog.workspace_base = 0;
    e.frag.binding = {};
    e.frag.binding.kernel_id = ::shelf::runtime::KernelId::PAGED_ATTENTION;
    e.frag.binding.descriptor_slot = 1;
    e.frag.binding.input_slots[0] = 0;
    e.frag.binding.output_slots[0] = 1;
    e.frag.binding.input_count = 1;
    e.frag.binding.output_count = 1;
    e.frag.binding.group_size = e.p.head_dim;
    e.frag.binding.scalar_a = e.p.scale;
    return true;
}

hipError_t launch_correctness(const Env& e, hipStream_t s) {
    if (hipMemsetAsync(e.frag.error_word, 0, sizeof(uint32_t), s) != hipSuccess)
        return hipGetLastError();
    auto st = shelf::kernel::launch_model_dispatch_correctness(
        e.frag.prog, e.frag.binding, (::shelf::runtime::DeviceBatchContext*)e.frag.ctx_dev,
        e.state, (float*)e.frag.scratch, (uint32_t*)e.frag.error_word, s);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

hipError_t launch_optimized(const Env& e, hipStream_t s) {
    if (e.splits > 1u) {
        Env* env = const_cast<Env*>(&e);
        if (env->partials == nullptr) {
            const size_t bytes = (size_t)e.common.rows * e.common.kv_heads * e.splits
                * (e.common.q_heads / e.common.kv_heads)
                * (2u + ::shelf::kernel::kPagedAttentionHeadLimit) * sizeof(float);
            if (hipMalloc(reinterpret_cast<void**>(&env->partials), bytes) != hipSuccess)
                return hipErrorOutOfMemory;
        }
        if (e.p.fp8) {
            hipError_t herr = ::shelf::kernel::launch_attention_paged_fp8_split(
                e.common, (const fp8*)e.k_pool, (const fp8*)e.v_pool,
                (const float*)e.k_scale, (const float*)e.v_scale,
                env->partials, e.splits, s);
            if (herr != hipSuccess) return herr;
        } else {
            hipError_t herr = ::shelf::kernel::launch_attention_paged_bf16_split(
                e.common, (const shelf::bf16_t*)e.k_pool, (const shelf::bf16_t*)e.v_pool,
                env->partials, e.splits, s);
            if (herr != hipSuccess) return herr;
        }
        return ::shelf::kernel::launch_attention_paged_bf16_split_reduce(
            e.common, env->partials, e.splits, s);
    }
    if (e.p.fp8)
        return ::shelf::kernel::launch_attention_paged_fp8_e4m3(
            e.common, (const fp8*)e.k_pool, (const fp8*)e.v_pool, (const float*)e.k_scale,
            (const float*)e.v_scale, s);
    return ::shelf::kernel::launch_attention_paged_bf16(
        e.common, (const shelf::bf16_t*)e.k_pool, (const shelf::bf16_t*)e.v_pool, s);
}

bool check_output(const Env& e, hipError_t (*launch)(const Env&, hipStream_t)) {
    (void)launch_correctness(e, nullptr);
    const hipError_t hc = hipGetLastError();
    const hipError_t ho = launch(e, nullptr);
    // A refused launch must not read as a numeric mismatch: without this the output buffer
    // keeps its old contents and the comparison reports every element wrong.
    if (hc != hipSuccess || ho != hipSuccess) {
        std::printf("  (launch corr=%s opt=%s)\n", hipGetErrorString(hc), hipGetErrorString(ho));
        return false;
    }
    if (hipDeviceSynchronize() != hipSuccess) return false;
    const size_t out_elems = (size_t)e.rows * e.q_features;
    std::vector<float> corr(out_elems), opt(out_elems);
    if (e.p.f32_output) {
        if (hipMemcpy(corr.data(), e.out_corr, out_elems * 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
        if (hipMemcpy(opt.data(), e.out_opt, out_elems * 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
    } else {
        std::vector<uint16_t> cb(out_elems), ob(out_elems);
        if (hipMemcpy(cb.data(), e.out_corr, out_elems * 2, hipMemcpyDeviceToHost) != hipSuccess) return false;
        if (hipMemcpy(ob.data(), e.out_opt, out_elems * 2, hipMemcpyDeviceToHost) != hipSuccess) return false;
        for (size_t i = 0; i < out_elems; ++i) {
            corr[i] = f32_from_bf16(cb[i]);
            opt[i] = f32_from_bf16(ob[i]);
        }
    }
    double max_rel = 0.0;
    size_t mm = 0;
    for (size_t i = 0; i < out_elems; ++i) {
        const double denom = std::fmax(1.0, std::fabs((double)corr[i]));
        const double rel = std::fabs((double)opt[i] - (double)corr[i]) / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > 2e-3) ++mm;
    }
    if (mm != 0) std::printf("  (max_rel=%.3e mm=%zu)\n", max_rel, mm);
    return mm == 0;
}

std::vector<Profile> extract_actual_profiles(const std::string& model_dir, double arena_gib,
                                              int device, hipStream_t stream) {
    std::size_t arena_bytes = (std::size_t)(arena_gib * 1024.0 * 1024.0 * 1024.0);
    auto arena_res = shelf::gpu::GpuArena::create(device, arena_bytes);
    if (!arena_res.ok()) return {};
    auto arena = arena_res.release();
    shelf::qwen35::Qwen35LoadOptions lo;
    auto model_res = shelf::qwen35::Qwen35Model::load_from_safetensors(model_dir, arena, stream, lo);
    if (!model_res.ok()) return {};
    auto model = model_res.release();
    auto lp = shelf::qwen35::lower_qwen35_to_primitives(model.text_config(), model.weights());
    if (!lp.ok()) return {};
    auto lowered = lp.release();
    std::vector<::shelf::runtime::WeightSlot> ws = std::move(lowered.weights);
    std::vector<::shelf::runtime::StaticParameterSlot> shelf = std::move(lowered.parameters);
    ::shelf::runtime::WeightTableView wv{nullptr, 0};
    wv.slots = ws.data();
    wv.count = ws.size();
    ::shelf::runtime::ProgramBuildOptions opts;
    opts.validate_workspace = false;
    auto prog_res = ::shelf::runtime::build_program(lowered.graph, wv, ::shelf::runtime::RowBucket::R16,
                                                 ::shelf::runtime::ExecutionClass::DECODE, opts);
    if (!prog_res.ok()) return {};
    auto prog = prog_res.release();
    std::set<std::string> seen;
    std::vector<Profile> out;
    for (size_t i = 0; i < prog.dispatches.size(); ++i) {
        if (prog.dispatches[i].kernel_id != ::shelf::runtime::KernelId::PAGED_ATTENTION) continue;
        const auto& b = prog.dispatches[i];
        const uint32_t hd = b.group_size;
        if (hd == 0 || b.input_count != 1) continue;
        auto qv = prog.find_value(::shelf::runtime::ValueId{b.input_slots[0]});
        auto ov = prog.find_value(::shelf::runtime::ValueId{b.output_slots[0]});
        if (!qv || !ov) continue;
        if (qv->feature_count % hd != 0) continue;
        Profile p;
        p.head_dim = hd;
        p.q_heads = qv->feature_count / hd;
        p.kv_heads = p.q_heads;
        p.f32_output = (ov->dtype == ::shelf::runtime::ValueDType::F32);
        p.scale = b.scalar_a;
        char key[80];
        std::snprintf(key, sizeof key, "%u|%u|%u", p.q_heads, p.head_dim, (int)p.f32_output);
        if (seen.insert(key).second) out.push_back(p);
    }
    return out;
}

const char* dt_name(bool fp8) {
    return fp8 ? "fp8" : "bf16";
}

namespace {

hipError_t launch_prefill(const Env& e, hipStream_t s) {
    if (e.p.fp8)
        return ::shelf::kernel::launch_attention_paged_prefill_fp8_e4m3(
            e.common, (const fp8*)e.k_pool, (const fp8*)e.v_pool, (const float*)e.k_scale,
            (const float*)e.v_scale, s);
    return ::shelf::kernel::launch_attention_paged_prefill_bf16(
        e.common, (const shelf::bf16_t*)e.k_pool, (const shelf::bf16_t*)e.v_pool, s);
}

}  // namespace

}  // namespace

int run_paged_attention(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else if (a == "--variant") o.variant = next();
        else if (a == "--rows") { if (!parse_list(next(), o.rows)) std::exit(2); }
        else if (a == "--visible") { if (!parse_list(next(), o.visible)) std::exit(2); }
        else if (a == "--kv-dtype") {
            const std::string v = next();
            if (v != "bf16" && v != "fp8") {
                std::fprintf(stderr, "valid --kv-dtype: bf16|fp8 (got %s)\n", v.c_str());
                std::exit(2);
            }
            o.fp8 = (v == "fp8");
        }
        else if (a == "--out-dtype") o.f32_output = (std::string(next()) == "f32");
        else if (a == "--q-heads") o.q_heads = (uint32_t)std::stoul(next());
        else if (a == "--kv-heads") o.kv_heads = (uint32_t)std::stoul(next());
        else if (a == "--head-dim") o.head_dim = (uint32_t)std::stoul(next());
        else if (a == "--page-tokens") o.page_tokens = (uint32_t)std::stoul(next());
        else if (a == "--splits") o.splits = (uint32_t)std::stoul(next());
        else if (a == "--dump-out") o.dump_out = next();
        else if (a == "--warmup") o.warmup = std::atoi(next());
        else if (a == "--samples") o.samples = std::atoi(next());
        else if (a == "--launches") o.launches = std::atoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::atoi(next());
        else if (a == "--check") o.check = true;
        else if (a == "--out") o.out = next();
        else if (a == "--model-dir") o.model_dir = next();
        else if (a == "--actual-qwen-profiles") o.actual_qwen_profiles = true;
        else if (a == "--arena-gib") o.arena_gib = std::atof(next());
        else { usage(stderr); return 2; }
    }
    if (o.variant != "correctness" && o.variant != "optimized" && o.variant != "prefill" &&
        o.variant != "auto") { usage(stderr); return 2; }
    if (o.samples <= 0 || o.launches <= 0) return 2;
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::vector<Profile> profiles;
    if (o.actual_qwen_profiles) {
        if (o.model_dir.empty()) { std::fprintf(stderr, "--model-dir required\n"); return 1; }
        profiles = extract_actual_profiles(o.model_dir, o.arena_gib, o.device, stream);
        if (profiles.empty()) { std::fprintf(stderr, "no paged-attention profiles\n"); return 1; }
        for (auto& p : profiles) {
            p.fp8 = o.fp8;
            p.page_tokens = o.page_tokens;
            p.kv_heads = o.kv_heads;
        }
        std::printf("extracted %zu actual paged-attention profiles\n", profiles.size());
    } else {
        Profile p;
        p.fp8 = o.fp8;
        p.f32_output = o.f32_output;
        p.q_heads = o.q_heads;
        p.kv_heads = o.kv_heads;
        p.head_dim = o.head_dim;
        p.page_tokens = o.page_tokens;
        profiles.push_back(p);
    }

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f,
                             "variant,kv_dtype,out_dtype,q_heads,kv_heads,head_dim,page_tokens,rows,"
                             "visible,warmup,samples,launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (const Profile& prof : profiles) {
        for (uint32_t rows : o.rows) {
            for (uint32_t visible : o.visible) {
                Env e;
                e.p = prof;
                e.splits = o.splits;
                e.rows = rows;
                e.visible = visible;
                e.q_features = prof.q_heads * prof.head_dim;
                e.ept = prof.kv_heads * prof.head_dim;
                if (!build_state(e, o.seed)) { std::printf("build_state fail rows=%u vis=%u\n", rows, visible); rc = 1; continue; }
                if (!build_correctness(e)) { std::printf("frag build fail\n"); rc = 1; continue; }
                if (hipDeviceSynchronize() != hipSuccess) { rc = 1; continue; }
                bool auto_opt = true;
                if (o.variant == "auto") {
                    ::shelf::execution::PagedAttentionSelectorInput si;
                    si.kv_dtype = prof.fp8 ? ::shelf::kernel::KVCacheDType::FP8_E4M3
                                           : ::shelf::kernel::KVCacheDType::BF16;
                    si.output_dtype = prof.f32_output ? ::shelf::runtime::ValueDType::F32 : ::shelf::runtime::ValueDType::BF16;
                    si.rows = rows;
                    si.q_heads = prof.q_heads;
                    si.kv_heads = prof.kv_heads;
                    si.head_dim = prof.head_dim;
                    si.page_tokens = prof.page_tokens;
                    si.max_visible_tokens = visible;
                    auto_opt = ::shelf::execution::select_paged_attention_implementation(si) ==
                               ::shelf::execution::PagedAttentionImplementation::Optimized;
                }
                const bool use_opt = (o.variant == "optimized") || (o.variant == "prefill") ||
                                     (o.variant == "auto" && auto_opt);
                hipError_t (*launch)(const Env&, hipStream_t) =
                    (o.variant == "prefill") ? launch_prefill : launch_optimized;
                const bool run_opt = use_opt;
                auto run = [&, launch, run_opt](hipStream_t s) -> hipError_t {
                    return run_opt ? launch(e, s) : launch_correctness(e, s);
                };
                std::string vtag = o.variant;
                if (o.variant == "auto") vtag = auto_opt ? "optimized" : "correctness";
                if (o.check) {
                    bool okc = check_output(e, run_opt ? launch : launch_correctness);
                    std::printf("CHECK %-9s dt=%-4s out=%-3s qh=%u kvh=%u hd=%u pt=%u rows=%-5u vis=%-5u %s\n",
                                vtag.c_str(), dt_name(prof.fp8), prof.f32_output ? "f32" : "bf16",
                                prof.q_heads, prof.kv_heads, prof.head_dim, prof.page_tokens, rows,
                                visible, okc ? "PASS" : "FAIL");
                    if (!okc) rc = 1;
                }
                for (int i = 0; i < o.warmup; ++i)
                    if (run(stream) != hipSuccess) rc = 1;
                if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; continue; }
                hipEvent_t ev0 = nullptr, ev1 = nullptr;
                if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; continue; }
                std::vector<double> per;
                per.reserve(o.samples);
                for (int s = 0; s < o.samples; ++s) {
                    if (hipEventRecord(ev0, stream) != hipSuccess) break;
                    for (int l = 0; l < o.launches; ++l)
                        if (run(stream) != hipSuccess) break;
                    if (hipEventRecord(ev1, stream) != hipSuccess) break;
                    if (hipEventSynchronize(ev1) != hipSuccess) break;
                    float ms = 0.0f;
                    if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) break;
                    per.push_back(ms * 1000.0 / (double)o.launches);
                }
                shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
                shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
                if (per.empty()) { rc = 1; continue; }
                const double mn = percentile(per, 0.0);
                const double p50 = percentile(per, 0.5);
                const double p95 = percentile(per, 0.95);
                double mean = 0.0;
                for (double x : per) mean += x;
                mean /= (double)per.size();
                if (!o.out.empty()) {
                    FILE* f = std::fopen(o.out.c_str(), "a");
                    if (f) {
                        std::fprintf(f, "%s,%s,%s,%u,%u,%u,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                                      vtag.c_str(), dt_name(prof.fp8), prof.f32_output ? "f32" : "bf16",
                                      prof.q_heads, prof.kv_heads, prof.head_dim, prof.page_tokens, rows,
                                      visible, o.warmup, o.samples, o.launches, mn, p50, p95, mean);
                        std::fclose(f);
                    }
                }
                if (!o.dump_out.empty()) {
                    std::vector<float> dump((size_t)e.rows * e.q_features);
                    if (hipMemcpy(dump.data(), e.out_opt, dump.size() * 4, hipMemcpyDeviceToHost) ==
                            hipSuccess) {
                        FILE* f = std::fopen(o.dump_out.c_str(), "wb");
                        if (f) {
                            std::fwrite(dump.data(), 4, dump.size(), f);
                            std::fclose(f);
                        }
                    }
                }
                std::printf("%-9s dt=%-4s out=%-3s qh=%u kvh=%u hd=%u pt=%u rows=%-5u vis=%-5u p50=%.2fus mean=%.2fus\n",
                            vtag.c_str(), dt_name(prof.fp8), prof.f32_output ? "f32" : "bf16",
                            prof.q_heads, prof.kv_heads, prof.head_dim, prof.page_tokens, rows,
                            visible, p50, mean);
            }
        }
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
