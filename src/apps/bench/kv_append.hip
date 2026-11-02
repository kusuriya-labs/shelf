#include <shelf/kernels/optimized/kv_append.h>
#include <shelf/kernels/kv_types.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
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
using fp8 = ::ps::kernel::fp8e4m3_storage_t;
constexpr float kFp8E4M3Max = 448.0f;

struct Profile {
    bool fp8 = false;
    uint32_t kv_heads = 4;
    uint32_t head_dim = 256;
    uint32_t page_tokens = 16;
    uint32_t num_pages = 128;
    uint32_t num_layers = 1;
    uint32_t layer = 0;
    uint32_t block_table_stride = 128;
    uint32_t max_sequences = 8;
};

struct Options {
    std::string variant = "optimized";
    std::vector<uint32_t> rows = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    bool fp8 = false;
    uint32_t kv_heads = 4;
    uint32_t head_dim = 256;
    uint32_t page_tokens = 16;
    uint32_t num_pages = 128;
    uint32_t num_layers = 1;
    int warmup = 100;
    int samples = 50;
    int launches = 20;
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
                 "usage: shelf-bench kv-append [options]\n"
                 "  --variant correctness|optimized|auto\n"
                 "  --rows 1,2,4,...\n"
                 "  --kv-dtype bf16|fp8\n"
                 "  --kv-heads N\n"
                 "  --head-dim N\n"
                 "  --page-tokens N\n"
                 "  --num-pages N\n"
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

bool parse_rows(const std::string& s, std::vector<uint32_t>& out) {
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
    return ps::f32_to_bf16_rne((float)x);
}

struct Frag {
    void* values_dev = nullptr;
    void* states_dev = nullptr;
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* error_word = nullptr;
    ::ps::runtime::DeviceProgramView prog{};
    ::ps::runtime::DispatchBinding binding{};
    ~Frag() { teardown(); }
    void teardown() {
        if (values_dev) ps::gpu::discard_cleanup_result(hipFree(values_dev));
        if (states_dev) ps::gpu::discard_cleanup_result(hipFree(states_dev));
        if (ctx_dev) ps::gpu::discard_cleanup_result(hipFree(ctx_dev));
        if (scratch) ps::gpu::discard_cleanup_result(hipFree(scratch));
        if (error_word) ps::gpu::discard_cleanup_result(hipFree(error_word));
        values_dev = states_dev = ctx_dev = scratch = error_word = nullptr;
        prog = {};
    }
};

struct Env {
    Profile p;
    uint32_t rows = 0;
    uint32_t ept = 0;
    void* k_in = nullptr;
    void* v_in = nullptr;
    void* row_seq = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* k_scale = nullptr;
    void* v_scale = nullptr;
    Frag frag;
    ::ps::kernel::ModelDispatchStateView state{};
    ::ps::kernel::KvAppendCommonArgs common{};
    std::vector<uint16_t> h_kin, h_vin;
    std::vector<uint32_t> h_row_seq, h_row_pos, h_bt;
    ~Env() { teardown(); }
    void teardown() {
        void** bufs[9] = {&k_in, &v_in, &row_seq, &row_pos, &block_tables, &k_pool, &v_pool,
                          &k_scale, &v_scale};
        for (auto b : bufs)
            if (*b) ps::gpu::discard_cleanup_result(hipFree(*b)), *b = nullptr;
    }
};

bool upload(Env& e, uint32_t seed) {
    const size_t input_elems = (size_t)e.rows * e.ept;
    const size_t bt_elems = (size_t)e.p.max_sequences * e.p.block_table_stride;
    const uint64_t pool_elems = (uint64_t)e.p.num_layers * e.p.num_pages * e.p.page_tokens * e.ept;
    const uint32_t scale_elems = e.p.num_layers * e.p.num_pages * e.p.page_tokens * e.p.kv_heads;
    size_t k_pool_bytes = 0, v_pool_bytes = 0, k_scale_bytes = 0, v_scale_bytes = 0;
    if (e.p.fp8) {
        k_pool_bytes = v_pool_bytes = (size_t)pool_elems;
        k_scale_bytes = v_scale_bytes = (size_t)scale_elems * 4u;
    } else {
        k_pool_bytes = v_pool_bytes = (size_t)pool_elems * 2u;
    }
    void** buf[9] = {&e.k_in, &e.v_in, &e.row_seq, &e.row_pos, &e.block_tables, &e.k_pool,
                     &e.v_pool, &e.k_scale, &e.v_scale};
    size_t sz[9] = {input_elems * 2, input_elems * 2, (size_t)e.rows * 4, (size_t)e.rows * 4,
                    bt_elems * 4, k_pool_bytes, v_pool_bytes, k_scale_bytes, v_scale_bytes};
    for (int i = 0; i < 9; ++i) {
        if (sz[i] == 0) continue;
        if (hipMalloc(buf[i], sz[i]) != hipSuccess) return false;
    }
    e.h_kin.resize(input_elems);
    e.h_vin.resize(input_elems);
    e.h_row_seq.resize(e.rows);
    e.h_row_pos.resize(e.rows);
    e.h_bt.resize(bt_elems);
    for (size_t i = 0; i < input_elems; ++i) {
        e.h_kin[i] = bf16_bits(((double)(h32(i * 7919ull + seed) % 2001u) - 1000.0) / 128.0);
        e.h_vin[i] = bf16_bits(((double)(h32(i * 7919ull + seed + 7u) % 2001u) - 1000.0) / 128.0);
    }
    for (uint32_t r = 0; r < e.rows; ++r) {
        e.h_row_seq[r] = r % e.p.max_sequences;
        e.h_row_pos[r] = r;
    }
    for (uint32_t slot = 0; slot < e.p.max_sequences; ++slot)
        for (uint32_t lb = 0; lb < e.p.block_table_stride; ++lb)
            e.h_bt[(size_t)slot * e.p.block_table_stride + lb] = lb % e.p.num_pages;
    if (hipMemcpy(e.k_in, e.h_kin.data(), input_elems * 2, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemcpy(e.v_in, e.h_vin.data(), input_elems * 2, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemcpy(e.row_seq, e.h_row_seq.data(), e.rows * 4, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemcpy(e.row_pos, e.h_row_pos.data(), e.rows * 4, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemcpy(e.block_tables, e.h_bt.data(), bt_elems * 4, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (k_pool_bytes != 0 && hipMemset(e.k_pool, 0, k_pool_bytes) != hipSuccess) return false;
    if (v_pool_bytes != 0 && hipMemset(e.v_pool, 0, v_pool_bytes) != hipSuccess) return false;
    if (k_scale_bytes != 0 && hipMemset(e.k_scale, 0, k_scale_bytes) != hipSuccess) return false;
    if (v_scale_bytes != 0 && hipMemset(e.v_scale, 0, v_scale_bytes) != hipSuccess) return false;
    return true;
}

bool build_correctness(Env& e) {
    const uint32_t ept = e.ept;
    const uint32_t epp = e.p.page_tokens * ept;
    const uint32_t epl = e.p.num_pages * epp;
    ::ps::runtime::DeviceValueBinding vals[2];
    vals[0].logical_value = 0;
    vals[0].storage = ::ps::runtime::ValueStorage::EXTERNAL_INPUT;
    vals[0].dtype = ::ps::runtime::ValueDType::BF16;
    vals[0].row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    vals[0].slot = 0;
    vals[0].feature_count = ept;
    vals[0].row_stride = ept;
    vals[0].offset = (uint64_t)(uintptr_t)e.k_in;
    vals[0].bytes = (uint64_t)e.rows * ept * 2;
    vals[1] = vals[0];
    vals[1].logical_value = 1;
    vals[1].offset = (uint64_t)(uintptr_t)e.v_in;
    if (hipMalloc(&e.frag.values_dev, 2 * sizeof(::ps::runtime::DeviceValueBinding)) != hipSuccess)
        return false;
    std::memcpy(e.frag.values_dev, vals, 2 * sizeof(::ps::runtime::DeviceValueBinding));

    ::ps::runtime::DeviceStateBinding st;
    st.slot = 0;
    st.kind = (uint8_t)::ps::runtime::PrimitiveStateKind::KV_CACHE;
    st.state_index = e.p.layer;
    if (hipMalloc(&e.frag.states_dev, sizeof(st)) != hipSuccess) return false;
    std::memcpy(e.frag.states_dev, &st, sizeof(st));

    ::ps::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = e.rows;
    ctx.row_sequence_slots = (uint32_t*)e.row_seq;
    ctx.row_positions = (uint32_t*)e.row_pos;
    if (hipMalloc(&e.frag.ctx_dev, sizeof(ctx)) != hipSuccess) return false;
    if (hipMemcpy(e.frag.ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMalloc(&e.frag.scratch, (size_t)64 * 2048 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&e.frag.error_word, sizeof(uint32_t)) != hipSuccess) return false;

    e.frag.prog = {};
    e.frag.prog.values = (const ::ps::runtime::DeviceValueBinding*)e.frag.values_dev;
    e.frag.prog.value_count = 2;
    e.frag.prog.states = (const ::ps::runtime::DeviceStateBinding*)e.frag.states_dev;
    e.frag.prog.state_count = 1;
    e.frag.prog.workspace_base = 0;
    e.frag.binding = {};
    e.frag.binding.kernel_id = ::ps::runtime::KernelId::KV_APPEND;
    e.frag.binding.descriptor_slot = 1;
    e.frag.binding.input_slots[0] = 0;
    e.frag.binding.input_slots[1] = 1;
    e.frag.binding.input_count = 2;
    e.frag.binding.output_count = 0;
    e.frag.binding.group_size = e.p.head_dim;

    e.state = {};
    e.state.kv_dtype = e.p.fp8 ? 1u : 0u;
    e.state.block_table.block_tables = (uint32_t*)e.block_tables;
    e.state.block_table.max_sequences = e.p.max_sequences;
    e.state.block_table.block_table_stride = e.p.block_table_stride;
    if (e.p.fp8) {
        e.state.kv_fp8.k_pool = (fp8*)e.k_pool;
        e.state.kv_fp8.v_pool = (fp8*)e.v_pool;
        e.state.kv_fp8.k_scale_pool = (float*)e.k_scale;
        e.state.kv_fp8.v_scale_pool = (float*)e.v_scale;
        e.state.kv_fp8.page_tokens = e.p.page_tokens;
        e.state.kv_fp8.num_pages = e.p.num_pages;
        e.state.kv_fp8.num_attention_layers = e.p.num_layers;
        e.state.kv_fp8.kv_heads = e.p.kv_heads;
        e.state.kv_fp8.head_dim = e.p.head_dim;
        e.state.kv_fp8.elems_per_token = ept;
        e.state.kv_fp8.elems_per_page = epp;
        e.state.kv_fp8.elems_per_layer = epl;
    } else {
        e.state.kv_bf16.k_pool = (ps::bf16_t*)e.k_pool;
        e.state.kv_bf16.v_pool = (ps::bf16_t*)e.v_pool;
        e.state.kv_bf16.page_tokens = e.p.page_tokens;
        e.state.kv_bf16.num_pages = e.p.num_pages;
        e.state.kv_bf16.num_attention_layers = e.p.num_layers;
        e.state.kv_bf16.kv_heads = e.p.kv_heads;
        e.state.kv_bf16.head_dim = e.p.head_dim;
        e.state.kv_bf16.elems_per_token = ept;
        e.state.kv_bf16.elems_per_page = epp;
        e.state.kv_bf16.elems_per_layer = epl;
    }

    e.common = {};
    e.common.k_input = (const ps::bf16_t*)e.k_in;
    e.common.v_input = (const ps::bf16_t*)e.v_in;
    e.common.k_row_stride = ept;
    e.common.v_row_stride = ept;
    e.common.row_sequence_slots = (const uint32_t*)e.row_seq;
    e.common.row_positions = (const uint32_t*)e.row_pos;
    e.common.block_tables = (const uint32_t*)e.block_tables;
    e.common.block_table_stride = e.p.block_table_stride;
    e.common.max_sequences = e.p.max_sequences;
    e.common.rows = e.rows;
    e.common.layer = e.p.layer;
    e.common.page_tokens = e.p.page_tokens;
    e.common.num_pages = e.p.num_pages;
    e.common.num_attention_layers = e.p.num_layers;
    e.common.kv_heads = e.p.kv_heads;
    e.common.head_dim = e.p.head_dim;
    e.common.elems_per_token = ept;
    e.common.elems_per_page = epp;
    e.common.elems_per_layer = epl;
    return true;
}

hipError_t launch_correctness(const Env& e, hipStream_t s) {
    if (hipMemsetAsync(e.frag.error_word, 0, sizeof(uint32_t), s) != hipSuccess)
        return hipGetLastError();
    auto st = ps::kernel::launch_model_dispatch_correctness(
        e.frag.prog, e.frag.binding, (::ps::runtime::DeviceBatchContext*)e.frag.ctx_dev,
        e.state, (float*)e.frag.scratch, (uint32_t*)e.frag.error_word, s);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

hipError_t launch_optimized(const Env& e, hipStream_t s) {
    if (e.p.fp8)
        return ::ps::kernel::launch_kv_append_fp8_e4m3(
                   e.common, (fp8*)e.k_pool, (fp8*)e.v_pool, (float*)e.k_scale,
                   (float*)e.v_scale, s);
    return ::ps::kernel::launch_kv_append_bf16(e.common, (ps::bf16_t*)e.k_pool,
                                                       (ps::bf16_t*)e.v_pool, s);
}

bool check_output(const Env& e) {
    const uint32_t ept = e.ept;
    const uint32_t epp = e.p.page_tokens * ept;
    const uint32_t epl = e.p.num_pages * epp;
    const uint64_t pool_elems = (uint64_t)e.p.num_layers * epl;
    const uint64_t pool_byte_len = e.p.fp8 ? pool_elems : pool_elems * 2;
    std::vector<uint8_t> got_k(pool_byte_len), got_v(pool_byte_len);
    if (hipMemcpy(got_k.data(), e.k_pool, pool_byte_len, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(got_v.data(), e.v_pool, pool_byte_len, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    std::vector<uint8_t> exp_k(pool_byte_len, 0), exp_v(pool_byte_len, 0);
    const uint32_t scale_elems = e.p.num_layers * e.p.num_pages * e.p.page_tokens * e.p.kv_heads;
    std::vector<float> exp_ks, exp_vs;
    if (e.p.fp8) {
        exp_ks.assign(scale_elems, 0.0f);
        exp_vs.assign(scale_elems, 0.0f);
    }
    for (uint32_t r = 0; r < e.rows; ++r) {
        const uint32_t slot = e.h_row_seq[r];
        const uint32_t pos = e.h_row_pos[r];
        const uint32_t lb = pos / e.p.page_tokens;
        const uint32_t off = pos % e.p.page_tokens;
        const uint32_t page = e.h_bt[(size_t)slot * e.p.block_table_stride + lb];
        const uint64_t base = (uint64_t)e.p.layer * epl + (uint64_t)page * epp +
                              (uint64_t)off * ept;
        for (uint32_t h = 0; h < e.p.kv_heads; ++h) {
            const uint64_t hb = base + (uint64_t)h * e.p.head_dim;
            if (e.p.fp8) {
                float k_amax = 0.f, v_amax = 0.f;
                for (uint32_t i = 0; i < e.p.head_dim; ++i) {
                    const float kf = f32_from_bf16(e.h_kin[(size_t)r * ept + (size_t)h * e.p.head_dim + i]);
                    const float vf = f32_from_bf16(e.h_vin[(size_t)r * ept + (size_t)h * e.p.head_dim + i]);
                    k_amax = std::fmax(k_amax, std::fabs(kf));
                    v_amax = std::fmax(v_amax, std::fabs(vf));
                }
                const float k_scale = k_amax == 0.f ? 1.f : k_amax / kFp8E4M3Max;
                const float v_scale = v_amax == 0.f ? 1.f : v_amax / kFp8E4M3Max;
                const size_t si = ((size_t)e.p.layer * e.p.num_pages + page) * e.p.page_tokens + off;
                const size_t sih = si * e.p.kv_heads + h;
                exp_ks[sih] = k_scale;
                exp_vs[sih] = v_scale;
                for (uint32_t i = 0; i < e.p.head_dim; ++i) {
                    const float kf = f32_from_bf16(e.h_kin[(size_t)r * ept + (size_t)h * e.p.head_dim + i]);
                    const float vf = f32_from_bf16(e.h_vin[(size_t)r * ept + (size_t)h * e.p.head_dim + i]);
                    exp_k[hb + i] = (uint8_t)__hip_cvt_float_to_fp8(kf / k_scale, __HIP_SATFINITE, __HIP_E4M3);
                    exp_v[hb + i] = (uint8_t)__hip_cvt_float_to_fp8(vf / v_scale, __HIP_SATFINITE, __HIP_E4M3);
                }
            } else {
                for (uint32_t i = 0; i < e.p.head_dim; ++i) {
                    const uint16_t kb = e.h_kin[(size_t)r * ept + (size_t)h * e.p.head_dim + i];
                    const uint16_t vb = e.h_vin[(size_t)r * ept + (size_t)h * e.p.head_dim + i];
                    exp_k[hb * 2 + i * 2] = kb & 0xFF;
                    exp_k[hb * 2 + i * 2 + 1] = kb >> 8;
                    exp_v[hb * 2 + i * 2] = vb & 0xFF;
                    exp_v[hb * 2 + i * 2 + 1] = vb >> 8;
                }
            }
        }
    }
    size_t k_mm = 0, v_mm = 0;
    for (uint64_t i = 0; i < pool_byte_len; ++i) {
        if (got_k[i] != exp_k[i]) ++k_mm;
        if (got_v[i] != exp_v[i]) ++v_mm;
    }
    if (k_mm != 0 || v_mm != 0) return false;
    if (e.p.fp8) {
        std::vector<float> got_ks(scale_elems), got_vs(scale_elems);
        if (hipMemcpy(got_ks.data(), e.k_scale, scale_elems * 4, hipMemcpyDeviceToHost) != hipSuccess)
            return false;
        if (hipMemcpy(got_vs.data(), e.v_scale, scale_elems * 4, hipMemcpyDeviceToHost) != hipSuccess)
            return false;
        for (uint32_t i = 0; i < scale_elems; ++i)
            if (got_ks[i] != exp_ks[i] || got_vs[i] != exp_vs[i]) return false;
    }
    return true;
}

std::vector<Profile> extract_actual_profiles(const std::string& model_dir, double arena_gib,
                                             int device, hipStream_t stream) {
    std::size_t arena_bytes = (std::size_t)(arena_gib * 1024.0 * 1024.0 * 1024.0);
    auto arena_res = ps::gpu::GpuArena::create(device, arena_bytes);
    if (!arena_res.ok()) return {};
    auto arena = arena_res.release();
    ps::qwen35::Qwen35LoadOptions lo;
    auto model_res = ps::qwen35::Qwen35Model::load_from_safetensors(model_dir, arena, stream, lo);
    if (!model_res.ok()) return {};
    auto model = model_res.release();
    auto lp = ps::qwen35::lower_qwen35_to_primitives(model.text_config(), model.weights());
    if (!lp.ok()) return {};
    auto lowered = lp.release();
    std::vector<::ps::runtime::WeightSlot> ws = std::move(lowered.weights);
    std::vector<::ps::runtime::StaticParameterSlot> ps = std::move(lowered.parameters);
    ::ps::runtime::WeightTableView wv{nullptr, 0};
    wv.slots = ws.data();
    wv.count = ws.size();
    ::ps::runtime::ProgramBuildOptions opts;
    opts.validate_workspace = false;
    auto prog_res = ::ps::runtime::build_program(lowered.graph, wv, ::ps::runtime::RowBucket::R16,
                                                 ::ps::runtime::ExecutionClass::DECODE, opts);
    if (!prog_res.ok()) return {};
    auto prog = prog_res.release();
    std::set<std::string> seen;
    std::vector<Profile> out;
    for (size_t i = 0; i < prog.dispatches.size(); ++i) {
        if (prog.dispatches[i].kernel_id != ::ps::runtime::KernelId::KV_APPEND) continue;
        const auto& b = prog.dispatches[i];
        const uint32_t hd = b.group_size;
        if (hd == 0 || b.input_count != 2) continue;
        auto kin = prog.find_value(::ps::runtime::ValueId{b.input_slots[0]});
        if (!kin) continue;
        if (kin->feature_count % hd != 0) continue;
        Profile p;
        p.head_dim = hd;
        p.kv_heads = kin->feature_count / hd;
        char key[80];
        std::snprintf(key, sizeof key, "%u|%u", p.kv_heads, p.head_dim);
        if (seen.insert(key).second) out.push_back(p);
    }
    return out;
}

const char* dt_name(bool fp8) {
    return fp8 ? "fp8" : "bf16";
}

}  // namespace

int run_kv_append(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing arg for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            usage(stdout);
            return 0;
        } else if (a == "--variant") {
            o.variant = next();
        } else if (a == "--rows") {
            if (!parse_rows(next(), o.rows)) std::exit(2);
        } else if (a == "--kv-dtype") {
            const std::string v = next();
            if (v != "bf16" && v != "fp8") {
                std::fprintf(stderr, "valid --kv-dtype: bf16|fp8 (got %s)\n", v.c_str());
                std::exit(2);
            }
            o.fp8 = (v == "fp8");
        } else if (a == "--kv-heads") {
            o.kv_heads = (uint32_t)std::stoul(next());
        } else if (a == "--head-dim") {
            o.head_dim = (uint32_t)std::stoul(next());
        } else if (a == "--page-tokens") {
            o.page_tokens = (uint32_t)std::stoul(next());
        } else if (a == "--num-pages") {
            o.num_pages = (uint32_t)std::stoul(next());
        } else if (a == "--warmup") {
            o.warmup = std::atoi(next());
        } else if (a == "--samples") {
            o.samples = std::atoi(next());
        } else if (a == "--launches") {
            o.launches = std::atoi(next());
        } else if (a == "--seed") {
            o.seed = (uint32_t)std::stoul(next());
        } else if (a == "--device") {
            o.device = std::atoi(next());
        } else if (a == "--check") {
            o.check = true;
        } else if (a == "--out") {
            o.out = next();
        } else if (a == "--model-dir") {
            o.model_dir = next();
        } else if (a == "--actual-qwen-profiles") {
            o.actual_qwen_profiles = true;
        } else if (a == "--arena-gib") {
            o.arena_gib = std::atof(next());
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (o.variant != "correctness" && o.variant != "optimized" && o.variant != "auto") {
        usage(stderr);
        return 2;
    }
    if (o.samples <= 0 || o.launches <= 0) return 2;
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::vector<Profile> profiles;
    if (o.actual_qwen_profiles) {
        if (o.model_dir.empty()) {
            std::fprintf(stderr, "--model-dir required with --actual-qwen-profiles\n");
            return 1;
        }
        profiles = extract_actual_profiles(o.model_dir, o.arena_gib, o.device, stream);
        if (profiles.empty()) {
            std::fprintf(stderr, "no kv-append profiles matched\n");
            return 1;
        }
        for (auto& p : profiles) {
            p.fp8 = o.fp8;
            p.page_tokens = o.page_tokens;
        }
        std::printf("extracted %zu actual kv-append profiles\n", profiles.size());
    } else {
        Profile p;
        p.fp8 = o.fp8;
        p.kv_heads = o.kv_heads;
        p.head_dim = o.head_dim;
        p.page_tokens = o.page_tokens;
        p.num_pages = o.num_pages;
        profiles.push_back(p);
    }

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f,
                             "variant,kv_dtype,kv_heads,head_dim,page_tokens,rows,warmup,samples,"
                             "launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (const Profile& prof : profiles) {
        for (uint32_t rows : o.rows) {
            Env e;
            e.p = prof;
            e.rows = rows;
            e.ept = prof.kv_heads * prof.head_dim;
            if (!upload(e, o.seed)) {
                std::printf("upload fail rows=%u\n", rows);
                rc = 1;
                continue;
            }
            if (!build_correctness(e)) {
                std::printf("frag build fail rows=%u\n", rows);
                rc = 1;
                continue;
            }
            if (hipDeviceSynchronize() != hipSuccess) {
                rc = 1;
                continue;
            }
            bool auto_opt = true;
            if (o.variant == "auto") {
                ::ps::execution::KvAppendSelectorInput si;
                si.kv_dtype = prof.fp8 ? ::ps::kernel::KVCacheDType::FP8_E4M3
                                       : ::ps::kernel::KVCacheDType::BF16;
                si.rows = rows;
                si.kv_heads = prof.kv_heads;
                si.head_dim = prof.head_dim;
                si.page_tokens = prof.page_tokens;
                auto_opt = ::ps::execution::select_kv_append_implementation(si) ==
                           ::ps::execution::KvAppendImplementation::Optimized;
            }
            const bool use_opt = (o.variant == "optimized") || (o.variant == "auto" && auto_opt);
            auto launch = [&, use_opt](hipStream_t s) -> hipError_t {
                return use_opt ? launch_optimized(e, s) : launch_correctness(e, s);
            };
            std::string vtag = o.variant;
            if (o.variant == "auto") vtag = auto_opt ? "optimized" : "correctness";
            if (o.check) {
                (void)launch(stream);
                if (hipDeviceSynchronize() != hipSuccess) rc = 1;
                bool okc = check_output(e);
                std::printf("CHECK %-9s dt=%-4s kvh=%u hd=%u pt=%u rows=%-5u %s\n", vtag.c_str(),
                            dt_name(prof.fp8), prof.kv_heads, prof.head_dim, prof.page_tokens, rows,
                            okc ? "PASS" : "FAIL");
                if (!okc) rc = 1;
            }
            for (int i = 0; i < o.warmup; ++i)
                if (launch(stream) != hipSuccess) rc = 1;
            if (hipStreamSynchronize(stream) != hipSuccess) {
                rc = 1;
                continue;
            }
            hipEvent_t ev0 = nullptr, ev1 = nullptr;
            if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) {
                rc = 1;
                continue;
            }
            std::vector<double> per;
            per.reserve(o.samples);
            for (int s = 0; s < o.samples; ++s) {
                if (hipEventRecord(ev0, stream) != hipSuccess) break;
                for (int l = 0; l < o.launches; ++l)
                    if (launch(stream) != hipSuccess) break;
                if (hipEventRecord(ev1, stream) != hipSuccess) break;
                if (hipEventSynchronize(ev1) != hipSuccess) break;
                float ms = 0.0f;
                if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) break;
                per.push_back(ms * 1000.0 / (double)o.launches);
            }
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev0));
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            if (per.empty()) {
                rc = 1;
                continue;
            }
            const double mn = percentile(per, 0.0);
            const double p50 = percentile(per, 0.5);
            const double p95 = percentile(per, 0.95);
            double mean = 0.0;
            for (double v : per) mean += v;
            mean /= (double)per.size();
            if (!o.out.empty()) {
                FILE* f = std::fopen(o.out.c_str(), "a");
                if (f) {
                    std::fprintf(
                        f, "%s,%s,%u,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n", vtag.c_str(),
                        dt_name(prof.fp8), prof.kv_heads, prof.head_dim, prof.page_tokens, rows,
                        o.warmup, o.samples, o.launches, mn, p50, p95, mean);
                    std::fclose(f);
                }
            }
            std::printf("%-9s dt=%-4s kvh=%-2u hd=%-4u pt=%-3u rows=%-5u min=%.1f p50=%.1f "
                        "p95=%.1f mean=%.1f us\n",
                        vtag.c_str(), dt_name(prof.fp8), prof.kv_heads, prof.head_dim,
                        prof.page_tokens, rows, mn, p50, p95, mean);
        }
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
