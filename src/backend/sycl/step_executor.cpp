// shelf backend/sycl — GPU step executor implementation (SYCL port of
// backend/hip step_executor.cpp).
//
// Numerics policy: bf16 activations with fp32-accumulating kernels (the
// vendored PhaseShift family). The CPU reference executor (f32) remains
// the numerical golden; this executor is the speed path.
#include <shelf/backend/sycl/step_executor.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>

#include <shelf/backend/sycl/kernels/elementwise/aux_ops.h>
#include <shelf/backend/sycl/kernels/elementwise/embedding.h>
#include <shelf/backend/sycl/kernels/elementwise/elementwise.h>
#include <shelf/backend/sycl/kernels/gdn/gdn_conv_silu.h>
#include <shelf/backend/sycl/kernels/gdn/gdn_qk_prepare.h>
#include <shelf/backend/sycl/kernels/gdn/gdn_recurrence.h>
#include <shelf/backend/sycl/kernels/gemm/linear_bf16.h>
#include <shelf/backend/sycl/kernels/attention/kv_append.h>
#include <shelf/backend/sycl/kernels/attention/paged_attention.h>
#include <shelf/backend/sycl/kernels/attention/rope.h>
#include <shelf/backend/sycl/kernels/elementwise/rmsnorm.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>
#include <shelf/foundation/numeric.hpp>

namespace shelf::backend::sycl {

using namespace shelf::syclk;  // vendored kernel launchers

namespace {

std::size_t numel(const TensorShape& s) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < s.rank(); ++i) n *= s.dim(i);
    return n;
}

std::uint32_t u32(std::size_t v) { return static_cast<std::uint32_t>(v); }

} // namespace

SyclStepExecutor::SyclStepExecutor(SyclStepExecutorOptions opts) : opts_(opts) {
    if (!::shelf::sycl::set_device(opts_.device).ok()) { init_failed_ = true; return; }
    auto s = ::shelf::sycl::Stream::create();
    if (!s.ok()) { init_failed_ = true; return; }
    stream_.emplace(std::move(s.value()));

    const auto try_malloc = [&](void** p, std::size_t bytes) -> bool {
        return malloc_dev(p, bytes).ok();
    };
    bool ok = true;
    ok = ok && try_malloc(&arena_dev_, opts_.arena_bytes);
    ok = ok && try_malloc(reinterpret_cast<void**>(&token_dev_), kMaxStepRows * sizeof(std::int32_t));
    ok = ok && try_malloc(reinterpret_cast<void**>(&posrep_dev_), kPosScratch * sizeof(std::uint32_t));
    ok = ok && try_malloc(reinterpret_cast<void**>(&rowpos_dev_), kPosScratch * sizeof(std::uint32_t));
    ok = ok && try_malloc(reinterpret_cast<void**>(&slot_zero_dev_), sizeof(std::uint32_t));
    ok = ok && try_malloc(reinterpret_cast<void**>(&errword_dev_), sizeof(std::uint32_t));
    ok = ok && try_malloc(reinterpret_cast<void**>(&req_dev_), sizeof(syclk::GdnRequestDescriptor));
    ok = ok && try_malloc(reinterpret_cast<void**>(&inv_freq_dev_),
                          std::max<std::size_t>(2, opts_.rope_rotary_dim / 2) * sizeof(double));
    if (!ok) { init_failed_ = true; return; }
    (void)memset_dev(slot_zero_dev_, 0, sizeof(std::uint32_t));
    posrep_host_.assign(kPosScratch, 0u);
}

Status SyclStepExecutor::malloc_dev(void** out, std::size_t bytes) {
    try {
        void* p = ::sycl::malloc_device(bytes, stream_->queue());
        if (!p) return Status::insufficient_memory("sycl::malloc_device failed",
                                                   __FILE__, __LINE__);
        *out = p;
        return Status::make_ok();
    } catch (...) {
        return ::shelf::syclk::status_from_sycl("malloc_dev", __FILE__, __LINE__);
    }
}

Status SyclStepExecutor::memcpy_h2d_sync(void* dst, const void* src, std::size_t bytes) {
    try {
        stream_->queue().memcpy(dst, src, bytes).wait_and_throw();
    } catch (...) {
        return ::shelf::syclk::status_from_sycl("memcpy h2d", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status SyclStepExecutor::memcpy_d2h_sync(void* dst, const void* src, std::size_t bytes) {
    try {
        stream_->queue().memcpy(dst, src, bytes).wait_and_throw();
    } catch (...) {
        return ::shelf::syclk::status_from_sycl("memcpy d2h", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status SyclStepExecutor::memset_dev(void* dst, int value, std::size_t bytes) {
    try {
        stream_->queue().memset(dst, value, bytes).wait_and_throw();
    } catch (...) {
        return ::shelf::syclk::status_from_sycl("memset", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

void SyclStepExecutor::free_all_dev() {
    // USM free requires all referencing commands to be done (hipFree was
    // synchronous; the SYCL runtime is not).
    if (stream_) stream_->synchronize();
    auto fr = [&](void* p) { if (p) (void)::sycl::free(p, stream_->queue()); };
    for (auto& [id, loc] : state_dev_) {
        fr(loc.dev);
        fr(loc.k_pool);
        fr(loc.v_pool);
        fr(loc.table);
    }
    for (auto& [id, p] : weight_dev_) fr(p);
    fr(arena_dev_);
    fr(token_dev_);
    fr(posrep_dev_);
    fr(rowpos_dev_);
    fr(slot_zero_dev_);
    fr(errword_dev_);
    fr(req_dev_);
    fr(inv_freq_dev_);
    arena_dev_ = nullptr;
    token_dev_ = nullptr;
    posrep_dev_ = nullptr;
    rowpos_dev_ = nullptr;
    slot_zero_dev_ = nullptr;
    errword_dev_ = nullptr;
    inv_freq_dev_ = nullptr;
    req_dev_ = nullptr;
}

SyclStepExecutor::~SyclStepExecutor() {
    if (stream_) free_all_dev();
}

Status SyclStepExecutor::upload_weight(TensorId id, const std::vector<float>& f32) {
    if (weight_dev_.count(id.value()))
        return Status::invalid_state("weight uploaded twice", __FILE__, __LINE__);
    std::vector<bf16_t> b(f32.size());
    for (std::size_t i = 0; i < f32.size(); ++i) b[i] = f32_to_bf16(f32[i]);
    void* dev = nullptr;
    SHELF_RETURN_IF_ERROR(malloc_dev(&dev, b.size() * sizeof(bf16_t)));
    SHELF_RETURN_IF_ERROR(memcpy_h2d_sync(dev, b.data(), b.size() * sizeof(bf16_t)));
    weight_dev_[id.value()] = dev;
    return Status::make_ok();
}

Status SyclStepExecutor::upload_weight_bf16(TensorId id, const bf16_t* data, std::size_t n) {
    if (weight_dev_.count(id.value()))
        return Status::invalid_state("weight uploaded twice", __FILE__, __LINE__);
    void* dev = nullptr;
    SHELF_RETURN_IF_ERROR(malloc_dev(&dev, n * sizeof(bf16_t)));
    SHELF_RETURN_IF_ERROR(memcpy_h2d_sync(dev, data, n * sizeof(bf16_t)));
    weight_dev_[id.value()] = dev;
    return Status::make_ok();
}

Status SyclStepExecutor::declare_states(const ir::Graph& g) {
    if (init_failed_)
        return Status::invalid_state("executor init failed (device/allocation)",
                                     __FILE__, __LINE__);
    weight_shape_.clear();
    for (const auto& w : g.all_weights()) weight_shape_[w.id.value()] = w.shape;

    for (const auto& s : g.all_states()) {
        state_ids_.push_back(s.id);
        StateLoc loc;
        loc.kind = s.kind;
        const TensorShape& sh = s.logical_shape;
        if (s.kind == ir::StateKind::kv_cache) {
            const std::size_t cap = sh.dim(0), kvh = sh.dim(2), D = sh.dim(3);
            const std::size_t bs = opts_.kv_block_size;
            loc.pages = (cap + bs - 1) / bs;
            const std::size_t pool_elems = loc.pages * bs * kvh * D;
            SHELF_RETURN_IF_ERROR(malloc_dev(&loc.k_pool, pool_elems * sizeof(bf16_t)));
            SHELF_RETURN_IF_ERROR(malloc_dev(&loc.v_pool, pool_elems * sizeof(bf16_t)));
            SHELF_RETURN_IF_ERROR(memset_dev(loc.k_pool, 0, pool_elems * sizeof(bf16_t)));
            SHELF_RETURN_IF_ERROR(memset_dev(loc.v_pool, 0, pool_elems * sizeof(bf16_t)));
            std::vector<std::uint32_t> t(loc.pages);
            for (std::size_t i = 0; i < loc.pages; ++i) t[i] = u32(i);
            SHELF_RETURN_IF_ERROR(malloc_dev(reinterpret_cast<void**>(&loc.table),
                                             t.size() * sizeof(std::uint32_t)));
            SHELF_RETURN_IF_ERROR(memcpy_h2d_sync(loc.table, t.data(),
                                                  t.size() * sizeof(std::uint32_t)));
        } else {
            loc.elems = numel(sh);
            SHELF_RETURN_IF_ERROR(malloc_dev(&loc.dev, loc.elems * sizeof(float)));
            SHELF_RETURN_IF_ERROR(memset_dev(loc.dev, 0, loc.elems * sizeof(float)));
        }
        state_dev_[s.id.value()] = loc;
    }

    syclk::GdnRequestDescriptor req{0u, 0u, 1u};
    SHELF_RETURN_IF_ERROR(memcpy_h2d_sync(req_dev_, &req, sizeof(req)));
    SHELF_RETURN_IF_ERROR(memset_dev(errword_dev_, 0, sizeof(std::uint32_t)));
    SHELF_RETURN_IF_ERROR(launch_rope_inv_freq_init(
        inv_freq_dev_, u32(opts_.rope_rotary_dim / 2),
        static_cast<float>(opts_.rope_theta), *stream_));
    states_declared_ = true;
    return Status::make_ok();
}

Status SyclStepExecutor::reset_states() {
    for (auto& [id, loc] : state_dev_) {
        loc.rows = 0;
        if (loc.dev)
            SHELF_RETURN_IF_ERROR(memset_dev(loc.dev, 0, loc.elems * sizeof(float)));
    }
    // KV pools keep stale bytes; rowpos/rows=0 makes them unreachable by the
    // paged addressing (kernels read positions < rows only).
    return Status::make_ok();
}

void* SyclStepExecutor::arena_alloc(std::size_t bytes) {
    arena_top_ = (arena_top_ + 255u) & ~std::size_t(255u);
    if (arena_top_ + bytes > arena_cap_) return nullptr;
    void* p = static_cast<char*>(arena_dev_) + arena_top_;
    arena_top_ += bytes;
    return p;
}

Status SyclStepExecutor::run_step(const ir::Graph& g, const std::int32_t* tokens,
                                  std::size_t n, std::uint32_t base_pos) {
    if (!states_declared_)
        return Status::invalid_state("declare_states first", __FILE__, __LINE__);
    if (n == 0 || n > kMaxStepRows)
        return Status::invalid_argument("run_step: rows out of range", __FILE__, __LINE__);
    if (!::shelf::syclk::wave32_supported(stream_->queue().get_device()))
        return Status::unsupported(
            "this device lacks 32-lane sub-groups (wave32 kernels unsupported; "
            "e.g. OpenMP/CPU device)", __FILE__, __LINE__);
    arena_cap_ = opts_.arena_bytes;
    arena_reset();
    value_dev_.clear();
    step_rows_ = n;
    step_base_pos_ = base_pos;

    const auto t0 = std::chrono::steady_clock::now();

    try {
        stream_->queue().memcpy(token_dev_, tokens, n * sizeof(std::int32_t));
    } catch (...) {
        return ::shelf::syclk::status_from_sycl("token upload", __FILE__, __LINE__);
    }
    for (std::size_t i = 0; i < n; ++i) posrep_host_[i] = base_pos + static_cast<std::uint32_t>(i);
    syclk::GdnRequestDescriptor req{0u, 0u, u32(n)};
    try {
        stream_->queue().memcpy(req_dev_, &req, sizeof(req));
        stream_->queue().memcpy(rowpos_dev_, posrep_host_.data(),
                               n * sizeof(std::uint32_t));
    } catch (...) {
        return ::shelf::syclk::status_from_sycl("scratch upload", __FILE__, __LINE__);
    }

    const auto& inputs = g.all_inputs();
    if (inputs.empty())
        return Status::invalid_argument("graph lacks token input", __FILE__, __LINE__);
    value_dev_[inputs[0].id.value()] = ValueLoc{token_dev_, false, n};

    for (const auto& p : g.primitives()) {
        SHELF_RETURN_IF_ERROR(run_primitive(g, p, step_base_pos_));
        if (const char* d = std::getenv("SHELF_SYCL_DEBUG_SYNC")) { (void)d;
            try {
                stream_->queue().wait_and_throw();
            } catch (...) {
                return ::shelf::syclk::status_from_sycl(
                    (std::string("after ") + ir::kind_name(p.kind)).c_str(),
                    __FILE__, __LINE__);
            }
        }
    }

    try {
        stream_->queue().wait_and_throw();
    } catch (...) {
        return ::shelf::syclk::status_from_sycl("step sync", __FILE__, __LINE__);
    }
    stats_.last_step_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
    return Status::make_ok();
}

Result<std::vector<float>> SyclStepExecutor::fetch_value_debug(const ir::Graph& /*g*/,
                                                               ir::ValueId id) {
    auto it = value_dev_.find(id.value());
    if (it == value_dev_.end() || !it->second.dev)
        return Status::not_found("value not materialized", __FILE__, __LINE__);
    const ValueLoc& l = it->second;
    std::vector<bf16_t> tmp;
    std::vector<float> out(l.elems);
    if (l.f32) {
        SHELF_RETURN_IF_ERROR(memcpy_d2h_sync(out.data(), l.dev, l.elems * sizeof(float)));
    } else if (id.value() == 0u && l.elems == 1 && !l.f32) {
        // token input: i32[1]
        std::int32_t t;
        SHELF_RETURN_IF_ERROR(memcpy_d2h_sync(&t, l.dev, 4));
        out[0] = static_cast<float>(t);
    } else {
        tmp.resize(l.elems);
        SHELF_RETURN_IF_ERROR(memcpy_d2h_sync(tmp.data(), l.dev, l.elems * sizeof(bf16_t)));
        for (std::size_t i = 0; i < l.elems; ++i)
            out[i] = ::shelf::bf16_to_f32(tmp[i]);
    }
    return out;
}

Result<std::vector<float>> SyclStepExecutor::fetch_logits(const ir::Graph& g) {
    if (!logits_dev_)
        return Status::invalid_state("logits not produced (run_step first)", __FILE__,
                                     __LINE__);
    std::size_t n = 0;
    for (const auto& p : g.primitives())
        if (p.kind == ir::Kind::output_projection) n = numel(p.output.shape);
    if (!n) return Status::invalid_state("graph lacks output_projection", __FILE__, __LINE__);
    std::vector<float> out(n);
    SHELF_RETURN_IF_ERROR(memcpy_d2h_sync(out.data(), logits_dev_, n * sizeof(float)));
    return out;
}

Status SyclStepExecutor::run_primitive(const ir::Graph& g, const ir::Primitive& p,
                                       std::uint32_t /*position*/) {
    using K = ir::Kind;
    const auto in = [&](std::size_t i) -> ValueLoc* {
        if (i >= p.inputs.size()) return nullptr;
        auto it = value_dev_.find(p.inputs[i].value());
        return it == value_dev_.end() ? nullptr : &it->second;
    };
    const auto wt = [&](std::size_t i) -> void* {
        if (i >= p.weights.size()) return nullptr;
        auto it = weight_dev_.find(p.weights[i].value());
        return it == weight_dev_.end() ? nullptr : it->second;
    };
    const std::size_t out_n = numel(p.output.shape);
    const auto alloc = [&](bool f32) -> void* {
        void* p_ = arena_alloc(out_n * (f32 ? sizeof(float) : sizeof(bf16_t)));
        return p_;
    };
    const auto set_out = [&](bool f32, void* dev) -> Status {
        if (!dev) return Status::insufficient_memory("arena exhausted", __FILE__, __LINE__);
        value_dev_[p.output.id.value()] = ValueLoc{dev, f32, out_n};
        return Status::make_ok();
    };
    const auto need = [&](ValueLoc* v, bool f32, const char* what) -> Status {
        if (!v) return Status::invalid_state(std::string(what) + ": missing input", __FILE__,
                                             __LINE__);
        if (v->f32 != f32)
            return Status::invalid_state(std::string(what) + ": dtype policy mismatch",
                                         __FILE__, __LINE__);
        return Status::make_ok();
    };

    switch (p.kind) {
    case K::embedding: {
        const std::size_t H = p.output.shape.dim(p.output.shape.rank() - 1);
        const auto ws = weight_shape_.find(p.weights[0].value());
        const std::size_t vocab = ws == weight_shape_.end() ? 0 : ws->second.dim(0);
        if (!vocab) return Status::invalid_state("embedding: unknown weight", __FILE__, __LINE__);
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(launch_embedding_bf16(
            static_cast<const bf16_t*>(wt(0)), token_dev_, static_cast<bf16_t*>(o),
            u32(step_rows_), u32(vocab), u32(H), u32(H), errword_dev_, *stream_));
        return set_out(false, o);
    }
    case K::linear:
    case K::output_projection: {
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "linear"));
        const bool f32_out = (p.kind == K::output_projection);
        void* o = alloc(f32_out);
        const std::size_t N = p.output.shape.dim(p.output.shape.rank() - 1);
        const std::size_t rows = step_rows_;
        const std::size_t Kd = x->elems / rows;  // input [rows, K]
        // bench_gemv_hip on the model's real shapes: rows==1 -> ExactRows
        // wins everywhere; rows>=2 -> ExactRowsDot2 wins on most N (k/v/o/
        // mlp/gdn_out), losing only ~1-4 us on q_gate/gdn_qkv. Net positive.
        // (SYCL port: Dot2 falls back to the identical scalar accumulation;
        // see kernels/gemm/linear_bf16.h note.)
        Bf16GemmConfig cfg{rows >= 2u ? Bf16GemmConfigId::ExactRowsDot2
                                      : Bf16GemmConfigId::ExactRows,
                           static_cast<std::uint8_t>(rows)};
        SHELF_RETURN_IF_ERROR(launch_gemm_bf16(
            cfg, static_cast<const bf16_t*>(wt(0)), static_cast<const bf16_t*>(x->dev), o,
            f32_out ? Bf16GemmOutputDType::F32 : Bf16GemmOutputDType::BF16,
            u32(rows), u32(N), u32(Kd), u32(Kd), u32(N), *stream_));
        if (f32_out) logits_dev_ = static_cast<float*>(o);
        return set_out(f32_out, o);
    }
    case K::rmsnorm: {
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "rmsnorm"));  // bf16 activations
        if (!wt(0)) return Status::invalid_state("rmsnorm: missing weight", __FILE__, __LINE__);
        const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);
        const std::size_t rows = out_n / F;
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(launch_rmsnorm(
            x->dev, RmsNormDataType::BF16, wt(0), RmsNormDataType::BF16,
            RmsNormWeightLayout::PER_FEATURE,
            p.attr.gamma_plus_one ? RmsNormWeightMode::ONE_PLUS : RmsNormWeightMode::DIRECT,
            o, RmsNormDataType::BF16, u32(rows), u32(F), u32(F), u32(F), u32(F),
            p.attr.epsilon, *stream_));
        return set_out(false, o);
    }
    case K::rope: {
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "rope"));
        // vendored rope computes f32 sincos and takes f32 input; expand bf16 first
        void* xf = arena_alloc(out_n * sizeof(float));
        void* o = alloc(false);
        if (!xf || !o) return Status::insufficient_memory("arena exhausted", __FILE__, __LINE__);
        SHELF_RETURN_IF_ERROR(
            launch_bf16_to_f32(static_cast<const bf16_t*>(x->dev),
                               static_cast<float*>(xf), u32(out_n), *stream_));
        const std::size_t D = p.output.shape.dim(p.output.shape.rank() - 1);
        const std::size_t rows = out_n / D;               // P * heads_per_token
        const std::size_t per_tok = rows / step_rows_;    // q or kv heads
        if (rows % step_rows_) return Status::invalid_state("rope: row count not multiple of batch", __FILE__, __LINE__);
        const std::size_t rd = static_cast<std::size_t>(p.attr.rotary_dim);
        // rope position of row r = base_pos + r/per_tok (all heads of one token share pos)
        for (std::size_t r = 0; r < rows; ++r)
            posrep_host_[r] = step_base_pos_ + static_cast<std::uint32_t>(r / per_tok);
        try {
            stream_->queue().memcpy(posrep_dev_, posrep_host_.data(),
                                   rows * sizeof(std::uint32_t));
        } catch (...) {
            return ::shelf::syclk::status_from_sycl("rope pos upload", __FILE__, __LINE__);
        }
        SHELF_RETURN_IF_ERROR(launch_rope_f32_bf16(
            static_cast<const float*>(xf), u32(D), static_cast<bf16_t*>(o), u32(D),
            posrep_dev_, u32(rows), u32(D), u32(D), u32(rd),
            static_cast<float>(opts_.rope_theta), inv_freq_dev_, 1.0f, *stream_));
        return set_out(false, o);
    }
    case K::slice: {
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "slice"));
        const std::size_t F_in = x->elems ? [&] {
            const auto* s = g.find_value(p.inputs[0]);
            return s ? s->shape.dim(s->shape.rank() - 1) : 0;
        }() : 0;
        if (!F_in) return Status::invalid_state("slice: unknown input width", __FILE__, __LINE__);
        const std::size_t begin = static_cast<std::size_t>(p.attr.slice_begin);
        const std::size_t end = p.attr.slice_end < 0 ? F_in
                                                     : static_cast<std::size_t>(p.attr.slice_end);
        const std::size_t rows = out_n / (end - begin);
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(launch_bf16_row_slice(
            static_cast<const bf16_t*>(x->dev), static_cast<bf16_t*>(o), u32(rows),
            u32(end - begin), u32(begin), u32(F_in), *stream_));
        return set_out(false, o);
    }
    case K::reshape: {
        ValueLoc* x = in(0);
        if (!x) return Status::invalid_state("reshape: missing input", __FILE__, __LINE__);
        return set_out(x->f32, x->dev);  // pure view: alias
    }
    case K::sigmoid_gate: {
        ValueLoc* x = in(0), *z = in(1);
        SHELF_RETURN_IF_ERROR(need(x, false, "sigmoid_gate"));
        SHELF_RETURN_IF_ERROR(need(z, false, "sigmoid_gate"));
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(launch_bf16_sigmoid_gate(
            static_cast<const bf16_t*>(x->dev), static_cast<const bf16_t*>(z->dev),
            static_cast<bf16_t*>(o), u32(out_n), *stream_));
        return set_out(false, o);
    }
    case K::silu: {
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "silu"));
        void* o = alloc(true);  // bf16 -> f32 (no bf16->bf16 silu in family)
        const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);
        SHELF_RETURN_IF_ERROR(launch_silu_bf16_f32(
            static_cast<const bf16_t*>(x->dev), u32(F), static_cast<float*>(o),
            u32(F), u32(out_n / F), u32(F), *stream_));
        return set_out(true, o);
    }
    case K::elem_mul: {
        ValueLoc* a = in(0), *b = in(1);
        if (!a || !b) return Status::invalid_state("elem_mul: missing input", __FILE__, __LINE__);
        // mixed f32 (from silu) x bf16 -> convert bf16 to f32 then mul_f32_f32_bf16
        const bool a32 = a->f32, b32 = b->f32;
        void* af = nullptr; void* bf = nullptr;
        const float *pa32 = nullptr, *pb32 = nullptr;
        const std::size_t Fx = p.output.shape.dim(p.output.shape.rank() - 1);
        std::size_t ars = Fx, brs = Fx;   // row strides (elements)
        if (a32) pa32 = static_cast<const float*>(a->dev);
        else { af = arena_alloc(out_n * 4); pa32 = static_cast<const float*>(af);
               SHELF_RETURN_IF_ERROR(launch_bf16_to_f32(
                   static_cast<const bf16_t*>(a->dev), static_cast<float*>(af), u32(a->elems),
                   *stream_)); }
        if (b32) pb32 = static_cast<const float*>(b->dev);
        else { bf = arena_alloc(out_n * 4); pb32 = static_cast<const float*>(bf);
               SHELF_RETURN_IF_ERROR(launch_bf16_to_f32(
                   static_cast<const bf16_t*>(b->dev), static_cast<float*>(bf), u32(b->elems),
                   *stream_)); }
        void* o = alloc(false);
        const std::size_t Fm = p.output.shape.dim(p.output.shape.rank() - 1);
        SHELF_RETURN_IF_ERROR(launch_mul_f32_f32_bf16(
            pa32, u32(ars), pb32, u32(brs), static_cast<bf16_t*>(o), u32(Fm),
            u32(out_n / Fm), u32(Fm), *stream_));
        (void)af; (void)bf;
        return set_out(false, o);
    }
    case K::residual_add: {
        ValueLoc* a = in(0), *b = in(1);
        SHELF_RETURN_IF_ERROR(need(a, false, "residual_add"));
        SHELF_RETURN_IF_ERROR(need(b, false, "residual_add"));
        const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(launch_residual_add_bf16(
            static_cast<const bf16_t*>(a->dev), u32(F), static_cast<const bf16_t*>(b->dev),
            u32(F), static_cast<bf16_t*>(o), u32(F), u32(out_n / F), u32(F), *stream_));
        return set_out(false, o);
    }
    case K::causal_conv_step: {
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "causal_conv_step"));
        auto sit = state_dev_.find(p.state_write[0].value());
        if (sit == state_dev_.end() || !wt(0))
            return Status::invalid_state("causal_conv_step: state/weight missing", __FILE__,
                                         __LINE__);
        StateLoc& st = sit->second;
        const auto* cs = g.find_value(p.inputs[0]);
        const std::size_t C = cs ? cs->shape.dim(cs->shape.rank() - 1) : 0;
        if (!C) return Status::invalid_state("causal_conv_step: unknown C", __FILE__, __LINE__);
        void* of = arena_alloc(x->elems * sizeof(float));
        void* o = alloc(false);
        FusedGdnConvSiluArgs args{};
        args.conv.input = static_cast<const bf16_t*>(x->dev);
        args.conv.output = static_cast<float*>(of);
        args.conv.conv_weight = static_cast<const bf16_t*>(wt(0));
        args.conv.conv_state = static_cast<float*>(st.dev);
        args.conv.requests = req_dev_;
        args.conv.num_requests = 1u;
        args.conv.actual_rows = u32(step_rows_);
        args.conv.max_request_rows = u32(step_rows_);
        args.conv.input_row_stride = u32(C);
        args.conv.output_row_stride = u32(C);
        args.conv.conv_dim = u32(C);
        args.conv.conv_history = 3u;
        args.conv.state_index = 0u;
        args.conv.max_sequences = 1u;
        args.conv.conv_layer_stride = C * 3;
        args.conv.conv_slot_stride = C * 3;
        args.act_out = static_cast<bf16_t*>(o);
        args.act_row_stride = u32(C);
        SHELF_RETURN_IF_ERROR(launch_fused_gdn_conv_silu(args, *stream_));
        return set_out(false, o);
    }
    case K::gdn_step: {
        ValueLoc* q = in(0), *k = in(1), *v = in(2), *b = in(3), *a = in(4);
        SHELF_RETURN_IF_ERROR(need(q, false, "gdn_step"));
        SHELF_RETURN_IF_ERROR(need(k, false, "gdn_step"));
        SHELF_RETURN_IF_ERROR(need(v, false, "gdn_step"));
        SHELF_RETURN_IF_ERROR(need(b, false, "gdn_step"));
        SHELF_RETURN_IF_ERROR(need(a, false, "gdn_step"));
        auto sit = state_dev_.find(p.state_write[0].value());
        if (sit == state_dev_.end() || !wt(0) || !wt(1))
            return Status::invalid_state("gdn_step: state/weights missing", __FILE__, __LINE__);
        StateLoc& st = sit->second;
        const TensorShape& qs = g.find_value(p.inputs[0])->shape;  // [nkh,dk]|[P,nkh,dk]
        const TensorShape& vs = g.find_value(p.inputs[2])->shape;  // [P,nvh,dv]
        const std::size_t nkh = qs.dim(qs.rank() - 2), dk = qs.dim(qs.rank() - 1);
        const std::size_t nvh = vs.dim(vs.rank() - 2), dv = vs.dim(vs.rank() - 1);
        const std::size_t qk_n = nkh * dk;

        const std::size_t prep_n = step_rows_ * qk_n;   // one row per token
        void* nqf = arena_alloc(prep_n * 4);
        void* sqf = arena_alloc(prep_n * 4);
        void* nkf = arena_alloc(prep_n * 4);
        void* o = alloc(true);  // f32 [nvh,dv]
        if (!nqf || !sqf || !nkf || !o)
            return Status::insufficient_memory("arena exhausted", __FILE__, __LINE__);

        FusedGdnQkPrepareArgs prep{};
        prep.q = static_cast<const bf16_t*>(q->dev);
        prep.q_row_stride = u32(qk_n);   // q value [rows, nkh*dk]
        prep.k = static_cast<const bf16_t*>(k->dev);
        prep.k_row_stride = u32(qk_n);
        prep.nq_out = static_cast<float*>(nqf);
        prep.nq_row_stride = u32(qk_n);
        prep.nk_out = static_cast<float*>(nkf);
        prep.nk_row_stride = u32(qk_n);
        prep.sq_out = static_cast<float*>(sqf);
        prep.sq_row_stride = u32(qk_n);
        prep.eps = 1e-6f;
        prep.scale = 1.0f / std::sqrt(static_cast<float>(dk));
        prep.rows = u32(step_rows_);
        prep.features = u32(qk_n);
        prep.group_size = u32(dk);
        SHELF_RETURN_IF_ERROR(launch_fused_gdn_qk_prepare(prep, *stream_));

        GdnRecurrenceArgs rec{};
        rec.q = static_cast<const float*>(sqf);
        rec.k = static_cast<const float*>(nkf);
        rec.v = static_cast<const bf16_t*>(v->dev);
        rec.a = static_cast<const bf16_t*>(a->dev);
        rec.b = static_cast<const bf16_t*>(b->dev);
        rec.dt_bias = static_cast<const bf16_t*>(wt(1));
        rec.a_log = static_cast<const bf16_t*>(wt(0));
        rec.recurrent_state = static_cast<float*>(st.dev);
        rec.output = static_cast<float*>(o);
        rec.requests = req_dev_;
        rec.num_requests = 1u;
        rec.actual_rows = u32(step_rows_);
        rec.q_row_stride = u32(qk_n);
        rec.k_row_stride = u32(qk_n);
        rec.v_row_stride = u32(nvh * dv);
        rec.a_row_stride = u32(nvh);
        rec.b_row_stride = u32(nvh);
        rec.output_row_stride = u32(nvh * dv);
        rec.recurrent_slot_stride = nvh * dk * dv;
        rec.recurrent_layer_stride = nvh * dk * dv;
        rec.state_index = 0u;
        rec.max_sequences = 1u;
        rec.key_heads = u32(nkh);
        rec.num_v_heads = u32(nvh);
        rec.head_k = u32(dk);
        rec.head_v = u32(dv);
        if (std::getenv("SHELF_DEBUG_GDN")) {
            stream_->queue().wait();
            std::vector<float> sqh(qk_n), nkh_(qk_n), oh(nvh * dv), sh(std::min<std::size_t>(8, st.elems));
            std::vector<bf16_t> vh(nvh * dv), ah(nvh), bh(nvh);
            memcpy_d2h_sync(sqh.data(), sqf, qk_n * 4);
            memcpy_d2h_sync(nkh_.data(), nkf, qk_n * 4);
            memcpy_d2h_sync(vh.data(), v->dev, nvh * dv * 2);
            memcpy_d2h_sync(ah.data(), a->dev, nvh * 2);
            memcpy_d2h_sync(bh.data(), b->dev, nvh * 2);
            memcpy_d2h_sync(oh.data(), o, nvh * dv * 4);
            std::fprintf(stderr, "[ptr] q=%p k=%p nq=%p sq=%p nk=%p v=%p out=%p\n", q->dev, k->dev, nqf, sqf, nkf, v->dev, o);
            std::fprintf(stderr, "[gdn P=%zu pre] sq[0..3]=%.6g,%.6g,%.6g,%.6g nk[0..3]=%.6g,%.6g,%.6g,%.6g v0=%.6g a0=%.6g b0=%.6g out0=%.6g out_last=%.6g st0=%.6g\n",
                step_rows_, sqh[0], sqh[1], sqh[2], sqh[3], nkh_[0], nkh_[1], nkh_[2], nkh_[3],
                ::shelf::bf16_to_f32(vh[0]),
                ::shelf::bf16_to_f32(ah[0]),
                ::shelf::bf16_to_f32(bh[0]),
                oh[0], oh[oh.size()-1], sh[0]);
        }
        SHELF_RETURN_IF_ERROR(launch_gdn_recurrence_f32(rec, *stream_));
        if (std::getenv("SHELF_DEBUG_GDN")) {
            stream_->queue().wait();
            std::vector<float> oh(nvh * dv), sh(std::min<std::size_t>(4, st.elems));
            memcpy_d2h_sync(oh.data(), o, nvh * dv * 4);
            memcpy_d2h_sync(sh.data(), st.dev, sh.size() * 4);
            std::fprintf(stderr, "[gdn P=%zu post] out0=%.9g outlast=%.9g st0=%.9g st1=%.9g\n",
                         step_rows_, oh[0], oh[oh.size()-1], sh[0], sh[sh.size()-1]);
        }
        st.rows += step_rows_;
        return set_out(true, o);
    }
    case K::rmsnorm_gated: {
        // out = rmsnorm(x, eps) * gamma_plain * silu(z): rmsnorm then swiglu kernel
        ValueLoc* x = in(0), *z = in(1);
        if (!x || !z || !wt(0))
            return Status::invalid_state("rmsnorm_gated: missing input/weight", __FILE__, __LINE__);
        const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);  // dv
        const std::size_t rows = out_n / F;
        void* norm;
        if (x->f32) {
            norm = alloc(true);
            SHELF_RETURN_IF_ERROR(launch_rmsnorm(
                x->dev, RmsNormDataType::F32, wt(0), RmsNormDataType::BF16,
                RmsNormWeightLayout::PER_FEATURE, RmsNormWeightMode::DIRECT, norm,
                RmsNormDataType::F32, u32(rows), u32(F), u32(F), u32(F), u32(F),
                p.attr.epsilon, *stream_));
            void* normb = arena_alloc(out_n * sizeof(bf16_t));
            SHELF_RETURN_IF_ERROR(launch_f32_to_bf16(
                static_cast<const float*>(norm), static_cast<bf16_t*>(normb), u32(out_n),
                *stream_));
            norm = normb;
        } else {
            norm = arena_alloc(out_n * sizeof(bf16_t));
            SHELF_RETURN_IF_ERROR(launch_rmsnorm(
                x->dev, RmsNormDataType::BF16, wt(0), RmsNormDataType::BF16,
                RmsNormWeightLayout::PER_FEATURE, RmsNormWeightMode::DIRECT, norm,
                RmsNormDataType::BF16, u32(rows), u32(F), u32(F), u32(F), u32(F),
                p.attr.epsilon, *stream_));
        }
        SHELF_RETURN_IF_ERROR(need(z, false, "rmsnorm_gated z"));
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(launch_swiglu_bf16(
            static_cast<const bf16_t*>(z->dev), u32(F), static_cast<const bf16_t*>(norm),
            u32(F), static_cast<bf16_t*>(o), u32(F), u32(rows), u32(F), *stream_));
        return set_out(false, o);
    }
    case K::attention_gqa_step: {
        ValueLoc* q = in(0), *k = in(1), *v = in(2);
        SHELF_RETURN_IF_ERROR(need(q, false, "attention_gqa_step"));
        SHELF_RETURN_IF_ERROR(need(k, false, "attention_gqa_step"));
        SHELF_RETURN_IF_ERROR(need(v, false, "attention_gqa_step"));
        auto sit = state_dev_.find(p.state_write[0].value());
        if (sit == state_dev_.end())
            return Status::invalid_state("attention_gqa_step: state missing", __FILE__, __LINE__);
        StateLoc& st = sit->second;
        const ir::StateDesc* desc = g.find_state(p.state_write[0]);
        if (st.rows + step_rows_ > desc->logical_shape.dim(0))
            return Status::out_of_range("attention_gqa_step: KV context full", __FILE__,
                                        __LINE__);
        const std::size_t bs = opts_.kv_block_size;
        const std::size_t kvh = desc->logical_shape.dim(2);
        const std::size_t D = desc->logical_shape.dim(3);
        const std::size_t P = step_rows_;
        const std::size_t nq = q->elems / (D * P);   // per-token query heads
        if (q->elems % (D * P)) return Status::invalid_state("attention: q width mismatch", __FILE__, __LINE__);
        const std::size_t ept = kvh * D, epp = bs * ept, epl = st.pages * epp;

        KvAppendCommonArgs ap{};
        ap.k_input = static_cast<const bf16_t*>(k->dev);
        ap.v_input = static_cast<const bf16_t*>(v->dev);
        ap.k_row_stride = u32(k->elems / P);
        ap.v_row_stride = u32(v->elems / P);
        ap.row_sequence_slots = slot_zero_dev_;
        ap.row_positions = rowpos_dev_;
        ap.block_tables = st.table;
        ap.block_table_stride = u32(st.pages);
        ap.max_sequences = 1u;
        ap.rows = u32(step_rows_);
        ap.layer = 0u;
        ap.page_tokens = u32(bs);
        ap.num_pages = u32(st.pages);
        ap.num_attention_layers = 1u;
        ap.kv_heads = u32(kvh);
        ap.head_dim = u32(D);
        ap.elems_per_token = ept;
        ap.elems_per_page = epp;
        ap.elems_per_layer = epl;
        SHELF_RETURN_IF_ERROR(launch_kv_append_bf16(ap, static_cast<bf16_t*>(st.k_pool),
                                                    static_cast<bf16_t*>(st.v_pool),
                                                    *stream_));

        void* o = alloc(false);
        PagedAttentionCommonArgs at{};
        at.q = static_cast<const bf16_t*>(q->dev);
        at.output = o;
        at.output_dtype = PagedAttentionOutputDType::BF16;
        at.q_row_stride = u32(nq * D);
        at.output_row_stride = u32(nq * D);
        at.row_sequence_slots = slot_zero_dev_;
        at.row_positions = rowpos_dev_;
        at.block_tables = st.table;
        at.block_table_stride = u32(st.pages);
        at.max_sequences = 1u;
        at.rows = u32(step_rows_);
        at.layer = 0u;
        at.page_tokens = u32(bs);
        at.num_pages = u32(st.pages);
        at.num_attention_layers = 1u;
        at.q_heads = u32(nq);
        at.kv_heads = u32(kvh);
        at.head_dim = u32(D);
        at.elems_per_token = ept;
        at.elems_per_page = epp;
        at.elems_per_layer = epl;
        at.scale = p.attr.score_scale != 0.0f ? p.attr.score_scale
                                              : 1.0f / std::sqrt(static_cast<float>(D));
        SHELF_RETURN_IF_ERROR(launch_attention_paged_bf16(
            at, static_cast<const bf16_t*>(st.k_pool), static_cast<const bf16_t*>(st.v_pool),
            *stream_));
        st.rows += step_rows_;
        return set_out(false, o);
    }
    default:
        return Status::unsupported((std::string(ir::kind_name(p.kind)) +
                                    " not wired on sycl executor").c_str(),
                                   __FILE__, __LINE__);
    }
}

} // namespace shelf::backend::sycl
