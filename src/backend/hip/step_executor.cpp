// shelf backend/hip — GPU step executor implementation.
//
// Numerics policy: bf16 activations with fp32-accumulating kernels (the
// vendored PhaseShift family). The CPU reference executor (f32) remains
// the numerical golden; this executor is the speed path.
// FP8 経路（approx 側）: 重みは E4M3 コードのまま断片順で載せ、活性化は
// ステップごとに per-token スケールで量子化する。dequant はカーネル内
// （行列命令が E4M3 を直接読む）。契約は src/quant/fp8_w8a8.hpp。
#include <shelf/backend/hip/step_executor.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>

#include <shelf/backend/hip/kernels/elementwise/aux_ops.h>
#include <shelf/backend/hip/kernels/elementwise/activation_quantize_fp8.h>
#include <shelf/backend/hip/kernels/elementwise/embedding.h>
#include <shelf/backend/hip/kernels/elementwise/elementwise.h>
#include <shelf/backend/hip/kernels/gdn/gdn_chunk.h>
#include <shelf/backend/hip/kernels/gdn/gdn_conv_silu.h>
#include <shelf/backend/hip/kernels/gdn/gdn_qk_prepare.h>
#include <shelf/backend/hip/kernels/gdn/gdn_recurrence.h>
#include <shelf/backend/hip/kernels/gemm/linear_bf16.h>
#include <shelf/backend/hip/kernels/gemm/linear_fp8.h>
#include <shelf/backend/hip/kernels/attention/kv_append.h>
#include <shelf/backend/hip/kernels/attention/paged_attention.h>
#include <shelf/backend/hip/kernels/attention/rope.h>
#include <shelf/backend/hip/kernels/elementwise/rmsnorm.h>
#include <shelf/foundation/numeric.hpp>
#include <shelf/quant/fp8_w8a8.hpp>

namespace shelf::backend::hip {

using namespace shelf::hipk;  // vendored kernel launchers

namespace {

Status hipok(hipError_t e, const char* what) {
    if (e != hipSuccess)
        return Status::from(Status::Code::device_error,
                            std::string(what) + ": " + hipGetErrorString(e),
                            __FILE__, __LINE__);
    return Status::make_ok();
}

std::size_t numel(const TensorShape& s) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < s.rank(); ++i) n *= s.dim(i);
    return n;
}

// 1 ステップ分のアリーナ必要量は declare 相（compiler/arena_offsets.hpp）が決める。
// ここにあった「64 MiB + 2×Σ出力(f32)」の自動算出は D4 で撤去した（offsets 表が
// 寿命の重ならない値を再利用するため、同じグラフでも必要量は小さくなる）。

std::uint32_t u32(std::size_t v) { return static_cast<std::uint32_t>(v); }

std::size_t mib(std::size_t bytes) { return bytes >> 20; }

// hipMalloc wants void**; passing reinterpret_cast<Typed**> is a strict
// aliasing violation (observed to miscompile the typed loads at -O2).
// Allocate into a local void*, then assign through the correctly-typed lvalue.
template <class T> hipError_t malloc_dev(T** out, std::size_t count) {
    void* p = nullptr;
    const hipError_t e = hipMalloc(&p, count * sizeof(T));
    *out = static_cast<T*>(p);
    return e;
}

} // namespace

// ---- device 確保の計上（予算 enforcement） ----------------------------------
// 予算（vram_budget_bytes）はこの実行器が確保した合計への上限として働き、
// hipMalloc を試みる前に止める。確保済み・要求・予算を MiB で出すので、
// 行数や状態容量をどう動かせば収まるかが読み出し可能になる。
Status HipStepExecutor::dev_reserve(void** out, std::size_t bytes, const char* what) {
    if (out == nullptr || bytes == 0u)
        return Status::invalid_argument("dev_reserve: null/zero", __FILE__, __LINE__);
    if (opts_.vram_budget_bytes > 0 && dev_bytes_ + bytes > opts_.vram_budget_bytes) {
        return Status::insufficient_memory(
            std::string(what) + ": VRAM 予算超過（確保済み " + std::to_string(mib(dev_bytes_)) +
                " MiB + 要求 " + std::to_string(mib(bytes)) + " MiB > 予算 " +
                std::to_string(mib(opts_.vram_budget_bytes)) + " MiB）",
            __FILE__, __LINE__);
    }
    const hipError_t e = hipMalloc(out, bytes);
    if (e != hipSuccess)
        return Status::from(Status::Code::device_error,
                            std::string(what) + ": " + hipGetErrorString(e) +
                                "（確保済み " + std::to_string(mib(dev_bytes_)) +
                                " MiB / 要求 " + std::to_string(mib(bytes)) + " MiB）",
                            __FILE__, __LINE__);
    dev_allocs_[*out] = bytes;
    dev_bytes_ += bytes;
    return Status::make_ok();
}

void HipStepExecutor::dev_release(void* p) {
    if (p == nullptr) return;
    const auto it = dev_allocs_.find(p);
    if (it != dev_allocs_.end()) {
        dev_bytes_ -= it->second;
        dev_allocs_.erase(it);
    }
    (void)hipFree(p);
}

HipStepExecutor::HipStepExecutor(HipStepExecutorOptions opts) : opts_(opts) {
    // KV ページ（ブロック）寸法は既定 16（解決済み kernel 由来）。
    // decode attention は 16 トークンごとに block_table を引くので、
    // 間接参照の本数が律速の可能性を試すため `SHELF_KV_PAGE` で変えられるようにする。
    if (const char* e = std::getenv("SHELF_KV_PAGE")) {
        const long v = std::strtol(e, nullptr, 10);
        if (v == 16 || v == 32 || v == 64 || v == 128) opts_.kv_block_size = static_cast<std::size_t>(v);
    }
    (void)hipSetDevice(opts_.device);
    hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
    // arena_bytes == 0 ならアリーナは declare_states のグラフから自動算出する。
    hipError_t e0 = hipSuccess;
    if (opts_.arena_bytes > 0) {
        e0 = hipMalloc(&arena_dev_, opts_.arena_bytes);
        if (e0 == hipSuccess) arena_cap_ = opts_.arena_bytes;
    }
    const hipError_t e1 = malloc_dev(&token_dev_, kMaxStepRows);
    const hipError_t e2 = malloc_dev(&posrep_dev_, kPosScratch);
    const hipError_t e3 = malloc_dev(&rowpos_dev_, kPosScratch);
    const hipError_t e4 = malloc_dev(&slot_zero_dev_, kMaxStepRows);
    const hipError_t e5 = malloc_dev(&errword_dev_, 1u);
    const hipError_t e6 = malloc_dev(&req_dev_, 1u);
    const hipError_t e7 = malloc_dev(&inv_freq_dev_,
                                      std::max<std::size_t>(2, opts_.rope_rotary_dim / 2));
    (void)e0; (void)e1; (void)e2; (void)e3; (void)e4; (void)e5; (void)e6; (void)e7;
    // 起動時 scratch は確保失敗を無視する（その値を使う時点でエラーになる）。
    // 予算 accounting には成功分だけを計上する。
    const auto charge = [&](void* p, std::size_t bytes) {
        if (p != nullptr) {
            dev_allocs_[p] = bytes;
            dev_bytes_ += bytes;
        }
    };
    charge(arena_dev_, arena_cap_);
    charge(token_dev_, kMaxStepRows * sizeof(std::int32_t));
    charge(posrep_dev_, kPosScratch * sizeof(std::uint32_t));
    charge(rowpos_dev_, kPosScratch * sizeof(std::uint32_t));
    charge(slot_zero_dev_, kMaxStepRows * sizeof(std::uint32_t));
    charge(errword_dev_, sizeof(std::uint32_t));
    charge(req_dev_, sizeof(std::uint32_t));
    charge(inv_freq_dev_, std::max<std::size_t>(2, opts_.rope_rotary_dim / 2) * sizeof(double));
    hipMemset(slot_zero_dev_, 0, kMaxStepRows * sizeof(std::uint32_t));
    posrep_host_.assign(kPosScratch, 0u);
}

HipStepExecutor::~HipStepExecutor() {
    (void)hipSetDevice(opts_.device);
    // 解放の先にキューを消化する。発射済みカーネル／コピーが生きているまま
    // メモリを解放すると HSA のメモリフォルト／ハングになり得る
    // （エラー経路でここへ来ることがある）。
    if (stream_) (void)hipStreamSynchronize(stream_);
    for (auto& [id, loc] : state_dev_) {
        if (loc.dev) dev_release(loc.dev);
        if (loc.k_pool) dev_release(loc.k_pool);
        if (loc.v_pool) dev_release(loc.v_pool);
        if (loc.table) dev_release(loc.table);
    }
    for (auto& [id, p] : weight_dev_) dev_release(p);
    for (auto& [id, w] : weight_fp8_) {
        if (w.codes) dev_release(w.codes);
        if (w.scales) dev_release(w.scales);
    }
    dev_release(fp8_codes_dev_);
    dev_release(fp8_scales_dev_);
    dev_release(arena_dev_);
    dev_release(token_dev_);
    dev_release(posrep_dev_);
    dev_release(rowpos_dev_);
    dev_release(slot_zero_dev_);
    dev_release(errword_dev_);
    dev_release(req_dev_);
    dev_release(inv_freq_dev_);
    hipStreamDestroy(stream_);
}

Status HipStepExecutor::upload_weight(TensorId id, const std::vector<float>& f32) {
    (void)hipSetDevice(opts_.device);
    if (weight_dev_.count(id.value()))
        return Status::invalid_state("weight uploaded twice", __FILE__, __LINE__);
    std::vector<bf16_t> b(f32.size());
    for (std::size_t i = 0; i < f32.size(); ++i) b[i] = f32_to_bf16(f32[i]);
    void* dev = nullptr;
    SHELF_RETURN_IF_ERROR(dev_reserve(&dev, b.size() * sizeof(bf16_t), "malloc weight"));
    SHELF_RETURN_IF_ERROR(hipok(
        hipMemcpy(dev, b.data(), b.size() * sizeof(bf16_t), hipMemcpyHostToDevice),
        "upload weight"));
    weight_dev_[id.value()] = dev;
    return Status::make_ok();
}

Status HipStepExecutor::upload_weight_bf16(TensorId id, const bf16_t* data, std::size_t n) {
    (void)hipSetDevice(opts_.device);
    if (weight_dev_.count(id.value()))
        return Status::invalid_state("weight uploaded twice", __FILE__, __LINE__);
    bf16_t* dev = nullptr;
    SHELF_RETURN_IF_ERROR(dev_alloc(&dev, n, "malloc weight bf16"));
    SHELF_RETURN_IF_ERROR(hipok(
        hipMemcpy(dev, data, n * sizeof(bf16_t), hipMemcpyHostToDevice), "upload weight bf16"));
    weight_dev_[id.value()] = dev;
    return Status::make_ok();
}

Status HipStepExecutor::upload_weight_fp8(TensorId id, const std::uint8_t* codes,
                                          std::size_t n_codes, const float* scales,
                                          std::size_t features, std::uint32_t k_padded,
                                          bool weight_block128) {
    (void)hipSetDevice(opts_.device);
    using namespace shelf::quant::fp8;
    if (weight_fp8_.count(id.value()) || weight_dev_.count(id.value()))
        return Status::invalid_state("weight uploaded twice", __FILE__, __LINE__);
    if (features == 0u || (k_padded % kTileK) != 0u)
        return Status::invalid_argument("upload_weight_fp8: 形の指定不正", __FILE__, __LINE__);
    // codes は 16 行単位で切り上げた tiled_bytes(features, k_padded)。
    if (n_codes != tiled_bytes(u32(features), k_padded))
        return Status::invalid_argument(
            "upload_weight_fp8: コード数がレイアウトと不一致（" + std::to_string(n_codes) +
            " != " + std::to_string(tiled_bytes(u32(features), k_padded)) + "）",
            __FILE__, __LINE__);
    const WeightScaleGranularity g =
        weight_block128 ? WeightScaleGranularity::Block128 : WeightScaleGranularity::PerChannel;
    const std::size_t want_scales = weight_scale_count(g, u32(features), k_padded);
    if (want_scales == 0u)
        return Status::invalid_argument("upload_weight_fp8: スケール数が不正", __FILE__, __LINE__);
    std::uint8_t* cdev = nullptr;
    float* sdev = nullptr;
    SHELF_RETURN_IF_ERROR(dev_alloc(&cdev, n_codes, "malloc weight fp8 codes"));
    SHELF_RETURN_IF_ERROR(dev_alloc(&sdev, want_scales, "malloc weight fp8 scales"));
    SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(cdev, codes, n_codes, hipMemcpyHostToDevice),
                                "upload weight fp8 codes"));
    SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(sdev, scales, want_scales * sizeof(float),
                                          hipMemcpyHostToDevice),
                                "upload weight fp8 scales"));
    weight_fp8_[id.value()] = Fp8Weight{cdev, sdev, weight_block128};
    return Status::make_ok();
}

Status HipStepExecutor::ensure_fp8_scratch(std::size_t code_bytes, std::size_t scale_floats) {
    if (code_bytes > fp8_codes_cap_) {
        if (fp8_codes_dev_) dev_release(fp8_codes_dev_);
        fp8_codes_dev_ = nullptr;
        SHELF_RETURN_IF_ERROR(dev_alloc(&fp8_codes_dev_, code_bytes, "malloc fp8 act codes"));
        fp8_codes_cap_ = code_bytes;
    }
    if (scale_floats > fp8_scales_cap_) {
        if (fp8_scales_dev_) dev_release(fp8_scales_dev_);
        fp8_scales_dev_ = nullptr;
        SHELF_RETURN_IF_ERROR(dev_alloc(&fp8_scales_dev_, scale_floats, "malloc fp8 act scales"));
        fp8_scales_cap_ = scale_floats;
    }
    return Status::make_ok();
}

Status HipStepExecutor::declare_states(const ir::Graph& g) {
    (void)hipSetDevice(opts_.device);
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
            SHELF_RETURN_IF_ERROR(
                dev_reserve(&loc.k_pool, pool_elems * sizeof(bf16_t), "kv pool K"));
            SHELF_RETURN_IF_ERROR(
                dev_reserve(&loc.v_pool, pool_elems * sizeof(bf16_t), "kv pool V"));
            SHELF_RETURN_IF_ERROR(
                hipok(hipMemset(loc.k_pool, 0, pool_elems * sizeof(bf16_t)), "zero kv K"));
            SHELF_RETURN_IF_ERROR(
                hipok(hipMemset(loc.v_pool, 0, pool_elems * sizeof(bf16_t)), "zero kv V"));
            std::vector<std::uint32_t> t(loc.pages);
            for (std::size_t i = 0; i < loc.pages; ++i) t[i] = u32(i);
            SHELF_RETURN_IF_ERROR(dev_alloc(&loc.table, t.size(), "kv table"));
            SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(loc.table, t.data(),
                                                  t.size() * sizeof(std::uint32_t),
                                                  hipMemcpyHostToDevice),
                                        "kv table upload"));
        } else {
            loc.elems = numel(sh);
            SHELF_RETURN_IF_ERROR(
                dev_reserve(&loc.dev, loc.elems * sizeof(float), "state alloc"));
            SHELF_RETURN_IF_ERROR(
                hipok(hipMemset(loc.dev, 0, loc.elems * sizeof(float)), "state zero"));
        }
        state_dev_[s.id.value()] = loc;
    }

    hipk::GdnRequestDescriptor req{0u, 0u, 1u};
    SHELF_RETURN_IF_ERROR(
        hipok(hipMemcpy(req_dev_, &req, sizeof(req), hipMemcpyHostToDevice), "gdn req"));
    SHELF_RETURN_IF_ERROR(hipok(hipMemset(errword_dev_, 0, sizeof(std::uint32_t)), "errword"));
    SHELF_RETURN_IF_ERROR(hipok(launch_rope_inv_freq_init(
                               inv_freq_dev_, u32(opts_.rope_rotary_dim / 2),
                               static_cast<float>(opts_.rope_theta), stream_),
                               "inv_freq init"));

    // アリーナが未指定なら、declare 相が決めた必要量をそのまま確保する。
    // 複数のグラフ（decode rows=1 / prefill rows=N）を同じ実行器で回す場合、
    // 足りなければ run_stage が広げる。
    if (opts_.arena_bytes == 0) {
        const std::size_t bytes = plan_for(g).peak_bytes;
        dev_release(arena_dev_);
        arena_dev_ = nullptr;
        arena_cap_ = 0;
        if (const Status st = dev_reserve(&arena_dev_, bytes, "arena malloc (declare phase)");
            !st.ok()) {
            arena_dev_ = nullptr;
            return st;
        }
        arena_cap_ = bytes;
    } else {
        const std::size_t need = plan_for(g).peak_bytes;
        if (need > arena_cap_)
            return Status::insufficient_memory(
                "アリーナが offsets 表に足りない（必要 " + std::to_string(need) +
                    " B / 確保 " + std::to_string(arena_cap_) + " B）",
                __FILE__, __LINE__);
    }

    states_declared_ = true;
    return Status::make_ok();
}

Status HipStepExecutor::reset_states() {
    (void)hipSetDevice(opts_.device);
    for (auto& [id, loc] : state_dev_) {
        loc.rows = 0;
        if (loc.dev)
            SHELF_RETURN_IF_ERROR(
                hipok(hipMemset(loc.dev, 0, loc.elems * sizeof(float)), "reset state"));
    }
    // KV pools keep stale bytes; rowpos/rows=0 makes them unreachable by the
    // paged addressing (kernels read positions < rows only).
    return Status::make_ok();
}

const compiler::ArenaPlan& HipStepExecutor::plan_for(const ir::Graph& g) {
    const std::uint32_t first_value =
        g.primitives().empty() ? 0u : g.primitives().front().output.id.value();
    const std::uint32_t last_value =
        g.primitives().empty() ? 0u : g.primitives().back().output.id.value();
    for (auto& c : plans_) {
        if (c->graph == &g && c->primitives == g.size() && c->first_value == first_value &&
            c->last_value == last_value)
            return c->plan;
    }
    auto c = std::make_unique<CachedPlan>();
    c->graph = &g;
    c->primitives = g.size();
    c->first_value = first_value;
    c->last_value = last_value;
    c->plan = compiler::plan_arena(g, opts_.arena_reuse);
    // 使うグラフは 1 実行器あたり高々 2 本（decode / prefill）。古いものから捨てる。
    if (plans_.size() >= 4) plans_.erase(plans_.begin());
    const auto& plan = c->plan;
    plans_.push_back(std::move(c));
    return plan;
}

// swiglu → fp8 活性化量子化の融合対象を集める（グラフ毎に一度）。
const HipStepExecutor::FuseCache& HipStepExecutor::fuse_for(const ir::Graph& g) {
    const std::uint32_t first_value =
        g.primitives().empty() ? 0u : g.primitives().front().output.id.value();
    const std::uint32_t last_value =
        g.primitives().empty() ? 0u : g.primitives().back().output.id.value();
    for (auto& c : fuses_)
        if (c->graph == &g && c->primitives == g.size() && c->first_value == first_value &&
            c->last_value == last_value)
            return *c;
    auto c = std::make_unique<FuseCache>();
    c->graph = &g;
    c->primitives = g.size();
    c->first_value = first_value;
    c->last_value = last_value;
    // `SHELF_FUSE_ACTQ=0` で融合を切れる（数値 A/B・ロールバック用）。
    const char* fq = std::getenv("SHELF_FUSE_ACTQ");
    const bool fuse_on = opts_.fp8_act_group != 0u && !(fq && fq[0] == '0');
    if (fuse_on) {
        using K = ir::Kind;
        std::unordered_map<std::uint32_t, std::uint32_t> users;
        for (const auto& p : g.primitives())
            for (const auto& iv : p.inputs) ++users[iv.value()];
        for (const auto& p : g.primitives()) {
            if (p.kind != K::swiglu) continue;
            const std::uint32_t v = p.output.id.value();
            if (users[v] != 1u) continue;
            const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);
            if (F == 0u || (F % shelf::quant::fp8::kActGroupK) != 0u) continue;
            if (shelf::quant::fp8::pad_k(u32(F)) != u32(F)) continue;
            for (const auto& q : g.primitives()) {
                if (q.inputs.empty() || q.inputs[0].value() != v) continue;
                if (q.kind != K::linear && q.kind != K::output_projection) continue;
                if (q.weights.empty() || weight_fp8_.count(q.weights[0].value()) == 0u) continue;
                c->swiglu_out.push_back(v);
                break;
            }
        }
        auto fp8_leaf = [&](std::uint32_t v0) {
            std::vector<std::uint32_t> stack{v0};
            bool any = false;
            bool all_fp8 = true;
            for (std::size_t guard = 0; !stack.empty() && guard < 4096u; ++guard) {
                const std::uint32_t cur = stack.back();
                stack.pop_back();
                for (const auto& q : g.primitives()) {
                    for (const auto& iv : q.inputs) {
                        if (iv.value() != cur) continue;
                        if (q.kind == K::reshape) {
                            stack.push_back(q.output.id.value());
                        } else if ((q.kind == K::linear ||
                                    q.kind == K::output_projection) &&
                                   !q.weights.empty() &&
                                   weight_fp8_.count(q.weights[0].value()) != 0u) {
                            any = true;
                        } else {
                            all_fp8 = false;
                        }
                    }
                }
            }
            return any && all_fp8;
        };
        // rmsnorm → fp8 活性化量子化の融合対象（消費者が全て fp8 GEMM の値）。
        // `SHELF_FUSE_NORMQ=0` で切れる（A/B・ロールバック用）。
        const char* nq = std::getenv("SHELF_FUSE_NORMQ");
        const bool normq_on = !(nq && nq[0] == '0');
        for (const auto& p : g.primitives()) {
            if (!normq_on) break;
            if (p.kind != K::rmsnorm || !p.attr.gamma_plus_one) continue;
            const std::uint32_t v = p.output.id.value();
            if (!fp8_leaf(v)) continue;
            const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);
            if (F == 0u || (F % shelf::quant::fp8::kActGroupK) != 0u) continue;
            if (shelf::quant::fp8::pad_k(u32(F)) != u32(F)) continue;
            c->rmsnorm_out.push_back(v);
        }
        // rmsnorm_gated（GDN 後段の gate 掛け算）→ fp8 量子化の融合対象。
        // 消費者が reshape 経由で fp8 GEMM につながるので reshape は透過扱いにする。
        for (const auto& p : g.primitives()) {
            if (p.kind != K::rmsnorm_gated) continue;
            const std::uint32_t v = p.output.id.value();
            if (!fp8_leaf(v)) continue;
            const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);
            const std::size_t flat = (step_rows_ != 0u) ? 0u : 0u;  // 実行時に平坦幅で判定する
            (void)F; (void)flat;
            c->gated_out.push_back(v);
        }
    }
    if (fuses_.size() >= 4) fuses_.erase(fuses_.begin());
    auto* raw = c.get();
    fuses_.push_back(std::move(c));
    return *raw;
}

void* HipStepExecutor::slot_ptr(ir::ValueId v, bool f32) const {
    if (plan_ == nullptr || arena_dev_ == nullptr) return nullptr;
    const auto* s = plan_->slot_of(v);
    if (s == nullptr || s->f32 != f32) return nullptr;
    return static_cast<char*>(arena_dev_) + s->offset;
}

void* HipStepExecutor::scratch_ptr(std::size_t prim_index, std::uint8_t slot,
                                   std::size_t need) const {
    if (plan_ == nullptr || arena_dev_ == nullptr) return nullptr;
    const auto* s = plan_->scratch_of(prim_index, slot);
    // 必要量が宣言より大きい = declare 相と実行器の方針が食い違っている。
    // 黙って隣を壊すより、ここで止める。
    if (s == nullptr || s->bytes < need) return nullptr;
    return static_cast<char*>(arena_dev_) + plan_->scratch_base + s->offset;
}

Status HipStepExecutor::run_step(const ir::Graph& g, const std::int32_t* tokens,
                                 std::size_t n, std::uint32_t base_pos) {
    return run_stage(g, tokens, nullptr, n, base_pos);
}

Status HipStepExecutor::run_stage(const ir::Graph& g, const std::int32_t* tokens,
                                  const bf16_t* hidden, std::size_t n,
                                  std::uint32_t base_pos) {
    (void)hipSetDevice(opts_.device);   // 複数デバイスの実行器を同じ process で使うため
    if (!states_declared_)
        return Status::invalid_state("declare_states first", __FILE__, __LINE__);
    if (n == 0 || n > kMaxStepRows)
        return Status::invalid_argument("run_step: rows out of range", __FILE__, __LINE__);
    // declare 相の offsets 表を選ぶ（グラフ毎に一度だけ計算し、以降は使い回す）。
    // auto: 複数グラフ（例: decode rows=1 と prefill rows=N）を同じ実行器で使うと
    // declare 時の確保で足りないことがある（shelf-run がまさにこの形）。足りなければ
    // ここで広げる（縮めない。明示指定時は触らない）。
    // scratch の中身はステップ毎に失効する（行数も変わる）。
    fp8_owner_value_ = 0u;
    const compiler::ArenaPlan& plan = plan_for(g);
    if (arena_cap_ < plan.peak_bytes) {
        if (opts_.arena_bytes > 0)
            return Status::insufficient_memory(
                "アリーナが offsets 表に足りない（必要 " + std::to_string(plan.peak_bytes) +
                    " B / 確保 " + std::to_string(arena_cap_) + " B）",
                __FILE__, __LINE__);
        dev_release(arena_dev_);  // 直前ステップは同期済み
        arena_dev_ = nullptr;
        arena_cap_ = 0;
        if (const Status st =
                dev_reserve(&arena_dev_, plan.peak_bytes, "arena malloc (declare phase, per graph)");
            !st.ok()) {
            arena_dev_ = nullptr;
            return st;
        }
        arena_cap_ = plan.peak_bytes;
    }
    plan_ = &plan;
    value_dev_.clear();
    logits_dev_ = nullptr;   // 解放済みアリーナを指さないよう消す
    step_rows_ = n;
    step_base_pos_ = base_pos;

    const auto t0 = std::chrono::steady_clock::now();

    // エラー時のみ未消化のカーネル／コピーを先に流し切ってから返す。
    // ここで返った後に呼び出し側がメモリを解放すると、生きているカーネルが
    // 解放済みメモリを触り HSA のメモリフォルト／ハングになるため。
    const auto sync_on_error = [&](const Status& s) -> Status {
        if (!s.ok()) (void)hipStreamSynchronize(stream_);
        return s;
    };

    for (std::size_t i = 0; i < n; ++i) posrep_host_[i] = base_pos + static_cast<std::uint32_t>(i);
    hipk::GdnRequestDescriptor req{0u, 0u, u32(n)};
    SHELF_RETURN_IF_ERROR(sync_on_error(
        hipok(hipMemcpyAsync(req_dev_, &req, sizeof(req), hipMemcpyHostToDevice, stream_),
              "gdn req upload")));
    SHELF_RETURN_IF_ERROR(sync_on_error(
        hipok(hipMemcpyAsync(rowpos_dev_, posrep_host_.data(),
                             n * sizeof(std::uint32_t), hipMemcpyHostToDevice, stream_),
              "rowpos upload")));

    // データ入力: i32 = トークン（先頭ステージ）、それ以外 = 前段の hidden（以降のステージ）。
    const auto& inputs = g.all_inputs();
    if (inputs.empty())
        return Status::invalid_argument("graph lacks data input", __FILE__, __LINE__);
    // データ入力は必ず先頭（i32 = トークン、それ以外 = 前段 hidden）。
    const ir::ValueDesc* tokens_in = (inputs[0].dtype == DataType::i32) ? &inputs[0] : nullptr;
    const ir::ValueDesc* hidden_in = (inputs[0].dtype == DataType::i32) ? nullptr : &inputs[0];
    if (tokens_in != nullptr) {
        if (tokens == nullptr)
            return Status::invalid_argument("run_stage: このステージには tokens が必要",
                                            __FILE__, __LINE__);
        SHELF_RETURN_IF_ERROR(sync_on_error(
            hipok(hipMemcpyAsync(token_dev_, tokens, n * sizeof(std::int32_t),
                                 hipMemcpyHostToDevice, stream_),
                  "token upload")));
        value_dev_[tokens_in->id.value()] = ValueLoc{token_dev_, false, n};
    }
    if (hidden_in != nullptr) {
        if (hidden == nullptr)
            return Status::invalid_argument("run_stage: このステージには hidden が必要",
                                            __FILE__, __LINE__);
        const std::size_t elems = numel(hidden_in->shape);
        if (elems != n * (elems / n))
            return Status::invalid_argument("run_stage: hidden の要素数が行数で割れない",
                                            __FILE__, __LINE__);
        void* dst = static_cast<char*>(arena_dev_) + plan.input_offset;
        if (elems * sizeof(bf16_t) > plan.input_bytes) {
            return Status::insufficient_memory(
                "hidden 入力が declare 相の受け口より大きい", __FILE__, __LINE__);
        }
        SHELF_RETURN_IF_ERROR(sync_on_error(
            hipok(hipMemcpyAsync(dst, hidden, elems * sizeof(bf16_t),
                                 hipMemcpyHostToDevice, stream_),
                  "hidden upload")));
        value_dev_[hidden_in->id.value()] = ValueLoc{dst, false, elems};
    }
    if (tokens_in == nullptr && hidden_in == nullptr)
        return Status::invalid_argument("graph lacks data input", __FILE__, __LINE__);

    const auto& prims = g.primitives();
    // 開発用の内訳計測（SHELF_HIP_PROFILE=1）。primitive ごとに 2 イベントで挟み、
    // ステップ終端で同期して 1 行ずつ出す。既定は無効（イベントを作らない）。
    const bool profile = std::getenv("SHELF_HIP_PROFILE") != nullptr;
    hipEvent_t ev_a = nullptr, ev_b = nullptr;
    if (profile) {
        (void)hipEventCreate(&ev_a);
        (void)hipEventCreate(&ev_b);
    }
    for (std::size_t pi = 0; pi < prims.size(); ++pi) {
        const auto& p = prims[pi];
        if (profile) (void)hipEventRecord(ev_a, stream_);
        SHELF_RETURN_IF_ERROR(sync_on_error(run_primitive(g, p, pi, step_base_pos_)));
        if (profile) {
            (void)hipEventRecord(ev_b, stream_);
            (void)hipEventSynchronize(ev_b);
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, ev_a, ev_b);
            const std::uint32_t nlast =
                p.output.shape.rank() ? static_cast<std::uint32_t>(p.output.shape.dim(
                                            p.output.shape.rank() - 1u))
                                      : 0u;
            std::uint32_t kdim = 0;
            if (!p.inputs.empty() && step_rows_ > 0u) {
                const auto* iv = g.find_value(p.inputs[0]);
                if (iv)
                    kdim = static_cast<std::uint32_t>(numel(iv->shape) / step_rows_);
            }
            std::fprintf(stderr, "[profile] %4zu %-16s %8.3f ms  n=%u k=%u\n", pi,
                         ir::kind_name(p.kind), static_cast<double>(ms), nlast, kdim);
        }
        if (const char* d = std::getenv("SHELF_HIP_DEBUG_SYNC")) { (void)d;
            hipError_t e = hipStreamSynchronize(stream_);
            if (e != hipSuccess)
                return Status::from(Status::Code::device_error,
                                    std::string("after ") + ir::kind_name(p.kind) + ": " +
                                    hipGetErrorString(e), __FILE__, __LINE__);
        }
    }

    if (profile) {
        (void)hipStreamSynchronize(stream_);
        (void)hipEventDestroy(ev_a);
        (void)hipEventDestroy(ev_b);
    }
    SHELF_RETURN_IF_ERROR(hipok(hipStreamSynchronize(stream_), "step sync"));
    stats_.last_step_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
    return Status::make_ok();
}

Result<std::vector<float>> HipStepExecutor::fetch_value_debug(const ir::Graph& /*g*/,
                                                              ir::ValueId id) {
    auto it = value_dev_.find(id.value());
    if (it == value_dev_.end() || !it->second.dev)
        return Status::not_found("value not materialized", __FILE__, __LINE__);
    const ValueLoc& l = it->second;
    std::vector<bf16_t> tmp;
    std::vector<float> out(l.elems);
    if (l.f32) {
        SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(out.data(), l.dev, l.elems * sizeof(float),
                                              hipMemcpyDeviceToHost), "val readback"));
    } else if (id.value() == 0u && l.elems == 1 && !l.f32) {
        // token input: i32[1]
        std::int32_t t; hipMemcpy(&t, l.dev, 4, hipMemcpyDeviceToHost);
        out[0] = static_cast<float>(t);
    } else {
        tmp.resize(l.elems);
        SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(tmp.data(), l.dev, l.elems * sizeof(bf16_t),
                                              hipMemcpyDeviceToHost), "val readback"));
        for (std::size_t i = 0; i < l.elems; ++i)
            out[i] = __bfloat162float(__ushort_as_bfloat16(tmp[i].data));
    }
    return out;
}

Result<std::vector<bf16_t>> HipStepExecutor::fetch_value_bf16(const ir::Graph& /*g*/,
                                                               ir::ValueId id) {
    (void)hipSetDevice(opts_.device);
    auto it = value_dev_.find(id.value());
    if (it == value_dev_.end() || !it->second.dev)
        return Status::not_found("value not materialized", __FILE__, __LINE__);
    const ValueLoc& l = it->second;
    if (l.f32)
        return Status::invalid_state("fetch_value_bf16: f32 の値は PP の hand-off に使えない",
                                     __FILE__, __LINE__);
    std::vector<bf16_t> out(l.elems);
    SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(out.data(), l.dev, l.elems * sizeof(bf16_t),
                                          hipMemcpyDeviceToHost), "stage readback"));
    return out;
}

Result<std::vector<float>> HipStepExecutor::fetch_logits(const ir::Graph& g) {
    (void)hipSetDevice(opts_.device);
    if (!logits_dev_)
        return Status::invalid_state("logits not produced (run_step first)", __FILE__,
                                     __LINE__);
    std::size_t n = 0;
    for (const auto& p : g.primitives())
        if (p.kind == ir::Kind::output_projection) n = numel(p.output.shape);
    if (!n) return Status::invalid_state("graph lacks output_projection", __FILE__, __LINE__);
    std::vector<float> out(n);
    SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(out.data(), logits_dev_, n * sizeof(float),
                                          hipMemcpyDeviceToHost),
                                "logits readback"));
    return out;
}

Status HipStepExecutor::run_primitive(const ir::Graph& g, const ir::Primitive& p,
                                      std::size_t prim_index, std::uint32_t /*position*/) {
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
    // 出力の置き場は declare 相（compiler/arena_offsets.hpp）が決めている。
    // run 相は offset を引くだけで、割り当てはしない。f32 の別が食い違うと
    // nullptr になり、下の arena_ok が理由付きで止める。
    const auto alloc = [&](bool f32) -> void* { return slot_ptr(p.output.id, f32); };
    // アリーナ枯渇を「null 出力へのカーネル発射」（デバイスフォルトの原因）に
    // しないため、alloc の直後に必ず検査してから発射する。
    const auto arena_ok = [&](const void* ptr) -> Status {
        if (ptr) return Status::make_ok();
        const auto* slot = plan_ == nullptr ? nullptr : plan_->slot_of(p.output.id);
        std::string why = slot == nullptr ? "offsets 表に無い"
                                          : "f32/bf16 の方針が食い違う";
        return Status::insufficient_memory(
            std::string("declare 相の出力置き場が使えない（") + ir::kind_name(p.kind) + ": " +
                why + "）",
            __FILE__, __LINE__);
    };
    const auto scratch_ok = [&](const void* ptr) -> Status {
        if (ptr) return Status::make_ok();
        return Status::insufficient_memory(
            std::string("declare 相の一時領域が足りない（") + ir::kind_name(p.kind) +
                ": offsets 表の宣言より大きい）",
            __FILE__, __LINE__);
    };
    // primitive 内の一時領域（rope の f32 展開など）。宣言量より多く要れば止める。
    const auto scratch = [&](std::uint8_t slot, std::size_t bytes) -> void* {
        return scratch_ptr(prim_index, slot, bytes);
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
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        SHELF_RETURN_IF_ERROR(hipok(launch_embedding_bf16(
            static_cast<const bf16_t*>(wt(0)), token_dev_, static_cast<bf16_t*>(o),
            u32(step_rows_), u32(vocab), u32(H), u32(H), errword_dev_, stream_), "embedding"));
        return set_out(false, o);
    }
    case K::linear:
    case K::output_projection: {
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "linear"));
        const bool f32_out = (p.kind == K::output_projection);
        void* o = alloc(f32_out);
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        const std::size_t N = p.output.shape.dim(p.output.shape.rank() - 1);
        // 行数はこの primitive の出力から決める（`row_slice` で hidden を最終行に
        // 絞る経路では step_rows_ と違う行数的で走らせる）。
        if (N == 0u || out_n % N != 0u)
            return Status::invalid_state("linear: 出力形が不正", __FILE__, __LINE__);
        const std::size_t rows = out_n / N;
        if (rows == 0u || x->elems % rows != 0u)
            return Status::invalid_state("linear: 入力行数が不一致", __FILE__, __LINE__);
        const std::size_t K = x->elems / rows;  // input [rows, K]
        // 重みが FP8（E4M3）で upload されていれば W8A8 経路。同じ層の中でも
        // linear_attn 側は bf16 のまま（checkpoint の ignore）なので、この分岐は
        // 重み単位で決まる。
        const auto f8 = p.weights.empty()
                            ? weight_fp8_.end()
                            : weight_fp8_.find(p.weights[0].value());
        if (f8 != weight_fp8_.end()) {
            using namespace shelf::quant::fp8;
            const std::uint32_t kp = pad_k(u32(K));
            if (kp != u32(K))
                return Status::unsupported(
                    "linear(fp8): K が 32 の倍数でないと断片順に置けない", __FILE__, __LINE__);
            if (!fp8_gemm_shape_supported(u32(rows), u32(N), kp))
                return Status::unsupported("linear(fp8): 形が fp8 GEMM の受理範囲外",
                                           __FILE__, __LINE__);
            Fp8GemmScaleArgs sa;
            // 128 で割り切れない K ではグループ量子化を使えない（行全体に落とす）。
            sa.act_group =
                (opts_.fp8_act_group != 0u && (kp % kActGroupK) == 0u) ? kActGroupK : 0u;
            sa.weight_block128 = f8->second.block128;
            sa.activation_scale_stride = act_scale_groups(kp, sa.act_group);
            sa.weight_scale_k_blocks = scale_k_blocks(kp);
            if (!fp8_scale_args_supported(sa, u32(rows), u32(N), kp))
                return Status::unsupported("linear(fp8): スケール配置が不正", __FILE__,
                                           __LINE__);
            // scratch がすでに入力と同じ (値, 行数, K 切り, 粒度) を持てば再量子化しない。
            const bool reuse = p.inputs[0].value() == fp8_owner_value_ &&
                               fp8_owner_rows_ == u32(rows) && fp8_owner_kp_ == kp &&
                               fp8_owner_group_ == sa.act_group;
            if (!reuse) {
                SHELF_RETURN_IF_ERROR(ensure_fp8_scratch(fp8_activation_bytes(u32(rows), kp),
                                                         activation_scale_count(u32(rows), kp,
                                                                                sa.act_group)));
                SHELF_RETURN_IF_ERROR(hipok(launch_activation_quantize_fp8(
                                              static_cast<const bf16_t*>(x->dev), fp8_codes_dev_,
                                              fp8_scales_dev_, u32(rows), u32(K), kp, u32(K),
                                              sa.act_group, stream_),
                                          "fp8 act quantize"));
                fp8_owner_value_ = p.inputs[0].value();
                fp8_owner_rows_ = u32(rows);
                fp8_owner_kp_ = kp;
                fp8_owner_group_ = sa.act_group;
            }
            SHELF_RETURN_IF_ERROR(hipok(launch_gemm_fp8_w8a8(
                                          Fp8GemmConfig{}, f8->second.codes, f8->second.scales,
                                          fp8_codes_dev_, fp8_scales_dev_, o,
                                          f32_out ? Fp8GemmOutputDType::F32
                                                  : Fp8GemmOutputDType::BF16,
                                          u32(rows), u32(N), kp, u32(N), sa, stream_),
                                      "gemm fp8"));
            if (f32_out) logits_dev_ = static_cast<float*>(o);
            return set_out(f32_out, o);
        }
        // bench_gemv_hip on the model's real shapes: rows==1 -> ExactRows
        // wins everywhere; rows>=2 -> ExactRowsDot2 wins on most N (k/v/o/
        // mlp/gdn_out), losing only ~1-4 us on q_gate/gdn_qkv. Net positive.
        // prefill（64 行の束）は 2 次元タイルで回す。ExactRows 系は 16 行ずつ 4 回に
        // 分かれるため行方向の再利用が無く、実効 71 GB/s（DRAM 境界の 1/8）だった。
        Bf16GemmConfig cfg;
        if (rows >= 64u && bf16_prefill2d_supported(u32(rows), u32(N), u32(K))) {
            cfg.id = Bf16GemmConfigId::Prefill2D;
            cfg.exact_rows = 0u;
        } else if (rows > 16u) {
            // 行数が 16 を超える形は ExactRows 系が使えない（カーネルは 16 行まで）。
            // 汎用の Wmma 経路に落とす（出力タイル 16 で任意の行数・出力数）。
            cfg.id = Bf16GemmConfigId::Wmma;
            cfg.exact_rows = 0u;
        } else {
            cfg.id = rows >= 2u ? Bf16GemmConfigId::ExactRowsDot2
                                : Bf16GemmConfigId::ExactRows;
            cfg.exact_rows = static_cast<std::uint8_t>(rows);
        }
        SHELF_RETURN_IF_ERROR(hipok(launch_gemm_bf16(
            cfg, static_cast<const bf16_t*>(wt(0)), static_cast<const bf16_t*>(x->dev), o,
            f32_out ? Bf16GemmOutputDType::F32 : Bf16GemmOutputDType::BF16,
            u32(rows), u32(N), u32(K), u32(K), u32(N), stream_), "gemm"));
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
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        // 融合: bf16 出力をもう一度読んで量子化する pass を消す（radiance の
        // `rmsnorm_quant_fp8` と同じ形）。bf16 出力も従来どおり書くので、
        // bf16 で読む消費者（PP の受け渡しやデバッグ）はそのまま動く。
        const auto& fuse = fuse_for(g);
        if (p.attr.gamma_plus_one &&
            std::find(fuse.rmsnorm_out.begin(), fuse.rmsnorm_out.end(),
                      p.output.id.value()) != fuse.rmsnorm_out.end() &&
            F % shelf::quant::fp8::kActGroupK == 0u &&
            shelf::quant::fp8::pad_k(u32(F)) == u32(F)) {
            using namespace shelf::quant::fp8;
            const std::uint32_t kp = u32(F);
            if (ensure_fp8_scratch(fp8_activation_bytes(u32(rows), kp),
                                   activation_scale_count(u32(rows), kp, kActGroupK)).ok()) {
                const hipError_t ef = launch_rmsnorm_bf16_quantize(
                    x->dev, wt(0), o, fp8_codes_dev_, fp8_scales_dev_, u32(rows), u32(F),
                    u32(F), u32(F), p.attr.epsilon, stream_);
                if (ef == hipSuccess) {
                    fp8_owner_value_ = p.output.id.value();
                    fp8_owner_rows_ = u32(rows);
                    fp8_owner_kp_ = kp;
                    fp8_owner_group_ = kActGroupK;
                    return set_out(false, o);
                }
                fp8_owner_value_ = 0u;   // NotSupported 等は従来経路
            }
        }
        SHELF_RETURN_IF_ERROR(hipok(launch_rmsnorm(
            x->dev, RmsNormDataType::BF16, wt(0), RmsNormDataType::BF16,
            RmsNormWeightLayout::PER_FEATURE,
            p.attr.gamma_plus_one ? RmsNormWeightMode::ONE_PLUS : RmsNormWeightMode::DIRECT,
            o, RmsNormDataType::BF16, u32(rows), u32(F), u32(F), u32(F), u32(F),
            p.attr.epsilon, stream_), "rmsnorm"));
        return set_out(false, o);
    }
    case K::rope: {
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "rope"));
        const std::size_t D = p.output.shape.dim(p.output.shape.rank() - 1);
        const std::size_t heads = (p.output.shape.rank() >= 3u)
                                      ? p.output.shape.dim(p.output.shape.rank() - 2)
                                      : 1u;
        const std::size_t rd0 = static_cast<std::size_t>(p.attr.rotary_dim);
        // ① bf16 入力・1 トークン 1 workgroup の pair kernel。
        //    f32 中間材料（往復 3 通過）を丸ごと消す。head_dim 256・rotary 64 のときだけ。
        if (D == kRopePairHeadDim && rd0 == kRopePairRotary &&
            out_n == step_rows_ * heads * D) {
            const std::size_t feats = heads * D;
            void* o = alloc(false);
            SHELF_RETURN_IF_ERROR(arena_ok(o));
            const hipError_t e = launch_rope_bf16_pair(
                static_cast<const bf16_t*>(x->dev), u32(feats), static_cast<bf16_t*>(o),
                u32(feats), rowpos_dev_, u32(step_rows_), u32(feats), u32(D), u32(rd0),
                static_cast<float>(opts_.rope_theta), inv_freq_dev_, 1.0f, stream_);
            if (e == hipSuccess) return set_out(false, o);
            if (e != hipErrorNotSupported) return hipok(e, "rope bf16 pair");
        }
        // ② 以降は従来経路（f32 中間材料を作る）
        void* xf = scratch(0, out_n * sizeof(float));
        void* o = alloc(false);
        if (!xf || !o) return Status::insufficient_memory("arena exhausted", __FILE__, __LINE__);
        SHELF_RETURN_IF_ERROR(hipok(
            launch_bf16_to_f32(static_cast<const bf16_t*>(x->dev),
                               static_cast<float*>(xf), u32(out_n), stream_),
            "rope cvt"));
        // head_dim 256 の pair kernel は **1 トークン 1 workgroup**で全ヘッドを回し、
        // sincos は 1 トークンあたり rotary/2 個だけ計算する（ヘッド数だけ作り直さない）。
        // 従来は (トークン × ヘッド) を行として渡していたため、同じ sincos を
        // ヘッド数ぶん重複計算していた（27B で 24 倍。rope 19.1 ms → 参照 2.7 ms の主因）。
        if (D == kRopePairHeadDim && rd0 == kRopePairRotary &&
            out_n == step_rows_ * heads * D) {
            const std::size_t feats = heads * D;
            void* o2 = alloc(false);
            SHELF_RETURN_IF_ERROR(arena_ok(o2));
            SHELF_RETURN_IF_ERROR(hipok(launch_rope_f32_bf16(
                                          static_cast<const float*>(xf), u32(feats),
                                          static_cast<bf16_t*>(o2), u32(feats), rowpos_dev_,
                                          u32(step_rows_), u32(feats), u32(D),
                                          u32(p.attr.rotary_dim),
                                          static_cast<float>(opts_.rope_theta),
                                          inv_freq_dev_, 1.0f, stream_),
                                      "rope pair/token"));
            return set_out(false, o2);
        }
        const std::size_t rows = out_n / D;               // P * heads_per_token
        const std::size_t per_tok = rows / step_rows_;    // q or kv heads
        if (rows % step_rows_) return Status::invalid_state("rope: row count not multiple of batch", __FILE__, __LINE__);
        const std::size_t rd = static_cast<std::size_t>(p.attr.rotary_dim);
        // rope position of row r = base_pos + r/per_tok (all heads of one token share pos)
        // 行数 × ヘッド数が確保数を超えたら伸ばす（512 行 × 64 head など）。
        if (rows > posrep_cap_) {
            std::size_t cap = posrep_cap_;
            while (cap < rows) cap *= 2u;
            dev_release(posrep_dev_);
            posrep_dev_ = nullptr;
            if (const Status st = dev_alloc(&posrep_dev_, cap, "pos scratch grow"); !st.ok()) {
                posrep_dev_ = nullptr;
                return st;
            }
            posrep_host_.assign(cap, 0u);
            posrep_cap_ = cap;
        }
        for (std::size_t r = 0; r < rows; ++r)
            posrep_host_[r] = step_base_pos_ + static_cast<std::uint32_t>(r / per_tok);
        SHELF_RETURN_IF_ERROR(hipok(hipMemcpyAsync(posrep_dev_, posrep_host_.data(),
                                                   rows * sizeof(std::uint32_t),
                                                   hipMemcpyHostToDevice, stream_),
                                    "rope pos upload"));
        SHELF_RETURN_IF_ERROR(hipok(launch_rope_f32_bf16(
            static_cast<const float*>(xf), u32(D), static_cast<bf16_t*>(o), u32(D),
            posrep_dev_, u32(rows), u32(D), u32(D), u32(rd),
            static_cast<float>(opts_.rope_theta), inv_freq_dev_, 1.0f, stream_),
            "rope"));
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
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        SHELF_RETURN_IF_ERROR(hipok(launch_bf16_row_slice(
            static_cast<const bf16_t*>(x->dev), static_cast<bf16_t*>(o), u32(rows),
            u32(end - begin), u32(begin), u32(F_in), stream_), "slice"));
        return set_out(false, o);
    }
    case K::row_slice: {
        // 先頭次元（行）の連続スライス。行はメモリ上で連続なので、読む行の
        // 先頭ポインタを進めるだけのコピーで済む（`slice` のような gather ではない）。
        // prefill の lm_head を最終行だけに絞る経路がこれを使う。
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "row_slice"));
        if (x->f32)
            return Status::invalid_state("row_slice: bf16 活性化のみ対応", __FILE__, __LINE__);
        const auto* xs = g.find_value(p.inputs[0]);
        const std::size_t R = xs ? xs->shape.dim(0) : 0u;
        if (R == 0u || x->elems % R != 0u)
            return Status::invalid_state("row_slice: 入力行数が不明", __FILE__, __LINE__);
        const std::size_t C = x->elems / R;   // 1 行の要素数
        // `row_last` は「このステップで実際に渡された行」の最終行（n < rows_ の
        // 部分チャンクでも正しい行を選ぶ）。それ以外は静的範囲。
        const std::size_t rb = p.attr.row_last ? step_rows_ - 1u
                                               : static_cast<std::size_t>(p.attr.row_begin);
        const std::size_t re = p.attr.row_last ? step_rows_
                                               : (p.attr.row_end < 0 ? R
                                                  : static_cast<std::size_t>(p.attr.row_end));
        if (rb >= re || re > R)
            return Status::invalid_state("row_slice: 行範囲が不正", __FILE__, __LINE__);
        if (out_n != (re - rb) * C)
            return Status::invalid_state("row_slice: 出力形が不一致", __FILE__, __LINE__);
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        SHELF_RETURN_IF_ERROR(hipok(launch_bf16_row_slice(
            static_cast<const bf16_t*>(x->dev) + rb * C, static_cast<bf16_t*>(o),
            u32(re - rb), u32(C), 0u, u32(C), stream_), "row_slice"));
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
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        SHELF_RETURN_IF_ERROR(hipok(launch_bf16_sigmoid_gate(
            static_cast<const bf16_t*>(x->dev), static_cast<const bf16_t*>(z->dev),
            static_cast<bf16_t*>(o), u32(out_n), stream_), "sigmoid_gate"));
        return set_out(false, o);
    }
    case K::silu: {
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "silu"));
        void* o = alloc(true);  // bf16 -> f32 (no bf16->bf16 silu in family)
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);
        SHELF_RETURN_IF_ERROR(hipok(launch_silu_bf16_f32(
            static_cast<const bf16_t*>(x->dev), u32(F), static_cast<float*>(o),
            u32(F), u32(out_n / F), u32(F), stream_), "silu"));
        return set_out(true, o);
    }
    case K::swiglu: {
        // gate / up は bf16（linear 出力）。f32 中間を作らず 1 pass で积を出す。
        // （局所名は gt/up。`g` はグラフの名前なので隠さない）
        ValueLoc* gt = in(0);
        ValueLoc* up = in(1);
        SHELF_RETURN_IF_ERROR(need(gt, false, "swiglu gate"));
        SHELF_RETURN_IF_ERROR(need(up, false, "swiglu up"));
        if (gt->f32 || up->f32)
            return Status::unsupported("swiglu: bf16 入力のみ対応", __FILE__, __LINE__);
        const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);
        const std::size_t rows = out_n / F;
        // 融合: この値を使う fp8 GEMM がすぐ来るなら、bf16 中間材料を書かずに
        // コード/スケールだけ生成する（bf16 の往復 2 回分と発射 1 回を浮かせられる）。
        const auto& fuse = fuse_for(g);
        const bool fusable = std::find(fuse.swiglu_out.begin(), fuse.swiglu_out.end(),
                                       p.output.id.value()) != fuse.swiglu_out.end() &&
                             F % shelf::quant::fp8::kActGroupK == 0u;
        if (fusable) {
            using namespace shelf::quant::fp8;
            const std::uint32_t kp = u32(F);
            if (ensure_fp8_scratch(fp8_activation_bytes(u32(rows), kp),
                                   activation_scale_count(u32(rows), kp, kActGroupK)).ok()) {
                const hipError_t ef = launch_swiglu_quantize_fp8(
                    static_cast<const bf16_t*>(gt->dev), static_cast<const bf16_t*>(up->dev),
                    nullptr,
                    fp8_codes_dev_, fp8_scales_dev_, u32(rows), kp, kp, u32(F), u32(F),
                    0u, kActGroupK, stream_);
                if (ef == hipSuccess) {
                    fp8_owner_value_ = p.output.id.value();
                    fp8_owner_rows_ = u32(rows);
                    fp8_owner_kp_ = kp;
                    fp8_owner_group_ = kActGroupK;
                    void* o = alloc(false);           // 使われないが offsets 表の枠は確保
                    SHELF_RETURN_IF_ERROR(arena_ok(o));
                    return set_out(false, o);
                }
                fp8_owner_value_ = 0u;                // NotSupported 等は従来経路へ
            }
        }
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        SHELF_RETURN_IF_ERROR(hipok(launch_swiglu_bf16(
            static_cast<const bf16_t*>(gt->dev), u32(F), static_cast<const bf16_t*>(up->dev),
            u32(F), static_cast<bf16_t*>(o), u32(F), u32(rows), u32(F), stream_), "swiglu"));
        return set_out(false, o);
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
        else { af = scratch(0, out_n * sizeof(float)); pa32 = static_cast<const float*>(af);
               SHELF_RETURN_IF_ERROR(scratch_ok(af));
               SHELF_RETURN_IF_ERROR(hipok(launch_bf16_to_f32(
                   static_cast<const bf16_t*>(a->dev), static_cast<float*>(af), u32(a->elems),
                   stream_), "mul cvt a")); }
        if (b32) pb32 = static_cast<const float*>(b->dev);
        else { bf = scratch(1, out_n * sizeof(float)); pb32 = static_cast<const float*>(bf);
               SHELF_RETURN_IF_ERROR(scratch_ok(bf));
               SHELF_RETURN_IF_ERROR(hipok(launch_bf16_to_f32(
                   static_cast<const bf16_t*>(b->dev), static_cast<float*>(bf), u32(b->elems),
                   stream_), "mul cvt b")); }
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        const std::size_t Fm = p.output.shape.dim(p.output.shape.rank() - 1);
        SHELF_RETURN_IF_ERROR(hipok(launch_mul_f32_f32_bf16(
            pa32, u32(ars), pb32, u32(brs), static_cast<bf16_t*>(o), u32(Fm),
            u32(out_n / Fm), u32(Fm), stream_), "elem_mul"));
        (void)af; (void)bf;
        return set_out(false, o);
    }
    case K::residual_add: {
        ValueLoc* a = in(0), *b = in(1);
        SHELF_RETURN_IF_ERROR(need(a, false, "residual_add"));
        SHELF_RETURN_IF_ERROR(need(b, false, "residual_add"));
        const std::size_t F = p.output.shape.dim(p.output.shape.rank() - 1);
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        SHELF_RETURN_IF_ERROR(hipok(launch_residual_add_bf16(
            static_cast<const bf16_t*>(a->dev), u32(F), static_cast<const bf16_t*>(b->dev),
            u32(F), static_cast<bf16_t*>(o), u32(F), u32(out_n / F), u32(F), stream_),
            "residual_add"));
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
        void* of = scratch(0, x->elems * sizeof(float));
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(arena_ok(of));
        SHELF_RETURN_IF_ERROR(arena_ok(o));
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
        SHELF_RETURN_IF_ERROR(hipok(launch_fused_gdn_conv_silu(args, stream_), "conv_silu"));
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
        const TensorShape& vs = g.find_value(p.inputs[2])->shape;  // [nvh,dv]|[P,nvh,dv]
        const std::size_t nkh = qs.dim(qs.rank() - 2), dk = qs.dim(qs.rank() - 1);
        const std::size_t nvh = vs.dim(vs.rank() - 2), dv = vs.dim(vs.rank() - 1);
        const std::size_t qk_n = nkh * dk;

        const std::size_t prep_n = step_rows_ * qk_n;   // one row per token
        void* nqf = scratch(0, prep_n * sizeof(float));
        void* sqf = scratch(1, prep_n * sizeof(float));
        void* nkf = scratch(2, prep_n * sizeof(float));
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
        SHELF_RETURN_IF_ERROR(hipok(launch_fused_gdn_qk_prepare(prep, stream_), "qk_prepare"));

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
            hipStreamSynchronize(stream_);
            std::vector<float> sqh(qk_n), nkh_(qk_n), oh(nvh * dv), sh(std::min<std::size_t>(8, st.elems));
            std::vector<bf16_t> vh(nvh * dv), ah(nvh), bh(nvh);
            hipMemcpy(sqh.data(), sqf, qk_n * 4, hipMemcpyDeviceToHost);
            hipMemcpy(nkh_.data(), nkf, qk_n * 4, hipMemcpyDeviceToHost);
            hipMemcpy(vh.data(), v->dev, nvh * dv * 2, hipMemcpyDeviceToHost);
            hipMemcpy(ah.data(), a->dev, nvh * 2, hipMemcpyDeviceToHost);
            hipMemcpy(bh.data(), b->dev, nvh * 2, hipMemcpyDeviceToHost);
            hipMemcpy(oh.data(), o, nvh * dv * 4, hipMemcpyDeviceToHost);
            std::fprintf(stderr, "[ptr] q=%p k=%p nq=%p sq=%p nk=%p v=%p out=%p\n", q->dev, k->dev, nqf, sqf, nkf, v->dev, o);
            std::fprintf(stderr, "[gdn P=%zu pre] sq[0..3]=%.6g,%.6g,%.6g,%.6g nk[0..3]=%.6g,%.6g,%.6g,%.6g v0=%.6g a0=%.6g b0=%.6g out0=%.6g out_last=%.6g st0=%.6g\n",
                step_rows_, sqh[0], sqh[1], sqh[2], sqh[3], nkh_[0], nkh_[1], nkh_[2], nkh_[3],
                __bfloat162float(__ushort_as_bfloat16(vh[0].data)),
                __bfloat162float(__ushort_as_bfloat16(ah[0].data)),
                __bfloat162float(__ushort_as_bfloat16(bh[0].data)),
                oh[0], oh[oh.size()-1], sh[0]);
        }
        // prefill（複数行）はチャンク版（FP16 WMMA）で回す。1 行の decode は
        // 逐次の方が速いので従来のカーネルのまま。
        // `SHELF_GDN_CHUNK=0` で従来経路に固定できる（数値の A/B 用）。
        const char* chunk_env = std::getenv("SHELF_GDN_CHUNK");
        const bool want_chunk =
            step_rows_ > 1u && (chunk_env == nullptr || chunk_env[0] != '0') &&
            hipk::gdn_chunk_supported(rec);
        if (want_chunk) {
            // 1 発射（チャンクはカーネル内ループ）。チャンク毎に発射すると
            // 2048 行で 64 回/層 = 3,072 dispatch になり、発射間 gap が実額で出る。
            SHELF_RETURN_IF_ERROR(
                hipok(hipk::launch_gdn_rows(rec, 0u, u32(step_rows_), stream_), "gdn chunk"));
        } else {
            // decode（1 行）。WMMA 版を試み、受理できなければスカラー f32 版に戻る。
            // SHELF_GDN_DECODE1 = off（スカラー）| lossy（f16 一成分）| exact（hi/lo・既定）
            const char* d1 = std::getenv("SHELF_GDN_DECODE1");
            const bool use_wmma = !(d1 && d1[0] == 'o') &&
                                  hipk::gdn_recurrence_wmma_decode1_supported(rec);
            if (use_wmma) {
                const bool lossy = d1 && d1[0] == 'l';
                SHELF_RETURN_IF_ERROR(hipok(
                    hipk::launch_gdn_recurrence_wmma_decode1(rec, lossy, stream_),
                    "gdn decode1 wmma"));
            } else {
                SHELF_RETURN_IF_ERROR(
                    hipok(launch_gdn_recurrence_f32(rec, stream_), "recurrence"));
            }
        }
        if (std::getenv("SHELF_DEBUG_GDN")) {
            hipStreamSynchronize(stream_);
            std::vector<float> oh(nvh * dv), sh(std::min<std::size_t>(4, st.elems));
            hipMemcpy(oh.data(), o, nvh * dv * 4, hipMemcpyDeviceToHost);
            hipMemcpy(sh.data(), st.dev, sh.size() * 4, hipMemcpyDeviceToHost);
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
            // f32 入力でも bf16 で出力する（本体が store 前に f32→bf16 の丸めを
            // 噛ませるので、別 pass で変換するのと 1 バイト違わない）。変換 pass
            // （読み 4 B + 書き 2 B / 要素 × 48 レイヤ）を消す。
            norm = scratch(0, out_n * sizeof(bf16_t));
            SHELF_RETURN_IF_ERROR(scratch_ok(norm));
            SHELF_RETURN_IF_ERROR(hipok(launch_rmsnorm(
                x->dev, RmsNormDataType::F32, wt(0), RmsNormDataType::BF16,
                RmsNormWeightLayout::PER_FEATURE, RmsNormWeightMode::DIRECT, norm,
                RmsNormDataType::BF16, u32(rows), u32(F), u32(F), u32(F), u32(F),
                p.attr.epsilon, stream_), "gdn norm"));
        } else {
            norm = scratch(0, out_n * sizeof(bf16_t));
            SHELF_RETURN_IF_ERROR(scratch_ok(norm));
            SHELF_RETURN_IF_ERROR(hipok(launch_rmsnorm(
                x->dev, RmsNormDataType::BF16, wt(0), RmsNormDataType::BF16,
                RmsNormWeightLayout::PER_FEATURE, RmsNormWeightMode::DIRECT, norm,
                RmsNormDataType::BF16, u32(rows), u32(F), u32(F), u32(F), u32(F),
                p.attr.epsilon, stream_), "gdn norm"));
        }
        SHELF_RETURN_IF_ERROR(need(z, false, "rmsnorm_gated z"));
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        // gate 掛け算の结果を fp8 量子化まで 1 発射で出す（radiance の `gated_quant_fp8` 相当）。
        // bf16 出力も同時に書くので、消費者が owner 再利用に失敗しても値は正しい。
        // fp8 の行は **トークン**、K は平坦化した幅（[tokens, heads, dv] の dense 列）。
        const std::size_t feat_flat = step_rows_ != 0u ? out_n / step_rows_ : 0u;
        const auto& gfuse = fuse_for(g);
        // rows==1（decode）では 2 発射に負けた（54.19 → 54.36 ms/tok）ので prefill のみ。
        if (feat_flat != 0u && step_rows_ > 1u &&
            feat_flat % shelf::quant::fp8::kActGroupK == 0u &&
            shelf::quant::fp8::pad_k(u32(feat_flat)) == u32(feat_flat) &&
            std::find(gfuse.gated_out.begin(), gfuse.gated_out.end(),
                      p.output.id.value()) != gfuse.gated_out.end()) {
            using namespace shelf::quant::fp8;
            const std::uint32_t kp = u32(feat_flat);
            if (ensure_fp8_scratch(fp8_activation_bytes(u32(step_rows_), kp),
                                   activation_scale_count(u32(step_rows_), kp,
                                                         kActGroupK)).ok()) {
                const hipError_t ef = launch_swiglu_quantize_fp8(
                    static_cast<const bf16_t*>(z->dev), static_cast<const bf16_t*>(norm),
                    static_cast<bf16_t*>(o),
                    fp8_codes_dev_, fp8_scales_dev_, u32(step_rows_), kp, kp, kp, kp, kp,
                    kActGroupK, stream_);
                if (ef == hipSuccess) {
                    fp8_owner_value_ = p.output.id.value();
                    fp8_owner_rows_ = u32(step_rows_);
                    fp8_owner_kp_ = kp;
                    fp8_owner_group_ = kActGroupK;
                    return set_out(false, o);
                }
                fp8_owner_value_ = 0u;
            }
        }
        SHELF_RETURN_IF_ERROR(hipok(launch_swiglu_bf16(
            static_cast<const bf16_t*>(z->dev), u32(F), static_cast<const bf16_t*>(norm),
            u32(F), static_cast<bf16_t*>(o), u32(F), u32(rows), u32(F), stream_), "gdn post"));
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

        KvAppendBf16Args ap{};
        ap.common.k_input = static_cast<const bf16_t*>(k->dev);
        ap.common.v_input = static_cast<const bf16_t*>(v->dev);
        ap.common.k_row_stride = u32(k->elems / P);
        ap.common.v_row_stride = u32(v->elems / P);
        ap.common.row_sequence_slots = slot_zero_dev_;
        ap.common.row_positions = rowpos_dev_;
        ap.common.block_tables = st.table;
        ap.common.block_table_stride = u32(st.pages);
        ap.common.max_sequences = 1u;
        ap.common.rows = u32(step_rows_);
        ap.common.layer = 0u;
        ap.common.page_tokens = u32(bs);
        ap.common.num_pages = u32(st.pages);
        ap.common.num_attention_layers = 1u;
        ap.common.kv_heads = u32(kvh);
        ap.common.head_dim = u32(D);
        ap.common.elems_per_token = ept;
        ap.common.elems_per_page = epp;
        ap.common.elems_per_layer = epl;
        ap.k_pool = static_cast<bf16_t*>(st.k_pool);
        ap.v_pool = static_cast<bf16_t*>(st.v_pool);
        SHELF_RETURN_IF_ERROR(hipok(launch_kv_append_bf16(ap.common, ap.k_pool, ap.v_pool,
                                                          stream_), "kv_append"));
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(arena_ok(o));
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
        // 複数行（prefill）で受理条件を満たすならタイル化 prefill kernel。
        // 1 行（decode）は従来どおり（grid が 1 CTA に直列化して遅くなる）。
        // `SHELF_ATTN_PREFILL=0` で per-query 経路に固定できる（数値の A/B 用）。
        const char* attn_env = std::getenv("SHELF_ATTN_PREFILL");
        const bool prefill_path = attention_paged_prefill_bf16_supported(at) &&
                                  (attn_env == nullptr || attn_env[0] != '0');
        {
            const hipError_t ea = prefill_path
                ? launch_attention_paged_prefill_bf16(
                      at, static_cast<const bf16_t*>(st.k_pool),
                      static_cast<const bf16_t*>(st.v_pool), stream_)
                : launch_attention_paged_bf16(
                      at, static_cast<const bf16_t*>(st.k_pool),
                      static_cast<const bf16_t*>(st.v_pool), stream_);
            SHELF_RETURN_IF_ERROR(hipok(ea, "attention paged"));
        }
        // NOTE(split-KV decode): `launch_attention_paged_bf16_split` + `..._split_reduce`
        //   は rows==1 で context を splits に割る実装（radiance の split-KV decode と同形）で、
        //   実行もできたが ctx 2048/4096 で step_ms は 54.06 → 54.10/54.14（4096 で 55.87 横ばい）。
        //   decode attention の律速は workgroup 数ではない。配線は入れていない。
        
        if (std::getenv("SHELF_DEBUG_ATTN"))
            std::fprintf(stderr,
                         "[attn] rows=%zu D=%zu page=%zu qh=%zu kvh=%zu stride=%u npages=%u "
                         "path=%s\n", step_rows_, D, bs, nq, kvh, u32(st.pages),
                         u32(st.pages), prefill_path ? "prefill_wmma" : "per_query");
        st.rows += step_rows_;
        return set_out(false, o);
    }
    default:
        return Status::unsupported((std::string(ir::kind_name(p.kind)) +
                                    " not wired on hip executor").c_str(),
                                   __FILE__, __LINE__);
    }
}

} // namespace shelf::backend::hip
