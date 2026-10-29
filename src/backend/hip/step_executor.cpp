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

HipStepExecutor::HipStepExecutor(HipStepExecutorOptions opts) : opts_(opts) {
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
    const hipError_t e4 = malloc_dev(&slot_zero_dev_, 1u);
    const hipError_t e5 = malloc_dev(&errword_dev_, 1u);
    const hipError_t e6 = malloc_dev(&req_dev_, 1u);
    const hipError_t e7 = malloc_dev(&inv_freq_dev_,
                                      std::max<std::size_t>(2, opts_.rope_rotary_dim / 2));
    (void)e0; (void)e1; (void)e2; (void)e3; (void)e4; (void)e5; (void)e6; (void)e7;
    hipMemset(slot_zero_dev_, 0, sizeof(std::uint32_t));
    posrep_host_.assign(kPosScratch, 0u);
}

HipStepExecutor::~HipStepExecutor() {
    (void)hipSetDevice(opts_.device);
    // 解放の先にキューを消化する。発射済みカーネル／コピーが生きているまま
    // メモリを解放すると HSA のメモリフォルト／ハングになり得る
    // （エラー経路でここへ来ることがある）。
    if (stream_) (void)hipStreamSynchronize(stream_);
    for (auto& [id, loc] : state_dev_) {
        if (loc.dev) hipFree(loc.dev);
        if (loc.k_pool) hipFree(loc.k_pool);
        if (loc.v_pool) hipFree(loc.v_pool);
        if (loc.table) hipFree(loc.table);
    }
    for (auto& [id, p] : weight_dev_) hipFree(p);
    for (auto& [id, w] : weight_fp8_) {
        if (w.codes) hipFree(w.codes);
        if (w.scales) hipFree(w.scales);
    }
    hipFree(fp8_codes_dev_);
    hipFree(fp8_scales_dev_);
    hipFree(arena_dev_);
    hipFree(token_dev_);
    hipFree(posrep_dev_);
    hipFree(rowpos_dev_);
    hipFree(slot_zero_dev_);
    hipFree(errword_dev_);
    hipFree(req_dev_);
    hipFree(inv_freq_dev_);
    hipStreamDestroy(stream_);
}

Status HipStepExecutor::upload_weight(TensorId id, const std::vector<float>& f32) {
    (void)hipSetDevice(opts_.device);
    if (weight_dev_.count(id.value()))
        return Status::invalid_state("weight uploaded twice", __FILE__, __LINE__);
    std::vector<bf16_t> b(f32.size());
    for (std::size_t i = 0; i < f32.size(); ++i) b[i] = f32_to_bf16(f32[i]);
    void* dev = nullptr;
    SHELF_RETURN_IF_ERROR(hipok(hipMalloc(&dev, b.size() * sizeof(bf16_t)), "malloc weight"));
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
    SHELF_RETURN_IF_ERROR(hipok(malloc_dev(&dev, n), "malloc weight bf16"));
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
    SHELF_RETURN_IF_ERROR(hipok(malloc_dev(&cdev, n_codes), "malloc weight fp8 codes"));
    SHELF_RETURN_IF_ERROR(hipok(hipMalloc(&sdev, want_scales * sizeof(float)),
                                "malloc weight fp8 scales"));
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
        if (fp8_codes_dev_) hipFree(fp8_codes_dev_);
        fp8_codes_dev_ = nullptr;
        SHELF_RETURN_IF_ERROR(hipok(malloc_dev(&fp8_codes_dev_, code_bytes),
                                    "malloc fp8 act codes"));
        fp8_codes_cap_ = code_bytes;
    }
    if (scale_floats > fp8_scales_cap_) {
        if (fp8_scales_dev_) hipFree(fp8_scales_dev_);
        fp8_scales_dev_ = nullptr;
        SHELF_RETURN_IF_ERROR(hipok(hipMalloc(&fp8_scales_dev_, scale_floats * sizeof(float)),
                                    "malloc fp8 act scales"));
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
                hipok(hipMalloc(&loc.k_pool, pool_elems * sizeof(bf16_t)), "kv pool K"));
            SHELF_RETURN_IF_ERROR(
                hipok(hipMalloc(&loc.v_pool, pool_elems * sizeof(bf16_t)), "kv pool V"));
            SHELF_RETURN_IF_ERROR(
                hipok(hipMemset(loc.k_pool, 0, pool_elems * sizeof(bf16_t)), "zero kv K"));
            SHELF_RETURN_IF_ERROR(
                hipok(hipMemset(loc.v_pool, 0, pool_elems * sizeof(bf16_t)), "zero kv V"));
            std::vector<std::uint32_t> t(loc.pages);
            for (std::size_t i = 0; i < loc.pages; ++i) t[i] = u32(i);
            SHELF_RETURN_IF_ERROR(hipok(malloc_dev(&loc.table, t.size()), "kv table"));
            SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(loc.table, t.data(),
                                                  t.size() * sizeof(std::uint32_t),
                                                  hipMemcpyHostToDevice),
                                        "kv table upload"));
        } else {
            loc.elems = numel(sh);
            SHELF_RETURN_IF_ERROR(
                hipok(hipMalloc(&loc.dev, loc.elems * sizeof(float)), "state alloc"));
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
        if (arena_dev_) {
            (void)hipFree(arena_dev_);
            arena_dev_ = nullptr;
        }
        const hipError_t ea = hipMalloc(&arena_dev_, bytes);
        if (ea != hipSuccess) {
            arena_cap_ = 0;
            return hipok(ea, "arena malloc (declare phase)");
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
    const compiler::ArenaPlan& plan = plan_for(g);
    if (arena_cap_ < plan.peak_bytes) {
        if (opts_.arena_bytes > 0)
            return Status::insufficient_memory(
                "アリーナが offsets 表に足りない（必要 " + std::to_string(plan.peak_bytes) +
                    " B / 確保 " + std::to_string(arena_cap_) + " B）",
                __FILE__, __LINE__);
        if (arena_dev_) (void)hipFree(arena_dev_);  // 直前ステップは同期済み
        arena_dev_ = nullptr;
        const hipError_t ea = hipMalloc(&arena_dev_, plan.peak_bytes);
        if (ea != hipSuccess) {
            arena_cap_ = 0;
            return hipok(ea, "arena malloc (declare phase, per graph)");
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
    for (std::size_t pi = 0; pi < prims.size(); ++pi) {
        const auto& p = prims[pi];
        SHELF_RETURN_IF_ERROR(sync_on_error(run_primitive(g, p, pi, step_base_pos_)));
        if (const char* d = std::getenv("SHELF_HIP_DEBUG_SYNC")) { (void)d;
            hipError_t e = hipStreamSynchronize(stream_);
            if (e != hipSuccess)
                return Status::from(Status::Code::device_error,
                                    std::string("after ") + ir::kind_name(p.kind) + ": " +
                                    hipGetErrorString(e), __FILE__, __LINE__);
        }
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
        const std::size_t rows = step_rows_;
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
            SHELF_RETURN_IF_ERROR(ensure_fp8_scratch(fp8_activation_bytes(u32(rows), kp),
                                                     activation_scale_count(u32(rows), kp,
                                                                            sa.act_group)));
            SHELF_RETURN_IF_ERROR(hipok(launch_activation_quantize_fp8(
                                          static_cast<const bf16_t*>(x->dev), fp8_codes_dev_,
                                          fp8_scales_dev_, u32(rows), u32(K), kp, u32(K),
                                          sa.act_group, stream_),
                                      "fp8 act quantize"));
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
        Bf16GemmConfig cfg{rows >= 2u ? Bf16GemmConfigId::ExactRowsDot2
                                      : Bf16GemmConfigId::ExactRows,
                           static_cast<std::uint8_t>(rows)};
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
        // vendored rope computes f32 sincos and takes f32 input; expand bf16 first
        void* xf = scratch(0, out_n * sizeof(float));
        void* o = alloc(false);
        if (!xf || !o) return Status::insufficient_memory("arena exhausted", __FILE__, __LINE__);
        SHELF_RETURN_IF_ERROR(hipok(
            launch_bf16_to_f32(static_cast<const bf16_t*>(x->dev),
                               static_cast<float*>(xf), u32(out_n), stream_),
            "rope cvt"));
        const std::size_t D = p.output.shape.dim(p.output.shape.rank() - 1);
        const std::size_t rows = out_n / D;               // P * heads_per_token
        const std::size_t per_tok = rows / step_rows_;    // q or kv heads
        if (rows % step_rows_) return Status::invalid_state("rope: row count not multiple of batch", __FILE__, __LINE__);
        const std::size_t rd = static_cast<std::size_t>(p.attr.rotary_dim);
        // rope position of row r = base_pos + r/per_tok (all heads of one token share pos)
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
        SHELF_RETURN_IF_ERROR(hipok(launch_gdn_recurrence_f32(rec, stream_), "recurrence"));
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
            norm = scratch(0, out_n * sizeof(float));
            SHELF_RETURN_IF_ERROR(scratch_ok(norm));
            SHELF_RETURN_IF_ERROR(hipok(launch_rmsnorm(
                x->dev, RmsNormDataType::F32, wt(0), RmsNormDataType::BF16,
                RmsNormWeightLayout::PER_FEATURE, RmsNormWeightMode::DIRECT, norm,
                RmsNormDataType::F32, u32(rows), u32(F), u32(F), u32(F), u32(F),
                p.attr.epsilon, stream_), "gdn norm"));
            void* normb = scratch(1, out_n * sizeof(bf16_t));
            SHELF_RETURN_IF_ERROR(scratch_ok(normb));
            SHELF_RETURN_IF_ERROR(hipok(launch_f32_to_bf16(
                static_cast<const float*>(norm), static_cast<bf16_t*>(normb), u32(out_n),
                stream_), "gdn norm cvt"));
            norm = normb;
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
        SHELF_RETURN_IF_ERROR(hipok(launch_attention_paged_bf16(
            at, static_cast<const bf16_t*>(st.k_pool), static_cast<const bf16_t*>(st.v_pool),
            stream_), "attn_paged"));
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
