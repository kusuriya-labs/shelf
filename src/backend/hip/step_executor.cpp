// shelf backend/hip — GPU step executor implementation.
//
// Numerics policy: bf16 activations with fp32-accumulating kernels (the
// vendored PhaseShift family). The CPU reference executor (f32) remains
// the numerical golden; this executor is the speed path.
// FP8 経路（approx 側）: 重みは E4M3 コードのまま断片順で載せ、活性化は
// ステップごとに per-token スケールで量子化する。dequant はカーネル内
// （行列命令が E4M3 を直接読む）。契約は src/quant/fp8_perchannel.hpp。
#include <shelf/backend/hip/step_executor.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <thread>

#include <shelf/backend/hip/kernels/elementwise/aux_ops.h>
#include <shelf/backend/hip/kernels/elementwise/activation_quantize_fp8.h>
#include <shelf/backend/hip/kernels/elementwise/activation_quantize_mxfp4.h>
#include <shelf/backend/hip/kernels/elementwise/embedding.h>
#include <shelf/backend/hip/kernels/elementwise/elementwise.h>
#include <shelf/backend/hip/kernels/gdn/gdn_chunk.h>
#include <shelf/backend/hip/kernels/gdn/gdn_conv_silu.h>
#include <shelf/backend/hip/kernels/gdn/gdn_qk_prepare.h>
#include <shelf/backend/hip/kernels/gdn/gdn_recurrence.h>
#include <shelf/backend/hip/kernels/gemm/linear_bf16.h>
#include <shelf/backend/hip/kernels/gemm/linear_fp8.h>
#include <shelf/backend/hip/kernels/gemm/linear_mxfp4.h>
#include <shelf/backend/hip/kernels/attention/kv_append.h>
#include <shelf/backend/hip/kernels/attention/paged_attention.h>
#include <shelf/backend/hip/kernels/attention/rope.h>
#include <shelf/backend/hip/kernels/elementwise/rmsnorm.h>
#include <shelf/foundation/numeric.hpp>
#include <shelf/quant/fp8_perchannel.hpp>
#include <shelf/quant/fp8_finegrained.hpp>
#include <shelf/quant/mxfp4.hpp>

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

// ---- T17: CPU 側のエキスパート計算（pinned host の重み）----
//
// GPU と同じ値になることを優先した素朴な実装。量子化重みは復号しながら内積を取る
// （重みを f32 に展開して持つとメモリが 4 倍になるため、その場で E2M1 を引く）。
inline float bf16_host_to_float(bf16_t v) {
    const std::uint32_t bits = static_cast<std::uint32_t>(v.data) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// bf16 への丸め（GPU の swiglu / bf16 GEMM と同じ bit にする）。
inline float bf16_round_host(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    u = (u + 0x8000u + ((u >> 16) & 1u)) >> 16;
    const std::uint32_t bits = u << 16;
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

inline float e2m1_lut(std::uint8_t nib) {
    static const float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float m = kMag[nib & 7u];
    return (nib & 8u) != 0u ? -m : m;
}

inline float e8m0_lut(std::uint8_t b) {
    return std::ldexp(1.0f, static_cast<int>(b) - 127);
}

float dot_bf16_row(const bf16_t* w, const float* x, std::size_t k) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < k; ++i) acc += bf16_host_to_float(w[i]) * x[i];
    return acc;
}

// 断片順の MXFP4 1 行（出力 1 個）と f32 ベクトルの内積。
float dot_mxfp4_row(const std::uint8_t* codes, const std::uint8_t* scales, std::uint32_t row,
                    const float* x, std::uint32_t k, std::uint32_t k_padded) {
    const std::uint32_t nb = k_padded / shelf::quant::mxfp4::kBlockK;
    const std::size_t tile_base =
        static_cast<std::size_t>(row / shelf::quant::mxfp4::kTileRows) * (nb * 256u) +
        static_cast<std::size_t>(row % shelf::quant::mxfp4::kTileRows) *
            shelf::quant::mxfp4::kLaneBytes;
    const std::uint8_t* sc = scales + static_cast<std::size_t>(row) * nb;
    float acc = 0.0f;
    for (std::uint32_t kb = 0; kb < nb; ++kb) {
        const float s = e8m0_lut(sc[kb]);
        const std::uint8_t* bp = codes + tile_base + static_cast<std::size_t>(kb) * 256u;
        const float* xb = x + static_cast<std::size_t>(kb) * 32u;
        for (std::uint32_t j = 0; j < 16u; ++j) {
            if (kb * 32u + j >= k) break;
            acc += xb[j] * e2m1_lut(bp[j] & 0x0Fu) * s;
            if (kb * 32u + 16u + j < k)
                acc += xb[16u + j] * e2m1_lut(bp[j] >> 4) * s;
        }
    }
    return acc;
}


// GPU の活性化量子化と同じ契約で f32 ベクトルを作り直す（CPU 分担用）。
// W4A4 は E2M1 + E8M0（32 要素ごと）、W4A8 は E4M3 + 動的スケール。
void quantize_act_mxfp4_host(const float* x, std::uint32_t k, std::vector<float>& out) {
    using namespace shelf::quant::mxfp4;
    const std::uint32_t kp = pad_k(k);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(kp) / 2u);
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(kp) / kBlockK);
    if (!quantize_row(x, k, kp, codes.data(), scales.data()).ok()) {
        out.assign(x, x + k);
        return;
    }
    out.resize(k);
    for (std::uint32_t i = 0; i < k; ++i) {
        const std::uint8_t byte = codes[i / 2u];
        const std::uint8_t nib = (i % 2u == 0u) ? (byte & 0x0Fu) : (byte >> 4);
        out[i] = e2m1_lut(nib) * e8m0_lut(scales[i / kBlockK]);
    }
}

void quantize_act_fp8_host(const float* x, std::uint32_t k, std::uint32_t group,
                           std::vector<float>& out) {
    using namespace shelf::quant::fp8;
    const std::uint32_t g = group == 0u ? k : group;
    out.resize(k);
    for (std::uint32_t b = 0; b < k; b += g) {
        const std::uint32_t e = std::min(k, b + g);
        float amax = 0.0f;
        for (std::uint32_t i = b; i < e; ++i) amax = std::max(amax, std::fabs(x[i]));
        const float s = amax_scale(amax);
        for (std::uint32_t i = b; i < e; ++i) {
            const std::uint8_t code = e4m3_encode(s > 0.0f ? x[i] / s : 0.0f);
            out[i] = e4m3_decode(code) * s;
        }
    }
}

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

// ---- device 確保の計上 ------------------------------------------------------
Status HipStepExecutor::dev_reserve(void** out, std::size_t bytes, const char* what) {
    if (out == nullptr || bytes == 0u)
        return Status::invalid_argument("dev_reserve: null/zero", __FILE__, __LINE__);
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

bool HipStepExecutor::is_pinned_dev(void* p) const noexcept {
    for (const auto& [host, dev] : pinned_allocs_) {
        (void)host;
        if (dev == p) return true;
    }
    return false;
}

void HipStepExecutor::release_weight_ptr(void* p) noexcept {
    if (p == nullptr || is_pinned_dev(p)) return;
    dev_release(p);
}

Status HipStepExecutor::alloc_weight_payload(void** dev_out, std::size_t bytes,
                                             const void* src, const char* what, TensorId id,
                                             int kind) {
    if (dev_out == nullptr || bytes == 0u)
        return Status::invalid_argument("alloc_weight_payload: null/zero", __FILE__, __LINE__);
    // pinned host に確保する（明示指定・縮退の両方から使う）。
    const auto alloc_pinned = [&]() -> Status {
        void* host = nullptr;
        hipError_t e = hipHostMalloc(&host, bytes, hipHostMallocMapped);
        if (e != hipSuccess)
            return Status::from(Status::Code::device_error,
                                std::string(what) + ": hipHostMalloc(mapped): " +
                                    hipGetErrorString(e),
                                __FILE__, __LINE__);
        void* dev = nullptr;
        e = hipHostGetDevicePointer(&dev, host, 0);
        if (e != hipSuccess) {
            (void)hipHostFree(host);
            return Status::from(Status::Code::device_error,
                                std::string(what) + ": hipHostGetDevicePointer: " +
                                    hipGetErrorString(e),
                                __FILE__, __LINE__);
        }
        if (src != nullptr) std::memcpy(host, src, bytes);
        *dev_out = dev;
        pinned_allocs_.emplace_back(host, dev);
        host_weight_index_[id.value()] = host_weights_.size();
        host_weights_.push_back(HostWeight{id, kind, host, bytes, false, {}, 0u, 0u});
        host_weight_bytes_ += bytes;
        return Status::make_ok();
    };
    if (pending_weight_pinned_ || fail_dev_alloc_ > 0) {
        if (fail_dev_alloc_ > 0) --fail_dev_alloc_;
        return alloc_pinned();
    }
    const Status ds = dev_reserve(dev_out, bytes, what);
    if (!ds.ok()) {
        // T17 の縮退: VRAM が足りない / 転送失敗でも止めない。pinned host に
        // 置いて CPU（MoE）または mapped host 経由の GPU で継続する。
        if (!allow_host_fallback_) return ds;
        return alloc_pinned();
    }
    if (src != nullptr)
        SHELF_RETURN_IF_ERROR(
            hipok(hipMemcpy(*dev_out, src, bytes, hipMemcpyHostToDevice), what));
    return Status::make_ok();
}

Status HipStepExecutor::alloc_pinned_aux(void** dev_out, std::size_t bytes, const void* src,
                                         const char* what) {
    if (dev_out == nullptr || bytes == 0u)
        return Status::invalid_argument("alloc_pinned_aux: null/zero", __FILE__, __LINE__);
    void* host = nullptr;
    hipError_t e = hipHostMalloc(&host, bytes, hipHostMallocMapped);
    if (e != hipSuccess)
        return Status::from(Status::Code::device_error,
                            std::string(what) + ": hipHostMalloc(mapped): " +
                                hipGetErrorString(e),
                            __FILE__, __LINE__);
    void* dev = nullptr;
    e = hipHostGetDevicePointer(&dev, host, 0);
    if (e != hipSuccess) {
        (void)hipHostFree(host);
        return Status::from(Status::Code::device_error,
                            std::string(what) + ": hipHostGetDevicePointer: " +
                                hipGetErrorString(e),
                            __FILE__, __LINE__);
    }
    if (src != nullptr) std::memcpy(host, src, bytes);
    *dev_out = dev;
    pinned_allocs_.emplace_back(host, dev);
    host_weight_bytes_ += bytes;
    return Status::make_ok();
}

Status HipStepExecutor::promote_host_weights() {
    (void)hipSetDevice(opts_.device);
    for (auto& hw : host_weights_) {
        if (hw.promoted) continue;
        void* dev = nullptr;
        SHELF_RETURN_IF_ERROR(dev_reserve(&dev, hw.bytes, "promote weight"));
        SHELF_RETURN_IF_ERROR(
            hipok(hipMemcpyAsync(dev, hw.host, hw.bytes, hipMemcpyHostToDevice, stream_),
                  "promote weight copy"));
        SHELF_RETURN_IF_ERROR(hipok(hipStreamSynchronize(stream_), "promote sync"));
        switch (hw.kind) {
            case 0:
                weight_dev_[hw.id.value()] = dev;
                break;
            case 1: {
                const auto it = weight_fp8_.find(hw.id.value());
                if (it == weight_fp8_.end())
                    return Status::invalid_state("promote: fp8 weight not found", __FILE__,
                                                 __LINE__);
                it->second.codes = static_cast<std::uint8_t*>(dev);
                break;
            }
            case 2: {
                const auto it = weight_mxfp4_.find(hw.id.value());
                if (it == weight_mxfp4_.end())
                    return Status::invalid_state("promote: mxfp4 weight not found", __FILE__,
                                                 __LINE__);
                it->second.codes = static_cast<std::uint8_t*>(dev);
                break;
            }
            default:
                return Status::invalid_state("promote: unknown kind", __FILE__, __LINE__);
        }
        hw.promoted = true;
        promoted_weight_bytes_ += hw.bytes;
    }
    return Status::make_ok();
}

Status HipStepExecutor::alloc_state_pool(void** dev_out, void** host_out, std::size_t bytes,
                                         const char* what) {
    if (dev_out == nullptr || host_out == nullptr || bytes == 0u)
        return Status::invalid_argument("alloc_state_pool: null/zero", __FILE__, __LINE__);
    if (!opts_.kv_pinned_host) {
        *host_out = nullptr;
        return dev_reserve(dev_out, bytes, what);
    }
    void* host = nullptr;
    hipError_t e = hipHostMalloc(&host, bytes, hipHostMallocMapped);
    if (e != hipSuccess)
        return Status::from(Status::Code::device_error,
                            std::string(what) + ": hipHostMalloc(mapped): " +
                                hipGetErrorString(e),
                            __FILE__, __LINE__);
    void* dev = nullptr;
    e = hipHostGetDevicePointer(&dev, host, 0);
    if (e != hipSuccess) {
        (void)hipHostFree(host);
        return Status::from(Status::Code::device_error,
                            std::string(what) + ": hipHostGetDevicePointer: " +
                                hipGetErrorString(e),
                            __FILE__, __LINE__);
    }
    *dev_out = dev;
    *host_out = host;
    pinned_allocs_.emplace_back(host, dev);
    host_kv_bytes_ += bytes;
    return Status::make_ok();
}

HipStepExecutor::HipStepExecutor(HipStepExecutorOptions opts) : opts_(opts) {
    max_sequences_ = std::max<std::size_t>(opts_.max_sequences, 1u);
    allow_host_fallback_ = opts_.allow_host_fallback;
    // テスト用の故障注入: 先頭 N 回の device 確保を失敗させる（縮退の検証）。
    if (const char* e = std::getenv("SHELF_TEST_FAIL_DEV_ALLOC")) {
        fail_dev_alloc_ = static_cast<int>(std::strtol(e, nullptr, 10));
    }
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
    const hipError_t e6 = malloc_dev(&req_dev_, std::max<std::size_t>(max_sequences_, 1u));
    const hipError_t e8 = malloc_dev(&rowslot_dev_, kMaxStepRows);
    const hipError_t e9 = malloc_dev(&slot_window_dev_, max_sequences_);
    const hipError_t e7 = malloc_dev(&inv_freq_dev_,
                                      std::max<std::size_t>(2, opts_.rope_rotary_dim / 2));
    (void)e0; (void)e1; (void)e2; (void)e3; (void)e4; (void)e5; (void)e6; (void)e7; (void)e8;
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
    charge(req_dev_, max_sequences_ * sizeof(hipk::GdnRequestDescriptor));
    charge(rowslot_dev_, kMaxStepRows * sizeof(std::uint32_t));
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
        if (loc.dev) release_weight_ptr(loc.dev);
        if (loc.k_pool) release_weight_ptr(loc.k_pool);
        if (loc.v_pool) release_weight_ptr(loc.v_pool);
        if (loc.table) release_weight_ptr(loc.table);
    }
    for (auto& [id, p] : weight_dev_) release_weight_ptr(p);
    for (auto& [id, w] : weight_fp8_) {
        release_weight_ptr(w.codes);
        release_weight_ptr(w.scales);
    }
    for (auto& [id, w] : weight_mxfp4_) {
        release_weight_ptr(w.codes);
        release_weight_ptr(w.scales);
    }
    // pinned（mapped host）は device 側の hipFree では解放できない。
    for (auto& [host, dev] : pinned_allocs_) {
        (void)dev;
        if (host) (void)hipHostFree(host);
    }
    pinned_allocs_.clear();
    dev_release(fp8_codes_dev_);
    dev_release(fp8_scales_dev_);
    dev_release(mxfp4_act_codes_dev_);
    dev_release(mxfp4_act_scales_dev_);
    dev_release(moe_scratch_dev_);
    dev_release(moe_act_codes_dev_);
    dev_release(moe_act_scales_dev_);
    dev_release(arena_dev_);
    dev_release(token_dev_);
    dev_release(posrep_dev_);
    dev_release(rowpos_dev_);
    dev_release(slot_zero_dev_);
    dev_release(errword_dev_);
    dev_release(req_dev_);
    dev_release(rowslot_dev_);
    dev_release(slot_window_dev_);
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
    SHELF_RETURN_IF_ERROR(alloc_weight_payload(&dev, b.size() * sizeof(bf16_t), b.data(),
                                               "upload weight", id, 0));
    weight_dev_[id.value()] = dev;
    return Status::make_ok();
}

Status HipStepExecutor::upload_weight_bf16(TensorId id, const bf16_t* data, std::size_t n) {
    (void)hipSetDevice(opts_.device);
    if (weight_dev_.count(id.value()))
        return Status::invalid_state("weight uploaded twice", __FILE__, __LINE__);
    void* dev = nullptr;
    SHELF_RETURN_IF_ERROR(alloc_weight_payload(&dev, n * sizeof(bf16_t), data,
                                               "upload weight bf16", id, 0));
    weight_dev_[id.value()] = dev;
    return Status::make_ok();
}

Status HipStepExecutor::upload_weight_mxfp4(TensorId id, const std::uint8_t* codes,
                                            std::size_t n_codes, const std::uint8_t* scales,
                                            std::size_t n_scales, std::size_t features,
                                            std::uint32_t k_padded) {
    (void)hipSetDevice(opts_.device);
    using namespace shelf::quant::mxfp4;
    if (weight_mxfp4_.count(id.value()) || weight_fp8_.count(id.value()) ||
        weight_dev_.count(id.value()))
        return Status::invalid_state("weight uploaded twice", __FILE__, __LINE__);
    if (features == 0u || (k_padded % kBlockK) != 0u)
        return Status::invalid_argument("upload_weight_mxfp4: 形の指定不正", __FILE__, __LINE__);
    const std::size_t want_codes = tiled_bytes(u32(features), k_padded);
    const std::size_t want_scales =
        static_cast<std::size_t>(features) * (k_padded / kBlockK);
    if (n_codes != want_codes)
        return Status::invalid_argument(
            "upload_weight_mxfp4: コード数がレイアウトと不一致（" + std::to_string(n_codes) +
            " != " + std::to_string(want_codes) + "）", __FILE__, __LINE__);
    if (n_scales != want_scales)
        return Status::invalid_argument(
            "upload_weight_mxfp4: スケール数が不一致（" + std::to_string(n_scales) + " != " +
            std::to_string(want_scales) + "）", __FILE__, __LINE__);
    const std::size_t pinned_before = host_weights_.size();
    std::uint8_t* cdev = nullptr;
    std::uint8_t* sdev = nullptr;
    SHELF_RETURN_IF_ERROR(alloc_weight_payload(reinterpret_cast<void**>(&cdev), n_codes,
                                               codes, "upload mxfp4 codes", id, 2));
    // スケールも device に置くが、確保に失敗したら pinned host へ落とす（縮退）。
    {
        Status ss = Status::make_ok();
        if (fail_dev_alloc_ > 0) {
            --fail_dev_alloc_;
            ss = Status::from(Status::Code::device_error, "mxfp4 scales: 故障注入",
                              __FILE__, __LINE__);
        } else {
            ss = dev_reserve(reinterpret_cast<void**>(&sdev), n_scales, "malloc mxfp4 scales");
        }
        if (!ss.ok()) {
            if (!allow_host_fallback_) return ss;
            SHELF_RETURN_IF_ERROR(alloc_pinned_aux(reinterpret_cast<void**>(&sdev), n_scales,
                                                   scales, "pinned mxfp4 scales"));
        } else {
            SHELF_RETURN_IF_ERROR(hipok(
                hipMemcpy(sdev, scales, n_scales, hipMemcpyHostToDevice),
                "upload mxfp4 scales"));
        }
    }
    weight_mxfp4_[id.value()] = Mxfp4Weight{cdev, sdev, features, k_padded};
    // pinned host に置いた場合、CPU 計算用にスケールの正本も持つ
    // （コードは `host_weights_.back().host` が正本。scales だけ device に
    //  行ってしまうため、ここで複製する）。
    if (host_weights_.size() != pinned_before) {
        HostWeight& hw = host_weights_.back();
        hw.host_scales.assign(scales, scales + n_scales);
        hw.features = features;
        hw.k_padded = k_padded;
    }
    return Status::make_ok();
}

Status HipStepExecutor::upload_weight_fp8(TensorId id, const std::uint8_t* codes,
                                          std::size_t n_codes, const float* scales,
                                          std::size_t features, std::uint32_t k_padded,
                                          bool weight_block128) {
    (void)hipSetDevice(opts_.device);
    using namespace shelf::quant::fp8;
using namespace shelf::quant::fp8::fine;
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
    SHELF_RETURN_IF_ERROR(alloc_weight_payload(reinterpret_cast<void**>(&cdev), n_codes,
                                               codes, "upload weight fp8 codes", id, 1));
    {
        Status ss = Status::make_ok();
        if (fail_dev_alloc_ > 0) {
            --fail_dev_alloc_;
            ss = Status::from(Status::Code::device_error, "fp8 scales: 故障注入", __FILE__,
                              __LINE__);
        } else {
            ss = dev_alloc(&sdev, want_scales, "malloc weight fp8 scales");
        }
        if (!ss.ok()) {
            if (!allow_host_fallback_) return ss;
            SHELF_RETURN_IF_ERROR(alloc_pinned_aux(
                reinterpret_cast<void**>(&sdev), want_scales * sizeof(float), scales,
                "pinned fp8 scales"));
        } else {
            SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(sdev, scales, want_scales * sizeof(float),
                                                  hipMemcpyHostToDevice),
                                        "upload weight fp8 scales"));
        }
    }
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

Status HipStepExecutor::ensure_mxfp4_scratch(std::size_t code_bytes, std::size_t scale_bytes) {
    if (code_bytes > mxfp4_act_codes_cap_) {
        if (mxfp4_act_codes_dev_) dev_release(mxfp4_act_codes_dev_);
        mxfp4_act_codes_dev_ = nullptr;
        SHELF_RETURN_IF_ERROR(
            dev_alloc(&mxfp4_act_codes_dev_, code_bytes, "malloc mxfp4 act codes"));
        mxfp4_act_codes_cap_ = code_bytes;
    }
    if (scale_bytes > mxfp4_act_scales_cap_) {
        if (mxfp4_act_scales_dev_) dev_release(mxfp4_act_scales_dev_);
        mxfp4_act_scales_dev_ = nullptr;
        SHELF_RETURN_IF_ERROR(
            dev_alloc(&mxfp4_act_scales_dev_, scale_bytes, "malloc mxfp4 act scales"));
        mxfp4_act_scales_cap_ = scale_bytes;
    }
    return Status::make_ok();
}

Status HipStepExecutor::ensure_moe_scratch(std::size_t bytes) {
    if (bytes <= moe_scratch_cap_) return Status::make_ok();
    if (moe_scratch_dev_) dev_release(moe_scratch_dev_);
    moe_scratch_dev_ = nullptr;
    SHELF_RETURN_IF_ERROR(dev_alloc(&moe_scratch_dev_, bytes, "malloc moe scratch"));
    moe_scratch_cap_ = bytes;
    return Status::make_ok();
}

Status HipStepExecutor::ensure_moe_act_scratch(std::size_t code_bytes,
                                               std::size_t scale_bytes) {
    if (code_bytes > moe_act_codes_cap_) {
        if (moe_act_codes_dev_) dev_release(moe_act_codes_dev_);
        moe_act_codes_dev_ = nullptr;
        SHELF_RETURN_IF_ERROR(
            dev_alloc(&moe_act_codes_dev_, code_bytes, "malloc moe act codes"));
        moe_act_codes_cap_ = code_bytes;
    }
    if (scale_bytes > moe_act_scales_cap_) {
        if (moe_act_scales_dev_) dev_release(moe_act_scales_dev_);
        moe_act_scales_dev_ = nullptr;
        SHELF_RETURN_IF_ERROR(
            dev_alloc(&moe_act_scales_dev_, scale_bytes, "malloc moe act scales"));
        moe_act_scales_cap_ = scale_bytes;
    }
    return Status::make_ok();
}

// MoE の 1 行（1 トークン）を計算する。ルーティングだけホストで行い、
// エキスパートの行列積はデバイスのカーネルに任せる（正しさ優先の実装。
// GPU 側で分担を決める最適化は後段。判定は「この呼び出しの先頭」で固定する）。
Status HipStepExecutor::run_moe_row(const ir::Primitive& p, const bf16_t* x_row,
                                    bf16_t* out_row, std::size_t H, std::size_t E,
                                    std::size_t I, std::size_t Is, bool shared) {
    const std::size_t W = std::max(I, Is);
    // ---- 重みの形式（エキスパートは全員同じ形式。ルーターは bf16）----
    const auto dev_of = [&](std::size_t i) -> void* {
        const auto it = weight_dev_.find(p.weights[i].value());
        return it == weight_dev_.end() ? nullptr : it->second;
    };
    const auto mx_of = [&](std::size_t i) -> const Mxfp4Weight* {
        const auto it = weight_mxfp4_.find(p.weights[i].value());
        return it == weight_mxfp4_.end() ? nullptr : &it->second;
    };
    if (dev_of(0) == nullptr)
        return Status::unsupported("moe_mlp: router は bf16 のみ対応", __FILE__, __LINE__);
    const bool expert_mx = mx_of(1) != nullptr;
    if (!expert_mx && dev_of(1) == nullptr)
        return Status::unsupported("moe_mlp: エキスパート重みが bf16/MXFP4 以外",
                                   __FILE__, __LINE__);

    // ---- 一時領域 ----
    std::size_t off = 0;
    const auto take = [&off](std::size_t bytes) {
        off = (off + 15u) & ~std::size_t(15u);
        const std::size_t at = off;
        off += bytes;
        return at;
    };
    const std::size_t off_gate = take(W * sizeof(bf16_t));
    const std::size_t off_up = take(W * sizeof(bf16_t));
    const std::size_t off_mid = take(W * sizeof(bf16_t));
    const std::size_t off_eout = take(H * sizeof(bf16_t));
    const std::size_t off_acc = take(H * sizeof(float));
    const std::size_t off_logits = take(E * sizeof(float));
    // CPU で計算した寄与を H2D するための f32 1 行分。
    const std::size_t off_cpu = take(H * sizeof(float));
    const std::size_t total = take(16u);
    SHELF_RETURN_IF_ERROR(ensure_moe_scratch(total));
    auto* base = static_cast<std::uint8_t*>(moe_scratch_dev_);
    auto* gate_buf = reinterpret_cast<bf16_t*>(base + off_gate);
    auto* up_buf = reinterpret_cast<bf16_t*>(base + off_up);
    auto* mid_buf = reinterpret_cast<bf16_t*>(base + off_mid);
    auto* eout_buf = reinterpret_cast<bf16_t*>(base + off_eout);
    auto* acc_buf = reinterpret_cast<float*>(base + off_acc);
    auto* logits_dev = reinterpret_cast<float*>(base + off_logits);
    auto* cpu_acc_dev = reinterpret_cast<float*>(base + off_cpu);

    // ---- ルーター（1 行分の logits をデバイスで作ってホストで top-k）----
    SHELF_RETURN_IF_ERROR(hipok(
        launch_moe_router_logit(x_row, static_cast<const bf16_t*>(dev_of(0)), logits_dev,
                                u32(E), u32(H), stream_),
        "moe router logits"));
    std::vector<float> logits(E);
    std::vector<bf16_t> x_host(H);   // CPU 分担の入力（GPU と同じ bf16 の生ビット）
    SHELF_RETURN_IF_ERROR(hipok(
        hipMemcpyAsync(logits.data(), logits_dev, E * sizeof(float), hipMemcpyDeviceToHost,
                       stream_),
        "moe router readback"));
    SHELF_RETURN_IF_ERROR(hipok(
        hipMemcpyAsync(x_host.data(), x_row, H * sizeof(bf16_t), hipMemcpyDeviceToHost,
                       stream_),
        "moe cpu input readback"));
    SHELF_RETURN_IF_ERROR(hipok(hipStreamSynchronize(stream_), "moe router sync"));

    const std::size_t k = static_cast<std::size_t>(p.attr.top_k);
    if (k == 0u || k > E)
        return Status::invalid_state("moe_mlp: top_k out of range", __FILE__, __LINE__);
    // softmax（f32）→ top-k（同値は小さい index）→ 選択分を再正規化。
    float mx = -INFINITY;
    for (float v : logits) mx = std::max(mx, v);
    float sum = 0.0f;
    for (auto& v : logits) {
        v = std::exp(v - mx);
        sum += v;
    }
    for (auto& v : logits) v /= sum;
    std::vector<char> used(E, 0);
    std::vector<std::size_t> sel(k, 0);
    std::vector<float> sel_w(k, 0.0f);
    float keep = 0.0f;
    for (std::size_t j = 0; j < k; ++j) {
        std::size_t best = E;
        for (std::size_t e = 0; e < E; ++e) {
            if (used[e]) continue;
            if (best == E || logits[e] > logits[best]) best = e;
        }
        used[best] = 1;
        sel[j] = best;
        sel_w[j] = logits[best];
        keep += logits[best];
    }
    if (keep <= 0.0f)
        return Status::invalid_state("moe_mlp: top-k weights are zero", __FILE__, __LINE__);
    for (auto& v : sel_w) v /= keep;

    SHELF_RETURN_IF_ERROR(hipok(
        hipMemsetAsync(acc_buf, 0, H * sizeof(float), stream_), "moe acc zero"));

    // ---- MXFP4 の活性化 scratch（1 行分）。x は全エキスパートで共通。----
    const bool w4a4 = opts_.mxfp4_activation;
    std::uint32_t x_kp = 0;
    std::uint32_t x_group = 0;
    if (expert_mx) {
        using namespace shelf::quant::mxfp4;
        x_kp = pad_k(u32(H));
        if (x_kp != u32(H))
            return Status::unsupported("moe_mlp: H が 32 の倍数でない", __FILE__, __LINE__);
        if (w4a4) {
            SHELF_RETURN_IF_ERROR(
                ensure_mxfp4_scratch(tiled_bytes(1u, x_kp), scale_bytes(1u, x_kp)));
            SHELF_RETURN_IF_ERROR(hipok(
                launch_activation_quantize_mxfp4(x_row, mxfp4_act_codes_dev_,
                                                 mxfp4_act_scales_dev_, 1u, u32(H), x_kp,
                                                 u32(H), stream_),
                "moe mxfp4 act quantize"));
        } else {
            x_group = (opts_.fp8_act_group != 0u &&
                       (x_kp % shelf::quant::fp8::fine::kActGroupK) == 0u)
                          ? shelf::quant::fp8::fine::kActGroupK
                          : 0u;
            SHELF_RETURN_IF_ERROR(ensure_fp8_scratch(
                shelf::quant::fp8::tiled_bytes(1u, x_kp),
                shelf::quant::fp8::fine::activation_scale_count(1u, x_kp, x_group)));
            SHELF_RETURN_IF_ERROR(hipok(
                launch_activation_quantize_fp8(x_row, fp8_codes_dev_, fp8_scales_dev_, 1u,
                                               u32(H), x_kp, u32(H), x_group, stream_),
                "moe mxfp4 act quantize (fp8)"));
        }
    }

    // エキスパート 1 個の MLP（gate/up → swiglu → down → 累積）。
    const auto run_expert = [&](std::size_t w_idx, std::size_t width, float weight) -> Status {
        const Mxfp4Weight* mx_g = mx_of(w_idx);
        if (mx_g != nullptr) {
            using namespace shelf::quant::mxfp4;
            const std::uint32_t kp = pad_k(u32(H));
            const Mxfp4Weight* mx_u = mx_of(w_idx + 1);
            if (mx_u == nullptr)
                return Status::invalid_state("moe_mlp: expert weight kind mismatch",
                                             __FILE__, __LINE__);
            Mxfp4GemmScaleArgs sa;
            sa.weight_scale_stride = kp / kBlockK;
            if (w4a4) {
                sa.act_group = 0u;
                sa.activation_scale_stride = kp / kBlockK;
                SHELF_RETURN_IF_ERROR(hipok(
                    launch_gemm_mxfp4_w4a4_decode1(mx_g->codes, mx_g->scales,
                                                   mxfp4_act_codes_dev_,
                                                   mxfp4_act_scales_dev_, gate_buf,
                                                   Mxfp4GemmOutputDType::BF16, 1u, u32(width),
                                                   kp, u32(width), sa, stream_),
                    "moe expert gate (w4a4)"));
                SHELF_RETURN_IF_ERROR(hipok(
                    launch_gemm_mxfp4_w4a4_decode1(mx_u->codes, mx_u->scales,
                                                   mxfp4_act_codes_dev_,
                                                   mxfp4_act_scales_dev_, up_buf,
                                                   Mxfp4GemmOutputDType::BF16, 1u, u32(width),
                                                   kp, u32(width), sa, stream_),
                    "moe expert up (w4a4)"));
            } else {
                sa.act_group = x_group;
                sa.activation_scale_stride =
                    x_group == shelf::quant::fp8::fine::kActGroupK ? kp / x_group : 1u;
                SHELF_RETURN_IF_ERROR(hipok(
                    launch_gemm_mxfp4_w4a8_decode1(mx_g->codes, mx_g->scales, fp8_codes_dev_,
                                                   fp8_scales_dev_, gate_buf,
                                                   Mxfp4GemmOutputDType::BF16, 1u, u32(width),
                                                   kp, u32(width), sa, stream_),
                    "moe expert gate (w4a8)"));
                SHELF_RETURN_IF_ERROR(hipok(
                    launch_gemm_mxfp4_w4a8_decode1(mx_u->codes, mx_u->scales, fp8_codes_dev_,
                                                   fp8_scales_dev_, up_buf,
                                                   Mxfp4GemmOutputDType::BF16, 1u, u32(width),
                                                   kp, u32(width), sa, stream_),
                    "moe expert up (w4a8)"));
            }
        } else {
            Bf16GemmConfig cfg;
            cfg.id = Bf16GemmConfigId::ExactRows;
            cfg.exact_rows = 1u;
            SHELF_RETURN_IF_ERROR(hipok(
                launch_gemm_bf16(cfg, static_cast<const bf16_t*>(dev_of(w_idx)), x_row,
                                 gate_buf, Bf16GemmOutputDType::BF16, 1u, u32(width), u32(H),
                                 u32(H), u32(width), stream_),
                "moe expert gate"));
            SHELF_RETURN_IF_ERROR(hipok(
                launch_gemm_bf16(cfg, static_cast<const bf16_t*>(dev_of(w_idx + 1)), x_row,
                                 up_buf, Bf16GemmOutputDType::BF16, 1u, u32(width), u32(H),
                                 u32(H), u32(width), stream_),
                "moe expert up"));
        }
        SHELF_RETURN_IF_ERROR(hipok(
            launch_swiglu_bf16(gate_buf, u32(width), up_buf, u32(width), mid_buf,
                               u32(width), 1u, u32(width), stream_),
            "moe expert swiglu"));
        // down: mid（bf16, 幅 width）→ [H]
        const Mxfp4Weight* mx_d = mx_of(w_idx + 2);
        if (mx_d != nullptr) {
            using namespace shelf::quant::mxfp4;
            const std::uint32_t kp = pad_k(u32(width));
            if (kp != u32(width))
                return Status::unsupported("moe_mlp: down の K が 32 の倍数でない", __FILE__,
                                           __LINE__);
            Mxfp4GemmScaleArgs sa;
            sa.weight_scale_stride = kp / kBlockK;
            if (w4a4) {
                SHELF_RETURN_IF_ERROR(
                    ensure_moe_act_scratch(tiled_bytes(1u, kp), scale_bytes(1u, kp)));
                SHELF_RETURN_IF_ERROR(hipok(
                    launch_activation_quantize_mxfp4(mid_buf, moe_act_codes_dev_,
                                                     moe_act_scales_dev_, 1u, u32(width), kp,
                                                     u32(width), stream_),
                    "moe expert down act quantize (w4a4)"));
                sa.act_group = 0u;
                sa.activation_scale_stride = kp / kBlockK;
                SHELF_RETURN_IF_ERROR(hipok(
                    launch_gemm_mxfp4_w4a4_decode1(mx_d->codes, mx_d->scales,
                                                   moe_act_codes_dev_, moe_act_scales_dev_,
                                                   eout_buf, Mxfp4GemmOutputDType::BF16, 1u,
                                                   u32(H), kp, u32(H), sa, stream_),
                    "moe expert down (w4a4)"));
            } else {
                const std::uint32_t group =
                    (opts_.fp8_act_group != 0u &&
                     (kp % shelf::quant::fp8::fine::kActGroupK) == 0u)
                        ? shelf::quant::fp8::fine::kActGroupK
                        : 0u;
                SHELF_RETURN_IF_ERROR(ensure_moe_act_scratch(
                    shelf::quant::fp8::tiled_bytes(1u, kp),
                    shelf::quant::fp8::fine::activation_scale_count(1u, kp, group) *
                        sizeof(float)));
                SHELF_RETURN_IF_ERROR(hipok(
                    launch_activation_quantize_fp8(mid_buf, moe_act_codes_dev_,
                                                   reinterpret_cast<float*>(moe_act_scales_dev_),
                                                   1u, u32(width), kp, u32(width), group,
                                                   stream_),
                    "moe expert down act quantize"));
                sa.act_group = group;
                sa.activation_scale_stride =
                    group == shelf::quant::fp8::fine::kActGroupK ? kp / group : 1u;
                SHELF_RETURN_IF_ERROR(hipok(
                    launch_gemm_mxfp4_w4a8_decode1(
                        mx_d->codes, mx_d->scales, moe_act_codes_dev_,
                        reinterpret_cast<const float*>(moe_act_scales_dev_), eout_buf,
                        Mxfp4GemmOutputDType::BF16, 1u, u32(H), kp, u32(H), sa, stream_),
                    "moe expert down (w4a8)"));
            }
        } else {
            Bf16GemmConfig cfg;
            cfg.id = Bf16GemmConfigId::ExactRows;
            cfg.exact_rows = 1u;
            SHELF_RETURN_IF_ERROR(hipok(
                launch_gemm_bf16(cfg, static_cast<const bf16_t*>(dev_of(w_idx + 2)), mid_buf,
                                 eout_buf, Bf16GemmOutputDType::BF16, 1u, u32(H), u32(width),
                                 u32(width), u32(H), stream_),
                "moe expert down"));
        }
        return hipok(launch_moe_accumulate(acc_buf, eout_buf, weight, u32(H), stream_),
                     "moe expert accumulate");
    };

    // ---- 分担の判定（この行の先頭で固定する）----
    // 3 本とも pinned host のまま（VRAM に載せていない）エキスパートは CPU で計算する。
    // 混在（一部だけ pinned）は GPU に任せる（mapped host を読むので遅いが正しい）。
    const auto pinned_hosted = [&](std::size_t w_idx) -> bool {
        // 切分けの検証用: CPU 分担を止めて GPU（mapped host 読み）に固定する。
        if (std::getenv("SHELF_MOE_NO_CPU") != nullptr) return false;
        for (std::size_t t = 0; t < 3u; ++t) {
            const auto it = host_weight_index_.find(p.weights[w_idx + t].value());
            if (it == host_weight_index_.end()) return false;
            if (host_weights_[it->second].promoted) return false;
        }
        return true;
    };
    std::vector<std::size_t> cpu_sel;
    for (std::size_t j = 0; j < k; ++j) {
        const std::size_t e = sel[j];
        if (pinned_hosted(1u + 3u * e)) {
            cpu_sel.push_back(j);
        } else {
            ++moe_gpu_experts_;
            SHELF_RETURN_IF_ERROR(run_expert(1u + 3u * e, I, sel_w[j]));
        }
    }

    // ---- CPU 分担の本体（GPU のカーネルが流れている間に計算する）----
    std::vector<float> cpu_acc(cpu_sel.empty() ? 0u : H, 0.0f);
    std::vector<float> xf(x_host.size());
    for (std::size_t i = 0; i < x_host.size(); ++i) xf[i] = bf16_host_to_float(x_host[i]);
    // 量子化エキスパートの入力は全エキスパートで共通なので 1 回だけ量子化する。
    std::vector<float> xq_row;
    bool cpu_mx = false;
    for (std::size_t j : cpu_sel) {
        const auto it = host_weight_index_.find(p.weights[1u + 3u * sel[j]].value());
        if (it != host_weight_index_.end() && host_weights_[it->second].kind == 2) cpu_mx = true;
    }
    if (cpu_mx) {
        if (w4a4) {
            quantize_act_mxfp4_host(xf.data(), u32(H), xq_row);
        } else {
            quantize_act_fp8_host(xf.data(), u32(H), x_group, xq_row);
        }
    }
    const auto cpu_expert = [&](std::size_t w_idx, std::size_t width, float weight,
                                float* acc_out) -> Status {
        const auto host_of = [&](std::size_t i) -> const HostWeight* {
            const auto it = host_weight_index_.find(p.weights[i].value());
            if (it == host_weight_index_.end() || host_weights_[it->second].promoted)
                return nullptr;
            return &host_weights_[it->second];
        };
        const HostWeight* g = host_of(w_idx);
        const HostWeight* u = host_of(w_idx + 1u);
        const HostWeight* d = host_of(w_idx + 2u);
        if (!g || !u || !d)
            return Status::invalid_state(
                "moe cpu: pinned weight missing (w_idx=" + std::to_string(w_idx) +
                    ", ids=" + std::to_string(p.weights[w_idx].value()) + "/" +
                    std::to_string(p.weights[w_idx + 1u].value()) + "/" +
                    std::to_string(p.weights[w_idx + 2u].value()) + ")",
                __FILE__, __LINE__);
        std::vector<float> gate(width), up(width), mid(width), out(H);
        const bool mx = g->kind == 2;
        if (mx) {
            const auto* gc = static_cast<const std::uint8_t*>(g->host);
            const auto* uc = static_cast<const std::uint8_t*>(u->host);
            const auto* dc = static_cast<const std::uint8_t*>(d->host);
            // GPU と同じ活性化量子化を通す（W4A4 / W4A8）。これで VRAM 予算の
            // 違い（CPU 分担か GPU か）で数値が変わらない方向に揃う。
            // 入力 x は行で共通なので、量子化は `xq_row` を 1 回だけ作って使う。
            const float* act = xq_row.empty() ? xf.data() : xq_row.data();
            for (std::size_t i = 0; i < width; ++i) {
                gate[i] = dot_mxfp4_row(gc, g->host_scales.data(), u32(i), act, u32(H),
                                        g->k_padded);
                up[i] = dot_mxfp4_row(uc, u->host_scales.data(), u32(i), act, u32(H),
                                      u->k_padded);
            }
            for (std::size_t i = 0; i < width; ++i) {
                // GPU は gate/up を bf16 で書き、swiglu も bf16 を読む。
                // 同じ丸めを通してから量子化しないと、mid の E2M1 コードが
                // 境界で食い違う（実測: 共有/近接 top-1 の入れ替わり）。
                const float gv = bf16_round_host(gate[i]);
                const float uv = bf16_round_host(up[i]);
                mid[i] = bf16_round_host((gv / (1.0f + std::exp(-gv))) * uv);
            }
            std::vector<float> mq;
            if (w4a4) {
                quantize_act_mxfp4_host(mid.data(), u32(width), mq);
            } else {
                const bool g128 = opts_.fp8_act_group != 0u &&
                                  (width % shelf::quant::fp8::fine::kActGroupK) == 0u;
                quantize_act_fp8_host(mid.data(), u32(width),
                                      g128 ? shelf::quant::fp8::fine::kActGroupK : 0u, mq);
            }
            for (std::size_t h = 0; h < H; ++h)
                out[h] = dot_mxfp4_row(dc, d->host_scales.data(), u32(h), mq.data(),
                                       u32(width), d->k_padded);
        } else {
            const auto* gw = static_cast<const bf16_t*>(g->host);
            const auto* uw = static_cast<const bf16_t*>(u->host);
            const auto* dw = static_cast<const bf16_t*>(d->host);
            for (std::size_t i = 0; i < width; ++i) {
                gate[i] = dot_bf16_row(gw + i * H, xf.data(), H);
                up[i] = dot_bf16_row(uw + i * H, xf.data(), H);
            }
            for (std::size_t i = 0; i < width; ++i) {
                const float gv = bf16_round_host(gate[i]);
                const float uv = bf16_round_host(up[i]);
                mid[i] = bf16_round_host((gv / (1.0f + std::exp(-gv))) * uv);
            }
            for (std::size_t h = 0; h < H; ++h)
                out[h] = dot_bf16_row(dw + h * width, mid.data(), width);
        }
        for (std::size_t h = 0; h < H; ++h) acc_out[h] += weight * out[h];
        return Status::make_ok();
    };

    // ---- 共有エキスパート（sigmoid ゲート）----
    if (shared) {
        const std::size_t sbase = 1u + 3u * E;
        // ゲート値は logits バッファを 1 要素分だけ使い回す。
        SHELF_RETURN_IF_ERROR(hipok(
            launch_moe_router_logit(x_row, static_cast<const bf16_t*>(dev_of(sbase + 3u)),
                                    logits_dev, 1u, u32(H), stream_),
            "moe shared gate logit"));
        float gv = 0.0f;
        SHELF_RETURN_IF_ERROR(hipok(
            hipMemcpyAsync(&gv, logits_dev, sizeof(float), hipMemcpyDeviceToHost, stream_),
            "moe shared gate readback"));
        SHELF_RETURN_IF_ERROR(hipok(hipStreamSynchronize(stream_), "moe shared gate sync"));
        const float sg = 1.0f / (1.0f + std::exp(-gv));
        // 共有エキスパートは bf16（checkpoint の量子化除外宣言）。
        if (dev_of(sbase) == nullptr || dev_of(sbase + 1u) == nullptr ||
            dev_of(sbase + 2u) == nullptr)
            return Status::unsupported("moe_mlp: 共有エキスパートは bf16 のみ対応", __FILE__,
                                       __LINE__);
        Bf16GemmConfig cfg;
        cfg.id = Bf16GemmConfigId::ExactRows;
        cfg.exact_rows = 1u;
        SHELF_RETURN_IF_ERROR(hipok(
            launch_gemm_bf16(cfg, static_cast<const bf16_t*>(dev_of(sbase)), x_row, gate_buf,
                             Bf16GemmOutputDType::BF16, 1u, u32(Is), u32(H), u32(H), u32(Is),
                             stream_),
            "moe shared gate"));
        SHELF_RETURN_IF_ERROR(hipok(
            launch_gemm_bf16(cfg, static_cast<const bf16_t*>(dev_of(sbase + 1u)), x_row,
                             up_buf, Bf16GemmOutputDType::BF16, 1u, u32(Is), u32(H), u32(H),
                             u32(Is), stream_),
            "moe shared up"));
        SHELF_RETURN_IF_ERROR(hipok(
            launch_swiglu_bf16(gate_buf, u32(Is), up_buf, u32(Is), mid_buf, u32(Is), 1u,
                               u32(Is), stream_),
            "moe shared swiglu"));
        SHELF_RETURN_IF_ERROR(hipok(
            launch_gemm_bf16(cfg, static_cast<const bf16_t*>(dev_of(sbase + 2u)), mid_buf,
                             eout_buf, Bf16GemmOutputDType::BF16, 1u, u32(H), u32(Is), u32(Is),
                             u32(H), stream_),
            "moe shared down"));
        SHELF_RETURN_IF_ERROR(
            hipok(launch_moe_accumulate(acc_buf, eout_buf, sg, u32(H), stream_),
                  "moe shared accumulate"));
    }

    // ---- CPU 分担の合流 ----
    // GPU のカーネル（エキスパート・共有）は enqueue 済みで流れている。その間に
    // CPU が pinned のエキスパートを計算し、最後に 1 回だけ H2D して足す。
    if (!cpu_sel.empty()) {
        const auto cpu_t0 = std::chrono::steady_clock::now();
        // エキスパート単位で分割する（結果は固定の分割・固定の足し順なので決定的）。
        unsigned nt = std::thread::hardware_concurrency();
        if (nt == 0u) nt = 1u;
        if (const char* e = std::getenv("SHELF_MOE_CPU_THREADS"))
            nt = static_cast<unsigned>(std::strtoul(e, nullptr, 10));
        nt = std::max(1u, std::min<unsigned>(nt, 16u));
        nt = std::min<unsigned>(nt, static_cast<unsigned>(cpu_sel.size()));
        std::vector<std::vector<float>> partial(
            nt, std::vector<float>(H, 0.0f));
        std::vector<Status> errs(nt, Status::make_ok());
        if (nt <= 1u) {
            for (std::size_t idx = 0; idx < cpu_sel.size(); ++idx) {
                const std::size_t pos = cpu_sel[idx];
                const std::size_t e = sel[pos];
                errs[0] = cpu_expert(1u + 3u * e, I, sel_w[pos], partial[0].data());
                if (!errs[0].ok()) break;
            }
        } else {
            std::vector<std::thread> workers;
            workers.reserve(nt);
            for (unsigned t = 0; t < nt; ++t) {
                workers.emplace_back([&, t] {
                    for (std::size_t idx = t; idx < cpu_sel.size(); idx += nt) {
                        const std::size_t pos = cpu_sel[idx];
                        const std::size_t e = sel[pos];
                        errs[t] = cpu_expert(1u + 3u * e, I, sel_w[pos],
                                             partial[t].data());
                        if (!errs[t].ok()) return;
                    }
                });
            }
            for (auto& w : workers) w.join();
        }
        for (unsigned t = 0; t < nt; ++t)
            SHELF_RETURN_IF_ERROR(errs[t]);
        for (unsigned t = 0; t < nt; ++t)
            for (std::size_t h = 0; h < H; ++h) cpu_acc[h] += partial[t][h];
        moe_cpu_experts_ += cpu_sel.size();
        moe_cpu_ms_ += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - cpu_t0)
                           .count();
        SHELF_RETURN_IF_ERROR(hipok(
            hipMemcpyAsync(cpu_acc_dev, cpu_acc.data(), H * sizeof(float),
                           hipMemcpyHostToDevice, stream_),
            "moe cpu acc upload"));
        SHELF_RETURN_IF_ERROR(hipok(hipStreamSynchronize(stream_), "moe cpu acc sync"));
        SHELF_RETURN_IF_ERROR(
            hipok(launch_moe_add_f32(acc_buf, cpu_acc_dev, u32(H), stream_),
                  "moe cpu acc add"));
    }

    return hipok(launch_f32_to_bf16(acc_buf, out_row, u32(H), stream_), "moe output");
}

Status HipStepExecutor::declare_states(const ir::Graph& g) {
    const auto declare_t0 = std::chrono::steady_clock::now();
    kv_shared_pool_ = opts_.kv_blocks != 0u;
    kv_total_pages_ = 0;
    kv_pages_per_seq_ = 0;
    struct DeclareTimer {
        std::chrono::steady_clock::time_point t0;
        double* out;
        ~DeclareTimer() {
            *out = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
        }
    } declare_timer{declare_t0, &stats_.declare_ms};
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
            // T16: グローバルプール。既定はスロットごとの固定範囲（従来と同じ）。
            const std::size_t pages = loc.pages;
            loc.total_pages = opts_.kv_blocks != 0u ? opts_.kv_blocks
                                                    : pages * max_sequences_;
            if (loc.total_pages < pages)
                return Status::invalid_argument(
                    "kv_blocks が 1 シーケンス分（" + std::to_string(pages) +
                        "）より小さい",
                    __FILE__, __LINE__);
            if (kv_pages_per_seq_ != 0 && kv_pages_per_seq_ != pages)
                return Status::invalid_state("KV 層で 1 シーケンスのページ数が食い違う",
                                             __FILE__, __LINE__);
            if (kv_total_pages_ != 0 && kv_total_pages_ != loc.total_pages)
                return Status::invalid_state("KV 層でグローバル block 数が食い違う",
                                             __FILE__, __LINE__);
            kv_pages_per_seq_ = pages;
            kv_total_pages_ = loc.total_pages;
            const std::size_t pool_elems = loc.total_pages * bs * kvh * D;
            loc.pool_bytes = pool_elems * sizeof(bf16_t);   // reset で 0 クリアする
            const std::size_t pool_bytes = pool_elems * sizeof(bf16_t);
            SHELF_RETURN_IF_ERROR(alloc_state_pool(&loc.k_pool, &loc.host_k, pool_bytes,
                                                   "kv pool K"));
            SHELF_RETURN_IF_ERROR(alloc_state_pool(&loc.v_pool, &loc.host_v, pool_bytes,
                                                   "kv pool V"));
            // T14: online KV（直近ページの VRAM ミラー）。
            if (opts_.online_kv_pages != 0u) {
                if (loc.host_k == nullptr)
                    return Status::invalid_argument(
                        "online_kv_pages は kv_pinned_host のときだけ使える", __FILE__,
                        __LINE__);
                const std::size_t per_page = bs * kvh * D;
                const std::size_t online_elems =
                    max_sequences_ * opts_.online_kv_pages * per_page;
                const std::size_t online_bytes = online_elems * sizeof(bf16_t);
                SHELF_RETURN_IF_ERROR(
                    dev_reserve(&loc.online_k, online_bytes, "online kv K"));
                SHELF_RETURN_IF_ERROR(
                    dev_reserve(&loc.online_v, online_bytes, "online kv V"));
                SHELF_RETURN_IF_ERROR(
                    hipok(hipMemset(loc.online_k, 0, online_bytes), "zero online kv K"));
                SHELF_RETURN_IF_ERROR(
                    hipok(hipMemset(loc.online_v, 0, online_bytes), "zero online kv V"));
                loc.online_pages = opts_.online_kv_pages;
                loc.online_bytes = online_bytes;
            }
            if (loc.host_k != nullptr) {
                std::memset(loc.host_k, 0, pool_bytes);
            } else {
                SHELF_RETURN_IF_ERROR(
                    hipok(hipMemset(loc.k_pool, 0, pool_bytes), "zero kv K"));
            }
            if (loc.host_v != nullptr) {
                std::memset(loc.host_v, 0, pool_bytes);
            } else {
                SHELF_RETURN_IF_ERROR(
                    hipok(hipMemset(loc.v_pool, 0, pool_bytes), "zero kv V"));
            }
            // ブロック表は slot ごとに `pages` エントリ。既定は
            // `slot * pages + 論理 page`（従来の identity）。共有プールでは
            // `assign_kv_blocks` が任意のグローバル block を指す。
            std::vector<std::uint32_t> t(max_sequences_ * pages);
            for (std::size_t slot = 0; slot < max_sequences_; ++slot)
                for (std::size_t p = 0; p < pages; ++p) {
                    const std::size_t v = slot * pages + p;
                    t[slot * pages + p] =
                        static_cast<std::uint32_t>(v < loc.total_pages ? v
                                                                      : loc.total_pages - 1);
                }
            SHELF_RETURN_IF_ERROR(dev_alloc(&loc.table, t.size(), "kv table"));
            SHELF_RETURN_IF_ERROR(hipok(hipMemcpy(loc.table, t.data(),
                                                  t.size() * sizeof(std::uint32_t),
                                                  hipMemcpyHostToDevice),
                                        "kv table upload"));
        } else {
            loc.elems = numel(sh) * max_sequences_;   // スロット先頭オフセットで並べる
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
    // **必ず実行ストリーム上で** 0 クリアする。`hipMemset`（既定ストリーム）は
    // 実行器の non-blocking stream_ と順序が保証されず、直前のカーネルと競合する
    // （実測: 2 回目のシーケンスだけ結果が変わる原因だった）。
    // pinned host の KV はストリーム上で memset できない。実行中のカーネルを
    // 待ってからホスト側で 0 を書く（次のステップはその後で発射する）。
    bool any_host_kv = false;
    for (const auto& [id, loc] : state_dev_)
        if (loc.host_k != nullptr || loc.host_v != nullptr) any_host_kv = true;
    if (any_host_kv) (void)hipStreamSynchronize(stream_);
    for (auto& [id, loc] : state_dev_) {
        loc.rows = 0;
        if (loc.dev)
            SHELF_RETURN_IF_ERROR(hipok(
                hipMemsetAsync(loc.dev, 0, loc.elems * sizeof(float), stream_), "reset state"));
        if (loc.k_pool != nullptr && loc.pool_bytes != 0) {
            if (loc.host_k != nullptr)
                std::memset(loc.host_k, 0, loc.pool_bytes);
            else
                SHELF_RETURN_IF_ERROR(hipok(
                    hipMemsetAsync(loc.k_pool, 0, loc.pool_bytes, stream_), "reset kv K"));
        }
        if (loc.v_pool != nullptr && loc.pool_bytes != 0) {
            if (loc.host_v != nullptr)
                std::memset(loc.host_v, 0, loc.pool_bytes);
            else
                SHELF_RETURN_IF_ERROR(hipok(
                    hipMemsetAsync(loc.v_pool, 0, loc.pool_bytes, stream_), "reset kv V"));
        }
        if (loc.online_k != nullptr && loc.online_bytes != 0)
            SHELF_RETURN_IF_ERROR(hipok(
                hipMemsetAsync(loc.online_k, 0, loc.online_bytes, stream_), "reset online K"));
        if (loc.online_v != nullptr && loc.online_bytes != 0)
            SHELF_RETURN_IF_ERROR(hipok(
                hipMemsetAsync(loc.online_v, 0, loc.online_bytes, stream_), "reset online V"));
    }
    // KV プールも 0 に戻す。rows=0 なら「読まれない」はずだが、**ブロック単位で
    // 読むカーネル**（1 ブロック 16 トークン等）では未書き込み域の残骸が見える経路が
    // あり、実測で 2 回目のシーケンスが 1 回目と変わった（`shelf-validate` の T0 が
    // bf16 / FP8 の両方で diverged）。declare 直後と同じ条件へ戻すためゼロで埋める。
    //
    // アリーナ（活性化＋ declare 相の scratch）はステップ内で再利用するため、
    // **前のシーケンスの最終ステップの値**が残る。1 ステップ目に scratch を
    // 読むカーネル（rope の展開・GDN の準備領域など）があると、2 回目以降の
    // リクエストが 1 回目と違う結果になる（実測: 同一プロンプト・greedy でも
    // 2 回目が変わる）。新しいシーケンスの 1 ステップ目でアリーナを 0 クリアして
    // 「プロセス起動直後と同じ条件」に戻す。
    // アリーナ（ステップ scratch）も 0 に戻す。前シーケンスの最終ステップの値が
    // 残ると、1 ステップ目に scratch を読むカーネル（rope 展開・GDN 準備）が
    // 影響を受ける。reset_states は要求境界なので、ここで一度だけ払う。
    if (arena_dev_ != nullptr && arena_cap_ != 0)
        SHELF_RETURN_IF_ERROR(
            hipok(hipMemsetAsync(arena_dev_, 0, arena_cap_, stream_), "reset arena"));
    arena_zero_pending_ = false;
    return Status::make_ok();
}

Status HipStepExecutor::reset_slot(std::size_t slot) {
    if (slot >= max_sequences_)
        return Status::invalid_argument("reset_slot: slot >= max_sequences", __FILE__, __LINE__);
    (void)hipSetDevice(opts_.device);
    bool any_host_kv = false;
    for (const auto& [id, loc] : state_dev_)
        if (loc.host_k != nullptr || loc.host_v != nullptr) any_host_kv = true;
    if (any_host_kv) (void)hipStreamSynchronize(stream_);
    for (auto& [id, loc] : state_dev_) {
        loc.rows = 0;
        if (loc.kind == ir::StateKind::kv_cache && kv_shared_pool_) {
            // 共有プールでは KV ページを 0 にしない（他スロットが参照しうる）。
            // スロットのブロック表だけ無効化し、呼び手が assign し直す。
            if (loc.table != nullptr && loc.pages != 0)
                SHELF_RETURN_IF_ERROR(hipok(
                    hipMemsetAsync(loc.table + slot * loc.pages, 0,
                                   loc.pages * sizeof(std::uint32_t), stream_),
                    "reset slot table"));
        } else if (loc.kind == ir::StateKind::kv_cache) {
            // pool_bytes は全スロット分。1 スロット分に割って先頭をずらす。
            const std::size_t bytes =
                loc.pool_bytes / std::max<std::size_t>(max_sequences_, 1u);
            const std::size_t off = slot * bytes;
            if (loc.host_k != nullptr)
                std::memset(static_cast<char*>(loc.host_k) + off, 0, bytes);
            else
                SHELF_RETURN_IF_ERROR(hipok(
                    hipMemsetAsync(static_cast<char*>(loc.k_pool) + off, 0, bytes, stream_),
                    "reset slot kv K"));
            if (loc.host_v != nullptr)
                std::memset(static_cast<char*>(loc.host_v) + off, 0, bytes);
            else
                SHELF_RETURN_IF_ERROR(hipok(
                    hipMemsetAsync(static_cast<char*>(loc.v_pool) + off, 0, bytes, stream_),
                    "reset slot kv V"));
        } else if (loc.dev != nullptr) {
            const std::size_t per = loc.elems / std::max<std::size_t>(max_sequences_, 1u);
            SHELF_RETURN_IF_ERROR(hipok(
                hipMemsetAsync(static_cast<char*>(loc.dev) +
                                   slot * per * sizeof(float),
                               0, per * sizeof(float), stream_),
                "reset slot state"));
        }
        // T14: スロット再利用時に前の要求の K/V がミラーに残らないようにする。
        if (loc.online_k != nullptr && loc.online_bytes != 0) {
            const std::size_t per = loc.online_bytes / std::max<std::size_t>(max_sequences_, 1u);
            SHELF_RETURN_IF_ERROR(hipok(
                hipMemsetAsync(static_cast<char*>(loc.online_k) + slot * per, 0, per,
                               stream_),
                "reset slot online K"));
            SHELF_RETURN_IF_ERROR(hipok(
                hipMemsetAsync(static_cast<char*>(loc.online_v) + slot * per, 0, per,
                               stream_),
                "reset slot online V"));
        }
    }
    // 共有 scratch（アリーナ）の残骸も落とす（他スロットの状態は壊さない）。
    if (arena_dev_ != nullptr && arena_cap_ != 0)
        SHELF_RETURN_IF_ERROR(
            hipok(hipMemsetAsync(arena_dev_, 0, arena_cap_, stream_), "reset slot arena"));
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
            if (F == 0u || (F % shelf::quant::fp8::fine::kActGroupK) != 0u) continue;
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
            if (F == 0u || (F % shelf::quant::fp8::fine::kActGroupK) != 0u) continue;
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

Result<std::vector<float>> HipStepExecutor::export_gdn_state(std::size_t slot) const {
    if (slot >= max_sequences_)
        return Status::invalid_argument("export_gdn_state: slot が範囲外", __FILE__,
                                        __LINE__);
    (void)hipSetDevice(opts_.device);
    (void)hipStreamSynchronize(stream_);
    std::vector<float> out;
    for (const auto id : state_ids_) {
        const auto it = state_dev_.find(id.value());
        if (it == state_dev_.end()) continue;
        const StateLoc& loc = it->second;
        if (loc.kind == ir::StateKind::kv_cache || loc.dev == nullptr) continue;
        const std::size_t per =
            loc.elems / std::max<std::size_t>(max_sequences_, 1u);
        const std::size_t at = out.size();
        out.resize(at + per);
        const hipError_t e =
            hipMemcpy(out.data() + at,
                      static_cast<const char*>(loc.dev) + slot * per * sizeof(float),
                      per * sizeof(float), hipMemcpyDeviceToHost);
        if (e != hipSuccess)
            return Status::from(Status::Code::device_error,
                                std::string("export_gdn_state: ") + hipGetErrorString(e),
                                __FILE__, __LINE__);
    }
    return out;
}

Status HipStepExecutor::import_gdn_state(std::size_t slot, const std::vector<float>& data) {
    if (slot >= max_sequences_)
        return Status::invalid_argument("import_gdn_state: slot が範囲外", __FILE__,
                                        __LINE__);
    (void)hipSetDevice(opts_.device);
    (void)hipStreamSynchronize(stream_);
    std::size_t at = 0;
    for (const auto id : state_ids_) {
        const auto it = state_dev_.find(id.value());
        if (it == state_dev_.end()) continue;
        StateLoc& loc = it->second;
        if (loc.kind == ir::StateKind::kv_cache || loc.dev == nullptr) continue;
        const std::size_t per =
            loc.elems / std::max<std::size_t>(max_sequences_, 1u);
        if (at + per > data.size())
            return Status::invalid_argument("import_gdn_state: データが短い", __FILE__,
                                            __LINE__);
        const hipError_t e =
            hipMemcpy(static_cast<char*>(loc.dev) + slot * per * sizeof(float),
                      data.data() + at, per * sizeof(float), hipMemcpyHostToDevice);
        if (e != hipSuccess)
            return Status::from(Status::Code::device_error,
                                std::string("import_gdn_state: ") + hipGetErrorString(e),
                                __FILE__, __LINE__);
        at += per;
    }
    if (at != data.size())
        return Status::invalid_argument("import_gdn_state: データが長すぎる", __FILE__,
                                        __LINE__);
    return Status::make_ok();
}

Status HipStepExecutor::assign_kv_blocks(std::size_t slot, const std::uint32_t* blocks,
                                         std::size_t count, std::size_t offset) {
    if (slot >= max_sequences_)
        return Status::invalid_argument("assign_kv_blocks: slot が範囲外", __FILE__,
                                        __LINE__);
    if (count == 0u) return Status::make_ok();
    if (blocks == nullptr)
        return Status::invalid_argument("assign_kv_blocks: blocks が null", __FILE__,
                                        __LINE__);
    if (kv_pages_per_seq_ == 0u || offset + count > kv_pages_per_seq_)
        return Status::invalid_argument("assign_kv_blocks: 論理 page が範囲外", __FILE__,
                                        __LINE__);
    for (std::size_t i = 0; i < count; ++i)
        if (blocks[i] >= kv_total_pages_)
            return Status::invalid_argument("assign_kv_blocks: グローバル block が範囲外",
                                            __FILE__, __LINE__);
    for (auto& [id, loc] : state_dev_) {
        if (loc.kind != ir::StateKind::kv_cache || loc.table == nullptr) continue;
        const std::size_t base = slot * loc.pages + offset;
        SHELF_RETURN_IF_ERROR(hipok(
            hipMemcpyAsync(loc.table + base, blocks, count * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, stream_),
            "assign kv blocks"));
        // T16 + T14: prefix で共有したページは kv_append を通らないため、
        // online ミラーへも複製しておく（窓内で読まれる可能性がある）。
        if (loc.online_pages != 0u && loc.online_k != nullptr &&
            loc.k_pool != nullptr && loc.total_pages != 0u) {
            const std::size_t page_bytes = loc.pool_bytes / loc.total_pages;
            for (std::size_t i = 0; i < count; ++i) {
                const std::uint32_t logical = u32(offset + i);
                const std::size_t dst =
                    (slot * loc.online_pages + (logical % loc.online_pages)) * page_bytes;
                const std::size_t src = static_cast<std::size_t>(blocks[i]) * page_bytes;
                SHELF_RETURN_IF_ERROR(hipok(
                    hipMemcpyAsync(static_cast<char*>(loc.online_k) + dst,
                                   static_cast<const char*>(loc.k_pool) + src, page_bytes,
                                   hipMemcpyDeviceToDevice, stream_),
                    "prefix -> online K"));
                SHELF_RETURN_IF_ERROR(hipok(
                    hipMemcpyAsync(static_cast<char*>(loc.online_v) + dst,
                                   static_cast<const char*>(loc.v_pool) + src, page_bytes,
                                   hipMemcpyDeviceToDevice, stream_),
                    "prefix -> online V"));
            }
        }
    }
    return Status::make_ok();
}
Status HipStepExecutor::run_step(const ir::Graph& g, const std::int32_t* tokens,
                                 std::size_t n, std::uint32_t base_pos) {
    return run_stage(g, tokens, nullptr, n, base_pos);
}

Status HipStepExecutor::run_stage(const ir::Graph& g, const std::int32_t* tokens,
                                  const bf16_t* hidden, std::size_t n,
                                  std::uint32_t base_pos) {
    return run_impl(g, tokens, hidden, n, base_pos, nullptr);
}

Status HipStepExecutor::run_batch(const ir::Graph& g, const std::int32_t* tokens,
                                  const bf16_t* hidden, std::size_t n, const BatchRows& rows) {
    return run_impl(g, tokens, hidden, n, 0, &rows);
}

Status HipStepExecutor::run_stage_batch(const ir::Graph& g, const std::int32_t* tokens,
                                        const bf16_t* hidden, std::size_t n,
                                        const RowBatch& rows) {
    const BatchRows br{rows.slots, rows.positions};
    return run_batch(g, tokens, hidden, n, br);
}

Status HipStepExecutor::run_impl(const ir::Graph& g, const std::int32_t* tokens,
                                 const bf16_t* hidden, std::size_t n,
                                 std::uint32_t base_pos, const BatchRows* rows) {
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
        // 新しいアリーナはゼロ初期化されていない（hipMalloc は未定義）。最初の
        // ステップの scratch を読むカーネルのために、確保したら 0 で埋める。
        arena_zero_pending_ = true;
    }
    plan_ = &plan;
    value_dev_.clear();
    logits_dev_ = nullptr;   // 解放済みアリーナを指さないよう消す
    step_rows_ = n;
    step_base_pos_ = base_pos;

    const auto t0 = std::chrono::steady_clock::now();

    // ステップ内訳（SHELF_HIP_STEP_PROFILE=1）。イベントは 3 点:
    //   sa = 入力アップロード前 / sk = カーネル列の前 / sb = カーネル列の後。
    // 既定では作らない（ホットパスに追加コストを載せない）。
    const bool step_profile = std::getenv("SHELF_HIP_STEP_PROFILE") != nullptr;
    hipEvent_t ev_sa = nullptr, ev_sk = nullptr, ev_sb = nullptr;
    struct EventGuard {
        hipEvent_t* e;
        ~EventGuard() {
            if (*e) (void)hipEventDestroy(*e);
        }
    } guard_sa{&ev_sa}, guard_sk{&ev_sk}, guard_sb{&ev_sb};
    if (step_profile) {
        (void)hipEventCreate(&ev_sa);
        (void)hipEventCreate(&ev_sk);
        (void)hipEventCreate(&ev_sb);
        (void)hipEventRecord(ev_sa, stream_);
    }

    // エラー時のみ未消化のカーネル／コピーを先に流し切ってから返す。
    // ここで返った後に呼び出し側がメモリを解放すると、生きているカーネルが
    // 解放済みメモリを触り HSA のメモリフォルト／ハングになるため。
    const auto sync_on_error = [&](const Status& s) -> Status {
        if (!s.ok()) (void)hipStreamSynchronize(stream_);
        return s;
    };

    // 行ごとの slot / 位置（run_batch でなければ全行 slot 0・連続位置）。
    rowslot_host_.assign(n, 0u);
    cur_max_pos_ = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t slot = (rows != nullptr && rows->slots != nullptr)
                                       ? rows->slots[i]
                                       : 0u;
        if (slot >= max_sequences_)
            return Status::invalid_argument("run_batch: slot >= max_sequences", __FILE__,
                                            __LINE__);
        const std::uint32_t pos = (rows != nullptr && rows->positions != nullptr)
                                      ? rows->positions[i]
                                      : base_pos + static_cast<std::uint32_t>(i);
        rowslot_host_[i] = slot;
        posrep_host_[i] = pos;
        cur_max_pos_ = std::max(cur_max_pos_, pos + 1u);
    }
    // T14: online KV の窓（スロットごとに書き込まれた論理ページ数）。
    // 行の位置からスロット最大を取り、device 配列に載せる（attention が読む）。
    slot_window_host_.assign(max_sequences_, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t pages =
            (posrep_host_[i] + 1u + u32(opts_.kv_block_size) - 1u) /
            u32(opts_.kv_block_size);
        std::uint32_t& w = slot_window_host_[rowslot_host_[i]];
        w = std::max(w, pages);
    }
    SHELF_RETURN_IF_ERROR(sync_on_error(
        hipok(hipMemcpyAsync(slot_window_dev_, slot_window_host_.data(),
                             max_sequences_ * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                             stream_),
              "slot window upload")));
    // 複数スロットか（行は slot ごとに連続している前提）。
    rows_multi_slot_ = false;
    for (std::size_t i = 1; i < n; ++i)
        if (rowslot_host_[i] != rowslot_host_[0]) rows_multi_slot_ = true;

    // GDN のリクエスト記述子: 連続する同一 slot を 1 リクエストにまとめる。
    std::vector<hipk::GdnRequestDescriptor> descs(max_sequences_);
    cur_requests_ = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (i == 0 || rowslot_host_[i] != rowslot_host_[i - 1]) {
            descs[cur_requests_] = hipk::GdnRequestDescriptor{
                rowslot_host_[i], u32(i), 0u};
            ++cur_requests_;
        }
    }
    for (std::uint32_t r = 0; r < cur_requests_; ++r) {
        const std::size_t begin = descs[r].row_begin;
        const std::size_t end =
            (r + 1u < cur_requests_) ? descs[r + 1u].row_begin : n;
        descs[r].row_count = u32(end - begin);
    }
    SHELF_RETURN_IF_ERROR(sync_on_error(
        hipok(hipMemcpyAsync(req_dev_, descs.data(),
                             cur_requests_ * sizeof(hipk::GdnRequestDescriptor),
                             hipMemcpyHostToDevice, stream_),
              "gdn req upload")));
    SHELF_RETURN_IF_ERROR(sync_on_error(
        hipok(hipMemcpyAsync(rowpos_dev_, posrep_host_.data(),
                             n * sizeof(std::uint32_t), hipMemcpyHostToDevice, stream_),
              "rowpos upload")));
    SHELF_RETURN_IF_ERROR(sync_on_error(
        hipok(hipMemcpyAsync(rowslot_dev_, rowslot_host_.data(),
                             n * sizeof(std::uint32_t), hipMemcpyHostToDevice, stream_),
              "rowslot upload")));

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

    if (step_profile) (void)hipEventRecord(ev_sk, stream_);

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
    if (step_profile) (void)hipEventRecord(ev_sb, stream_);
    SHELF_RETURN_IF_ERROR(hipok(hipStreamSynchronize(stream_), "step sync"));
    stats_.last_step_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
    if (step_profile) {
        float total = 0.0f, kern = 0.0f, up = 0.0f;
        (void)hipEventElapsedTime(&total, ev_sa, ev_sb);
        (void)hipEventElapsedTime(&kern, ev_sk, ev_sb);
        (void)hipEventElapsedTime(&up, ev_sa, ev_sk);
        stats_.last_gpu_ms = static_cast<double>(total);
        stats_.last_kernel_ms = static_cast<double>(kern);
        stats_.last_upload_ms = static_cast<double>(up);
        stats_.last_host_ms = std::max(0.0, stats_.last_step_ms - stats_.last_gpu_ms);
    } else {
        stats_.last_gpu_ms = 0.0;
        stats_.last_kernel_ms = 0.0;
        stats_.last_upload_ms = 0.0;
        stats_.last_host_ms = 0.0;
    }
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
        // 重みが MXFP4（quark: E2M1 ＋ E8M0）のとき、活性化は checkpoint の
        // 宣言で W4A4（A も E2M1+E8M0）と W4A8（A は既存の FP8）を選ぶ。
        //   * W4A4: `global_quant_config.input_tensors.dtype == fp4`（T09）
        //   * W4A8: それ以外（従来動作）
        const auto mx = p.weights.empty()
                            ? weight_mxfp4_.end()
                            : weight_mxfp4_.find(p.weights[0].value());
        if (mx != weight_mxfp4_.end()) {
            using namespace shelf::quant::mxfp4;
            const std::uint32_t kp = pad_k(u32(K));
            if (kp != u32(K))
                return Status::unsupported(
                    "linear(mxfp4): K が 32 の倍数でない", __FILE__, __LINE__);
            // prefill 用カーネルは 128 行タイルだが、端数の行は読み込み側で clamp
            // 済み（`linear_mxfp4_prefill.hip`。単体テストで 16/32/64/96 行が誤差 0）。
            // カーネルの前提は K が 128 の倍数であること（W4A4/W4A8 とも）。
            if ((kp % 128u) != 0u)
                return Status::unsupported(
                    "linear(mxfp4): prefill は K が 128 の倍数のみ（k_padded " +
                        std::to_string(kp) + "）",
                    __FILE__, __LINE__);
            const auto dt = f32_out ? Mxfp4GemmOutputDType::F32 : Mxfp4GemmOutputDType::BF16;
            Mxfp4GemmScaleArgs sa;
            sa.weight_scale_stride = kp / kBlockK;
            if (opts_.mxfp4_activation) {
                // ---- W4A4: 活性化を E2M1 + E8M0 へ（断片順ニブル + E8M0）----
                if (!activation_quantize_mxfp4_supported(u32(K), kp))
                    return Status::unsupported(
                        "linear(mxfp4/w4a4): 活性化の形が受理範囲外", __FILE__, __LINE__);
                const std::size_t codes_n = tiled_bytes(u32(rows), kp);
                const std::size_t scales_n = scale_bytes(u32(rows), kp);
                SHELF_RETURN_IF_ERROR(ensure_mxfp4_scratch(codes_n, scales_n));
                SHELF_RETURN_IF_ERROR(hipok(
                    launch_activation_quantize_mxfp4(
                        static_cast<const bf16_t*>(x->dev), mxfp4_act_codes_dev_,
                        mxfp4_act_scales_dev_, u32(rows), u32(K), kp, u32(K), stream_),
                    "mxfp4/w4a4 act quantize"));
                sa.act_group = 0u;
                sa.activation_scale_stride = kp / kBlockK;
                if (rows == 1u && mxfp4_decode1_supported(u32(rows), u32(N), kp))
                    SHELF_RETURN_IF_ERROR(hipok(
                        launch_gemm_mxfp4_w4a4_decode1(
                            mx->second.codes, mx->second.scales, mxfp4_act_codes_dev_,
                            mxfp4_act_scales_dev_, o, dt, u32(rows), u32(N), kp, u32(N), sa,
                            stream_),
                        "gemm mxfp4 w4a4 decode1"));
                else
                    SHELF_RETURN_IF_ERROR(hipok(
                        launch_gemm_mxfp4_w4a4_prefill(
                            mx->second.codes, mx->second.scales, mxfp4_act_codes_dev_,
                            mxfp4_act_scales_dev_, o, dt, u32(rows), u32(N), kp, u32(N), sa,
                            stream_),
                        "gemm mxfp4 w4a4 prefill"));
            } else {
                // ---- W4A8: 活性化は既存の FP8 量子化 ----
                const std::uint32_t act_group =
                    (opts_.fp8_act_group != 0u &&
                     (kp % shelf::quant::fp8::fine::kActGroupK) == 0u)
                        ? shelf::quant::fp8::fine::kActGroupK
                        : 0u;
                const std::size_t act_scales =
                    shelf::quant::fp8::fine::activation_scale_count(u32(rows), kp, act_group);
                SHELF_RETURN_IF_ERROR(ensure_fp8_scratch(
                    shelf::quant::fp8::tiled_bytes(u32(rows), kp), act_scales));
                SHELF_RETURN_IF_ERROR(hipok(launch_activation_quantize_fp8(
                                              static_cast<const bf16_t*>(x->dev),
                                              fp8_codes_dev_, fp8_scales_dev_, u32(rows),
                                              u32(K), kp, u32(K), act_group, stream_),
                                          "mxfp4 act quantize (fp8/uint8)"));
                sa.act_group = act_group;
                sa.activation_scale_stride =
                    static_cast<std::uint32_t>(act_scales / (rows == 0 ? 1 : rows));
                if (rows == 1u && mxfp4_decode1_supported(u32(rows), u32(N), kp))
                    SHELF_RETURN_IF_ERROR(hipok(launch_gemm_mxfp4_w4a8_decode1(
                                                  mx->second.codes, mx->second.scales,
                                                  fp8_codes_dev_, fp8_scales_dev_, o, dt,
                                                  u32(rows), u32(N), kp, u32(N), sa, stream_),
                                              "gemm mxfp4 w4a8 decode1"));
                else
                    SHELF_RETURN_IF_ERROR(hipok(launch_gemm_mxfp4_w4a8_prefill(
                                                  mx->second.codes, mx->second.scales,
                                                  fp8_codes_dev_, fp8_scales_dev_, o, dt,
                                                  u32(rows), u32(N), kp, u32(N), sa, stream_),
                                              "gemm mxfp4 w4a8 prefill"));
            }
            if (f32_out) logits_dev_ = static_cast<float*>(o);
            return set_out(f32_out, o);
        }

        const auto f8 = p.weights.empty()
                            ? weight_fp8_.end()
                            : weight_fp8_.find(p.weights[0].value());
        if (f8 != weight_fp8_.end()) {
            using namespace shelf::quant::fp8;
using namespace shelf::quant::fp8::fine;
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
            F % shelf::quant::fp8::fine::kActGroupK == 0u &&
            shelf::quant::fp8::pad_k(u32(F)) == u32(F)) {
            using namespace shelf::quant::fp8;
using namespace shelf::quant::fp8::fine;
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
        // W4A4 のときは FP8 への融合をしない（次の linear が E2M1 で量子化し直すため）。
        const bool fusable = !opts_.mxfp4_activation &&
                             std::find(fuse.swiglu_out.begin(), fuse.swiglu_out.end(),
                                       p.output.id.value()) != fuse.swiglu_out.end() &&
                             F % shelf::quant::fp8::fine::kActGroupK == 0u;
        if (fusable) {
            using namespace shelf::quant::fp8;
using namespace shelf::quant::fp8::fine;
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
        args.conv.num_requests = cur_requests_;
        args.conv.actual_rows = u32(step_rows_);
        args.conv.max_request_rows = u32(step_rows_);
        args.conv.input_row_stride = u32(C);
        args.conv.output_row_stride = u32(C);
        args.conv.conv_dim = u32(C);
        args.conv.conv_history = 3u;
        args.conv.state_index = 0u;
        args.conv.max_sequences = u32(max_sequences_);
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
        rec.num_requests = cur_requests_;
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
        rec.max_sequences = u32(max_sequences_);
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
        st.rows = std::max<std::size_t>(st.rows, cur_max_pos_);
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
        if (!opts_.mxfp4_activation && feat_flat != 0u && step_rows_ > 1u &&
            feat_flat % shelf::quant::fp8::fine::kActGroupK == 0u &&
            shelf::quant::fp8::pad_k(u32(feat_flat)) == u32(feat_flat) &&
            std::find(gfuse.gated_out.begin(), gfuse.gated_out.end(),
                      p.output.id.value()) != gfuse.gated_out.end()) {
            using namespace shelf::quant::fp8;
using namespace shelf::quant::fp8::fine;
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
        // バッチでは行ごとに位置が違うので、最大位置 + 1 で容量を判定する。
        if (cur_max_pos_ > desc->logical_shape.dim(0))
            return Status::out_of_range("attention_gqa_step: KV context full", __FILE__,
                                        __LINE__);
        const std::size_t bs = opts_.kv_block_size;
        const std::size_t kvh = desc->logical_shape.dim(2);
        const std::size_t D = desc->logical_shape.dim(3);
        const std::size_t P = step_rows_;
        const std::size_t nq = q->elems / (D * P);   // per-token query heads
        if (q->elems % (D * P)) return Status::invalid_state("attention: q width mismatch", __FILE__, __LINE__);
        const std::size_t ept = kvh * D, epp = bs * ept,
                          epl = (st.total_pages != 0 ? st.total_pages : st.pages) * epp;

        KvAppendBf16Args ap{};
        ap.common.k_input = static_cast<const bf16_t*>(k->dev);
        ap.common.v_input = static_cast<const bf16_t*>(v->dev);
        ap.common.k_row_stride = u32(k->elems / P);
        ap.common.v_row_stride = u32(v->elems / P);
        ap.common.row_sequence_slots = rowslot_dev_;
        ap.common.row_positions = rowpos_dev_;
        ap.common.block_tables = st.table;
        ap.common.block_table_stride = u32(st.pages);
        ap.common.max_sequences = u32(max_sequences_);
        ap.common.rows = u32(step_rows_);
        ap.common.layer = 0u;
        ap.common.page_tokens = u32(bs);
        // num_pages はプール全体のページ数（kernel は block_table の値をこれで
        // bound する）。block_table_stride は 1 シーケンス分（st.pages）。
        ap.common.num_pages = u32(st.total_pages != 0 ? st.total_pages : st.pages);
        ap.common.num_attention_layers = 1u;
        ap.common.kv_heads = u32(kvh);
        ap.common.head_dim = u32(D);
        ap.common.elems_per_token = ept;
        ap.common.elems_per_page = epp;
        ap.common.elems_per_layer = epl;
        ap.k_pool = static_cast<bf16_t*>(st.k_pool);
        ap.v_pool = static_cast<bf16_t*>(st.v_pool);
        ap.online_k = static_cast<bf16_t*>(st.online_k);
        ap.online_v = static_cast<bf16_t*>(st.online_v);
        ap.online_pages = u32(st.online_pages);
        SHELF_RETURN_IF_ERROR(hipok(launch_kv_append_bf16(
                                        ap.common, ap.k_pool, ap.v_pool, stream_,
                                        ap.online_k, ap.online_v, ap.online_pages),
                                    "kv_append"));
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        PagedAttentionCommonArgs at{};
        at.q = static_cast<const bf16_t*>(q->dev);
        at.output = o;
        at.output_dtype = PagedAttentionOutputDType::BF16;
        at.q_row_stride = u32(nq * D);
        at.output_row_stride = u32(nq * D);
        at.row_sequence_slots = rowslot_dev_;
        at.row_positions = rowpos_dev_;
        at.block_tables = st.table;
        at.block_table_stride = u32(st.pages);
        at.max_sequences = u32(max_sequences_);
        at.rows = u32(step_rows_);
        at.layer = 0u;
        at.page_tokens = u32(bs);
        at.num_pages = u32(st.total_pages != 0 ? st.total_pages : st.pages);
        at.num_attention_layers = 1u;
        at.q_heads = u32(nq);
        at.kv_heads = u32(kvh);
        at.head_dim = u32(D);
        at.elems_per_token = ept;
        at.elems_per_page = epp;
        at.elems_per_layer = epl;
        at.scale = p.attr.score_scale != 0.0f ? p.attr.score_scale
                                              : 1.0f / std::sqrt(static_cast<float>(D));
        at.slot_window_end = slot_window_dev_;
        // 複数行（prefill）で受理条件を満たすならタイル化 prefill kernel。
        // 1 行（decode）は従来どおり（grid が 1 CTA に直列化して遅くなる）。
        // `SHELF_ATTN_PREFILL=0` で per-query 経路に固定できる（数値の A/B 用）。
        // prefill は既定で転置コア版（S^T = K·Q^T）。`SHELF_ATTN_PREFILL_ST=0` で
        // 従来の S = Q·K^T 版に戻せる（A/B と切り分け用）。
        const char* attn_env = std::getenv("SHELF_ATTN_PREFILL");
        const bool st_env = [] {
            const char* e = std::getenv("SHELF_ATTN_PREFILL_ST");
            return e == nullptr || e[0] != '0';
        }();
        // 複数スロットのバッチでは prefill 用タイルカーネルを使わない
        // （あちらは 1 タイル = 1 スロット前提で row0 の slot を共有する）。
        // decode 用カーネルは行ごとに slot / 位置 / ブロック表を引く。
        const bool multi_slot = rows_multi_slot_;
        const bool st_path = !multi_slot && st_env &&
                             attention_paged_prefill_st_bf16_supported(at);
        const bool prefill_path = !multi_slot && attention_paged_prefill_bf16_supported(at) &&
                                  (attn_env == nullptr || attn_env[0] != '0');
        {
            const hipError_t ea = st_path
                ? launch_attention_paged_prefill_st_bf16(
                      at, static_cast<const bf16_t*>(st.k_pool),
                      static_cast<const bf16_t*>(st.v_pool), stream_,
                      static_cast<const bf16_t*>(st.online_k),
                      static_cast<const bf16_t*>(st.online_v), u32(st.online_pages))
                : (prefill_path
                      ? launch_attention_paged_prefill_bf16(
                            at, static_cast<const bf16_t*>(st.k_pool),
                            static_cast<const bf16_t*>(st.v_pool), stream_,
                            static_cast<const bf16_t*>(st.online_k),
                            static_cast<const bf16_t*>(st.online_v), u32(st.online_pages))
                      : launch_attention_paged_bf16(
                            at, static_cast<const bf16_t*>(st.k_pool),
                            static_cast<const bf16_t*>(st.v_pool), stream_,
                            static_cast<const bf16_t*>(st.online_k),
                            static_cast<const bf16_t*>(st.online_v), u32(st.online_pages)));
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
        st.rows = std::max<std::size_t>(st.rows, cur_max_pos_);
        return set_out(false, o);
    }
    case K::moe_mlp: {
        // 疎な MoE ブロック。ルーティングはホスト（top-k の決定は enqueue と
        // 同時刻に固定する必要があるため、この行の先頭で決める）、行列積はデバイス。
        ValueLoc* x = in(0);
        SHELF_RETURN_IF_ERROR(need(x, false, "moe_mlp"));
        if (p.output.shape.rank() != 2)
            return Status::invalid_state("moe_mlp: output must be rank 2", __FILE__, __LINE__);
        const std::size_t H = p.output.shape.dim(1);
        if (H == 0u || out_n % H != 0u)
            return Status::invalid_state("moe_mlp: output width", __FILE__, __LINE__);
        const std::size_t rows = out_n / H;
        if (rows == 0u || x->elems != rows * H)
            return Status::invalid_state("moe_mlp: input rows", __FILE__, __LINE__);
        const auto rit = weight_shape_.find(p.weights[0].value());
        const auto git = weight_shape_.find(p.weights[1].value());
        if (rit == weight_shape_.end() || git == weight_shape_.end())
            return Status::invalid_state("moe_mlp: weight shape missing", __FILE__, __LINE__);
        const std::size_t E = rit->second.dim(0);
        const std::size_t I = git->second.dim(0);
        const bool shared = p.weights.size() == 3u * E + 5u;
        std::size_t Is = 0;
        if (shared) {
            const auto sit = weight_shape_.find(p.weights[3u * E + 2u].value());
            if (sit == weight_shape_.end())
                return Status::invalid_state("moe_mlp: shared shape missing", __FILE__, __LINE__);
            Is = sit->second.dim(0);
        }
        void* o = alloc(false);
        SHELF_RETURN_IF_ERROR(arena_ok(o));
        auto* xb = static_cast<const bf16_t*>(x->dev);
        auto* ob = static_cast<bf16_t*>(o);
        for (std::size_t r = 0; r < rows; ++r)
            SHELF_RETURN_IF_ERROR(
                run_moe_row(p, xb + r * H, ob + r * H, H, E, I, Is, shared));
        return set_out(false, o);
    }
    default:
        return Status::unsupported((std::string(ir::kind_name(p.kind)) +
                                    " not wired on hip executor").c_str(),
                                   __FILE__, __LINE__);
    }
}

} // namespace shelf::backend::hip
