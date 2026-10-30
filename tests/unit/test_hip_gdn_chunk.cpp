// shelf unit test — GDN のチャンク版（FP16 WMMA）が逐次版と一致すること。
//
// 同じ入力・同じ初期状態から
//   ① 逐次カーネル（1 行ずつ）
//   ② チャンクカーネル（kGdnChunk 行ずつ、末尾は端数チャンク）
// を回し、出力と最終状態を比べる。チャンク版は行列積を fp16 の断片で行うため
// ビット一致は望めない。**相対誤差が丸めの範囲**であることを見る。
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <shelf/backend/hip/kernels/gdn/gdn_chunk.h>
#include <shelf/backend/hip/kernels/gdn/gdn_recurrence.h>
#include <shelf/foundation/numeric.hpp>

namespace {

using shelf::bf16_t;
using shelf::hipk::GdnRecurrenceArgs;
using shelf::hipk::GdnRequestDescriptor;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

constexpr std::uint32_t kDK = 128u;
constexpr std::uint32_t kDV = 128u;

bf16_t to_bf16(float v) {
    return bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
}

struct DevBuf {
    void* p = nullptr;
    std::size_t bytes = 0;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() {
        if (p) (void)hipFree(p);
    }
    bool alloc(std::size_t n) {
        void* q = nullptr;
        if (hipMalloc(&q, n) != hipSuccess) return false;
        p = q;
        bytes = n;
        return true;
    }
    bool put(const void* src, std::size_t n) {
        return hipMemcpy(p, src, n, hipMemcpyHostToDevice) == hipSuccess;
    }
    bool fetch(void* dst, std::size_t n) const {
        return hipMemcpy(dst, p, n, hipMemcpyDeviceToHost) == hipSuccess;
    }
};

// 1 ケース分のホスト側データとデバイス側の複製。
struct Case {
    std::uint32_t rows = 0, nvh = 0, nkh = 0;
    std::vector<float> q, k, state, out;
    std::vector<bf16_t> v, a, b, dt_bias, a_log;

    DevBuf dq, dk_, dv_, da, db, ddt, dal, dstate, dout, dreq;

    bool upload() {
        const std::size_t qn = q.size() * sizeof(float);
        const std::size_t vn = v.size() * sizeof(bf16_t);
        const std::size_t an = a.size() * sizeof(bf16_t);
        if (!dq.alloc(qn) || !dk_.alloc(qn) || !dv_.alloc(vn) || !da.alloc(an) ||
            !db.alloc(an) || !ddt.alloc(nvh * sizeof(bf16_t)) ||
            !dal.alloc(nvh * sizeof(bf16_t)) || !dstate.alloc(state.size() * sizeof(float)) ||
            !dout.alloc(out.size() * sizeof(float)) ||
            !dreq.alloc(sizeof(GdnRequestDescriptor)))
            return false;
        return dq.put(q.data(), qn) && dk_.put(k.data(), qn) && dv_.put(v.data(), vn) &&
               da.put(a.data(), an) && db.put(b.data(), an) &&
               ddt.put(dt_bias.data(), nvh * sizeof(bf16_t)) &&
               dal.put(a_log.data(), nvh * sizeof(bf16_t)) &&
               dstate.put(state.data(), state.size() * sizeof(float));
    }
    bool set_request(std::uint32_t row_begin, std::uint32_t row_count) {
        const GdnRequestDescriptor d{0u, row_begin, row_count};
        return dreq.put(&d, sizeof d);
    }
    void download() {
        (void)dstate.fetch(state.data(), state.size() * sizeof(float));
        (void)dout.fetch(out.data(), out.size() * sizeof(float));
    }
    GdnRecurrenceArgs args() const {
        GdnRecurrenceArgs r;
        r.q = static_cast<const float*>(dq.p);
        r.k = static_cast<const float*>(dk_.p);
        r.v = static_cast<const bf16_t*>(dv_.p);
        r.a = static_cast<const bf16_t*>(da.p);
        r.b = static_cast<const bf16_t*>(db.p);
        r.dt_bias = static_cast<const bf16_t*>(ddt.p);
        r.a_log = static_cast<const bf16_t*>(dal.p);
        r.recurrent_state = static_cast<float*>(dstate.p);
        r.output = static_cast<float*>(dout.p);
        r.requests = static_cast<const GdnRequestDescriptor*>(dreq.p);
        r.num_requests = 1u;
        r.actual_rows = rows;
        r.q_row_stride = nkh * kDK;
        r.k_row_stride = nkh * kDK;
        r.v_row_stride = nvh * kDV;
        r.a_row_stride = nvh;
        r.b_row_stride = nvh;
        r.output_row_stride = nvh * kDV;
        r.recurrent_slot_stride = static_cast<std::uint64_t>(nvh) * kDK * kDV;
        r.recurrent_layer_stride = static_cast<std::uint64_t>(nvh) * kDK * kDV;
        r.state_index = 0u;
        r.max_sequences = 1u;
        r.key_heads = nkh;
        r.num_v_heads = nvh;
        r.head_k = kDK;
        r.head_v = kDV;
        return r;
    }
};

void fill(Case* c, std::uint32_t rows, std::uint32_t nvh, std::uint32_t nkh) {
    c->rows = rows;
    c->nvh = nvh;
    c->nkh = nkh;
    std::mt19937 rng(12345u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    c->q.assign(static_cast<std::size_t>(rows) * nkh * kDK, 0.0f);
    c->k.assign(c->q.size(), 0.0f);
    // q/k は qk_prepare 後（L2 正規化・q は 1/sqrt(dk) 倍）に相当する値。
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t h = 0; h < nkh; ++h) {
            const std::size_t base = (static_cast<std::size_t>(r) * nkh + h) * kDK;
            float nq = 0.0f, nk = 0.0f;
            for (std::uint32_t i = 0; i < kDK; ++i) {
                const float qv = nd(rng), kv = nd(rng);
                c->q[base + i] = qv;
                c->k[base + i] = kv;
                nq += qv * qv;
                nk += kv * kv;
            }
            const float sq = 1.0f / std::sqrt(std::max(nq, 1e-12f));
            const float sk = 1.0f / std::sqrt(std::max(nk, 1e-12f));
            for (std::uint32_t i = 0; i < kDK; ++i) {
                c->q[base + i] *= sq / std::sqrt(static_cast<float>(kDK));
                c->k[base + i] *= sk;
            }
        }
    }
    c->v.assign(static_cast<std::size_t>(rows) * nvh * kDV, bf16_t{0u});
    c->a.assign(static_cast<std::size_t>(rows) * nvh, bf16_t{0u});
    c->b.assign(c->a.size(), bf16_t{0u});
    for (auto& x : c->v) x = to_bf16(nd(rng));
    for (std::size_t i = 0; i < c->a.size(); ++i) {
        c->a[i] = to_bf16(nd(rng) * 0.5f);
        c->b[i] = to_bf16(nd(rng) * 0.8f);
    }
    c->dt_bias.assign(nvh, bf16_t{0u});
    c->a_log.assign(nvh, bf16_t{0u});
    for (std::uint32_t h = 0; h < nvh; ++h) {
        c->dt_bias[h] = to_bf16(-1.0f + 0.1f * static_cast<float>(h));
        c->a_log[h] = to_bf16(0.5f);
    }
    c->state.assign(static_cast<std::size_t>(nvh) * kDK * kDV, 0.0f);
    for (auto& x : c->state) x = nd(rng) * 0.1f;
    c->out.assign(static_cast<std::size_t>(rows) * nvh * kDV, 0.0f);
}

void diff(const std::vector<float>& a, const std::vector<float>& b, double* worst, double* mag) {
    *worst = 0.0;
    *mag = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        *worst = std::max(*worst, std::fabs(static_cast<double>(a[i]) - b[i]));
        *mag = std::max(*mag, std::fabs(static_cast<double>(b[i])));
    }
}

int run_case(std::uint32_t rows, std::uint32_t nvh, std::uint32_t nkh) {
    Case seq, chunk;
    fill(&seq, rows, nvh, nkh);
    fill(&chunk, rows, nvh, nkh);
    chunk.q = seq.q;
    chunk.k = seq.k;
    chunk.v = seq.v;
    chunk.a = seq.a;
    chunk.b = seq.b;
    chunk.state = seq.state;

    if (!seq.upload() || !chunk.upload()) {
        std::printf("FAIL: デバイスへの転送\n");
        return 1;
    }

    // ① 逐次（1 行ずつ）
    for (std::uint32_t t = 0; t < rows; ++t) {
        if (!seq.set_request(t, 1u)) return 1;
        const GdnRecurrenceArgs a = seq.args();
        const hipError_t e = shelf::hipk::launch_gdn_recurrence_f32(a, nullptr);
        if (e != hipSuccess) {
            std::printf("FAIL: 逐次カーネル: %s\n", hipGetErrorString(e));
            return 1;
        }
    }
    // ② チャンク（kGdnChunk 行ずつ）
    for (std::uint32_t done = 0; done < rows;) {
        const std::uint32_t n = std::min<std::uint32_t>(shelf::hipk::kGdnChunk, rows - done);
        if (!chunk.set_request(done, n)) return 1;
        const GdnRecurrenceArgs a = chunk.args();
        const hipError_t e = shelf::hipk::launch_gdn_chunk(a, done, n, nullptr);
        if (e != hipSuccess) {
            std::printf("FAIL: チャンクカーネル: %s\n", hipGetErrorString(e));
            return 1;
        }
        done += n;
    }
    {
        const hipError_t se = hipDeviceSynchronize();
        if (se != hipSuccess) {
            std::printf("FAIL: 同期: %s\n", hipGetErrorString(se));
            return 1;
        }
    }
    seq.download();
    chunk.download();

    double worst_out = 0.0, mag_out = 0.0, worst_st = 0.0, mag_st = 0.0;
    diff(seq.out, chunk.out, &worst_out, &mag_out);
    diff(seq.state, chunk.state, &worst_st, &mag_st);
    const double rel_out = mag_out > 0.0 ? worst_out / mag_out : 0.0;
    const double rel_st = mag_st > 0.0 ? worst_st / mag_st : 0.0;
    std::printf("  rows=%u nvh=%u nkh=%u  out: worst=%.3e rel=%.3e  state: worst=%.3e rel=%.3e\n",
                rows, nvh, nkh, worst_out, rel_out, worst_st, rel_st);
    check(rel_out < 2.0e-2, "出力が逐次版と丸めの範囲で一致");
    check(rel_st < 2.0e-2, "最終状態が逐次版と丸めの範囲で一致");
    return 0;
}

}  // namespace

// decode 専用 WMMA（1 行）を逐次 f32 版と比較する。
// exact = f16 hi/lo の 4 回 WMMA、lossy = hi のみ 1 回。受理境界も見る。
int run_decode1_case(std::uint32_t nvh, std::uint32_t nkh) {
    Case seq, ex, ls;
    fill(&seq, 1u, nvh, nkh);
    fill(&ex, 1u, nvh, nkh);
    fill(&ls, 1u, nvh, nkh);
    for (Case* c : {&ex, &ls}) {
        c->q = seq.q; c->k = seq.k; c->v = seq.v; c->a = seq.a; c->b = seq.b;
        c->state = seq.state;
    }
    if (!seq.upload() || !ex.upload() || !ls.upload()) {
        std::printf("FAIL: decode1: デバイスへの転送\n");
        return 1;
    }
    if (!seq.set_request(0u, 1u) || !ex.set_request(0u, 1u) || !ls.set_request(0u, 1u)) return 1;
    if (hipError_t e = launch_gdn_recurrence_f32(seq.args(), nullptr); e != hipSuccess) {
        std::printf("FAIL: decode1 逐次: %s\n", hipGetErrorString(e)); return 1;
    }
    // 受理境界: 1 行以外（row_count != 1）は不受理
    if (ex.set_request(0u, 2u) && shelf::hipk::gdn_recurrence_wmma_decode1_supported(ex.args())) {
        std::printf("FAIL: decode1 は row_count=2 を拒否すべき\n");
        ++failures;
    }
    if (!ex.set_request(0u, 1u)) return 1;
    if (!shelf::hipk::gdn_recurrence_wmma_decode1_supported(ex.args())) {
        std::printf("FAIL: decode1 受理条件（rows=1・head_k=128・head_v の 64 倍）\n");
        ++failures;
        return 1;
    }
    if (hipError_t e = shelf::hipk::launch_gdn_recurrence_wmma_decode1(ex.args(), false, nullptr);
        e != hipSuccess) {
        std::printf("FAIL: decode1 exact: %s\n", hipGetErrorString(e)); return 1;
    }
    if (hipError_t e = shelf::hipk::launch_gdn_recurrence_wmma_decode1(ls.args(), true, nullptr);
        e != hipSuccess) {
        std::printf("FAIL: decode1 lossy: %s\n", hipGetErrorString(e)); return 1;
    }
    if (hipError_t se = hipDeviceSynchronize(); se != hipSuccess) {
        std::printf("FAIL: decode1 同期: %s\n", hipGetErrorString(se)); return 1;
    }
    seq.download(); ex.download(); ls.download();
    double wo = 0.0, mo = 0.0, ws = 0.0, ms = 0.0, wo2 = 0.0, mo2 = 0.0, ws2 = 0.0, ms2 = 0.0;
    diff(seq.out, ex.out, &wo, &mo); diff(seq.state, ex.state, &ws, &ms);
    diff(seq.out, ls.out, &wo2, &mo2); diff(seq.state, ls.state, &ws2, &ms2);
    std::printf("  decode1 nvh=%u nkh=%u exact: out rel=%.3e state rel=%.3e | lossy: out rel=%.3e "
                "state rel=%.3e\n", nvh, nkh, wo / std::max(mo, 1e-9), ws / std::max(ms, 1e-9),
                wo2 / std::max(mo2, 1e-9), ws2 / std::max(ms2, 1e-9));
    check(wo / std::max(mo, 1e-9) < 2.0e-2, "decode1 exact: 出力が逐次版と範囲内");
    check(ws / std::max(ms, 1e-9) < 2.0e-2, "decode1 exact: 状態が逐次版と範囲内");
    check(wo2 / std::max(mo2, 1e-9) < 2.0e-2, "decode1 lossy: 出力が逐次版と範囲内");
    check(ws2 / std::max(ms2, 1e-9) < 2.0e-2, "decode1 lossy: 状態が逐次版と範囲内");
    check(wo / std::max(mo, 1e-9) <= wo2 / std::max(mo2, 1e-9) + 1e-12,
          "decode1: exact が lossy 以下の誤差");
    return 0;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const hipError_t dev = hipSetDevice(0);
    if (dev != hipSuccess) {
        std::printf("FAIL: hipSetDevice: %s\n", hipGetErrorString(dev));
        return 1;
    }
    if (run_case(32u, 4u, 2u) != 0) return 1;
    if (run_case(64u, 4u, 2u) != 0) return 1;
    if (run_case(20u, 2u, 1u) != 0) return 1;   // 端数チャンク（rows < kGdnChunk）
    if (run_case(128u, 4u, 2u) != 0) return 1;  // 4 チャンク（誤差の蓄積を見る）
    if (run_case(256u, 4u, 2u) != 0) return 1;  // 8 チャンク
    if (run_decode1_case(4u, 2u) != 0) return 1;    // 1 行・head_v 128（WMMA 2 workgroup）
    if (run_decode1_case(2u, 2u) != 0) return 1;    // repeat=1
    std::printf("test_hip_gdn_chunk: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
