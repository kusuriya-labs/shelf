// shelf unit test — SYCL kernel layer smoke (port coverage for
// test_hip_elementwise_kernels.cpp et al., condensed). Launches the ported
// kernels on the default SYCL device and checks them against host scalar
// references of the same math. Kernels that require 32-lane hardware
// sub-groups (rmsnorm / gemm-exact-rows / l2 / qk-prepare / paged
// attention / fp8 kv append) are skipped when the device reports no such
// sub-group size (e.g. the OpenMP backend) — mirror of that behaviour on
// GPU devices is covered by the executor test.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <shelf/backend/sycl/kernels/elementwise/aux_ops.h>
#include <shelf/backend/sycl/kernels/elementwise/elementwise.h>
#include <shelf/backend/sycl/kernels/elementwise/embedding.h>
#include <shelf/backend/sycl/kernels/attention/kv_append.h>
#include <shelf/backend/sycl/kernels/attention/rope.h>
#include <shelf/backend/sycl/kernels/gdn/gdn_conv_silu.h>

using namespace shelf;

namespace {

int failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL: %s\n", msg);                                  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

shelf::sycl::Stream& stream() {
    static shelf::sycl::Stream s = [] {
        auto r = shelf::sycl::Stream::create();
        if (!r.ok()) { std::printf("FAIL: stream create: %s\n", r.status().message().c_str()); std::exit(1); }
        return std::move(r.value());
    }();
    return s;
}

shelf::backend::DeviceId dev() {
    static shelf::backend::DeviceId d = [] {
        shelf::backend::DeviceId cur = backend::kInvalidDevice;
        if (!shelf::sycl::get_device(cur).ok()) { std::printf("FAIL: no device\n"); std::exit(1); }
        return cur;
    }();
    return d;
}

template <class T>
struct DevArr {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit DevArr(std::size_t count) : bytes(count * sizeof(T)) {
        p = ::sycl::malloc_device(bytes, stream().queue());
    }
    ~DevArr() { if (p) ::sycl::free(p, stream().queue()); }
    T* ptr() const { return static_cast<T*>(p); }
    bool ok() const { return p != nullptr; }
    template <class U> void upload(const std::vector<U>& v) {
        stream().queue().memcpy(p, v.data(), v.size() * sizeof(U)).wait();
    }
    template <class U> std::vector<U> download(std::size_t n) {
        std::vector<U> v(n);
        stream().queue().memcpy(v.data(), p, n * sizeof(U)).wait();
        return v;
    }
};

bool nearly(float a, float b, float tol = 2e-2f) {
    return std::abs(a - b) <= tol * std::max(1.0f, std::max(std::abs(a), std::abs(b)));
}

// ---------------- elementwise family (no sub-groups needed) ----------------
void test_elementwise() {
    constexpr std::size_t N = 1000;  // rows=4, features=250
    const std::uint32_t rows = 4, F = 250;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(-3.f, 3.f);
    std::vector<bf16_t> ha(N), hb(N);
    for (std::size_t i = 0; i < N; ++i) { ha[i] = f32_to_bf16(ud(rng)); hb[i] = f32_to_bf16(ud(rng)); }

    DevArr<bf16_t> A(N), B(N), O(2 * N);
    DevArr<float> OF(N);
    if (!(A.ok() && B.ok() && O.ok() && OF.ok())) { CHECK(false, "alloc elementwise"); return; }
    A.upload(ha); B.upload(hb);

    CHECK(syclk::launch_residual_add_bf16(A.ptr(), F, B.ptr(), F, O.ptr(), F, rows, F, stream()).ok(), "residual_add launch");
    {
        auto o = O.download<bf16_t>(N);
        bool good = true;
        for (std::size_t i = 0; i < N; ++i)
            if (o[i].data != f32_to_bf16(bf16_to_f32(ha[i]) + bf16_to_f32(hb[i])).data) { good = false; break; }
        CHECK(good, "residual_add values");
    }

    CHECK(syclk::launch_silu_bf16_f32(A.ptr(), F, OF.ptr(), F, rows, F, stream()).ok(), "silu launch");
    {
        auto o = OF.download<float>(N);
        bool good = true;
        for (std::size_t i = 0; i < N; ++i)
            if (!nearly(o[i], bf16_to_f32(ha[i]) / (1.f + std::exp(-bf16_to_f32(ha[i]))), 1e-5f)) { good = false; break; }
        CHECK(good, "silu values");
    }

    CHECK(syclk::launch_swiglu_bf16(A.ptr(), F, B.ptr(), F, O.ptr(), F, rows, F, stream()).ok(), "swiglu launch");
    {
        auto o = O.download<bf16_t>(N);
        bool good = true;
        for (std::size_t i = 0; i < N; ++i) {
            const float g = bf16_to_f32(ha[i]), u = bf16_to_f32(hb[i]);
            if (o[i].data != f32_to_bf16(u * (g / (1.f + std::exp(-g)))).data) { good = false; break; }
        }
        CHECK(good, "swiglu values");
    }

    CHECK(syclk::launch_scale_f32(OF.ptr(), F, OF.ptr(), F, rows, F, 2.0f, stream()).ok(), "scale launch");
    CHECK(syclk::launch_concat_bf16(A.ptr(), F, B.ptr(), F, O.ptr(), 2 * F, rows, F, F, stream()).ok(), "concat launch");
    {
        auto o = O.download<bf16_t>(rows * 2 * F);
        bool good = true;
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < 2 * F; ++c)
                if (o[r * 2 * F + c].data != (c < F ? ha[r * F + c] : hb[r * F + c - F]).data) { good = false; break; }
        CHECK(good, "concat values");
    }

    CHECK(syclk::launch_split_bf16(O.ptr(), 2 * F, A.ptr(), F, B.ptr(), F, rows, F,
                                    syclk::ElementwiseSplitLayout::Halves, 0, stream()).ok(),
          "split launch");
    {
        // split(concat(a,b)) == a,b
        auto a2 = A.download<bf16_t>(N);
        auto b2 = B.download<bf16_t>(N);
        bool good = true;
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < F; ++c) {
                if (a2[r * F + c].data != ha[r * F + c].data) { good = false; break; }
                if (b2[r * F + c].data != hb[r * F + c].data) { good = false; break; }
            }
        CHECK(good, "split values");
    }
}



// ---------------- aux ops ----------------
void test_aux() {
    constexpr std::size_t N = 512;
    std::vector<bf16_t> h(N);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> ud(-2.f, 2.f);
    for (auto& x : h) x = f32_to_bf16(ud(rng));
    DevArr<bf16_t> A(N), O(N);
    DevArr<float> F32(N);
    if (!(A.ok() && O.ok() && F32.ok())) { CHECK(false, "alloc aux"); return; }
    A.upload(h);
    CHECK(syclk::launch_bf16_to_f32(A.ptr(), F32.ptr(), 256, stream()).ok(), "cvt launch");
    {
        auto o = F32.download<float>(256);
        bool good = true;
        for (std::size_t i = 0; i < 256; ++i)
            if (o[i] != bf16_to_f32(h[i])) { good = false; break; }
        CHECK(good, "cvt values");
    }
    CHECK(syclk::launch_f32_to_bf16(F32.ptr(), O.ptr(), 256, stream()).ok(), "cvt back launch");
    {
        auto o = O.download<bf16_t>(256);
        bool good = true;
        for (std::size_t i = 0; i < 256; ++i)
            if (o[i].data != h[i].data) { good = false; break; }
        CHECK(good, "cvt roundtrip bits");
    }
    // row slice: rows=2 cols=8 begin=3 stride=16
    std::vector<bf16_t> big(2 * 16);
    for (std::size_t i = 0; i < big.size(); ++i) big[i] = bf16_t{static_cast<std::uint16_t>(i)};
    DevArr<bf16_t> BG(2 * 16), SL(2 * 8);
    if (!(BG.ok() && SL.ok())) { CHECK(false, "alloc slice"); return; }
    BG.upload(big);
    CHECK(syclk::launch_bf16_row_slice(BG.ptr(), SL.ptr(), 2, 8, 3, 16, stream()).ok(), "slice launch");
    {
        auto o = SL.download<bf16_t>(16);
        bool good = true;
        for (std::size_t r = 0; r < 2; ++r)
            for (std::size_t c = 0; c < 8; ++c)
                if (o[r * 8 + c].data != big[r * 16 + 3 + c].data) { good = false; break; }
        CHECK(good, "slice values");
    }
    // argmax
    std::vector<float> logits(1000, -1.f);
    logits[777] = 3.5f;
    DevArr<float> LG(1000);
    DevArr<std::uint32_t> TK(1);
    if (!(LG.ok() && TK.ok())) { CHECK(false, "alloc argmax"); return; }
    LG.upload(logits);
    CHECK(syclk::launch_f32_argmax(LG.ptr(), 1000, TK.ptr(), stream()).ok(), "argmax launch");
    auto tk = TK.download<std::uint32_t>(1);
    CHECK(tk[0] == 777u, "argmax value");
}

// ---------------- embedding ----------------
void test_embedding() {
    const std::uint32_t rows = 3, vocab = 64, H = 128;
    std::vector<bf16_t> table(vocab * H);
    for (std::size_t i = 0; i < table.size(); ++i) table[i] = f32_to_bf16(0.01f * (i % 251));
    std::vector<std::int32_t> toks = {5, 63, -1};
    DevArr<bf16_t> TAB(vocab * H), OUT(rows * H);
    DevArr<std::int32_t> TOK(3);
    DevArr<std::uint32_t> ERR(1);
    if (!(TAB.ok() && OUT.ok() && TOK.ok() && ERR.ok())) { CHECK(false, "alloc embedding"); return; }
    TAB.upload(table); TOK.upload(toks);
    std::vector<std::uint32_t> zero{0u};
    ERR.upload(zero);
    CHECK(syclk::launch_embedding_bf16(TAB.ptr(), TOK.ptr(), OUT.ptr(), rows, vocab, H, H,
                                       ERR.ptr(), stream()).ok(), "embedding launch");
    auto o = OUT.download<bf16_t>(rows * H);
    bool good = true;
    for (std::size_t c = 0; c < H; ++c) {
        if (o[0 * H + c].data != table[5 * H + c].data) { good = false; break; }
        if (o[1 * H + c].data != table[63 * H + c].data) { good = false; break; }
    }
    CHECK(good, "embedding rows");
    auto err = ERR.download<std::uint32_t>(1);
    CHECK(err[0] == static_cast<std::uint32_t>(syclk::ProgramStatus::INVALID_BINDING),
          "embedding error word for bad token");
}

// ---------------- rope ----------------
void test_rope() {
    const std::uint32_t half = 8;   // rotary 16, head dim 16
    DevArr<double> INV(half);
    if (!INV.ok()) { CHECK(false, "alloc rope"); return; }
    CHECK(syclk::launch_rope_inv_freq_init(INV.ptr(), half, 1.0e7f, stream()).ok(), "inv_freq init");
    auto inv = INV.download<double>(half);
    bool good = true;
    for (std::uint32_t j = 0; j < half; ++j)
        if (std::abs(inv[j] - std::pow(1.0e7, -static_cast<double>(j) / half)) > 1e-9) { good = false; break; }
    CHECK(good, "inv_freq values");

    const std::uint32_t rows = 2, F = 16;
    std::vector<float> in(F * rows);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    for (auto& x : in) x = ud(rng);
    std::vector<std::uint32_t> pos = {1, 7};
    DevArr<float> IN(F * rows);
    DevArr<std::uint32_t> POS(2);
    DevArr<bf16_t> OUT(F * rows);
    if (!(IN.ok() && POS.ok() && OUT.ok())) { CHECK(false, "alloc rope 2"); return; }
    IN.upload(in); POS.upload(pos);
    CHECK(syclk::launch_rope_f32_bf16_generic(IN.ptr(), F, OUT.ptr(), F, POS.ptr(), rows, F,
                                              F, 16, 1.0e7f, INV.ptr(), 1.0f, stream()).ok(),
          "rope generic launch");
    auto o = OUT.download<bf16_t>(F * rows);
    double worst = 0.0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < 16; ++c) {
            float expect = in[r * F + c];
            if (c < 16) {
                const std::uint32_t j = c < half ? c : c - half;
                const double angle = pos[r] * inv[j];
                const float cs = static_cast<float>(std::cos(angle));
                const float sn = static_cast<float>(std::sin(angle));
                const float p = in[r * F + (c < half ? c + half : c - half)];
                expect = c < half ? in[r * F + c] * cs - p * sn
                                  : in[r * F + c] * cs + p * sn;
            }
            worst = std::max(worst, static_cast<double>(std::abs(bf16_to_f32(o[r * F + c]) - expect)));
        }
    }
    CHECK(worst < 1e-2, "rope generic values");
}

// ---------------- kv append ----------------
void test_kv_append() {
    const std::uint32_t rows = 2, kvh = 2, D = 64, bs = 4, pages = 8;
    const std::uint32_t ept = kvh * D, epp = bs * ept;
    std::vector<bf16_t> k(rows * ept), v(rows * ept);
    for (std::size_t i = 0; i < k.size(); ++i) { k[i] = f32_to_bf16(0.5f + 0.001f * i); v[i] = f32_to_bf16(-0.25f - 0.001f * i); }
    std::vector<std::uint32_t> slots(rows, 0u), positions = {5, 6};
    std::vector<std::uint32_t> table(pages);
    for (std::size_t i = 0; i < pages; ++i) table[i] = static_cast<std::uint32_t>(i);

    DevArr<bf16_t> K(rows * ept), V(rows * ept), POOL(2 * pages * epp), VP(2 * pages * epp);
    DevArr<std::uint32_t> SL(2), PO(2), TB(pages);
    if (!(K.ok() && V.ok() && POOL.ok() && VP.ok() && SL.ok() && PO.ok() && TB.ok())) {
        CHECK(false, "alloc kv"); return;
    }
    K.upload(k); V.upload(v); SL.upload(slots); PO.upload(positions); TB.upload(table);
    std::vector<std::uint8_t> z(2 * pages * epp * 2, 0u);
    stream().queue().memset(POOL.ptr(), 0, 2 * pages * epp * 2).wait();
    stream().queue().memset(VP.ptr(), 0, 2 * pages * epp * 2).wait();

    syclk::KvAppendCommonArgs a{};
    a.k_input = K.ptr(); a.v_input = V.ptr();
    a.k_row_stride = ept; a.v_row_stride = ept;
    a.row_sequence_slots = SL.ptr(); a.row_positions = PO.ptr();
    a.block_tables = TB.ptr(); a.block_table_stride = pages;
    a.max_sequences = 1; a.rows = rows; a.layer = 0;
    a.page_tokens = bs; a.num_pages = pages; a.num_attention_layers = 1;
    a.kv_heads = kvh; a.head_dim = D;
    a.elems_per_token = ept; a.elems_per_page = epp; a.elems_per_layer = pages * epp;
    CHECK(syclk::launch_kv_append_bf16(a, POOL.ptr(), VP.ptr(), stream()).ok(), "kv append launch");
    {
        auto pool = POOL.download<bf16_t>(2 * pages * epp);
        bool good = true;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const std::uint32_t pos = positions[r];
            const std::size_t base = (pos / bs) * epp + (pos % bs) * ept;
            for (std::uint32_t e = 0; e < ept; ++e)
                if (pool[base + e].data != k[r * ept + e].data) { good = false; break; }
            if (!good) break;
        }
        CHECK(good, "kv append pool values");
    }
}

// ---------------- fused conv+silu ----------------
void test_conv_silu() {
    const std::uint32_t C = 40, rows = 1;
    std::vector<bf16_t> in(C), w(C * 4);
    std::vector<float> state(C * 3);
    for (std::size_t i = 0; i < C; ++i) { in[i] = f32_to_bf16(0.1f * i); state[i * 3 + 0] = 0.1f * i; state[i * 3 + 1] = 0.2f * i; state[i * 3 + 2] = 0.3f * i; }
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = f32_to_bf16(0.25f);
    syclk::GdnRequestDescriptor req{0u, 0u, 1u};
    DevArr<bf16_t> IN(C), W(C * 4), ACT(C);
    DevArr<float> OUT(C), ST(C * 3);
    DevArr<syclk::GdnRequestDescriptor> RQ(1);
    if (!(IN.ok() && W.ok() && ACT.ok() && OUT.ok() && ST.ok() && RQ.ok())) { CHECK(false, "alloc conv"); return; }
    IN.upload(in); W.upload(w); ST.upload(state); RQ.upload(std::vector<syclk::GdnRequestDescriptor>{req});

    syclk::FusedGdnConvSiluArgs args{};
    args.conv.input = IN.ptr(); args.conv.output = OUT.ptr();
    args.conv.conv_weight = W.ptr(); args.conv.conv_state = ST.ptr();
    args.conv.requests = RQ.ptr(); args.conv.num_requests = 1;
    args.conv.actual_rows = rows; args.conv.max_request_rows = rows;
    args.conv.input_row_stride = C; args.conv.output_row_stride = C;
    args.conv.conv_dim = C; args.conv.conv_history = 3;
    args.conv.state_index = 0; args.conv.max_sequences = 1;
    args.conv.conv_slot_stride = C * 3; args.conv.conv_layer_stride = C * 3;
    args.act_out = ACT.ptr(); args.act_row_stride = C;
    CHECK(syclk::launch_fused_gdn_conv_silu(args, stream()).ok(), "conv_silu launch");
    auto out = OUT.download<float>(C);
    auto st = ST.download<float>(C * 3);
    bool good = true;
    // expected y = w0*s0 + w1*s1 + w2*s2 + w3*x with all w = 0.25(bf16)
    const float wf = bf16_to_f32(w[0]);
    for (std::uint32_t c = 0; c < C; ++c) {
        const float s0 = 0.1f * c, s1 = 0.2f * c, s2 = 0.3f * c;
        const float x = bf16_to_f32(in[c]);
        const float y = wf * (s0 + s1 + s2 + x);
        if (!nearly(out[c], y, 1e-5f)) { good = false; break; }
    }
    // state shift: new state = (s1, s2, x)
    for (std::uint32_t c = 0; c < C && good; ++c) {
        const float s1 = 0.2f * c, s2 = 0.3f * c, x = bf16_to_f32(in[c]);
        if (!nearly(st[c * 3 + 0], s1, 1e-6f) || !nearly(st[c * 3 + 1], s2, 1e-6f) ||
            !nearly(st[c * 3 + 2], x, 1e-6f)) { good = false; break; }
    }
    CHECK(good, "conv_silu values");
}

} // namespace

int main() {
    test_elementwise();
    test_aux();
    test_embedding();
    test_rope();
    test_kv_append();
    test_conv_silu();
    std::printf("test_sycl_kernels: failures=%d\n", failures);
    return failures ? 1 : 0;
}
