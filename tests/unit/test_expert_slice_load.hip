#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/io/safetensors_reader.h>
#include <shelf/kernels/correctness/standalone/gemm_bf16_correctness.h>
#include <shelf/kernels/correctness/standalone/gemm_block_scaled_correctness.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/quantization/codec/e4m3.h>
#include <shelf/quantization/fp8/fp8_native.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>

#include "expert_slice_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace expert_slice_fixture;
using namespace ps;

static int g_fail = 0;

static void fail(const std::string& msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static std::vector<std::uint8_t> d2h_bytes(const void* dev, std::size_t bytes) {
    std::vector<std::uint8_t> host(bytes);
    if (bytes == 0) return host;
    if (hipMemcpy(host.data(), dev, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        fail("d2h_bytes");
        host.assign(bytes, 0);
    }
    return host;
}

static std::vector<std::uint16_t> d2h_u16(const void* dev, std::size_t count) {
    std::vector<std::uint16_t> host(count);
    if (count == 0) return host;
    if (hipMemcpy(host.data(), dev, count * sizeof(std::uint16_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        fail("d2h_u16");
        host.assign(count, 0);
    }
    return host;
}

static bf16_t* alloc_bf16(gpu::GpuArena& arena, std::size_t count, const std::string& tag) {
    auto v = arena.allocate_aligned(count * sizeof(bf16_t), 16);
    if (!v.ok()) {
        fail(tag + " alloc");
        return nullptr;
    }
    return static_cast<bf16_t*>(v.release().data());
}

struct ChainOut {
    std::vector<std::uint16_t> mid;
    std::vector<std::uint16_t> out;
};

static ChainOut run_bf16_chain(gpu::GpuArena& arena, hipStream_t stream,
                               const weights::MatrixWeight& wg,
                               const weights::MatrixWeight& wd,
                               const std::vector<std::uint16_t>& input_bf16) {
    ChainOut r;
    bf16_t* x = alloc_bf16(arena, std::size_t{kTokens} * kHidden, "bf16 chain x");
    bf16_t* y1 = alloc_bf16(arena, std::size_t{kTokens} * kGateUpRows, "bf16 chain y1");
    bf16_t* act = alloc_bf16(arena, std::size_t{kTokens} * kInter, "bf16 chain act");
    bf16_t* y3 = alloc_bf16(arena, std::size_t{kTokens} * kHidden, "bf16 chain y3");
    if (x == nullptr || y1 == nullptr || act == nullptr || y3 == nullptr) return r;

    check(hipMemcpy(x, input_bf16.data(), input_bf16.size() * sizeof(std::uint16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "bf16 chain h2d");
    check(kernel::detail::launch_gemm_bf16_baseline(
              wg.bf16.data<bf16_t>(), x, y1, kGateUpRows, kTokens, kHidden, stream) == hipSuccess,
          "bf16 chain gemm1 launch");
    check(kernel::launch_swiglu_bf16(y1, kGateUpRows, y1 + kInter, kGateUpRows,
                                     act, kInter, kTokens, kInter, stream) == hipSuccess,
          "bf16 chain swiglu launch");
    check(kernel::detail::launch_gemm_bf16_baseline(
              wd.bf16.data<bf16_t>(), act, y3, kHidden, kTokens, kInter, stream) == hipSuccess,
          "bf16 chain gemm2 launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "bf16 chain sync");
    r.mid = d2h_u16(y1, std::size_t{kTokens} * kGateUpRows);
    r.out = d2h_u16(y3, std::size_t{kTokens} * kHidden);
    return r;
}

static kernel::detail::BlockScaledWeightView fp8_weight_view(const weights::MatrixWeight& w) {
    kernel::detail::BlockScaledWeightView v;
    v.kind = kernel::detail::BlockScaledKind::Fp8;
    v.codes = static_cast<const std::uint8_t*>(w.codes.data<void>());
    v.scales = w.scales.data<void>();
    v.out_features = w.rows;
    v.k = w.cols;
    v.k_padded = w.k_padded;
    return v;
}

static ChainOut run_fp8_chain(gpu::GpuArena& arena, hipStream_t stream,
                              const weights::MatrixWeight& wg,
                              const weights::MatrixWeight& wd,
                              const std::vector<std::uint16_t>& input_bf16) {
    ChainOut r;
    bf16_t* x = alloc_bf16(arena, std::size_t{kTokens} * kHidden, "fp8 chain x");
    bf16_t* y1 = alloc_bf16(arena, std::size_t{kTokens} * kGateUpRows, "fp8 chain y1");
    bf16_t* act = alloc_bf16(arena, std::size_t{kTokens} * kInter, "fp8 chain act");
    bf16_t* y3 = alloc_bf16(arena, std::size_t{kTokens} * kHidden, "fp8 chain y3");
    if (x == nullptr || y1 == nullptr || act == nullptr || y3 == nullptr) return r;

    check(hipMemcpy(x, input_bf16.data(), input_bf16.size() * sizeof(std::uint16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "fp8 chain h2d");
    check(kernel::detail::launch_gemm_block_scaled_reference(fp8_weight_view(wg), x, y1, kTokens, stream) ==
              hipSuccess,
          "fp8 chain gemm1 launch");
    check(kernel::launch_swiglu_bf16(y1, kGateUpRows, y1 + kInter, kGateUpRows,
                                     act, kInter, kTokens, kInter, stream) == hipSuccess,
          "fp8 chain swiglu launch");
    check(kernel::detail::launch_gemm_block_scaled_reference(fp8_weight_view(wd), act, y3, kTokens, stream) ==
              hipSuccess,
          "fp8 chain gemm2 launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "fp8 chain sync");
    r.mid = d2h_u16(y1, std::size_t{kTokens} * kGateUpRows);
    r.out = d2h_u16(y3, std::size_t{kTokens} * kHidden);
    return r;
}

static void swiglu_oracle(const std::vector<std::uint16_t>& mid_bits,
                          std::vector<float>& act) {
    for (std::uint32_t t = 0; t < kTokens; ++t) {
        for (std::uint32_t i = 0; i < kInter; ++i) {
            const float gate = bf16_to_f32(mid_bits[static_cast<std::size_t>(t) * kGateUpRows + i]);
            const float up =
                bf16_to_f32(mid_bits[static_cast<std::size_t>(t) * kGateUpRows + kInter + i]);
            const float silu = gate / (1.0f + std::exp(-gate));
            act[static_cast<std::size_t>(t) * kInter + i] =
                bf16_to_f32(f32_to_bf16(up * silu));
        }
    }
}

static void bf16_gemm_oracle(const std::vector<float>& weight, std::uint32_t n, std::uint32_t k,
                             const std::vector<float>& x, std::vector<std::uint16_t>& y_bits) {
    for (std::uint32_t t = 0; t < kTokens; ++t) {
        const float* xrow = x.data() + static_cast<std::size_t>(t) * k;
        for (std::uint32_t m = 0; m < n; ++m) {
            const float* wrow = weight.data() + static_cast<std::size_t>(m) * k;
            float acc = 0.0f;
            for (std::uint32_t ki = 0; ki < k; ++ki) {
                acc = std::fmaf(wrow[ki], xrow[ki], acc);
            }
            y_bits[static_cast<std::size_t>(t) * n + m] = f32_to_bf16(acc);
        }
    }
}

static void fp8_gemm_oracle(const std::uint8_t* codes, const std::vector<float>& scales,
                            std::uint32_t n, std::uint32_t k,
                            const std::vector<float>& x,
                            std::vector<std::uint16_t>& y_bits) {
    const std::uint32_t scale_k = k / kScaleBlockK;
    std::vector<std::uint8_t> a_codes(k);
    for (std::uint32_t t = 0; t < kTokens; ++t) {
        const float* xrow = x.data() + static_cast<std::size_t>(t) * k;
        float peak = 0.0f;
        for (std::uint32_t i = 0; i < k; ++i) peak = std::max(peak, std::fabs(xrow[i]));
        const float ascale = peak > 0.0f ? peak / 448.0f : 0.0f;
        for (std::uint32_t i = 0; i < k; ++i) {
            a_codes[i] = ascale > 0.0f
                ? quantization::codec::e4m3_encode_u8(xrow[i] / ascale)
                : static_cast<std::uint8_t>(0);
        }
        for (std::uint32_t m = 0; m < n; ++m) {
            const std::uint8_t* wrow = codes + static_cast<std::size_t>(m) * k;
            const float* srow = scales.data() + static_cast<std::size_t>(m / kScaleBlockN) * scale_k;
            float acc = 0.0f;
            for (std::uint32_t kb = 0; kb < scale_k; ++kb) {
                float p = 0.0f;
                const std::uint32_t base = kb * kScaleBlockK;
                for (std::uint32_t j = 0; j < kScaleBlockK; ++j) {
                    p = std::fmaf(quantization::codec::e4m3_decode_f32(wrow[base + j]),
                                  quantization::codec::e4m3_decode_f32(a_codes[base + j]), p);
                }
                acc += p * srow[kb];
            }
            y_bits[static_cast<std::size_t>(t) * n + m] = f32_to_bf16(acc * ascale);
        }
    }
}

static ChainOut oracle_bf16_chain(const Fixture& f, std::uint32_t e,
                                  const std::vector<float>& input) {
    const std::vector<float> wg = bf16_region_f32(
        f.gate_up_bf16, gate_up_expert_elements(),
        static_cast<std::uint64_t>(e) * gate_up_expert_elements());
    const std::vector<float> wd = bf16_region_f32(
        f.down_bf16, down_expert_elements(),
        static_cast<std::uint64_t>(e) * down_expert_elements());
    ChainOut r;
    r.mid.resize(std::size_t{kTokens} * kGateUpRows);
    bf16_gemm_oracle(wg, kGateUpRows, kHidden, input, r.mid);
    std::vector<float> act(std::size_t{kTokens} * kInter);
    swiglu_oracle(r.mid, act);
    r.out.resize(std::size_t{kTokens} * kHidden);
    bf16_gemm_oracle(wd, kDownRows, kInter, act, r.out);
    return r;
}

static ChainOut oracle_fp8_chain(const Fixture& f, std::uint32_t e,
                                 const std::vector<float>& input) {
    const std::vector<float> wg_scales = scale_region_f32(
        f.gate_up_scales, gate_up_scale_elements(),
        static_cast<std::uint64_t>(e) * gate_up_scale_elements());
    const std::vector<float> wd_scales = scale_region_f32(
        f.down_scales, down_scale_elements(),
        static_cast<std::uint64_t>(e) * down_scale_elements());
    const std::uint8_t* wg_codes =
        f.gate_up_codes.data() + static_cast<std::uint64_t>(e) * gate_up_expert_elements();
    const std::uint8_t* wd_codes =
        f.down_codes.data() + static_cast<std::uint64_t>(e) * down_expert_elements();

    ChainOut r;
    r.mid.resize(std::size_t{kTokens} * kGateUpRows);
    fp8_gemm_oracle(wg_codes, wg_scales, kGateUpRows, kHidden, input, r.mid);
    std::vector<float> act(std::size_t{kTokens} * kInter);
    swiglu_oracle(r.mid, act);
    r.out.resize(std::size_t{kTokens} * kHidden);
    fp8_gemm_oracle(wd_codes, wd_scales, kDownRows, kInter, act, r.out);
    return r;
}

static void compare_bits(const std::string& tag, const std::vector<std::uint16_t>& got,
                         const std::vector<std::uint16_t>& want) {
    if (got.size() != want.size()) {
        fail(tag + " size");
        return;
    }
    float max_abs = 0.0f;
    float ref_scale = 0.0f;
    std::size_t argmax = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float g = bf16_to_f32(got[i]);
        const float w = bf16_to_f32(want[i]);
        if (std::fabs(g - w) > max_abs) {
            max_abs = std::fabs(g - w);
            argmax = i;
        }
        ref_scale = std::max(ref_scale, std::fabs(w));
    }
    const float tol = 0.02f * ref_scale + 1e-4f;
    std::printf("  %s: max_abs=%.6f ref_scale=%.4f tol=%.6f idx=%zu got=%.6f want=%.6f\n",
                tag.c_str(), max_abs, ref_scale, tol, argmax,
                bf16_to_f32(got[argmax]), bf16_to_f32(want[argmax]));
    check(max_abs <= tol, tag + " within tolerance");
}

static void compare_chain(const std::string& tag, const ChainOut& got, const ChainOut& want) {
    compare_bits(tag + " mid", got.mid, want.mid);
    compare_bits(tag + " out", got.out, want.out);
}

static void check_fp8_geometry(const weights::MatrixWeight& w, std::uint32_t rows,
                               std::uint32_t k, const std::string& tag) {
    check(w.encoding == weights::MatrixEncoding::Fp8, tag + " encoding");
    check(w.rows == rows && w.cols == k && w.k_padded == k, tag + " geometry");
    check(w.weight_scale_group == kScaleBlockK, tag + " weight scale group");
    check(w.scales.ndim() == 2 && w.scales.dim(0) == rows / kScaleBlockN &&
              w.scales.dim(1) == k / kScaleBlockK,
          tag + " scale grid shape");
}

static void test_bf16_experts(gpu::GpuArena& arena, hipStream_t stream, const Fixture& f,
                              const weights::SafetensorsCollection& col) {
    const std::vector<float> input = bf16_region_f32(
        f.input_bf16, static_cast<std::uint64_t>(kTokens) * kHidden, 0);
    const std::uint64_t slice_bytes = (gate_up_expert_elements() + down_expert_elements()) * 2;
    for (std::uint32_t e : {0u, 2u, kExperts - 1}) {
        const std::string tag = "bf16 e=" + std::to_string(e);
        const std::size_t used_before = arena.used();
        auto wg_result = weights::load_bf16_matrix_axis_slice(col, kGateUpName, e, 1, arena, stream);
        if (!wg_result.ok()) fail(tag + " gate load: " + wg_result.status().message());
        auto wd_result = weights::load_bf16_matrix_axis_slice(col, kDownName, e, 1, arena, stream);
        if (!wd_result.ok()) fail(tag + " down load: " + wd_result.status().message());
        if (!wg_result.ok() || !wd_result.ok()) continue;
        const weights::MatrixWeight& wg = wg_result.value();
        const weights::MatrixWeight& wd = wd_result.value();

        check(wg.encoding == weights::MatrixEncoding::Bf16, tag + " gate encoding");
        check(wg.rows == kGateUpRows && wg.cols == kHidden && wg.k_padded == kHidden,
              tag + " gate geometry");
        check(wd.rows == kDownRows && wd.cols == kInter && wd.k_padded == kInter,
              tag + " down geometry");

        const std::size_t used = arena.used() - used_before;
        check(used == slice_bytes, tag + " arena bytes is the slice only");
        check(used < f.gate_up_bf16.size() + f.down_bf16.size(),
              tag + " arena bytes below full fused tensors");

        const std::vector<std::uint8_t> got_gate =
            d2h_bytes(wg.bf16.data<void>(), gate_up_expert_elements() * 2);
        const std::size_t gate_off = static_cast<std::size_t>(e) * gate_up_expert_elements() * 2;
        check(std::equal(got_gate.begin(), got_gate.end(), f.gate_up_bf16.begin() + gate_off),
              tag + " gate bytes");
        const std::vector<std::uint8_t> got_down =
            d2h_bytes(wd.bf16.data<void>(), down_expert_elements() * 2);
        const std::size_t down_off = static_cast<std::size_t>(e) * down_expert_elements() * 2;
        check(std::equal(got_down.begin(), got_down.end(), f.down_bf16.begin() + down_off),
              tag + " down bytes");

        const ChainOut got = run_bf16_chain(arena, stream, wg, wd, f.input_bf16);
        const ChainOut want = oracle_bf16_chain(f, e, input);
        compare_chain(tag, got, want);
    }
    check(!weights::load_bf16_matrix_axis_slice(col, kGateUpName, kExperts, 1, arena, stream).ok(),
          "bf16 invalid expert rejected");
    check(!weights::load_bf16_matrix_axis_slice(col, kGateUpName, 0, 0, arena, stream).ok(),
          "bf16 empty range rejected");
}

static quantization::fp8::Fp8NativeHost preshuffle_expert_reference(const Fixture& f,
                                                                    std::uint32_t e) {
    quantization::fp8::Fp8BlockCanonicalView cv;
    cv.batch = 1;
    cv.n = kGateUpRows;
    cv.padded_k = kHidden;
    cv.codes = f.gate_up_codes.data() + static_cast<std::uint64_t>(e) * gate_up_expert_elements();
    cv.codes_bytes = gate_up_expert_elements();
    const std::vector<float> scales = scale_region_f32(
        f.gate_up_scales, gate_up_scale_elements(),
        static_cast<std::uint64_t>(e) * gate_up_scale_elements());
    cv.scales = scales.data();
    cv.scales_bytes = scales.size() * sizeof(float);
    cv.scale_n = kGateUpRows / kScaleBlockN;
    cv.scale_k = kHidden / kScaleBlockK;
    cv.scale_block_k = kScaleBlockK;

    quantization::fp8::Fp8NativeHost native;
    const bool ok = quantization::fp8::preshuffle_fp8_native(cv, native);
    check(ok, "preshuffle reference accepts expert");
    return native;
}

static void test_fp8_experts(gpu::GpuArena& arena, hipStream_t stream, const Fixture& f,
                             const weights::SafetensorsCollection& col) {
    weights::WeightLoadOptions canonical_options;
    canonical_options.preshuffle = false;
    const std::vector<float> input = bf16_region_f32(
        f.input_bf16, static_cast<std::uint64_t>(kTokens) * kHidden, 0);
    const std::uint64_t slice_bytes =
        gate_up_expert_elements() + down_expert_elements() +
        (gate_up_scale_elements() + down_scale_elements()) * sizeof(float);
    for (std::uint32_t e : {0u, 2u, kExperts - 1}) {
        const std::string tag = "fp8 e=" + std::to_string(e);
        const std::size_t used_before = arena.used();
        auto wg_result = weights::load_fp8_matrix_axis_slice(
            col, kGateUpName, kGateUpScaleName, e, 1, arena, stream, canonical_options);
        if (!wg_result.ok()) fail(tag + " gate load: " + wg_result.status().message());
        auto wd_result = weights::load_fp8_matrix_axis_slice(
            col, kDownName, kDownScaleName, e, 1, arena, stream, canonical_options);
        if (!wd_result.ok()) fail(tag + " down load: " + wd_result.status().message());
        if (!wg_result.ok() || !wd_result.ok()) continue;
        const weights::MatrixWeight& wg = wg_result.value();
        const weights::MatrixWeight& wd = wd_result.value();

        check_fp8_geometry(wg, kGateUpRows, kHidden, tag + " gate");
        check_fp8_geometry(wd, kDownRows, kInter, tag + " down");
        check(!wg.preshuffled && !wd.preshuffled, tag + " canonical storage");

        const std::size_t used = arena.used() - used_before;
        check(used == slice_bytes, tag + " arena bytes is the slice only");
        check(used < (f.gate_up_codes.size() + f.down_codes.size()) / kExperts * (kExperts - 1),
              tag + " arena bytes below full fused codes");

        const std::vector<std::uint8_t> got_codes =
            d2h_bytes(wg.codes.data<void>(), gate_up_expert_elements());
        const std::size_t codes_off = static_cast<std::size_t>(e) * gate_up_expert_elements();
        check(std::equal(got_codes.begin(), got_codes.end(), f.gate_up_codes.begin() + codes_off),
              tag + " gate codes bytes");
        const std::vector<float> want_scales = scale_region_f32(
            f.gate_up_scales, gate_up_scale_elements(),
            static_cast<std::uint64_t>(e) * gate_up_scale_elements());
        const std::vector<std::uint8_t> got_scale_bytes =
            d2h_bytes(wg.scales.data<void>(), want_scales.size() * sizeof(float));
        float max_scale_diff = 0.0f;
        for (std::size_t i = 0; i < want_scales.size(); ++i) {
            float got_scale = 0.0f;
            std::memcpy(&got_scale, got_scale_bytes.data() + i * sizeof(float), sizeof(float));
            max_scale_diff = std::max(max_scale_diff, std::fabs(got_scale - want_scales[i]));
        }
        check(max_scale_diff == 0.0f, tag + " gate scales exact");

        auto wg_native_result = weights::load_fp8_matrix_axis_slice(
            col, kGateUpName, kGateUpScaleName, e, 1, arena, stream);
        check(wg_native_result.ok(), tag + " gate native load");
        if (wg_native_result.ok()) {
            const weights::MatrixWeight& wg_native = wg_native_result.value();
            check(wg_native.preshuffled, tag + " native preshuffled flag");
            const quantization::fp8::Fp8NativeHost native = preshuffle_expert_reference(f, e);
            const std::vector<std::uint8_t> got_native =
                d2h_bytes(wg_native.codes.data<void>(), native.codes_bytes);
            check(std::equal(got_native.begin(), got_native.end(), native.codes.get()),
                  tag + " native codes match preshuffle");
            const std::vector<std::uint8_t> got_native_scales =
                d2h_bytes(wg_native.scales.data<void>(), native.scale_count * sizeof(float));
            check(std::equal(got_native_scales.begin(), got_native_scales.end(),
                             reinterpret_cast<const std::uint8_t*>(native.scales.get())),
                  tag + " native scales match preshuffle");
        }

        const ChainOut got = run_fp8_chain(arena, stream, wg, wd, f.input_bf16);
        const ChainOut want = oracle_fp8_chain(f, e, input);
        compare_chain(tag, got, want);
    }
    check(!weights::load_fp8_matrix_axis_slice(col, kGateUpName, kGateUpScaleName, kExperts, 1,
                                               arena, stream)
               .ok(),
          "fp8 invalid expert rejected");
    check(!weights::load_fp8_matrix_axis_slice(col, kGateUpName, kGateUpScaleName, 0, 0, arena,
                                               stream)
               .ok(),
          "fp8 empty range rejected");
    check(!weights::load_fp8_matrix_axis_slice(col, kGateUpName, kDownScaleName, 0, 1, arena,
                                               stream)
               .ok(),
          "fp8 scale grid mismatch rejected");
}

int main() {
    std::printf("=== Expert axis-slice load (GPU) ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto ar = gpu::GpuArena::create(0, 512ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        fail("arena");
        return 1;
    }
    gpu::GpuArena arena = ar.release();

    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path(ec) / ("ps_expert_slice_gpu_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    const Fixture f = make_fixture();
    const fs::path bf16_dir = root / "bf16";
    const fs::path fp8_dir = root / "fp8";
    check(write_bf16_model_dir(bf16_dir, f), "bf16 fixture written");
    check(write_fp8_model_dir(fp8_dir, f), "fp8 fixture written");

    auto bf16_col = weights::SafetensorsCollection::open(bf16_dir.string());
    check(bf16_col.ok(), "bf16 collection open");
    auto fp8_col = weights::SafetensorsCollection::open(fp8_dir.string());
    check(fp8_col.ok(), "fp8 collection open");
    if (bf16_col.ok() && fp8_col.ok()) {
        test_bf16_experts(arena, stream, f, bf16_col.value());
        test_fp8_experts(arena, stream, f, fp8_col.value());
    }

    fs::remove_all(root, ec);
    arena.shutdown();
    gpu::discard_cleanup_result(hipStreamDestroy(stream));

    std::printf("test_expert_slice_load: %s (fail=%d)\n",
                g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
