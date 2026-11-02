#include <phaseshift/quantization/fp8/fp8.h>
#include <phaseshift/quantization/manifest/runtime_resolution.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ps::quantization;
using namespace ps::quantization::fp8;
using namespace ps::quantization::manifest;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static uint64_t align128(uint64_t k) {
    return ((k + 127u) / 128u) * 128u;
}

static void test_byte_counts() {
    for (uint64_t n : {1u, 127u, 128u, 129u, 256u}) {
        for (uint64_t k : {1u, 127u, 128u, 129u, 200u}) {
            const uint64_t kp = align128(k);
            const uint64_t sn = (n + 127u) / 128u;
            const uint64_t sk = kp / 128u;
            check(quant_codes_bytes(n, kp, kFp8FormatDesc) == n * kp,
                  "fp8 codes bytes n=" + std::to_string(n) + " k=" + std::to_string(k));
            check(quant_metadata1_bytes(n, kp, kFp8FormatDesc) == sn * sk * 4u,
                  "fp8 scale bytes n=" + std::to_string(n) + " k=" + std::to_string(k));
            // Batch geometry must not collapse N into the batch dimension.
            check(quant_metadata_bytes(3, n, kp, kFp8FormatDesc, 0) == 3u * sn * sk * 4u,
                  "fp8 batched scale bytes n=" + std::to_string(n));
        }
    }
}

static void test_scale_shape() {
    {
        Fp8BlockCanonicalStore store;
        store.init(1, 129, 129, 256);
        const Fp8BlockCanonicalView v = store.view();
        check(v.scale_n == 2, "fp8 n=129 scale_n == 2");
        check(v.scale_k == 2, "fp8 k=129 scale_k == 2");
        check(v.validate(), "fp8 view validate");
        check(store.codes.size() == 129u * 256u, "fp8 codes size");
        check(store.scales.size() == 4u, "fp8 scales size");
    }
    {
        Fp8BlockCanonicalStore store;
        store.init(1, 128, 127, 128);
        const Fp8BlockCanonicalView v = store.view();
        check(v.scale_n == 1, "fp8 n=128 scale_n == 1");
        check(v.scale_k == 1, "fp8 k=127 scale_k == 1");
        check(v.validate(), "fp8 view validate narrow");
    }
}

static void test_roundtrip() {
    for (uint64_t n : {1u, 127u, 128u, 129u, 300u}) {
        for (uint64_t k : {64u, 127u, 128u, 129u, 260u}) {
            const uint64_t kp = align128(k);
            std::vector<float> src(n * k);
            uint32_t state = 7u;
            for (auto& f : src) {
                state = state * 1664525u + 1013904223u;
                f = (static_cast<float>((state >> 8) & 0xFFFFu) / 65535.0f * 2.0f - 1.0f) * 9.0f;
            }
            Fp8BlockCanonicalStore store;
            store.init(1, n, k, kp);
            check(store.quantize_matrix(0, src.data(), k), "fp8 quantize_matrix");
            check(store.view().validate(), "fp8 roundtrip validate");

            std::vector<float> dst(n * kp, 0.0f);
            dequantize_fp8_canonical_matrix(store.view(), 0, dst.data(), kp);
            const uint64_t sn = (n + 127u) / 128u;
            const uint64_t sk = kp / 128u;
            for (uint64_t r = 0; r < n; ++r) {
                const uint64_t tb = r / 128u;
                for (uint64_t c = 0; c < kp; ++c) {
                    const float scale = store.scales[(tb * sk + c / 128u)];
                    const float want = (c < k) ? src[r * k + c] : 0.0f;
                    const float bound = std::max(0.08f * std::fabs(want), scale / 1024.0f);
                    check(std::fabs(dst[r * kp + c] - want) <= bound + 1e-9f,
                          "fp8 roundtrip n=" + std::to_string(n) + " k=" + std::to_string(k));
                }
            }
            (void)sn;
        }
    }
}

static void test_zero_tile() {
    const uint64_t n = 128, k = 128;
    std::vector<float> src(n * k, 0.0f);
    Fp8BlockCanonicalStore store;
    store.init(1, n, k, k);
    check(store.quantize_matrix(0, src.data(), k), "fp8 zero quantize");
    check(store.scales.size() == 1 && store.scales[0] == 0.0f, "fp8 zero scale");
    for (uint8_t c : store.codes) check(c == 0, "fp8 zero codes");
}

static void test_nonfinite() {
    const uint64_t n = 128, k = 128;
    std::vector<float> src(n * k, 1.0f);
    src[0] = std::nanf("");
    src[1] = INFINITY;
    src[2] = -INFINITY;
    Fp8BlockCanonicalStore store;
    store.init(1, n, k, k);
    check(store.quantize_matrix(0, src.data(), k), "fp8 nonfinite quantize");
    const float scale = 1.0f / kE4m3Max;
    check(store.scales[0] == scale, "fp8 nonfinite scale ignores nonfinite");
    check(store.codes[0] == 0, "fp8 nan code");
    check(store.codes[1] == 0, "fp8 inf code");
    check(store.codes[2] == 0, "fp8 -inf code");
}

static void test_matrix_batch() {
    const uint64_t batch = 2, n = 200, k = 128;
    const uint64_t kp = 128;
    Fp8BlockCanonicalStore store;
    store.init(batch, n, k, kp);
    std::vector<float> m0(n * k, 1.0f);
    std::vector<float> m1(n * k, 100.0f);
    check(store.quantize_matrix(0, m0.data(), k), "fp8 batch matrix 0");
    check(store.quantize_matrix(1, m1.data(), k), "fp8 batch matrix 1");
    const uint64_t sn = (n + 127u) / 128u;
    const uint64_t sk = kp / 128u;
    check(store.scales.size() == batch * sn * sk, "fp8 batch scale count");
    for (uint64_t tb = 0; tb < sn; ++tb) {
        check(store.scales[(0 * sn + tb) * sk] == 1.0f / kE4m3Max, "fp8 batch matrix 0 scale");
        check(store.scales[(1 * sn + tb) * sk] == 100.0f / kE4m3Max, "fp8 batch matrix 1 scale");
    }
    check(store.view().validate(), "fp8 batch validate");
}

static QuantizedTensorMetadata fp8_tensor() {
    QuantizedTensorMetadata t;
    t.encoding = QuantizedEncoding::Fp8;
    t.layout = kFp8LayoutName;
    t.logical_shape = {8, 128};
    t.k_padded = 128;
    t.codes = QuantizedTensorRef{"w.__phaseshift_codes", "U8", "0x00000000"};
    t.metadata1 = QuantizedTensorRef{"w.__phaseshift_metadata1", "F32", "0x00000000"};
    return t;
}

static void test_manifest_contract() {
    QuantizedManifest m;
    m.tensors["w"] = fp8_tensor();
    check(validate_quantized_manifest_contract(m).ok(), "fp8 manifest contract");
    auto resolved = resolve_quant_spec(m, m.tensors["w"]);
    check(resolved.ok(), "fp8 resolve");
    if (resolved.ok()) {
        check(resolved.value().spec_id == QuantSpecId::FP8_E4M3_F32_V1, "fp8 spec id");
        check(resolved.value().descriptor.group_size == 128 &&
                  resolved.value().descriptor.block_size == 128, "fp8 group size");
        check(resolved.value().descriptor.scale_type == StorageDType::FP32, "fp8 scale dtype");
    }
    check(preferred_gemm_compute_spec(QuantSpecId::FP8_E4M3_F32_V1).value() ==
              ComputeSpecId::FP8_W8A8_F32_BF16, "fp8 compute spec");

    QuantizedManifest bad_dtype = m;
    bad_dtype.tensors["w"].metadata1->dtype = "U8";
    check(!validate_quantized_manifest_contract(bad_dtype).ok(), "fp8 manifest rejects scale dtype");

    QuantizedManifest bad_k = m;
    bad_k.tensors["w"].k_padded = 96;
    check(!validate_quantized_manifest_contract(bad_k).ok(), "fp8 manifest rejects k_padded");

    QuantizedManifest bad_layout = m;
    bad_layout.tensors["w"].layout = "wrong";
    check(!validate_quantized_manifest_contract(bad_layout).ok(), "fp8 manifest rejects layout");

    QuantizedManifest bad_codes = m;
    bad_codes.tensors["w"].codes->dtype = "BF16";
    check(!validate_quantized_manifest_contract(bad_codes).ok(), "fp8 manifest rejects codes dtype");
}

static void test_encoding_names() {
    check(quantized_encoding_name(QuantizedEncoding::Fp8) == kFp8EncodingName,
          "fp8 encoding name");
    auto parsed = parse_quantized_encoding(kFp8EncodingName);
    check(parsed.ok() && parsed.value() == QuantizedEncoding::Fp8, "fp8 parse");
}

int main() {
    test_byte_counts();
    test_scale_shape();
    test_roundtrip();
    test_zero_tile();
    test_nonfinite();
    test_matrix_batch();
    test_manifest_contract();
    test_encoding_names();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
