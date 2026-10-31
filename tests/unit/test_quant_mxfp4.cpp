// shelf unit test — quant/mxfp4（E2M1・E8M0 コーデックと断片順 relayout）。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <shelf/quant/mxfp4.hpp>

namespace {

using namespace shelf::quant;

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

// ---- E2M1 コーデック --------------------------------------------------------

void test_e2m1_table() {
    // 大きさフィールド 0..7 が {0, 0.5, 1, 1.5, 2, 3, 4, 6} であること。
    for (std::uint8_t i = 0; i < 8u; ++i) {
        check(mxfp4::e2m1_decode_nibble(i) == mxfp4::kE2M1Magnitudes[i], "E2M1 の大きさ");
        check(mxfp4::e2m1_decode_nibble(static_cast<std::uint8_t>(i | 0x08u)) ==
                  -mxfp4::kE2M1Magnitudes[i],
              "E2M1 の符号");
    }
    // 全 16 ニブルで decode -> encode が恒等（表現可能な値だから）。
    for (std::uint8_t n = 0; n < 16u; ++n) {
        check(mxfp4::e2m1_encode_nibble(mxfp4::e2m1_decode_nibble(n)) == n,
              "E2M1 の roundtrip");
    }
}

void test_e2m1_rounding() {
    // 最近傍（同距離は偶数側）と飽和。
    check(mxfp4::e2m1_encode_nibble(0.0f) == 0x0u, "0 -> 0");
    check(mxfp4::e2m1_encode_nibble(-0.0f) == 0x8u, "-0 -> 符号付き 0");
    check(mxfp4::e2m1_encode_nibble(0.24f) == 0x0u, "0.24 -> 0");
    check(mxfp4::e2m1_encode_nibble(0.26f) == 0x1u, "0.26 -> 0.5");
    check(mxfp4::e2m1_encode_nibble(0.25f) == 0x0u, "0.25（同距離）-> 0");
    check(mxfp4::e2m1_encode_nibble(0.75f) == 0x2u, "0.75（同距離）-> 1.0");
    check(mxfp4::e2m1_encode_nibble(5.9f) == 0x7u, "5.9 -> 6");
    check(mxfp4::e2m1_encode_nibble(100.0f) == 0x7u, "飽和 -> 6");
    check(mxfp4::e2m1_encode_nibble(-100.0f) == 0xFu, "負の飽和 -> -6");
    check(mxfp4::e2m1_encode_nibble(std::nanf("")) == 0x0u, "NaN -> 0");
}

// ---- E8M0 コーデック --------------------------------------------------------

void test_e8m0() {
    check(mxfp4::e8m0_decode_u8(127u) == 1.0f, "127 -> 1");
    check(mxfp4::e8m0_decode_u8(128u) == 2.0f, "128 -> 2");
    check(mxfp4::e8m0_decode_u8(126u) == 0.5f, "126 -> 0.5");
    check(std::isnan(mxfp4::e8m0_decode_u8(255u)), "255 -> NaN");
    check(mxfp4::e8m0_decode_u8(0u) == std::ldexp(1.0f, -127), "0 -> 2^-127");
    // encode は 2 の累乗をそのまま返し、その間の値は切り上げる。
    check(mxfp4::e8m0_encode_f32(1.0f) == 127u, "1 -> 127");
    check(mxfp4::e8m0_encode_f32(2.0f) == 128u, "2 -> 128");
    check(mxfp4::e8m0_encode_f32(1.5f) == 128u, "1.5 -> 128（切り上げ）");
    check(mxfp4::e8m0_encode_f32(0.0f) == 0u, "0 -> 0");
}

// ---- 量子化 -> 逆量子化 -----------------------------------------------------

void test_quantize_error() {
    // 32 要素ごとに E8M0 スケールを決め、各値は E2M1 の格子へ丸める。
    // 誤差は格子の半歩（= scale * 0.5、大きい側では scale * 1）以内に収まる。
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    constexpr std::uint32_t kCols = 96u;   // 32 の倍数 3 ブロック
    constexpr std::uint32_t kRows = 16u;
    const std::uint32_t kp = mxfp4::pad_k(kCols);

    std::vector<float> src(static_cast<std::size_t>(kRows) * kCols);
    for (auto& v : src) v = dist(rng);

    std::vector<std::uint8_t> canon(static_cast<std::size_t>(kRows) * (kp / 2u));
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(kRows) * (kp / mxfp4::kBlockK));
    for (std::uint32_t r = 0; r < kRows; ++r) {
        const auto st = mxfp4::quantize_row(src.data() + static_cast<std::size_t>(r) * kCols, kCols, kp,
                                            canon.data() + static_cast<std::size_t>(r) * (kp / 2u),
                                            scales.data() + static_cast<std::size_t>(r) * (kp / mxfp4::kBlockK));
        check(st.ok(), "quantize_row が成功");
    }

    std::vector<std::uint8_t> tiled(mxfp4::tiled_bytes(kRows, kp));
    check(mxfp4::preshuffle_codes(canon.data(), kRows, kCols, kp, tiled.data()).ok(),
          "preshuffle_codes が成功");

    std::vector<float> back(static_cast<std::size_t>(kRows) * kCols);
    check(mxfp4::dequantize(tiled.data(), scales.data(), kRows, kCols, kp, back.data()).ok(),
          "dequantize が成功");

    double worst = 0.0;
    for (std::uint32_t r = 0; r < kRows; ++r) {
        for (std::uint32_t c = 0; c < kCols; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * kCols + c;
            const float s = mxfp4::e8m0_decode_u8(scales[static_cast<std::size_t>(r) * (kp / 32u) +
                                                          c / 32u]);
            const double err = std::fabs(static_cast<double>(back[i]) - src[i]);
            // 格子の間隔は最大 1.0 * scale（4 -> 6 の区間）。
            const double bound = static_cast<double>(s) * 1.0 + 1e-6;
            if (err > bound) worst = err;
        }
    }
    check(worst == 0.0, "量子化誤差が格子の半歩以内");
    std::printf("  quantize: 最大誤差超過 = %.3e\n", worst);
}

// ---- 断片順 relayout --------------------------------------------------------

void test_preshuffle_roundtrip() {
    constexpr std::uint32_t kRows = 40u;   // 16 の倍数でない（切り上げを確認）
    constexpr std::uint32_t kCols = 96u;
    const std::uint32_t kp = mxfp4::pad_k(kCols);
    const std::size_t row_bytes = kp / 2u;

    std::vector<std::uint8_t> canon(static_cast<std::size_t>(kRows) * row_bytes);
    for (std::size_t i = 0; i < canon.size(); ++i) {
        canon[i] = static_cast<std::uint8_t>((i * 37u) & 0xFFu);
    }
    std::vector<std::uint8_t> tiled(mxfp4::tiled_bytes(kRows, kp));
    std::vector<std::uint8_t> back(canon.size());

    check(mxfp4::preshuffle_codes(canon.data(), kRows, kCols, kp, tiled.data()).ok(),
          "preshuffle が成功");
    check(mxfp4::untile_codes(tiled.data(), kRows, kCols, kp, back.data()).ok(),
          "untile が成功");
    bool same = true;
    for (std::uint32_t r = 0; r < kRows && same; ++r) {
        for (std::uint32_t c = 0; c < kCols && same; ++c) {
            const std::uint8_t a = static_cast<std::uint8_t>(
                (canon[static_cast<std::size_t>(r) * row_bytes + (c >> 1u)] >>
                 (4u * (c & 1u))) & 0x0Fu);
            const std::uint8_t b = static_cast<std::uint8_t>(
                (back[static_cast<std::size_t>(r) * row_bytes + (c >> 1u)] >>
                 (4u * (c & 1u))) & 0x0Fu);
            if (a != b) same = false;
        }
    }
    check(same, "preshuffle -> untile が恒等");
}

void test_offsets_match_layout() {
    // tiled_offset が指すバイトのニブルが、そのまま (row, col) の値になっていること。
    // 参照実装（16 行 × 32 K = 256 B、行 ol のバイト b が k=b と k=16+b）と同じ配置か。
    constexpr std::uint32_t kRows = 16u;
    constexpr std::uint32_t kCols = 64u;
    const std::uint32_t kp = mxfp4::pad_k(kCols);
    std::vector<std::uint8_t> canon(static_cast<std::size_t>(kRows) * (kp / 2u), 0u);
    std::vector<std::uint8_t> tiled(mxfp4::tiled_bytes(kRows, kp));
    // 各 (row, col) に一意なニブルを入れる（col 0..15 と 16..31 の対応を確認するため）。
    for (std::uint32_t r = 0; r < kRows; ++r) {
        for (std::uint32_t c = 0; c < kCols; ++c) {
            const std::uint8_t nib = static_cast<std::uint8_t>((r + c) & 0x0Fu);
            canon[static_cast<std::size_t>(r) * (kp / 2u) + (c >> 1u)] = static_cast<std::uint8_t>(
                canon[static_cast<std::size_t>(r) * (kp / 2u) + (c >> 1u)] |
                (nib << (4u * (c & 1u))));
        }
    }
    check(mxfp4::preshuffle_codes(canon.data(), kRows, kCols, kp, tiled.data()).ok(),
          "preshuffle が成功（オフセット検査）");
    bool ok = true;
    for (std::uint32_t r = 0; r < kRows && ok; ++r) {
        for (std::uint32_t c = 0; c < kCols && ok; ++c) {
            const std::size_t off = mxfp4::tiled_offset(r, c, kp);
            const std::uint8_t nib = static_cast<std::uint8_t>(
                (tiled[off] >> mxfp4::tiled_nibble_shift(c)) & 0x0Fu);
            if (nib != static_cast<std::uint8_t>((r + c) & 0x0Fu)) ok = false;
            // タイル内の行位置は 16 B 境界（レーン 16 B）。
            if ((off % mxfp4::kLaneBytes) != (c % 16u)) ok = false;
        }
    }
    check(ok, "tiled_offset がレイアウトと一致");
    // 先頭タイル・先頭ブロックの行 0 はオフセット 0、行 1 は 16、ブロック 1 は 256。
    check(mxfp4::tiled_offset(0u, 0u, kp) == 0u, "offset(0,0) = 0");
    check(mxfp4::tiled_offset(1u, 0u, kp) == 16u, "offset(1,0) = 16");
    check(mxfp4::tiled_offset(0u, 32u, kp) == 256u, "offset(0,32) = 256");
    check(mxfp4::tiled_offset(16u, 0u, kp) == 2u * 256u, "offset(16,0) = 512（次のタイル）");
    check(mxfp4::tiled_nibble_shift(0u) == 0u, "col 0 は下位ニブル");
    check(mxfp4::tiled_nibble_shift(16u) == 4u, "col 16 は上位ニブル");
    check(mxfp4::scale_offset(3u, 64u, kp) == 3u * 2u + 2u, "scale_offset が行優先");
}

void test_padding_zero() {
    // K の詰め（cols < k_padded）と行の詰めは 0 で埋まること。
    constexpr std::uint32_t kRows = 16u;
    constexpr std::uint32_t kCols = 40u;
    const std::uint32_t kp = mxfp4::pad_k(kCols);   // 64
    std::vector<std::uint8_t> canon(static_cast<std::size_t>(kRows) * (kp / 2u), 0xAAu);
    std::vector<std::uint8_t> tiled(mxfp4::tiled_bytes(kRows, kp));
    check(mxfp4::preshuffle_codes(canon.data(), kRows, kCols, kp, tiled.data()).ok(),
          "preshuffle が成功（詰め）");
    bool zero = true;
    for (std::uint32_t r = 0; r < kRows; ++r) {
        for (std::uint32_t c = kCols; c < kp; ++c) {
            const std::uint8_t byte = tiled[mxfp4::tiled_offset(r, c, kp)];
            if (((byte >> mxfp4::tiled_nibble_shift(c)) & 0x0Fu) != 0u) zero = false;
        }
    }
    check(zero, "K の詰め領域が 0");
}

}  // namespace

int main() {
    test_e2m1_table();
    test_e2m1_rounding();
    test_e8m0();
    test_quantize_error();
    test_preshuffle_roundtrip();
    test_offsets_match_layout();
    test_padding_zero();
    if (failures == 0) {
        std::printf("unit.quant_mxfp4: OK\n");
        return 0;
    }
    std::printf("unit.quant_mxfp4: %d 件失敗\n", failures);
    return 1;
}
