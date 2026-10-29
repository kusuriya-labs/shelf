// shelf unit test — quant/fp8_w8a8（E4M3 コーデックと断片順 relayout）。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include <shelf/quant/fp8_w8a8.hpp>

namespace {

using namespace shelf::quant;

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

// ---- E4M3 コーデック --------------------------------------------------------

void test_decode_table() {
    // 有限最大値 448（0x7E）と最小正規 2^-6（0x08）、非正規の量子 2^-9。
    check(fp8::e4m3_decode(0x7E) == 448.0f, "0x7E -> 448");
    check(fp8::e4m3_decode(0xFE) == -448.0f, "0xFE -> -448");
    check(fp8::e4m3_decode(0x08) == 0x1p-6f, "0x08 -> 2^-6 (最小正規)");
    check(fp8::e4m3_decode(0x01) == 0x1p-9f, "0x01 -> 2^-9 (非正規の量子)");
    check(fp8::e4m3_decode(0x07) == 7.0f * 0x1p-9f, "0x07 -> 7*2^-9");
    check(fp8::e4m3_decode(0x00) == 0.0f, "0x00 -> 0");
    check(fp8::e4m3_decode(0x80) == 0.0f, "0x80 -> -0");
    check(std::signbit(fp8::e4m3_decode(0x80)), "0x80 は負のゼロ");
    check(std::isnan(fp8::e4m3_decode(0x7F)), "0x7F -> NaN");
    check(std::isnan(fp8::e4m3_decode(0xFF)), "0xFF -> NaN (-)");
    // 1.0 = E(7+7=14)? -> 指数 field 7+0=... 1.0 = 2^0 -> E=7, m=0
    check(fp8::e4m3_decode(0x38) == 1.0f, "0x38 -> 1.0");
    check(fp8::e4m3_decode(0x3C) == 1.5f, "0x3C -> 1.5");
    check(fp8::e4m3_decode(0x3B) == 1.375f, "0x3B -> 1.375");
}

void test_encode_roundtrip_all_codes() {
    // 全コード: decode した値を encode すると同一コードに戻る（NaN は除く）。
    int diff = 0;
    for (int c = 0; c < 256; ++c) {
        const std::uint8_t code = static_cast<std::uint8_t>(c);
        if ((code & 0x7Fu) == 0x7Fu) {
            check(fp8::e4m3_encode(fp8::e4m3_decode(code)) == code, "NaN code 保持");
            continue;
        }
        const float v = fp8::e4m3_decode(code);
        const std::uint8_t back = fp8::e4m3_encode(v);
        if (back != code) ++diff;
    }
    check(diff == 0, "全有限コードの往復一致");
}

void test_encode_specific() {
    check(fp8::e4m3_encode(0.0f) == 0x00, "encode +0");
    check(fp8::e4m3_encode(-0.0f) == 0x80, "encode -0");
    check(fp8::e4m3_encode(1.0f) == 0x38, "encode 1.0");
    check(fp8::e4m3_encode(448.0f) == 0x7E, "encode 448");
    check(fp8::e4m3_encode(1000.0f) == 0x7E, "satfinite 正");
    check(fp8::e4m3_encode(-1000.0f) == 0xFE, "satfinite 負");
    check(fp8::e4m3_encode(std::numeric_limits<float>::infinity()) == 0x7E, "inf -> 448");
    check(fp8::e4m3_encode(-std::numeric_limits<float>::infinity()) == 0xFE, "-inf -> -448");
    check((fp8::e4m3_encode(std::numeric_limits<float>::quiet_NaN()) & 0x7Fu) == 0x7Fu,
          "NaN -> 0x7F");
    // 448 と次の間（464）は境界: 半分ちょうどなので偶数側（448 = 0x7E）へ。
    check(fp8::e4m3_encode(464.0f) == 0x7E, "464 は tie -> 448");
    check(fp8::e4m3_encode(0.0f) == 0x00, "0 保持");
    // 最小正規の直下（2^-6 - 量子/2）は 2^-6 へ繰上げになる境界を確認
    check(fp8::e4m3_encode(0x1p-6f - 0x1p-10f) == 0x08, "最小正規の少し下は 2^-6");
    check(fp8::e4m3_encode(0x1p-10f) == 0x00, "半量子未満は 0（tie は偶数側）");
    // 3*2^-10 は量子 1 と 2 の中間ちょうど -> RNE で偶数コード 0x02 へ
    check(fp8::e4m3_encode(3.0f * 0x1p-10f) == 0x02, "1.5 量子は tie -> 偶数コード");
    // 2.5 量子も tie（2 と 3 の中間）なので偶数コード 0x02 へ。
    check(fp8::e4m3_encode(5.0f * 0x1p-10f) == 0x02, "2.5 量子は tie -> 偶数コード");
    check(fp8::e4m3_encode(5.5f * 0x1p-10f) == 0x03, "2.5 量子超は 3 へ");
}

void test_encode_is_nearest_even() {
    // ランダム値: 候補コード（隣接する 2 コード）のうち必ず decode 誤差が
    // 最小になる方を選ぶこと（同値なら偶数コードへ）。
    std::mt19937 rng(20261029u);
    std::uniform_real_distribution<float> d(-480.0f, 480.0f);
    int worse = 0, ties = 0, non_even_tie = 0;
    for (int i = 0; i < 200000; ++i) {
        const float x = d(rng);
        const std::uint8_t c = fp8::e4m3_encode(x);
        if ((c & 0x7Fu) == 0x7Fu) continue;
        const float mine = std::fabs(fp8::e4m3_decode(c) - x);
        // 隣接コード（符号を保持したままコード値を ±1）
        for (int dc = -1; dc <= 1; dc += 2) {
            const int cc = ((c & 0x80u) << 1) | ((c & 0x7Fu) + dc);
            if ((cc & 0x7F) > 0x7E) continue;
            const std::uint8_t nb = static_cast<std::uint8_t>((cc & 0x100u) | (cc & 0x7Fu));
            const float alt = std::fabs(fp8::e4m3_decode(nb) - x);
            if (alt < mine) ++worse;
            else if (alt == mine) {
                ++ties;
                if ((c & 1u) != 0u && (nb & 1u) == 0u) ++non_even_tie;
            }
        }
    }
    check(worse == 0, "近傍コードより誤差が大きい選択なし");
    check(non_even_tie == 0, "タイは偶数コードへ丸めている");
}

// ---- 断片順 relayout --------------------------------------------------------

void test_tiled_offset_is_bijection() {
    // 詰め込み後の全バイトがちょうど 1 回だけ書かれること（取りこぼし・重複防止）。
    const std::uint32_t rows = 40, cols = 100, kp = fp8::pad_k(cols);   // 100 -> 128
    std::vector<int> hits(fp8::tiled_bytes(rows, kp), 0);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < cols; ++c) ++hits[fp8::tiled_offset(r, c, kp)];
    bool dup = false;
    std::size_t ones = 0;
    for (const int h : hits) {
        if (h > 1) dup = true;
        if (h == 1) ++ones;
    }
    check(!dup && ones == static_cast<std::size_t>(rows) * cols,
          "有効領域を 1 回だけ覆い、詰め領域は未使用");
    check(fp8::pad_k(96) == 96 && fp8::pad_k(97) == 128, "pad_k は 32 倍数へ切り上げ");
    check(fp8::pad_rows(17) == 32, "pad_rows は 16 行単位");
    check(fp8::tiled_bytes(16, 32) == 512, "タイル 16x32 = 512 バイト");
}

void test_preshuffle_roundtrip() {
    struct Case { std::uint32_t rows, cols; } cases[] = {{1, 32}, {16, 32}, {17, 96},
                                                         {64, 4096}, {4096, 12288 / 4}};
    for (const auto& cs : cases) {
        std::vector<std::uint8_t> src(static_cast<std::size_t>(cs.rows) * cs.cols);
        for (std::size_t i = 0; i < src.size(); ++i)
            src[i] = static_cast<std::uint8_t>((i * 37u + 11u) & 0x7Fu);   // NaN コードは避ける
        const std::uint32_t kp = fp8::pad_k(cs.cols);
        std::vector<std::uint8_t> tiled(fp8::tiled_bytes(cs.rows, kp), 0xEEu);
        check(fp8::preshuffle_codes(src.data(), cs.rows, cs.cols, kp, tiled.data()).ok(),
              "preshuffle ok");
        // タイルの詰め領域（K 方向）が 0 になっていること
        bool pad_zero = true;
        for (std::uint32_t r = 0; r < fp8::pad_rows(cs.rows); ++r)
            for (std::uint32_t c = cs.cols; c < kp; ++c)
                if (tiled[fp8::tiled_offset(r, c, kp)] != 0u) pad_zero = false;
        check(pad_zero, "K 方向の詰めは 0");

        std::vector<std::uint8_t> back(static_cast<std::size_t>(cs.rows) * cs.cols, 0u);
        check(fp8::untile_codes(tiled.data(), cs.rows, cs.cols, kp, back.data()).ok(),
              "untile ok");
        check(back == src, "preshuffle -> untile で元に戻るか");
    }
}

void test_preshuffle_matches_fragment_addresses() {
    // カーネル側のアドレス式（16 行タイル優先 → 32 K タイル → 16 K 半分行優先）と
    // 一致する例を直計算で確認する（レイアウトの取り違え検出）。
    const std::uint32_t kp = 64;
    // (row 3, col 20): ntile 0, ktile 0, half 1 (20/16), 0*512 + 256 + 3*16 + 4 = 308
    check(fp8::tiled_offset(3, 20, kp) == 256 + 3u * 16u + 4u, "半分行優先の例");
    // (row 17, col 40): ntile 1 (1*kp*16 = 1024) + ktile 1 (512) + half 0（40 は
    // K[32,64) の前半 16 側）+ 1*16 + 8
    check(fp8::tiled_offset(17, 40, kp) == 1024u + 512u + 16u + 8u, "タイル境界の例");
    // (row 17, col 56): 同上で half 1（+256）
    check(fp8::tiled_offset(17, 56, kp) == 1024u + 512u + 256u + 16u + 8u, "16 K 半分の例");
}

void test_scale_contract() {
    check(fp8::token_scale(0.0f) == 0.0f, "amax 0 -> scale 0");
    check(fp8::token_scale(448.0f) == 1.0f, "amax 448 -> scale 1");
    check(fp8::kE4m3Max == 448.0f, "E4M3 の最大値は 448");
    check(fp8::weight_scale_count(fp8::WeightScaleGranularity::PerChannel, 4096u, 12288u) == 4096u,
          "per-channel は出力特征ごと");
    check(fp8::weight_scale_count(fp8::WeightScaleGranularity::Block128, 4096u, 12288u) ==
              32u * 96u,
          "block128 は 128(出力) × 128(K) タイルごと");
    check(fp8::scale_k_blocks(12288u) == 96u && fp8::scale_k_blocks(4096u) == 32u,
          "K 方向のスケールブロック数");
    check(fp8::scale_n_blocks(fp8::WeightScaleGranularity::Block128, 4096u) == 32u,
          "N 方向のスケールブロック数");
    check(fp8::act_scale_groups(12288u, fp8::kActGroupK) == 96u, "act_group=128 のグループ数");
    check(fp8::act_scale_groups(12288u, fp8::kActGroupWholeRow) == 1u, "act_group=0 は 1 グループ");
    check(fp8::activation_scale_count(64u, 4096u, fp8::kActGroupK) == 64u * 32u,
          "活性化スケールの総数 = rows × グループ数");
    check(fp8::amax_scale(896.0f) == 2.0f, "amax からスケール");
    check(!fp8::validate_tiling(4, 4, 30).ok(), "k_padded は 32 倍数でないと拒否");
    check(!fp8::validate_tiling(4, 40, 32).ok(), "k_padded < cols は拒否");
    check(fp8::validate_tiling(4, 40, 64).ok(), "受理される形");
}

}  // namespace

int main() {
    test_decode_table();
    test_encode_roundtrip_all_codes();
    test_encode_specific();
    test_encode_is_nearest_even();
    test_tiled_offset_is_bijection();
    test_preshuffle_roundtrip();
    test_preshuffle_matches_fragment_addresses();
    test_scale_contract();
    std::printf("test_quant_fp8: failed=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
