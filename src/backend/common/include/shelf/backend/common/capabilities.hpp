// shelf backend/common — バックエンドの機能表（D6）。
//
// 役割: 「このビルド／この装置で何ができるか」を**値**として表し、
//       起動要求との突き合わせを 1 箇所（`runtime/boot::evaluate`）で行えるようにする。
//       HIP 型は置かない（このヘッダは HIP ヘッダを include しない）。
//
// 使い方: probe が作った `FeatureSet` に要求を突き合わせ、足りない機能を
// **名前付きで**返す。アプリは要求を宣言するだけで、可否の判断はしない。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace shelf::backend {

// 機能の識別子。**追加は末尾へ**（ビット位置を既存の意味に重ねない）。
enum class Feature : std::uint8_t {
    cpu_reference = 0,     // backend/cpu の参照経路が使える
    device_execution,      // GPU で実行できる（HIP ランタイム + カーネル）
    fp8_w8a8,              // FP8 W8A8（E4M3 行列命令・per-channel または block スケール）
    fp8_finegrained,       // FP8 の細粒度（1x128 など）スケール
    mxfp4,                 // MXFP4（E2M1 + E8M0 ブロック）
    paged_kv,              // paged KV キャッシュ
    gdn,                   // GDN（linear_attention）の融合カーネル
    pipeline_parallel,     // 複数デバイスへのレイヤ分割実行
    count,
};

inline constexpr std::size_t kFeatureCount = static_cast<std::size_t>(Feature::count);

const char* feature_name(Feature f) noexcept;

// 機能の集合（ビット集合）。順序を持たないので、表示は `feature_name` の順で固定する。
class FeatureSet {
public:
    FeatureSet() = default;

    FeatureSet& add(Feature f) noexcept {
        bits_ |= (std::uint64_t{1} << static_cast<std::uint64_t>(f));
        return *this;
    }
    FeatureSet& remove(Feature f) noexcept {
        bits_ &= ~(std::uint64_t{1} << static_cast<std::uint64_t>(f));
        return *this;
    }
    bool has(Feature f) const noexcept {
        return (bits_ & (std::uint64_t{1} << static_cast<std::uint64_t>(f))) != 0;
    }
    bool empty() const noexcept { return bits_ == 0; }
    std::size_t size() const noexcept;
    // `other` のうち **this に無いもの**（= 不足分）。
    FeatureSet missing_from(const FeatureSet& other) const noexcept {
        FeatureSet out;
        out.bits_ = other.bits_ & ~bits_;
        return out;
    }
    bool contains_all(const FeatureSet& other) const noexcept {
        return (other.bits_ & ~bits_) == 0;
    }
    FeatureSet operator|(const FeatureSet& o) const noexcept {
        FeatureSet r;
        r.bits_ = bits_ | o.bits_;
        return r;
    }
    FeatureSet& operator|=(const FeatureSet& o) noexcept {
        bits_ |= o.bits_;
        return *this;
    }
    bool operator==(const FeatureSet& o) const noexcept { return bits_ == o.bits_; }
    bool operator!=(const FeatureSet& o) const noexcept { return bits_ != o.bits_; }
    std::uint64_t bits() const noexcept { return bits_; }

    // "fp8_w8a8, paged_kv"（空なら "なし"）。ログ・エラー文言用。
    std::string join() const;

private:
    std::uint64_t bits_ = 0;
};

// このビルドで実装済みの機能（コンパイル時に決まる事実）。
// `device_runtime` は「HIP ランタイムとカーネルが組み込まれているか」を呼び手が渡す
// （このヘッダは `SHELF_HAVE_HIP` を知らない = 純粋に保つ）。
FeatureSet build_features(bool device_runtime) noexcept;

// 指定デバイス台数で pipeline-parallel が成立するか（1 台なら常に真）。
inline bool pipeline_ok(std::size_t stages) noexcept { return stages <= 1u; }

}  // namespace shelf::backend
