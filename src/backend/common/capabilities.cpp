// shelf backend/common — 機能表の実装（純粋。HIP 型を持たない）。
#include <shelf/backend/common/capabilities.hpp>

namespace shelf::backend {
namespace {

constexpr const char* kFeatureNames[kFeatureCount] = {
    "cpu_reference", "device_execution", "fp8_w8a8",   "fp8_finegrained",
    "mxfp4",         "paged_kv",         "gdn",        "pipeline_parallel",
};

static_assert(kFeatureCount == 8, "feature_name の表と Feature の個数を合わせる");

}  // namespace

const char* feature_name(Feature f) noexcept {
    const auto i = static_cast<std::size_t>(f);
    return (i < kFeatureCount) ? kFeatureNames[i] : "unknown";
}

std::size_t FeatureSet::size() const noexcept {
    std::uint64_t v = bits_;
    std::size_t n = 0;
    while (v != 0) {
        v &= (v - 1);
        ++n;
    }
    return n;
}

std::string FeatureSet::join() const {
    if (bits_ == 0) return "なし";
    std::string s;
    for (std::size_t i = 0; i < kFeatureCount; ++i) {
        const auto f = static_cast<Feature>(i);
        if (!has(f)) continue;
        if (!s.empty()) s += ", ";
        s += feature_name(f);
    }
    return s;
}

FeatureSet build_features(bool device_runtime) noexcept {
    FeatureSet s;
    s.add(Feature::cpu_reference);
    if (device_runtime) {
        // GPU 実行・FP8・paged KV・GDN・PP は同じビルド単位（backend/hip）に入る。
        // 個別の可否が分かれるようになったら、ここで分けて宣言する。
        s.add(Feature::device_execution);
        s.add(Feature::fp8_w8a8);
        s.add(Feature::fp8_finegrained);
        s.add(Feature::mxfp4);
        s.add(Feature::paged_kv);
        s.add(Feature::gdn);
        s.add(Feature::pipeline_parallel);
    }
    return s;
}

}  // namespace shelf::backend
