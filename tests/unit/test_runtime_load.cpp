// shelf unit test — runtime/load の検証と実体化（D3）。GPU 不要。
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <shelf/quant/fp8_w8a8.hpp>
#include <shelf/runtime/load/materialize.hpp>

namespace {

using shelf::DataType;
using shelf::Status;
using shelf::bf16_t;
using shelf::runtime::load::Fp8ModelFormat;
using shelf::runtime::load::MaterializedWeight;
using shelf::runtime::load::TensorBlob;
using shelf::runtime::load::TensorSource;
using shelf::runtime::load::WeightKind;
using shelf::runtime::load::WeightRequest;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

// メモリ上の供給元（apps の ShardSource と同じ契約）。
class MemSource : public TensorSource {
public:
    void put_bf16(const std::string& n, const std::vector<bf16_t>& v) {
        TensorBlob b;
        b.dtype = DataType::bf16;
        b.bytes.resize(v.size() * sizeof(bf16_t));
        std::memcpy(b.bytes.data(), v.data(), b.bytes.size());
        data_[n] = std::move(b);
    }
    void put_f32(const std::string& n, const std::vector<float>& v) {
        TensorBlob b;
        b.dtype = DataType::f32;
        b.bytes.resize(v.size() * sizeof(float));
        std::memcpy(b.bytes.data(), v.data(), b.bytes.size());
        data_[n] = std::move(b);
    }
    void put_u8(const std::string& n, DataType dt, const std::vector<std::uint8_t>& v) {
        TensorBlob b;
        b.dtype = dt;
        b.bytes = v;
        data_[n] = std::move(b);
    }
    bool has(const std::string& n) const override { return data_.count(n) != 0u; }
    shelf::Result<TensorBlob> read(const std::string& n) const override {
        const auto it = data_.find(n);
        if (it == data_.end())
            return Status::not_found("no tensor: " + n, __FILE__, __LINE__);
        return it->second;
    }

private:
    std::map<std::string, TensorBlob> data_;
};

std::vector<bf16_t> bf16_seq(std::size_t n, float scale) {
    std::vector<bf16_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = shelf::f32_to_bf16(scale * static_cast<float>(i + 1));
    return v;
}

}  // namespace

int main() {
    using namespace shelf::quant::fp8;

    // 1. bf16 の 1 次元（norm）も通る（移設時に 2 次元要求で落ちた回帰の門）
    {
        MemSource src;
        src.put_bf16("n.weight", bf16_seq(8, 0.5f));
        WeightRequest req{"n.weight", {8}};
        auto m = shelf::runtime::load::materialize(src, req, Fp8ModelFormat{});
        check(m.ok(), "1 次元 bf16 を実体化できる");
        if (m.ok()) {
            check(m.value().kind == WeightKind::Bf16, "kind は Bf16");
            check(m.value().bf16.size() == 8, "要素数が一致");
        }
    }
    // 2. bf16 の 2 次元（Linear）
    {
        MemSource src;
        src.put_bf16("w.weight", bf16_seq(2 * 32, 1.0f));
        WeightRequest req{"w.weight", {2, 32}};
        auto m = shelf::runtime::load::materialize(src, req, Fp8ModelFormat{});
        check(m.ok() && m.value().bf16.size() == 64, "2 次元 bf16 を実体化できる");
    }
    // 3. f32
    {
        MemSource src;
        src.put_f32("w.weight", std::vector<float>(6, 1.5f));
        WeightRequest req{"w.weight", {2, 3}};
        auto m = shelf::runtime::load::materialize(src, req, Fp8ModelFormat{});
        check(m.ok() && m.value().kind == WeightKind::F32 && m.value().f32.size() == 6,
              "f32 を実体化できる");
    }
    // 4. FP8（per-channel + `_scale`）: 断片順へ relayout される
    {
        MemSource src;
        std::vector<std::uint8_t> codes(2 * 32, 0x3C);
        src.put_u8("w.weight", DataType::fp8_e4m3, codes);
        src.put_bf16("w.weight_scale", {shelf::f32_to_bf16(0.5f), shelf::f32_to_bf16(0.25f)});
        Fp8ModelFormat fmt;
        fmt.quantized = true;
        fmt.act_group = 0;
        fmt.scale_suffix = "_scale";
        WeightRequest req{"w.weight", {2, 32}};
        auto m = shelf::runtime::load::materialize(src, req, fmt);
        check(m.ok(), "fp8 を実体化できる");
        if (m.ok()) {
            check(m.value().kind == WeightKind::Fp8W8A8, "kind は Fp8W8A8");
            check(m.value().codes.size() == tiled_bytes(2, 32), "コードは断片順の容量");
            check(m.value().scales.size() == 2, "スケールは per-channel の 2 個");
            check(m.value().k_padded == 32 && m.value().features == 2, "shape の記録");
            check(!m.value().weight_block128, "per-channel は block128 ではない");
            check(m.value().scales[0] == 0.5f && m.value().scales[1] == 0.25f,
                  "スケール値が保たれる");
        }
    }
    // 5. FP8 blockwise（`_scale_inv`）: 27B 系の宣言
    {
        MemSource src;
        src.put_u8("w.weight", DataType::fp8_e4m3, std::vector<std::uint8_t>(128 * 128, 0x38));
        // [ceil(128/128), ceil(128/128)] = 1 個
        src.put_bf16("w.weight_scale_inv", {shelf::f32_to_bf16(0.125f)});
        Fp8ModelFormat fmt;
        fmt.quantized = true;
        fmt.weight = WeightScaleGranularity::Block128;
        fmt.scale_suffix = "_scale_inv";
        WeightRequest req{"w.weight", {128, 128}};
        auto m = shelf::runtime::load::materialize(src, req, fmt);
        check(m.ok() && m.value().weight_block128, "block128 を実体化できる");
        check(m.ok() && m.value().scales.size() == 1, "block128 のスケール数");
    }
    // 6. スケール欠落 → 明示エラー
    {
        MemSource src;
        src.put_u8("w.weight", DataType::fp8_e4m3, std::vector<std::uint8_t>(64, 1));
        Fp8ModelFormat fmt;
        fmt.quantized = true;
        WeightRequest req{"w.weight", {2, 32}};
        auto m = shelf::runtime::load::materialize(src, req, fmt);
        check(!m.ok(), "スケール欠落は失敗");
    }
    // 7. スケール個数の不一致 → 明示エラー
    {
        MemSource src;
        src.put_u8("w.weight", DataType::fp8_e4m3, std::vector<std::uint8_t>(64, 1));
        src.put_bf16("w.weight_scale", {shelf::f32_to_bf16(1.0f)});   // 2 個必要
        Fp8ModelFormat fmt;
        fmt.quantized = true;
        WeightRequest req{"w.weight", {2, 32}};
        auto m = shelf::runtime::load::materialize(src, req, fmt);
        check(!m.ok(), "スケール数の不一致は失敗");
    }
    // 8. CPU 参照は fp8 を明示的に拒否する
    {
        MemSource src;
        src.put_u8("w.weight", DataType::fp8_e4m3, std::vector<std::uint8_t>(8, 1));
        auto f = shelf::runtime::load::read_f32(src, "w.weight");
        check(!f.ok(), "read_f32 は fp8 を拒否する");
        check(f.status().message().find("FP8") != std::string::npos,
              "拒否理由に FP8 が出る");
    }

    std::printf("test_runtime_load: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
