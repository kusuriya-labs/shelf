// shelf runtime/load — テンソルの供給元（D3）。
//
// 役割: 読み取り（artifact）と、その後の検証・実体化（本モジュール）を分ける。
//       runtime は `shelf/artifact/` を知らない（境界ルール）ため、供給元は
//       アプリ側（composition root）が実装して差し込む。
//
// 契約: 返す `TensorBlob` は**所有権あり**（供給元の寿命に依存しない）。
//       ゼロコピーにしたい場合は、供給元が mapping を保持したうえで
//       `TensorSource` の実装を差し替える（runtime 側は関知しない）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <shelf/foundation/dtype.hpp>
#include <shelf/status.hpp>

namespace shelf::runtime::load {

// 供給元が返す 1 テンソル分の生バイト（所有権あり）。
struct TensorBlob {
    DataType dtype = DataType::unknown;
    std::vector<std::uint8_t> bytes;
};

class TensorSource {
public:
    TensorSource() = default;
    TensorSource(const TensorSource&) = delete;
    TensorSource& operator=(const TensorSource&) = delete;
    virtual ~TensorSource() = default;

    virtual bool has(const std::string& name) const = 0;
    // 無ければ not_found、dtype/サイズの異常は invalid_state で返す。
    virtual Result<TensorBlob> read(const std::string& name) const = 0;
};

}  // namespace shelf::runtime::load
