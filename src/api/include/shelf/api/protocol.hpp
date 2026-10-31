// shelf api — エンジン境界の型付き JSONL（`engine_contract.md` §1 / §2）。
//
// 1 行 = 1 オブジェクト。Server はこの型だけを知り、`shelf::Status::Code` を
// そのまま HTTP に直結させない（`serve_protocol.md` §2 の 3 段正規化）。
//
// 位置づけ: `src/api` は Server とエンジンの**境界**。backend も HIP も持たない。
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <shelf/api/sampling.hpp>
#include <shelf/status.hpp>

namespace shelf::api {

// ---------------------------------------------------------------------------
// 操作（Server → Engine）
// ---------------------------------------------------------------------------
struct GenerateRequest {
    std::string request_id;
    std::vector<std::int32_t> input_ids;   // [1, 2, 3]
    std::uint32_t max_new_tokens = 0;      // > 0
    Sampling sampling;                     // 5 パラメータ（省略可・既定は greedy）
};

struct CancelRequest {
    std::string request_id;
};

struct PingRequest {};

using Request = std::variant<GenerateRequest, CancelRequest, PingRequest>;

// 1 行を解析する。未知キー・型違い・範囲違反は `invalid_argument`（理由にフィールド名）。
// 空行・末尾改行は許容する（`\r` も落とす）。
Result<Request> parse_request(std::string_view line);

// 逆方向（エンジンのテストとフェイクで使う）。
std::string encode_request(const Request& r);

// ---------------------------------------------------------------------------
// イベント（Engine → Server）
// ---------------------------------------------------------------------------
// エンジンから Server へ出る `code`（`serve_protocol.md` §3 の code 列）。
// `Status::Code` の写像は `code_from_status`（ここが唯一の対応表）。
struct EngineError {
    std::optional<std::string> request_id;
    std::string code;      // "invalid_argument" / "overloaded" / "busy" / ...
    std::string message;
    std::string param;     // 分かるときだけ（sampling のフィールド名など）
};

struct TokenEvent {
    std::string request_id;
    std::int32_t token_id = 0;
};

enum class FinishReason : std::uint8_t { stop, length, cancelled, error };
const char* to_string(FinishReason r) noexcept;
std::optional<FinishReason> finish_reason_from(std::string_view s) noexcept;

struct DoneEvent {
    std::string request_id;
    std::vector<std::int32_t> generated_ids;
    FinishReason finish_reason = FinishReason::length;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t generated_tokens = 0;
    std::uint64_t seed = 0;   // 必ず返す（§3.4）
};

struct PongEvent {
    std::string backend;          // "hip" / "cpu"
    std::string model;            // 表示用（パスではない）
    std::uint64_t max_context = 0;
    std::vector<std::string> features;   // 機能名（`backend/common` の FeatureSet の写像）
};

struct ShutdownEvent {};

using Event = std::variant<TokenEvent, DoneEvent, EngineError, PongEvent, ShutdownEvent>;

// 1 行（末尾改行なし）。`request_id` は生成したイベントでは常に入る。
std::string encode_event(const Event& e);

// 1 行を解析する（Server 側のテスト・フェイク用）。未知 event は invalid_argument。
Result<Event> parse_event(std::string_view line);

// ---------------------------------------------------------------------------
// エラー写像（`serve_protocol.md` §3 の表が正本。ここは機械的な写像だけ）
// ---------------------------------------------------------------------------
// `invalid_state` は 2 分岐（busy = 再試行可 / state_error = 壊れている）。
// 各呼び出し点がどちらかを**宣言**する（宣言漏れは state_error 扱い）。
enum class InvalidStateKind : std::uint8_t { busy, state_error };

struct ErrorCode {
    std::string code;     // Server が HTTP へ写す code
    bool client_error;    // true = 400 系
};

ErrorCode code_from_status(Status::Code c, InvalidStateKind kind = InvalidStateKind::state_error);

// 例外を使わない経路用: Status から EngineError を作る。
EngineError error_from_status(const Status& st, InvalidStateKind kind = InvalidStateKind::state_error,
                              std::optional<std::string> request_id = std::nullopt,
                              std::string param = {});

}  // namespace shelf::api
