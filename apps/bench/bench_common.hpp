// shelf bench — 共通の引数解析・統計・出力・計測ヘルパ。
//
// 終了コード規約は phaseshift-bench と同一:
//   0 = 成功 / --help、1 = 実行時エラー、2 = 引数・用法の誤り
// --help は装置初期化とモデルロードより前に処理され、GPU が無くても exit 0。
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#ifdef SHELF_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

namespace shelf::bench {

inline constexpr int kExitOk = 0;
inline constexpr int kExitRuntime = 1;
inline constexpr int kExitUsage = 2;

// --key value と --key=value の両対応。それ以外は位置引数として残す。
// --help / -h は値を取らないフラグ（位置によっても動くようにするため）。
class Args {
public:
    bool parse(int argc, char** argv, std::string* error);

    bool has(const std::string& key) const;
    std::string str(const std::string& key, const std::string& def = "") const;
    // 値が無い・整数でない場合は *ok = false（def を返す）。
    std::int64_t i64(const std::string& key, std::int64_t def, bool* ok) const;
    // カンマ区切りの整数リスト。不正要素があれば *ok = false。
    std::vector<std::int64_t> i64_list(const std::string& key,
                                       const std::vector<std::int64_t>& def,
                                       bool* ok) const;

    const std::vector<std::string>& positional() const { return pos_; }
    // パース済みのキー一覧（出現順）。未知オプション検出に使う。
    const std::vector<std::string>& keys() const { return keys_; }

private:
    const std::string* find(const std::string& key) const;
    std::vector<std::pair<std::string, std::string>> kv_;
    std::vector<std::string> keys_;
    std::vector<std::string> pos_;
};

// 未知の --key を検出する（すべて既知なら true）。見つかれば *bad にキー名を入れる。
// shelf-validate と同じ規約: 未知のオプションは起動エラー（exit 2）。
bool only_known_keys(const Args& a, const std::vector<std::string>& known, std::string* bad);

// 計測統計（phaseshift-bench の集計規約に合わせる: min / p10 / p50 / p90 / p95 / mean / stddev）。
struct Stats {
    double min = 0.0;
    double p10 = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
};
Stats summarize(std::vector<double> xs);  // 空なら全 0

// stdout の key=value 行。
void kv_print(const char* key, double value, int decimals = 3);
void kv_print(const char* key, std::int64_t value);
void kv_print(const char* key, const std::string& value);

// CSV 追記。ファイルが無い・空ならヘッダを先に書く。
bool append_csv(const std::string& path, const std::string& header,
                const std::vector<std::string>& row, std::string* error);

// CSV 1 行分を "a","b","c" へ（カンマ・引用符は想定しない数値/短い識別子のみ）。
std::string csv_join(const std::vector<std::string>& row);

#ifdef SHELF_HAVE_HIP
// カーネル微計測用: 自前の HIP ストリームとイベント 2 点を作る。
// 規約（phaseshift-bench と同じ）: warmup 回破棄 → samples × launches を
// 1 イベント対で測り launches で割る → µs/launch の統計。
class StreamTimer {
public:
    StreamTimer() = default;
    StreamTimer(const StreamTimer&) = delete;
    StreamTimer& operator=(const StreamTimer&) = delete;
    ~StreamTimer();

    bool init(std::string* error);
    hipStream_t stream() const { return stream_; }

    template <class Fn>
    bool measure(int warmup, int samples, int launches, Fn&& fn, Stats* out,
                 std::string* error) {
        for (int i = 0; i < warmup; ++i) {
            const hipError_t e = fn(stream_);
            if (e != hipSuccess) return fail(error, "warmup launch", e);
        }
        if (hipStreamSynchronize(stream_) != hipSuccess)
            return fail(error, "warmup sync", hipGetLastError());

        std::vector<double> xs;
        xs.reserve(static_cast<std::size_t>(samples));
        for (int s = 0; s < samples; ++s) {
            if (hipEventRecord(ev0_, stream_) != hipSuccess)
                return fail(error, "record start", hipGetLastError());
            for (int l = 0; l < launches; ++l) {
                const hipError_t e = fn(stream_);
                if (e != hipSuccess) return fail(error, "launch", e);
            }
            if (hipEventRecord(ev1_, stream_) != hipSuccess)
                return fail(error, "record end", hipGetLastError());
            if (hipEventSynchronize(ev1_) != hipSuccess)
                return fail(error, "sync end", hipGetLastError());
            float ms = 0.0f;
            const hipError_t e = hipEventElapsedTime(&ms, ev0_, ev1_);
            if (e != hipSuccess) return fail(error, "elapsed", e);
            xs.push_back(static_cast<double>(ms) * 1000.0 / static_cast<double>(launches));
        }
        *out = summarize(std::move(xs));
        return true;
    }

    hipStream_t stream_ = nullptr;
    hipEvent_t ev0_ = nullptr;
    hipEvent_t ev1_ = nullptr;

private:
    static bool fail(std::string* error, const char* what, hipError_t e) {
        if (error) *error = std::string(what) + ": " + hipGetErrorString(e);
        return false;
    }
};
#endif  // SHELF_HAVE_HIP

}  // namespace shelf::bench
