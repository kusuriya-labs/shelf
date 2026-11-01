// shelf api — 生成パラメータの検証と抽選（`engine_contract.md` §3）。
#include <shelf/runtime/generate/sampling.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace shelf::runtime::generate {
namespace {

// 契約 §3.2 の文言。Server はこの message をそのまま envelope に載せ、
// `param` にフィールド名を入れる（写像は `serve_protocol.md` §3）。
Status bad(const char* param, const std::string& why) {
    std::string msg = "sampling ";
    msg += param;
    msg += ": ";
    msg += why;
    return Status::invalid_argument(msg, __FILE__, __LINE__);
}

bool finite_non_nan(double v) { return std::isfinite(v); }

}  // namespace

Status validate_sampling(const Sampling& s, std::string* param) {
    const auto set = [&](const char* p, const std::string& why) {
        if (param) *param = p;
        return bad(p, why);
    };
    if (!finite_non_nan(s.temperature))
        return set("temperature", "有限の値が必要（NaN / inf は不可）");
    if (s.temperature < 0.0) return set("temperature", "0 以上が必要");
    if (!finite_non_nan(s.top_p)) return set("top_p", "有限の値が必要（NaN / inf は不可）");
    if (!(s.top_p > 0.0 && s.top_p <= 1.0)) return set("top_p", "(0, 1] の範囲が必要");
    if (!finite_non_nan(s.min_p)) return set("min_p", "有限の値が必要（NaN / inf は不可）");
    if (!(s.min_p >= 0.0 && s.min_p <= 1.0)) return set("min_p", "[0, 1] の範囲が必要");
    // top_k は uint32 なので負値は表現できない（0 = off として扱う）。
    if (param) param->clear();
    return Status::make_ok();
}

std::array<std::uint32_t, 4> Philox4x32::bits(std::uint64_t seed,
                                              std::uint64_t position) noexcept {
    // Philox4x32-10（Salmon et al. 2011）。鍵 = seed の 64 bit を 2 ワードに分割、
    // カウンタ = (position 下位, position 上位, 0, 0)。
    constexpr std::uint32_t kM0 = 0xD2511F53u;
    constexpr std::uint32_t kM1 = 0xCD9E8D57u;
    constexpr std::uint32_t kW0 = 0x9E3779B9u;
    constexpr std::uint32_t kW1 = 0xBB67AE85u;

    std::array<std::uint32_t, 4> ctr{static_cast<std::uint32_t>(position & 0xFFFFFFFFu),
                                    static_cast<std::uint32_t>(position >> 32), 0u, 0u};
    const std::array<std::uint32_t, 2> key{static_cast<std::uint32_t>(seed & 0xFFFFFFFFu),
                                           static_cast<std::uint32_t>(seed >> 32)};

    const auto mulh = [](std::uint32_t a, std::uint32_t b, std::uint32_t* lo) {
        const std::uint64_t p = static_cast<std::uint64_t>(a) * b;
        *lo = static_cast<std::uint32_t>(p & 0xFFFFFFFFu);
        return static_cast<std::uint32_t>(p >> 32);
    };

    std::uint32_t k0 = key[0], k1 = key[1];
    for (int round = 0; round < 10; ++round) {
        std::uint32_t lo0 = 0, lo1 = 0;
        const std::uint32_t hi0 = mulh(kM0, ctr[0], &lo0);
        const std::uint32_t hi1 = mulh(kM1, ctr[2], &lo1);
        const std::uint32_t n0 = hi1 ^ ctr[1] ^ k0;
        const std::uint32_t n1 = hi0 ^ ctr[3] ^ k1;
        const std::uint32_t n2 = lo1 ^ ctr[1] ^ k0;
        const std::uint32_t n3 = lo0 ^ ctr[3] ^ k1;
        ctr = {n0, n1, n2, n3};
        k0 += kW0;
        k1 += kW1;
    }
    return ctr;
}

double Philox4x32::uniform(std::uint64_t seed, std::uint64_t position) noexcept {
    const auto r = bits(seed, position);
    // 32 bit の一様値を [0, 1) へ（仮数 53 bit のうち 32 bit を使う。中心化はしない）。
    return static_cast<double>(r[0]) * (1.0 / 4294967296.0);
}

std::uint64_t assign_seed(std::optional<std::uint64_t> requested, std::uint64_t nonce) noexcept {
    if (requested.has_value()) return *requested;
    // 未指定: (固定の塩, nonce) から Philox で作る（プロセス起動時刻に依存させない=
    // テストで再現できる。`nonce` は呼び手が「リクエスト通し番号」等を渡す）。
    constexpr std::uint64_t kSalt = 0x5348454C46ull;   // "SHELF"
    const auto r = Philox4x32::bits(kSalt, nonce);
    return (static_cast<std::uint64_t>(r[0]) << 32) | r[1];
}

Result<Draw> sample(const std::vector<float>& logits, const Sampling& s, std::uint64_t position) {
    return sample_range(logits.data(), logits.size(), s, position);
}

Result<Draw> sample_range(const float* logits, std::size_t n, const Sampling& s,
                          std::uint64_t position) {
    std::string param;
    SHELF_RETURN_IF_ERROR(validate_sampling(s, &param));
    if (logits == nullptr || n == 0u)
        return Status::invalid_argument("sampling: logits が空", __FILE__, __LINE__);

    // 1) 有限値だけを候補にする（NaN / -inf は落とす。全部落ちたら invalid 扱い）。
    std::vector<std::uint32_t> idx;
    idx.reserve(n);
    float max_logit = -std::numeric_limits<float>::infinity();
    for (std::uint32_t i = 0; i < n; ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) continue;
        idx.push_back(i);
        max_logit = std::max(max_logit, v);
    }
    if (idx.empty())
        return Status::invalid_argument("sampling: 有限な logit が無い", __FILE__, __LINE__);

    // greedy（既定）: argmax。同値は添字の小さい方（走査順で最初の最大）。
    if (s.is_greedy()) {
        std::uint32_t best = idx.front();
        for (const auto i : idx)
            if (logits[i] > logits[best]) best = i;
        return Draw{best, 1u, false};
    }

    // 2) 確率（温度を掛ける前の分布）で候補を切る: top_k → top_p → min_p。
    //    温度は最後（§3.1）。ここでは softmax を素の logits で作る。
    std::vector<std::uint32_t> cand = std::move(idx);
    if (s.top_k > 0 && s.top_k < cand.size()) {
        std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(s.top_k),
                          cand.end(), [&](std::uint32_t a, std::uint32_t b) {
                              if (logits[a] != logits[b]) return logits[a] > logits[b];
                              return a < b;
                          });
        cand.resize(s.top_k);
    }

    // softmax（f64 で計算。max_logit を引いて安定化）
    std::vector<double> prob(cand.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
        prob[i] = std::exp(static_cast<double>(logits[cand[i]]) - static_cast<double>(max_logit));
        sum += prob[i];
    }
    if (!(sum > 0.0))
        return Status::invalid_argument("sampling: 確率の総和が 0", __FILE__, __LINE__);
    for (auto& p : prob) p /= sum;

    // top_p: 確率の大きい順に累積し、P 以上になった所まで残す（必ず 1 個は残す）。
    if (s.top_p < 1.0) {
        std::vector<std::size_t> order(cand.size());
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return prob[a] > prob[b]; });
        double acc = 0.0;
        std::vector<char> keep(cand.size(), 0);
        for (const auto i : order) {
            keep[i] = 1;
            acc += prob[i];
            if (acc >= s.top_p) break;
        }
        std::vector<std::uint32_t> kept_cand;
        std::vector<double> kept_prob;
        for (std::size_t i = 0; i < cand.size(); ++i) {
            if (!keep[i]) continue;
            kept_cand.push_back(cand[i]);
            kept_prob.push_back(prob[i]);
        }
        cand = std::move(kept_cand);
        prob = std::move(kept_prob);
    }

    // min_p: 最大確率に対する相対閾値で落とす（必ず 1 個は残す）。
    if (s.min_p > 0.0) {
        const double pmax = *std::max_element(prob.begin(), prob.end());
        std::vector<std::uint32_t> kept_cand;
        std::vector<double> kept_prob;
        for (std::size_t i = 0; i < cand.size(); ++i) {
            if (prob[i] < s.min_p * pmax) continue;
            kept_cand.push_back(cand[i]);
            kept_prob.push_back(prob[i]);
        }
        if (!kept_cand.empty()) {
            cand = std::move(kept_cand);
            prob = std::move(kept_prob);
        }
    }

    // 3) 温度を最後に適用して再正規化。
    std::vector<double> w(cand.size());
    double wsum = 0.0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
        const double z = (static_cast<double>(logits[cand[i]]) - static_cast<double>(max_logit)) /
                         s.temperature;
        w[i] = std::exp(z);
        wsum += w[i];
    }
    if (!(wsum > 0.0))
        return Status::invalid_argument("sampling: 温度適用後の総和が 0", __FILE__, __LINE__);
    for (auto& v : w) v /= wsum;

    // 4) 抽選（(seed, position) の純粋関数）。
    const std::uint64_t seed = s.seed.value_or(assign_seed(s.seed, position));
    const double u = Philox4x32::uniform(seed, position);
    double acc = 0.0;
    std::uint32_t picked = cand.back();
    for (std::size_t i = 0; i < cand.size(); ++i) {
        acc += w[i];
        if (u < acc) {
            picked = cand[i];
            break;
        }
    }
    return Draw{picked, static_cast<std::uint32_t>(cand.size()), true};
}

}  // namespace shelf::runtime::generate
