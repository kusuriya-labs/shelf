// shelf unit test — 生成パラメータの契約（`engine_contract.md` §3）。
//
// 見ていること:
//   1. 検証規則（§3.2）と **フィールド名**つきの失敗
//   2. 既定（temperature 0）は argmax = 現行の greedy と一致（乱数を使わない）
//   3. 実行順序 top_k → top_p → min_p → temperature（§3.1）。順序を入れ替えると
//      結果が変わる配置で、契約どおりの結果になることを確かめる
//   4. seed は `(seed, position)` の純粋関数（同じ入力なら同じ結果、位置で変わる）
//   5. seed 未指定は採番され、`done.seed` 用の値が決定的に返る
#include <cmath>
#include <cstdio>
#include <vector>

#include <shelf/api/sampling.hpp>

namespace {

using namespace shelf::api;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

void test_validate() {
    Sampling s;   // 既定はすべて有効
    std::string param;
    check(validate_sampling(s, &param).ok(), "既定は有効");

    Sampling nan_t;
    nan_t.temperature = std::nan("");
    auto st = validate_sampling(nan_t, &param);
    check(!st.ok() && param == "temperature", "NaN の temperature は拒否（param つき）");
    check(st.message().find("temperature") != std::string::npos, "message にフィールド名");

    Sampling neg;
    neg.temperature = -1.0;
    check(!validate_sampling(neg, &param).ok() && param == "temperature", "負の温度は拒否");

    Sampling tp0;
    tp0.temperature = 1.0;
    tp0.top_p = 0.0;
    check(!validate_sampling(tp0, &param).ok() && param == "top_p", "top_p=0 は拒否");
    tp0.top_p = 1.5;
    check(!validate_sampling(tp0, &param).ok(), "top_p>1 は拒否");

    Sampling mp;
    mp.min_p = -0.1;
    check(!validate_sampling(mp, &param).ok() && param == "min_p", "min_p<0 は拒否");
    mp.min_p = 1.1;
    check(!validate_sampling(mp, &param).ok(), "min_p>1 は拒否");

    Sampling inf_p;
    inf_p.top_p = std::numeric_limits<double>::infinity();
    check(!validate_sampling(inf_p, &param).ok(), "inf の top_p は拒否");
}

void test_greedy() {
    const std::vector<float> logits{0.5f, 3.0f, -1.0f, 3.0f, 2.0f};
    const auto r = sample(logits, Sampling{}, 0);
    check(r.ok(), "既定で抽選できる");
    check(r.value().index == 1u, "argmax（同値は添字の小さい方）");
    check(!r.value().used_random, "greedy は乱数を使わない");
    check(r.value().candidates == 1u, "greedy の候補は 1");

    // NaN は無視される
    std::vector<float> with_nan{std::nanf(""), 1.0f, 2.0f};
    const auto rn = sample(with_nan, Sampling{}, 0);
    check(rn.ok() && rn.value().index == 2u, "NaN を無視して argmax");

    // 全部 NaN は失敗
    std::vector<float> all_nan{std::nanf(""), std::nanf("")};
    check(!sample(all_nan, Sampling{}, 0).ok(), "有限値が無ければ失敗");
    check(!sample({}, Sampling{}, 0).ok(), "空 logits は失敗");
}

void test_top_k() {
    // top_k=1 は argmax と同じ（温度を掛けても 1 候補しか残らない）
    std::vector<float> logits{1.0f, 2.0f, 0.5f, 1.5f};
    Sampling s;
    s.temperature = 1.0;
    s.top_k = 1;
    for (std::uint64_t pos = 0; pos < 16; ++pos) {
        const auto r = sample(logits, s, pos);
        check(r.ok() && r.value().index == 1u, "top_k=1 は常に最大確率の候補");
    }
}

void test_order() {
    // §3.1 の順序を確かめる配置:
    //   logits = [0, 0, 5]（3 候補）で top_k=2 → [0, 5] が残る。
    //   さらに top_p=0.5 では 1 個（5 の方）だけが残る。
    //   temperature を先に掛ける実装だと、温度で分布が平らになり top_p の
    //   切り方が変わって候補が 2 個残りうる。契約どおりなら 1 個に絞られる。
    std::vector<float> logits{0.0f, 0.0f, 5.0f};
    Sampling s;
    s.top_k = 2;
    s.top_p = 0.5;
    s.temperature = 3.0;   // 温度を先に掛けると分布が平らになる（順序の判別に効く）
    for (std::uint64_t pos = 0; pos < 32; ++pos) {
        const auto r = sample(logits, s, pos);
        check(r.ok() && r.value().index == 2u, "top_k → top_p の順（温度は最後）");
        if (r.ok()) check(r.value().candidates == 1u, "絞り込み後の候補は 1 個");
    }
}

void test_min_p() {
    // min_p は最大確率に対する相対閾値。ほぼ同率の 2 候補だけが残る形を作る。
    std::vector<float> logits{1.0f, 1.0f, 0.0f};
    Sampling s;
    s.temperature = 1.0;
    s.min_p = 0.5;   // p(0)=p(1)≈0.42, p(2)≈0.155 → 0.5*0.42=0.21 で 2 は落ちる
    bool saw_only_first_two = true;
    for (std::uint64_t pos = 0; pos < 64; ++pos) {
        const auto r = sample(logits, s, pos);
        if (!r.ok() || (r.value().index != 0u && r.value().index != 1u))
            saw_only_first_two = false;
        if (r.ok()) check(r.value().candidates == 2u, "min_p で 2 候補に絞られる");
    }
    check(saw_only_first_two, "min_p で落ちた候補は選ばれない");
}

void test_seed_purity() {
    // Philox: 同じ (seed, position) は同じ値、位置が違えば変わる。
    const auto a = Philox4x32::bits(12345, 7);
    const auto b = Philox4x32::bits(12345, 7);
    const auto c = Philox4x32::bits(12345, 8);
    check(a == b, "同じ (seed, position) は同じワード");
    check(a != c, "位置が違えば違うワード");
    check(a != Philox4x32::bits(12346, 7), "seed が違えば違うワード");

    // 一様性の粗い確認（平均 0.5 付近、範囲 [0,1)）
    double sum = 0.0;
    constexpr std::uint64_t kN = 4096;
    for (std::uint64_t i = 0; i < kN; ++i) {
        const double u = Philox4x32::uniform(99, i);
        check(u >= 0.0 && u < 1.0, "uniform は [0,1)");
        sum += u;
    }
    const double mean = sum / static_cast<double>(kN);
    std::printf("  uniform mean = %.4f\n", mean);
    check(std::fabs(mean - 0.5) < 0.02, "uniform の平均は 0.5 付近");

    // 抽選の再現性: 同じ seed・同じ位置なら同じ結果
    std::vector<float> logits{0.2f, 0.9f, 0.4f, 0.7f};
    Sampling s;
    s.temperature = 1.0;
    s.seed = 42;
    const auto r1 = sample(logits, s, 3);
    const auto r2 = sample(logits, s, 3);
    check(r1.ok() && r2.ok() && r1.value().index == r2.value().index,
          "同じ (seed, position) は同じ抽選結果");
    const auto r3 = sample(logits, s, 4);
    check(r3.ok(), "位置を変えても抽選できる");

    // seed 未指定は採番される（決定的。0 も正規の値）
    const auto s1 = assign_seed(std::nullopt, 1);
    const auto s2 = assign_seed(std::nullopt, 1);
    check(s1 == s2, "採番は同じ nonce で決定的");
    check(assign_seed(std::nullopt, 1) != assign_seed(std::nullopt, 2), "nonce で変わる");
    check(assign_seed(0, 5) == 0, "seed 0 は正規の値（採番しない）");
    check(assign_seed(7, 5) == 7, "指定された seed はそのまま");
}

void test_temperature_effect() {
    // 温度が高いほど分布が平らになる（同じ seed で、最高確率の候補の当選率が下がる）。
    std::vector<float> logits{4.0f, 0.0f, 0.0f};
    const auto share_top = [&](double temp) {
        Sampling s;
        s.temperature = temp;
        s.seed = 7;
        std::size_t hits = 0;
        constexpr std::uint64_t kN = 2048;
        for (std::uint64_t pos = 0; pos < kN; ++pos) {
            const auto r = sample(logits, s, pos);
            if (r.ok() && r.value().index == 0u) ++hits;
        }
        return static_cast<double>(hits) / static_cast<double>(kN);
    };
    const double hot = share_top(0.5);
    const double cold = share_top(2.0);
    std::printf("  top-1 share: T=0.5 %.3f / T=2.0 %.3f\n", hot, cold);
    check(hot > cold, "温度を上げると最大確率の候補の当選率が下がる");
    check(hot > 0.9, "低温ではほぼ argmax に張り付く");
}

}  // namespace

int main() {
    test_validate();
    test_greedy();
    test_top_k();
    test_order();
    test_min_p();
    test_seed_purity();
    test_temperature_effect();
    std::printf("test_api_sampling: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
