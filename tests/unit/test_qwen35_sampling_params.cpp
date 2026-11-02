#include <shelf/models/qwen35/runtime/scheduling_types.h>

#include <cmath>
#include <cstdio>
#include <limits>

using ps::qwen35::runtime::SamplingConfig;
using ps::qwen35::runtime::SamplingMode;
using ps::qwen35::runtime::make_sampling_request;
using ps::qwen35::runtime::sampling_config_is_default;
using ps::qwen35::runtime::validate_sampling_config;

static int g_fail = 0;

static void expect_ok(const SamplingConfig& c, const char* msg) {
    const ps::Status st = validate_sampling_config(c);
    const bool ok = st.ok();
    if (!ok) ++g_fail;
    std::printf("  [%-28s] %s %s\n", msg, ok ? "PASS" : "FAIL",
                ok ? "" : st.message().c_str());
}

static void expect_reject(const SamplingConfig& c, const char* msg) {
    const ps::Status st = validate_sampling_config(c);
    const bool ok = !st.ok();
    if (!ok) ++g_fail;
    std::printf("  [%-28s] %s\n", msg, ok ? "PASS" : "FAIL");
}

static void expect_mode(const SamplingConfig& c, bool sample, uint64_t index,
                        SamplingMode want, const char* msg) {
    const auto r = make_sampling_request(c, sample, index);
    const bool ok = r.mode == want && r.sample_index == index &&
                    r.temperature == c.temperature && r.top_p == c.top_p &&
                    r.min_p == c.min_p && r.top_k == c.top_k && r.seed == c.seed;
    if (!ok) ++g_fail;
    std::printf("  [%-28s] %s\n", msg, ok ? "PASS" : "FAIL");
}

int main() {
    std::printf("sampling config validation\n");

    SamplingConfig def;
    expect_ok(def, "default");
    if (!sampling_config_is_default(def)) {
        ++g_fail;
        std::printf("  [default is default]           FAIL\n");
    } else {
        std::printf("  [default is default]           PASS\n");
    }

    for (float t : {0.0f, 0.7f, 1.0f, 2.5f}) {
        SamplingConfig c;
        c.temperature = t;
        expect_ok(c, "temperature accepted");
    }
    for (float p : {0.1f, 0.5f, 0.9f, 1.0f}) {
        SamplingConfig c;
        c.temperature = 0.7f;
        c.top_p = p;
        expect_ok(c, "top_p accepted");
    }
    for (float p : {0.0f, 0.05f, 0.5f, 1.0f}) {
        SamplingConfig c;
        c.temperature = 0.7f;
        c.min_p = p;
        expect_ok(c, "min_p accepted");
    }
    {
        SamplingConfig c;
        c.temperature = 0.7f;
        c.top_p = 0.9f;
        c.min_p = 0.05f;
        c.top_k = 16u;
        c.seed = 42u;
        expect_ok(c, "top_k + min_p + seed accepted");
    }
    {
        SamplingConfig c;
        c.temperature = 0.7f;
        c.min_p = 0.1f;
        if (sampling_config_is_default(c)) {
            ++g_fail;
            std::printf("  [min_p is not default]         FAIL\n");
        } else {
            std::printf("  [min_p is not default]         PASS\n");
        }
    }

    {
        SamplingConfig c;
        c.temperature = -0.1f;
        expect_reject(c, "negative temperature");
    }
    {
        SamplingConfig c;
        c.temperature = std::numeric_limits<float>::quiet_NaN();
        expect_reject(c, "NaN temperature");
    }
    {
        SamplingConfig c;
        c.temperature = std::numeric_limits<float>::infinity();
        expect_reject(c, "inf temperature");
    }
    {
        SamplingConfig c;
        c.top_p = 0.0f;
        expect_reject(c, "zero top_p");
    }
    {
        SamplingConfig c;
        c.top_p = -0.5f;
        expect_reject(c, "negative top_p");
    }
    {
        SamplingConfig c;
        c.top_p = 1.0001f;
        expect_reject(c, "top_p above one");
    }
    {
        SamplingConfig c;
        c.top_p = std::numeric_limits<float>::quiet_NaN();
        expect_reject(c, "NaN top_p");
    }
    {
        SamplingConfig c;
        c.min_p = -0.1f;
        expect_reject(c, "negative min_p");
    }
    {
        SamplingConfig c;
        c.min_p = 1.0001f;
        expect_reject(c, "min_p above one");
    }
    {
        SamplingConfig c;
        c.min_p = std::numeric_limits<float>::quiet_NaN();
        expect_reject(c, "NaN min_p");
    }

    std::printf("sampling request mapping\n");
    expect_mode(def, false, 0u, SamplingMode::None, "sample=false -> None");
    expect_mode(def, true, 0u, SamplingMode::Greedy, "temp=0 -> Greedy");
    expect_mode(def, true, 7u, SamplingMode::Greedy, "index passthrough");

    SamplingConfig hot;
    hot.temperature = 0.7f;
    hot.top_p = 0.9f;
    hot.min_p = 0.05f;
    hot.top_k = 16u;
    hot.seed = 123u;
    expect_mode(hot, false, 3u, SamplingMode::None, "hot but sample=false");
    expect_mode(hot, true, 3u, SamplingMode::Stochastic, "temp>0 -> Stochastic");

    // top_p / top_k must not change the greedy decision.
    SamplingConfig greedy_topk;
    greedy_topk.temperature = 0.0f;
    greedy_topk.top_p = 0.5f;
    greedy_topk.top_k = 1u;
    expect_mode(greedy_topk, true, 0u, SamplingMode::Greedy, "temp=0 keeps Greedy");

    std::printf("%s: %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
