#include "quality_core.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::printf("FAIL %s\n", name);
        ++g_failures;
    }
}

void check_near(double actual, double expected, double tolerance, const char* name) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
        std::printf("FAIL %s actual=%.17g expected=%.17g\n", name, actual, expected);
        ++g_failures;
    }
}

void test_distribution_math() {
    const float uniform[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    check_near(ps::bench::quality::log_sum_exp(uniform, 4), std::log(4.0), 1e-12,
               "log_sum_exp uniform");
    check_near(ps::bench::quality::row_nll(uniform, 4, 2), std::log(4.0), 1e-12,
               "row_nll uniform");
    check(ps::bench::quality::argmax(uniform, 4) == 0u, "argmax tie keeps lowest id");

    std::vector<uint32_t> top;
    ps::bench::quality::topk_ids(uniform, 4, 2u, top);
    check(top.size() == 2u && top[0] == 0u && top[1] == 1u, "topk tie order");

    const float base[2] = {0.0f, 0.0f};
    const float candidate[2] = {0.0f, std::log(3.0f)};
    ps::bench::quality::Metrics metrics;
    ps::bench::quality::accumulate_row(base, candidate, 2u, 1, metrics);
    check(metrics.positions == 1u, "positions");
    check_near(metrics.nll_base, std::log(2.0), 1e-12, "nll base");
    check_near(metrics.nll_candidate, -std::log(0.75), 1e-6, "nll candidate");
    check_near(metrics.kl, 0.14384103622589045, 1e-6, "kl");
    check(metrics.top1_match == 0u, "top1 mismatch");
    check_near(ps::bench::quality::top1_agreement(metrics), 0.0, 0.0, "top1 rate");
    check_near(ps::bench::quality::top5_overlap(metrics), 0.4, 1e-12, "top5 overlap");
    check_near(ps::bench::quality::perplexity(metrics.nll_base, metrics.positions),
               std::exp(std::log(2.0)), 1e-12, "perplexity");

    ps::bench::quality::Metrics identical;
    ps::bench::quality::accumulate_row(base, base, 2u, 0, identical);
    check_near(identical.kl, 0.0, 0.0, "kl identical");
    check(identical.top1_match == 1u, "top1 identical");
    check_near(identical.max_abs_delta, 0.0, 0.0, "delta identical");
}

void test_dump_roundtrip() {
    const std::string path = "test_quality_metrics.dump";
    std::remove(path.c_str());
    const uint32_t vocab = 4u;
    ps::bench::quality::DumpWriter writer;
    check(writer.open(path, vocab, ps::bench::quality::KvDtype::Fp8E4M3), "writer open");
    const float row0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float row1[4] = {-1.0f, 0.5f, 0.0f, 2.5f};
    check(writer.write(3, 16u, row0), "writer row0");
    check(writer.write(0, 17u, row1), "writer row1");
    check(writer.close(), "writer close");

    ps::bench::quality::DumpReader reader;
    std::string error;
    check(reader.open(path, error), "reader open");
    check(reader.header().vocab == vocab, "reader vocab");
    check(reader.header().positions == 2u, "reader positions");
    check(reader.header().kv_dtype ==
              static_cast<uint32_t>(ps::bench::quality::KvDtype::Fp8E4M3),
          "reader dtype");

    std::vector<float> logits;
    int32_t target = 0;
    uint32_t position = 0u;
    check(reader.read(logits, target, position, error), "reader row0");
    check(target == 3 && position == 16u, "reader row0 header");
    check(logits.size() == vocab && logits[3] == 4.0f, "reader row0 logits");
    check(!reader.done(), "reader not done");
    check(reader.read(logits, target, position, error), "reader row1");
    check(target == 0 && position == 17u, "reader row1 header");
    check(reader.done(), "reader done");
    check(!reader.read(logits, target, position, error), "reader eof");
    check(reader.close(), "reader close");
    std::remove(path.c_str());
}

}

int main() {
    test_distribution_math();
    test_dump_roundtrip();
    if (g_failures != 0) {
        std::printf("quality metric failures: %d\n", g_failures);
        return 1;
    }
    std::printf("quality metric contract ok\n");
    return 0;
}
