#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/tp/tp_execution.h>

#include "phaseshift/runtime/tp/tp_transport_internal.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using ps::f32_to_bf16_rne;
using ps::runtime::ValueDType;
using ps::runtime::tp_bf16_bits_to_f32;
using ps::runtime::tp_reduce_f32_accumulate;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static std::uint16_t bf16(float value) {
    return f32_to_bf16_rne(value);
}

static void test_bf16_single_round() {
    const float tiny = 1.0f / 256.0f;
    std::vector<std::uint16_t> a{bf16(1.0f)};
    std::vector<std::uint16_t> b{bf16(tiny)};
    std::vector<std::uint16_t> c{bf16(tiny)};
    std::vector<const void*> srcs{a.data(), b.data(), c.data()};

    std::uint16_t out = 0;
    tp_reduce_f32_accumulate(srcs, 1, ValueDType::BF16, &out);

    const float acc =
        tp_bf16_bits_to_f32(a[0]) + tp_bf16_bits_to_f32(b[0]) + tp_bf16_bits_to_f32(c[0]);
    const std::uint16_t want = f32_to_bf16_rne(acc);
    check(out == want, "3 rank reduction rounds once in f32");

    const std::uint16_t pair1 =
        f32_to_bf16_rne(tp_bf16_bits_to_f32(a[0]) + tp_bf16_bits_to_f32(b[0]));
    const std::uint16_t pairwise =
        f32_to_bf16_rne(tp_bf16_bits_to_f32(pair1) + tp_bf16_bits_to_f32(c[0]));
    check(pairwise != want, "pairwise bf16 rounding differs for the fixture");
    check(out != pairwise, "reduction is not pairwise bf16 rounding");
}

static void test_bf16_two_rank_matches_single_round() {
    std::vector<std::uint16_t> a{bf16(3.5f)};
    std::vector<std::uint16_t> b{bf16(-1.25f)};
    std::vector<const void*> srcs{a.data(), b.data()};
    std::uint16_t out = 0;
    tp_reduce_f32_accumulate(srcs, 1, ValueDType::BF16, &out);
    const std::uint16_t want = f32_to_bf16_rne(
        tp_bf16_bits_to_f32(a[0]) + tp_bf16_bits_to_f32(b[0]));
    check(out == want, "2 rank reduction matches single f32 rounding");
}

static void test_f32_path() {
    std::vector<float> a{1.5f, -2.0f};
    std::vector<float> b{2.25f, 4.0f};
    std::vector<float> c{0.125f, 0.5f};
    std::vector<const void*> srcs{a.data(), b.data(), c.data()};
    std::vector<float> out(2, 0.0f);
    tp_reduce_f32_accumulate(srcs, 2, ValueDType::F32, out.data());
    check(out[0] == 1.5f + 2.25f + 0.125f, "f32 accumulate element 0");
    check(out[1] == -2.0f + 4.0f + 0.5f, "f32 accumulate element 1");
}

static void test_guards() {
    std::vector<std::uint16_t> a{bf16(1.0f)};
    std::vector<const void*> empty;
    std::uint16_t out = 0;
    tp_reduce_f32_accumulate(empty, 1, ValueDType::BF16, &out);
    check(out == 0, "empty source list leaves output untouched");
    std::vector<const void*> srcs{a.data()};
    tp_reduce_f32_accumulate(srcs, 0, ValueDType::BF16, &out);
    check(out == 0, "zero count leaves output untouched");
}

int main() {
    test_bf16_single_round();
    test_bf16_two_rank_matches_single_round();
    test_f32_path();
    test_guards();
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
