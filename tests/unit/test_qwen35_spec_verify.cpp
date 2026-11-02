#include <phaseshift/models/qwen35/runtime/spec_decode.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

using ps::qwen35::runtime::SpecPhase;
using ps::qwen35::runtime::SpecVerifyResult;

bool tokens_equal(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    return a == b;
}

void expect_result(
    const char* name,
    const std::vector<int32_t>& drafts,
    const std::vector<int32_t>& target_tokens,
    bool bonus_enabled,
    uint32_t expected_accepted,
    uint32_t expected_first_reject,
    bool expected_correction,
    int32_t expected_correction_token,
    bool expected_bonus,
    int32_t expected_bonus_token,
    const std::vector<int32_t>& expected_emitted,
    uint32_t expected_committed,
    const char* expected_name) {

    SpecVerifyResult out;
    auto st = ps::qwen35::runtime::spec_greedy_accept(
        drafts.empty() ? nullptr : drafts.data(),
        static_cast<uint32_t>(drafts.size()),
        target_tokens.data(),
        bonus_enabled,
        out);
    check(st.ok(), std::string(name) + ": accept ok");
    check(std::string(ps::qwen35::runtime::spec_phase_name(SpecPhase::IDLE)) == "IDLE",
          std::string(name) + ": phase name IDLE");
    (void)expected_name;
    check(out.num_drafts == drafts.size(), std::string(name) + ": num_drafts");
    check(out.num_accepted_drafts == expected_accepted, std::string(name) + ": num_accepted");
    check(out.first_reject_index == expected_first_reject, std::string(name) + ": first_reject");
    check(out.correction_valid == expected_correction, std::string(name) + ": correction_valid");
    if (expected_correction) {
        check(out.correction_token == expected_correction_token, std::string(name) + ": correction_token");
    }
    check(out.bonus_token_valid == expected_bonus, std::string(name) + ": bonus_valid");
    if (expected_bonus) {
        check(out.bonus_token == expected_bonus_token, std::string(name) + ": bonus_token");
    }
    check(tokens_equal(out.emitted_tokens, expected_emitted), std::string(name) + ": emitted");
    check(out.num_emitted_tokens == expected_emitted.size(), std::string(name) + ": emitted_count");
    check(out.num_committed_target_tokens == expected_committed, std::string(name) + ": committed");
}

}  // namespace

int main() {
    check(std::string(ps::qwen35::runtime::spec_phase_name(SpecPhase::IDLE)) == "IDLE", "phase IDLE");
    check(std::string(ps::qwen35::runtime::spec_phase_name(SpecPhase::DRAFTING)) == "DRAFTING", "phase DRAFTING");
    check(std::string(ps::qwen35::runtime::spec_phase_name(SpecPhase::VERIFYING)) == "VERIFYING", "phase VERIFYING");
    check(std::string(ps::qwen35::runtime::spec_phase_name(SpecPhase::COMMITTING)) == "COMMITTING", "phase COMMITTING");
    check(std::string(ps::qwen35::runtime::spec_phase_name(SpecPhase::ROLLING_BACK)) == "ROLLING_BACK", "phase ROLLING_BACK");

    expect_result(
        "all_accept_bonus",
        {10, 11, 12, 13},
        {10, 11, 12, 13, 99},
        true,
        4u, 0xFFFFFFFFu, false, -1, true, 99,
        {10, 11, 12, 13, 99}, 5u, "all_accept_bonus");

    expect_result(
        "all_accept_no_bonus",
        {10, 11, 12, 13},
        {10, 11, 12, 13, 99},
        false,
        4u, 0xFFFFFFFFu, false, -1, false, -1,
        {10, 11, 12, 13}, 5u, "all_accept_no_bonus");

    expect_result(
        "reject_first",
        {10, 11, 12, 13},
        {7, 88, 77, 66, 99},
        true,
        0u, 0u, true, 7, false, -1,
        {7}, 1u, "reject_first");

    expect_result(
        "reject_mid",
        {10, 11, 12, 13},
        {10, 11, 7, 13, 99},
        true,
        2u, 2u, true, 7, false, -1,
        {10, 11, 7}, 3u, "reject_mid");

    expect_result(
        "reject_last",
        {10, 11, 12, 13},
        {10, 11, 12, 7, 99},
        true,
        3u, 3u, true, 7, false, -1,
        {10, 11, 12, 7}, 4u, "reject_last");

    expect_result(
        "k1_accept",
        {5},
        {5, 9},
        true,
        1u, 0xFFFFFFFFu, false, -1, true, 9,
        {5, 9}, 2u, "k1_accept");

    expect_result(
        "k1_reject",
        {5},
        {4, 9},
        true,
        0u, 0u, true, 4, false, -1,
        {4}, 1u, "k1_reject");

    expect_result(
        "k0_no_drafts",
        {},
        {9},
        true,
        0u, 0u, false, -1, false, -1,
        {}, 0u, "k0_no_drafts");

    {
        SpecVerifyResult out;
        auto st = ps::qwen35::runtime::spec_greedy_accept(nullptr, 0u, nullptr, true, out);
        check(!st.ok(), "null target_tokens rejected");
    }
    {
        SpecVerifyResult out;
        auto st = ps::qwen35::runtime::spec_greedy_accept(nullptr, 2u, nullptr, true, out);
        check(!st.ok(), "null target_tokens with drafts rejected");
    }
    {
        const int32_t target[2] = {1, 2};
        SpecVerifyResult out;
        auto st = ps::qwen35::runtime::spec_greedy_accept(nullptr, 2u, target, true, out);
        check(!st.ok(), "null drafts with num_drafts>0 rejected");
    }

    std::printf("test_qwen35_spec_verify: passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
