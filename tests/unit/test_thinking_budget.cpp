#include <shelf/models/qwen35/runtime/runtime_request.h>

#include <cstdio>
#include <string>
#include <vector>

using shelf::qwen35::runtime::commit_sampled_token;
using shelf::qwen35::runtime::CommitAction;
using shelf::qwen35::runtime::FinishReason;
using shelf::qwen35::runtime::RuntimeRequest;
using shelf::qwen35::runtime::thinking_allowance;
using shelf::qwen35::runtime::thinking_budget_enabled;
using shelf::qwen35::StopTokens;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static RuntimeRequest make_request(uint32_t max_new_tokens, uint32_t budget,
                                   int32_t close_token) {
    RuntimeRequest request;
    request.max_new_tokens = max_new_tokens;
    request.thinking.tokens = budget;
    request.thinking.close_token = close_token;
    return request;
}

static void commit(RuntimeRequest& request, int32_t token) {
    (void)commit_sampled_token(request, token, StopTokens{});
}

static void injection_at_the_budget() {
    RuntimeRequest request = make_request(16, 3, 99);
    commit(request, 10);
    commit(request, 11);
    commit(request, 12);
    check(request.generated == std::vector<int32_t>{10, 11, 12},
          "tokens below the budget are kept");
    check(!request.thinking_closed, "thinking is open below the budget");

    commit(request, 13);
    check(request.generated == std::vector<int32_t>{10, 11, 12, 99},
          "the token at the budget becomes the close token");
    check(request.thinking_closed, "the injected close token ends the region");

    commit(request, 14);
    commit(request, 15);
    check(request.generated == std::vector<int32_t>{10, 11, 12, 99, 14, 15},
          "tokens after the close token are kept");
}

static void natural_close_token_is_not_injected() {
    RuntimeRequest request = make_request(16, 5, 99);
    commit(request, 10);
    commit(request, 99);
    check(request.generated == std::vector<int32_t>{10, 99},
          "the model's own close token is kept");
    check(request.thinking_closed, "the model's close token ends the region");

    commit(request, 11);
    commit(request, 12);
    commit(request, 13);
    commit(request, 14);
    commit(request, 15);
    check(request.generated == std::vector<int32_t>{10, 99, 11, 12, 13, 14, 15},
          "no injection after the region is closed");
}

static void budget_zero_means_no_budget() {
    RuntimeRequest request = make_request(8, 0, 99);
    check(!thinking_budget_enabled(request), "tokens == 0 disables the budget");
    check(thinking_allowance(request) == shelf::qwen35::runtime::kDFlash2NoThinkingAllowance,
          "no allowance when the budget is off");
    for (int32_t token = 10; token < 18; ++token) {
        commit(request, token);
    }
    check(request.generated_tokens == 8u, "the request ran to max_new_tokens");
    check(request.finish_reason == FinishReason::MaxNewTokens,
          "max_new_tokens still terminates");
    bool seen_close = false;
    for (const int32_t token : request.generated) {
        seen_close = seen_close || token == 99;
    }
    check(!seen_close, "no close token is injected without a budget");
}

static void missing_close_token_disables_the_budget() {
    RuntimeRequest request = make_request(8, 2, -1);
    check(!thinking_budget_enabled(request), "close_token < 0 disables the budget");
    commit(request, 10);
    commit(request, 11);
    commit(request, 12);
    check(request.generated == std::vector<int32_t>{10, 11, 12},
          "the sampled token survives");
}

static void allowance_tracks_generated_tokens() {
    RuntimeRequest request = make_request(16, 5, 99);
    check(thinking_allowance(request) == 5u, "allowance starts at the budget");
    commit(request, 10);
    check(thinking_allowance(request) == 4u, "one thinking token used");
    commit(request, 11);
    commit(request, 12);
    commit(request, 13);
    commit(request, 14);
    check(thinking_allowance(request) == 0u, "the budget is spent");
    check(thinking_allowance(request) == 0u, "allowance stays at zero");
}

static void eos_still_wins_over_the_budget_boundary() {
    RuntimeRequest request = make_request(3, 3, 99);
    StopTokens stop{257};
    (void)commit_sampled_token(request, 10, stop);
    (void)commit_sampled_token(request, 11, stop);
    const auto action = commit_sampled_token(request, 257, stop);
    check(action == CommitAction::Finished, "the stop token finishes the request");
    check(request.finish_reason == FinishReason::Eos,
          "eos wins when it coincides with the budget boundary");
    check(request.generated == std::vector<int32_t>{10, 11, 257},
          "the stop token is committed");
}

int main() {
    injection_at_the_budget();
    natural_close_token_is_not_injected();
    budget_zero_means_no_budget();
    missing_close_token_disables_the_budget();
    allowance_tracks_generated_tokens();
    eos_still_wins_over_the_budget_boundary();

    if (g_fail == 0) {
        std::printf("thinking_budget: ok\n");
        return 0;
    }
    std::printf("thinking_budget: failed=%d\n", g_fail);
    return 1;
}
