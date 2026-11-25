#include <shelf/runtime/runtime_request.h>

#include <cstdio>

using shelf::qwen_hybrid::StopTokens;
using shelf::runtime::CommitAction;
using shelf::runtime::FinishReason;
using shelf::runtime::RuntimeRequest;
using shelf::runtime::commit_sampled_token;

static int g_failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::printf("FAIL %s\n", name);
        ++g_failures;
    }
}

static StopTokens stop_tokens() {
    return StopTokens{120025};
}

int main() {
    RuntimeRequest normal;
    normal.max_new_tokens = 8;
    check(commit_sampled_token(normal, 120025, stop_tokens()) == CommitAction::Finished,
          "stop token finishes a normal request");
    check(normal.finish_reason == FinishReason::Eos, "normal finish reason is eos");

    RuntimeRequest ignored;
    ignored.max_new_tokens = 8;
    ignored.ignore_eos = true;
    check(commit_sampled_token(ignored, 120025, stop_tokens()) == CommitAction::Continue,
          "ignore_eos continues generation");
    check(ignored.finish_reason == FinishReason::None, "ignore_eos keeps finish reason none");
    check(ignored.generated_tokens == 1u && ignored.pending_decode_token == 120025,
          "ignored stop token is committed as a generated token");

    for (int i = 0; i < 6; ++i)
        check(commit_sampled_token(ignored, 11, stop_tokens()) == CommitAction::Continue,
              "generation continues up to the budget");
    check(commit_sampled_token(ignored, 12, stop_tokens()) == CommitAction::Finished,
          "ignore_eos still stops at max_new_tokens");
    check(ignored.finish_reason == FinishReason::MaxNewTokens,
          "length finish reason is preserved");
    return g_failures ? 1 : 0;
}
