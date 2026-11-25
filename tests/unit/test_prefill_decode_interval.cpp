#include <shelf/runtime/token_budget_scheduler.h>

#include <cstdio>
#include <vector>

using shelf::runtime::RequestState;
using shelf::runtime::RuntimeRequest;
using shelf::runtime::request_decode_ready;
using shelf::runtime::request_prefill_ready;

static int g_failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::printf("FAIL %s\n", name);
        ++g_failures;
    }
}

static RuntimeRequest make_request(RequestState state, uint32_t position,
                                   uint32_t input_tokens, bool waiting = false) {
    RuntimeRequest request;
    request.state = state;
    request.input_tokens.resize(input_tokens);
    request.sequence.position = position;
    request.pending_decode_token = position >= input_tokens ? 7 : -1;
    request.waiting_for_token_bitmask = waiting;
    return request;
}

int main() {
    RuntimeRequest prefill = make_request(RequestState::Active, 0u, 100u);
    RuntimeRequest decode = make_request(RequestState::Active, 100u, 100u);
    std::vector<const RuntimeRequest*> only_prefill{&prefill};
    std::vector<const RuntimeRequest*> only_decode{&decode};
    std::vector<const RuntimeRequest*> mixed{&prefill, &decode};

    check(request_prefill_ready(prefill), "prefill request is ready");
    check(!request_decode_ready(prefill), "prefill request is not a decode");
    check(request_decode_ready(decode), "decode request is ready");
    check(!request_prefill_ready(decode), "decode request is not a prefill");

    check(!shelf::runtime::prefill_decode_should_interrupt(mixed, 0u, 100u),
          "interval zero disables interruption");
    check(!shelf::runtime::prefill_decode_should_interrupt(mixed, 4u, 3u),
          "interruption waits for the interval");
    check(shelf::runtime::prefill_decode_should_interrupt(mixed, 4u, 4u),
          "interruption fires at the interval");
    check(shelf::runtime::prefill_decode_should_interrupt(mixed, 1u, 8u),
          "interruption fires after a long prefill sequence");
    check(!shelf::runtime::prefill_decode_should_interrupt(only_prefill, 1u, 8u),
          "no decode request to interrupt for");
    check(!shelf::runtime::prefill_decode_should_interrupt(only_decode, 1u, 8u),
          "no prefill request to interrupt");

    RuntimeRequest waiting = make_request(RequestState::Active, 100u, 100u, true);
    std::vector<const RuntimeRequest*> waiting_decode{&prefill, &waiting};
    check(!shelf::runtime::prefill_decode_should_interrupt(waiting_decode, 1u, 8u),
          "mask-waiting requests do not count as decodes");

    RuntimeRequest finished = make_request(RequestState::Finished, 100u, 100u);
    std::vector<const RuntimeRequest*> finished_decode{&prefill, &finished};
    check(!shelf::runtime::prefill_decode_should_interrupt(finished_decode, 1u, 8u),
          "finished requests do not count as decodes");
    return g_failures ? 1 : 0;
}
