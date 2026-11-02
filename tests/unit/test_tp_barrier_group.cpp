#include <shelf/runtime/tp/tp_barrier_group.h>

#include "shelf/runtime/tp/tp_transport_internal.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>

using shelf::Status;
using shelf::runtime::TpBarrierGroup;
using shelf::runtime::TpSumInvocation;
using shelf::runtime::TpSumTarget;
using shelf::runtime::TpTransport;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static Status make_reason() {
    return Status::invalid_state("root cause", __FILE__, __LINE__);
}

class ValidatingTransport final : public TpTransport {
public:
    const char* name() const noexcept override { return "validating"; }

    Status sum_hidden(const TpSumInvocation& invocation) override {
        calls.fetch_add(1);
        std::size_t bytes = 0;
        return shelf::runtime::validate_tp_sum_invocation(
            invocation, static_cast<std::size_t>(invocation.targets.size()), bytes);
    }

    std::atomic<int> calls{0};
};

static TpSumTarget empty_target() { return TpSumTarget{}; }

static TpSumTarget filled_target(void* ptr) {
    TpSumTarget target;
    target.ptr = ptr;
    target.rows = 1;
    target.row_stride = 16;
    target.feature_count = 4;
    target.dtype = shelf::runtime::ValueDType::F32;
    return target;
}

static void arrive_both(TpBarrierGroup& group, const TpSumTarget& rank0,
                        const TpSumTarget& rank1, Status& out0, Status& out1) {
    std::thread t0([&] { out0 = group.arrive(0, rank0, nullptr, false); });
    std::thread t1([&] { out1 = group.arrive(1, rank1, nullptr, false); });
    t0.join();
    t1.join();
}

static void test_abort_reports_reason() {
    TpBarrierGroup group(2, {0, 1}, nullptr);

    check(!group.aborted(), "fresh barrier group is not aborted");
    check(group.abort_status().ok(), "fresh barrier group has no abort reason");

    const Status reason = make_reason();
    group.abort(reason);

    check(group.aborted(), "abort sets aborted");
    const Status stored = group.abort_status();
    check(stored.code() == reason.code(), "abort status keeps code");
    check(stored.message() == reason.message(), "abort status keeps message");

    TpSumTarget target;
    const Status arrived = group.arrive(0, target, nullptr, false);
    check(arrived.message() == reason.message(), "arrive reports abort reason");

    group.abort(Status::invalid_state("second abort", __FILE__, __LINE__));
    check(group.abort_status().message() == reason.message(),
          "abort keeps first reason");
}

static void test_empty_targets_skip_sum() {
    ValidatingTransport transport;
    TpBarrierGroup group(2, {0, 1}, &transport);
    Status out0 = Status::make_ok();
    Status out1 = Status::make_ok();
    arrive_both(group, empty_target(), empty_target(), out0, out1);

    check(out0.ok() && out1.ok(), "all-empty barrier returns ok on both ranks");
    check(transport.calls.load() == 0, "all-empty barrier skips the sum");
    check(!group.aborted(), "all-empty barrier does not abort");
}

static void test_filled_targets_sum_once() {
    int storage = 0;
    ValidatingTransport transport;
    TpBarrierGroup group(2, {0, 1}, &transport);
    Status out0 = Status::make_ok();
    Status out1 = Status::make_ok();
    arrive_both(group, filled_target(&storage), filled_target(&storage), out0, out1);

    check(out0.ok() && out1.ok(), "filled barrier returns ok on both ranks");
    check(transport.calls.load() == 1, "filled barrier sums once");
    check(!group.aborted(), "filled barrier does not abort");
}

static void test_partial_empty_keeps_guard() {
    int storage = 0;
    ValidatingTransport transport;
    TpBarrierGroup group(2, {0, 1}, &transport);
    Status out0 = Status::make_ok();
    Status out1 = Status::make_ok();
    arrive_both(group, empty_target(), filled_target(&storage), out0, out1);

    check(transport.calls.load() == 1, "partially empty barrier still validates");
    check(!out0.ok() && !out1.ok(), "partially empty barrier fails on both ranks");
    check(out0.message().find("tp sum target is empty") != std::string::npos,
          "partially empty barrier reports the empty target");
    check(group.aborted(), "partially empty barrier aborts");
}

int main() {
    test_abort_reports_reason();
    test_empty_targets_skip_sum();
    test_filled_targets_sum_once();
    test_partial_empty_keeps_guard();
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
