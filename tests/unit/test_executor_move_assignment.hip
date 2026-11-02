#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/runtime/stream_bridge.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <utility>
#include <vector>

namespace {

int g_failed = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        std::printf("FAIL: %s\n", name);
        ++g_failed;
    }
}

void seed_host_state(::ps::qwen35::Executor& ex) {
    ex.stochastic_topk_partitions = 5u;
    ex.stochastic_topk_scratch_bytes = 4096u;
}

void check_moved_host_state(const ::ps::qwen35::Executor& ex, const char* tag) {
    check(ex.stochastic_topk_partitions == 5u, tag);
    check(ex.stochastic_topk_scratch_bytes == 4096u, tag);
}

void check_cleared_host_state(const ::ps::qwen35::Executor& ex, const char* tag) {
    check(ex.stochastic_topk_partitions == 0u, tag);
    check(ex.stochastic_topk_scratch_bytes == 0u, tag);
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto signal = ::ps::runtime::stream_signal_alloc(64u);
    if (!signal.ok()) {
        std::printf("signal alloc failed: %s\n", signal.status().message().c_str());
        return 1;
    }

    {
        ::ps::qwen35::Executor src;
        src.completion_signal = signal.release();
        seed_host_state(src);
        ::ps::qwen35::Executor dst;
        dst = std::move(src);

        check(dst.completion_signal.device_ptr != nullptr,
              "move assignment keeps device_ptr");
        check(dst.completion_signal.ptr != nullptr, "move assignment keeps host ptr");
        check(dst.completion_signal.bytes == 64u, "move assignment keeps bytes");
        check(src.completion_signal.device_ptr == nullptr,
              "move assignment clears source device_ptr");
        check(src.completion_signal.ptr == nullptr,
              "move assignment clears source host ptr");
        check_moved_host_state(dst, "move assignment carries host state");
        check_cleared_host_state(src, "move assignment clears source host state");

        const ::ps::Status dst_st = ::ps::qwen35::executor_shutdown(dst);
        check(dst_st.ok(), "destination shutdown");
        const ::ps::Status src_st = ::ps::qwen35::executor_shutdown(src);
        check(src_st.ok(), "source shutdown");
    }

    {
        auto signal2 = ::ps::runtime::stream_signal_alloc(64u);
        if (!signal2.ok()) {
            std::printf("signal2 alloc failed: %s\n", signal2.status().message().c_str());
            return 1;
        }
        ::ps::qwen35::Executor src;
        src.completion_signal = signal2.release();
        seed_host_state(src);
        ::ps::qwen35::Executor dst(std::move(src));

        check(dst.completion_signal.bytes == 64u,
              "move constructor keeps signal bytes");
        check_moved_host_state(dst, "move constructor carries host state");
        check_cleared_host_state(src, "move constructor clears source host state");

        const ::ps::Status dst_st = ::ps::qwen35::executor_shutdown(dst);
        check(dst_st.ok(), "ctor destination shutdown");
        const ::ps::Status src_st = ::ps::qwen35::executor_shutdown(src);
        check(src_st.ok(), "ctor source shutdown");
    }

    if (g_failed != 0) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("move assignment ok\n");
    return 0;
}
