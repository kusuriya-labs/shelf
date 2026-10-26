// shelf unit test — SYCL device/buffer/stream/event (port of
// test_hip_device.cpp; plan Phase 2 criteria).
#include <cstdio>
#include <cstring>
#include <vector>

#include <shelf/backend/sycl/sycl_api.hpp>

namespace {

using namespace shelf;

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

void test_enumeration() {
    auto devices = shelf::sycl::enumerate_devices();
    check(devices.ok(), "enumerate ok");
    if (!devices.ok()) return;
    check(!devices.value().empty(), "at least one device");
    for (const auto& info : devices.value()) {
        check(info.kind == backend::DeviceKind::gpu ||
              info.kind == backend::DeviceKind::host, "kind is gpu or host");
        check(!info.name.empty(), "name non-empty");
        if (info.kind == backend::DeviceKind::gpu)
            check(info.total_memory_bytes > (1ull << 30), "memory > 1GiB");
        std::printf("device %d: %s (%.1f GiB, kind=%d)\n", info.id, info.name.c_str(),
                    static_cast<double>(info.total_memory_bytes) / (1ull << 30),
                    static_cast<int>(info.kind));
    }
}

void test_roundtrip() {
    auto devices = shelf::sycl::enumerate_devices();
    if (!devices.ok() || devices.value().empty()) { check(false, "no device for roundtrip"); return; }
    const auto dev = devices.value()[0].id;

    auto stream = shelf::sycl::Stream::create();
    check(stream.ok(), "stream create");
    auto buf = shelf::sycl::DeviceBuffer::allocate(4096 * sizeof(float), dev);
    check(buf.ok(), "buffer allocate");
    if (!buf.ok() || !stream.ok()) return;

    std::vector<float> src(4096), dst(4096, 0.0f);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i) * 0.25f;

    check(shelf::sycl::copy_h2d(buf.value(), src.data(), src.size() * sizeof(float), stream.value()).ok(),
          "copy h2d enqueue");
    check(stream.value().synchronize().ok(), "stream sync after h2d");
    check(shelf::sycl::copy_d2h(dst.data(), buf.value(), dst.size() * sizeof(float), stream.value()).ok(),
          "copy d2h enqueue");
    check(stream.value().synchronize().ok(), "stream sync after d2h");
    check(std::memcmp(src.data(), dst.data(), src.size() * sizeof(float)) == 0,
          "roundtrip bytes identical");

    auto v = buf.value().view();
    check(v.kind == backend::MemoryKind::device && v.device == dev, "view metadata");
    check(v.contains(0, v.bytes).ok(), "view full range");
    check(v.contains(0, v.bytes + 1).code() == Status::Code::out_of_range, "view overflow detected");
}

void test_event_between_streams() {
    auto devices = shelf::sycl::enumerate_devices();
    if (!devices.ok() || devices.value().empty()) { check(false, "no device"); return; }
    const auto dev = devices.value()[0].id;

    auto producer = shelf::sycl::Stream::create();
    auto consumer = shelf::sycl::Stream::create();
    auto start = shelf::sycl::Event::create(/*timing=*/true);
    auto done = shelf::sycl::Event::create(/*timing=*/true);
    auto a = shelf::sycl::DeviceBuffer::allocate(1 << 20, dev);
    auto b = shelf::sycl::DeviceBuffer::allocate(1 << 20, dev);
    check(producer.ok() && consumer.ok() && start.ok() && done.ok() && a.ok() && b.ok(),
          "multi-stream setup");
    if (!(producer.ok() && consumer.ok() && start.ok() && done.ok() && a.ok() && b.ok())) return;

    // unrecorded waits must be rejected (lifetime/dependency bug early-warning)
    check(done.value().wait(consumer.value()).code() == Status::Code::invalid_state,
          "wait before record rejected");
    check(done.value().synchronize().code() == Status::Code::invalid_state,
          "synchronize before record rejected");

    // producer: fill a, record start/done around traffic
    check(shelf::sycl::fill_zero(a.value(), 1 << 20, producer.value()).ok(), "fill a");
    check(start.value().record(producer.value()).ok(), "record start");
    check(shelf::sycl::copy_d2d(b.value(), a.value(), 1 << 20, producer.value()).ok(), "d2d a->b");
    check(done.value().record(producer.value()).ok(), "record done");

    // consumer waits producer completion without host synchronization,
    // then moves b back to host
    check(done.value().wait(consumer.value()).ok(), "consumer waits event");
    std::vector<char> host(1 << 20, 0x5A);
    check(shelf::sycl::copy_d2h(host.data(), b.value(), 1 << 20, consumer.value()).ok(), "d2h b");
    check(consumer.value().synchronize().ok(), "consumer sync");
    bool all_zero = true;
    for (char c : host) { if (c != 0) { all_zero = false; break; } }
    check(all_zero, "event dependency produced filled data");

    auto elapsed = shelf::sycl::Event::elapsed_us(start.value(), done.value());
    check(elapsed.ok() && elapsed.value() >= 0.0, "timing event elapsed sane");
}

void test_buffer_lifetime_race() {
    // Destroy a buffer while async work is still queued; then allocate again
    // and prove the device is still usable (runtime must order reuse).
    auto devices = shelf::sycl::enumerate_devices();
    if (!devices.ok() || devices.value().empty()) { check(false, "no device"); return; }
    const auto dev = devices.value()[0].id;
    auto stream = shelf::sycl::Stream::create();
    {
        auto big = shelf::sycl::DeviceBuffer::allocate(64ull << 20, dev);
        if (!big.ok() || !stream.ok()) { check(false, "64MiB alloc"); return; }
        check(shelf::sycl::fill_zero(big.value(), 64ull << 20, stream.value()).ok(), "async fill 64MiB");
        // no synchronize here: destructor runs with work in flight
    }
    check(stream.value().synchronize().ok(), "sync after destroy-with-work-in-flight");
    auto again = shelf::sycl::DeviceBuffer::allocate(1 << 20, dev);
    check(again.ok(), "reuse allocation after destruction");
}

void test_guards() {
    auto devices = shelf::sycl::enumerate_devices();
    if (!devices.ok() || devices.value().empty()) { check(false, "no device"); return; }
    const auto dev = devices.value()[0].id;
    auto stream = shelf::sycl::Stream::create();
    auto buf = shelf::sycl::DeviceBuffer::allocate(256, dev);
    if (!stream.ok() || !buf.ok()) return check(false, "guard setup");

    char host[64] = {};
    check(shelf::sycl::copy_h2d(buf.value(), host, 0, stream.value()).code() ==
              Status::Code::invalid_argument,
          "zero-byte transfer rejected");
    check(shelf::sycl::copy_d2h(host, buf.value(), 257, stream.value()).code() ==
              Status::Code::out_of_range,
          "oversized transfer rejected");
    check(shelf::sycl::DeviceBuffer::allocate(0, dev).code() == Status::Code::invalid_argument,
          "zero-size allocation rejected");
}

} // namespace

int main() {
    test_enumeration();
    test_roundtrip();
    test_event_between_streams();
    test_buffer_lifetime_race();
    test_guards();

    std::printf("test_sycl_device: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
