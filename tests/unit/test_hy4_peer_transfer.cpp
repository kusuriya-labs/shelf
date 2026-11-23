#include <shelf/models/hy4_flash/ops.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace shelf::hy4;

extern "C" hipError_t __real_hipEventRecord(hipEvent_t, hipStream_t);
extern "C" hipError_t __wrap_hipEventRecord(hipEvent_t event, hipStream_t stream) {
    auto status = hipEventQuery(event);
    if (status == hipErrorNotReady) {
        std::fprintf(stderr, "完了前の転送 event が再使用されました\n");
        return hipErrorIllegalState;
    }
    return status == hipSuccess ? __real_hipEventRecord(event, stream) : status;
}

namespace {
void check_transfer(int source, int destination) {
    constexpr int passes = 8;
    constexpr size_t bytes = Hy4MoePlan::packet_bytes(1);
    constexpr size_t words = bytes / sizeof(uint32_t);
    Checkpoint model;
    Residency gpu(model, 64);
    gpu.ranks = 2;
    gpu.physical[0] = source;
    gpu.physical[1] = destination;
    checked(hipSetDevice(destination));
    if (source != destination) {
        int accessible = 0;
        checked(hipDeviceCanAccessPeer(&accessible, destination, source));
        if (!accessible)
            throw std::runtime_error("この向きの peer 読み出しは利用できません");
        auto status = hipDeviceEnablePeerAccess(source, 0);
        gpu.owns_peer[1][0] = status == hipSuccess;
        if (status == hipErrorPeerAccessAlreadyEnabled)
            (void)hipGetLastError();
        else
            checked(status);
    }
    gpu.peer[1][0] = true;
    std::array<uint32_t *, 2> history{}, packet{};
    for (int r = 0; r < 2; ++r) {
        checked(hipSetDevice(gpu.physical[r]));
        auto arena = shelf::gpu::GpuArena::create(gpu.physical[r], (passes + 1) * bytes);
        if (!arena.ok())
            throw std::runtime_error(arena.status().message());
        gpu.arenas.push_back(arena.release());
        auto memory = gpu.arenas.back().allocate_aligned((passes + 1) * bytes, 256);
        if (!memory.ok())
            throw std::runtime_error(memory.status().message());
        history[r] = static_cast<uint32_t *>(memory.value().data());
        packet[r] = history[r] + passes * words;
        checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
        auto &events = r == 0 ? gpu.ready[0][1] : gpu.ack[1][0];
        for (auto &event : events)
            checked(hipEventCreateWithFlags(&event, hipEventDisableTiming | hipEventReleaseToSystem));
    }
    std::vector<uint32_t> expected(passes * words), actual(expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = uint32_t(i + 1) * 0x9e3779b9u ^ uint32_t(source * 4 + destination);
    checked(hipSetDevice(source));
    checked(hipMemcpy(history[0], expected.data(), passes * bytes, hipMemcpyHostToDevice));
    if (source == destination)
        checked(hipLaunchHostFunc(gpu.compute[0], [](void *) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }, nullptr));
    for (int pass = 0; pass < passes; ++pass) {
        checked(hipSetDevice(source));
        peer_copy(packet[0], history[0] + pass * words, bytes, gpu.compute[0]);
        gpu.transfer(packet[1], 1, packet[0], 0, bytes);
        peer_copy(history[1] + pass * words, packet[1], bytes, gpu.compute[1]);
    }
    checked(hipStreamSynchronize(gpu.compute[1]));
    checked(hipMemcpy(actual.data(), history[1], passes * bytes, hipMemcpyDeviceToHost));
    gpu.shutdown();
    auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
    if (mismatch.first != expected.end()) {
        size_t i = mismatch.first - expected.begin();
        std::fprintf(stderr, "転送 %zu 回目の offset=%zu で期待値 %08x、受信値 %08x\n",
                     i / words + 1, (i % words) * sizeof(uint32_t), *mismatch.first, *mismatch.second);
        throw std::runtime_error("peer 転送の内容が一致しません");
    }
}
}

int main() {
    try {
        int count = 0;
        auto status = hipGetDeviceCount(&count);
        if (status == hipErrorNoDevice || (status == hipSuccess && count == 0))
            return 77;
        checked(status);
        for (int source = 0; source < count; ++source)
            for (int destination = 0; destination < count; ++destination)
                if (source != destination || count == 1) {
                    std::fprintf(stderr, "peer 転送 GPU %d → GPU %d\n", source, destination);
                    check_transfer(source, destination);
                }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
