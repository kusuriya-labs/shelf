#include <shelf/models/hy4_flash/ops.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
void check_probe_fallback(int source, int destination) {
    constexpr int passes = 8;
    constexpr size_t probe_bytes = 4096;
    constexpr size_t bytes = Hy4MoePlan::packet_bytes(1);
    constexpr size_t words = bytes / sizeof(uint32_t);
    for (const char *mode : {"push", "peer"}) {
        for (bool corrupt : {false, true}) {
            setenv("SHELF_HY4_EP_TRANSPORT", mode, 1);
            Checkpoint model;
            Residency gpu(model, 64);
            gpu.ranks = 2;
            gpu.physical[0] = source;
            gpu.physical[1] = destination;
            std::array<uint32_t *, 2> history{}, packet{};
            for (int r = 0; r < 2; ++r) {
                checked(hipSetDevice(gpu.physical[r]));
                const size_t total = probe_bytes + (passes + 1) * bytes;
                auto arena = shelf::gpu::GpuArena::create(gpu.physical[r], total);
                if (!arena.ok())
                    throw std::runtime_error(arena.status().message());
                gpu.arenas.push_back(arena.release());
                auto memory = gpu.arenas.back().allocate_aligned(total, 256);
                if (!memory.ok())
                    throw std::runtime_error(memory.status().message());
                auto *base = static_cast<uint8_t *>(memory.value().data());
                gpu.workspace[r] = {base, probe_bytes};
                history[r] = reinterpret_cast<uint32_t *>(base + probe_bytes);
                packet[r] = history[r] + passes * words;
                checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
            }
            const bool host_reduce = gpu.configure_transports(
                corrupt ? static_cast<Residency::PeerProbe>(
                              [](void *dst, int dst_rank, const void *src, int src_rank, size_t n, hipStream_t stream,
                                 bool) -> bool {
                                  if (src_rank == 0 && dst_rank == 1)
                                      return hipMemsetAsync(dst, 0x5a, n, stream) == hipSuccess;
                                  peer_copy(dst, src, n, stream);
                                  return true;
                              })
                        : static_cast<Residency::PeerProbe>(
                              [](void *dst, int dst_rank, const void *src, int src_rank, size_t n, hipStream_t stream,
                                 bool) -> bool {
                                  if (src_rank == 0 && dst_rank == 1)
                                      return false;
                                  peer_copy(dst, src, n, stream);
                                  return true;
                              }));
            unsetenv("SHELF_HY4_EP_TRANSPORT");
            const bool forward_peer = gpu.push ? gpu.peer[0][1] : gpu.peer[1][0];
            const bool reverse_peer = gpu.push ? gpu.peer[1][0] : gpu.peer[0][1];
            if (host_reduce || forward_peer || !gpu.small_staging[0][1] || !reverse_peer || gpu.small_staging[1][0])
                throw std::runtime_error("プローブ失敗時の方向別 host フォールバック状態が不正です");
            for (int dir = 0; dir < 2; ++dir) {
                const int src = dir;
                const int dst = 1 - dir;
                std::vector<uint32_t> expected(passes * words), actual(expected.size());
                for (size_t i = 0; i < expected.size(); ++i)
                    expected[i] = uint32_t(i + 1) * 0x9e3779b9u ^ uint32_t(src * 13 + dst * 7 + corrupt);
                checked(hipSetDevice(gpu.physical[src]));
                checked(hipMemcpy(history[src], expected.data(), passes * bytes, hipMemcpyHostToDevice));
                for (int pass = 0; pass < passes; ++pass) {
                    checked(hipSetDevice(gpu.physical[src]));
                    peer_copy(packet[src], history[src] + pass * words, bytes, gpu.compute[src]);
                    gpu.transfer(packet[dst], dst, packet[src], src, bytes);
                    peer_copy(history[dst] + pass * words, packet[dst], bytes, gpu.compute[dst]);
                }
                checked(hipSetDevice(gpu.physical[dst]));
                checked(hipStreamSynchronize(gpu.compute[dst]));
                checked(hipMemcpy(actual.data(), history[dst], passes * bytes, hipMemcpyDeviceToHost));
                if (actual != expected)
                    throw std::runtime_error("フォールバック経路での転送内容が一致しません");
            }
            gpu.shutdown();
            std::fprintf(stderr, "probe fallback mode=%s corrupt=%d pass\n", mode, int(corrupt));
        }
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
                    if (source == 0)
                        check_probe_fallback(source, destination);
                }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
