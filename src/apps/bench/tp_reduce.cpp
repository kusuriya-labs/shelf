#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/tp/tp_barrier_group.h>
#include <shelf/runtime/tp/tp_transports.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using shelf::Status;
using shelf::runtime::HipPeerTpTransport;
using shelf::runtime::HostMediatedTpTransport;
using shelf::runtime::TpSumInvocation;
using shelf::runtime::TpSumTarget;
using shelf::runtime::TpTransport;
using shelf::runtime::ValueDType;
using shelf::runtime::create_tp_transport;

namespace {

float bf16_to_f32_host(std::uint16_t v) {
    const std::uint32_t u = static_cast<std::uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct BenchTarget {
    double min_gbps_large = 0.0;
    double max_us_rows1 = 0.0;
    bool enforced = false;
};

BenchTarget target_for(const std::string& backend) {
    if (backend == "hip-p2p") {
        return BenchTarget{20.0, 150.0, true};
    }
    if (backend == "host-mediated") {
        return BenchTarget{1.2, 400.0, true};
    }
    return BenchTarget{0.0, 0.0, false};
}

void usage() {
    std::printf("usage: shelf-bench tp-reduce [--rows 1,2,...] [--features N] "
                "[--iters N] [--backend auto|p2p|host] [--signal] [--group] [--check]\n");
    std::printf("  measures tp sum_hidden latency for the implemented transports\n");
    std::printf("  default rows: 1,2,4,8,16,32,64,128,256,512,1024,2048\n");
    std::printf("  default features: 5120 (Qwen3.8 hidden), dtype bf16\n");
    std::printf("  bandwidth definition: 2 * payload_bytes / elapsed (one read + one write)\n");
    std::printf("  --check enforces the backend target and fails when it is missed\n");
    std::printf("  --group measures TpBarrierGroup arrive throughput on two host threads\n");
    std::printf("  --signal uses the stream-signal barrier\n");
    std::printf("  targets: hip-p2p >= 20.0 GB/s (large rows) and <= 150 us (rows=1)\n");
    std::printf("           host-mediated >= 1.2 GB/s (large rows) and <= 400 us (rows=1)\n");
}

std::vector<uint64_t> parse_rows(const char* text) {
    std::vector<uint64_t> rows;
    const char* p = text;
    while (*p != '\0') {
        char* end = nullptr;
        const unsigned long long value = std::strtoull(p, &end, 10);
        if (end == p) break;
        if (value > 0) rows.push_back(value);
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    return rows;
}

struct BenchPair {
    void* buffer_a = nullptr;
    void* buffer_b = nullptr;
    hipStream_t stream_a = nullptr;
    hipStream_t stream_b = nullptr;
    hipEvent_t ready_a = nullptr;
    hipEvent_t ready_b = nullptr;
};

bool allocate_pair(BenchPair& pair, std::size_t bytes) {
    for (int device = 0; device < 2; ++device) {
        auto scope = shelf::gpu::ScopedDevice::create(device);
        if (!scope.ok()) return false;
        void** buffer = device == 0 ? &pair.buffer_a : &pair.buffer_b;
        hipStream_t* stream = device == 0 ? &pair.stream_a : &pair.stream_b;
        hipEvent_t* ready = device == 0 ? &pair.ready_a : &pair.ready_b;
        if (hipStreamCreate(stream) != hipSuccess) return false;
        if (hipEventCreate(ready) != hipSuccess) return false;
        if (hipMalloc(buffer, bytes) != hipSuccess) return false;
    }
    return true;
}

void free_pair(BenchPair& pair) {
    for (int device = 0; device < 2; ++device) {
        auto scope = shelf::gpu::ScopedDevice::create(device);
        if (!scope.ok()) continue;
        void** buffer = device == 0 ? &pair.buffer_a : &pair.buffer_b;
        hipStream_t* stream = device == 0 ? &pair.stream_a : &pair.stream_b;
        hipEvent_t* ready = device == 0 ? &pair.ready_a : &pair.ready_b;
        if (*buffer != nullptr) shelf::gpu::discard_cleanup_result(hipFree(*buffer));
        if (*ready != nullptr) shelf::gpu::discard_cleanup_result(hipEventDestroy(*ready));
        if (*stream != nullptr) shelf::gpu::discard_cleanup_result(hipStreamDestroy(*stream));
    }
}

Status upload(int device, void* buffer, const std::vector<std::uint8_t>& host,
              hipStream_t stream) {
    auto scope = shelf::gpu::ScopedDevice::create(device);
    if (!scope.ok()) return scope.status();
    if (hipMemcpyAsync(buffer, host.data(), host.size(), hipMemcpyHostToDevice, stream) !=
        hipSuccess) {
        return Status::hip_error("tp-reduce upload", hipGetErrorString(hipGetLastError()),
                                 __FILE__, __LINE__);
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        return Status::hip_error("tp-reduce upload sync",
                                 hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

struct RowResult {
    uint64_t rows = 0;
    double us = 0.0;
    double gbps = 0.0;
    bool ok = false;
};

RowResult run_row(TpTransport* transport, BenchPair& pair,
                  const std::vector<std::uint8_t>& host_a,
                  const std::vector<std::uint8_t>& host_b, uint64_t rows,
                  uint32_t features, int iters, bool signal) {
    RowResult result;
    result.rows = rows;
    const std::size_t bytes = static_cast<std::size_t>(rows) * features * 2u;
    TpSumTarget target_a;
    target_a.ptr = pair.buffer_a;
    target_a.rows = rows;
    target_a.row_stride = features;
    target_a.feature_count = features;
    target_a.dtype = ValueDType::BF16;
    TpSumTarget target_b = target_a;
    target_b.ptr = pair.buffer_b;
    std::vector<TpSumTarget> targets{target_a, target_b};
    std::vector<hipStream_t> streams{pair.stream_a, pair.stream_b};
    std::vector<hipEvent_t> ready{pair.ready_a, pair.ready_b};
    TpSumInvocation invocation{targets, streams, ready};

    const auto reset = [&]() -> Status {
        Status st = upload(0, pair.buffer_a, host_a, pair.stream_a);
        if (!st.ok()) return st;
        st = upload(1, pair.buffer_b, host_b, pair.stream_b);
        if (!st.ok()) return st;
        if (hipEventRecord(pair.ready_a, pair.stream_a) != hipSuccess ||
            hipEventRecord(pair.ready_b, pair.stream_b) != hipSuccess) {
            return Status::hip_error("tp-reduce record", hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        return Status::make_ok();
    };
    const auto run_sum = [&]() -> Status {
        Status st = Status::make_ok();
        if (signal) {
            st = transport->sum_hidden_stream_signal(0, target_a, pair.stream_a, false);
            if (st.ok())
                st = transport->sum_hidden_stream_signal(1, target_b, pair.stream_b, false);
            transport->advance_barrier(0);
            transport->advance_barrier(1);
        } else {
            st = transport->sum_hidden(invocation);
        }
        if (!st.ok()) return st;
        if (hipStreamSynchronize(pair.stream_a) != hipSuccess ||
            hipStreamSynchronize(pair.stream_b) != hipSuccess) {
            return Status::hip_error("tp-reduce sync", hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        return Status::make_ok();
    };

    Status st = reset();
    if (!st.ok()) return result;
    st = run_sum();
    if (!st.ok()) return result;
    if (signal) {
        const std::size_t count = static_cast<std::size_t>(rows) * features;
        std::vector<std::uint16_t> got_a(count);
        std::vector<std::uint16_t> got_b(count);
        {
            auto scope = shelf::gpu::ScopedDevice::create(0);
            if (!scope.ok()) return result;
            if (hipMemcpy(got_a.data(), pair.buffer_a, count * sizeof(std::uint16_t),
                          hipMemcpyDeviceToHost) != hipSuccess)
                return result;
        }
        {
            auto scope = shelf::gpu::ScopedDevice::create(1);
            if (!scope.ok()) return result;
            if (hipMemcpy(got_b.data(), pair.buffer_b, count * sizeof(std::uint16_t),
                          hipMemcpyDeviceToHost) != hipSuccess)
                return result;
        }
        const auto* host_a16 = reinterpret_cast<const std::uint16_t*>(host_a.data());
        const auto* host_b16 = reinterpret_cast<const std::uint16_t*>(host_b.data());
        bool mismatch = false;
        for (std::size_t i = 0; i < count && !mismatch; ++i) {
            const float va = bf16_to_f32_host(host_a16[i]);
            const float vb = bf16_to_f32_host(host_b16[i]);
            const std::uint16_t want = shelf::f32_to_bf16_rne(va + vb);
            if (got_a[i] != want || got_b[i] != want) mismatch = true;
        }
        if (mismatch) {
            std::printf("signal barrier result mismatch rows=%llu\n",
                        static_cast<unsigned long long>(rows));
            return result;
        }
    }
    result.ok = true;

    double total_us = 0.0;
    for (int it = 0; it < iters; ++it) {
        st = reset();
        if (!st.ok()) return result;
        const auto start = std::chrono::steady_clock::now();
        st = run_sum();
        const auto stop = std::chrono::steady_clock::now();
        if (!st.ok()) return result;
        total_us += std::chrono::duration<double, std::micro>(stop - start).count();
    }
    result.us = total_us / iters;
    result.gbps = (2.0 * static_cast<double>(bytes)) / result.us / 1e3;
    return result;
}

bool run_group(TpTransport* transport, BenchPair& pair, uint64_t rows, uint32_t features,
               int iters) {
    shelf::runtime::TpBarrierGroup group(2, {0, 1}, transport);
    Status st = group.initialize();
    if (!st.ok()) {
        std::printf(" group init failed: %s\n", st.message().c_str());
        return false;
    }
    std::atomic<int> failed{0};
    auto worker = [&](int rank, void* buffer, hipStream_t stream) {
        auto scope = shelf::gpu::ScopedDevice::create(rank);
        if (!scope.ok()) {
            failed.store(1);
            return;
        }
        TpSumTarget target;
        target.ptr = buffer;
        target.rows = rows;
        target.row_stride = features;
        target.feature_count = features;
        target.dtype = ValueDType::BF16;
        for (int i = 0; i < iters; ++i) {
            Status result = group.arrive(rank, target, stream, false);
            if (!result.ok()) {
                failed.store(1);
                group.abort(result);
                return;
            }
        }
    };
    const auto start = std::chrono::steady_clock::now();
    std::thread rank0(worker, 0, pair.buffer_a, pair.stream_a);
    std::thread rank1(worker, 1, pair.buffer_b, pair.stream_b);
    rank0.join();
    rank1.join();
    {
        auto scope = shelf::gpu::ScopedDevice::create(0);
        if (!scope.ok() || hipStreamSynchronize(pair.stream_a) != hipSuccess) failed.store(1);
    }
    {
        auto scope = shelf::gpu::ScopedDevice::create(1);
        if (!scope.ok() || hipStreamSynchronize(pair.stream_b) != hipSuccess) failed.store(1);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double total_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    const double us = total_us / static_cast<double>(iters);
    std::printf(" group rows=%llu features=%u barriers=%d us/barrier=%.1f\n",
                static_cast<unsigned long long>(rows), features, iters, us);
    group.shutdown();
    return failed.load() == 0;
}

bool run_backend(TpTransport* transport, BenchPair& pair,
                 const std::vector<std::uint8_t>& host_a,
                 const std::vector<std::uint8_t>& host_b,
                 const std::vector<uint64_t>& rows, uint32_t features, int iters,
                 bool check, bool signal) {
    const std::string backend = transport->name();
    std::printf("backend: %s\n", backend.c_str());
    BenchTarget target = target_for(backend);
    double us_rows1 = 0.0;
    double gbps_large = 0.0;
    for (uint64_t r : rows) {
        RowResult result = run_row(transport, pair, host_a, host_b, r, features, iters,
                                   signal);
        std::printf("  %-14s rows=%-5llu payload=%-9llu us=%.1f GB/s=%.2f\n",
                    backend.c_str(), static_cast<unsigned long long>(r),
                    static_cast<unsigned long long>(r) * features * 2u, result.us,
                    result.gbps);
        if (!result.ok) {
            std::printf("  %-14s rows=%-5llu FAILED\n", backend.c_str(),
                        static_cast<unsigned long long>(r));
            return false;
        }
        if (r == 1) us_rows1 = result.us;
        if (r == rows.back()) gbps_large = result.gbps;
    }
    if (!target.enforced) return true;
    const bool gbps_ok = gbps_large >= target.min_gbps_large;
    const bool us_ok = us_rows1 <= target.max_us_rows1;
    std::printf("target %-14s GB/s>=%.2f (%s) us(rows=1)<=%.0f (%s) -> %s\n",
                backend.c_str(), target.min_gbps_large, gbps_ok ? "PASS" : "FAIL",
                target.max_us_rows1, us_ok ? "PASS" : "FAIL",
                (gbps_ok && us_ok) ? "PASS" : "FAIL");
    return !check || (gbps_ok && us_ok);
}

}

int run_tp_reduce(int argc, char** argv) {
    std::vector<uint64_t> rows{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    uint32_t features = 5120;
    int iters = 50;
    std::string backend = "auto";
    bool check = false;
    bool signal = false;
    bool group_bench = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else if (arg == "--rows" && i + 1 < argc) {
            rows = parse_rows(argv[++i]);
        } else if (arg == "--features" && i + 1 < argc) {
            features = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
        } else if (arg == "--check") {
            check = true;
        } else if (arg == "--signal") {
            signal = true;
        } else if (arg == "--group") {
            group_bench = true;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            usage();
            return 1;
        }
    }
    if (rows.empty() || features == 0 || iters <= 0) {
        std::fprintf(stderr, "invalid rows/features/iters\n");
        return 1;
    }
    if (backend != "auto" && backend != "p2p" && backend != "host") {
        std::fprintf(stderr, "invalid backend: %s\n", backend.c_str());
        return 1;
    }
    std::sort(rows.begin(), rows.end());

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::fprintf(stderr, "tp-reduce requires 2 GPUs (found %d)\n", device_count);
        return 1;
    }

    std::uint64_t max_rows = 0;
    for (uint64_t r : rows) max_rows = r > max_rows ? r : max_rows;
    const std::size_t max_bytes =
        static_cast<std::size_t>(max_rows) * features * sizeof(std::uint16_t);

    BenchPair pair;
    if (!allocate_pair(pair, max_bytes)) {
        std::fprintf(stderr, "failed to allocate benchmark buffers\n");
        return 1;
    }

    std::vector<std::uint8_t> host_a(max_bytes);
    std::vector<std::uint8_t> host_b(max_bytes);
    for (std::size_t i = 0; i < max_bytes; ++i) {
        host_a[i] = static_cast<std::uint8_t>(i * 31u + 7u);
        host_b[i] = static_cast<std::uint8_t>(i * 17u + 3u);
    }

    const std::vector<int> devices{0, 1};
    std::printf("tp-reduce devices=0,1 features=%u dtype=bf16 iters=%d\n", features,
                iters);
    std::printf("bandwidth definition: 2 * payload_bytes / elapsed\n");

    bool all_ok = true;
    bool measured = false;

    if (backend == "auto" || backend == "p2p") {
        std::unique_ptr<TpTransport> peer;
        if (backend == "p2p") {
            ::setenv("SHELF_TP_TRANSPORT", "p2p", 1);
        }
        auto selection = create_tp_transport(
            devices, signal ? shelf::runtime::TpCombineBackend::StreamSignal
                            : shelf::runtime::TpCombineBackend::Event);
        if (selection.ok() && std::string(selection.value().transport->name()) == "hip-p2p") {
            const Status verify = selection.value().transport->verify();
            if (verify.ok()) {
                peer = std::move(selection.value().transport);
            } else {
                std::printf("[tp] hip-p2p self-test failed (%s)\n", verify.message().c_str());
            }
        }
        if (backend == "p2p") {
            if (peer == nullptr) {
                std::printf("hip-p2p is not usable on this host\n");
                all_ok = false;
            }
        }
        if (peer != nullptr) {
            const Status reserve_status = peer->reserve(max_bytes);
            if (!reserve_status.ok()) {
                std::printf("tp inbox reserve failed: %s\n",
                            reserve_status.message().c_str());
                free_pair(pair);
                return 1;
            }
            if (signal && !peer->stream_signal_ready()) {
                std::printf("hip-p2p stream signal barrier is not available\n");
                free_pair(pair);
                return 1;
            }
            measured = true;
            if (group_bench) {
                all_ok = run_group(peer.get(), pair, rows.front(), features, iters) && all_ok;
            } else {
                all_ok = run_backend(peer.get(), pair, host_a, host_b, rows, features,
                                     iters, check, signal) &&
                         all_ok;
            }
        }
    }

    if (!group_bench && (backend == "auto" || backend == "host" || !measured)) {
        HostMediatedTpTransport host_transport(devices);
        const Status host_reserve = host_transport.reserve(max_bytes);
        if (!host_reserve.ok()) {
            std::printf("host transport reserve failed: %s\n",
                        host_reserve.message().c_str());
            free_pair(pair);
            return 1;
        }
        if (check && backend == "p2p") {
            std::printf("skipping host-mediated measurement for --backend p2p\n");
        } else {
            all_ok = run_backend(&host_transport, pair, host_a, host_b, rows, features,
                                 iters, check, false) &&
                     all_ok;
        }
    }

    free_pair(pair);
    return all_ok ? 0 : 1;
}
