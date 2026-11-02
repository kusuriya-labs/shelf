#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/runtime/tp/tp_transports.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using ps::Status;
using ps::runtime::HipPeerTpTransport;
using ps::runtime::HostMediatedTpTransport;
using ps::runtime::TpSumInvocation;
using ps::runtime::TpSumTarget;
using ps::runtime::TpTransport;
using ps::runtime::ValueDType;
using ps::runtime::create_tp_transport;

namespace {

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
    std::printf("usage: phaseshift-bench tp-reduce [--rows 1,2,...] [--features N] "
                "[--iters N] [--backend auto|p2p|host] [--check]\n");
    std::printf("  measures tp sum_hidden latency for the implemented transports\n");
    std::printf("  default rows: 1,2,4,8,16,32,64,128,256,512,1024,2048\n");
    std::printf("  default features: 5120 (Qwen3.8 hidden), dtype bf16\n");
    std::printf("  bandwidth definition: 2 * payload_bytes / elapsed (one read + one write)\n");
    std::printf("  --check enforces the backend target and fails when it is missed\n");
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
        auto scope = ps::gpu::ScopedDevice::create(device);
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
        auto scope = ps::gpu::ScopedDevice::create(device);
        if (!scope.ok()) continue;
        void** buffer = device == 0 ? &pair.buffer_a : &pair.buffer_b;
        hipStream_t* stream = device == 0 ? &pair.stream_a : &pair.stream_b;
        hipEvent_t* ready = device == 0 ? &pair.ready_a : &pair.ready_b;
        if (*buffer != nullptr) ps::gpu::discard_cleanup_result(hipFree(*buffer));
        if (*ready != nullptr) ps::gpu::discard_cleanup_result(hipEventDestroy(*ready));
        if (*stream != nullptr) ps::gpu::discard_cleanup_result(hipStreamDestroy(*stream));
    }
}

Status upload(int device, void* buffer, const std::vector<std::uint8_t>& host,
              hipStream_t stream) {
    auto scope = ps::gpu::ScopedDevice::create(device);
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
                  uint32_t features, int iters) {
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
        Status st = transport->sum_hidden(invocation);
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

bool run_backend(TpTransport* transport, BenchPair& pair,
                 const std::vector<std::uint8_t>& host_a,
                 const std::vector<std::uint8_t>& host_b,
                 const std::vector<uint64_t>& rows, uint32_t features, int iters,
                 bool check) {
    const std::string backend = transport->name();
    std::printf("backend: %s\n", backend.c_str());
    BenchTarget target = target_for(backend);
    double us_rows1 = 0.0;
    double gbps_large = 0.0;
    for (uint64_t r : rows) {
        RowResult result = run_row(transport, pair, host_a, host_b, r, features, iters);
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
            ::setenv("PHASESHIFT_TP_TRANSPORT", "p2p", 1);
        }
        auto selection = create_tp_transport(devices);
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
            measured = true;
            all_ok = run_backend(peer.get(), pair, host_a, host_b, rows, features, iters,
                                 check) &&
                     all_ok;
        }
    }

    if (backend == "auto" || backend == "host" || !measured) {
        HostMediatedTpTransport host_transport(devices);
        if (check && backend == "p2p") {
            std::printf("skipping host-mediated measurement for --backend p2p\n");
        } else {
            all_ok = run_backend(&host_transport, pair, host_a, host_b, rows, features,
                                 iters, check) &&
                     all_ok;
        }
    }

    free_pair(pair);
    return all_ok ? 0 : 1;
}
