#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(e)                                                                       \
    do {                                                                            \
        hipError_t r_ = (e);                                                        \
        if (r_ != hipSuccess) {                                                     \
            std::fprintf(stderr, "%d: %s\n", __LINE__, hipGetErrorString(r_));      \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

constexpr int kWords = 2048;

__global__ void bump_epoch_kernel(int* epoch_slot) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int epoch = *epoch_slot + 1;
    __threadfence();
    *epoch_slot = epoch;
}

__global__ void push_kernel(int* peer_mail, int* peer_flag, const int* epoch_slot, int rank) {
    const int epoch = *epoch_slot;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kWords; i += blockDim.x * gridDim.x)
        peer_mail[i] = epoch * 100000 + rank * 7 + i * 3;
    __threadfence_system();
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
        __hip_atomic_store(peer_flag, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void sum_kernel(const int* own_mail, volatile int* own_flag,
                           const int* epoch_slot, int rank, int* out) {
    const int epoch = *epoch_slot;
    if (threadIdx.x == 0) {
        for (int spins = 0; spins < 100000000; ++spins) {
            if (__hip_atomic_load((int*)own_flag, __ATOMIC_ACQUIRE,
                                  __HIP_MEMORY_SCOPE_SYSTEM) >= epoch)
                break;
            __builtin_amdgcn_s_sleep(32);
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kWords; i += blockDim.x * gridDim.x)
        out[i] = (epoch * 100000 + rank * 7 + i * 3) + own_mail[i];
}

struct Rank {
    int* own_mail = nullptr;
    int* peer_mail = nullptr;
    int* out = nullptr;
    int* epoch_slot = nullptr;
    hipStream_t stream = nullptr;
    hipGraphExec_t graph = nullptr;
};

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
    int* host_flags = nullptr;
    CK(hipHostMalloc(&host_flags, 256, hipHostMallocMapped));
    host_flags[0] = 0;
    host_flags[1] = 0;

    Rank ranks[2];
    for (int d = 0; d < 2; ++d) {
        CK(hipSetDevice(d));
        hipError_t peer = hipDeviceEnablePeerAccess(1 - d, 0);
        if (peer != hipSuccess && peer != hipErrorPeerAccessAlreadyEnabled) CK(peer);
        (void)hipGetLastError();
        CK(hipMalloc(&ranks[d].own_mail, kWords * 4));
        CK(hipMalloc(&ranks[d].out, kWords * 4));
        CK(hipMalloc(&ranks[d].epoch_slot, 4));
        CK(hipMemset(ranks[d].epoch_slot, 0, 4));
        CK(hipStreamCreateWithFlags(&ranks[d].stream, hipStreamNonBlocking));
    }
    CK(hipSetDevice(0));
    CK(hipMemset(ranks[0].own_mail, 0, kWords * 4));
    CK(hipSetDevice(1));
    CK(hipMemset(ranks[1].own_mail, 0, kWords * 4));
    ranks[0].peer_mail = ranks[1].own_mail;
    ranks[1].peer_mail = ranks[0].own_mail;

    hipGraph_t graphs[2] = {nullptr, nullptr};
    for (int d = 0; d < 2; ++d) {
        CK(hipSetDevice(d));
        CK(hipStreamBeginCapture(ranks[d].stream, hipStreamCaptureModeThreadLocal));
        bump_epoch_kernel<<<1, 32, 0, ranks[d].stream>>>(ranks[d].epoch_slot);
        push_kernel<<<8, 256, 0, ranks[d].stream>>>(ranks[d].peer_mail, host_flags + (1 - d),
                                                    ranks[d].epoch_slot, d);
        sum_kernel<<<8, 256, 0, ranks[d].stream>>>(ranks[d].own_mail, host_flags + d,
                                                   ranks[d].epoch_slot, d, ranks[d].out);
        CK(hipGetLastError());
        CK(hipStreamEndCapture(ranks[d].stream, &graphs[d]));
        CK(hipGraphInstantiate(&ranks[d].graph, graphs[d], nullptr, nullptr, 0));
        size_t nodes = 0;
        CK(hipGraphGetNodes(graphs[d], nullptr, &nodes));
        std::printf("rank %d graph nodes=%zu\n", d, nodes);
    }

    hipEvent_t t0 = nullptr, t1 = nullptr;
    CK(hipSetDevice(0));
    CK(hipEventCreate(&t0));
    CK(hipEventCreate(&t1));
    for (int warm = 0; warm < 10; ++warm) {
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            CK(hipGraphLaunch(ranks[d].graph, ranks[d].stream));
        }
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            CK(hipStreamSynchronize(ranks[d].stream));
        }
    }
    CK(hipSetDevice(0));
    CK(hipEventRecord(t0, ranks[0].stream));
    for (int it = 0; it < iters; ++it) {
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            CK(hipGraphLaunch(ranks[d].graph, ranks[d].stream));
        }
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            CK(hipStreamSynchronize(ranks[d].stream));
        }
    }

    CK(hipSetDevice(0));
    CK(hipEventRecord(t1, ranks[0].stream));
    CK(hipStreamSynchronize(ranks[0].stream));
    float pair_ms = 0.0f;
    CK(hipEventElapsedTime(&pair_ms, t0, t1));
    std::printf("graph_seq per_step_us=%.2f\n", pair_ms * 1000.0f / iters);

    int bad = 0;
    for (int d = 0; d < 2; ++d) {
        CK(hipSetDevice(d));
        int check[kWords];
        CK(hipMemcpy(check, ranks[d].out, sizeof(check), hipMemcpyDeviceToHost));
        const int epoch = iters + 10;
        for (int i = 0; i < kWords; ++i)
            if (check[i] != epoch * 200000 + 7 + i * 6) ++bad;
    }
    std::printf("graph_seq iters=%d bad=%d\n", iters, bad);
    return bad == 0 ? 0 : 2;
}
