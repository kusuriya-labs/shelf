#include <phaseshift/runtime/stream_bridge.h>

#include <atomic>

namespace ps {
namespace runtime {

namespace {

void zero_words(uint32_t* words, std::size_t bytes) {
    for (std::size_t i = 0; i < bytes / sizeof(uint32_t); ++i) words[i] = 0u;
}

}  // namespace

const char* stream_signal_kind_name(StreamSignalKind kind) {
    switch (kind) {
        case StreamSignalKind::ExtSignal:
            return "hipExtMallocWithFlags(hipMallocSignalMemory)";
        case StreamSignalKind::HostCoherent:
            return "hipHostMalloc(coherent)";
        case StreamSignalKind::HostMapped:
            return "hipHostMalloc(mapped)";
        case StreamSignalKind::Device:
            return "hipMalloc(device)";
        default:
            return "none";
    }
}

Result<int> stream_wait_value_supported(int device) {
    int supported = 0;
    const hipError_t err = hipDeviceGetAttribute(
        &supported, hipDeviceAttributeCanUseStreamWaitValue, device);
    if (err != hipSuccess) {
        return Status::hip_error("stream wait value capability query",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return supported;
}

Result<StreamSignal> stream_signal_alloc(std::size_t bytes) {
    if (bytes == 0u || (bytes % sizeof(uint32_t)) != 0u) {
        return Status::invalid_argument(
            "signal memory size must be a nonzero multiple of 4", __FILE__,
            __LINE__);
    }
    void* ptr = nullptr;
    StreamSignalKind kind = StreamSignalKind::None;

    if (hipExtMallocWithFlags(&ptr, bytes, hipMallocSignalMemory) == hipSuccess) {
        kind = StreamSignalKind::ExtSignal;
    } else {
        (void)hipGetLastError();
        ptr = nullptr;
        if (hipHostMalloc(&ptr, bytes,
                          hipHostMallocMapped | hipHostMallocCoherent) ==
            hipSuccess) {
            kind = StreamSignalKind::HostCoherent;
        } else {
            (void)hipGetLastError();
            ptr = nullptr;
            if (hipHostMalloc(&ptr, bytes, hipHostMallocMapped) == hipSuccess) {
                kind = StreamSignalKind::HostMapped;
            } else {
                (void)hipGetLastError();
                ptr = nullptr;
            }
        }
    }
    if (kind == StreamSignalKind::None || ptr == nullptr) {
        return Status::hip_error("signal memory allocation",
                                 hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);
    }
    auto* words = static_cast<uint32_t*>(ptr);
    void* dev = ptr;
    if (kind == StreamSignalKind::HostCoherent ||
        kind == StreamSignalKind::HostMapped) {
        if (hipHostGetDevicePointer(&dev, ptr, 0) != hipSuccess) {
            (void)hipHostFree(ptr);
            return Status::hip_error("signal memory device mapping",
                                     hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
    }
    if (kind != StreamSignalKind::Device) zero_words(words, bytes);
    StreamSignal memory{};
    memory.ptr = words;
    memory.device_ptr = static_cast<uint32_t*>(dev);
    memory.bytes = bytes;
    memory.kind = kind;
    return memory;
}

Status stream_signal_free(const StreamSignal& memory) noexcept {
    if (memory.ptr == nullptr) return Status::make_ok();
    hipError_t err = hipSuccess;
    switch (memory.kind) {
        case StreamSignalKind::HostCoherent:
        case StreamSignalKind::HostMapped:
            err = hipHostFree(memory.ptr);
            break;
        case StreamSignalKind::Device:
        case StreamSignalKind::ExtSignal:
            err = hipFree(memory.ptr);
            break;
        default:
            return Status::make_ok();
    }
    if (err != hipSuccess) {
        return Status::hip_error("signal memory free", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status stream_write_value32(hipStream_t stream,
                                    const StreamSignal& memory,
                                    uint32_t value) {
    if (memory.device_ptr == nullptr) {
        return Status::invalid_argument("null signal memory", __FILE__, __LINE__);
    }
    const hipError_t err = hipStreamWriteValue32(
        stream, static_cast<void*>(memory.device_ptr), value, 0u);
    if (err != hipSuccess) {
        return Status::hip_error("stream write value32", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status stream_wait_value32(hipStream_t stream,
                                   const StreamSignal& memory,
                                   uint32_t value) {
    if (memory.device_ptr == nullptr) {
        return Status::invalid_argument("null signal memory", __FILE__, __LINE__);
    }
    const hipError_t err = hipStreamWaitValue32(
        stream, static_cast<void*>(memory.device_ptr), value,
        hipStreamWaitValueEq, 0xffffffffu);
    if (err != hipSuccess) {
        return Status::hip_error("stream wait value32", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

uint32_t stream_signal_load(const StreamSignal& memory) noexcept {
    if (memory.ptr == nullptr) return 0u;
    if (memory.kind != StreamSignalKind::HostCoherent &&
        memory.kind != StreamSignalKind::HostMapped) {
        return 0u;
    }
    return std::atomic_ref<uint32_t>(*memory.ptr)
        .load(std::memory_order_acquire);
}

}  // namespace runtime
}  // namespace ps
