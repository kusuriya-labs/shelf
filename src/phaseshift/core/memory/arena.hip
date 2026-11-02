#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/status.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/core/memory/device_allocation_view.h>
#include <cstring>
#include <iostream>

namespace ps {
namespace gpu {

namespace {

constexpr std::size_t kTargetCommitQuantum = 256ull * 1024ull * 1024ull;

bool vmm_requested() {
    const char* e = std::getenv("PHASESHIFT_ARENA_VMM");
    return e != nullptr && std::strcmp(e, "1") == 0;
}

Status round_up_multiple(std::size_t value, std::size_t multiple, std::size_t& out) {
    if (multiple == 0) {
        return Status::invalid_argument("multiple must be non-zero", __FILE__, __LINE__);
    }
    const std::size_t rem = value % multiple;
    if (rem == 0) {
        out = value;
        return Status::make_ok();
    }
    const std::size_t add = multiple - rem;
    if (value > std::size_t(-1) - add) {
        return Status::overflow("round-up overflow", __FILE__, __LINE__);
    }
    out = value + add;
    return Status::make_ok();
}

hipMemAllocationProp device_allocation_prop(int device) {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    return prop;
}

}

Result<GpuArena> GpuArena::create(int physical_device, std::size_t capacity_bytes) {
    if (physical_device < 0) {
        return Status::invalid_argument("invalid device", __FILE__, __LINE__);
    }

    if (capacity_bytes == 0) {
        return Status::invalid_argument("capacity must be non-zero", __FILE__, __LINE__);
    }

    auto dev_result = ScopedDevice::create(physical_device);
    if (!dev_result.ok()) {
        return dev_result.status();
    }

    GpuArena arena;
    arena.device_ = physical_device;
    arena.capacity_ = capacity_bytes;
    arena.offset_ = 0;

    if (!vmm_requested()) {
        const hipError_t e = hipMalloc(&arena.base_, capacity_bytes);
        if (e != hipSuccess || arena.base_ == nullptr) {
            return Status::insufficient_memory(
                std::string("hipMalloc failed: ") + hipGetErrorString(e), __FILE__, __LINE__);
        }
        arena.committed_bytes_ = capacity_bytes;
        return arena;
    }

    int supported = 0;
    hipError_t e = hipDeviceGetAttribute(
        &supported, hipDeviceAttributeVirtualMemoryManagementSupported, physical_device);
    if (e != hipSuccess) {
        return Status::hip_error(
            "hipDeviceGetAttribute", hipGetErrorString(e), __FILE__, __LINE__);
    }
    if (supported == 0) {
        return Status::unsupported("VMM not supported on device", __FILE__, __LINE__);
    }

    const auto prop = device_allocation_prop(physical_device);
    std::size_t minimum = 0;
    std::size_t recommended = 0;
    e = hipMemGetAllocationGranularity(
        &minimum, &prop, hipMemAllocationGranularityMinimum);
    if (e != hipSuccess || minimum == 0) {
        return Status::hip_error(
            "hipMemGetAllocationGranularity(min)", hipGetErrorString(e), __FILE__, __LINE__);
    }
    e = hipMemGetAllocationGranularity(
        &recommended, &prop, hipMemAllocationGranularityRecommended);
    if (e != hipSuccess || recommended == 0) {
        return Status::hip_error(
            "hipMemGetAllocationGranularity(rec)", hipGetErrorString(e), __FILE__, __LINE__);
    }

    arena.minimum_granularity_ = minimum;
    arena.recommended_granularity_ = recommended;

    std::size_t base_quantum = recommended;
    if (base_quantum < kTargetCommitQuantum) {
        base_quantum = kTargetCommitQuantum;
    }
    std::size_t commit_quantum = 0;
    auto qstatus = round_up_multiple(base_quantum, minimum, commit_quantum);
    if (!qstatus.ok()) {
        return qstatus;
    }
    arena.commit_quantum_ = commit_quantum;

    std::size_t reserved_bytes = 0;
    auto rstatus = round_up_multiple(capacity_bytes, minimum, reserved_bytes);
    if (!rstatus.ok()) {
        return rstatus;
    }
    arena.reserved_bytes_ = reserved_bytes;

    e = hipMemAddressReserve(&arena.base_, reserved_bytes, 0, nullptr, 0);
    if (e != hipSuccess || arena.base_ == nullptr) {
        return Status::hip_error(
            "hipMemAddressReserve", hipGetErrorString(e), __FILE__, __LINE__);
    }
    arena.vmm_ = true;
    arena.committed_bytes_ = 0;

    const std::size_t mapping_capacity =
        (reserved_bytes + commit_quantum - 1) / commit_quantum;
    arena.mappings_.reserve(mapping_capacity);

    return arena;
}

Status GpuArena::ensure_committed(std::size_t required_end) {
    if (!vmm_ || required_end <= committed_bytes_) {
        return Status::make_ok();
    }

    std::size_t target = 0;
    auto tstatus = round_up_multiple(required_end, commit_quantum_, target);
    if (!tstatus.ok()) {
        return tstatus;
    }
    if (target > reserved_bytes_) {
        target = reserved_bytes_;
    }

    const std::size_t bytes = target - committed_bytes_;
    if (bytes == 0 || bytes % minimum_granularity_ != 0) {
        return Status::invalid_state(
            "commit size not granularity aligned", __FILE__, __LINE__);
    }

    auto dev_result = ScopedDevice::create(device_);
    if (!dev_result.ok()) {
        return dev_result.status();
    }

    const auto prop = device_allocation_prop(device_);
    hipMemGenericAllocationHandle_t handle{};
    hipError_t e = hipMemCreate(&handle, bytes, &prop, 0);
    if (e != hipSuccess) {
        return Status::hip_error("hipMemCreate", hipGetErrorString(e), __FILE__, __LINE__);
    }

    void* map_address =
        reinterpret_cast<void*>(reinterpret_cast<char*>(base_) + committed_bytes_);
    e = hipMemMap(map_address, bytes, 0, handle, 0);
    if (e != hipSuccess) {
        discard_cleanup_result(hipMemRelease(handle));
        return Status::hip_error("hipMemMap", hipGetErrorString(e), __FILE__, __LINE__);
    }

    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = device_;
    access.flags = hipMemAccessFlagsProtReadWrite;
    e = hipMemSetAccess(map_address, bytes, &access, 1);
    if (e != hipSuccess) {
        discard_cleanup_result(hipMemUnmap(map_address, bytes));
        discard_cleanup_result(hipMemRelease(handle));
        return Status::hip_error(
            "hipMemSetAccess", hipGetErrorString(e), __FILE__, __LINE__);
    }

    mappings_.push_back(VmmMapping{committed_bytes_, bytes, handle});
    committed_bytes_ = target;
    return Status::make_ok();
}

namespace {

Status free_arena_base(
    int device, bool vmm, void* base, std::size_t reserved_bytes,
    const std::vector<GpuArena::VmmMapping>& mappings) {
    if (device < 0) {
        return Status::invalid_state("arena device invalid", __FILE__, __LINE__);
    }
    auto dev_result = ScopedDevice::create(device);
    if (!dev_result.ok()) {
        return dev_result.status();
    }
    if (vmm) {
        for (const auto& m : mappings) {
            discard_cleanup_result(hipMemUnmap(
                reinterpret_cast<void*>(
                    reinterpret_cast<char*>(base) + m.offset),
                m.bytes));
            discard_cleanup_result(hipMemRelease(m.handle));
        }
        discard_cleanup_result(hipMemAddressFree(base, reserved_bytes));
        return Status::make_ok();
    }
    const hipError_t e = hipFree(base);
    if (e != hipSuccess) {
        return Status::hip_error("hipFree", hipGetErrorString(e), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}

Status GpuArena::shutdown() {
    if (base_ == nullptr) {
        return Status::make_ok();
    }

    Status error = free_arena_base(device_, vmm_, base_, reserved_bytes_, mappings_);
    mappings_.clear();

    device_ = -1;
    base_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
    reserved_bytes_ = 0;
    committed_bytes_ = 0;
    minimum_granularity_ = 0;
    recommended_granularity_ = 0;
    commit_quantum_ = 0;
    shutdown_ = true;

    return error;
}

void GpuArena::release_noexcept() {
    if (base_) {
        auto error = free_arena_base(device_, vmm_, base_, reserved_bytes_, mappings_);
        if (!error.ok()) {
            std::cerr << "arena destructor: free failed: "
                      << error.message() << std::endl;
        }
        mappings_.clear();
        base_ = nullptr;
    }
    device_ = -1;
    capacity_ = 0;
    offset_ = 0;
    reserved_bytes_ = 0;
    committed_bytes_ = 0;
    minimum_granularity_ = 0;
    recommended_granularity_ = 0;
    commit_quantum_ = 0;
    shutdown_ = true;
}

GpuArena::~GpuArena() noexcept {
    release_noexcept();
}

GpuArena::GpuArena(GpuArena&& other) noexcept
    : device_(other.device_),
      base_(other.base_),
      capacity_(other.capacity_),
      offset_(other.offset_),
      vmm_(other.vmm_),
      reserved_bytes_(other.reserved_bytes_),
      committed_bytes_(other.committed_bytes_),
      minimum_granularity_(other.minimum_granularity_),
      recommended_granularity_(other.recommended_granularity_),
      commit_quantum_(other.commit_quantum_),
      mappings_(std::move(other.mappings_)) {
    other.device_ = -1;
    other.base_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
    other.reserved_bytes_ = 0;
    other.committed_bytes_ = 0;
    other.minimum_granularity_ = 0;
    other.recommended_granularity_ = 0;
    other.commit_quantum_ = 0;
    other.mappings_.clear();
    other.shutdown_ = true;
}

GpuArena& GpuArena::operator=(GpuArena&& other) noexcept {
    if (this != &other) {
        release_noexcept();
        device_ = other.device_;
        base_ = other.base_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        vmm_ = other.vmm_;
        reserved_bytes_ = other.reserved_bytes_;
        committed_bytes_ = other.committed_bytes_;
        minimum_granularity_ = other.minimum_granularity_;
        recommended_granularity_ = other.recommended_granularity_;
        commit_quantum_ = other.commit_quantum_;
        mappings_ = std::move(other.mappings_);
        other.device_ = -1;
        other.base_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
        other.reserved_bytes_ = 0;
        other.committed_bytes_ = 0;
        other.minimum_granularity_ = 0;
        other.recommended_granularity_ = 0;
        other.commit_quantum_ = 0;
        other.mappings_.clear();
        other.shutdown_ = true;
    }
    return *this;
}

Result<DeviceAllocationView> GpuArena::allocate_aligned(std::size_t bytes, std::size_t alignment) {
    if (shutdown_) {
        return Status::invalid_state("arena shutdown", __FILE__, __LINE__);
    }

    if (device_ < 0) {
        return Status::invalid_state("arena moved from", __FILE__, __LINE__);
    }

    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return Status::invalid_argument("alignment must be power of 2", __FILE__, __LINE__);
    }

    if (bytes == 0) {
        return DeviceAllocationView(nullptr, 0, device_);
    }

    std::size_t aligned_offset;
    auto align_status = checked_align_offset_from_base(
        base_, offset_, alignment, capacity_, aligned_offset, __FILE__, __LINE__);
    if (!align_status.ok()) {
        return align_status;
    }

    if (bytes > capacity_ - aligned_offset) {
        return Status::insufficient_memory("arena capacity exceeded", __FILE__, __LINE__);
    }

    const std::size_t end = aligned_offset + bytes;
    auto commit_status = ensure_committed(end);
    if (!commit_status.ok()) {
        return commit_status;
    }

    offset_ = end;
    void* aligned_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(base_) + aligned_offset);
    return DeviceAllocationView(aligned_ptr, bytes, device_);
}

Result<ArenaSet> ArenaSet::create(
    const std::vector<int>& devices,
    std::size_t capacity_bytes) {
    if (devices.empty()) {
        return Status::invalid_argument("devices must not be empty", __FILE__, __LINE__);
    }

    for (std::size_t i = 0; i < devices.size(); ++i) {
        for (std::size_t j = i + 1; j < devices.size(); ++j) {
            if (devices[i] == devices[j]) {
                return Status::invalid_argument("duplicate device in ArenaSet", __FILE__, __LINE__);
            }
        }
    }

    ArenaSet set;
    set.arenas_.reserve(devices.size());

    for (int device : devices) {
        auto arena_result = GpuArena::create(device, capacity_bytes);
        if (!arena_result.ok()) {
            return arena_result.status();
        }

        set.arenas_.push_back(DeviceArena{
            .physical_device = device,
            .arena = arena_result.release(),
        });
    }

    return set;
}

Result<GpuArena*> ArenaSet::for_device(int physical_device) noexcept {
    for (auto& da : arenas_) {
        if (da.physical_device == physical_device) {
            return &da.arena;
        }
    }
    return Status::out_of_range("device not in ArenaSet", __FILE__, __LINE__);
}

Status ArenaSet::shutdown() {
    if (shutdown_) {
        return Status::make_ok();
    }

    Status first_error = Status::make_ok();

    for (auto& da : arenas_) {
        Status err = da.arena.shutdown();
        if (!first_error.ok()) {
            continue;
        }
        if (!err.ok()) {
            first_error = err;
        }
    }

    shutdown_ = true;

    return first_error;
}

DeviceAllocationView GpuArena::make_view(void* data, std::size_t bytes, int physical_device) noexcept {
    return DeviceAllocationView(data, bytes, physical_device);
}

Result<DeviceAllocationView> GpuArena::used_prefix_view() const {
    if (shutdown_ || device_ < 0) {
        return Status::invalid_state("arena shutdown", __FILE__, __LINE__);
    }
    if (base_ == nullptr) {
        return Status::invalid_state("arena has no allocation", __FILE__, __LINE__);
    }
    if (offset_ == 0) {
        return Status::invalid_state("arena used prefix is empty", __FILE__, __LINE__);
    }
    if (vmm_ && offset_ > committed_bytes_) {
        return Status::invalid_state("arena used prefix not committed", __FILE__, __LINE__);
    }
    return DeviceAllocationView(base_, offset_, device_);
}

} // namespace gpu
} // namespace ps
