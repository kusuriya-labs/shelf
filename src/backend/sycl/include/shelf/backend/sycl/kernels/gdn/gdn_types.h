// vendored from PhaseShift (own work); namespace/type adapted only.
//
// shelf adaptation note: PhaseShift's GDN kernels read request metadata from
// ::ps::runtime::DeviceRequestDescriptor (phaseshift/runtime/batch/
// device_batch_context.h). shelf has no serving request system, so the three
// fields those kernels actually use (request_handle.slot, row_begin, row_count)
// are carried by this plain descriptor instead. Field meaning is unchanged.
// (SYCL port: identical layout.)
#pragma once

#include <cstdint>
#include <type_traits>

namespace shelf::syclk {

struct GdnRequestDescriptor {
    uint32_t slot = 0;        // state-slot index (was request_handle.slot)
    uint32_t row_begin = 0;   // first token row of this request in the batch
    uint32_t row_count = 0;   // token rows this request consumes (1 = decode)
};

static_assert(sizeof(GdnRequestDescriptor) == 12);
static_assert(std::is_trivially_copyable_v<GdnRequestDescriptor>);

} // namespace shelf::syclk
