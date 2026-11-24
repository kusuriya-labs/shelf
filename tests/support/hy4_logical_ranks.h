#pragma once

#include <shelf/models/hy4_flash/ops.h>
#include <cstdlib>

namespace shelf::hy4 {
inline void logical_ranks(Residency &gpu, int ranks) {
    gpu.ranks = gpu.ep_size = ranks;
    for (int r = 1; r < ranks; ++r) {
        gpu.physical[r] = 0;
        constexpr size_t work = 512 << 20, bytes = work + Residency::matrix_bytes;
        auto result = gpu::GpuArena::create(0, bytes);
        if (!result.ok())
            throw std::runtime_error(result.status().message());
        gpu.arenas.push_back(result.release());
        auto memory = gpu.arenas.back().allocate_aligned(bytes, 256);
        if (!memory.ok())
            throw std::runtime_error(memory.status().message());
        Region buffer{static_cast<uint8_t *>(memory.value().data()), bytes};
        gpu.reserve_experts(r, buffer);
        gpu.workspace[r] = {static_cast<uint8_t *>(buffer.next(work)), work};
        checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
    }
    const auto transport = std::getenv("SHELF_HY4_EP_TRANSPORT");
    const bool host = transport && std::string_view(transport) == "host";
    for (int r = 0; r < ranks; ++r)
        for (int other = 0; other < ranks; ++other)
            if (other != r) {
                gpu.peer[r][other] = !host;
                if (host)
                    checked(hipHostMalloc(&gpu.small_staging[r][other], gpu.transfer_packet_bytes(), hipHostMallocPortable));
                for (int p = 0; p < 2; ++p) {
                    gpu.create_event(gpu.ready[r][other][p], r);
                    gpu.create_event(gpu.ack[r][other][p], r);
                }
            }
    for (auto &layer : gpu.model.owners)
        for (int e = 0; e < 512; ++e)
            layer[e] = e % ranks;
    checked(hipMemcpy(gpu.moe_owners[0], gpu.model.owners.data(), sizeof(Owners), hipMemcpyHostToDevice));
    if (ranks > 1) {
        std::vector<Hy4ExpertProjection> all(Residency::matrix_bytes / sizeof(Hy4ExpertProjection));
        checked(hipMemcpy(all.data(), gpu.matrices[0], Residency::matrix_bytes, hipMemcpyDeviceToHost));
        for (int r = 0; r < ranks; ++r) {
            auto table = all;
            for (size_t layer = 0; layer < gpu.model.owners.size(); ++layer)
                for (int e = 0; e < 512; ++e)
                    if (gpu.model.owners[layer][e] != r)
                        std::fill_n(table.data() + (layer * 512 + e) * 3, 3, Hy4ExpertProjection{});
            checked(hipMemcpy(gpu.matrices[r], table.data(), Residency::matrix_bytes, hipMemcpyHostToDevice));
        }
    }
}
}
