#include <shelf/kernels/fused/exl3/decode.h>
#include <shelf/kernels/fused/exl3/grouped.h>
#include <cstdlib>

namespace shelf::kernel {

namespace {
constexpr uint32_t kOutBlockBase = kExl3W16A16OutBlock;
constexpr uint32_t kRowTile = kExl3W16A16RowTile;
}

hipError_t launch_gemm_exl3_int8_grouped(const Exl3GroupedArgs& args,
                                           hipStream_t stream) {
    if (args.trellis == nullptr || args.suh == nullptr || args.svh == nullptr ||
        args.input == nullptr || args.output == nullptr || args.group_expert == nullptr ||
        args.group_rows == nullptr || args.group_count == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.out_features == 0u || (args.out_features % kOutBlockBase) != 0u ||
        args.k == 0u || (args.k % 128u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.ka < 1u || args.ka > 8u || (args.half != 0u && args.ka > 3u)) {
        return hipErrorInvalidValue;
    }
    if (args.input_row_stride < args.k || args.output_row_stride < args.out_features ||
        args.group_capacity == 0u || args.group_capacity > 65535u) {
        return hipErrorInvalidValue;
    }
    if (args.half != 0u) return hipErrorInvalidValue;

    Exl3GemmArgs dense;
    dense.trellis = args.trellis;
    dense.suh = args.suh;
    dense.svh = args.svh;
    dense.input = args.input;
    dense.output = args.output;
    dense.rows = kRowTile;
    dense.out_features = args.out_features;
    dense.k = args.k;
    dense.input_row_stride = args.input_row_stride;
    dense.output_row_stride = args.output_row_stride;
    dense.output_dtype = args.output_dtype;
    dense.ka = args.ka;
    dense.half = args.half;
    static const uint32_t fuse_override = [] {
        const char* e = std::getenv("SHELF_EXL3_GROUPED_FUSE");
        if (e == nullptr) return kExl3GroupedFuse;
        return static_cast<uint32_t>(std::strtoul(e, nullptr, 10));
    }();
    static const uint32_t ntiles_override = [] {
        const char* e = std::getenv("SHELF_EXL3_GROUPED_NTILES");
        return e != nullptr ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    dense.n_tiles = ntiles_override;
    dense.groups = (args.expert_group_begin != nullptr && fuse_override > 1u) ? fuse_override
                                                                            : 1u;
    dense.group_expert = args.group_expert;
    dense.group_rows = args.group_rows;
    dense.group_count = args.group_count;
    dense.expert_group_begin = args.expert_group_begin;
    dense.row_token = args.row_token;
    dense.group_row_base = args.group_row_base;
    dense.trellis_batch_stride = args.trellis_batch_stride;
    dense.suh_batch_stride = args.suh_batch_stride;
    dense.svh_batch_stride = args.svh_batch_stride;

    const uint32_t n_tiles = dense.n_tiles != 0u ? dense.n_tiles :
        (dense.rows < 4u || dense.out_features % (kOutBlockBase * 2u) != 0u ||
         dense.out_features < kOutBlockBase * 8u ? 1u : 2u);
    if ((args.out_features % (kOutBlockBase * n_tiles)) != 0u) return hipErrorInvalidValue;
    return launch_exl3_mul24_grouped(dense, n_tiles,
                                     args.out_features / (kOutBlockBase * n_tiles),
                                     args.group_capacity, stream);
}

}
