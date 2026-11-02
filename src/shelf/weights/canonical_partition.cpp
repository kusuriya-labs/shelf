#include <shelf/weights/canonical_partition.h>

#include <shelf/quantization/quant_format.h>
#include <shelf/quantization/quant_spec.h>

#include <cstring>
#include <limits>

namespace ps::weights {

namespace {

using quantization::bundle::ByteSpan;
using quantization::manifest::QuantizedEncoding;
using quantization::bundle::QuantizedTensorView;
using quantization::MetaCountMode;
using quantization::QuantFormatDesc;
using quantization::QuantFormatId;
using quantization::QuantMetaDesc;
using quantization::quant_format_desc;

uint64_t row_count(const std::vector<int64_t>& shape) {
    uint64_t rows = 1;
    for (std::size_t i = 0; i + 1 < shape.size(); ++i) {
        rows *= static_cast<uint64_t>(shape[i]);
    }
    return rows;
}

uint64_t ceil_div(uint64_t value, uint64_t unit) {
    return (value + unit - 1u) / unit;
}

Status checked_mul(uint64_t left, uint64_t right, uint64_t& out) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) {
        return Status::overflow("tensor partition byte count overflow", __FILE__, __LINE__);
    }
    out = left * right;
    return Status::make_ok();
}

ByteSpan span_of(const std::vector<uint8_t>& buffer) {
    ByteSpan span;
    if (!buffer.empty()) {
        span.data = buffer.data();
        span.size = buffer.size();
    }
    return span;
}

const uint8_t* byte_ptr(const ByteSpan& stream) {
    return static_cast<const uint8_t*>(stream.data);
}

Status require_absent(const ByteSpan& stream, const char* what) {
    if (stream.data != nullptr || stream.size != 0) {
        return Status::invalid_argument(what, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status require_stream(const ByteSpan& stream, uint64_t expected_bytes, const char* what) {
    if (stream.data == nullptr) {
        return Status::invalid_argument(what, __FILE__, __LINE__);
    }
    if (stream.size != expected_bytes) {
        return Status::invalid_argument(what, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status compute_global_rows(const TensorPartitionDesc& partition,
                           const std::vector<int64_t>& local_shape,
                           std::vector<uint64_t>& out) {
    const std::size_t outer = partition.global_shape.size() - 1;
    std::vector<uint64_t> stride(outer, 1);
    uint64_t acc = 1;
    for (std::size_t i = outer; i-- > 0;) {
        stride[i] = acc;
        acc *= static_cast<uint64_t>(partition.global_shape[i]);
    }

    std::vector<uint64_t> prefix(partition.ranges.size(), 0);
    uint64_t sum = 0;
    for (std::size_t i = 0; i < partition.ranges.size(); ++i) {
        prefix[i] = sum;
        sum += partition.ranges[i].extent;
    }

    uint64_t rows = 1;
    for (std::size_t i = 0; i < outer; ++i) {
        rows *= static_cast<uint64_t>(local_shape[i]);
    }
    out.resize(rows);

    const std::size_t axis = static_cast<std::size_t>(partition.axis);
    std::vector<uint64_t> coord(outer, 0);
    for (uint64_t r = 0; r < rows; ++r) {
        uint64_t global_row = 0;
        for (std::size_t d = 0; d < outer; ++d) {
            uint64_t global_coord = coord[d];
            if (d == axis) {
                std::size_t ri = 0;
                while (ri + 1 < prefix.size() && coord[d] >= prefix[ri + 1]) ++ri;
                global_coord = partition.ranges[ri].global_offset +
                               (coord[d] - prefix[ri]);
            }
            global_row += global_coord * stride[d];
        }
        out[r] = global_row;
        for (std::size_t i = outer; i-- > 0;) {
            if (++coord[i] < static_cast<uint64_t>(local_shape[i])) break;
            coord[i] = 0;
        }
    }
    return Status::make_ok();
}

Status copy_rows(uint64_t row_bytes, const std::vector<uint64_t>& global_rows,
                 const uint8_t* src, std::vector<uint8_t>& dst) {
    uint64_t bytes = 0;
    Status st = checked_mul(static_cast<uint64_t>(global_rows.size()), row_bytes, bytes);
    if (!st.ok()) return st;
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        return Status::overflow("tensor partition byte count overflow", __FILE__, __LINE__);
    }
    dst.resize(static_cast<std::size_t>(bytes));
    for (std::size_t r = 0; r < global_rows.size(); ++r) {
        std::memcpy(dst.data() + r * row_bytes, src + global_rows[r] * row_bytes,
                    static_cast<std::size_t>(row_bytes));
    }
    return Status::make_ok();
}

void slice_row_units(const uint8_t* src_row, uint64_t unit_bytes,
                     const std::vector<TensorPartitionRange>& units,
                     uint8_t* dst_row) {
    uint64_t dst_offset = 0;
    for (const TensorPartitionRange& unit : units) {
        const uint64_t bytes = unit.extent * unit_bytes;
        std::memcpy(dst_row + dst_offset, src_row + unit.global_offset * unit_bytes,
                    static_cast<std::size_t>(bytes));
        dst_offset += bytes;
    }
}

Status materialize_bf16(const TensorPartitionDesc& partition,
                        const QuantizedTensorView& global,
                        const std::vector<int64_t>& local_shape,
                        bool k_axis,
                        OwnedCanonicalTensor& owned) {
    Status st = require_absent(global.codes, "bf16 tensor must not carry a codes stream");
    if (!st.ok()) return st;
    st = require_absent(global.metadata1, "bf16 tensor must not carry a metadata1 stream");
    if (!st.ok()) return st;
    st = require_absent(global.metadata2, "bf16 tensor must not carry a metadata2 stream");
    if (!st.ok()) return st;
    st = require_absent(global.metadata3, "bf16 tensor must not carry a metadata3 stream");
    if (!st.ok()) return st;
    st = require_absent(global.metadata4, "bf16 tensor must not carry a metadata4 stream");
    if (!st.ok()) return st;
    if (global.data.data == nullptr) {
        return Status::invalid_argument("bf16 tensor is missing the data stream",
                                        __FILE__, __LINE__);
    }

    const uint64_t rows_global = row_count(partition.global_shape);
    const uint64_t k_global = static_cast<uint64_t>(partition.global_shape.back());
    uint64_t expected = 0;
    st = checked_mul(rows_global, k_global, expected);
    if (!st.ok()) return st;
    st = checked_mul(expected, sizeof(uint16_t), expected);
    if (!st.ok()) return st;
    if (global.data.size != expected) {
        return Status::invalid_argument("bf16 tensor byte count mismatch", __FILE__, __LINE__);
    }

    owned.logical_shape = local_shape;
    owned.k_padded = static_cast<uint64_t>(local_shape.back());

    if (k_axis) {
        const uint64_t rows_local = row_count(local_shape);
        if (rows_local != rows_global) {
            return Status::invalid_argument("tensor partition row count mismatch",
                                            __FILE__, __LINE__);
        }
        const uint64_t k_local = static_cast<uint64_t>(local_shape.back());
        uint64_t row_bytes_local = 0;
        st = checked_mul(k_local, sizeof(uint16_t), row_bytes_local);
        if (!st.ok()) return st;
        uint64_t bytes = 0;
        st = checked_mul(rows_local, row_bytes_local, bytes);
        if (!st.ok()) return st;
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            return Status::overflow("tensor partition byte count overflow", __FILE__, __LINE__);
        }
        owned.data.resize(static_cast<std::size_t>(bytes));
        const uint64_t row_bytes_global = k_global * sizeof(uint16_t);
        for (uint64_t r = 0; r < rows_local; ++r) {
            slice_row_units(byte_ptr(global.data) + r * row_bytes_global, sizeof(uint16_t),
                            partition.ranges,
                            owned.data.data() + r * row_bytes_local);
        }
        return Status::make_ok();
    }

    std::vector<uint64_t> global_rows;
    st = compute_global_rows(partition, local_shape, global_rows);
    if (!st.ok()) return st;
    return copy_rows(k_global * sizeof(uint16_t), global_rows, byte_ptr(global.data), owned.data);
}

Status materialize_quant(QuantFormatId format_id,
                         const TensorPartitionDesc& partition,
                         const QuantizedTensorView& global,
                         const std::vector<int64_t>& local_shape,
                         bool k_axis,
                         OwnedCanonicalTensor& owned) {
    const QuantFormatDesc* fmt = quant_format_desc(format_id);
    if (fmt == nullptr || fmt->block_elements == 0) {
        return Status::unsupported("unsupported tensor partition encoding", __FILE__, __LINE__);
    }
    const uint64_t block = fmt->block_elements;
    if (global.k_padded <= 0 ||
        (static_cast<uint64_t>(global.k_padded) % block) != 0) {
        return Status::invalid_argument("global k_padded is not block aligned",
                                        __FILE__, __LINE__);
    }
    Status aligned = validate_tensor_partition_block_alignment(
        partition, k_axis ? block : 0);
    if (!aligned.ok()) return aligned;

    const uint64_t rows_global = row_count(partition.global_shape);
    const uint64_t k_global = static_cast<uint64_t>(partition.global_shape.back());
    if (static_cast<uint64_t>(global.k_padded) < k_global) {
        return Status::invalid_argument("global k_padded smaller than logical K",
                                        __FILE__, __LINE__);
    }
    const uint64_t kp_global = static_cast<uint64_t>(global.k_padded);
    const uint64_t blocks_global = kp_global / block;

    uint64_t codes_row = 0;
    Status st = checked_mul(blocks_global, fmt->code_bytes_per_block, codes_row);
    if (!st.ok()) return st;
    uint64_t codes_expected = 0;
    st = checked_mul(rows_global, codes_row, codes_expected);
    if (!st.ok()) return st;
    st = require_stream(global.codes, codes_expected,
                        "quantized tensor codes stream is missing or has the wrong size");
    if (!st.ok()) return st;

    const ByteSpan* meta_streams[4] = {
        &global.metadata1, &global.metadata2, &global.metadata3, &global.metadata4};
    uint64_t meta_unit_bytes[4] = {0, 0, 0, 0};
    uint64_t meta_units[4] = {0, 0, 0, 0};
    uint64_t meta_unit_rows[4] = {1, 1, 1, 1};
    uint64_t meta_block_k[4] = {block, block, block, block};
    for (uint32_t i = 0; i < 4; ++i) {
        const QuantMetaDesc& meta = fmt->meta[i];
        if (meta.count_mode == MetaCountMode::None) {
            st = require_absent(*meta_streams[i],
                                "quantized tensor carries an unexpected metadata stream");
            if (!st.ok()) return st;
            continue;
        }
        if (meta.element_bytes == 0) {
            return Status::unsupported(
                "tensor partition requires a block metadata layout",
                __FILE__, __LINE__);
        }
        if (meta.count_mode == MetaCountMode::MatrixBlock) {
            if (partition.global_shape.size() != 2) {
                return Status::unsupported(
                    "block-scaled tensor partition requires a 2D tensor",
                    __FILE__, __LINE__);
            }
            const uint64_t block_n = meta.block_n != 0 ? meta.block_n : 1u;
            const uint64_t block_k = meta.block_k != 0 ? meta.block_k : block;
            const uint64_t tile = k_axis ? block_k : block_n;
            for (const TensorPartitionRange& range : partition.ranges) {
                if ((range.global_offset % tile) != 0 || (range.extent % tile) != 0) {
                    return Status::invalid_argument(
                        "tensor partition range is not aligned to the block-scaled tile",
                        __FILE__, __LINE__);
                }
            }
            meta_unit_rows[i] = block_n;
            meta_block_k[i] = block_k;
            st = checked_mul(ceil_div(kp_global, block_k), meta.element_bytes,
                             meta_unit_bytes[i]);
            if (!st.ok()) return st;
            meta_units[i] = ceil_div(rows_global, block_n);
        } else if (meta.count_mode == MetaCountMode::PerBlock) {
            st = checked_mul(blocks_global, meta.element_bytes, meta_unit_bytes[i]);
            if (!st.ok()) return st;
            meta_units[i] = rows_global;
        } else {
            return Status::unsupported(
                "tensor partition requires a block metadata layout",
                __FILE__, __LINE__);
        }
        uint64_t expected = 0;
        st = checked_mul(meta_units[i], meta_unit_bytes[i], expected);
        if (!st.ok()) return st;
        st = require_stream(*meta_streams[i], expected,
                            "quantized tensor metadata stream is missing or has the wrong size");
        if (!st.ok()) return st;
    }

    owned.logical_shape = local_shape;
    const uint64_t kp_local =
        k_axis ? static_cast<uint64_t>(local_shape.back()) : kp_global;
    owned.k_padded = kp_local;
    const uint64_t blocks_local = kp_local / block;
    std::vector<std::vector<uint8_t>*> meta_owned = {
        &owned.metadata1, &owned.metadata2, &owned.metadata3, &owned.metadata4};

    if (k_axis) {
        std::vector<TensorPartitionRange> code_units;
        code_units.reserve(partition.ranges.size());
        for (const TensorPartitionRange& r : partition.ranges) {
            code_units.push_back(
                TensorPartitionRange{r.global_offset / block, r.extent / block});
        }

        uint64_t codes_row_local = 0;
        st = checked_mul(blocks_local, fmt->code_bytes_per_block, codes_row_local);
        if (!st.ok()) return st;
        uint64_t bytes = 0;
        st = checked_mul(rows_global, codes_row_local, bytes);
        if (!st.ok()) return st;
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            return Status::overflow("tensor partition byte count overflow", __FILE__, __LINE__);
        }
        owned.codes.resize(static_cast<std::size_t>(bytes));
        for (uint64_t r = 0; r < rows_global; ++r) {
            slice_row_units(byte_ptr(global.codes) + r * codes_row, fmt->code_bytes_per_block,
                            code_units, owned.codes.data() + r * codes_row_local);
        }

        for (uint32_t i = 0; i < 4; ++i) {
            if (fmt->meta[i].count_mode == MetaCountMode::None) continue;
            const uint64_t unit_k = meta_block_k[i];
            std::vector<TensorPartitionRange> units;
            units.reserve(partition.ranges.size());
            for (const TensorPartitionRange& r : partition.ranges) {
                units.push_back(
                    TensorPartitionRange{r.global_offset / unit_k, r.extent / unit_k});
            }
            const uint64_t unit_bytes_local =
                ceil_div(kp_local, unit_k) * fmt->meta[i].element_bytes;
            uint64_t meta_bytes = 0;
            st = checked_mul(meta_units[i], unit_bytes_local, meta_bytes);
            if (!st.ok()) return st;
            if (meta_bytes > std::numeric_limits<std::size_t>::max()) {
                return Status::overflow("tensor partition byte count overflow", __FILE__,
                                        __LINE__);
            }
            std::vector<uint8_t>& dst = *meta_owned[i];
            dst.resize(static_cast<std::size_t>(meta_bytes));
            const uint8_t* src = byte_ptr(*meta_streams[i]);
            for (uint64_t u = 0; u < meta_units[i]; ++u) {
                slice_row_units(src + u * meta_unit_bytes[i], fmt->meta[i].element_bytes,
                                units, dst.data() + u * unit_bytes_local);
            }
        }
        return Status::make_ok();
    }

    std::vector<uint64_t> global_rows;
    st = compute_global_rows(partition, local_shape, global_rows);
    if (!st.ok()) return st;
    st = copy_rows(codes_row, global_rows, byte_ptr(global.codes), owned.codes);
    if (!st.ok()) return st;
    for (uint32_t i = 0; i < 4; ++i) {
        if (fmt->meta[i].count_mode == MetaCountMode::None) continue;
        if (meta_unit_rows[i] == 1) {
            st = copy_rows(meta_unit_bytes[i], global_rows, byte_ptr(*meta_streams[i]),
                           *meta_owned[i]);
            if (!st.ok()) return st;
            continue;
        }
        const uint64_t unit_rows = meta_unit_rows[i];
        if (global_rows.empty() || (global_rows.size() % unit_rows) != 0) {
            return Status::invalid_argument(
                "local row count is not aligned to the block-scaled tile",
                __FILE__, __LINE__);
        }
        const uint64_t units_local = global_rows.size() / unit_rows;
        uint64_t meta_bytes = 0;
        st = checked_mul(units_local, meta_unit_bytes[i], meta_bytes);
        if (!st.ok()) return st;
        if (meta_bytes > std::numeric_limits<std::size_t>::max()) {
            return Status::overflow("tensor partition byte count overflow", __FILE__,
                                    __LINE__);
        }
        std::vector<uint8_t>& dst = *meta_owned[i];
        dst.resize(static_cast<std::size_t>(meta_bytes));
        const uint8_t* src = byte_ptr(*meta_streams[i]);
        for (uint64_t u = 0; u < units_local; ++u) {
            const uint64_t src_unit = global_rows[u * unit_rows] / unit_rows;
            std::memcpy(dst.data() + u * meta_unit_bytes[i],
                        src + src_unit * meta_unit_bytes[i],
                        static_cast<std::size_t>(meta_unit_bytes[i]));
        }
    }
    return Status::make_ok();
}

}

Status materialize_rank_local_canonical(
    const TensorPartitionDesc& partition,
    const QuantizedTensorView& global,
    OwnedCanonicalTensor& owned,
    QuantizedTensorView& local) {
    Status valid = validate_tensor_partition(partition);
    if (!valid.ok()) return valid;
    if (global.scale_block_n != 0 || global.scale_block_k != 0) {
        const quantization::WeightBlockShape expected =
            quantization::default_weight_block(global.encoding);
        const uint32_t declared_n = global.scale_block_n != 0 ? global.scale_block_n : expected.n;
        const uint32_t declared_k = global.scale_block_k != 0 ? global.scale_block_k : expected.k;
        if (declared_n != expected.n || declared_k != expected.k) {
            return Status::unsupported(
                "tp partition requires the default weight scale block",
                __FILE__, __LINE__);
        }
    }
    if (partition.global_shape != global.logical_shape) {
        return Status::invalid_argument(
            "partition global_shape does not match tensor logical shape",
            __FILE__, __LINE__);
    }
    auto local_shape_result = tensor_partition_local_shape(partition);
    if (!local_shape_result.ok()) return local_shape_result.status();
    const std::vector<int64_t>& local_shape = local_shape_result.value();
    const bool k_axis =
        static_cast<std::size_t>(partition.axis) + 1 == partition.global_shape.size();

    Status st = Status::make_ok();
    switch (global.encoding) {
        case QuantizedEncoding::Bf16:
            st = materialize_bf16(partition, global, local_shape, k_axis, owned);
            break;
        case QuantizedEncoding::Fp8:
            st = materialize_quant(QuantFormatId::Fp8Finegrained, partition, global,
                                   local_shape, k_axis, owned);
            break;
    }
    if (!st.ok()) return st;

    local.name = global.name;
    local.encoding = global.encoding;
    local.logical_shape = owned.logical_shape;
    local.k_padded = static_cast<int64_t>(owned.k_padded);
    local.data = span_of(owned.data);
    local.codes = span_of(owned.codes);
    local.metadata1 = span_of(owned.metadata1);
    local.metadata2 = span_of(owned.metadata2);
    local.metadata3 = span_of(owned.metadata3);
    local.metadata4 = span_of(owned.metadata4);
    return Status::make_ok();
}

}  // namespace ps::weights
