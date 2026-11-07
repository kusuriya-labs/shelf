#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/quantization/fp8/fp8.h>
#include <shelf/weights/canonical_partition.h>
#include <shelf/weights/tensor_partition.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using shelf::Status;
using shelf::weights::materialize_rank_local_canonical;
using shelf::weights::OwnedCanonicalTensor;
using shelf::weights::TensorPartitionDesc;
using shelf::weights::TensorPartitionRange;
using shelf::weights::tensor_partition_local_shape;
using shelf::weights::validate_tensor_partition;
using namespace shelf::quantization;
using namespace shelf::quantization::manifest;
using namespace shelf::quantization::bundle;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static TensorPartitionRange rng(uint64_t offset, uint64_t extent) {
    TensorPartitionRange r;
    r.global_offset = offset;
    r.extent = extent;
    return r;
}

static TensorPartitionDesc desc_of(int32_t axis, std::vector<int64_t> shape,
                                   uint32_t index, uint32_t count,
                                   std::vector<TensorPartitionRange> ranges) {
    TensorPartitionDesc d;
    d.axis = axis;
    d.index = index;
    d.count = count;
    d.global_shape = std::move(shape);
    d.ranges = std::move(ranges);
    return d;
}

static QuantizedTensorView make_bf16_view(const std::vector<uint8_t>& bytes,
                                          std::vector<int64_t> shape) {
    QuantizedTensorView v;
    v.name = "w";
    v.encoding = QuantizedEncoding::Bf16;
    v.logical_shape = std::move(shape);
    v.k_padded = static_cast<int64_t>(v.logical_shape.back());
    v.data = ByteSpan{bytes.data(), bytes.size()};
    return v;
}

static void check_bytes(const std::string& tag, const std::vector<uint8_t>& got,
                        const std::vector<uint8_t>& want) {
    if (got.size() != want.size()) {
        fail(tag + " size " + std::to_string(got.size()) + " vs " + std::to_string(want.size()));
        return;
    }
    if (!got.empty() && std::memcmp(got.data(), want.data(), got.size()) != 0) {
        fail(tag + " bytes differ");
    }
}

static std::vector<float> make_source(uint64_t rows, uint64_t k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> src(rows * k);
    for (float& f : src) f = dist(rng);
    return src;
}

static std::vector<uint8_t> gather_rows(const std::vector<uint8_t>& src, uint64_t row_bytes,
                                         uint64_t first_row, uint64_t row_count) {
    std::vector<uint8_t> out;
    const uint8_t* begin = src.data() + first_row * row_bytes;
    out.insert(out.end(), begin, begin + row_count * row_bytes);
    return out;
}

static void test_bf16_partitions() {
    {
        const uint64_t rows = 4, k = 8;
        std::vector<uint8_t> bytes(rows * k * 2);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(i * 7 + 3);
        }
        QuantizedTensorView g = make_bf16_view(bytes, {4, 8});

        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {4, 8}, 0, 2, {rng(0, 2)}), g, owned, local);
        check(st.ok(), "bf16 axis0 materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({2, 8}), "bf16 axis0 local shape");
            check(local.k_padded == 8, "bf16 axis0 local k_padded");
            const std::vector<uint8_t> want(bytes.begin(), bytes.begin() + 2 * k * 2);
            check_bytes("bf16 axis0 rank0", owned.data, want);
        }

        st = materialize_rank_local_canonical(
            desc_of(0, {4, 8}, 1, 2, {rng(2, 2)}), g, owned, local);
        check(st.ok(), "bf16 axis0 rank1 materialize");
        if (st.ok()) {
            const std::vector<uint8_t> want(bytes.begin() + 2 * k * 2, bytes.end());
            check_bytes("bf16 axis0 rank1", owned.data, want);
        }

        st = materialize_rank_local_canonical(
            desc_of(1, {4, 8}, 1, 2, {rng(3, 5)}), g, owned, local);
        check(st.ok(), "bf16 axis1 materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({4, 5}), "bf16 axis1 local shape");
            std::vector<uint8_t> want;
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row = bytes.data() + r * k * 2 + 3 * 2;
                want.insert(want.end(), row, row + 5 * 2);
            }
            check_bytes("bf16 axis1 slice", owned.data, want);
        }
    }
    {
        std::vector<uint8_t> bytes(16 * 2);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(i * 11 + 5);
        }
        QuantizedTensorView g = make_bf16_view(bytes, {16});
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {16}, 1, 2, {rng(4, 4)}), g, owned, local);
        check(st.ok(), "bf16 1D materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({4}), "bf16 1D local shape");
            const std::vector<uint8_t> want(bytes.begin() + 8, bytes.begin() + 16);
            check_bytes("bf16 1D slice", owned.data, want);
        }
    }
    {
        const uint64_t e = 2, n = 3, k = 4;
        std::vector<uint8_t> bytes(e * n * k * 2);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(i * 13 + 1);
        }
        QuantizedTensorView g = make_bf16_view(bytes, {2, 3, 4});
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {2, 3, 4}, 0, 2, {rng(1, 1)}), g, owned, local);
        check(st.ok(), "bf16 3D axis0 materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({1, 3, 4}),
                  "bf16 3D local shape");
            const std::size_t row_bytes = n * k * 2;
            const std::vector<uint8_t> want(bytes.begin() + row_bytes,
                                            bytes.begin() + 2 * row_bytes);
            check_bytes("bf16 3D row slice", owned.data, want);
        }
    }
    {
        const uint64_t rows = 6, k = 4;
        std::vector<uint8_t> bytes(rows * k * 2);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(i);
        }
        QuantizedTensorView g = make_bf16_view(bytes, {6, 4});
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {6, 4}, 0, 2, {rng(0, 2), rng(4, 2)}), g, owned, local);
        check(st.ok(), "bf16 segmented materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({4, 4}),
                  "bf16 segmented local shape");
            std::vector<uint8_t> want;
            const std::size_t row_bytes = k * 2;
            want.insert(want.end(), bytes.begin(), bytes.begin() + 2 * row_bytes);
            want.insert(want.end(), bytes.begin() + 4 * row_bytes, bytes.end());
            check_bytes("bf16 segmented slice", owned.data, want);
        }
    }
}

static void test_segmented_gdn() {
    const uint64_t qk = 64, value = 64;
    const uint64_t conv = 2 * qk + value;
    const uint64_t k = 32;
    std::vector<uint8_t> bytes(conv * k * 2u);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>(i * 5 + 2);
    }
    QuantizedTensorView g = make_bf16_view(
        bytes, {static_cast<int64_t>(conv), static_cast<int64_t>(k)});

    const std::vector<TensorPartitionRange> rank0 = {
        rng(0, 32), rng(64, 32), rng(128, 32)};
    OwnedCanonicalTensor owned;
    QuantizedTensorView local;
    Status st = materialize_rank_local_canonical(
        desc_of(0, {static_cast<int64_t>(conv), static_cast<int64_t>(k)}, 0, 2, rank0),
        g, owned, local);
    check(st.ok(), "segmented GDN materialize");
    if (!st.ok()) return;
    check(local.logical_shape == std::vector<int64_t>({96, 32}),
          "segmented GDN local shape");

    const uint64_t row_bytes = k * 2u;
    std::vector<uint8_t> want;
    const uint64_t expected_rows[3] = {0, 64, 128};
    for (uint64_t seg = 0; seg < 3; ++seg) {
        const std::vector<uint8_t> seg_bytes =
            gather_rows(bytes, row_bytes, expected_rows[seg], 32);
        want.insert(want.end(), seg_bytes.begin(), seg_bytes.end());
    }
    check_bytes("segmented GDN data", owned.data, want);
}

static void test_expert_3d_partition() {
    const uint64_t e = 8, n = 16, k = 32;
    const uint64_t rows = e * n;
    std::vector<uint8_t> bytes(rows * k * 2u);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>(i * 3 + 1);
    }
    QuantizedTensorView g = make_bf16_view(
        bytes, {static_cast<int64_t>(e), static_cast<int64_t>(n),
                static_cast<int64_t>(k)});

    OwnedCanonicalTensor owned;
    QuantizedTensorView local;
    Status st = materialize_rank_local_canonical(
        desc_of(0, {8, 16, 32}, 1, 2, {rng(4, 4)}), g, owned, local);
    check(st.ok(), "3D expert axis0 materialize");
    if (!st.ok()) return;
    check(local.logical_shape == std::vector<int64_t>({4, 16, 32}),
          "3D expert local shape");

    const uint64_t row_bytes = k * 2u;
    check_bytes("3D expert data", owned.data,
                gather_rows(bytes, row_bytes, 4 * n, 4 * n));

    TensorPartitionDesc gate_up = desc_of(
        0, {512, 1280, 2560}, 1, 2, {rng(256, 256)});
    auto gate_up_local = tensor_partition_local_shape(gate_up);
    check(gate_up_local.ok(), "expert gate_up plan computes");
    if (gate_up_local.ok()) {
        check(gate_up_local.value() == std::vector<int64_t>({256, 1280, 2560}),
              "expert gate_up local shape");
    }
    TensorPartitionDesc down = desc_of(0, {512, 2560, 640}, 0, 2, {rng(0, 256)});
    auto down_local = tensor_partition_local_shape(down);
    check(down_local.ok() && down_local.value() == std::vector<int64_t>({256, 2560, 640}),
          "expert down local shape");
    check(validate_tensor_partition(gate_up).ok(), "expert gate_up descriptor valid");
}

static void test_fp8_partition() {
    const uint64_t rows = 256, k = 256;
    fp8::Fp8BlockCanonicalStore store;
    store.init(1, rows, k, k);
    const std::vector<float> src = make_source(rows, k, 909);
    if (!store.quantize_matrix(0, src.data(), k)) fail("fp8 quantize_matrix");

    QuantizedTensorView g;
    g.name = "w";
    g.encoding = QuantizedEncoding::Fp8;
    g.logical_shape = {256, 256};
    g.k_padded = 256;
    g.codes = ByteSpan{store.codes.data(), store.codes.size()};
    g.metadata1 = ByteSpan{store.scales.data(), store.scales.size() * sizeof(float)};

    const uint64_t codes_row = k;
    const uint64_t scale_row = (k / 128u) * sizeof(float);
    const uint64_t grid_rows = rows / 128u;
    const std::vector<uint8_t> want_scales(
        reinterpret_cast<const uint8_t*>(store.scales.data()),
        reinterpret_cast<const uint8_t*>(store.scales.data()) +
            store.scales.size() * sizeof(float));

    OwnedCanonicalTensor r0, r1;
    QuantizedTensorView v0, v1;

    Status n0 = materialize_rank_local_canonical(
        desc_of(0, {256, 256}, 0, 2, {rng(0, 128)}), g, r0, v0);
    Status n1 = materialize_rank_local_canonical(
        desc_of(0, {256, 256}, 1, 2, {rng(128, 128)}), g, r1, v1);
    check(n0.ok() && n1.ok(), "fp8 N partition materializes");
    if (n0.ok() && n1.ok()) {
        check(v0.logical_shape == std::vector<int64_t>({128, 256}), "fp8 N rank0 shape");
        check(v1.logical_shape == std::vector<int64_t>({128, 256}), "fp8 N rank1 shape");
        check(v0.k_padded == 256 && v1.k_padded == 256, "fp8 N local k_padded");
        std::vector<uint8_t> codes = r0.codes;
        codes.insert(codes.end(), r1.codes.begin(), r1.codes.end());
        check_bytes("fp8 N codes round trip", codes, store.codes);
        std::vector<uint8_t> scales = r0.metadata1;
        scales.insert(scales.end(), r1.metadata1.begin(), r1.metadata1.end());
        check_bytes("fp8 N scales round trip", scales, want_scales);
        check(r0.metadata1.size() == scale_row && r1.metadata1.size() == scale_row,
              "fp8 N scale row is one grid row");
    }

    Status k0 = materialize_rank_local_canonical(
        desc_of(1, {256, 256}, 0, 2, {rng(0, 128)}), g, r0, v0);
    Status k1 = materialize_rank_local_canonical(
        desc_of(1, {256, 256}, 1, 2, {rng(128, 128)}), g, r1, v1);
    check(k0.ok() && k1.ok(), "fp8 K partition materializes");
    if (k0.ok() && k1.ok()) {
        check(v0.logical_shape == std::vector<int64_t>({256, 128}), "fp8 K rank0 shape");
        check(v1.logical_shape == std::vector<int64_t>({256, 128}), "fp8 K rank1 shape");
        check(v0.k_padded == 128 && v1.k_padded == 128, "fp8 K local k_padded");
        std::vector<uint8_t> codes(rows * k);
        for (uint64_t r = 0; r < rows; ++r) {
            std::memcpy(codes.data() + r * k, r0.codes.data() + r * 128u, 128u);
            std::memcpy(codes.data() + r * k + 128u, r1.codes.data() + r * 128u, 128u);
        }
        check_bytes("fp8 K codes round trip", codes, store.codes);
        std::vector<uint8_t> scales(grid_rows * scale_row);
        for (uint64_t u = 0; u < grid_rows; ++u) {
            std::memcpy(scales.data() + u * scale_row, r0.metadata1.data() + u * 4u, 4u);
            std::memcpy(scales.data() + u * scale_row + 4u,
                        r1.metadata1.data() + u * 4u, 4u);
        }
        check_bytes("fp8 K scales round trip", scales, want_scales);
    }

    OwnedCanonicalTensor bad;
    QuantizedTensorView bad_local;
    Status n_misaligned = materialize_rank_local_canonical(
        desc_of(0, {256, 256}, 0, 2, {rng(0, 64)}), g, bad, bad_local);
    check(!n_misaligned.ok() &&
              n_misaligned.code() == Status::Code::invalid_argument,
          "fp8 N range off the 128-row scale tile rejected");

    Status k_misaligned = materialize_rank_local_canonical(
        desc_of(1, {256, 256}, 0, 2, {rng(0, 64)}), g, bad, bad_local);
    check(!k_misaligned.ok() &&
              k_misaligned.code() == Status::Code::invalid_argument,
          "fp8 K range off the 128-K scale tile rejected");

    std::vector<uint8_t> codes3(256u * 256u, 0u);
    QuantizedTensorView fp8_nd;
    fp8_nd.name = "w";
    fp8_nd.encoding = QuantizedEncoding::Fp8;
    fp8_nd.logical_shape = {2, 128, 256};
    fp8_nd.k_padded = 256;
    fp8_nd.codes = ByteSpan{codes3.data(), codes3.size()};
    Status nd = materialize_rank_local_canonical(
        desc_of(1, {2, 128, 256}, 0, 2, {rng(0, 128)}), fp8_nd, bad, bad_local);
    check(!nd.ok() && nd.code() == Status::Code::unsupported,
          "fp8 block128 partition outside 2D rejected as unsupported");
}

static void test_reject_cases() {
    const uint64_t rows = 256, k = 128;
    std::vector<uint8_t> codes(rows * k, 0u);
    std::vector<uint8_t> scales(2u * sizeof(float), 0u);
    QuantizedTensorView g;
    g.name = "w";
    g.encoding = QuantizedEncoding::Fp8;
    g.logical_shape = {256, 128};
    g.k_padded = 128;
    g.codes = ByteSpan{codes.data(), codes.size()};
    g.metadata1 = ByteSpan{scales.data(), scales.size()};

    OwnedCanonicalTensor owned;
    QuantizedTensorView local;

    Status shape_mismatch = materialize_rank_local_canonical(
        desc_of(0, {8, 32}, 0, 2, {rng(0, 4)}), g, owned, local);
    check(!shape_mismatch.ok(), "global_shape mismatch rejected");

    QuantizedTensorView truncated = g;
    truncated.codes = ByteSpan{codes.data(), codes.size() - 1};
    Status bad_size = materialize_rank_local_canonical(
        desc_of(0, {256, 128}, 0, 2, {rng(0, 128)}), truncated, owned, local);
    check(!bad_size.ok(), "truncated codes rejected");

    QuantizedTensorView missing = g;
    missing.codes = ByteSpan{};
    Status missing_codes = materialize_rank_local_canonical(
        desc_of(0, {256, 128}, 0, 2, {rng(0, 128)}), missing, owned, local);
    check(!missing_codes.ok(), "missing codes rejected");

    QuantizedTensorView no_payload;
    no_payload.name = "w";
    no_payload.encoding = QuantizedEncoding::Fp8;
    no_payload.logical_shape = {8, 64};
    no_payload.k_padded = 128;
    Status fp8_status = materialize_rank_local_canonical(
        desc_of(0, {8, 64}, 0, 2, {rng(0, 4)}), no_payload, owned, local);
    check(!fp8_status.ok(), "fp8 view carrying no payload rejected");

    QuantizedTensorView bad_bf16 = g;
    bad_bf16.encoding = QuantizedEncoding::Bf16;
    Status bf16_codes = materialize_rank_local_canonical(
        desc_of(0, {256, 128}, 0, 2, {rng(0, 128)}), bad_bf16, owned, local);
    check(!bf16_codes.ok(), "bf16 view carrying codes rejected");
}

static QuantizedTensorView make_exl3_view(const std::vector<uint8_t>& codes,
                                          const std::vector<uint8_t>& suh,
                                          const std::vector<uint8_t>& svh,
                                          const std::vector<uint8_t>& marker,
                                          std::vector<int64_t> shape) {
    QuantizedTensorView v;
    v.name = "w";
    v.encoding = QuantizedEncoding::Exl3;
    v.logical_shape = std::move(shape);
    v.k_padded = v.logical_shape.back();
    v.exl3_ka = 1;
    v.exl3_half = false;
    v.codes = ByteSpan{codes.data(), codes.size()};
    v.suh = ByteSpan{suh.data(), suh.size()};
    v.svh = ByteSpan{svh.data(), svh.size()};
    v.marker = ByteSpan{marker.data(), marker.size()};
    return v;
}

static void test_exl3_partition() {
    const uint64_t m = 32, k = 32;
    const uint64_t kt = k / 16, nt = m / 16;
    const uint64_t row_bytes = 32;
    std::vector<uint8_t> codes(kt * nt * row_bytes);
    for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<uint8_t>(i);
    std::vector<uint8_t> suh(k * 2), svh(m * 2);
    for (uint64_t i = 0; i < k; ++i) {
        suh[2 * i] = static_cast<uint8_t>(0x10 + i);
        suh[2 * i + 1] = static_cast<uint8_t>(0x40 + i);
    }
    for (uint64_t i = 0; i < m; ++i) {
        svh[2 * i] = static_cast<uint8_t>(0x20 + i);
        svh[2 * i + 1] = static_cast<uint8_t>(0x50 + i);
    }
    const std::vector<uint8_t> marker{0xAA, 0xBB, 0xCC, 0xDD};
    const QuantizedTensorView g = make_exl3_view(codes, suh, svh, marker, {32, 32});

    {
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {32, 32}, 0, 2, {rng(0, 16)}), g, owned, local);
        check(st.ok(), "exl3 axis0 rank0 materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({16, 32}),
                  "exl3 axis0 local shape");
            check(local.k_padded == 32, "exl3 axis0 local k_padded");
            check(local.exl3_ka == 1 && !local.exl3_half, "exl3 axis0 carries the bit rate");
            check(local.marker.size == 4, "exl3 axis0 keeps the marker");
            std::vector<uint8_t> want;
            want.insert(want.end(), codes.begin(), codes.begin() + row_bytes);
            want.insert(want.end(), codes.begin() + 2 * row_bytes,
                        codes.begin() + 3 * row_bytes);
            check_bytes("exl3 axis0 rank0 trellis", owned.codes, want);
            check_bytes("exl3 axis0 rank0 suh", owned.suh, suh);
            check_bytes("exl3 axis0 rank0 svh", owned.svh,
                        std::vector<uint8_t>(svh.begin(), svh.begin() + 16 * 2));
            check_bytes("exl3 axis0 rank0 marker", owned.marker, marker);
        }
    }
    {
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {32, 32}, 1, 2, {rng(16, 16)}), g, owned, local);
        check(st.ok(), "exl3 axis0 rank1 materialize");
        if (st.ok()) {
            std::vector<uint8_t> want;
            want.insert(want.end(), codes.begin() + row_bytes,
                        codes.begin() + 2 * row_bytes);
            want.insert(want.end(), codes.begin() + 3 * row_bytes,
                        codes.begin() + 4 * row_bytes);
            check_bytes("exl3 axis0 rank1 trellis", owned.codes, want);
            check_bytes("exl3 axis0 rank1 svh", owned.svh,
                        std::vector<uint8_t>(svh.begin() + 16 * 2, svh.end()));
        }
    }
    {
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(1, {32, 32}, 1, 2, {rng(16, 16)}), g, owned, local);
        check(st.ok(), "exl3 axis1 rank1 materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({32, 16}),
                  "exl3 axis1 local shape");
            check(local.k_padded == 16, "exl3 axis1 local k_padded");
            std::vector<uint8_t> want(codes.begin() + 2 * row_bytes, codes.end());
            check_bytes("exl3 axis1 rank1 trellis", owned.codes, want);
            check_bytes("exl3 axis1 rank1 suh", owned.suh,
                        std::vector<uint8_t>(suh.begin() + 16 * 2, suh.end()));
            check_bytes("exl3 axis1 rank1 svh", owned.svh, svh);
        }
    }
    {
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        QuantizedTensorView bad = g;
        bad.suh = ByteSpan{};
        Status st = materialize_rank_local_canonical(
            desc_of(0, {32, 32}, 0, 2, {rng(0, 16)}), bad, owned, local);
        check(!st.ok(), "exl3 view missing suh rejected");
    }
}

int main() {
    test_bf16_partitions();
    test_segmented_gdn();
    test_expert_3d_partition();
    test_fp8_partition();
    test_exl3_partition();
    test_reject_cases();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
