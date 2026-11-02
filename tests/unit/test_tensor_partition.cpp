#include <phaseshift/weights/tensor_partition.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using ps::Status;
using ps::weights::TensorPartitionDesc;
using ps::weights::TensorPartitionPlan;
using ps::weights::TensorPartitionRange;
using ps::weights::tensor_partition_local_shape;
using ps::weights::validate_tensor_partition;
using ps::weights::validate_tensor_partition_block_alignment;
using ps::weights::validate_tensor_partition_plan;

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

static TensorPartitionDesc single_range(int32_t axis, std::vector<int64_t> shape,
                                        uint32_t index, uint32_t count,
                                        uint64_t offset, uint64_t extent) {
    TensorPartitionDesc d;
    d.axis = axis;
    d.index = index;
    d.count = count;
    d.global_shape = std::move(shape);
    d.ranges = {rng(offset, extent)};
    return d;
}

static void test_valid_descriptors() {
    check(validate_tensor_partition(single_range(0, {32, 64}, 0, 2, 0, 16)).ok(),
          "single range rank0 valid");
    check(validate_tensor_partition(single_range(0, {32, 64}, 1, 2, 16, 16)).ok(),
          "single range rank1 valid");
    check(validate_tensor_partition(single_range(1, {32, 64}, 1, 2, 32, 32)).ok(),
          "K axis single range valid");

    TensorPartitionDesc seg = single_range(0, {768}, 0, 2, 0, 128);
    seg.ranges = {rng(0, 128), rng(256, 128), rng(512, 256)};
    check(validate_tensor_partition(seg).ok(), "segmented ranges valid");

    TensorPartitionDesc ordered = single_range(0, {768}, 0, 2, 0, 128);
    ordered.ranges = {rng(512, 256), rng(0, 128), rng(256, 128)};
    check(validate_tensor_partition(ordered).ok(), "unordered disjoint ranges valid");
    check(ordered.ranges[0].global_offset == 512 && ordered.ranges[1].global_offset == 0,
          "validate does not reorder ranges");

    auto local = tensor_partition_local_shape(seg);
    check(local.ok(), "segmented local shape computes");
    if (local.ok()) {
        check(local.value().size() == 1 && local.value()[0] == 512,
              "segmented local axis = sum of extents");
    }
    auto local2 = tensor_partition_local_shape(single_range(1, {32, 64}, 0, 2, 16, 32));
    check(local2.ok() && local2.value() == std::vector<int64_t>({32, 32}),
          "K partition local shape keeps other dims");
}

static void test_invalid_descriptors() {
    TensorPartitionDesc base = single_range(0, {32, 64}, 0, 2, 0, 16);

    {
        TensorPartitionDesc d = base;
        d.count = 0;
        check(!validate_tensor_partition(d).ok(), "count=0 rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.count = 1;
        check(!validate_tensor_partition(d).ok(), "count=1 rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.index = 2;
        check(!validate_tensor_partition(d).ok(), "index>=count rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.global_shape.clear();
        check(!validate_tensor_partition(d).ok(), "empty global_shape rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.global_shape = {0, 64};
        check(!validate_tensor_partition(d).ok(), "non-positive global dim rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.axis = -1;
        check(!validate_tensor_partition(d).ok(), "axis<0 rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.axis = 2;
        check(!validate_tensor_partition(d).ok(), "axis>=ndim rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.ranges.clear();
        check(!validate_tensor_partition(d).ok(), "empty ranges rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.ranges = {rng(0, 0)};
        check(!validate_tensor_partition(d).ok(), "zero extent rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.ranges = {rng(32, 16)};
        check(!validate_tensor_partition(d).ok(), "offset at axis end rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.ranges = {rng(24, 16)};
        check(!validate_tensor_partition(d).ok(), "range beyond axis end rejected");
    }
    {
        TensorPartitionDesc d = single_range(0, {64}, 0, 2, 0, 32);
        d.ranges = {rng(0, 32), rng(16, 16)};
        check(!validate_tensor_partition(d).ok(), "overlapping ranges rejected");
    }
    {
        TensorPartitionDesc d = single_range(0, {64}, 0, 2, 0, 32);
        d.ranges = {rng(48, 16), rng(16, 40)};
        check(!validate_tensor_partition(d).ok(), "overlapping unordered ranges rejected");
    }
    {
        TensorPartitionDesc d = base;
        d.count = 0;
        check(validate_tensor_partition(base).ok(), "base still valid");
        check(!tensor_partition_local_shape(d).ok(), "local shape rejects invalid desc");
    }
}

static void test_block_alignment() {
    TensorPartitionDesc k_axis = single_range(1, {16, 2560}, 0, 2, 1280, 1280);
    check(validate_tensor_partition_block_alignment(k_axis, 32).ok(),
          "K range aligned to 32 accepted");
    check(validate_tensor_partition_block_alignment(k_axis, 0).ok(),
          "block_elements=0 skips alignment");

    TensorPartitionDesc misaligned_offset = single_range(1, {16, 2560}, 0, 2, 16, 1280);
    check(!validate_tensor_partition_block_alignment(misaligned_offset, 32).ok(),
          "K offset not multiple of 32 rejected");

    TensorPartitionDesc misaligned_extent = single_range(1, {16, 2560}, 0, 2, 0, 48);
    check(!validate_tensor_partition_block_alignment(misaligned_extent, 32).ok(),
          "K extent not multiple of 32 rejected");

    TensorPartitionDesc row_axis = single_range(0, {16, 64}, 0, 2, 8, 8);
    check(validate_tensor_partition_block_alignment(row_axis, 32).ok(),
          "row axis partition has no K block constraint");
}

static void test_plan() {
    TensorPartitionPlan plan;
    plan.tp_size = 2;
    plan.tp_rank = 0;
    check(validate_tensor_partition_plan(plan).ok(), "empty plan valid");
    plan.tensors["w"] = single_range(0, {32, 64}, 0, 2, 0, 16);
    check(validate_tensor_partition_plan(plan).ok(), "matching plan entry valid");

    TensorPartitionPlan bad_size;
    bad_size.tp_size = 0;
    check(!validate_tensor_partition_plan(bad_size).ok(), "tp_size=0 rejected");

    TensorPartitionPlan bad_rank;
    bad_rank.tp_size = 2;
    bad_rank.tp_rank = 2;
    check(!validate_tensor_partition_plan(bad_rank).ok(), "tp_rank>=tp_size rejected");

    TensorPartitionPlan mismatch = plan;
    mismatch.tensors["w"] = single_range(0, {32, 64}, 1, 2, 16, 16);
    check(!validate_tensor_partition_plan(mismatch).ok(),
          "entry index must match plan tp_rank");

    TensorPartitionPlan count_mismatch = plan;
    count_mismatch.tensors["w"] = single_range(0, {32, 64}, 0, 4, 0, 8);
    check(!validate_tensor_partition_plan(count_mismatch).ok(),
          "entry count must match plan tp_size");

    TensorPartitionPlan single;
    single.tp_size = 1;
    single.tp_rank = 0;
    single.tensors["w"] = single_range(0, {32, 64}, 0, 2, 0, 16);
    check(!validate_tensor_partition_plan(single).ok(),
          "tp_size=1 plan must not carry partition entries");
}

int main() {
    test_valid_descriptors();
    test_invalid_descriptors();
    test_block_alignment();
    test_plan();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
