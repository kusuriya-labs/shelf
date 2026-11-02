#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/device_allocation_view.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <cstdio>
#include <vector>
#include <limits>

#define HIP_ASSERT(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        return 1; \
    } \
} while(0)

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        printf("PASS: %s\n", name);
    } else {
        failed++;
        printf("FAIL: %s\n", name);
    }
}

}

int main() {
    printf("=== Tensor GPU Tests ===\n");

    int device_count = 0;
    HIP_ASSERT(hipGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("SKIP: no GPU available\n");
        return 77;
    }

    HIP_ASSERT(hipSetDevice(0));

    auto arena_result = ps::gpu::GpuArena::create(0, 1024 * 1024);
    if (!arena_result.ok()) {
        printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    auto alloc_result = arena.allocate_aligned(1024, 16);
    if (!alloc_result.ok()) {
        printf("FAIL: arena allocate: %s\n", alloc_result.status().message().c_str());
        return 1;
    }
    auto backing = alloc_result.release();

    auto alloc_result2 = arena.allocate_aligned(1024, 16);
    if (!alloc_result2.ok()) {
        printf("FAIL: arena allocate2: %s\n", alloc_result2.status().message().c_str());
        return 1;
    }
    auto backing2 = alloc_result2.release();

    auto t_result = ps::gpu::Tensor::create<float>(
        backing, std::vector<std::size_t>{2, 3, 4}, std::vector<std::size_t>{12, 4, 1});
    check(t_result.ok(), "tensor create ok");
    if (!t_result.ok()) {
        printf("FAIL: tensor create: %s\n", t_result.status().message().c_str());
        printf("\nResults: %d passed, %d failed\n", passed, failed);
        return 1;
    }
    auto t = t_result.release();

    check(t.ndim() == 3, "ndim == 3");
    check(t.dim(0) == 2, "dim(0) == 2");
    check(t.dim(1) == 3, "dim(1) == 3");
    check(t.dim(2) == 4, "dim(2) == 4");

    check(t.physical_device() == 0, "physical_device from backing is 0");

    {
        std::vector<float> tiny(3, 0.0f);
        auto small_backing = ps::gpu::Tensor::create<float>(
            backing, std::vector<std::size_t>{2, 3, 4}, std::vector<std::size_t>{12, 4, 1});
        check(small_backing.ok(), "create with sufficient backing succeeds");
    }

    {
        auto s = t.slice(0, 1, 1);
        check(s.ok(), "slice ok");
        if (s.ok()) {
            auto sv = s.release();
            check(sv.ndim() == 3, "slice ndim == 3");
            check(sv.dim(0) == 1 && sv.dim(1) == 3 && sv.dim(2) == 4, "slice shape [1,3,4]");
            check(t.dim(0) == 2, "original dim(0) unchanged after slice");
            check(t.view_offset_bytes() == 0, "original view_offset unchanged");
        }
    }

    {
        auto invalid = t.slice(0, 99, 1);
        check(!invalid.ok(), "invalid slice fails (start >= dim)");
    }

    {
        auto dim_out_of_range = t.slice(10, 0, 1);
        check(!dim_out_of_range.ok(), "dimension out of range fails");
    }

    {
        auto slice_overflow = t.slice(0, 1, 2);
        check(!slice_overflow.ok(), "slice overflow fails (start+count > dim)");
    }

    {
        auto slice_count_zero = t.slice(0, 0, 0);
        check(!slice_count_zero.ok(), "slice count zero fails");
    }

    {
        auto shape_mismatch = ps::gpu::Tensor::create<float>(
            backing, std::vector<std::size_t>{2, 3, 4}, std::vector<std::size_t>{12, 4});
        check(!shape_mismatch.ok(), "shape/stride mismatch fails");
    }

    {
        auto strides_mismatch = ps::gpu::Tensor::create<float>(
            backing, std::vector<std::size_t>{2, 3}, std::vector<std::size_t>{12, 4, 1});
        check(!strides_mismatch.ok(), "stride/shape mismatch fails");
    }

    {
        auto initializer_list_mismatch = ps::gpu::Tensor::create<float>(
            backing, {2, 3}, {3});
        check(!initializer_list_mismatch.ok(), "initializer-list shape/stride mismatch fails");
    }

    {
        const std::size_t shape[] = {2, 3};
        const std::size_t strides[] = {3, 1};
        auto null_shape = ps::gpu::Tensor::create<float>(
            backing, static_cast<const std::size_t*>(nullptr), 2, strides);
        auto null_strides = ps::gpu::Tensor::create<float>(
            backing, shape, 2, static_cast<const std::size_t*>(nullptr));
        check(!null_shape.ok(), "null shape pointer fails");
        check(!null_strides.ok(), "null strides pointer fails");
    }

    {
        auto same_shape_result = ps::gpu::Tensor::create<float>(backing, {2, 3, 4}, {12, 4, 1});
        auto different_shape_result = ps::gpu::Tensor::create<float>(backing, {2, 4, 3}, {12, 3, 1});
        check(same_shape_result.ok(), "same-shape tensor create succeeds");
        check(different_shape_result.ok(), "different-shape tensor create succeeds");
        if (same_shape_result.ok() && different_shape_result.ok()) {
            auto same_shape = same_shape_result.release();
            auto different_shape = different_shape_result.release();
            check(t.same_shape(same_shape), "same_shape accepts equal dimensions");
            check(!t.same_shape(different_shape), "same_shape rejects different dimensions");
        }
    }

    {
        auto base_ptr = t.allocation_base();
        check(base_ptr == backing.data(), "allocation_base matches backing data");
        check(t.allocation_bytes() == backing.bytes(), "allocation_bytes from backing");
        check(t.view_offset_bytes() == 0, "original view_offset_bytes is 0");
    }

    {
        auto s = t.slice(0, 1, 1);
        check(s.ok(), "slice for boundary test");
        if (s.ok()) {
            auto sv = s.release();
            check(sv.allocation_base() == t.allocation_base(),
                  "sliced allocation_base preserved");
            check(sv.allocation_bytes() == t.allocation_bytes(),
                  "sliced allocation_bytes preserved");
            check(sv.view_offset_bytes() == 12 * sizeof(float),
                  "sliced view_offset_bytes updated correctly");
            check(sv.physical_device() == t.physical_device(),
                  "sliced physical_device preserved");
        }
    }

    {
        auto s0 = t.slice(1, 0, 1);
        check(s0.ok(), "slice dim 1");
        if (s0.ok()) {
            auto sv0 = s0.release();
            auto s1 = sv0.slice(2, 2, 2);
            check(s1.ok(), "double slice");
            if (s1.ok()) {
                auto sv1 = s1.release();
                check(sv1.allocation_base() == t.allocation_base(),
                      "double-sliced allocation_base preserved");
                check(sv1.allocation_bytes() == t.allocation_bytes(),
                      "double-sliced allocation_bytes preserved");
                check(sv1.physical_device() == t.physical_device(),
                      "double-sliced physical_device preserved");
            }
        }
    }

    {
        auto s = t.slice(0, 1, 1);
        check(s.ok(), "slice for data() test");
        if (s.ok()) {
            auto sv = s.release();
            auto expected_ptr = reinterpret_cast<float*>(
                reinterpret_cast<std::byte*>(backing.data()) + 12 * sizeof(float));
            check(sv.data<float>() == expected_ptr,
                  "sliced data() points to correct offset");
        }
    }

    {
        std::size_t span;
        auto status = t.required_span_bytes(span);
        check(status.ok(), "required_span_bytes status ok");
        check(span == 96, "required_span_bytes correct for [2,3,4]");
    }

    {
        auto bad_stride = ps::gpu::Tensor::create<float>(
            backing, std::vector<std::size_t>{2}, std::vector<std::size_t>{std::numeric_limits<std::size_t>::max()});
        check(!bad_stride.ok(), "shape={2}, stride={SIZE_MAX} rejected");
    }

    {
        std::vector<std::size_t> huge_shape = {std::numeric_limits<std::size_t>::max()};
        std::vector<std::size_t> huge_strides = {1};
        auto huge = ps::gpu::Tensor::create<float>(
            backing, huge_shape, huge_strides);
        check(!huge.ok(), "max_offset exactly SIZE_MAX rejected");
    }

    {
        std::size_t huge_val = std::numeric_limits<std::size_t>::max() - 3;
        std::vector<std::size_t> shape = {2};
        std::vector<std::size_t> strides = {huge_val};
        auto overflow_plus1 = ps::gpu::Tensor::create<float>(
            backing, shape, strides);
        check(!overflow_plus1.ok(), "max_offset + 1 overflow rejected");
    }

    {
        std::size_t big_offset = std::numeric_limits<std::size_t>::max() / sizeof(float);
        std::vector<std::size_t> shape = {2};
        std::vector<std::size_t> strides = {big_offset};
        auto mul_overflow = ps::gpu::Tensor::create<float>(
            backing, shape, strides);
        check(!mul_overflow.ok(), "required byte multiplication overflow rejected");
    }

    {
        auto s0 = t.slice(0, 1, 1);
        check(s0.ok(), "first slice for nested test");
        if (s0.ok()) {
            auto sv0 = s0.release();
            auto s1 = sv0.slice(1, 0, 1);
            check(s1.ok(), "nested slice remains within allocation");
        }
    }

    {
        auto last = t.slice(0, 1, 1);
        check(last.ok(), "last valid element slice succeeds");
    }

    {
        auto alloc_exact = arena.allocate_aligned(96, 16);
        check(alloc_exact.ok(), "96 byte alloc ok");
        if (alloc_exact.ok()) {
            auto vb = alloc_exact.release();
            auto exact = ps::gpu::Tensor::create<float>(
                vb, std::vector<std::size_t>{24}, std::vector<std::size_t>{1});
            check(exact.ok(), "96 byte alloc can create 96 byte Tensor");
        }

        auto alloc_small = arena.allocate_aligned(96, 16);
        check(alloc_small.ok(), "96 byte alloc2 ok");
        if (alloc_small.ok()) {
            auto vb = alloc_small.release();
            auto too_big = ps::gpu::Tensor::create<float>(
                vb, std::vector<std::size_t>{25}, std::vector<std::size_t>{1});
            check(!too_big.ok(), "96 byte alloc cannot create 100 byte Tensor");
        }
    }

    {
        auto t2 = ps::gpu::Tensor::create<float>(
            backing2, std::vector<std::size_t>{10}, std::vector<std::size_t>{1});
        check(t2.ok(), "t2 from backing2");
        if (t2.ok()) {
            auto tv = t2.release();
            check(tv.allocation_base() == backing2.data(), "t2 base is backing2");
            auto s = t.slice(0, 1, 1);
            check(s.ok(), "t slice");
            if (s.ok()) {
                auto sv = s.release();
                check(sv.allocation_base() != backing2.data(),
                      "t slice cannot reach backing2");
            }
        }
    }

    {
        auto orig_base = t.allocation_base();
        auto orig_bytes = t.allocation_bytes();
        auto orig_device = t.physical_device();
        auto moved = std::move(t);
        check(moved.allocation_base() == orig_base, "move preserves base");
        check(moved.allocation_bytes() == orig_bytes, "move preserves bytes");
        check(moved.physical_device() == orig_device, "move preserves device");
    }

    {
        const std::size_t H = 64;
        auto backing_bf16 = arena.allocate_aligned(3 * H * sizeof(ps::bf16_t), 16);
        check(backing_bf16.ok(), "bf16 backing alloc ok");
        if (backing_bf16.ok()) {
            auto b = backing_bf16.release();
            auto t_bf16 = ps::gpu::Tensor::create<ps::bf16_t>(
                b, std::vector<std::size_t>{3, H}, std::vector<std::size_t>{H, 1});
            check(t_bf16.ok(), "bf16 [3,H] create ok");
            if (t_bf16.ok()) {
                auto tv = t_bf16.release();
                check(tv.element_size() == sizeof(ps::bf16_t), "bf16 element_size is 2");
                auto row1 = tv.slice(0, 1, 1);
                auto row2 = tv.slice(0, 2, 1);
                check(row1.ok() && row2.ok(), "bf16 row slices ok");
                if (row1.ok() && row2.ok()) {
                    check(row1.release().view_offset_bytes() == H * sizeof(ps::bf16_t),
                          "bf16 row1 slice offset == H * sizeof(bf16)");
                    check(row2.release().view_offset_bytes() == 2 * H * sizeof(ps::bf16_t),
                          "bf16 row2 slice offset == 2 * H * sizeof(bf16)");
                }
            }
        }
    }

    {
        const std::size_t V = 128;
        auto backing_f32 = arena.allocate_aligned(3 * V * sizeof(float), 16);
        check(backing_f32.ok(), "f32 backing alloc ok");
        if (backing_f32.ok()) {
            auto b = backing_f32.release();
            auto t_f32 = ps::gpu::Tensor::create<float>(
                b, std::vector<std::size_t>{3, V}, std::vector<std::size_t>{V, 1});
            check(t_f32.ok(), "f32 [3,V] create ok");
            if (t_f32.ok()) {
                auto tv = t_f32.release();
                check(tv.element_size() == sizeof(float), "f32 element_size is 4");
                auto row1 = tv.slice(0, 1, 1);
                auto row2 = tv.slice(0, 2, 1);
                check(row1.ok() && row2.ok(), "f32 row slices ok");
                if (row1.ok() && row2.ok()) {
                    check(row1.release().view_offset_bytes() == V * sizeof(float),
                          "f32 row1 slice offset == V * sizeof(float)");
                    check(row2.release().view_offset_bytes() == 2 * V * sizeof(float),
                          "f32 row2 slice offset == 2 * V * sizeof(float)");
                }
            }
        }
    }

    {
        auto backing_u32 = arena.allocate_aligned(4 * sizeof(uint32_t), 16);
        check(backing_u32.ok(), "u32 backing alloc ok");
        if (backing_u32.ok()) {
            auto b = backing_u32.release();
            auto t_u32 = ps::gpu::Tensor::create<uint32_t>(
                b, std::vector<std::size_t>{4}, std::vector<std::size_t>{1});
            check(t_u32.ok(), "u32 [4] create ok");
            if (t_u32.ok()) {
                auto tv = t_u32.release();
                check(tv.element_size() == sizeof(uint32_t), "u32 element_size is 4");
                auto row1 = tv.slice(0, 1, 1);
                check(row1.ok(), "u32 row slice ok");
                if (row1.ok()) {
                    check(row1.release().view_offset_bytes() == sizeof(uint32_t),
                          "u32 row1 slice offset == sizeof(uint32_t)");
                }
            }
        }
    }

    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
