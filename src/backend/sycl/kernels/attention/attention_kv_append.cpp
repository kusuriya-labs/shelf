// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/attention/attention_kv_append.hip.)
#include <shelf/backend/sycl/kernels/attention/kv_append.h>
#include <shelf/backend/sycl/kernels/attention/kv_types.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>
#include <shelf/backend/sycl/kernels/e4m3_codec.h>

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cfloat>

namespace shelf::syclk {

namespace {
namespace op_kv_append_bf16 {

using bf16 = shelf::bf16_t;

struct Locals {
    uint64_t* token_base_shared;
    uint32_t* valid_shared;
};

inline void kv_append_bf16_body(
    ::sycl::nd_item<1>& it, Locals L,
    const KvAppendCommonArgs& a, bf16* k_pool, bf16* v_pool) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t task = static_cast<uint32_t>(it.get_group(0));
    const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());
    const uint32_t row = task / a.kv_heads;
    const uint32_t head = task % a.kv_heads;

    if (tid == 0u) {
        uint64_t token_base = 0;
        uint32_t valid = 1u;
        const uint32_t slot = a.row_sequence_slots[row];
        if (slot >= a.max_sequences) {
            valid = 0u;
        } else {
            const uint32_t pos = a.row_positions[row];
            const uint32_t logical_block = pos / a.page_tokens;
            const uint32_t offset = pos % a.page_tokens;
            if (logical_block >= a.block_table_stride) {
                valid = 0u;
            } else {
                const uint32_t page = a.block_tables[static_cast<size_t>(slot) * a.block_table_stride + logical_block];
                if (page >= a.num_pages) {
                    valid = 0u;
                } else {
                    token_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer
                               + static_cast<uint64_t>(page) * a.elems_per_page
                               + static_cast<uint64_t>(offset) * a.elems_per_token;
                }
            }
        }
        *L.token_base_shared = token_base;
        *L.valid_shared = valid;
    }
    it.barrier(::sycl::access::fence_space::local_space);
    if (*L.valid_shared == 0u) return;

    const uint64_t dst_base = *L.token_base_shared + static_cast<uint64_t>(head) * head_dim;
    const bf16* k_src = reinterpret_cast<const bf16*>(a.k_input) + static_cast<size_t>(row) * a.k_row_stride + static_cast<size_t>(head) * head_dim;
    const bf16* v_src = reinterpret_cast<const bf16*>(a.v_input) + static_cast<size_t>(row) * a.v_row_stride + static_cast<size_t>(head) * head_dim;
    bf16* k_dst = k_pool + dst_base;
    bf16* v_dst = v_pool + dst_base;

    const uint32_t full = head_dim >> 2;
    const uint32_t rem = head_dim & 3u;
    const bool aligned8 =
        ::shelf::syclk::aligned8(k_src) && ::shelf::syclk::aligned8(k_dst) &&
        ::shelf::syclk::aligned8(v_src) && ::shelf::syclk::aligned8(v_dst);

    if (tid < full) {
        const uint32_t i = tid << 2;
        if (aligned8) {
            const uint64_t kb = load_u64(k_src + i);
            const uint64_t vb = load_u64(v_src + i);
            store_u64(k_dst + i, kb);
            store_u64(v_dst + i, vb);
        } else {
            k_dst[i] = k_src[i]; k_dst[i + 1] = k_src[i + 1];
            k_dst[i + 2] = k_src[i + 2]; k_dst[i + 3] = k_src[i + 3];
            v_dst[i] = v_src[i]; v_dst[i + 1] = v_src[i + 1];
            v_dst[i + 2] = v_src[i + 2]; v_dst[i + 3] = v_src[i + 3];
        }
    } else if (tid == full && rem != 0u) {
        const uint32_t i = full << 2;
        for (uint32_t j = 0; j < rem; ++j) {
            k_dst[i + j] = k_src[i + j];
            v_dst[i + j] = v_src[i + j];
        }
    }
}

} // namespace op_kv_append_bf16
} // namespace

::shelf::Status launch_kv_append_bf16(const KvAppendCommonArgs& common, bf16_t* k_pool,
                                      bf16_t* v_pool, const ::shelf::sycl::Stream& stream) {
    if (common.rows == 0u || common.head_dim == 0u || common.kv_heads == 0u)
        return ::shelf::Status::invalid_argument("bad kv_append shape", __FILE__, __LINE__);
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return ::shelf::Status::invalid_argument("bad kv_append paging", __FILE__, __LINE__);
    if (common.k_input == nullptr || common.v_input == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    const uint32_t grid_x = common.rows * common.kv_heads;
    const uint32_t chunks = (common.head_dim + 3u) / 4u;
    const uint32_t block = chunks <= 32u ? 32u : chunks <= 64u ? 64u : chunks <= 128u ? 128u : 256u;
    const auto a = common;
    using bf16 = shelf::bf16_t;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        ::sycl::local_accessor<uint64_t, 1> token_base(::sycl::range<1>{1}, cgh);
        ::sycl::local_accessor<uint32_t, 1> valid(::sycl::range<1>{1}, cgh);
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid_x * block}, ::sycl::range<1>{block}},
            [=](::sycl::nd_item<1> it) {
                op_kv_append_bf16::kv_append_bf16_body(
                    it,
                    op_kv_append_bf16::Locals{token_base.get_pointer().get(),
                                              valid.get_pointer().get()},
                    a, reinterpret_cast<bf16*>(k_pool), reinterpret_cast<bf16*>(v_pool));
            });
    });
}

namespace {
namespace op_kv_append_fp8 {

using bf16 = shelf::bf16_t;
using fp8 = ::shelf::syclk::fp8e4m3_storage_t;

constexpr float kFp8E4M3Max = 448.0f;

inline fp8 cvt_fp8(float v) { return ::shelf::syclk::e4m3_encode_u8(v); }

struct Locals {
    uint64_t* token_base_shared;
    uint32_t* page_shared;
    uint32_t* offset_shared;
    uint32_t* valid_shared;
    float* k_wave_max;
    float* v_wave_max;
    float* k_scale_shared;
    float* v_scale_shared;
};

inline void impl(::sycl::nd_item<1>& it, Locals L,
                 const KvAppendCommonArgs a, fp8* k_pool, fp8* v_pool,
                 float* k_scale_pool, float* v_scale_pool) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t task = static_cast<uint32_t>(it.get_group(0));
    const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());
    const uint32_t row = task / a.kv_heads;
    const uint32_t head = task % a.kv_heads;

    if (tid == 0u) {
        uint64_t token_base = 0;
        uint32_t page = 0, offset = 0, valid = 1u;
        const uint32_t slot = a.row_sequence_slots[row];
        if (slot >= a.max_sequences) {
            valid = 0u;
        } else {
            const uint32_t pos = a.row_positions[row];
            const uint32_t logical_block = pos / a.page_tokens;
            const uint32_t off = pos % a.page_tokens;
            if (logical_block >= a.block_table_stride) {
                valid = 0u;
            } else {
                const uint32_t pg = a.block_tables[static_cast<size_t>(slot) * a.block_table_stride + logical_block];
                if (pg >= a.num_pages) {
                    valid = 0u;
                } else {
                    page = pg;
                    offset = off;
                    token_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer
                               + static_cast<uint64_t>(page) * a.elems_per_page
                               + static_cast<uint64_t>(offset) * a.elems_per_token;
                }
            }
        }
        *L.token_base_shared = token_base;
        *L.page_shared = page;
        *L.offset_shared = offset;
        *L.valid_shared = valid;
    }
    it.barrier(::sycl::access::fence_space::local_space);
    if (*L.valid_shared == 0u) return;

    const uint64_t dst_base = *L.token_base_shared + static_cast<uint64_t>(head) * head_dim;
    const bf16* k_src = reinterpret_cast<const bf16*>(a.k_input) + static_cast<size_t>(row) * a.k_row_stride + static_cast<size_t>(head) * head_dim;
    const bf16* v_src = reinterpret_cast<const bf16*>(a.v_input) + static_cast<size_t>(row) * a.v_row_stride + static_cast<size_t>(head) * head_dim;
    fp8* k_dst = k_pool + dst_base;
    fp8* v_dst = v_pool + dst_base;

    const auto sg = it.get_sub_group();
    float k_max = 0.0f;
    float v_max = 0.0f;
    const uint32_t bdim = static_cast<uint32_t>(it.get_local_range(0));
    const uint32_t step = bdim << 2;
    for (uint32_t base = tid << 2; base < head_dim; base += step) {
        const uint32_t n = base + 4u <= head_dim ? 4u : (head_dim - base);
        for (uint32_t j = 0; j < n; ++j) {
            const float kf = ::shelf::bf16_to_f32(k_src[base + j]);
            const float vf = ::shelf::bf16_to_f32(v_src[base + j]);
            k_max = ::sycl::fmax(k_max, ::sycl::fabs(kf));
            v_max = ::sycl::fmax(v_max, ::sycl::fabs(vf));
        }
    }
    const uint32_t num_waves = bdim >> 5;
    const uint32_t wave_id = tid >> 5;
    k_max = wave_max_all(sg, k_max);
    v_max = wave_max_all(sg, v_max);
    if ((tid & 31u) == 0u) {
        L.k_wave_max[wave_id] = k_max;
        L.v_wave_max[wave_id] = v_max;
    }
    it.barrier(::sycl::access::fence_space::local_space);
    if (wave_id == 0u) {
        float kf = (tid < num_waves) ? L.k_wave_max[tid] : 0.0f;
        float vf = (tid < num_waves) ? L.v_wave_max[tid] : 0.0f;
        kf = wave_max_all(sg, kf);
        vf = wave_max_all(sg, vf);
        if (tid == 0u) {
            *L.k_scale_shared = (kf == 0.0f) ? 1.0f : kf / kFp8E4M3Max;
            *L.v_scale_shared = (vf == 0.0f) ? 1.0f : vf / kFp8E4M3Max;
        }
    }
    it.barrier(::sycl::access::fence_space::local_space);
    const float k_scale = *L.k_scale_shared;
    const float v_scale = *L.v_scale_shared;

    if (tid == 0u) {
        const size_t si = ((static_cast<size_t>(a.layer) * a.num_pages + *L.page_shared) * a.page_tokens + *L.offset_shared) * a.kv_heads + head;
        k_scale_pool[si] = k_scale;
        v_scale_pool[si] = v_scale;
    }

    for (uint32_t base = tid << 2; base < head_dim; base += step) {
        const uint32_t n = base + 4u <= head_dim ? 4u : (head_dim - base);
        fp8 kb[4];
        fp8 vb[4];
        for (uint32_t j = 0; j < n; ++j) {
            kb[j] = cvt_fp8(::shelf::bf16_to_f32(k_src[base + j]) / k_scale);
            vb[j] = cvt_fp8(::shelf::bf16_to_f32(v_src[base + j]) / v_scale);
        }
        if (n == 4u &&
            (reinterpret_cast<uintptr_t>(k_dst + base) & 3u) == 0u &&
            (reinterpret_cast<uintptr_t>(v_dst + base) & 3u) == 0u) {
            const uint32_t kp = static_cast<uint32_t>(kb[0])
                | (static_cast<uint32_t>(kb[1]) << 8)
                | (static_cast<uint32_t>(kb[2]) << 16)
                | (static_cast<uint32_t>(kb[3]) << 24);
            const uint32_t vp = static_cast<uint32_t>(vb[0])
                | (static_cast<uint32_t>(vb[1]) << 8)
                | (static_cast<uint32_t>(vb[2]) << 16)
                | (static_cast<uint32_t>(vb[3]) << 24);
            std::memcpy(k_dst + base, &kp, 4u);
            std::memcpy(v_dst + base, &vp, 4u);
        } else {
            for (uint32_t j = 0; j < n; ++j) {
                k_dst[base + j] = kb[j];
                v_dst[base + j] = vb[j];
            }
        }
    }
}

} // namespace op_kv_append_fp8

} // namespace

::shelf::Status launch_kv_append_fp8_e4m3(const KvAppendCommonArgs& common,
    ::shelf::syclk::fp8e4m3_storage_t* k_pool, ::shelf::syclk::fp8e4m3_storage_t* v_pool,
    float* k_scale_pool, float* v_scale_pool, const ::shelf::sycl::Stream& stream) {
    if (common.rows == 0u || common.head_dim == 0u || common.kv_heads == 0u)
        return ::shelf::Status::invalid_argument("bad kv_append shape", __FILE__, __LINE__);
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return ::shelf::Status::invalid_argument("bad kv_append paging", __FILE__, __LINE__);
    if (common.k_input == nullptr || common.v_input == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "kv_append fp8 requires wave32 sub-groups", __FILE__, __LINE__);
    const uint32_t grid_x = common.rows * common.kv_heads;
    const uint32_t block = common.head_dim <= 128u ? 32u
                        : common.head_dim <= 256u ? 64u
                        : common.head_dim <= 512u ? 128u
                        : 256u;
    const auto a = common;
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        ::sycl::local_accessor<uint64_t, 1> token_base(::sycl::range<1>{1}, cgh);
        ::sycl::local_accessor<uint32_t, 1> page(::sycl::range<1>{1}, cgh);
        ::sycl::local_accessor<uint32_t, 1> offset(::sycl::range<1>{1}, cgh);
        ::sycl::local_accessor<uint32_t, 1> valid(::sycl::range<1>{1}, cgh);
        ::sycl::local_accessor<float, 1> k_wave_max(::sycl::range<1>{8}, cgh);
        ::sycl::local_accessor<float, 1> v_wave_max(::sycl::range<1>{8}, cgh);
        ::sycl::local_accessor<float, 1> k_scale(::sycl::range<1>{1}, cgh);
        ::sycl::local_accessor<float, 1> v_scale(::sycl::range<1>{1}, cgh);
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid_x * block}, ::sycl::range<1>{block}},
            [=](::sycl::nd_item<1> it) {
                op_kv_append_fp8::impl(
                    it,
                    op_kv_append_fp8::Locals{
                        token_base.get_pointer().get(), page.get_pointer().get(),
                        offset.get_pointer().get(), valid.get_pointer().get(),
                        k_wave_max.get_pointer().get(), v_wave_max.get_pointer().get(),
                        k_scale.get_pointer().get(), v_scale.get_pointer().get()},
                    a, k_pool, v_pool, k_scale_pool, v_scale_pool);
            });
    });
}

} // namespace shelf::syclk
