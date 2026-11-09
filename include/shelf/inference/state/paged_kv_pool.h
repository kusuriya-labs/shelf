#pragma once

#include <shelf/inference/state/paged_types.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/kernels/kv_types.h>
#include <hip/hip_runtime.h>
#include <functional>
#include <vector>

namespace shelf {
namespace state {

using ::shelf::kernel::KVCacheDType;
using ::shelf::kernel::fp8e4m3_storage_t;
using ::shelf::kernel::OctaveKVCacheFormat;

struct PagedKVBatchDeviceView {
    bf16_t* k_pool = nullptr;
    bf16_t* v_pool = nullptr;

    const uint32_t* block_tables = nullptr;
    uint32_t block_table_stride = 0;

    const SequenceSlotId* request_slots = nullptr;
    const uint32_t* query_start_loc = nullptr;
    const uint32_t* prefix_lens = nullptr;
    const uint32_t* sequence_lens = nullptr;

    uint32_t num_requests = 0;

    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t attention_layer = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
};

struct PagedKVBatchFP8DeviceView {
    fp8e4m3_storage_t* k_pool = nullptr;
    fp8e4m3_storage_t* v_pool = nullptr;
    float* k_scale_pool = nullptr;
    float* v_scale_pool = nullptr;

    const uint32_t* block_tables = nullptr;
    uint32_t block_table_stride = 0;

    const SequenceSlotId* request_slots = nullptr;
    const uint32_t* query_start_loc = nullptr;
    const uint32_t* prefix_lens = nullptr;
    const uint32_t* sequence_lens = nullptr;

    uint32_t num_requests = 0;

    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t attention_layer = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
};

struct PagedKVPoolBF16View {
    bf16_t* k_pool = nullptr;
    bf16_t* v_pool = nullptr;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
};

struct PagedKVPoolFP8View {
    fp8e4m3_storage_t* k_pool = nullptr;
    fp8e4m3_storage_t* v_pool = nullptr;
    float* k_scale_pool = nullptr;
    float* v_scale_pool = nullptr;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
};

struct PagedKVPoolOctaveView {
    uint8_t* k_pool = nullptr;
    uint8_t* v_pool = nullptr;
    const float* levels = nullptr;
    bf16_t* k_scratch = nullptr;
    bf16_t* v_scratch = nullptr;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t k_slot_bytes = 0;
    uint32_t v_slot_bytes = 0;
    uint32_t k_bytes_per_token = 0;
    uint32_t v_bytes_per_token = 0;
    uint32_t k_bytes_per_page = 0;
    uint32_t v_bytes_per_page = 0;
    uint32_t k_bytes_per_layer = 0;
    uint32_t v_bytes_per_layer = 0;
    OctaveKVCacheFormat format{};
};

class PagedKVPool {
 public:
    static Result<PagedKVPool> create(
        gpu::GpuArena& arena,
        uint32_t num_pages,
        uint32_t page_tokens,
        uint32_t kv_heads,
        uint32_t head_dim);

    static Result<PagedKVPool> create(
        gpu::GpuArena& arena,
        uint32_t num_pages,
        uint32_t page_tokens,
        uint32_t num_attention_layers,
        uint32_t kv_heads,
        uint32_t head_dim);

    static Result<PagedKVPool> create(
        gpu::GpuArena& arena,
        uint32_t num_pages,
        uint32_t page_tokens,
        uint32_t num_attention_layers,
        uint32_t kv_heads,
        uint32_t head_dim,
        KVCacheDType dtype,
        OctaveKVCacheFormat octave_format = {});

    static std::size_t page_bytes(
        uint32_t page_tokens,
        uint32_t num_attention_layers,
        uint32_t kv_heads,
        uint32_t head_dim,
        KVCacheDType dtype,
        OctaveKVCacheFormat octave_format = {}) noexcept;

    Result<PageId> allocate();

    Status adopt(PageId page);

    Status attach(PageId page);

    Status retain_cache(PageId page);

    Status release_cache(PageId page);

    Status release(PageId page);

    Status release_many(const std::vector<PageId>& pages);

    Status validate_release_many(const std::vector<PageId>& pages) const;

    gpu::Tensor k_view(uint32_t layer, PageId page) const;

    gpu::Tensor v_view(uint32_t layer, PageId page) const;

    Status copy_page_from(
        PageId dst_page,
        const PagedKVPool& src,
        PageId src_page,
        hipStream_t stream);

    const gpu::Tensor& k_pool() const noexcept { return k_pool_; }
    const gpu::Tensor& v_pool() const noexcept { return v_pool_; }
    const gpu::Tensor& k_scale_pool() const noexcept { return k_scale_pool_; }
    const gpu::Tensor& v_scale_pool() const noexcept { return v_scale_pool_; }
    const gpu::Tensor& octave_levels() const noexcept { return octave_levels_; }
    const gpu::Tensor& octave_k_scratch() const noexcept { return octave_k_scratch_; }
    const gpu::Tensor& octave_v_scratch() const noexcept { return octave_v_scratch_; }
    KVCacheDType dtype() const noexcept { return dtype_; }
    OctaveKVCacheFormat octave_format() const noexcept { return octave_format_; }

    uint32_t num_pages() const noexcept { return num_pages_; }
    uint32_t page_tokens() const noexcept { return page_tokens_; }
    uint32_t num_attention_layers() const noexcept { return num_attention_layers_; }
    uint32_t kv_heads() const noexcept { return kv_heads_; }
    uint32_t head_dim() const noexcept { return head_dim_; }

    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    std::size_t bytes_per_page() const noexcept {
        return reserved_bytes_ / static_cast<std::size_t>(num_pages_);
    }

    uint32_t num_used_pages() const noexcept { return used_count_; }

    uint32_t num_cached_pages() const noexcept { return cached_count_; }

    uint32_t num_free_pages() const noexcept { return num_pages_ - used_count_; }

    uint32_t num_truly_free_pages() const noexcept {
        return static_cast<uint32_t>(free_list_.size());
    }

    void set_reclaimer(std::function<bool(uint32_t)> reclaimer) {
        reclaimer_ = std::move(reclaimer);
    }

    std::size_t used_bytes() const noexcept {
        return static_cast<std::size_t>(used_count_) * bytes_per_page();
    }

    std::size_t free_bytes() const noexcept {
        return reserved_bytes_ - used_bytes();
    }

 private:
    gpu::Tensor k_pool_;
    gpu::Tensor v_pool_;
    gpu::Tensor k_scale_pool_;
    gpu::Tensor v_scale_pool_;
    gpu::Tensor octave_levels_;
    gpu::Tensor octave_k_scratch_;
    gpu::Tensor octave_v_scratch_;
    KVCacheDType dtype_ = KVCacheDType::BF16;
    OctaveKVCacheFormat octave_format_{};

    uint32_t num_pages_;
    uint32_t page_tokens_;
    uint32_t num_attention_layers_;
    uint32_t kv_heads_;
    uint32_t head_dim_;

    std::vector<bool> in_use_;
    std::vector<uint32_t> free_list_;

    enum PageState : uint8_t {
        kPageSeq = 0,
        kPageFree = 1,
        kPageCached = 2,
    };

    void free_remove(PageId page);
    void cached_remove(PageId page);

    std::vector<uint32_t> seq_refs_;
    std::vector<uint32_t> cache_refs_;
    std::vector<uint8_t> page_state_;
    std::vector<uint32_t> free_pos_;
    std::vector<uint32_t> cached_list_;
    std::vector<uint32_t> cached_pos_;
    uint32_t cached_count_ = 0;
    std::function<bool(uint32_t)> reclaimer_;

    std::size_t reserved_bytes_;
    uint32_t used_count_;

    PagedKVPool() = default;
};

inline PagedKVPoolBF16View make_paged_kv_bf16_view(const PagedKVPool& pool) {
    PagedKVPoolBF16View v{};
    v.k_pool = pool.k_pool().data<bf16_t>();
    v.v_pool = pool.v_pool().data<bf16_t>();
    v.page_tokens = pool.page_tokens();
    v.num_pages = pool.num_pages();
    v.num_attention_layers = pool.num_attention_layers();
    v.kv_heads = pool.kv_heads();
    v.head_dim = pool.head_dim();
    v.elems_per_token = pool.kv_heads() * pool.head_dim();
    v.elems_per_page = v.page_tokens * v.elems_per_token;
    v.elems_per_layer = v.num_pages * v.elems_per_page;
    return v;
}

inline PagedKVPoolFP8View make_paged_kv_fp8_view(const PagedKVPool& pool) {
    PagedKVPoolFP8View v{};
    v.k_pool = pool.k_pool().data<fp8e4m3_storage_t>();
    v.v_pool = pool.v_pool().data<fp8e4m3_storage_t>();
    v.k_scale_pool = pool.k_scale_pool().data<float>();
    v.v_scale_pool = pool.v_scale_pool().data<float>();
    v.page_tokens = pool.page_tokens();
    v.num_pages = pool.num_pages();
    v.num_attention_layers = pool.num_attention_layers();
    v.kv_heads = pool.kv_heads();
    v.head_dim = pool.head_dim();
    v.elems_per_token = pool.kv_heads() * pool.head_dim();
    v.elems_per_page = v.page_tokens * v.elems_per_token;
    v.elems_per_layer = v.num_pages * v.elems_per_page;
    return v;
}

inline PagedKVPoolOctaveView make_paged_kv_octave_view(const PagedKVPool& pool) {
    PagedKVPoolOctaveView v{};
    v.k_pool = pool.k_pool().data<uint8_t>();
    v.v_pool = pool.v_pool().data<uint8_t>();
    v.levels = pool.octave_levels().data<float>();
    v.k_scratch = pool.octave_k_scratch().data<bf16_t>();
    v.v_scratch = pool.octave_v_scratch().data<bf16_t>();
    v.page_tokens = pool.page_tokens();
    v.num_pages = pool.num_pages();
    v.num_attention_layers = pool.num_attention_layers();
    v.kv_heads = pool.kv_heads();
    v.head_dim = pool.head_dim();
    v.format = pool.octave_format();
    v.k_slot_bytes = ::shelf::kernel::octave_k_slot_bytes(v.format);
    v.v_slot_bytes = ::shelf::kernel::octave_v_slot_bytes(v.format);
    v.k_bytes_per_token = v.kv_heads * v.k_slot_bytes;
    v.v_bytes_per_token = v.kv_heads * v.v_slot_bytes;
    v.k_bytes_per_page = v.page_tokens * v.k_bytes_per_token;
    v.v_bytes_per_page = v.page_tokens * v.v_bytes_per_token;
    v.k_bytes_per_layer = v.num_pages * v.k_bytes_per_page;
    v.v_bytes_per_layer = v.num_pages * v.v_bytes_per_page;
    return v;
}



}
}
