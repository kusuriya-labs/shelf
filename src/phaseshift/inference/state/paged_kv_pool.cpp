#include <phaseshift/inference/state/paged_kv_pool.h>

namespace ps {
namespace state {

std::size_t PagedKVPool::page_bytes(
    uint32_t page_tokens,
    uint32_t num_attention_layers,
    uint32_t kv_heads,
    uint32_t head_dim,
    KVCacheDType dtype) noexcept {
    if (page_tokens == 0u || num_attention_layers == 0u || kv_heads == 0u ||
        head_dim == 0u) {
        return 0u;
    }
    const std::size_t layer_page =
        static_cast<std::size_t>(page_tokens) * kv_heads * head_dim;
    const std::size_t body =
        2u * static_cast<std::size_t>(num_attention_layers) * layer_page;
    if (dtype == KVCacheDType::FP8_E4M3) {
        const std::size_t scales =
            2u * static_cast<std::size_t>(num_attention_layers) *
            static_cast<std::size_t>(page_tokens) * kv_heads;
        return body * sizeof(fp8e4m3_storage_t) + scales * sizeof(float);
    }
    return body * sizeof(bf16_t);
}

Result<PagedKVPool> PagedKVPool::create(
    gpu::GpuArena& arena,
    uint32_t num_pages,
    uint32_t page_tokens,
    uint32_t kv_heads,
    uint32_t head_dim) {
    return create(arena, num_pages, page_tokens, 6, kv_heads, head_dim,
                  KVCacheDType::BF16);
}

Result<PagedKVPool> PagedKVPool::create(
    gpu::GpuArena& arena,
    uint32_t num_pages,
    uint32_t page_tokens,
    uint32_t num_attention_layers,
    uint32_t kv_heads,
    uint32_t head_dim) {
    return create(arena, num_pages, page_tokens, num_attention_layers,
                  kv_heads, head_dim, KVCacheDType::BF16);
}

Result<PagedKVPool> PagedKVPool::create(
    gpu::GpuArena& arena,
    uint32_t num_pages,
    uint32_t page_tokens,
    uint32_t num_attention_layers,
    uint32_t kv_heads,
    uint32_t head_dim,
    KVCacheDType dtype) {
    if (num_pages == 0) return Status::invalid_argument("num_pages must be > 0", __FILE__, __LINE__);
    if (page_tokens == 0) return Status::invalid_argument("page_tokens must be > 0", __FILE__, __LINE__);
    if (num_attention_layers == 0) return Status::invalid_argument("num_attention_layers must be > 0", __FILE__, __LINE__);
    if (kv_heads == 0) return Status::invalid_argument("kv_heads must be > 0", __FILE__, __LINE__);
    if (head_dim == 0) return Status::invalid_argument("head_dim must be > 0", __FILE__, __LINE__);
    if (dtype != KVCacheDType::BF16 && dtype != KVCacheDType::FP8_E4M3) {
        return Status::invalid_argument("unknown KV cache dtype", __FILE__, __LINE__);
    }

    PagedKVPool pool;
    pool.num_pages_ = num_pages;
    pool.page_tokens_ = page_tokens;
    pool.num_attention_layers_ = num_attention_layers;
    pool.kv_heads_ = kv_heads;
    pool.head_dim_ = head_dim;
    pool.dtype_ = dtype;
    pool.used_count_ = 0;

    const std::size_t pool_elems =
        static_cast<std::size_t>(num_attention_layers)
        * static_cast<std::size_t>(num_pages)
        * static_cast<std::size_t>(page_tokens)
        * static_cast<std::size_t>(kv_heads)
        * static_cast<std::size_t>(head_dim);

    const std::size_t scale_elems =
        static_cast<std::size_t>(num_attention_layers)
        * static_cast<std::size_t>(num_pages)
        * static_cast<std::size_t>(page_tokens)
        * static_cast<std::size_t>(kv_heads);

    if (dtype == KVCacheDType::FP8_E4M3) {
        auto k_alloc = arena.allocate_aligned(pool_elems * sizeof(fp8e4m3_storage_t), 256);
        if (!k_alloc.ok()) return k_alloc.status();
        auto k_tensor = gpu::Tensor::create<fp8e4m3_storage_t>(k_alloc.release(), {pool_elems}, {1});
        if (!k_tensor.ok()) return k_tensor.status();
        pool.k_pool_ = k_tensor.release();

        auto v_alloc = arena.allocate_aligned(pool_elems * sizeof(fp8e4m3_storage_t), 256);
        if (!v_alloc.ok()) return v_alloc.status();
        auto v_tensor = gpu::Tensor::create<fp8e4m3_storage_t>(v_alloc.release(), {pool_elems}, {1});
        if (!v_tensor.ok()) return v_tensor.status();
        pool.v_pool_ = v_tensor.release();

        auto ks_alloc = arena.allocate_aligned(scale_elems * sizeof(float), 256);
        if (!ks_alloc.ok()) return ks_alloc.status();
        auto ks_tensor = gpu::Tensor::create<float>(ks_alloc.release(), {scale_elems}, {1});
        if (!ks_tensor.ok()) return ks_tensor.status();
        pool.k_scale_pool_ = ks_tensor.release();

        auto vs_alloc = arena.allocate_aligned(scale_elems * sizeof(float), 256);
        if (!vs_alloc.ok()) return vs_alloc.status();
        auto vs_tensor = gpu::Tensor::create<float>(vs_alloc.release(), {scale_elems}, {1});
        if (!vs_tensor.ok()) return vs_tensor.status();
        pool.v_scale_pool_ = vs_tensor.release();

        pool.reserved_bytes_ =
            2 * pool_elems * sizeof(fp8e4m3_storage_t)
            + 2 * scale_elems * sizeof(float);
    } else {
        auto k_alloc = arena.allocate_aligned(pool_elems * sizeof(bf16_t), 256);
        if (!k_alloc.ok()) return k_alloc.status();
        auto k_tensor = gpu::Tensor::create<bf16_t>(k_alloc.release(), {pool_elems}, {1});
        if (!k_tensor.ok()) return k_tensor.status();
        pool.k_pool_ = k_tensor.release();

        auto v_alloc = arena.allocate_aligned(pool_elems * sizeof(bf16_t), 256);
        if (!v_alloc.ok()) return v_alloc.status();
        auto v_tensor = gpu::Tensor::create<bf16_t>(v_alloc.release(), {pool_elems}, {1});
        if (!v_tensor.ok()) return v_tensor.status();
        pool.v_pool_ = v_tensor.release();

        pool.reserved_bytes_ = 2 * pool_elems * sizeof(bf16_t);
    }

    pool.in_use_.assign(num_pages, false);
    pool.free_list_.reserve(num_pages);
    for (uint32_t i = 0; i < num_pages; ++i) {
        pool.free_list_.push_back(i);
    }

    return Result<PagedKVPool>(std::move(pool));
}

Result<PageId> PagedKVPool::allocate() {
    if (free_list_.empty()) return Status::insufficient_memory("KV page pool exhausted", __FILE__, __LINE__);

    PageId page = free_list_.back();
    free_list_.pop_back();
    in_use_[page] = true;
    ++used_count_;

    return page;
}

Status PagedKVPool::release(PageId page) {
    if (page >= num_pages_) return Status::invalid_argument("page id out of range", __FILE__, __LINE__);
    if (!in_use_[page]) return Status::invalid_argument("page not in use", __FILE__, __LINE__);

    in_use_[page] = false;
    free_list_.push_back(page);
    --used_count_;
    return Status::make_ok();
}

Status PagedKVPool::validate_release_many(const std::vector<PageId>& pages) const {
    for (PageId page : pages) {
        if (page == kInvalidPageId) return Status::invalid_argument("invalid page id", __FILE__, __LINE__);
        if (page >= num_pages_) return Status::invalid_argument("page id out of range", __FILE__, __LINE__);
        if (!in_use_[page]) return Status::invalid_argument("page not in use", __FILE__, __LINE__);
    }

    for (std::size_t i = 0; i < pages.size(); ++i) {
        for (std::size_t j = i + 1; j < pages.size(); ++j) {
            if (pages[i] == pages[j]) return Status::invalid_argument("duplicate page id", __FILE__, __LINE__);
        }
    }

    return Status::make_ok();
}

Status PagedKVPool::release_many(const std::vector<PageId>& pages) {
    auto validate_st = validate_release_many(pages);
    if (!validate_st.ok()) return validate_st;

    for (PageId page : pages) {
        in_use_[page] = false;
        free_list_.push_back(page);
        --used_count_;
    }

    return Status::make_ok();
}

gpu::Tensor PagedKVPool::k_view(uint32_t layer, PageId page) const {
    if (layer >= num_attention_layers_) return gpu::Tensor{};
    if (page >= num_pages_) return gpu::Tensor{};

    std::size_t elems_per_page = static_cast<std::size_t>(page_tokens_) * kv_heads_ * head_dim_;
    std::size_t elems_per_layer = static_cast<std::size_t>(num_pages_) * elems_per_page;

    std::size_t layer_offset = static_cast<std::size_t>(layer) * elems_per_layer;
    std::size_t page_offset = layer_offset + static_cast<std::size_t>(page) * elems_per_page;

    std::size_t shape0 = k_pool_.dim(0);
    std::size_t layer_size = shape0 / num_attention_layers_;

    auto layer_slice = k_pool_.slice(0, layer_offset, layer_size);
    if (!layer_slice.ok()) return gpu::Tensor{};

    auto page_slice = layer_slice.release().slice(0, page_offset - layer_offset, elems_per_page);
    if (!page_slice.ok()) return gpu::Tensor{};

    return page_slice.release();
}

gpu::Tensor PagedKVPool::v_view(uint32_t layer, PageId page) const {
    if (layer >= num_attention_layers_) return gpu::Tensor{};
    if (page >= num_pages_) return gpu::Tensor{};

    std::size_t elems_per_page = static_cast<std::size_t>(page_tokens_) * kv_heads_ * head_dim_;
    std::size_t elems_per_layer = static_cast<std::size_t>(num_pages_) * elems_per_page;

    std::size_t layer_offset = static_cast<std::size_t>(layer) * elems_per_layer;
    std::size_t page_offset = layer_offset + static_cast<std::size_t>(page) * elems_per_page;

    std::size_t shape0 = v_pool_.dim(0);
    std::size_t layer_size = shape0 / num_attention_layers_;

    auto layer_slice = v_pool_.slice(0, layer_offset, layer_size);
    if (!layer_slice.ok()) return gpu::Tensor{};

    auto page_slice = layer_slice.release().slice(0, page_offset - layer_offset, elems_per_page);
    if (!page_slice.ok()) return gpu::Tensor{};

    return page_slice.release();
}

Status PagedKVPool::copy_page_from(
    PageId dst_page,
    const PagedKVPool& src,
    PageId src_page,
    hipStream_t stream) {
    if (dtype_ != src.dtype_) {
        return Status::invalid_argument("KV page copy dtype mismatch", __FILE__, __LINE__);
    }
    if (page_tokens_ != src.page_tokens_ ||
        num_attention_layers_ != src.num_attention_layers_ ||
        kv_heads_ != src.kv_heads_ ||
        head_dim_ != src.head_dim_) {
        return Status::invalid_argument("KV page copy layout mismatch", __FILE__, __LINE__);
    }
    if (dst_page >= num_pages_) {
        return Status::invalid_argument("KV page copy destination out of range", __FILE__, __LINE__);
    }
    if (src_page >= src.num_pages_) {
        return Status::invalid_argument("KV page copy source out of range", __FILE__, __LINE__);
    }

    const std::size_t elems_per_page =
        static_cast<std::size_t>(page_tokens_) * kv_heads_ * head_dim_;
    const std::size_t kv_bytes = dtype_ == KVCacheDType::FP8_E4M3
        ? elems_per_page * sizeof(fp8e4m3_storage_t)
        : elems_per_page * sizeof(bf16_t);
    const std::size_t scale_elems_per_page =
        static_cast<std::size_t>(page_tokens_) * kv_heads_;
    const std::size_t scale_bytes = scale_elems_per_page * sizeof(float);

    for (uint32_t layer = 0; layer < num_attention_layers_; ++layer) {
        const gpu::Tensor src_k = src.k_view(layer, src_page);
        const gpu::Tensor src_v = src.v_view(layer, src_page);
        const gpu::Tensor dst_k = k_view(layer, dst_page);
        const gpu::Tensor dst_v = v_view(layer, dst_page);
        if (src_k.ndim() == 0 || src_v.ndim() == 0 || dst_k.ndim() == 0 || dst_v.ndim() == 0) {
            return Status::invalid_state("KV page copy view unavailable", __FILE__, __LINE__);
        }

        if (dtype_ == KVCacheDType::FP8_E4M3) {
            hipError_t err = hipMemcpyAsync(
                dst_k.data<fp8e4m3_storage_t>(), src_k.data<fp8e4m3_storage_t>(),
                kv_bytes, hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error("copy_page_from K", hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                dst_v.data<fp8e4m3_storage_t>(), src_v.data<fp8e4m3_storage_t>(),
                kv_bytes, hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error("copy_page_from V", hipGetErrorString(err), __FILE__, __LINE__);
            }

            const std::size_t dst_scale_offset =
                (static_cast<std::size_t>(layer) * num_pages_ + dst_page) * scale_elems_per_page;
            const std::size_t src_scale_offset =
                (static_cast<std::size_t>(layer) * src.num_pages_ + src_page) * scale_elems_per_page;
            err = hipMemcpyAsync(
                k_scale_pool_.data<float>() + dst_scale_offset,
                src.k_scale_pool_.data<float>() + src_scale_offset,
                scale_bytes, hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error("copy_page_from K scale", hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                v_scale_pool_.data<float>() + dst_scale_offset,
                src.v_scale_pool_.data<float>() + src_scale_offset,
                scale_bytes, hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error("copy_page_from V scale", hipGetErrorString(err), __FILE__, __LINE__);
            }
        } else {
            hipError_t err = hipMemcpyAsync(
                dst_k.data<bf16_t>(), src_k.data<bf16_t>(),
                kv_bytes, hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error("copy_page_from K", hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                dst_v.data<bf16_t>(), src_v.data<bf16_t>(),
                kv_bytes, hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error("copy_page_from V", hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
    }

    return Status::make_ok();
}

}
}
