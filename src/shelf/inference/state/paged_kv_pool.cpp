#include <shelf/inference/state/paged_kv_pool.h>

#include <algorithm>
#include <cmath>

namespace shelf {
namespace state {
namespace {

double octave_phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * 3.14159265358979323846);
}

double octave_Phi(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

std::vector<float> make_octave_levels() {
    std::vector<float> table(7u * 64u, 0.0f);
    for (int bits = 3; bits <= 6; ++bits) {
        const int n = 1 << bits;
        std::vector<double> levels(n);
        for (int i = 0; i < n; ++i)
            levels[i] = -2.5 + 5.0 * (static_cast<double>(i) + 0.5) / n;
        for (int it = 0; it < 4000; ++it) {
            std::vector<double> boundaries(n + 1);
            boundaries[0] = -12.0;
            boundaries[n] = 12.0;
            for (int i = 1; i < n; ++i)
                boundaries[i] = 0.5 * (levels[i - 1] + levels[i]);
            for (int i = 0; i < n; ++i) {
                const double mass = octave_Phi(boundaries[i + 1]) - octave_Phi(boundaries[i]);
                if (mass > 1e-300)
                    levels[i] = (octave_phi(boundaries[i]) - octave_phi(boundaries[i + 1])) / mass;
            }
        }
        for (int i = 0; i < n; ++i)
            table[static_cast<std::size_t>(bits) * 64u + i] = static_cast<float>(levels[i]);
    }
    return table;
}

}  // namespace

std::size_t PagedKVPool::page_bytes(
    uint32_t page_tokens,
    uint32_t num_attention_layers,
    uint32_t kv_heads,
    uint32_t head_dim,
    KVCacheDType dtype,
    OctaveKVCacheFormat octave_format) noexcept {
    if (page_tokens == 0u || num_attention_layers == 0u || kv_heads == 0u ||
        head_dim == 0u) {
        return 0u;
    }
    const std::size_t layer_page =
        static_cast<std::size_t>(page_tokens) * kv_heads * head_dim;
    const std::size_t body =
        2u * static_cast<std::size_t>(num_attention_layers) * layer_page;
    if (dtype == KVCacheDType::OCTAVE) {
        if (head_dim != 256u || !::shelf::kernel::octave_format_valid(octave_format))
            return 0u;
        return static_cast<std::size_t>(num_attention_layers) * page_tokens * kv_heads *
               (::shelf::kernel::octave_k_slot_bytes(octave_format) +
                ::shelf::kernel::octave_v_slot_bytes(octave_format)) +
               2u * static_cast<std::size_t>(page_tokens) * kv_heads * head_dim *
                   sizeof(bf16_t);
    }
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
    KVCacheDType dtype,
    OctaveKVCacheFormat octave_format) {
    if (num_pages == 0) return Status::invalid_argument("num_pages must be > 0", __FILE__, __LINE__);
    if (page_tokens == 0) return Status::invalid_argument("page_tokens must be > 0", __FILE__, __LINE__);
    if (num_attention_layers == 0) return Status::invalid_argument("num_attention_layers must be > 0", __FILE__, __LINE__);
    if (kv_heads == 0) return Status::invalid_argument("kv_heads must be > 0", __FILE__, __LINE__);
    if (head_dim == 0) return Status::invalid_argument("head_dim must be > 0", __FILE__, __LINE__);
    if (dtype != KVCacheDType::BF16 && dtype != KVCacheDType::FP8_E4M3 &&
        dtype != KVCacheDType::OCTAVE) {
        return Status::invalid_argument("unknown KV cache dtype", __FILE__, __LINE__);
    }
    if (dtype == KVCacheDType::OCTAVE && head_dim != 256u) {
        return Status::invalid_argument("Octave KV cache requires head_dim 256", __FILE__, __LINE__);
    }
    if (dtype == KVCacheDType::OCTAVE &&
        !::shelf::kernel::octave_format_valid(octave_format)) {
        return Status::invalid_argument("invalid Octave KV cache format", __FILE__, __LINE__);
    }

    PagedKVPool pool;
    pool.num_pages_ = num_pages;
    pool.page_tokens_ = page_tokens;
    pool.num_attention_layers_ = num_attention_layers;
    pool.kv_heads_ = kv_heads;
    pool.head_dim_ = head_dim;
    pool.dtype_ = dtype;
    pool.octave_format_ = octave_format;
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

    if (dtype == KVCacheDType::OCTAVE) {
        const std::size_t slots = static_cast<std::size_t>(num_attention_layers) *
                                  num_pages * page_tokens * kv_heads;
        const std::size_t k_bytes =
            slots * ::shelf::kernel::octave_k_slot_bytes(octave_format);
        const std::size_t v_bytes =
            slots * ::shelf::kernel::octave_v_slot_bytes(octave_format);
        auto k_alloc = arena.allocate_aligned(k_bytes, 256);
        if (!k_alloc.ok()) return k_alloc.status();
        auto k_tensor = gpu::Tensor::create<uint8_t>(k_alloc.release(), {k_bytes}, {1});
        if (!k_tensor.ok()) return k_tensor.status();
        pool.k_pool_ = k_tensor.release();

        auto v_alloc = arena.allocate_aligned(v_bytes, 256);
        if (!v_alloc.ok()) return v_alloc.status();
        auto v_tensor = gpu::Tensor::create<uint8_t>(v_alloc.release(), {v_bytes}, {1});
        if (!v_tensor.ok()) return v_tensor.status();
        pool.v_pool_ = v_tensor.release();
        const std::vector<float> levels = make_octave_levels();
        auto levels_alloc = arena.allocate_aligned(levels.size() * sizeof(float), 256);
        if (!levels_alloc.ok()) return levels_alloc.status();
        auto levels_tensor = gpu::Tensor::create<float>(levels_alloc.release(), {levels.size()}, {1});
        if (!levels_tensor.ok()) return levels_tensor.status();
        pool.octave_levels_ = levels_tensor.release();
        const hipError_t levels_copy = hipMemcpy(
            pool.octave_levels_.data<float>(), levels.data(), levels.size() * sizeof(float),
            hipMemcpyHostToDevice);
        if (levels_copy != hipSuccess) {
            return Status::hip_error("upload Octave Lloyd-Max levels",
                                     hipGetErrorString(levels_copy), __FILE__, __LINE__);
        }
        const std::size_t scratch_elems = static_cast<std::size_t>(num_pages) *
            page_tokens * kv_heads * head_dim;
        auto k_scratch_alloc = arena.allocate_aligned(scratch_elems * sizeof(bf16_t), 256);
        if (!k_scratch_alloc.ok()) return k_scratch_alloc.status();
        auto k_scratch_tensor = gpu::Tensor::create<bf16_t>(
            k_scratch_alloc.release(), {scratch_elems}, {1});
        if (!k_scratch_tensor.ok()) return k_scratch_tensor.status();
        pool.octave_k_scratch_ = k_scratch_tensor.release();
        auto v_scratch_alloc = arena.allocate_aligned(scratch_elems * sizeof(bf16_t), 256);
        if (!v_scratch_alloc.ok()) return v_scratch_alloc.status();
        auto v_scratch_tensor = gpu::Tensor::create<bf16_t>(
            v_scratch_alloc.release(), {scratch_elems}, {1});
        if (!v_scratch_tensor.ok()) return v_scratch_tensor.status();
        pool.octave_v_scratch_ = v_scratch_tensor.release();
        pool.reserved_bytes_ = k_bytes + v_bytes + 2u * scratch_elems * sizeof(bf16_t);
    } else if (dtype == KVCacheDType::FP8_E4M3) {
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
    pool.seq_refs_.assign(num_pages, 0);
    pool.cache_refs_.assign(num_pages, 0);
    pool.page_state_.assign(num_pages, kPageFree);
    pool.free_pos_.assign(num_pages, 0);
    pool.cached_pos_.assign(num_pages, 0);
    for (uint32_t i = 0; i < num_pages; ++i) {
        pool.free_list_.push_back(i);
        pool.free_pos_[i] = i;
    }

    return Result<PagedKVPool>(std::move(pool));
}

void PagedKVPool::free_remove(PageId page) {
    const uint32_t pos = free_pos_[page];
    const PageId last = free_list_.back();
    free_list_[pos] = last;
    free_pos_[last] = pos;
    free_list_.pop_back();
    free_pos_[page] = kInvalidPageId;
}

void PagedKVPool::cached_remove(PageId page) {
    const uint32_t pos = cached_pos_[page];
    const PageId last = cached_list_.back();
    cached_list_[pos] = last;
    cached_pos_[last] = pos;
    cached_list_.pop_back();
    cached_pos_[page] = kInvalidPageId;
}

Result<PageId> PagedKVPool::allocate() {
    if (free_list_.empty() && reclaimer_) {
        (void)reclaimer_(1u);
    }
    if (free_list_.empty()) {
        return Status::insufficient_memory("KV page pool exhausted", __FILE__, __LINE__);
    }

    PageId page = free_list_.back();
    free_list_.pop_back();
    free_pos_[page] = kInvalidPageId;
    page_state_[page] = kPageSeq;
    seq_refs_[page] = 1u;
    in_use_[page] = true;
    ++used_count_;

    return page;
}

Status PagedKVPool::adopt(PageId page) {
    if (page >= num_pages_) {
        return Status::invalid_argument("page id out of range", __FILE__, __LINE__);
    }
    if (page_state_[page] != kPageFree) {
        return Status::invalid_argument("page already in use", __FILE__, __LINE__);
    }
    free_remove(page);
    page_state_[page] = kPageSeq;
    seq_refs_[page] = 1u;
    in_use_[page] = true;
    ++used_count_;
    return Status::make_ok();
}

Status PagedKVPool::attach(PageId page) {
    if (page >= num_pages_) {
        return Status::invalid_argument("page id out of range", __FILE__, __LINE__);
    }
    if (page_state_[page] == kPageSeq) {
        ++seq_refs_[page];
        return Status::make_ok();
    }
    if (page_state_[page] == kPageCached) {
        cached_remove(page);
        --cached_count_;
    } else {
        free_remove(page);
    }
    page_state_[page] = kPageSeq;
    seq_refs_[page] = 1u;
    in_use_[page] = true;
    ++used_count_;
    return Status::make_ok();
}

Status PagedKVPool::retain_cache(PageId page) {
    if (page >= num_pages_) {
        return Status::invalid_argument("page id out of range", __FILE__, __LINE__);
    }
    if (page_state_[page] == kPageFree) {
        free_remove(page);
        page_state_[page] = kPageCached;
        ++cached_count_;
    }
    ++cache_refs_[page];
    return Status::make_ok();
}

Status PagedKVPool::release_cache(PageId page) {
    if (page >= num_pages_) {
        return Status::invalid_argument("page id out of range", __FILE__, __LINE__);
    }
    if (cache_refs_[page] == 0u) {
        return Status::invalid_state("page has no cache reference", __FILE__, __LINE__);
    }
    --cache_refs_[page];
    if (cache_refs_[page] != 0u) return Status::make_ok();
    if (page_state_[page] == kPageCached) {
        cached_remove(page);
        --cached_count_;
        page_state_[page] = kPageFree;
        free_pos_[page] = static_cast<uint32_t>(free_list_.size());
        free_list_.push_back(page);
    }
    return Status::make_ok();
}

Status PagedKVPool::release(PageId page) {
    if (page >= num_pages_) {
        return Status::invalid_argument("page id out of range", __FILE__, __LINE__);
    }
    if (page_state_[page] != kPageSeq || seq_refs_[page] == 0u) {
        return Status::invalid_argument("page not in use", __FILE__, __LINE__);
    }

    --seq_refs_[page];
    if (seq_refs_[page] != 0u) return Status::make_ok();

    in_use_[page] = false;
    --used_count_;
    if (cache_refs_[page] != 0u) {
        page_state_[page] = kPageCached;
        cached_pos_[page] = static_cast<uint32_t>(cached_list_.size());
        cached_list_.push_back(page);
        ++cached_count_;
    } else {
        page_state_[page] = kPageFree;
        free_pos_[page] = static_cast<uint32_t>(free_list_.size());
        free_list_.push_back(page);
    }
    return Status::make_ok();
}

Status PagedKVPool::validate_release_many(const std::vector<PageId>& pages) const {
    for (PageId page : pages) {
        if (page == kInvalidPageId) return Status::invalid_argument("invalid page id", __FILE__, __LINE__);
        if (page >= num_pages_) return Status::invalid_argument("page id out of range", __FILE__, __LINE__);
        if (page_state_[page] != kPageSeq || seq_refs_[page] == 0u) {
            return Status::invalid_argument("page not in use", __FILE__, __LINE__);
        }
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
        (void)release(page);
    }

    return Status::make_ok();
}

gpu::Tensor PagedKVPool::k_view(uint32_t layer, PageId page) const {
    if (layer >= num_attention_layers_) return gpu::Tensor{};
    if (page >= num_pages_) return gpu::Tensor{};

    std::size_t elems_per_page = dtype_ == KVCacheDType::OCTAVE
        ? static_cast<std::size_t>(page_tokens_) * kv_heads_ *
              ::shelf::kernel::octave_k_slot_bytes(octave_format_)
        : static_cast<std::size_t>(page_tokens_) * kv_heads_ * head_dim_;
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

    std::size_t elems_per_page = dtype_ == KVCacheDType::OCTAVE
        ? static_cast<std::size_t>(page_tokens_) * kv_heads_ *
              ::shelf::kernel::octave_v_slot_bytes(octave_format_)
        : static_cast<std::size_t>(page_tokens_) * kv_heads_ * head_dim_;
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
    if (dtype_ == KVCacheDType::OCTAVE && octave_format_ != src.octave_format_) {
        return Status::invalid_argument("KV page copy Octave format mismatch", __FILE__, __LINE__);
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
    const std::size_t octave_k_bytes = static_cast<std::size_t>(page_tokens_) * kv_heads_ *
        ::shelf::kernel::octave_k_slot_bytes(octave_format_);
    const std::size_t octave_v_bytes = static_cast<std::size_t>(page_tokens_) * kv_heads_ *
        ::shelf::kernel::octave_v_slot_bytes(octave_format_);
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

        if (dtype_ == KVCacheDType::OCTAVE) {
            hipError_t err = hipMemcpyAsync(
                dst_k.data<uint8_t>(), src_k.data<uint8_t>(), octave_k_bytes,
                hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error("copy_page_from Octave K", hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                dst_v.data<uint8_t>(), src_v.data<uint8_t>(), octave_v_bytes,
                hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error("copy_page_from Octave V", hipGetErrorString(err), __FILE__, __LINE__);
            }
        } else if (dtype_ == KVCacheDType::FP8_E4M3) {
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
