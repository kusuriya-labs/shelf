// shelf backend/common — Paged KV physical ABI (kernel-visible contract).
//
// This is the layout a paged-attention kernel sees. Porting a kernel must
// be an adapter, never a redesign. (Revision 2: the vendored gfx1201 kernel
// family — paged decode bf16/fp8, split-K reduce, kv_append — addresses
// TWO separate K/V pools; making that the canonical packing removes a
// per-step conversion copy.)
// Physical truth lives HERE; the IR's kv_cache StateDesc stays LOGICAL
// (capacity/head semantics). The mapping logical -> physical is a lowering/
// runtime concern (runtime/state).
//
// Packing, in elements of `dtype`:
//
//  SeparatePools (canonical) — offset WITHIN ONE POOL (caller picks K or V):
//    element(b, s, h, d) = b*block_stride + s*slot_stride + h*head_stride + d
//    canonical: slot_stride = kv_heads*head_dim ("token-major"),
//               head_stride = head_dim, block_stride = block_size*slot_stride
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/foundation/dtype.hpp>
#include <shelf/status.hpp>

namespace shelf::backend::common {

// Tokens per KV block. NOT a core constant: it comes from the resolved
// attention kernel's constraint. 16 is what the current paged kernels are
// compiled for; other values need kernels for them.
enum class KVPacking : std::uint8_t {
    SeparatePools,  // two allocations; kind = which pool pointer
};

struct PagedKVLayout {
    std::size_t block_size = 16;   // tokens per block
    std::size_t kv_heads = 0;      // kvh (GQA)
    std::size_t head_dim = 0;      // D
    std::size_t num_blocks = 0;    // pool capacity in blocks
    DataType dtype = DataType::bf16;
    KVPacking packing = KVPacking::SeparatePools;

    // strides in ELEMENTS (not bytes), semantics per packing (see above)
    std::size_t block_stride = 0;
    std::size_t head_stride = 0;
    std::size_t slot_stride = 0;

    // Validate internal consistency (const) — strides must already be set.
    Status check() const;
    // Fill canonical contiguous strides when all zero, then check().
    Status prepare();

    // 1 プール（K または V）あたりの要素数。
    std::size_t elements_per_block() const noexcept {
        return block_size * kv_heads * head_dim;
    }
};

// Element offset. Kind: 0 = K, 1 = V だが未使用（InterleavedKV 削除後は
// K/V が常に別プールなので、呼び出し側が k_pool / v_pool の base を渡す）。
// 返されるのは、選ばれた 1 プールの先頭からのオフセット。
inline std::size_t paged_kv_offset(const PagedKVLayout& L, std::size_t block,
                                   std::size_t kv_head, std::size_t slot_in_block,
                                   std::size_t kind, std::size_t dim) noexcept {
    (void)kind;
    return block * L.block_stride + slot_in_block * L.slot_stride +
           kv_head * L.head_stride + dim;
}

// Per-batch block table: physical block id per (sequence, logical block).
// int32 rows of width max_blocks, row stride `row_stride` (>= max_blocks).
// -1 marks an unallocated tail (legal: kernels must not touch beyond seqused).
// Kernel-native tables may store uint32 instead: -1 bit-casts to 0xFFFFFFFF,
// which uint32-consuming kernels treat as ">= num_pages" invalid — the two
// conventions are bit-compatible for the valid range.
struct BlockTableView {
    const std::int32_t* ids = nullptr;
    std::size_t num_seqs = 0;
    std::size_t max_blocks = 0;
    std::size_t row_stride = 0;

    std::int32_t at(std::size_t seq, std::size_t blk) const noexcept {
        return ids[seq * row_stride + blk];
    }
    Status validate() const {
        if (!ids) return Status::invalid_argument("block table: null ids", __FILE__, __LINE__);
        if (num_seqs == 0 || max_blocks == 0)
            return Status::invalid_argument("block table: empty dims", __FILE__, __LINE__);
        if (row_stride < max_blocks)
            return Status::invalid_argument("block table: row_stride < max_blocks",
                                            __FILE__, __LINE__);
        return Status::make_ok();
    }
};

// Live token count per sequence (device-side in kernels; host mirror here).
struct SeqUsedView {
    const std::int32_t* used = nullptr;
    std::size_t num_seqs = 0;
};

// fp8-KV dequant descales; null layout for bf16/f32 KV. Carried in the ABI
// now so an fp8 kernel needs no ABI change.
// NOTE: the vendored fp8 kernel uses PerTokenHead granularity
// ([layer][page][off][kvh] inside the pool span); the neutral view keeps
// PerSeqHead for simple runtimes. fp8 KV activation will add the token
// table view at that stage (both fit this struct family).
struct DescaleView {
    const float* k = nullptr;   // [num_seqs, kv_heads] (PerSeqHead)
    const float* v = nullptr;   // [num_seqs, kv_heads] (PerSeqHead)
    std::size_t num_seqs = 0;
    std::size_t kv_heads = 0;
    bool enabled() const noexcept { return k != nullptr && v != nullptr; }
};

// One kernel launch's worth of paged-KV addressing inputs.
struct PagedKVArgs {
    PagedKVLayout layout;
    BlockTableView table;
    SeqUsedView used;
    DescaleView descale;
    float scale = 0.0f;         // attention score scale (0 => 1/sqrt(head_dim))
    std::size_t max_ctx = 0;    // host-visible context bound

    Status validate() const {
        SHELF_RETURN_IF_ERROR(layout.check());
        SHELF_RETURN_IF_ERROR(table.validate());
        if (!used.used || used.num_seqs != table.num_seqs)
            return Status::invalid_argument("paged-kv: seqused mismatch", __FILE__, __LINE__);
        if (descale.enabled() &&
            (descale.num_seqs != table.num_seqs || descale.kv_heads != layout.kv_heads))
            return Status::invalid_argument("paged-kv: descale dims mismatch",
                                            __FILE__, __LINE__);
        for (std::size_t s = 0; s < table.num_seqs; ++s) {
            const std::size_t blocks_needed =
                (static_cast<std::size_t>(used.used[s]) + layout.block_size - 1) /
                layout.block_size;
            if (blocks_needed > table.max_blocks)
                return Status::out_of_range("paged-kv: seqused exceeds table width",
                                            __FILE__, __LINE__);
            for (std::size_t b = 0; b < blocks_needed; ++b) {
                if (table.at(s, b) < 0 ||
                    static_cast<std::size_t>(table.at(s, b)) >= layout.num_blocks) {
                    return Status::out_of_range("paged-kv: block id outside pool",
                                                __FILE__, __LINE__);
                }
            }
        }
        return Status::make_ok();
    }
};

} // namespace shelf::backend::common
