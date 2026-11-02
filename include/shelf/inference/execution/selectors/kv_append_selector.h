#pragma once
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/kernels/kv_types.h>
#include <cstdint>

namespace shelf::execution {

enum class KvAppendImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct KvAppendSelectorInput {
    ::shelf::kernel::KVCacheDType kv_dtype = ::shelf::kernel::KVCacheDType::BF16;
    uint32_t rows = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t page_tokens = 0;
};

KvAppendImplementation select_kv_append_implementation(const KvAppendSelectorInput& input);

}  // namespace shelf::execution
