#pragma once
#include <cstdint>

namespace shelf::execution {

enum class EmbeddingStorage : uint8_t {
    Bf16 = 0,
};

enum class EmbeddingImplementation : uint8_t {
    Correctness = 0,
    Optimized = 1,
};

struct EmbeddingSelectorInput {
    EmbeddingStorage storage = EmbeddingStorage::Bf16;
    uint32_t rows = 0;
    uint32_t hidden_size = 0;
};

struct EmbeddingRule {
    EmbeddingStorage storage;
    uint32_t hidden_size;
    uint32_t min_rows;
    uint32_t max_rows;
};

EmbeddingImplementation select_embedding_implementation(const EmbeddingSelectorInput& in);

}  // namespace shelf::execution
