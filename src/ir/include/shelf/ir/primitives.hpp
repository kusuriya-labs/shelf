// shelf ir — Primitive operations contract (plan §4.3, Phase 4).
//
// Operations carry MATHEMATICAL meaning only: shapes, dtype expectations,
// state read/write and attributes. No kernel selection, no devices, no
// model names (enforced by tests/architecture).
#pragma once

#include <cstdint>
#include <vector>

#include <shelf/foundation/dtype.hpp>
#include <shelf/foundation/shape.hpp>
#include <shelf/foundation/tensor_id.hpp>
#include <shelf/status.hpp>

namespace shelf::ir {

// ---- identity --------------------------------------------------------------

class ValueId {  // SSA-like activation value produced by exactly one primitive
public:
    static constexpr std::uint32_t kInvalidValue = 0xFFFFFFFEu;
    constexpr ValueId() = default;
    static constexpr ValueId from_uint32(std::uint32_t v) { ValueId id; id.value_ = v; return id; }
    std::uint32_t value() const noexcept { return value_; }
    bool valid() const noexcept { return value_ != kInvalidValue; }
    auto operator<=>(const ValueId&) const = default;
private:
    std::uint32_t value_ = kInvalidValue;
};

class StateId {  // session-persistent state slot (KV cache, recurrent)
public:
    static constexpr std::uint32_t kInvalidValue = 0xFFFFFFFEu;
    constexpr StateId() = default;
    static constexpr StateId from_uint32(std::uint32_t v) { StateId id; id.value_ = v; return id; }
    std::uint32_t value() const noexcept { return value_; }
    bool valid() const noexcept { return value_ != kInvalidValue; }
    auto operator<=>(const StateId&) const = default;
private:
    std::uint32_t value_ = kInvalidValue;
};

// ---- primitive kinds (Phase 4 candidate set) --------------------------------

enum class Kind : std::uint8_t {
    embedding,          // int tokens -> rows of weight table
    rmsnorm,            // x * rsqrt(mean(x^2)+eps) * gamma
    linear,             // out[m,n] = sum_k in[m,k] * w[n,k]   (weights [N,K])
    silu,               // x * sigmoid(x)
    elem_mul,           // a * b (same shape)
    residual_add,       // a + b (same shape)
    rope,               // rotary(x, cos, sin) over rotary_dim (0 = full)
    attention_decode,   // single query vs KV state; appends new K/V row
    moe_route,          // expert logits -> softmax probs [E]
    moe_merge,          // sum_e p_e * expert_out[e] -> [H]
    output_projection,  // linear to vocabulary logits (same math as linear)
};

inline const char* kind_name(Kind kind) noexcept {
    switch (kind) {
        case Kind::embedding: return "embedding";
        case Kind::rmsnorm: return "rmsnorm";
        case Kind::linear: return "linear";
        case Kind::silu: return "silu";
        case Kind::elem_mul: return "elem_mul";
        case Kind::residual_add: return "residual_add";
        case Kind::rope: return "rope";
        case Kind::attention_decode: return "attention_decode";
        case Kind::moe_route: return "moe_route";
        case Kind::moe_merge: return "moe_merge";
        case Kind::output_projection: return "output_projection";
        default: return "unknown";
    }
}

// Attributes: numeric parameters of the operation semantics. NEVER a global
// mutable config (plan §4.3: gate functions etc. must be per-primitive).
struct Attr {
    float epsilon = 0.0f;      // rmsnorm
    std::int64_t rotary_dim = 0;  // rope (0 = full width)
    std::int64_t top_k = 0;    // moe (routing contract only; see merge)
    float score_scale = 0.0f;  // attention score scale (0 => 1/sqrt(head_dim))
};

enum class StateKind : std::uint8_t {
    kv_cache = 0,
    recurrent,
};

struct StateDesc {
    StateId id;
    StateKind kind = StateKind::kv_cache;
    TensorShape logical_shape;      // kv: [capacity, head_dim]
    DataType dtype = DataType::f32;
};

// Activation value contract: shape + dtype (reference dtype is f32 for
// floats; embedding tokens are i32).
struct ValueDesc {
    ValueId id;
    TensorShape shape;
    DataType dtype = DataType::f32;
};

// One primitive node. Weights are logical TensorIds resolved through
// weights/ binding — the IR never sees containers or quantized payloads.
struct Primitive {
    Kind kind = Kind::linear;
    Attr attr;
    std::vector<ValueId> inputs;      // activation inputs
    std::vector<TensorId> weights;    // logical weight references
    std::vector<StateId> state_read;  // state inputs
    std::vector<StateId> state_write; // state outputs
    ValueDesc output;
};

} // namespace shelf::ir
