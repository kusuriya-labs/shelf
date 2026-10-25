// shelf backend/hip — GPU step executor for the single-token Model Graph.
//
// Executes one decode step of an ir::Graph on the device:
//   - weights pre-uploaded as bf16 (raw checkpoint bits),
//   - activations bf16 in a bump arena (per-step reset),
//   - f32 where the kernels demand it (GDN prepared q/k, logits),
//   - states device-resident (KV / conv / recurrent), usage rows persist.
//
// Neutral by contract: sees ir:: + foundation:: + its own kernels only
// (no model includes; model name scan applies). The composition root
// (apps/) wires model plans into this executor.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <shelf/backend/hip/kernels/gdn_types.h>
#include <shelf/foundation/result.hpp>
#include <shelf/foundation/status.hpp>
#include <shelf/ir/graph.hpp>

namespace shelf::backend::hip {

struct HipStepExecutorOptions {
    int device = 0;
    std::size_t arena_bytes = 64u << 20;   // activations per step
};

class HipStepExecutor {
public:
    explicit HipStepExecutor(HipStepExecutorOptions opts = {});
    ~HipStepExecutor();

    // Weight upload (host f32 -> device bf16, RNE). For states that keep
    // f32 parameters (A_log/dt_bias) this stays bf16 on device because the
    // GDN kernel takes them as bf16 [nvh] vectors.
    Status upload_weight(ir::TensorId id, const std::vector<float>& f32);

    // State allocation from the graph's StateDesc (zero-initialized).
    Status declare_states(const ir::Graph& g);
    // Zero the value/state buffers between independent runs (KV rows reset).
    Status reset_states(const ir::Graph& g);

    // One decode step: token id + rope cos/sin (host f32[rotary_dim]).
    // Produces logits in device memory (fetch with fetch_logits()).
    Status run_step(const ir::Graph& g, std::int64_t token,
                    const std::vector<float>& cos, const std::vector<float>& sin);

    Result<std::vector<float>> fetch_logits(const ir::Graph& g);

    struct Stats { double last_step_ms = 0.0; };
    const Stats& stats() const { return stats_; }

private:
    struct ValueLoc { void* dev = nullptr; bool f32 = false; std::size_t elems = 0; };

    Status run_primitive(const ir::Graph& g, const ir::Primitive& p);

    void* arena_alloc(std::size_t bytes);
    void arena_reset() { arena_top_ = 0; }

    HipStepExecutorOptions opts_;
    hipStream_t stream_ = nullptr;

    std::vector<char> arena_host_;      // staging; actual arena is device
    void* arena_dev_ = nullptr;
    std::size_t arena_cap_ = 0, arena_top_ = 0;

    std::unordered_map<std::uint32_t, void*> weight_dev_;       // TensorId -> bf16*
    struct StateLoc { void* dev = nullptr; std::size_t rows = 0; std::size_t elems = 0; };
    std::unordered_map<std::uint32_t, StateLoc> state_dev_;     // StateId -> loc
    std::unordered_map<std::uint32_t, ValueLoc> value_dev_;     // ValueId -> loc

    // per-run scratch that must survive across primitives
    void* token_dev_ = nullptr;      // int32[1] input
    void* cos_dev_ = nullptr;        // f32[rd]
    void* sin_dev_ = nullptr;        // f32[rd]
    GdnRequestDescriptor* req_dev_ = nullptr;   // single seq: {slot0,row0,n}
    float* logits_dev_ = nullptr;    // f32[vocab]
    ir::ValueId logits_value_;

    Stats stats_;
};

} // namespace shelf::backend::hip
