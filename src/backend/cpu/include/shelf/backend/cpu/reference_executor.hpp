// shelf backend/cpu — CPU reference executor (plan Phase 4).
//
// Correctness-first naive implementations in f32. This is the golden
// reference every fast kernel (CPU SIMD / HIP) must match within the
// defined tolerance. No performance ambition here, no model names.
#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include <shelf/ir/graph.hpp>
#include <shelf/status.hpp>

namespace shelf::backend::cpu {

class HostStorage {
public:
    // float payloads
    Status set_value(ir::ValueId id, std::vector<float> data) {
        return set_float(values_, id.value(), std::move(data), "value");
    }
    Status set_weight(TensorId id, std::vector<float> data) {
        return set_float(weights_, id.value(), std::move(data), "weight");
    }
    Status init_state(ir::StateId id, std::vector<float> data) {
        return set_float(states_, id.value(), std::move(data), "state");
    }
    // int payloads (embedding token ids)
    Status set_value_int(ir::ValueId id, std::vector<std::int32_t> data) {
        auto& slot = values_[id.value()];
        if (slot.set) return Status::invalid_state("int value overwritten", __FILE__, __LINE__);
        slot.ints = std::move(data);
        slot.is_int = true;
        slot.set = true;
        return Status::make_ok();
    }

    const std::vector<float>* value(ir::ValueId id) const { return get_float(values_, id.value()); }
    const std::vector<std::int32_t>* value_int(ir::ValueId id) const {
        auto it = values_.find(id.value());
        return (it == values_.end() || !it->second.is_int) ? nullptr : &it->second.ints;
    }
    const std::vector<float>* weight(TensorId id) const { return get_float(weights_, id.value()); }
    std::vector<float>* state(ir::StateId id) {
        auto it = states_.find(id.value());
        return it == states_.end() ? nullptr : &it->second.floats;
    }
    const std::vector<float>* state_const(ir::StateId id) const { return get_float(states_, id.value()); }

    // Step loop support: activations are per-step, weights and state are not.
    void reset_values() { values_.clear(); }
    // New sequence: zero every state buffer and drop tracked usage rows.
    // (HipStepExecutor::reset_states() と対になる CPU 側の入口)
    void reset_states();
    // Usage rows tracked by the executor for stateful primitives (KV/GDN).
    std::size_t state_rows(ir::StateId id) const {
        auto it = rows_.find(id.value());
        return it == rows_.end() ? 0 : it->second;
    }

private:
    struct Slot {
        std::vector<float> floats;
        std::vector<std::int32_t> ints;
        bool is_int = false;
        bool set = false;
    };
    static Status set_float(std::map<std::uint32_t, Slot>& m, std::uint32_t key,
                            std::vector<float> data, const char* what);
    static const std::vector<float>* get_float(const std::map<std::uint32_t, Slot>& m, std::uint32_t key) {
        auto it = m.find(key);
        return (it == m.end() || !it->second.set || it->second.is_int) ? nullptr : &it->second.floats;
    }

    std::map<std::uint32_t, Slot> values_;
    std::map<std::uint32_t, Slot> weights_;
    std::map<std::uint32_t, Slot> states_;
    std::map<std::uint32_t, std::size_t> rows_;  // StateId -> usage rows (persists across run()s)
    friend class ReferenceExecutor;
};

class ReferenceExecutor {
public:
    // Executes primitives in graph order. Every value/weight/state size is
    // validated against the graph declaration before use. KV usage rows are
    // tracked explicitly (no content heuristics).
    static Status run(const ir::Graph& graph, HostStorage& storage);

private:
    static Status run_primitive(const ir::Graph& graph, const ir::Primitive& p, HostStorage& storage);
};

} // namespace shelf::backend::cpu
