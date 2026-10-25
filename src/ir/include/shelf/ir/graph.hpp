// shelf ir — Graph: ordered primitives with SSA validation (§4.3/§4.4 boundary).
//
// Lowering emits primitives in execution order. The Graph validates:
//   * values defined exactly once, used only after definition
//   * per-kind signature: input/weight/state counts AND shape algebra
//   * state slots declared before use; read/write consistency
#pragma once

#include <cstdint>
#include <vector>

#include <shelf/ir/primitives.hpp>

namespace shelf::ir {

class Graph {
public:
    // External activation inputs (token ids, RoPE tables, new K/V rows, ...).
    Status declare_input(ValueDesc desc) {
        if (!desc.id.valid()) return bad("declare_input: invalid ValueId");
        if (desc.shape.empty()) return bad("declare_input: empty shape");
        if (find_value(desc.id) != nullptr) return bad("declare_input: duplicate ValueId");
        inputs_.push_back(desc);
        return Status::make_ok();
    }

    Status declare_weight(TensorId id, TensorShape shape) {
        if (!id.valid()) return bad("declare_weight: invalid TensorId");
        if (shape.empty()) return bad("declare_weight: empty shape");
        for (const auto& w : weights_) {
            if (w.id == id) return bad("declare_weight: duplicate TensorId");
        }
        weights_.push_back(WeightDecl{id, shape});
        return Status::make_ok();
    }

    Status declare_state(StateDesc desc) {
        if (!desc.id.valid()) return bad("declare_state: invalid StateId");
        if (desc.logical_shape.empty()) return bad("declare_state: empty shape");
        for (const auto& s : states_) {
            if (s.id == desc.id) return bad("declare_state: duplicate StateId");
        }
        states_.push_back(desc);
        return Status::make_ok();
    }

    Status add_primitive(Primitive p) {
        if (p.output.id.valid() && find_value(p.output.id) != nullptr) {
            return bad(std::string("duplicate output ValueId in primitive ") + kind_name(p.kind));
        }
        for (const ValueId in : p.inputs) {
            if (find_value(in) == nullptr) {
                return bad(std::string("undefined input ValueId in primitive ") + kind_name(p.kind));
            }
        }
        for (const StateId s : p.state_read) {
            if (find_state(s) == nullptr) return bad("undeclared state read");
        }
        for (const StateId s : p.state_write) {
            if (find_state(s) == nullptr) return bad("undeclared state write");
        }
        SHELF_RETURN_IF_ERROR(validate_signature(p));
        values_.push_back(p.output);
        primitives_.push_back(std::move(p));
        return Status::make_ok();
    }

    const std::vector<Primitive>& primitives() const noexcept { return primitives_; }
    std::size_t size() const noexcept { return primitives_.size(); }

    const ValueDesc* find_value(ValueId id) const {
        for (const auto& v : values_) if (v.id == id) return &v;
        for (const auto& v : inputs_) if (v.id == id) return &v;
        return nullptr;
    }
    const StateDesc* find_state(StateId id) const {
        for (const auto& s : states_) if (s.id == id) return &s;
        return nullptr;
    }
    const TensorShape* find_weight(TensorId id) const {
        for (const auto& w : weights_) if (w.id == id) return &w.shape;
        return nullptr;
    }

    Status validate() const {
        for (const auto& p : primitives_) {
            SHELF_RETURN_IF_ERROR(validate_signature(p));
        }
        return Status::make_ok();
    }

private:
    struct WeightDecl { TensorId id; TensorShape shape; };

    static Status bad(const std::string& msg) {
        return Status::invalid_argument("ir/graph: " + msg, __FILE__, __LINE__);
    }

    static bool same_shape(const TensorShape& a, const TensorShape& b) {
        if (a.rank() != b.rank()) return false;
        for (std::size_t i = 0; i < a.rank(); ++i) {
            if (a.dim(i) != b.dim(i)) return false;
        }
        return true;
    }

    Status validate_signature(const Primitive& p) const {
        auto in = [&](std::size_t i) -> const ValueDesc* {
            return i < p.inputs.size() ? find_value(p.inputs[i]) : nullptr;
        };
        auto wshape = [&](std::size_t i) -> const TensorShape* {
            return i < p.weights.size() ? find_weight(p.weights[i]) : nullptr;
        };
        const ValueDesc* out = find_value(p.output.id);  // may be null when adding

        switch (p.kind) {
            case Kind::embedding: {
                if (p.inputs.size() != 1 || p.weights.size() != 1) return bad("embedding arity");
                const ValueDesc* t = in(0);
                const TensorShape* tbl = wshape(0);
                if (!t || t->dtype != DataType::i32 || t->shape.rank() != 1) return bad("embedding tokens i32 [T]");
                if (!tbl || tbl->rank() != 2) return bad("embedding weight table [vocab,H]");
                const std::size_t H = tbl->dim(1);
                return expect_out(p, TensorShape::create({t->shape.dim(0), H}).release(), out);
            }
            case Kind::rmsnorm: {
                if (p.inputs.size() != 1 || p.weights.size() != 1) return bad("rmsnorm arity");
                const ValueDesc* x = in(0);
                const TensorShape* g = wshape(0);
                if (!x || x->shape.rank() != 2) return bad("rmsnorm input [M,H]");
                if (!g || g->rank() != 1 || g->dim(0) != x->shape.dim(1)) return bad("rmsnorm gamma [H]");
                return expect_out(p, x->shape, out);
            }
            case Kind::linear:
            case Kind::output_projection: {
                if (p.inputs.size() != 1 || p.weights.size() != 1) return bad("linear arity");
                const ValueDesc* x = in(0);
                const TensorShape* W = wshape(0);
                if (!x || x->shape.rank() != 2) return bad("linear input [M,K]");
                if (!W || W->rank() != 2) return bad("linear weight [N,K]");
                if (W->dim(1) != x->shape.dim(1)) return bad("linear K mismatch (weights [N,K])");
                return expect_out(p, TensorShape::create({x->shape.dim(0), W->dim(0)}).release(), out);
            }
            case Kind::silu: {
                if (p.inputs.size() != 1 || !p.weights.empty()) return bad("silu arity");
                const ValueDesc* x = in(0);
                if (!x) return bad("silu input");
                return expect_out(p, x->shape, out);
            }
            case Kind::elem_mul:
            case Kind::residual_add: {
                if (p.inputs.size() != 2 || !p.weights.empty()) return bad("binary arity");
                const ValueDesc* a = in(0);
                const ValueDesc* b = in(1);
                if (!a || !b || !same_shape(a->shape, b->shape)) return bad("binary needs identical shapes");
                return expect_out(p, a->shape, out);
            }
            case Kind::rope: {
                if (p.inputs.size() != 3 || !p.weights.empty()) return bad("rope arity (x, cos, sin)");
                const ValueDesc* x = in(0);
                const ValueDesc* c = in(1);
                const ValueDesc* sn = in(2);
                if (!x || !c || !sn) return bad("rope inputs");
                if (!same_shape(x->shape, c->shape) || !same_shape(x->shape, sn->shape)) {
                    return bad("rope cos/sin shape must match x");
                }
                if (x->shape.rank() != 2) return bad("rope x [H] as [1,H]");
                const std::size_t rd = p.attr.rotary_dim == 0 ? x->shape.dim(1)
                                                              : static_cast<std::size_t>(p.attr.rotary_dim);
                if (rd == 0 || rd % 2 != 0 || rd > x->shape.dim(1)) return bad("rope rotary_dim invalid");
                return expect_out(p, x->shape, out);
            }
            case Kind::attention_decode: {
                if (p.inputs.size() != 3 || !p.weights.empty()) return bad("attention_decode arity (q,k,v)");
                if (p.state_read.size() != 1 || p.state_write.size() != 1 ||
                    p.state_read[0] != p.state_write[0]) {
                    return bad("attention_decode needs one KV state read+write (append)");
                }
                const StateDesc* kv = find_state(p.state_read[0]);
                if (!kv || kv->kind != StateKind::kv_cache || kv->logical_shape.rank() != 2) {
                    return bad("KV state [capacity, head_dim]");
                }
                const std::size_t D = kv->logical_shape.dim(1);
                const ValueDesc* q = in(0);
                const ValueDesc* k = in(1);
                const ValueDesc* v = in(2);
                if (!q || !k || !v) return bad("attention_decode inputs");
                const TensorShape d1 = TensorShape::create({D}).value();
                const TensorShape d1m = TensorShape::create({1, D}).value();
                if (!(q->shape == d1 || q->shape == d1m) ||
                    !(k->shape == d1) || !(v->shape == d1)) {
                    return bad("attention_decode q must be [head_dim] or [1,head_dim], k/v [head_dim]");
                }
                // output may be declared as [head_dim] or [1,head_dim] (compose-friendly)
                if (!(p.output.shape == d1 || p.output.shape == d1m) ||
                    (out != nullptr && !(out->shape == d1 || out->shape == d1m))) {
                    return bad("attention_decode output must be [head_dim] or [1,head_dim]");
                }
                return Status::make_ok();
            }
            case Kind::moe_route: {
                if (p.inputs.size() != 1 || !p.weights.empty()) return bad("moe_route arity");
                const ValueDesc* logits = in(0);
                if (!logits || (logits->shape.rank() != 1 &&
                                !(logits->shape.rank() == 2 && logits->shape.dim(0) == 1))) {
                    return bad("moe_route logits [E] or [1,E]");
                }
                return expect_out(p, logits->shape, out);
            }
            case Kind::moe_merge: {
                if (p.inputs.size() != 2 || !p.weights.empty()) return bad("moe_merge arity (probs, expert_out)");
                const ValueDesc* probs = in(0);
                const ValueDesc* eo = in(1);
                if (!probs || !eo || eo->shape.rank() != 2) {
                    return bad("moe_merge needs probs [E], expert_out [E,H]");
                }
                const std::size_t pe = probs->shape.num_elements().ok()
                                           ? probs->shape.num_elements().value() : 0;
                if (eo->shape.dim(0) != pe) return bad("moe_merge E mismatch");
                return expect_out(p, TensorShape::create({eo->shape.dim(1)}).release(), out);
            }
            default:
                return bad("unknown primitive kind");
        }
    }

    Status expect_out(const Primitive& p, TensorShape expect, const ValueDesc* out) const {
        // When adding, p.output carries the declaration; when re-validating,
        // `out` points at the stored definition. Enforce shape on whichever.
        if (!p.output.shape.empty() && !same_shape(p.output.shape, expect)) {
            return bad(std::string(kind_name(p.kind)) + ": output shape mismatch with signature");
        }
        if (out != nullptr && !same_shape(out->shape, expect)) {
            return bad(std::string(kind_name(p.kind)) + ": stored output shape mismatch");
        }
        return Status::make_ok();
    }

    std::vector<Primitive> primitives_;
    std::vector<ValueDesc> values_;
    std::vector<ValueDesc> inputs_;
    std::vector<StateDesc> states_;
    std::vector<WeightDecl> weights_;
};

} // namespace shelf::ir
