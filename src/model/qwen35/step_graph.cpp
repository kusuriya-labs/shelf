// shelf model/qwen35 — single decode-step Model Graph lowering.
#include <utility>

#include <shelf/model/qwen35/step_graph.hpp>

namespace shelf::model::qwen35 {
namespace {

using namespace shelf;
using namespace shelf::ir;

Status bad(const std::string& msg) {
    return Status::invalid_argument("qwen35/step_graph: " + msg, __FILE__, __LINE__);
}

class Builder {
public:
    Builder(const TextConfig& cfg, std::size_t kv_capacity, std::size_t rows)
        : cfg_(cfg), cap_(kv_capacity), rows_(rows) {}

    Result<StepGraphPlan> build() {
        StepGraphPlan plan;
        plan.kv_capacity = cap_;
        plan.tied_embeddings = cfg_.tie_word_embeddings;

        tokens_ = nid();
        SHELF_RETURN_IF_ERROR(g_.declare_input(vdesc(tokens_, {rows_}, DataType::i32)));
        plan.tokens = tokens_;
        cos_ = nid();
        sin_ = nid();
        const std::size_t rd = cfg_.rotary_dim;
        SHELF_RETURN_IF_ERROR(g_.declare_input(vdesc(cos_, {rd})));
        SHELF_RETURN_IF_ERROR(g_.declare_input(vdesc(sin_, {rd})));
        plan.cos = cos_;
        plan.sin = sin_;

        ValueId h = add_embedding();
        if (!h.valid()) return last_;
        for (std::size_t l = 0; l < cfg_.num_layers; ++l) {
            h = add_layer(l, h);
            if (!h.valid()) return last_;
        }
        ValueId logits = add_final(h);
        if (!logits.valid()) return last_;
        plan.logits = logits;

        plan.graph = std::move(g_);
        plan.weights = std::move(weights_);
        plan.states = std::move(states_);
        return plan;
    }

private:
    ValueId nid() { return ValueId::from_uint32(next_value_++); }

    ValueDesc vdesc(ValueId id, std::initializer_list<std::size_t> dims,
                    DataType dt = DataType::f32) {
        return ValueDesc{id, TensorShape::create(dims).value(), dt};
    }
    ValueDesc vdesc(ValueId id, const std::vector<std::size_t>& dims,
                    DataType dt = DataType::f32) {
        return ValueDesc{id, TensorShape::create(dims).value(), dt};
    }

    // Batch-optional shape: leading rows_ only when rows_ > 1 (rank-1 decode
    // graphs keep their historical shapes; prefill graphs carry the batch).
    std::vector<std::size_t> B(std::initializer_list<std::size_t> d) {
        std::vector<std::size_t> v;
        if (rows_ > 1) v.push_back(rows_);
        for (auto x : d) v.push_back(x);
        return v;
    }

    TensorId wid(const std::string& name, std::initializer_list<std::size_t> shape) {
        TensorId id = TensorId::parse(static_cast<std::uint64_t>(weights_.size()) + 1).value();
        std::vector<std::size_t> sh(shape);
        auto st = g_.declare_weight(id, TensorShape::create(sh).value());
        if (!st.ok()) { last_ = st; return TensorId{}; }
        weights_.push_back(WeightRef{id, name, sh});
        return id;
    }

    ir::StateId add_state(ir::StateKind kind, std::initializer_list<std::size_t> shape) {
        ir::StateId id = ir::StateId::from_uint32(static_cast<std::uint32_t>(states_.size()));
        std::vector<std::size_t> sh(shape);
        auto st = g_.declare_state(ir::StateDesc{id, kind, TensorShape::create(sh).value(),
                                                 DataType::f32});
        if (!st.ok()) { last_ = st; return ir::StateId{}; }
        states_.push_back(StateRef{id, kind, sh});
        return id;
    }

    ValueId prim(Kind kind, std::vector<ValueId> ins, std::vector<TensorId> ws,
                 std::initializer_list<std::size_t> out_dims, Attr attr = {}) {
        Primitive p;
        p.kind = kind;
        p.attr = attr;
        p.inputs = std::move(ins);
        p.weights = std::move(ws);
        p.output = vdesc(nid(), out_dims);
        auto st = g_.add_primitive(std::move(p));
        if (!st.ok()) { last_ = st; return ValueId{}; }
        return p_kind_value(st, kind);
    }

    ValueId prim_vec(Kind kind, std::vector<ValueId> ins, std::vector<TensorId> ws,
                     const std::vector<std::size_t>& out_dims, Attr attr) {
        Primitive p;
        p.kind = kind;
        p.attr = std::move(attr);
        p.inputs = std::move(ins);
        p.weights = std::move(ws);
        p.output = vdesc(nid(), out_dims);
        auto st = g_.add_primitive(std::move(p));
        if (!st.ok()) { last_ = st; return ValueId{}; }
        return g_.primitives().back().output.id;
    }

    ValueId p_kind_value(const Status&, Kind) { return g_.primitives().back().output.id; }

    ValueId reshape_to(ValueId x, std::initializer_list<std::size_t> dims) {
        return prim(Kind::reshape, {x}, {}, dims);
    }
    ValueId reshape_to(ValueId x, const std::vector<std::size_t>& dims) {
        Primitive p;
        p.kind = Kind::reshape;
        p.inputs = {x};
        p.output = vdesc(nid(), dims);
        auto st = g_.add_primitive(std::move(p));
        if (!st.ok()) { last_ = st; return ValueId{}; }
        return g_.primitives().back().output.id;
    }

    ValueId add_embedding() {
        TensorId table = wid("model.language_model.embed_tokens.weight",
                             {cfg_.vocab_size, cfg_.hidden_size});
        if (!table.valid()) return ValueId{};
        embed_id_ = table;
        return prim(Kind::embedding, {tokens_}, {table}, {rows_, cfg_.hidden_size});
    }

    ValueId add_norm(ValueId x, const std::string& wname, bool plus_one) {
        TensorId g = wid(wname, {cfg_.hidden_size});
        if (!g.valid()) return ValueId{};
        Attr a;
        a.epsilon = static_cast<float>(cfg_.rms_eps);
        a.gamma_plus_one = plus_one;
        return prim(Kind::rmsnorm, {x}, {g}, {rows_, cfg_.hidden_size}, a);
    }

    ValueId add_mlp(ValueId x, std::size_t l) {
        const std::string base = "model.language_model.layers." + std::to_string(l) + ".mlp.";
        TensorId gate = wid(base + "gate_proj.weight", {cfg_.intermediate_size, cfg_.hidden_size});
        TensorId up = wid(base + "up_proj.weight", {cfg_.intermediate_size, cfg_.hidden_size});
        TensorId down = wid(base + "down_proj.weight", {cfg_.hidden_size, cfg_.intermediate_size});
        if (!gate.valid() || !up.valid() || !down.valid()) return ValueId{};
        ValueId g = prim(Kind::linear, {x}, {gate}, {rows_, cfg_.intermediate_size});
        if (!g.valid()) return g;
        ValueId s = prim(Kind::silu, {g}, {}, {rows_, cfg_.intermediate_size});
        if (!s.valid()) return s;
        ValueId u = prim(Kind::linear, {x}, {up}, {rows_, cfg_.intermediate_size});
        if (!u.valid()) return u;
        ValueId m = prim(Kind::elem_mul, {s, u}, {}, {rows_, cfg_.intermediate_size});
        if (!m.valid()) return m;
        return prim(Kind::linear, {m}, {down}, {rows_, cfg_.hidden_size});
    }

    ValueId add_full_attention(ValueId x, std::size_t l) {
        const std::string base = "model.language_model.layers." + std::to_string(l) + ".self_attn.";
        const std::size_t D = cfg_.head_dim, nq = cfg_.num_heads, kvh = cfg_.num_kv_heads;
        TensorId wq = wid(base + "q_proj.weight", {cfg_.q_gate_out_features(), cfg_.hidden_size});
        TensorId wk = wid(base + "k_proj.weight", {cfg_.kv_out_features(), cfg_.hidden_size});
        TensorId wv = wid(base + "v_proj.weight", {cfg_.kv_out_features(), cfg_.hidden_size});
        TensorId wo = wid(base + "o_proj.weight", {cfg_.hidden_size, nq * D});
        TensorId wqn = wid(base + "q_norm.weight", {D});
        TensorId wkn = wid(base + "k_norm.weight", {D});
        if (!wq.valid() || !wk.valid() || !wv.valid() || !wo.valid() || !wqn.valid() || !wkn.valid())
            return ValueId{};

        Attr eps;
        eps.epsilon = static_cast<float>(cfg_.rms_eps);
        eps.gamma_plus_one = true;
        Attr rope;
        rope.rotary_dim = static_cast<std::int64_t>(cfg_.rotary_dim);

        ValueId qg = prim(Kind::linear, {x}, {wq}, {rows_, cfg_.q_gate_out_features()});
        if (!qg.valid()) return qg;
        // q_proj view [1, nq, 2D]: per-head [q | gate] interleave — split on the
        // LAST dim after reshaping, not on the flattened row.
        ValueId qg3 = reshape_to(qg, {rows_, nq, 2 * D});
        if (!qg3.valid()) return qg3;
        Attr sl_q;  sl_q.slice_begin = 0;  sl_q.slice_end = static_cast<std::int64_t>(D);
        Attr sl_g;  sl_g.slice_begin = static_cast<std::int64_t>(D);
                      sl_g.slice_end = static_cast<std::int64_t>(2 * D);
        ValueId q3 = prim(Kind::slice, {qg3}, {}, {rows_, nq, D}, sl_q);
        if (!q3.valid()) return q3;
        ValueId gate = prim(Kind::slice, {qg3}, {}, {rows_, nq, D}, sl_g);
        if (!gate.valid()) return gate;
        ValueId qn = prim(Kind::rmsnorm, {q3}, {wqn}, {rows_, nq, D}, eps);
        if (!qn.valid()) return qn;
        ValueId qr = prim(Kind::rope, {qn, cos_, sin_}, {}, {rows_, nq, D}, rope);
        if (!qr.valid()) return qr;
        ValueId q2 = reshape_to(qr, B({nq, D}));
        if (!q2.valid()) return q2;

        ValueId kflat = prim(Kind::linear, {x}, {wk}, {rows_, kvh * D});
        if (!kflat.valid()) return kflat;
        ValueId k3 = reshape_to(kflat, {rows_, kvh, D});
        if (!k3.valid()) return k3;
        ValueId kn = prim(Kind::rmsnorm, {k3}, {wkn}, {rows_, kvh, D}, eps);
        if (!kn.valid()) return kn;
        ValueId kr = prim(Kind::rope, {kn, cos_, sin_}, {}, {rows_, kvh, D}, rope);
        if (!kr.valid()) return kr;
        ValueId k2 = reshape_to(kr, B({kvh, D}));
        if (!k2.valid()) return k2;

        ValueId vflat = prim(Kind::linear, {x}, {wv}, {rows_, kvh * D});
        if (!vflat.valid()) return vflat;
        ValueId v2 = reshape_to(vflat, B({kvh, D}));
        if (!v2.valid()) return v2;

        ir::StateId kv = add_state(ir::StateKind::kv_cache, {cap_, 2, kvh, D});
        if (!kv.valid()) return ValueId{};
        Primitive att;
        att.kind = Kind::attention_gqa_step;
        att.inputs = {q2, k2, v2};
        att.state_read = {kv};
        att.state_write = {kv};
        att.output = vdesc(nid(), B({nq, D}));
        auto st = g_.add_primitive(std::move(att));
        if (!st.ok()) { last_ = st; return ValueId{}; }
        ValueId attn = g_.primitives().back().output.id;

        ValueId attn2 = reshape_to(attn, {rows_, nq * D});
        if (!attn2.valid()) return attn2;
        ValueId gate2 = reshape_to(gate, {rows_, nq * D});
        if (!gate2.valid()) return gate2;
        ValueId gated = prim(Kind::sigmoid_gate, {attn2, gate2}, {}, {rows_, nq * D});
        if (!gated.valid()) return gated;
        return prim(Kind::linear, {gated}, {wo}, {rows_, cfg_.hidden_size});
    }

    ValueId add_linear_attention(ValueId x, std::size_t l) {
        const std::string base = "model.language_model.layers." + std::to_string(l) + ".linear_attn.";
        const std::size_t C = cfg_.conv_channels();
        const std::size_t Kd = cfg_.lin_key_heads * cfg_.lin_key_head_dim;
        const std::size_t Vd = cfg_.lin_value_heads * cfg_.lin_value_head_dim;
        const std::size_t nkh = cfg_.lin_key_heads, nvh = cfg_.lin_value_heads;
        const std::size_t dk = cfg_.lin_key_head_dim, dv = cfg_.lin_value_head_dim;

        TensorId wqkv = wid(base + "in_proj_qkv.weight", {C, cfg_.hidden_size});
        TensorId wz = wid(base + "in_proj_z.weight", {Vd, cfg_.hidden_size});
        TensorId wb = wid(base + "in_proj_b.weight", {nvh, cfg_.hidden_size});
        TensorId wa = wid(base + "in_proj_a.weight", {nvh, cfg_.hidden_size});
        TensorId wconv = wid(base + "conv1d.weight", {C, cfg_.conv_kernel});
        TensorId wal = wid(base + "A_log", {nvh});
        TensorId wdt = wid(base + "dt_bias", {nvh});
        TensorId wn = wid(base + "norm.weight", {dv});
        TensorId wout = wid(base + "out_proj.weight", {cfg_.hidden_size, Vd});
        if (!wqkv.valid() || !wz.valid() || !wb.valid() || !wa.valid() || !wconv.valid() ||
            !wal.valid() || !wdt.valid() || !wn.valid() || !wout.valid())
            return ValueId{};

        ValueId qkv = prim(Kind::linear, {x}, {wqkv}, {rows_, C});
        if (!qkv.valid()) return qkv;
        ValueId qkv1 = reshape_to(qkv, B({C}));
        if (!qkv1.valid()) return qkv1;

        ir::StateId convst = add_state(ir::StateKind::conv, {C, cfg_.conv_kernel - 1});
        if (!convst.valid()) return ValueId{};
        Primitive conv;
        conv.kind = Kind::causal_conv_step;
        conv.inputs = {qkv1};
        conv.weights = {wconv};
        conv.state_read = {convst};
        conv.state_write = {convst};
        conv.output = vdesc(nid(), B({C}));
        auto st = g_.add_primitive(std::move(conv));
        if (!st.ok()) { last_ = st; return ValueId{}; }
        ValueId mixed = g_.primitives().back().output.id;

        auto split = [&](std::size_t b, std::size_t e) {
            Attr sl;
            sl.slice_begin = static_cast<std::int64_t>(b);
            sl.slice_end = static_cast<std::int64_t>(e);
            return prim_vec(Kind::slice, {mixed}, {}, B({e - b}), sl);
        };
        ValueId qf = split(0, Kd);
        if (!qf.valid()) return qf;
        ValueId kf = split(Kd, 2 * Kd);
        if (!kf.valid()) return kf;
        ValueId vf = split(2 * Kd, C);
        if (!vf.valid()) return vf;
        ValueId q = reshape_to(qf, B({nkh, dk}));
        if (!q.valid()) return q;
        ValueId k = reshape_to(kf, B({nkh, dk}));
        if (!k.valid()) return k;
        ValueId v = reshape_to(vf, B({nvh, dv}));
        if (!v.valid()) return v;

        ValueId z = prim(Kind::linear, {x}, {wz}, {rows_, Vd});
        if (!z.valid()) return z;
        ValueId z2 = reshape_to(z, B({nvh, dv}));
        if (!z2.valid()) return z2;
        ValueId b = prim(Kind::linear, {x}, {wb}, {rows_, nvh});
        if (!b.valid()) return b;
        ValueId b1 = reshape_to(b, B({nvh}));
        if (!b1.valid()) return b1;
        ValueId a = prim(Kind::linear, {x}, {wa}, {rows_, nvh});
        if (!a.valid()) return a;
        ValueId a1 = reshape_to(a, B({nvh}));
        if (!a1.valid()) return a1;

        ir::StateId S = add_state(ir::StateKind::recurrent, {nvh, dk, dv});
        if (!S.valid()) return ValueId{};
        Primitive gdn;
        gdn.kind = Kind::gdn_step;
        gdn.inputs = {q, k, v, b1, a1};
        gdn.weights = {wal, wdt};
        gdn.state_read = {S};
        gdn.state_write = {S};
        gdn.output = vdesc(nid(), B({nvh, dv}));
        auto st2 = g_.add_primitive(std::move(gdn));
        if (!st2.ok()) { last_ = st2; return ValueId{}; }
        ValueId core = g_.primitives().back().output.id;

        Attr eps;
        eps.epsilon = 1e-6f;  // RMSNormGated variance eps (HF default; GDN contract)
        ValueId yg = prim_vec(Kind::rmsnorm_gated, {core, z2}, {wn}, B({nvh, dv}), eps);
        if (!yg.valid()) return yg;
        ValueId yflat = reshape_to(yg, {rows_, Vd});
        if (!yflat.valid()) return yflat;
        return prim(Kind::linear, {yflat}, {wout}, {rows_, cfg_.hidden_size});
    }

    ValueId add_layer(std::size_t l, ValueId h) {
        const std::string base = "model.language_model.layers." + std::to_string(l) + ".";
        ValueId x = add_norm(h, base + "input_layernorm.weight", true);
        if (!x.valid()) return x;
        ValueId att = cfg_.layer_kinds[l] == LayerKind::full_attention
                          ? add_full_attention(x, l)
                          : add_linear_attention(x, l);
        if (!att.valid()) return att;
        ValueId h2 = prim(Kind::residual_add, {h, att}, {}, {rows_, cfg_.hidden_size});
        if (!h2.valid()) return h2;
        ValueId x2 = add_norm(h2, base + "post_attention_layernorm.weight", true);
        if (!x2.valid()) return x2;
        ValueId m = add_mlp(x2, l);
        if (!m.valid()) return m;
        return prim(Kind::residual_add, {h2, m}, {}, {rows_, cfg_.hidden_size});
    }

    ValueId add_final(ValueId h) {
        ValueId hn = add_norm(h, "model.language_model.norm.weight", true);
        if (!hn.valid()) return hn;
        if (!embed_id_.valid()) { last_ = bad("embedding weight missing"); return ValueId{}; }
        // tied embeddings: lm_head IS the embedding table (same logical weight id)
        TensorId out = embed_id_;
        if (!cfg_.tie_word_embeddings) {
            // untied (tie_word_embeddings = false): the checkpoint carries its own
            // lm_head.weight. Reusing the embedding here computes logits against the
            // wrong table (Qwen3.5-9B で観測: hidden は HF と一致するのに出力が劣化)。
            out = wid("lm_head.weight", {cfg_.vocab_size, cfg_.hidden_size});
            if (!out.valid()) return ValueId{};   // 失敗時 Status は wid が last_ に設定済み
        }
        return prim(Kind::output_projection, {hn}, {out}, {rows_, cfg_.vocab_size});
    }

    const TextConfig& cfg_;
    std::size_t cap_;
    ir::Graph g_;
    std::vector<WeightRef> weights_;
    std::vector<StateRef> states_;
    std::uint32_t next_value_ = 1;
    ValueId tokens_, cos_, sin_;
    std::size_t rows_;
    TensorId embed_id_;
    Status last_ = Status::make_ok();
};

} // namespace

Result<StepGraphPlan> build_step_graph(const TextConfig& cfg, std::size_t kv_capacity,
                                       std::size_t rows) {
    Builder b(cfg, kv_capacity, rows);
    return b.build();
}

} // namespace shelf::model::qwen35
