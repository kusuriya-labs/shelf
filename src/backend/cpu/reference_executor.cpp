// shelf backend/cpu — reference executor implementation (naive, f32).
#include <algorithm>
#include <cmath>
#include <utility>

#include <shelf/backend/cpu/reference_executor.hpp>

namespace shelf::backend::cpu {
namespace {

using namespace shelf::ir;

Status err(const std::string& msg) {
    return Status::invalid_state("cpu_ref: " + msg, __FILE__, __LINE__);
}

std::size_t elements(const TensorShape& s) {
    auto n = s.num_elements();
    return n.ok() ? n.value() : 0;
}

float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

// softplus(x) = log1p(exp(x)); linear past 20 keeps f32 away from overflow.
float softplus_f(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

} // namespace

Status HostStorage::set_float(std::map<std::uint32_t, Slot>& m, std::uint32_t key,
                              std::vector<float> data, const char* what) {
    auto& slot = m[key];
    if (slot.set) return err(std::string(what) + " overwritten");
    slot.floats = std::move(data);
    slot.is_int = false;
    slot.set = true;
    return Status::make_ok();
}

Status ReferenceExecutor::run(const Graph& graph, HostStorage& storage) {
    SHELF_RETURN_IF_ERROR(graph.validate());
    for (const auto& p : graph.primitives()) {
        SHELF_RETURN_IF_ERROR(run_primitive(graph, p, storage));
    }
    return Status::make_ok();
}

Status ReferenceExecutor::run_primitive(const Graph& graph, const Primitive& p, HostStorage& storage) {
    const std::size_t out_n = elements(p.output.shape);

    // ---- shared fetchers with size validation ----
    auto get = [&](ValueId id) -> const std::vector<float>* {
        const ValueDesc* desc = graph.find_value(id);
        const std::vector<float>* data = storage.value(id);
        return (desc && data && data->size() == elements(desc->shape)) ? data : nullptr;
    };
    auto get_w = [&](TensorId id) -> const std::vector<float>* {
        const TensorShape* shape = graph.find_weight(id);
        const std::vector<float>* data = storage.weight(id);
        return (shape && data && data->size() == elements(*shape)) ? data : nullptr;
    };
    // state payload sized against its declared logical shape (any rank)
    auto get_state = [&](StateId id) -> std::vector<float>* {
        const StateDesc* desc = graph.find_state(id);
        std::vector<float>* data = storage.state(id);
        return (desc && data && data->size() == elements(desc->logical_shape)) ? data : nullptr;
    };

    switch (p.kind) {
        case Kind::embedding: {
            const std::vector<std::int32_t>* tokens = storage.value_int(p.inputs[0]);
            const std::vector<float>* tbl = get_w(p.weights[0]);
            if (!tokens || !tbl) return err("embedding inputs missing");
            const TensorShape* wt = graph.find_weight(p.weights[0]);
            const std::size_t vocab = wt->dim(0), H = wt->dim(1);
            std::vector<float> out(out_n);
            for (std::size_t t = 0; t < tokens->size(); ++t) {
                const std::int32_t tok = (*tokens)[t];
                if (tok < 0 || static_cast<std::size_t>(tok) >= vocab) {
                    return Status::out_of_range("embedding token id out of vocab", __FILE__, __LINE__);
                }
                for (std::size_t h = 0; h < H; ++h) {
                    out[t * H + h] = (*tbl)[static_cast<std::size_t>(tok) * H + h];
                }
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::rmsnorm: {
            const std::vector<float>* x = get(p.inputs[0]);
            const std::vector<float>* gamma = get_w(p.weights[0]);
            if (!x || !gamma) return err("rmsnorm inputs missing");
            const std::size_t H = p.output.shape.dim(p.output.shape.rank() - 1);
            std::vector<float> out(out_n);
            const float eps = p.attr.epsilon;
            const bool g1 = p.attr.gamma_plus_one;  // offset-weight convention: *(1 + w)
            for (std::size_t m = 0; m < out_n / H; ++m) {
                float sum = 0.0f;
                for (std::size_t d = 0; d < H; ++d) {
                    const float v = (*x)[m * H + d];
                    sum += v * v;
                }
                const float inv = 1.0f / std::sqrt(sum / static_cast<float>(H) + eps);
                for (std::size_t d = 0; d < H; ++d) {
                    const float w = g1 ? 1.0f + (*gamma)[d] : (*gamma)[d];
                    out[m * H + d] = (*x)[m * H + d] * inv * w;
                }
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::linear:
        case Kind::output_projection: {
            const std::vector<float>* x = get(p.inputs[0]);
            const std::vector<float>* W = get_w(p.weights[0]);
            if (!x || !W) return err("linear inputs missing");
            const std::size_t M = p.output.shape.dim(0);
            const std::size_t N = p.output.shape.dim(1);
            const TensorShape& xs = graph.find_value(p.inputs[0])->shape;
            const std::size_t K = xs.dim(1);
            std::vector<float> out(out_n, 0.0f);
            for (std::size_t m = 0; m < M; ++m) {
                for (std::size_t n = 0; n < N; ++n) {
                    float acc = 0.0f;
                    for (std::size_t k = 0; k < K; ++k) {
                        acc += (*x)[m * K + k] * (*W)[n * K + k];
                    }
                    out[m * N + n] = acc;
                }
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::silu: {
            const std::vector<float>* x = get(p.inputs[0]);
            if (!x) return err("silu input missing");
            std::vector<float> out(out_n);
            for (std::size_t i = 0; i < out_n; ++i) {
                out[i] = silu_f((*x)[i]);
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::elem_mul:
        case Kind::residual_add: {
            const std::vector<float>* a = get(p.inputs[0]);
            const std::vector<float>* b = get(p.inputs[1]);
            if (!a || !b) return err("binary inputs missing");
            std::vector<float> out(out_n);
            for (std::size_t i = 0; i < out_n; ++i) {
                out[i] = p.kind == Kind::elem_mul ? (*a)[i] * (*b)[i] : (*a)[i] + (*b)[i];
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::rope: {
            const std::vector<float>* x = get(p.inputs[0]);
            const std::vector<float>* c = get(p.inputs[1]);
            const std::vector<float>* s = get(p.inputs[2]);
            if (!x || !c || !s) return err("rope inputs missing");
            const std::size_t H = p.output.shape.dim(p.output.shape.rank() - 1);
            const std::size_t M = out_n / H;  // rows over the last dim (any rank >= 2)
            const std::size_t rd = p.attr.rotary_dim == 0 ? H
                                                          : static_cast<std::size_t>(p.attr.rotary_dim);
            const std::size_t half = rd / 2;
            // cos/sin: same shape as x (per-position table) or rank-1 [rd] broadcast to every row
            const bool full = c->size() == out_n && s->size() == out_n;
            const bool narrow = c->size() == rd && s->size() == rd;
            if (!full && !narrow) return err("rope cos/sin size mismatch");
            auto ci = [&](std::size_t m, std::size_t i) { return full ? m * H + i : i; };
            std::vector<float> out(*x);  // dims beyond rotary pass through
            for (std::size_t m = 0; m < M; ++m) {
                for (std::size_t i = 0; i < half; ++i) {
                    const std::size_t lo = m * H + i;
                    const std::size_t hi = m * H + i + half;
                    const float a = (*x)[lo], b = (*x)[hi];
                    out[lo] = a * (*c)[ci(m, i)] - b * (*s)[ci(m, i)];
                    out[hi] = b * (*c)[ci(m, i + half)] + a * (*s)[ci(m, i + half)];
                }
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::attention_decode: {
            const std::vector<float>* q = get(p.inputs[0]);
            const std::vector<float>* k = get(p.inputs[1]);
            const std::vector<float>* v = get(p.inputs[2]);
            std::vector<float>* kv = storage.state(p.state_read[0]);
            const StateDesc* desc = graph.find_state(p.state_read[0]);
            if (!q || !k || !v || !kv || !desc) return err("attention state/inputs missing");
            const std::size_t D = desc->logical_shape.dim(1);
            const std::size_t cap = desc->logical_shape.dim(0);
            // Layout: K rows [0, cap*D), V rows [cap*D, 2*cap*D).
            // init_state must pre-size with 2*cap*D zeros.
            if (kv->size() != 2 * cap * D) return err("KV state must be 2*capacity*D");
            std::size_t& rows = storage.rows_[p.state_read[0].value()];
            if (rows >= cap) return err("KV cache capacity exceeded");
            for (std::size_t d = 0; d < D; ++d) {
                (*kv)[rows * D + d] = (*k)[d];
                (*kv)[cap * D + rows * D + d] = (*v)[d];
            }
            ++rows;
            const float scale = p.attr.score_scale != 0.0f
                                    ? p.attr.score_scale
                                    : 1.0f / std::sqrt(static_cast<float>(D));
            const std::size_t n = rows;  // rows already includes the appended one
            std::vector<float> scores(n);
            float max_s = -INFINITY;
            for (std::size_t r = 0; r < n; ++r) {
                float acc = 0.0f;
                for (std::size_t d = 0; d < D; ++d) acc += (*q)[d] * (*kv)[r * D + d];
                scores[r] = acc * scale;
                max_s = std::max(max_s, scores[r]);
            }
            float sum = 0.0f;
            for (auto& sc : scores) { sc = std::exp(sc - max_s); sum += sc; }
            std::vector<float> out(D, 0.0f);
            for (std::size_t r = 0; r < n; ++r) {
                const float w = scores[r] / sum;
                for (std::size_t d = 0; d < D; ++d) {
                    out[d] += w * (*kv)[cap * D + r * D + d];
                }
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::moe_route: {
            const std::vector<float>* logits = get(p.inputs[0]);
            if (!logits) return err("moe_route input missing");
            float max_s = -INFINITY;
            for (float s : *logits) max_s = std::max(max_s, s);
            std::vector<float> out(out_n);
            float sum = 0.0f;
            for (std::size_t e = 0; e < out_n; ++e) {
                out[e] = std::exp((*logits)[e] - max_s);
                sum += out[e];
            }
            for (auto& p_e : out) p_e /= sum;
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::moe_merge: {
            const std::vector<float>* probs = get(p.inputs[0]);
            const std::vector<float>* eo = get(p.inputs[1]);
            if (!probs || !eo) return err("moe_merge inputs missing");
            const std::size_t E = probs->size();
            const std::size_t H = out_n;
            if (eo->size() != E * H) return err("moe_merge expert_out size");
            std::vector<float> out(H, 0.0f);
            for (std::size_t e = 0; e < E; ++e) {
                const float w = (*probs)[e];
                for (std::size_t h = 0; h < H; ++h) out[h] += w * (*eo)[e * H + h];
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::slice: {
            const std::vector<float>* x = get(p.inputs[0]);
            if (!x) return err("slice input missing");
            const TensorShape& xs = graph.find_value(p.inputs[0])->shape;
            const std::size_t rank = xs.rank();
            const std::size_t N = xs.dim(rank - 1);
            const std::int64_t b = p.attr.slice_begin;
            const std::int64_t e = p.attr.slice_end == -1 ? static_cast<std::int64_t>(N)
                                                          : p.attr.slice_end;
            if (b < 0 || e <= b || static_cast<std::size_t>(e) > N) {
                return err("slice range invalid for last dim of size " + std::to_string(N));
            }
            const std::size_t len = static_cast<std::size_t>(e - b);
            if (out_n != (x->size() / N) * len) return err("slice output shape mismatch");
            std::vector<float> out(out_n);
            for (std::size_t r = 0; r < x->size() / N; ++r) {
                for (std::size_t d = 0; d < len; ++d) {
                    out[r * len + d] = (*x)[r * N + static_cast<std::size_t>(b) + d];
                }
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::reshape: {
            const std::vector<float>* x = get(p.inputs[0]);
            if (!x) return err("reshape input missing");
            if (x->size() != out_n) return err("reshape element count mismatch");
            // pure view in the model; SSA copy here (element order unchanged)
            return storage.set_value(p.output.id, std::vector<float>(*x));
        }
        case Kind::sigmoid_gate: {
            const std::vector<float>* a = get(p.inputs[0]);
            const std::vector<float>* b = get(p.inputs[1]);
            if (!a || !b) return err("sigmoid_gate inputs missing");
            std::vector<float> out(out_n);
            for (std::size_t i = 0; i < out_n; ++i) {
                out[i] = (*a)[i] * sigmoid_f((*b)[i]);
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::rmsnorm_gated: {
            const std::vector<float>* x = get(p.inputs[0]);
            const std::vector<float>* z = get(p.inputs[1]);
            const std::vector<float>* gamma = get_w(p.weights[0]);
            if (!x || !z || !gamma) return err("rmsnorm_gated inputs missing");
            const std::size_t H = p.output.shape.dim(p.output.shape.rank() - 1);
            std::vector<float> out(out_n);
            const float eps = p.attr.epsilon;
            for (std::size_t m = 0; m < out_n / H; ++m) {
                float sum = 0.0f;
                for (std::size_t d = 0; d < H; ++d) {
                    const float v = (*x)[m * H + d];
                    sum += v * v;
                }
                const float inv = 1.0f / std::sqrt(sum / static_cast<float>(H) + eps);
                // gate AFTER norm+gamma (HF reference): norm(x) * gamma * silu(z)
                for (std::size_t d = 0; d < H; ++d) {
                    out[m * H + d] = (*x)[m * H + d] * inv * (*gamma)[d] * silu_f((*z)[m * H + d]);
                }
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::causal_conv_step: {
            const std::vector<float>* x = get(p.inputs[0]);
            const std::vector<float>* convW = get_w(p.weights[0]);
            std::vector<float>* st = get_state(p.state_read[0]);
            if (!x || !convW || !st) return err("causal_conv_step inputs/state missing");
            const std::size_t C = x->size();
            if (st->size() != C * 3 || convW->size() != C * 4) return err("causal_conv_step sizes");
            std::vector<float> out(C);
            // state holds the previous 3 inputs oldest-first; u = [s0, s1, s2, x]
            for (std::size_t c = 0; c < C; ++c) {
                const float u[4] = {(*st)[c * 3 + 0], (*st)[c * 3 + 1], (*st)[c * 3 + 2], (*x)[c]};
                float acc = 0.0f;
                for (std::size_t k = 0; k < 4; ++k) acc += (*convW)[c * 4 + k] * u[k];
                out[c] = silu_f(acc);
            }
            for (std::size_t c = 0; c < C; ++c) {
                const float s1 = (*st)[c * 3 + 1];
                const float s2 = (*st)[c * 3 + 2];
                (*st)[c * 3 + 0] = s1;
                (*st)[c * 3 + 1] = s2;
                (*st)[c * 3 + 2] = (*x)[c];
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::gdn_step: {
            const std::vector<float>* q = get(p.inputs[0]);
            const std::vector<float>* k = get(p.inputs[1]);
            const std::vector<float>* v = get(p.inputs[2]);
            const std::vector<float>* b = get(p.inputs[3]);
            const std::vector<float>* a = get(p.inputs[4]);
            const std::vector<float>* A_log = get_w(p.weights[0]);
            const std::vector<float>* dt_bias = get_w(p.weights[1]);
            std::vector<float>* S = get_state(p.state_read[0]);
            if (!q || !k || !v || !b || !a || !A_log || !dt_bias || !S) {
                return err("gdn_step inputs/state missing");
            }
            const TensorShape& qs = graph.find_value(p.inputs[0])->shape;
            const TensorShape& vs = graph.find_value(p.inputs[2])->shape;
            const std::size_t nkh = qs.dim(0), dk = qs.dim(1);
            const std::size_t nvh = vs.dim(0), dv = vs.dim(1);
            if (S->size() != nvh * dk * dv) return err("gdn_step recurrent state size");
            std::size_t rep = p.attr.head_rep != 0 ? static_cast<std::size_t>(p.attr.head_rep)
                                                   : (nvh % nkh == 0 ? nvh / nkh : 0);
            if (rep == 0 || nkh * rep != nvh) return err("gdn_step head_rep mismatch");
            std::vector<float> out(out_n, 0.0f);
            for (std::size_t vh = 0; vh < nvh; ++vh) {
                const std::size_t kh = vh / rep;
                const float beta = sigmoid_f((*b)[vh]);
                const float g = -std::exp((*A_log)[vh]) * softplus_f((*a)[vh] + (*dt_bias)[vh]);
                // l2-normalize q/k rows, then scale q by 1/sqrt(dk)
                std::vector<float> ql2(dk), kl2(dk);
                float qsum = 0.0f, ksum = 0.0f;
                for (std::size_t d = 0; d < dk; ++d) {
                    const float qv = (*q)[kh * dk + d], kv = (*k)[kh * dk + d];
                    qsum += qv * qv;
                    ksum += kv * kv;
                }
                const float qinv = 1.0f / std::sqrt(qsum + 1e-6f);
                const float kinv = 1.0f / std::sqrt(ksum + 1e-6f);
                const float dkinv = 1.0f / std::sqrt(static_cast<float>(dk));
                for (std::size_t d = 0; d < dk; ++d) {
                    ql2[d] = (*q)[kh * dk + d] * qinv * dkinv;
                    kl2[d] = (*k)[kh * dk + d] * kinv;
                }
                float* Sh = S->data() + vh * dk * dv;
                const float decay = std::exp(g);
                for (std::size_t i = 0; i < dk; ++i) {
                    for (std::size_t j = 0; j < dv; ++j) Sh[i * dv + j] *= decay;
                }
                std::vector<float> delta(dv);
                for (std::size_t j = 0; j < dv; ++j) {
                    float kv_mem = 0.0f;
                    for (std::size_t i = 0; i < dk; ++i) kv_mem += Sh[i * dv + j] * kl2[i];
                    delta[j] = ((*v)[vh * dv + j] - kv_mem) * beta;
                }
                for (std::size_t i = 0; i < dk; ++i) {
                    for (std::size_t j = 0; j < dv; ++j) Sh[i * dv + j] += kl2[i] * delta[j];
                }
                for (std::size_t j = 0; j < dv; ++j) {
                    float acc = 0.0f;
                    for (std::size_t i = 0; i < dk; ++i) acc += Sh[i * dv + j] * ql2[i];
                    out[vh * dv + j] = acc;
                }
            }
            ++storage.rows_[p.state_read[0].value()];  // step counter persists across run()s
            return storage.set_value(p.output.id, std::move(out));
        }
        case Kind::attention_gqa_step: {
            const std::vector<float>* q = get(p.inputs[0]);
            const std::vector<float>* kn = get(p.inputs[1]);
            const std::vector<float>* vn = get(p.inputs[2]);
            std::vector<float>* kv = get_state(p.state_read[0]);
            const StateDesc* desc = graph.find_state(p.state_read[0]);
            if (!q || !kn || !vn || !kv || !desc) return err("attention_gqa_step inputs/state missing");
            const std::size_t cap = desc->logical_shape.dim(0);
            const std::size_t kvh = desc->logical_shape.dim(2);
            const std::size_t D = desc->logical_shape.dim(3);
            // Layout [capacity,2,kvh,D]: index ((r*2 + kv)*kvh + h)*D + d
            if (kv->size() != 2 * cap * kvh * D) return err("KV state must be 2*capacity*kvh*D");
            const TensorShape& qs = graph.find_value(p.inputs[0])->shape;
            const std::size_t nq = qs.dim(0);
            if (nq % kvh != 0) return err("attention_gqa_step nq % kvh");
            std::size_t& rows = storage.rows_[p.state_read[0].value()];
            if (rows >= cap) return err("KV cache capacity exceeded");
            for (std::size_t h = 0; h < kvh; ++h) {
                for (std::size_t d = 0; d < D; ++d) {
                    (*kv)[((rows * 2 + 0) * kvh + h) * D + d] = (*kn)[h * D + d];
                    (*kv)[((rows * 2 + 1) * kvh + h) * D + d] = (*vn)[h * D + d];
                }
            }
            ++rows;
            const float scale = p.attr.score_scale != 0.0f
                                    ? p.attr.score_scale
                                    : 1.0f / std::sqrt(static_cast<float>(D));
            const std::size_t n = rows;  // rows already includes the appended one
            const std::size_t group = nq / kvh;
            std::vector<float> out(out_n, 0.0f);
            std::vector<float> scores(n);
            for (std::size_t hq = 0; hq < nq; ++hq) {
                const std::size_t h = hq / group;
                float max_s = -INFINITY;
                for (std::size_t r = 0; r < n; ++r) {
                    float acc = 0.0f;
                    for (std::size_t d = 0; d < D; ++d) {
                        acc += (*q)[hq * D + d] * (*kv)[((r * 2 + 0) * kvh + h) * D + d];
                    }
                    scores[r] = acc * scale;
                    max_s = std::max(max_s, scores[r]);
                }
                float sum = 0.0f;
                for (auto& sc : scores) { sc = std::exp(sc - max_s); sum += sc; }
                for (std::size_t r = 0; r < n; ++r) {
                    const float w = scores[r] / sum;
                    for (std::size_t d = 0; d < D; ++d) {
                        out[hq * D + d] += w * (*kv)[((r * 2 + 1) * kvh + h) * D + d];
                    }
                }
            }
            return storage.set_value(p.output.id, std::move(out));
        }
        default:
            return err("unknown primitive kind");
    }
}

} // namespace shelf::backend::cpu
