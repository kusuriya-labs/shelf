// shelf backend/cpu — reference executor implementation (naive, f32).
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
    std::map<std::uint32_t, std::size_t> kv_rows;  // StateId -> used rows (explicit)
    for (const auto& p : graph.primitives()) {
        SHELF_RETURN_IF_ERROR(run_primitive(graph, p, storage, kv_rows));
    }
    return Status::make_ok();
}

Status ReferenceExecutor::run_primitive(const Graph& graph, const Primitive& p, HostStorage& storage,
                                        std::map<std::uint32_t, std::size_t>& kv_rows) {
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
            for (std::size_t m = 0; m < out_n / H; ++m) {
                float sum = 0.0f;
                for (std::size_t d = 0; d < H; ++d) {
                    const float v = (*x)[m * H + d];
                    sum += v * v;
                }
                const float inv = 1.0f / std::sqrt(sum / static_cast<float>(H) + eps);
                for (std::size_t d = 0; d < H; ++d) {
                    out[m * H + d] = (*x)[m * H + d] * inv * (*gamma)[d];
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
                const float v = (*x)[i];
                out[i] = v / (1.0f + std::exp(-v));
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
            const std::size_t M = p.output.shape.dim(0);
            const std::size_t H = p.output.shape.dim(1);
            const std::size_t rd = p.attr.rotary_dim == 0 ? H
                                                          : static_cast<std::size_t>(p.attr.rotary_dim);
            const std::size_t half = rd / 2;
            std::vector<float> out(*x);  // dims beyond rotary pass through
            for (std::size_t m = 0; m < M; ++m) {
                for (std::size_t i = 0; i < half; ++i) {
                    const std::size_t lo = m * H + i;
                    const std::size_t hi = m * H + i + half;
                    const float a = (*x)[lo], b = (*x)[hi];
                    out[lo] = a * (*c)[lo] - b * (*s)[lo];
                    out[hi] = b * (*c)[hi] + a * (*s)[hi];
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
            std::size_t& rows = kv_rows[p.state_read[0].value()];
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
        default:
            return err("unknown primitive kind");
    }
}

} // namespace shelf::backend::cpu
