// shelf-run — end-to-end text driver for the Qwen3.5-family CPU reference
// path. Loads safetensors shards, replays the single-step Model Graph
// token-by-token (prefill == decode here; slow but exact-by-construction),
// greedy-samples and dumps final logits for oracle parity checks.
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <shelf/artifact/safetensors/reader.hpp>
#include <shelf/backend/cpu/reference_executor.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

#if defined(SHELF_HAVE_HIP) || defined(SHELF_HAVE_SYCL)
#include <memory>
#endif
#ifdef SHELF_HAVE_HIP
#include <shelf/backend/hip/step_executor.hpp>
#endif
#ifdef SHELF_HAVE_SYCL
#include <shelf/backend/sycl/step_executor.hpp>
#endif

namespace {

using namespace shelf;

int fail(const std::string& msg) {
    std::fprintf(stderr, "shelf-run: %s\n", msg.c_str());
    return 1;
}

std::string read_file_str(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::size_t numel(const std::vector<std::size_t>& s) {
    std::size_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

// Read one tensor with checkpoint-native bf16 bits (no conversion).
Result<std::vector<bf16_t>> read_bf16(
        const std::map<std::string, artifact::safetensors::Reader*>& where,
        const std::string& name) {
    auto it = where.find(name);
    if (it == where.end()) return Status::not_found("tensor not in shards: " + name, __FILE__, __LINE__);
    const auto* reader = it->second;
    const auto* entry = reader->catalog().find(name);
    if (!entry) return Status::not_found("catalog miss: " + name, __FILE__, __LINE__);
    if (entry->dtype != DataType::bf16)
        return Status::unsupported("not bf16 on disk: " + name, __FILE__, __LINE__);
    const std::size_t bytes = entry->end - entry->begin;
    std::vector<bf16_t> out(bytes / 2);
    SHELF_RETURN_IF_ERROR(reader->read_tensor(name, out.data(), bytes));
    return out;
}

// Read one tensor from whichever shard holds it, converted to f32.
Result<std::vector<float>> read_f32(
        const std::map<std::string, artifact::safetensors::Reader*>& where,
        const std::string& name) {
    auto it = where.find(name);
    if (it == where.end()) return Status::not_found("tensor not in shards: " + name, __FILE__, __LINE__);
    const auto* reader = it->second;
    const auto* entry = reader->catalog().find(name);
    if (!entry) return Status::not_found("catalog miss: " + name, __FILE__, __LINE__);
    const std::size_t bytes = entry->end - entry->begin;
    if (entry->dtype == DataType::f32) {
        std::vector<float> out(bytes / sizeof(float));
        SHELF_RETURN_IF_ERROR(reader->read_tensor(name, out.data(), bytes));
        return out;
    }
    if (entry->dtype == DataType::bf16) {
        std::vector<bf16_t> raw(bytes / 2);
        auto st = reader->read_tensor(name, raw.data(), bytes);
        if (!st.ok()) return st;
        std::vector<float> out(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) out[i] = bf16_to_f32(raw[i]);
        return out;
    }
    return Status::unsupported("read_f32: dtype for " + name, __FILE__, __LINE__);
}

} // namespace

int main(int argc, char** argv) {
    std::string model_dir, ids_csv, out_dir = "shelf-out", device = "cpu";
    int gpu = 0;
    int steps = 16;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") model_dir = next("model dir");
        else if (a == "--ids") ids_csv = next("ids");
        else if (a == "--steps") steps = std::atoi(next("steps").c_str());
        else if (a == "--out") out_dir = next("out");
        else if (a == "--device") device = next("cpu|hip|sycl");
        else if (a == "--gpu") gpu = std::atoi(next("gpu index").c_str());
        else return fail("usage: shelf-run --model DIR --ids 1,2,3 [--steps N] [--out DIR] "
                         "[--device cpu|hip|sycl]");
    }
    if (model_dir.empty() || ids_csv.empty()) return fail("--model and --ids required");

    // ---- config ----
    auto cfg_r = model::qwen35::load_text_config_json(read_file_str(model_dir + "/config.json"));
    if (!cfg_r.ok()) return fail(cfg_r.status().message());
    const auto& cfg = cfg_r.value();

    std::vector<std::int64_t> ids;
    {
        std::stringstream ss(ids_csv);
        std::string part;
        while (std::getline(ss, part, ',')) if (!part.empty()) ids.push_back(std::stoll(part));
    }
    if (ids.empty()) return fail("empty ids");
    const std::size_t capacity = ids.size() + static_cast<std::size_t>(steps) + 4;

    // ---- graph ----
    const std::size_t prefill_rows =
        (device == "hip" || device == "sycl") ? std::min<std::size_t>(ids.size(), 64) : 1;
    auto plan_r = model::qwen35::build_step_graph(cfg, capacity);
    if (!plan_r.ok()) return fail(plan_r.status().message());
    auto& plan = plan_r.value();
    std::unique_ptr<model::qwen35::StepGraphPlan> pfplan;
    if (prefill_rows > 1) {
        auto pr = model::qwen35::build_step_graph(cfg, capacity, prefill_rows);
        if (!pr.ok()) return fail("prefill graph: " + pr.status().message());
        pfplan = std::make_unique<model::qwen35::StepGraphPlan>(pr.release());
    }
    std::printf("graph: %zu primitives, %zu weights, %zu states, kv_capacity=%zu\n",
                plan.graph.size(), plan.weights.size(), plan.states.size(), capacity);

    // ---- shards ----
    nlohmann::json index;
    try {
        index = nlohmann::json::parse(read_file_str(model_dir + "/model.safetensors.index.json"));
    } catch (...) { return fail("cannot parse model.safetensors.index.json"); }
    std::vector<std::string> shard_files;
    for (const auto& f : index.at("weight_map").items()) {
        std::string file = f.value().get<std::string>();
        if (std::find(shard_files.begin(), shard_files.end(), file) == shard_files.end())
            shard_files.push_back(file);
    }
    std::vector<artifact::safetensors::Reader> readers;
    for (const auto& f : shard_files) {
        auto r = artifact::safetensors::Reader::open(model_dir + "/" + f);
        if (!r.ok()) return fail("open " + f + ": " + r.status().message());
        readers.push_back(r.release());
    }
    std::map<std::string, artifact::safetensors::Reader*> where;
    for (auto& r : readers)
        for (const auto& name : r.list_tensors())
            if (!where.contains(name)) where.emplace(name, &r);

    // ---- weights ----
#ifdef SHELF_HAVE_HIP
    std::unique_ptr<backend::hip::HipStepExecutor> hip_exec;
#endif
#ifdef SHELF_HAVE_SYCL
    std::unique_ptr<backend::sycl::SyclStepExecutor> sycl_exec;
#endif
    backend::cpu::HostStorage storage;
    const auto upload_weights = [&](auto& ex) -> int {
        for (const auto& w : plan.weights) {
            auto tb = read_bf16(where, w.checkpoint_name);
            if (tb.ok()) {
                if (tb.value().size() != numel(w.shape))
                    return fail("shape mismatch for " + w.checkpoint_name);
                auto st = ex->upload_weight_bf16(w.id, tb.value().data(), tb.value().size());
                if (!st.ok()) return fail("upload " + w.checkpoint_name + ": " + st.message());
                continue;
            }
            auto t = read_f32(where, w.checkpoint_name);
            if (!t.ok()) return fail(t.status().message());
            if (t.value().size() != numel(w.shape))
                return fail("shape mismatch for " + w.checkpoint_name);
            auto st = ex->upload_weight(w.id, t.value());
            if (!st.ok()) return fail("upload " + w.checkpoint_name + ": " + st.message());
        }
        auto st = ex->declare_states(plan.graph);
        if (!st.ok()) return fail("declare_states: " + st.message());
        return 0;
    };
    if (device == "sycl") {
#ifdef SHELF_HAVE_SYCL
        backend::sycl::SyclStepExecutorOptions so;
        so.device = gpu;
        so.rope_theta = cfg.rope_theta;
        so.rope_rotary_dim = cfg.rotary_dim;
        sycl_exec = std::make_unique<backend::sycl::SyclStepExecutor>(so);
        if (int rc = upload_weights(sycl_exec)) return rc;
#else
        return fail("built without SYCL support");
#endif
    } else if (device == "hip") {
#ifdef SHELF_HAVE_HIP
        backend::hip::HipStepExecutorOptions ho;
        ho.device = gpu;
        ho.rope_theta = cfg.rope_theta;
        ho.rope_rotary_dim = cfg.rotary_dim;
        hip_exec = std::make_unique<backend::hip::HipStepExecutor>(ho);
        if (int rc = upload_weights(hip_exec)) return rc;
#else
        return fail("built without HIP support");
#endif
    } else {
        for (const auto& w : plan.weights) {
            auto t = read_f32(where, w.checkpoint_name);
            if (!t.ok()) return fail(t.status().message());
            if (t.value().size() != numel(w.shape)) {
                return fail("shape mismatch for " + w.checkpoint_name + ": file has " +
                            std::to_string(t.value().size()) + ", IR contract needs " +
                            std::to_string(numel(w.shape)));
            }
            auto st = storage.set_weight(w.id, t.release());
            if (!st.ok()) return fail("set_weight " + w.checkpoint_name + ": " + st.message());
        }
        for (const auto& s : plan.states) {
            auto st = storage.init_state(s.id, std::vector<float>(numel(s.shape), 0.0f));
            if (!st.ok()) return fail("init_state: " + st.message());
        }
    }

    // ---- rope tables (text path: all mrope channels equal -> identity; HF
    //      computes inv_freq/cos/sin in f32 — we compute in double and round) ----
    const std::size_t rd = cfg.rotary_dim;
    std::vector<double> inv(rd / 2);
    for (std::size_t j = 0; j < rd / 2; ++j)
        inv[j] = std::pow(cfg.rope_theta, -2.0 * static_cast<double>(j) / static_cast<double>(rd));

    auto run_step = [&](std::int64_t token, std::size_t pos, std::vector<float>* logits_out)
        -> Status {
#ifdef SHELF_HAVE_HIP
        if (hip_exec) {
            SHELF_RETURN_IF_ERROR(
                hip_exec->run_step(plan.graph, static_cast<std::int32_t>(token),
                                   static_cast<std::uint32_t>(pos)));
            if (logits_out) {
                auto lg = hip_exec->fetch_logits(plan.graph);
                if (!lg.ok()) return lg.status();
                *logits_out = lg.value();
            }
            std::printf("  hip step: %.2f ms\n", hip_exec->stats().last_step_ms);
            return Status::make_ok();
        }
#endif
#ifdef SHELF_HAVE_SYCL
        if (sycl_exec) {
            SHELF_RETURN_IF_ERROR(
                sycl_exec->run_step(plan.graph, static_cast<std::int32_t>(token),
                                    static_cast<std::uint32_t>(pos)));
            if (logits_out) {
                auto lg = sycl_exec->fetch_logits(plan.graph);
                if (!lg.ok()) return lg.status();
                *logits_out = lg.value();
            }
            std::printf("  sycl step: %.2f ms\n", sycl_exec->stats().last_step_ms);
            return Status::make_ok();
        }
#endif
        storage.reset_values();
        auto st = storage.set_value_int(plan.tokens, {static_cast<std::int32_t>(token)});
        if (!st.ok()) return st;
        std::vector<float> cosv(rd), sinv(rd);
        for (std::size_t j = 0; j < rd / 2; ++j) {
            const double f = static_cast<double>(pos) * inv[j];
            cosv[j] = cosv[j + rd / 2] = static_cast<float>(std::cos(f));
            sinv[j] = sinv[j + rd / 2] = static_cast<float>(std::sin(f));
        }
        SHELF_RETURN_IF_ERROR(storage.set_value(plan.cos, std::move(cosv)));
        SHELF_RETURN_IF_ERROR(storage.set_value(plan.sin, std::move(sinv)));
        SHELF_RETURN_IF_ERROR(backend::cpu::ReferenceExecutor::run(plan.graph, storage));
        if (logits_out) {
            const auto* lv = storage.value(plan.logits);
            if (!lv) return Status::invalid_state("logits missing after run", __FILE__, __LINE__);
            *logits_out = *lv;
        }
        return Status::make_ok();
    };

    // ---- replay prompt, then greedy decode ----
    std::vector<std::int64_t> seq = ids;
    std::vector<float> last_logits;
    const auto gpu_prefill = [&](auto& ex) -> int {
        std::vector<std::int32_t> win;
        for (std::size_t c = 0; c < ids.size(); c += prefill_rows) {
            const std::size_t n = std::min(prefill_rows, ids.size() - c);
            win.clear();
            for (std::size_t i = 0; i < n; ++i)
                win.push_back(static_cast<std::int32_t>(ids[c + i]));
            const bool last_chunk = c + n == ids.size();
            auto st = ex->run_step(pfplan->graph, win.data(), n,
                                   static_cast<std::uint32_t>(c));
            if (!st.ok()) return fail("prefill chunk " + std::to_string(c) + ": " + st.message());
            std::printf("prefill chunk rows=%zu base=%zu (%.2f ms)\n", n, c,
                        ex->stats().last_step_ms);
            if (last_chunk) {
                auto lg = ex->fetch_logits(pfplan->graph);
                if (!lg.ok()) return fail(lg.status().message());
                auto& full = lg.value();  // [n,V]: keep last row
                last_logits.assign(full.end() - static_cast<std::ptrdiff_t>(full.size() / n),
                                   full.end());
            }
        }
        return 0;
    };
#ifdef SHELF_HAVE_HIP
    if (hip_exec && pfplan) { if (int rc = gpu_prefill(hip_exec)) return rc; } else
#endif
#ifdef SHELF_HAVE_SYCL
    if (sycl_exec && pfplan) { if (int rc = gpu_prefill(sycl_exec)) return rc; } else
#endif
    for (std::size_t p = 0; p < ids.size(); ++p) {
        auto st = run_step(seq[p], p, p + 1 == ids.size() ? &last_logits : nullptr);
        if (!st.ok()) return fail("prefill step " + std::to_string(p) + ": " + st.message());
        std::printf("prefill pos=%zu tok=%lld\n", p, static_cast<long long>(seq[p]));
    }
    // position of token seq[i] is i; the token sampled from prefill logits
    // occupies position T, so step s>=1 feeds seq.back() at position T+s-1.
    for (int s = 0; s < steps; ++s) {
        if (s > 0) {
            auto st = run_step(seq.back(), ids.size() + static_cast<std::size_t>(s) - 1,
                               &last_logits);
            if (!st.ok()) return fail("decode step " + std::to_string(s) + ": " + st.message());
        }
        std::size_t best = 0;
        for (std::size_t i = 1; i < last_logits.size(); ++i)
            if (last_logits[i] > last_logits[best]) best = i;
        seq.push_back(static_cast<std::int64_t>(best));
        std::printf("decode pos=%zu tok=%zu\n", ids.size() + static_cast<std::size_t>(s), best);
        if (static_cast<std::int64_t>(best) == cfg.eos_token_id) break;
    }

    // ---- outputs ----
    std::string ids_line;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        if (i) ids_line += ",";
        ids_line += std::to_string(seq[i]);
    }
    std::printf("seq: %s\n", ids_line.c_str());
    if (!last_logits.empty()) {
        std::filesystem::create_directories(out_dir);
        std::ofstream f(out_dir + "/logits_f32.bin", std::ios::binary);
        if (!f) return fail("cannot write " + out_dir + "/logits_f32.bin");
        f.write(reinterpret_cast<const char*>(last_logits.data()),
                static_cast<std::streamsize>(last_logits.size() * sizeof(float)));
        std::printf("wrote %s/logits_f32.bin (%zu floats)\n", out_dir.c_str(), last_logits.size());
    }
    return 0;
}
