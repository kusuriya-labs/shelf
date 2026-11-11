#include "single_request.h"

#include "compute_common.h"

#include <shelf/core/gpu/scoped_device.h>
#include <shelf/models/qwen35/dflash2/context_state.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include "private_target_weights.h"
#include <shelf/models/qwen35/runtime/decode_backend.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/models/qwen35/runtime/mtp_drafter.h>
#include <shelf/models/qwen35/runtime/dflash2_session_pool.h>
#include <shelf/models/qwen35/runtime/runtime_request.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <cstring>
#include <memory>
#include <vector>

namespace shelf::app {
namespace {



std::size_t safetensors_dir_bytes(const std::string& dir) {
    std::size_t total = 0;
    std::error_code ec;
    const std::filesystem::path root(dir);
    if (!std::filesystem::is_directory(root, ec)) {
        return 0;
    }
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        constexpr std::size_t kSuffix = 12;
        if (name.size() <= kSuffix ||
            name.compare(name.size() - kSuffix, kSuffix, ".safetensors") != 0) {
            continue;
        }
        total += static_cast<std::size_t>(entry.file_size(ec));
    }
    return total;
}

}  // namespace



void print_model_geometry(const shelf::models::LoadedModel& model) {
    const auto& text_config = model.hybrid_config();
    const uint32_t kv_heads = static_cast<uint32_t>(text_config.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(text_config.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(text_config);
    const auto gdn_layout =
        shelf::qwen35::make_qwen35_gdn_state_layout(text_config);

    printf("MODEL_HIDDEN_SIZE=%zu\n", text_config.hidden_size);
    printf("MODEL_FAMILY=%s\n", model.family() == shelf::models::ModelFamily::Plamo3
        ? "plamo3" : (model.runs_grouped_experts() ? "moe" : "dense"));
    if (const auto* dense = model.dense_model()) {
        printf("MODEL_INTERMEDIATE_SIZE=%zu\n", dense->text_config().ffn.intermediate_size);
    }
    if (const auto* moe = model.moe_model()) {
        printf("MODEL_EXPERTS=%zu\n", moe->text_config().ffn.num_experts);
        printf("MODEL_EXPERTS_PER_TOK=%zu\n", moe->text_config().ffn.num_experts_per_tok);
        printf("MODEL_MOE_INTERMEDIATE_SIZE=%zu\n",
               moe->text_config().ffn.moe_intermediate_size);
        printf("MODEL_SHARED_EXPERT_INTERMEDIATE_SIZE=%zu\n",
               moe->text_config().ffn.shared_expert_intermediate_size);
    }
    printf("MODEL_LAYERS=%zu\n", text_config.num_hidden_layers);
    printf("MODEL_GDN_LAYERS=%u\n", gdn_layout.num_gdn_states);
    printf("MODEL_ATTENTION_LAYERS=%u\n", num_attn_layers);
    printf("MODEL_Q_HEADS=%zu\n", text_config.num_attention_heads);
    printf("MODEL_KV_HEADS=%u\n", kv_heads);
    printf("MODEL_HEAD_DIM=%u\n", head_dim);
    printf("MODEL_GDN_KEY_HEADS=%zu\n", text_config.linear_num_key_heads);
    printf("MODEL_GDN_VALUE_HEADS=%u\n", gdn_layout.num_v_heads);
    printf("MODEL_GDN_KEY_HEAD_DIM=%u\n", gdn_layout.head_k);
    printf("MODEL_GDN_VALUE_HEAD_DIM=%u\n", gdn_layout.head_v);
    printf("MODEL_GDN_CONV_DIM=%u\n", gdn_layout.conv_dim);
    printf("MODEL_GDN_CONV_HISTORY=%u\n", gdn_layout.conv_history);
    fflush(stdout);
}

void dump_mem_debug(shelf::app::Qwen35ComputeRuntime& runtime) {
    const auto& model = runtime.model();
    auto& executor = runtime.executor();
    const auto& gdn_pool = runtime.gdn_pool();
    const auto* wts = model.dense_model() ? &model.dense_model()->weights() : nullptr;
    uint64_t bf16_bytes = 0, fp8_codes = 0, fp8_scales = 0;
    uint64_t w4_codes = 0, w4_sscales = 0, w4_cscales = 0;
    uint64_t w8_codes = 0, w8_sscales = 0, w8_cscales = 0;
    auto add_matrix = [&](const shelf::weights::MatrixWeight& m) {
        using E = shelf::weights::MatrixEncoding;
        if (m.rows == 0) return;
        const uint64_t r = m.rows;
        if (m.encoding == E::Bf16) {
            bf16_bytes += r * m.cols * 2u;
        } else if (m.encoding == E::Fp8Finegrained) {
            fp8_codes += r * m.k_padded;
            fp8_scales += ((r + 127u) / 128u) * (m.k_padded / 128u) * 4u;
        }
    };
    // The per-encoding breakdown is read from a dense weight set. A model with grouped
    // experts reports its resident bytes through ARENA_USED_BYTES instead.
    if (wts != nullptr) {
        add_matrix(wts->embed_tokens);
        if (!wts->lm_head_tied) add_matrix(wts->lm_head);
        for (const auto& l : wts->layers) {
            add_matrix(l.ffn.gate);
            add_matrix(l.ffn.up);
            add_matrix(l.ffn.down);
            if (l.is_gdn) {
                add_matrix(l.attn_in_proj_a);
                add_matrix(l.attn_in_proj_b);
                add_matrix(l.attn_in_proj_qkv);
                add_matrix(l.attn_in_proj_z);
                add_matrix(l.attn_out_proj);
            } else {
                add_matrix(l.attn_q_proj);
                add_matrix(l.attn_k_proj);
                add_matrix(l.attn_v_proj);
                add_matrix(l.attn_o_proj);
            }
        }
    }
    uint64_t staging_dispatch_count = 0;
    uint64_t staging_max_image_bytes = 0;
    uint64_t staging_shared_program_bytes = 0;
    for (size_t b = 0; b < shelf::runtime::ProgramSet::kRowBucketCount; ++b) {
        const auto& p = executor.program_set.programs[b];
        printf("ROWBUCKET_%zu_WORKSPACE_BYTES=%zu\n", b, p.workspace.total_bytes);
        staging_dispatch_count += p.dispatches.size();
        const uint64_t slot_base = executor.staging_pool.slot(static_cast<uint32_t>(b), 0);
        for (size_t i = 0; i < p.dispatches.size(); ++i) {
            const uint64_t slot = slot_base + i;
            if (slot < executor.staging_pool.bytes.size()) {
                const uint64_t image = executor.staging_pool.bytes[slot];
                if (image > staging_max_image_bytes) staging_max_image_bytes = image;
            }
        }
        if (b < executor.program_staging_meta.size()) {
            staging_shared_program_bytes += executor.program_staging_meta[b].device_bytes;
        }
    }
    printf("WEIGHTS_BF16_BYTES=%llu\n", (unsigned long long)bf16_bytes);
    printf("WEIGHTS_W4_CODES_BYTES=%llu\n", (unsigned long long)w4_codes);
    printf("WEIGHTS_W4_STORAGE_SCALE_BYTES=%llu\n", (unsigned long long)w4_sscales);
    printf("WEIGHTS_W4_COMPUTE_SCALE_BYTES=%llu\n", (unsigned long long)w4_cscales);
    printf("WEIGHTS_W8_CODES_BYTES=%llu\n", (unsigned long long)w8_codes);
    printf("WEIGHTS_W8_STORAGE_SCALE_BYTES=%llu\n", (unsigned long long)w8_sscales);
    printf("WEIGHTS_W8_COMPUTE_SCALE_BYTES=%llu\n", (unsigned long long)w8_cscales);
    printf("WEIGHTS_FP8_CODES_BYTES=%llu\n", (unsigned long long)fp8_codes);
    printf("WEIGHTS_FP8_SCALE_BYTES=%llu\n", (unsigned long long)fp8_scales);
    printf("DISPATCH_STAGING_DEVICE_BYTES=%llu\n",
           (unsigned long long)executor.staging_pool.pool_bytes);
    printf("DISPATCH_STAGING_HOST_BYTES=%llu\n",
           (unsigned long long)executor.dispatch_staging_bytes);
    printf("STAGING_DISPATCH_COUNT=%llu\n", (unsigned long long)staging_dispatch_count);
    printf("STAGING_MAX_IMAGE_BYTES=%llu\n", (unsigned long long)staging_max_image_bytes);
    printf("STAGING_SHARED_PROGRAM_BYTES=%llu\n", (unsigned long long)staging_shared_program_bytes);
    printf("EXECUTION_WORKSPACE_BYTES=%zu\n", executor.execution_workspace.allocation_bytes());
    printf("GDN_STATE_BYTES=%zu\n", gdn_pool.reserved_bytes());
    printf("ARENA_USED_BYTES=%zu\n", runtime.arena().used());
    printf("ARENA_CAPACITY_BYTES=%zu\n", runtime.arena().capacity());
    printf("ARENA_COMMITTED_BYTES=%zu\n", runtime.arena().committed());
    printf("ARENA_VMM=%d\n", runtime.arena().is_vmm() ? 1 : 0);
}

struct LogitsDumper {
    FILE* file = nullptr;
    std::size_t vocab = 0;
    std::vector<std::size_t> step_rows;

    bool open(const std::string& path, std::size_t vocab_size) {
        vocab = vocab_size;
        file = std::fopen(path.c_str(), "wb");
        return file != nullptr;
    }

    bool step(shelf::qwen35::Executor& executor) {
        if (file == nullptr) return true;
        const auto& lg = executor.logits;
        if (lg.ndim() != 2 || lg.dim(1) != vocab || lg.element_size() != sizeof(float)) {
            fprintf(stderr,
                    "dump-logits: unexpected logits shape ndim=%zu dim0=%zu dim1=%zu elem=%zu (vocab=%zu)\n",
                    lg.ndim(), lg.ndim() > 0 ? lg.dim(0) : 0,
                    lg.ndim() > 1 ? lg.dim(1) : 0, lg.element_size(), vocab);
            return false;
        }
        const std::size_t rows = lg.dim(0);
        std::vector<float> host(rows * vocab);
        hipError_t herr = hipMemcpy(host.data(), lg.data<float>(),
                                    rows * vocab * sizeof(float), hipMemcpyDeviceToHost);
        if (herr != hipSuccess) {
            fprintf(stderr, "dump-logits copy failed: %s\n", hipGetErrorString(herr));
            return false;
        }
        if (std::fwrite(host.data(), sizeof(float), rows * vocab, file) != rows * vocab) {
            fprintf(stderr, "dump-logits write failed\n");
            return false;
        }
        step_rows.push_back(rows);
        return true;
    }

    void finish() {
        if (file == nullptr) return;
        std::fclose(file);
        file = nullptr;
        std::size_t total_rows = 0;
        printf("LOGITS_VOCAB=%zu\n", vocab);
        printf("LOGITS_STEP_ROWS=");
        for (std::size_t i = 0; i < step_rows.size(); ++i) {
            if (i > 0) printf(",");
            printf("%zu", step_rows[i]);
            total_rows += step_rows[i];
        }
        printf("\n");
        printf("LOGITS_TOTAL_ROWS=%zu\n", total_rows);
    }
};

void dump_hidden_debug(shelf::qwen35::Executor& executor, const char* tag) {
    const char* h = std::getenv("SHELF_DEBUG_HIDDEN");
    if (h == nullptr || h[0] == '\0') return;
    for (int t = 0; t < 2; ++t) {
        const auto& ten = (t == 0) ? executor.output_hidden : executor.token_hidden;
        if (ten.ndim() != 2 || ten.element_size() != sizeof(std::uint16_t)) {
            fprintf(stderr, "HIDDENDBG %s t=%d bad shape ndim=%zu\n", tag, t, ten.ndim());
            continue;
        }
        const std::size_t rows = ten.dim(0);
        const std::size_t cols = ten.dim(1);
        std::vector<std::uint16_t> raw(rows * cols);
        hipError_t he = hipMemcpy(raw.data(), ten.data<void>(),
                                rows * cols * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
        if (he != hipSuccess) {
            fprintf(stderr, "HIDDENDBG %s t=%d copy failed: %s\n", tag, t, hipGetErrorString(he));
            continue;
        }
        double mx = 0.0;
        std::size_t nz = 0;
        for (std::size_t i = 0; i < raw.size(); ++i) {
            const std::uint32_t u = static_cast<std::uint32_t>(raw[i]) << 16;
            float fv;
            std::memcpy(&fv, &u, sizeof(fv));
            const double a = std::abs(fv);
            if (a > mx) mx = a;
            if (u != 0) ++nz;
        }
        fprintf(stderr, "HIDDENDBG %s t=%d rows=%zu cols=%zu max=%.6e nonzero=%zu\n",
                tag, t, rows, cols, mx, nz);
        const char* dbg_file = std::getenv("SHELF_DEBUG_HIDDEN_FILE");
        if (dbg_file != nullptr && dbg_file[0] != '\0') {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/%s_t%d.bin", dbg_file, tag, t);
            FILE* hf = std::fopen(path, "wb");
            if (hf != nullptr) {
                std::vector<float> fv(rows * cols);
                for (std::size_t i = 0; i < raw.size(); ++i) {
                    const std::uint32_t u = static_cast<std::uint32_t>(raw[i]) << 16;
                    std::memcpy(&fv[i], &u, sizeof(float));
                }
                std::fwrite(fv.data(), sizeof(float), fv.size(), hf);
                std::fclose(hf);
            }
        }
    }
}

void dump_tap_hidden(shelf::qwen35::Executor& executor, const char* tag, std::size_t rows) {
    const uint32_t taps = executor.config.target_hidden_tap_count;
    if (taps == 0) return;
    const char* dbg_file = std::getenv("SHELF_DEBUG_HIDDEN_FILE");
    for (uint32_t t = 0; t < taps; ++t) {
        const auto& ten = executor.dflash_target_hidden[t];
        const std::size_t layer = executor.config.target_hidden_taps[t];
        if (ten.ndim() != 2 || ten.element_size() != sizeof(std::uint16_t)) {
            fprintf(stderr, "TAPDBG %s layer=%zu bad shape ndim=%zu\n", tag, layer, ten.ndim());
            continue;
        }
        std::size_t use_rows = rows < ten.dim(0) ? rows : ten.dim(0);
        const std::size_t cols = ten.dim(1);
        std::vector<std::uint16_t> raw(use_rows * cols);
        hipError_t he = hipMemcpy(raw.data(), ten.data<void>(),
                                   raw.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
        if (he != hipSuccess) {
            fprintf(stderr, "TAPDBG %s layer=%zu copy failed: %s\n",
                    tag, layer, hipGetErrorString(he));
            continue;
        }
        double mx = 0.0;
        for (std::size_t i = 0; i < raw.size(); ++i) {
            const std::uint32_t u = static_cast<std::uint32_t>(raw[i]) << 16;
            float fv;
            std::memcpy(&fv, &u, sizeof(fv));
            const double a = std::abs(fv);
            if (a > mx) mx = a;
        }
        fprintf(stderr, "TAPDBG %s layer=%zu rows=%zu cols=%zu max=%.6e\n",
                tag, layer, use_rows, cols, mx);
        if (dbg_file != nullptr && dbg_file[0] != '\0') {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/%s_L%zu.bin", dbg_file, tag, layer);
            FILE* hf = std::fopen(path, "wb");
            if (hf != nullptr) {
                std::vector<float> fv(raw.size());
                for (std::size_t i = 0; i < raw.size(); ++i) {
                    const std::uint32_t u = static_cast<std::uint32_t>(raw[i]) << 16;
                    std::memcpy(&fv[i], &u, sizeof(float));
                }
                std::fwrite(fv.data(), sizeof(float), fv.size(), hf);
                std::fclose(hf);
            }
        }
    }
}

int run_single_shot(shelf::app::Qwen35ComputeRuntime& runtime,
                    const std::vector<int32_t>& prompt_ids,
                    unsigned long max_new_tokens,
                    const shelf::qwen35::runtime::SamplingConfig& sampling,
                    const std::string& kv_cache_dtype,
                    const std::string& dump_logits_path) {
    auto& executor = runtime.executor();
    auto& kv_pool = runtime.kv_pool();
    print_model_geometry(runtime.model());
    if (std::getenv("SHELF_MEM_DEBUG") != nullptr) {
        dump_mem_debug(runtime);
    }

    LogitsDumper dumper;
    if (!dump_logits_path.empty()) {
        if (!dumper.open(dump_logits_path, runtime.model().hybrid_config().vocab_size)) {
            fprintf(stderr, "cannot open --dump-logits path %s\n", dump_logits_path.c_str());
            return 1;
        }
    }

    auto submit_result = runtime.submit(prompt_ids, static_cast<uint32_t>(max_new_tokens), sampling);
    if (!submit_result.ok()) {
        fprintf(stderr, "submit failed: %s\n", submit_result.status().message().c_str());
        return 1;
    }
    const uint64_t request_id = submit_result.release();

    auto t_prefill_start = std::chrono::steady_clock::now();
    uint32_t prefill_chunks = 0;
    for (;;) {
        const auto* prefill_request = runtime.request(request_id);
        if (prefill_request == nullptr ||
            prefill_request->state == shelf::qwen35::runtime::RequestState::Finished ||
            prefill_request->sequence.position >= prefill_request->input_tokens.size()) {
            break;
        }
        auto prefill_step = runtime.step();
        if (!prefill_step.ok()) {
            fprintf(stderr, "prefill step failed: %s\n",
                    prefill_step.status().message().c_str());
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after prefill failure: %s\n",
                        cleanup_st.message().c_str());
            }
            return 1;
        }
        if (!prefill_step.value().executed) {
            fprintf(stderr, "prefill step did not execute\n");
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after prefill failure: %s\n",
                        cleanup_st.message().c_str());
            }
            return 1;
        }
        ++prefill_chunks;
        hipError_t prefill_sync = hipStreamSynchronize(runtime.stream());
        if (prefill_sync != hipSuccess) {
            fprintf(stderr, "prefill sync failed: %s\n", hipGetErrorString(prefill_sync));
            return 1;
        }
        if (dumper.file != nullptr) {
            if (!dumper.step(executor)) return 1;
        }
    }
    hipError_t herr = hipStreamSynchronize(runtime.stream());
    if (herr != hipSuccess) {
        fprintf(stderr, "prefill sync failed: %s\n", hipGetErrorString(herr));
        return 1;
    }
    if (dumper.file != nullptr) {
        if (!dumper.step(executor)) return 1;
    }
    dump_hidden_debug(executor, "prefill");
    dump_tap_hidden(executor, "prefill", prompt_ids.size());
    auto t_prefill_end = std::chrono::steady_clock::now();

    auto t_decode_start = std::chrono::steady_clock::now();
    uint32_t decode_tokens = 0;
    while (runtime.has_pending()) {
        auto step_result = runtime.step();
        if (!step_result.ok()) {
            fprintf(stderr, "decode step failed: %s\n", step_result.status().message().c_str());
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after decode failure: %s\n", cleanup_st.message().c_str());
            }
            return 1;
        }
        if (!step_result.value().executed) {
            fprintf(stderr, "decode stuck: pending request remains but step did not execute\n");
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after stuck decode: %s\n", cleanup_st.message().c_str());
            }
            return 1;
        }
        if (dumper.file != nullptr) {
            herr = hipStreamSynchronize(runtime.stream());
            if (herr != hipSuccess) {
                fprintf(stderr, "decode sync failed: %s\n", hipGetErrorString(herr));
                return 1;
            }
            if (!dumper.step(executor)) return 1;
        }
        if (std::getenv("SHELF_DEBUG_HIDDEN") != nullptr ||
            executor.config.target_hidden_tap_count > 0) {
            herr = hipStreamSynchronize(runtime.stream());
            if (herr != hipSuccess) {
                fprintf(stderr, "decode sync failed: %s\n", hipGetErrorString(herr));
                return 1;
            }
            char tag[32];
            std::snprintf(tag, sizeof(tag), "decode%d", static_cast<int>(decode_tokens));
            dump_hidden_debug(executor, tag);
            dump_tap_hidden(executor, tag, step_result.value().num_tokens);
        }
        decode_tokens += step_result.value().num_tokens;
    }
    herr = hipStreamSynchronize(runtime.stream());
    if (herr != hipSuccess) {
        fprintf(stderr, "decode sync failed: %s\n", hipGetErrorString(herr));
        return 1;
    }
    auto t_decode_end = std::chrono::steady_clock::now();

    const auto* request = runtime.request(request_id);
    if (request == nullptr ||
        request->state != shelf::qwen35::runtime::RequestState::Finished) {
        fprintf(stderr, "request did not finish\n");
        return 1;
    }
    std::vector<int32_t> generated(request->generated);

    const auto prefill_ms =
        std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    const auto decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    const double decode_tps = decode_ms > 0.0 ? decode_tokens * 1000.0 / decode_ms : 0.0;

    printf("MODEL_LAYERS=%zu\n", runtime.model().hybrid_config().num_hidden_layers);
    printf("PROMPT_TOKENS=%zu\n", prompt_ids.size());
    printf("GENERATED_TOKENS=%zu\n", generated.size());
    printf("ARENA_USED_GB=%.3f\n", static_cast<double>(runtime.arena().used()) / (1024.0 * 1024.0 * 1024.0));
    printf("ARENA_COMMITTED_GB=%.3f\n",
           static_cast<double>(runtime.arena().committed()) / (1024.0 * 1024.0 * 1024.0));
    printf("MODEL_LOAD_MS=%.1f\n", runtime.model_load_ms());
    printf("PREFILL_MS=%.1f\n", prefill_ms);
    printf("PREFILL_CHUNKS=%u\n", prefill_chunks);
    printf("DECODE_MS=%.1f\n", decode_ms);
    printf("DECODE_TOKENS=%u\n", decode_tokens);
    printf("DECODE_TOKENS_PER_SEC=%.2f\n", decode_tps);
    printf("KV_CACHE_DTYPE=%s\n", kv_cache_dtype.c_str());
    printf("KV_POOL_RESERVED_BYTES=%zu\n", kv_pool.reserved_bytes());

    printf("GENERATED_IDS=");
    for (std::size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) printf(",");
        printf("%d", generated[i]);
    }
    printf("\n");
    dumper.finish();
    fflush(stdout);

    const uint32_t used_kv_pages = kv_pool.num_used_pages();
    const uint32_t used_seq_slots = runtime.slot_pool().num_used_slots();
    const uint32_t used_gdn_slots = runtime.gdn_pool().num_used_slots();
    if (used_kv_pages != 0 || used_seq_slots != 0 || used_gdn_slots != 0) {
        fprintf(stderr,
                "pool leak after generation: kv_pages=%u seq_slots=%u gdn_slots=%u\n",
                used_kv_pages, used_seq_slots, used_gdn_slots);
        return 1;
    }
    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

int run_dflash2_shot(shelf::app::Qwen35ComputeRuntime& runtime,
                     const std::vector<int32_t>& prompt_ids,
                     unsigned long max_new_tokens,
                     const shelf::qwen35::dflash2::DFlash2Config& dflash_config,
                     const std::string& dflash2_model_dir,
                     unsigned long dflash2_drafts,
                     int dflash2_stats,
                     bool dflash2_tree,
                     unsigned long dflash2_tree_extra,
                     unsigned long dflash2_tree_budget,
                     const std::string& kv_cache_dtype,
                     const shelf::qwen35::runtime::SamplingConfig& sampling) {
    auto& executor = runtime.executor();
    const auto& model = runtime.model();
    auto& slot_pool = runtime.slot_pool();
    auto& gdn_pool = runtime.gdn_pool();
    auto& kv_pool = runtime.kv_pool();
    auto& arena = runtime.arena();
    const hipStream_t stream = runtime.stream();
    const auto& text_config = model.hybrid_config();
    shelf::qwen35::runtime::TpCoordinator* tp_coord = runtime.tp_coordinator();
    std::vector<shelf::qwen35::runtime::DFlash2GdnRankBinding> rank_bindings;
    if (tp_coord != nullptr) {
        rank_bindings.reserve(static_cast<std::size_t>(tp_coord->tp_size()));
        for (int r = 0; r < tp_coord->tp_size(); ++r) {
            auto& rank = tp_coord->rank(static_cast<std::size_t>(r));
            rank_bindings.push_back(shelf::qwen35::runtime::DFlash2GdnRankBinding{
                &rank.gdn_pool(), &rank.arena(), rank.stream(), rank.device_id(),
                &rank.executor().host_model_state, &rank.executor()});
        }
    }
    print_model_geometry(model);
    if (std::getenv("SHELF_MEM_DEBUG") != nullptr) {
        dump_mem_debug(runtime);
    }

    auto dflash_weights_result = runtime.load_dflash2_weights(dflash_config);
    if (!dflash_weights_result.ok()) {
        fprintf(stderr, "dflash2 weight load failed: %s\n",
                dflash_weights_result.status().message().c_str());
        return 1;
    }
    shelf::qwen35::dflash2::DFlash2Weights dflash_weights = dflash_weights_result.release();

    const uint32_t max_seq = runtime.config().max_seq_len;

    shelf::weights::MatrixWeight* private_lm_head = nullptr;
    shelf::weights::MatrixWeight* private_embed = nullptr;
    const bool tp_sharded = runtime.config().tp > 1;
    if (tp_sharded) {
        auto* tp = runtime.tp_coordinator();
        if (tp == nullptr || tp->tp_size() == 0) {
            fprintf(stderr, "private target weights unavailable\n");
            return 1;
        }
        auto& rank0 = tp->rank(0);
        const auto* q4 = rank0.rank_model() != nullptr
                             ? rank0.rank_model()->qwen4exp_model()
                             : nullptr;
        if (q4 != nullptr) {
            private_lm_head = const_cast<shelf::weights::MatrixWeight*>(
                &q4->weights().lm_head);
            private_embed = const_cast<shelf::weights::MatrixWeight*>(
                &q4->weights().embed_tokens);
        } else {
            private_lm_head = &rank0.private_lm_head();
            private_embed = &rank0.private_embed_tokens();
        }
    }
    shelf::qwen35::dflash2::DFlash2ExecutorConfig draft_config;
    draft_config.max_rows = static_cast<uint32_t>(dflash_config.block_size);
    draft_config.max_context_rows = static_cast<uint32_t>(dflash_config.sliding_window);
    draft_config.target_lm_head =
        tp_sharded ? private_lm_head : &model.lm_head();
    draft_config.target_embed_tokens =
        tp_sharded ? private_embed : &model.embed_tokens();
    auto draft_result = shelf::qwen35::dflash2::create_dflash2_executor(
        dflash_config, dflash_weights, arena, draft_config);
    if (!draft_result.ok()) {
        fprintf(stderr, "dflash2 executor create failed: %s\n",
                draft_result.status().message().c_str());
        return 1;
    }
    shelf::qwen35::dflash2::DFlash2Executor draft = draft_result.release();

    auto sequence_result = shelf::state::create_paged_sequence_state(
        slot_pool, gdn_pool, kv_pool, max_seq, stream);
    if (!sequence_result.ok()) {
        fprintf(stderr, "dflash2 sequence create failed: %s\n",
                sequence_result.status().message().c_str());
        (void)shelf::qwen35::dflash2::dflash2_executor_shutdown(draft);
        return 1;
    }
    shelf::state::PagedSequenceState sequence = sequence_result.release();
    auto release_sequence = [&]() {
        if (tp_coord != nullptr) {
            (void)tp_coord->on_sequence_released(sequence);
        }
        (void)shelf::state::release_paged_sequence_state(sequence, stream);
    };
    if (tp_coord != nullptr) {
        const shelf::Status mirror_st = tp_coord->on_sequence_created(sequence);
        if (!mirror_st.ok()) {
            fprintf(stderr, "dflash2 tp mirror create failed: %s\n",
                    mirror_st.message().c_str());
            release_sequence();
            (void)shelf::qwen35::dflash2::dflash2_executor_shutdown(draft);
            return 1;
        }
    }

    auto context_result =
        shelf::qwen35::dflash2::create_dflash2_context_state(dflash_config, arena, 0u);
    if (!context_result.ok()) {
        fprintf(stderr, "dflash2 context create failed: %s\n",
                context_result.status().message().c_str());
        release_sequence();
        (void)shelf::qwen35::dflash2::dflash2_executor_shutdown(draft);
        return 1;
    }
    shelf::qwen35::dflash2::DFlash2ContextState context = context_result.release();

    shelf::qwen35::runtime::DFlash2SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = static_cast<uint32_t>(dflash2_drafts);
    decoder_config.tree = dflash2_tree;
    decoder_config.tree_extra = static_cast<uint32_t>(dflash2_tree_extra);
    decoder_config.tree_budget = static_cast<uint32_t>(dflash2_tree_budget);
    decoder_config.eos_tokens = text_config.stop_tokens;
    auto decoder_result = shelf::qwen35::runtime::create_dflash2_spec_decoder(
        executor, draft, context, sequence, gdn_pool, arena, decoder_config, stream,
        rank_bindings.empty() ? nullptr : &rank_bindings, tp_coord);
    if (!decoder_result.ok()) {
        fprintf(stderr, "dflash2 spec decoder create failed: %s\n",
                decoder_result.status().message().c_str());
        (void)shelf::qwen35::dflash2::dflash2_context_shutdown(context);
        release_sequence();
        (void)shelf::qwen35::dflash2::dflash2_executor_shutdown(draft);
        return 1;
    }
    shelf::qwen35::runtime::DFlash2SpecDecoder decoder = decoder_result.release();
    shelf::qwen35::runtime::DFlash2SpecTiming timing;
    decoder.timing = &timing;
    shelf::qwen35::runtime::dflash2_spec_decoder_set_sampling(decoder, sampling);

    std::vector<int32_t> generated;
    uint32_t decode_tokens = 0u;

    const auto t_prefill_start = std::chrono::steady_clock::now();
    auto prefill_result = shelf::qwen35::runtime::dflash2_spec_prefill(
        decoder, prompt_ids.data(), static_cast<uint32_t>(prompt_ids.size()));
    if (!prefill_result.ok()) {
        fprintf(stderr, "dflash2 prefill failed: %s\n",
                prefill_result.status().message().c_str());
        (void)shelf::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
        (void)shelf::qwen35::dflash2::dflash2_context_shutdown(context);
        release_sequence();
        (void)shelf::qwen35::dflash2::dflash2_executor_shutdown(draft);
        return 1;
    }
    int32_t pending = prefill_result.value().pending_token;
    generated.push_back(pending);
    const auto t_prefill_end = std::chrono::steady_clock::now();

    const auto t_decode_start = std::chrono::steady_clock::now();
    bool finished = shelf::qwen_hybrid::is_stop_token(decoder_config.eos_tokens, pending);
    while (!finished && generated.size() < max_new_tokens) {
        const uint32_t remaining =
            static_cast<uint32_t>(max_new_tokens - generated.size());
        auto step_result =
            dflash2_tree ? shelf::qwen35::runtime::dflash2_spec_step_tree(decoder, pending, remaining)
                         : shelf::qwen35::runtime::dflash2_spec_step(decoder, pending, remaining);
        if (!step_result.ok()) {
            fprintf(stderr, "dflash2 step failed: %s\n",
                    step_result.status().message().c_str());
            (void)shelf::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
            (void)shelf::qwen35::dflash2::dflash2_context_shutdown(context);
            release_sequence();
            (void)shelf::qwen35::dflash2::dflash2_executor_shutdown(draft);
            return 1;
        }
        auto step = step_result.release();
        for (uint32_t i = 0; i < step.emitted_count; ++i) {
            generated.push_back(step.emitted[i]);
            ++decode_tokens;
        }
        pending = step.pending_token;
        finished = step.finished;
    }
    hipError_t herr = hipStreamSynchronize(stream);
    if (herr != hipSuccess) {
        fprintf(stderr, "dflash2 sync failed: %s\n", hipGetErrorString(herr));
        return 1;
    }
    const auto t_decode_end = std::chrono::steady_clock::now();

    if (dflash2_stats != 0) {
        const float mean_accepted = timing.rounds > 0u
            ? static_cast<float>(timing.accepted_drafts) / static_cast<float>(timing.rounds)
            : 0.0f;
        const float full_rate = timing.rounds > 0u
            ? static_cast<float>(timing.full_accepts) / static_cast<float>(timing.rounds)
            : 0.0f;
        const float emitted_per_round = timing.rounds > 0u
            ? static_cast<float>(timing.generated_tokens) / static_cast<float>(timing.rounds)
            : 0.0f;
        printf("DFLASH2_DRAFTS=%u\n", decoder_config.num_drafts);
        printf("DFLASH2_BLOCK_SIZE=%zu\n", dflash_config.block_size);
        printf("DFLASH2_ROUNDS=%u\n", timing.rounds);
        printf("DFLASH2_FULL_ACCEPTS=%u\n", timing.full_accepts);
        printf("DFLASH2_PARTIAL_ACCEPTS=%u\n", timing.partial_accepts);
        printf("DFLASH2_GDN_HISTORY_BYTES=%llu\n",
               static_cast<unsigned long long>(timing.gdn_history_bytes));
        printf("DFLASH2_GDN_SNAPSHOT_BYTES=%llu\n",
               static_cast<unsigned long long>(decoder.gdn_snapshot_bytes));
        printf("DFLASH2_GDN_COMPACT_BYTES=%llu\n",
               static_cast<unsigned long long>(decoder.gdn_compact_bytes));
        printf("DFLASH2_GDN_MODE=%s\n", dflash2_tree ? "tree-history" : "compact");
        printf("DFLASH2_ACCEPTED_DRAFTS=%u\n", timing.accepted_drafts);
        printf("DFLASH2_MEAN_ACCEPTED=%.3f\n", mean_accepted);
        printf("DFLASH2_FULL_ACCEPT_RATE=%.3f\n", full_rate);
        printf("DFLASH2_EMITTED_PER_ROUND=%.3f\n", emitted_per_round);
        printf("DFLASH2_DRAFT_MS=%.2f\n", timing.draft_ms);
        printf("DFLASH2_D2H_MS=%.2f\n", timing.draft_d2h_ms);
        printf("DFLASH2_DRAFT_GPU_MS=%.2f\n", timing.draft_gpu_ms);
        printf("DFLASH2_PROPOSAL_WAIT_MS=%.2f\n", timing.proposal_wait_ms);
        printf("DFLASH2_PROPOSAL_COPY_MS=%.2f\n", timing.proposal_copy_ms);
        printf("DFLASH2_VERIFY_MS=%.2f\n", timing.verify_ms);
        printf("DFLASH2_VERIFY_GPU_MS=%.2f\n", timing.verify_gpu_ms);
        printf("DFLASH2_DECISION_WAIT_MS=%.2f\n", timing.decision_wait_ms);
        printf("DFLASH2_DECISION_D2H_MS=%.2f\n", timing.decision_copy_ms);
        printf("DFLASH2_DEVICE_TOKEN_BRIDGE=%d\n", decoder.device_token_bridge && !dflash2_tree ? 1 : 0);
        printf("DFLASH2_RERUN_MS=%.2f\n", timing.rerun_ms);
        printf("DFLASH2_TREE_SELECT_MS=%.2f\n", timing.tree_select_ms);
        if (timing.tree_probability_rows != 0u)
            printf("DFLASH2_TREE_RESIDUAL_MASS=%.6f\n", 1.0 - timing.tree_retained_mass / timing.tree_probability_rows);
        printf("DFLASH2_TREE_WALK_MS=%.2f\n", timing.tree_walk_ms);
        printf("DFLASH2_TREE_COMMIT_MS=%.2f\n", timing.tree_commit_ms);
        printf("DFLASH2_TREE_RESTORE_MS=%.2f\n", timing.tree_restore_ms);
        printf("DFLASH2_TREE_GATHER_MS=%.2f\n", timing.tree_gather_ms);
        printf("DFLASH2_TREE_APPEND_MS=%.2f\n", timing.tree_append_ms);
        printf("DFLASH2_TREE_REPLAYS=%u\n", timing.tree_replays);
        printf("DFLASH2_GDN_SNAPSHOT_MS=%.2f\n", timing.gdn_snapshot_ms);
        printf("DFLASH2_GDN_RESTORE_MS=%.2f\n", timing.gdn_restore_ms);
        printf("DFLASH2_COMMIT_MS=%.2f\n", timing.dflash_commit_ms);
        printf("DFLASH2_ROUND_MS=%.2f\n", timing.round_ms);
        fflush(stdout);
    }

    (void)shelf::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
    (void)shelf::qwen35::dflash2::dflash2_context_shutdown(context);
    release_sequence();
    (void)shelf::qwen35::dflash2::dflash2_executor_shutdown(draft);

    const auto prefill_ms =
        std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    const auto decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    const double decode_tps = decode_ms > 0.0 ? decode_tokens * 1000.0 / decode_ms : 0.0;

    printf("MODEL_LAYERS=%zu\n", text_config.num_hidden_layers);
    printf("PROMPT_TOKENS=%zu\n", prompt_ids.size());
    printf("GENERATED_TOKENS=%zu\n", generated.size());
    printf("ARENA_USED_GB=%.3f\n", static_cast<double>(arena.used()) / (1024.0 * 1024.0 * 1024.0));
    printf("ARENA_COMMITTED_GB=%.3f\n",
           static_cast<double>(arena.committed()) / (1024.0 * 1024.0 * 1024.0));
    printf("MODEL_LOAD_MS=%.1f\n", runtime.model_load_ms());
    printf("PREFILL_MS=%.1f\n", prefill_ms);
    printf("DECODE_MS=%.1f\n", decode_ms);
    printf("DECODE_TOKENS=%u\n", decode_tokens);
    printf("DECODE_TOKENS_PER_SEC=%.2f\n", decode_tps);
    printf("KV_CACHE_DTYPE=%s\n", kv_cache_dtype.c_str());
    printf("KV_POOL_RESERVED_BYTES=%zu\n", kv_pool.reserved_bytes());

    printf("GENERATED_IDS=");
    for (std::size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) printf(",");
        printf("%d", generated[i]);
    }
    printf("\n");
    fflush(stdout);

    const uint32_t used_kv_pages = kv_pool.num_used_pages();
    const uint32_t used_seq_slots = slot_pool.num_used_slots();
    const uint32_t used_gdn_slots = gdn_pool.num_used_slots();
    if (used_kv_pages != 0 || used_seq_slots != 0 || used_gdn_slots != 0) {
        fprintf(stderr,
                "pool leak after generation: kv_pages=%u seq_slots=%u gdn_slots=%u\n",
                used_kv_pages, used_seq_slots, used_gdn_slots);
        return 1;
    }
    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}


int run_mtp_shot(shelf::app::Qwen35ComputeRuntime& runtime,
                 const std::vector<int32_t>& prompt_ids,
                 unsigned long max_new_tokens,
                 uint32_t mtp_drafts,
                 int mtp_stats,
                 const std::string& kv_cache_dtype,
                 const shelf::qwen35::runtime::SamplingConfig& sampling) {
    auto& executor = runtime.executor();
    auto& slot_pool = runtime.slot_pool();
    auto& gdn_pool = runtime.gdn_pool();
    auto& kv_pool = runtime.kv_pool();
    auto& arena = runtime.arena();
    const hipStream_t stream = runtime.stream();
    const auto& text_config = runtime.model().hybrid_config();
    print_model_geometry(runtime.model());
    if (std::getenv("SHELF_MEM_DEBUG") != nullptr) {
        dump_mem_debug(runtime);
    }

    const uint32_t max_seq = runtime.config().max_seq_len;
    auto sequence_result = shelf::state::create_paged_sequence_state(
        slot_pool, gdn_pool, kv_pool, max_seq, stream);
    if (!sequence_result.ok()) {
        fprintf(stderr, "mtp target sequence create failed: %s\n",
                sequence_result.status().message().c_str());
        return 1;
    }
    shelf::state::PagedSequenceState sequence = sequence_result.release();

    shelf::qwen35::runtime::MtpDrafterConfig drafter_config;
    drafter_config.num_drafts = mtp_drafts;
    drafter_config.max_seq_len = max_seq;
    drafter_config.page_tokens = runtime.config().page_tokens;
    drafter_config.max_batch_rows = std::min(
        executor.config.max_batch_rows,
        shelf::qwen35::runtime::kMtpMaxPrefillRows);
    drafter_config.kv_dtype =
        runtime.config().kv_cache_dtype == shelf::kernel::KVCacheDType::OCTAVE
            ? shelf::kernel::KVCacheDType::BF16 : runtime.config().kv_cache_dtype;
    if (runtime.config().rope_profile.has_value()) {
        drafter_config.rope_profile = *runtime.config().rope_profile;
    }
    const shelf::weights::MatrixWeight* lm_head_override = nullptr;
    if (auto* tp = runtime.tp_coordinator(); tp != nullptr && tp->tp_size() > 0) {
        auto& private_lm_head = tp->rank(0).private_lm_head();
        if (private_lm_head.rows != 0u) lm_head_override = &private_lm_head;
    }
    auto drafter_input =
        runtime.model().mtp_drafter_input(drafter_config, lm_head_override);
    if (!drafter_input.ok()) {
        fprintf(stderr, "mtp drafter input failed: %s\n",
                drafter_input.status().message().c_str());
        (void)shelf::state::release_paged_sequence_state(sequence, stream);
        return 1;
    }
    std::vector<shelf::qwen35::runtime::MtpGdnRankBinding> extra_ranks;
    auto* tp = runtime.tp_coordinator();
    if (tp != nullptr) {
        for (int r = 1; r < tp->tp_size(); ++r) {
            auto& rank = tp->rank(static_cast<std::size_t>(r));
            shelf::qwen35::runtime::MtpGdnRankBinding binding;
            binding.pool = &rank.gdn_pool();
            binding.arena = &rank.arena();
            binding.stream = rank.stream();
            binding.device = rank.device_id();
            binding.executor = &rank.executor();
            extra_ranks.push_back(binding);
        }
    }
    auto drafter_result = shelf::qwen35::runtime::create_mtp_drafter(
        drafter_input.value(), &executor, gdn_pool, arena, stream, drafter_config,
        extra_ranks.empty() ? nullptr : &extra_ranks, tp);
    if (!drafter_result.ok()) {
        fprintf(stderr, "mtp drafter create failed: %s\n",
                drafter_result.status().message().c_str());
        (void)shelf::state::release_paged_sequence_state(sequence, stream);
        return 1;
    }
    std::unique_ptr<shelf::qwen35::runtime::MtpDrafter> drafter =
        drafter_result.release();
    if (tp != nullptr) {
        Status mirrored = tp->on_sequence_created(sequence);
        if (!mirrored.ok()) {
            fprintf(stderr, "mtp tp mirror create failed: %s\n",
                    mirrored.message().c_str());
            (void)shelf::qwen35::runtime::mtp_drafter_shutdown(*drafter, stream);
            drafter.reset();
            (void)shelf::state::release_paged_sequence_state(sequence, stream);
            return 1;
        }
    }
    auto cleanup = [&]() {
        (void)shelf::qwen35::runtime::mtp_drafter_shutdown(*drafter, stream);
        drafter.reset();
        if (tp != nullptr) (void)tp->on_sequence_released(sequence);
        (void)shelf::state::release_paged_sequence_state(sequence, stream);
    };

    const auto t_prefill_start = std::chrono::steady_clock::now();
    auto prefill_result = shelf::qwen35::runtime::mtp_spec_prefill(
        *drafter, executor, sequence, prompt_ids.data(),
        static_cast<uint32_t>(prompt_ids.size()), sampling);
    if (!prefill_result.ok()) {
        fprintf(stderr, "mtp prefill failed: %s\n",
                prefill_result.status().message().c_str());
        cleanup();
        return 1;
    }
    int32_t pending = prefill_result.value().pending_token;
    std::vector<int32_t> generated;
    generated.push_back(pending);
    uint32_t decode_tokens = 0u;
    const auto t_prefill_end = std::chrono::steady_clock::now();

    const auto t_decode_start = std::chrono::steady_clock::now();
    bool finished = shelf::qwen_hybrid::is_stop_token(text_config.stop_tokens, pending);
    while (!finished && generated.size() < max_new_tokens) {
        const uint32_t remaining =
            static_cast<uint32_t>(max_new_tokens - generated.size());
        auto step_result = shelf::qwen35::runtime::mtp_spec_step(
            *drafter, executor, sequence, pending, remaining);
        if (!step_result.ok()) {
            fprintf(stderr, "mtp step failed: %s\n",
                    step_result.status().message().c_str());
            cleanup();
            return 1;
        }
        auto step = step_result.release();
        for (uint32_t i = 0u; i < step.emitted_count; ++i) {
            generated.push_back(step.emitted[i]);
            ++decode_tokens;
        }
        pending = step.pending_token;
        finished = step.finished;
    }
    const hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        fprintf(stderr, "mtp sync failed: %s\n", hipGetErrorString(sync_err));
        cleanup();
        return 1;
    }
    const auto t_decode_end = std::chrono::steady_clock::now();

    if (mtp_stats != 0) {
        const auto& timing = drafter->timing;
        const float mean_accepted = timing.rounds > 0u
            ? static_cast<float>(timing.accepted_drafts) /
                  static_cast<float>(timing.rounds)
            : 0.0f;
        const float full_rate = timing.rounds > 0u
            ? static_cast<float>(timing.full_accepts) /
                  static_cast<float>(timing.rounds)
            : 0.0f;
        const float emitted_per_round = timing.rounds > 0u
            ? static_cast<float>(timing.generated_tokens) /
                  static_cast<float>(timing.rounds)
            : 0.0f;
        printf("MTP_DRAFTS=%u\n", drafter_config.num_drafts);
        printf("MTP_ROUNDS=%u\n", timing.rounds);
        printf("MTP_FULL_ACCEPTS=%u\n", timing.full_accepts);
        printf("MTP_PARTIAL_ACCEPTS=%u\n", timing.partial_accepts);
        printf("MTP_ACCEPTED_DRAFTS=%u\n", timing.accepted_drafts);
        printf("MTP_MEAN_ACCEPTED=%.3f\n", mean_accepted);
        printf("MTP_FULL_ACCEPT_RATE=%.3f\n", full_rate);
        printf("MTP_EMITTED_PER_ROUND=%.3f\n", emitted_per_round);
        printf("MTP_VERIFY_ROWS=%u\n", timing.verify_rows);
        printf("MTP_GDN_RESTORES=%u\n", timing.gdn_restores);
        printf("MTP_DRAFT_MS=%.2f\n", timing.draft_ms);
        printf("MTP_VERIFY_MS=%.2f\n", timing.verify_ms);
        printf("MTP_GDN_RESTORE_MS=%.2f\n", timing.gdn_restore_ms);
        printf("MTP_COMPACT_REPLAY_MS=%.2f\n", timing.compact_replay_ms);
        printf("MTP_ROUND_MS=%.2f\n", timing.round_ms);
        fflush(stdout);
    }

    cleanup();

    const auto prefill_ms =
        std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    const auto decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    const double decode_tps =
        decode_ms > 0.0 ? decode_tokens * 1000.0 / decode_ms : 0.0;

    printf("PROMPT_TOKENS=%zu\n", prompt_ids.size());
    printf("GENERATED_TOKENS=%zu\n", generated.size());
    printf("ARENA_USED_GB=%.3f\n",
           static_cast<double>(arena.used()) / (1024.0 * 1024.0 * 1024.0));
    printf("ARENA_COMMITTED_GB=%.3f\n",
           static_cast<double>(arena.committed()) / (1024.0 * 1024.0 * 1024.0));
    printf("MODEL_LOAD_MS=%.1f\n", runtime.model_load_ms());
    printf("PREFILL_MS=%.1f\n", prefill_ms);
    printf("DECODE_MS=%.1f\n", decode_ms);
    printf("DECODE_TOKENS=%u\n", decode_tokens);
    printf("DECODE_TOKENS_PER_SEC=%.2f\n", decode_tps);
    printf("KV_CACHE_DTYPE=%s\n", kv_cache_dtype.c_str());
    printf("KV_POOL_RESERVED_BYTES=%zu\n", kv_pool.reserved_bytes());

    printf("GENERATED_IDS=");
    for (std::size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) printf(",");
        printf("%d", generated[i]);
    }
    printf("\n");
    fflush(stdout);

    const uint32_t used_kv_pages = kv_pool.num_used_pages();
    const uint32_t used_seq_slots = slot_pool.num_used_slots();
    const uint32_t used_gdn_slots = gdn_pool.num_used_slots();
    if (used_kv_pages != 0 || used_seq_slots != 0 || used_gdn_slots != 0) {
        fprintf(stderr,
                "pool leak after generation: kv_pages=%u seq_slots=%u gdn_slots=%u\n",
                used_kv_pages, used_seq_slots, used_gdn_slots);
        return 1;
    }
    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

}  // namespace shelf::app
