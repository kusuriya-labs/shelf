#include <shelf/models/qwen4exp/model/qwen4exp_config.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

namespace fs = std::filesystem;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        ++passed;
        printf("PASS: %s\n", msg);
    } else {
        ++failed;
        printf("FAIL: %s\n", msg);
    }
}

static std::string g_dir;

static void write_json(const std::string& name, const std::string& body) {
    std::ofstream f(g_dir + "/" + name, std::ios::trunc);
    f << body;
}

static const char* kFullTextConfig =
    R"({"model_type":"qwen4_exp","architectures":["Qwen4ExpForConditionalGeneration"],)"
    R"("text_config":{)"
    R"("vocab_size":248320,"hidden_size":2560,"num_hidden_layers":48,)"
    R"("num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,)"
    R"("full_attention_interval":4,)"
    R"("linear_num_key_heads":16,"linear_num_value_heads":48,)"
    R"("linear_key_head_dim":128,"linear_value_head_dim":128,)"
    R"("linear_conv_kernel_dim":4,)"
    R"("num_experts":512,"num_experts_per_tok":10,)"
    R"("moe_intermediate_size":640,"shared_expert_intermediate_size":640,)"
    R"("hc_count":4,"hc_lowrank":320,)"
    R"("ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,)"
    R"("make_ngram_vocab_size_divisible_by":128,"split_ngram_parts":128,)"
    R"("seed":1234,"ple_layer_ids":[2],"ple_embed_dim":2560,)"
    R"("ple_conv_kernel_size":4,)"
    R"("indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,)"
    R"("indexer_budget":2048,"indexer_compress_ratio":4,)"
    R"("rms_norm_eps":1e-6,"output_gate_type":"sigmoid","tie_word_embeddings":false,)"
    R"("rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,)"
    R"("mrope_section":[11,11,10],"mrope_interleaved":true,"rope_type":"default"},)"
    R"("eos_token_id":248044,)"
    R"("layer_types":[)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention",)"
    R"("linear_attention","linear_attention","linear_attention","full_attention"]}})";

static void test_full_config() {
    write_json("config.json", kFullTextConfig);
    auto cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(cfg.ok(), "config: parses");
    if (!cfg.ok()) {
        printf("  reason: %s\n", cfg.status().message().c_str());
        return;
    }
    const auto& c = cfg.value();
    check(c.vocab_size == 248320, "config: vocab_size");
    check(c.hidden_size == 2560, "config: hidden_size");
    check(c.num_hidden_layers == 48, "config: num_hidden_layers");
    check(c.num_attention_heads == 24 && c.num_key_value_heads == 2, "config: qsa heads");
    check(c.head_dim == 256, "config: head_dim");
    check(c.rotary_dim == 64, "config: rotary_dim = head_dim * 0.25");
    check(c.full_attention_interval == 4, "config: full_attention_interval");
    check(c.linear_num_key_heads == 16 && c.linear_num_value_heads == 48,
          "config: gdn heads");
    check(c.linear_conv_kernel_dim == 4, "config: gdn conv kernel");
    check(c.num_experts == 512 && c.num_experts_per_tok == 10, "config: moe");
    check(c.moe_intermediate_size == 640 && c.shared_expert_intermediate_size == 640,
          "config: moe intermediate");
    check(c.hc_count == 4 && c.hc_lowrank == 320, "config: hyper connection");
    check(c.ngram_size == 3 && c.heads_per_ngram == 8, "config: ngram");
    check(c.ngram_seed == 1234, "config: seed");
    check(c.ple_layer_index == 1, "config: ple layer index is 0-indexed (1)");
    check(c.ple_slot == 0, "config: ple slot inside ple_layer_ids is 0");
    check(c.ple_embed_dim == 2560 && c.ple_conv_kernel_size == 4, "config: ple dims");
    check(c.indexer_n_heads == 4 && c.indexer_kv_heads == 1, "config: indexer heads");
    check(c.indexer_head_dim == 128, "config: indexer head dim");
    check(c.indexer_budget == 2048 && c.indexer_compress_ratio == 4,
          "config: indexer budget");
    check(c.block_topk == 512, "config: block_topk = 2048 / 4");
    check(c.output_gate_silu == false, "config: sigmoid gate -> output_gate_silu false");
    check(c.norm_topk_prob == true, "config: norm_topk_prob default true");
    check(c.tie_word_embeddings == false, "config: not tied");
    check(c.rope_theta == 10000000.0f, "config: rope_theta");
    check(c.partial_rotary_factor == 0.25f, "config: partial_rotary_factor");
    check(c.stop_tokens.size() == 1 && c.stop_tokens[0] == 248044, "config: stop token");
    check(c.layer_types.size() == 48, "config: 48 layer types");

    std::size_t qsa = 0;
    std::size_t gdn = 0;
    for (std::size_t i = 0; i < 48; ++i) {
        if (c.layer_is_qsa(i)) ++qsa;
        else ++gdn;
    }
    check(qsa == 12 && gdn == 36, "config: 12 QSA / 36 GDN layers");
    check(c.layer_is_qsa(3) && c.layer_is_qsa(47), "config: layer 3 and 47 are QSA");
    check(!c.layer_is_qsa(0) && !c.layer_is_qsa(1), "config: layer 0 and 1 are GDN");
}

static void test_indexer_contract() {
    write_json("config.json",
               R"({"text_config":{"vocab_size":16,"hidden_size":64,)"
               R"("num_hidden_layers":4,"num_experts":8,"num_experts_per_tok":2,)"
               R"("moe_intermediate_size":32,"hc_count":4,"hc_lowrank":16,)"
               R"("indexer_n_heads":4,"indexer_kv_heads":2,)"
               R"("indexer_budget":2048,"indexer_compress_ratio":4}})");
    auto cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(!cfg.ok(), "config: indexer_kv_heads != 1 rejected");

    write_json("config.json",
               R"({"text_config":{"vocab_size":16,"hidden_size":64,)"
               R"("num_hidden_layers":4,"num_experts":8,"num_experts_per_tok":2,)"
               R"("moe_intermediate_size":32,"hc_count":4,"hc_lowrank":16,)"
               R"("indexer_n_heads":4,"indexer_kv_heads":1,)"
               R"("indexer_budget":2050,"indexer_compress_ratio":4}})");
    cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(!cfg.ok(), "config: indivisible indexer_budget rejected");

    write_json("config.json",
               R"({"text_config":{"vocab_size":16,"hidden_size":64,)"
               R"("num_hidden_layers":4,"num_experts":8,"num_experts_per_tok":2,)"
               R"("moe_intermediate_size":32,"hc_count":4,"hc_lowrank":16,)"
               R"("output_gate_type":"tanh"}})");
    cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(!cfg.ok(), "config: unknown output_gate_type rejected");
}

static void test_missing_and_mismatch() {
    write_json("config.json", R"({"model_type":"qwen4_exp"})");
    auto cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(!cfg.ok(), "config: missing text_config rejected");

    write_json("config.json", R"({"text_config":{"vocab_size":16}})");
    cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(!cfg.ok(), "config: required dimensions missing rejected");

    write_json("config.json",
               R"({"text_config":{"vocab_size":16,"hidden_size":64,)"
               R"("num_hidden_layers":4,"num_experts":8,"num_experts_per_tok":2,)"
               R"("moe_intermediate_size":32,"hc_count":4,"hc_lowrank":16,)"
               R"("layer_types":["linear_attention","full_attention"]}})");
    cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(!cfg.ok(), "config: layer_types length mismatch rejected");

    fs::remove(g_dir + "/config.json");
    cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(!cfg.ok(), "config: absent file rejected");
}

static void test_seed_default() {
    write_json("config.json",
               R"({"text_config":{"vocab_size":16,"hidden_size":64,)"
               R"("num_hidden_layers":4,"num_experts":8,"num_experts_per_tok":2,)"
               R"("moe_intermediate_size":32,"hc_count":4,"hc_lowrank":16}})");
    auto cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(cfg.ok(), "seed: config without seed parses");
    if (cfg.ok()) {
        check(cfg.value().ngram_seed == 1234,
              "seed: absent seed falls back to the documented default 1234");
        check(cfg.value().ple_slot == 0, "seed: ple slot default 0");
    }
}

static void test_layer_types_fallback() {
    write_json("config.json",
               R"({"text_config":{"vocab_size":16,"hidden_size":64,)"
               R"("num_hidden_layers":4,"num_experts":8,"num_experts_per_tok":2,)"
               R"("moe_intermediate_size":32,"hc_count":4,"hc_lowrank":16,)"
               R"("full_attention_interval":4}})");
    auto cfg = ps::qwen4exp::read_qwen4exp_text_config(g_dir);
    check(cfg.ok(), "config: layer_types-free config parses");
    if (!cfg.ok()) return;
    const auto& c = cfg.value();
    check(c.layer_types.empty(), "config: layer_types empty");
    check(c.layer_is_qsa(3), "config: interval fallback marks layer 3 QSA");
    check(!c.layer_is_qsa(0), "config: interval fallback marks layer 0 GDN");
}

int main() {
    printf("=== Qwen4Exp config ===\n");
    g_dir = "/tmp/ps_q4e_config_" + std::to_string(getpid());
    fs::remove_all(g_dir);
    fs::create_directories(g_dir);

    test_full_config();
    test_indexer_contract();
    test_missing_and_mismatch();
    test_seed_default();
    test_layer_types_fallback();

    fs::remove_all(g_dir);
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
