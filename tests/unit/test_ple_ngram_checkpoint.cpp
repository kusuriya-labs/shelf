#include <phaseshift/models/qwen4exp/ple/ngram_hash.h>
#include <phaseshift/weights/weight_loader.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ple = ps::qwen4exp::ple;

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

static const char* kPrefix =
    "model.language_model.layers.1.ple.ple_embedding.";

static bool read_i64(const ps::weights::SafetensorsCollection& col, const std::string& suffix,
                     std::vector<std::int64_t>& out, std::string& shard) {
    const std::string name = std::string(kPrefix) + suffix;
    auto sh = col.tensor_shard(name);
    if (sh.ok()) shard = sh.value();
    std::size_t bytes = 0;
    auto data = col.tensor_data(name, bytes);
    if (!data.ok()) {
        printf("  tensor_data failed for %s: %s\n", name.c_str(),
               data.status().message().c_str());
        return false;
    }
    if (bytes % sizeof(std::int64_t) != 0u) {
        printf("  tensor %s has %zu bytes, not I64 aligned\n", name.c_str(), bytes);
        return false;
    }
    out.resize(bytes / sizeof(std::int64_t));
    std::memcpy(out.data(), data.value(), bytes);
    return true;
}

struct PatternResult {
    bool sizes = false;
    bool offsets = false;
    bool multipliers = false;
};

static PatternResult evaluate(const ple::NgramHashParams& params,
                              const std::vector<std::int64_t>& actual_sizes,
                              const std::vector<std::int64_t>& actual_offsets,
                              const std::vector<std::int64_t>& actual_mults) {
    ple::NgramIdTables tables;
    const ps::Status st = ple::build_ngram_id_tables(params, tables);
    if (!st.ok()) {
        printf("  build failed: %s\n", st.message().c_str());
        return {};
    }
    PatternResult r;
    r.sizes = actual_sizes.size() == ple::kNgramHeads;
    r.offsets = actual_offsets.size() == ple::kNgramHeads;
    r.multipliers = actual_mults.size() == ple::kNgramSize;
    for (std::uint32_t h = 0; r.sizes && h < ple::kNgramHeads; ++h) {
        if (tables.vocab_sizes[h] != actual_sizes[h]) r.sizes = false;
    }
    for (std::uint32_t h = 0; r.offsets && h < ple::kNgramHeads; ++h) {
        if (tables.offsets[h] != actual_offsets[h]) r.offsets = false;
    }
    for (std::uint32_t k = 0; r.multipliers && k < ple::kNgramSize; ++k) {
        if (tables.multipliers[k] != actual_mults[k]) r.multipliers = false;
    }
    return r;
}

static void print_values(const char* name, const std::vector<std::int64_t>& v) {
    printf("  %s:", name);
    for (std::int64_t x : v) printf(" %lld", static_cast<long long>(x));
    printf("\n");
}

int main() {
    printf("=== PLE n-gram hash vs real checkpoint I64 tensors ===\n");
    const char* env = std::getenv("PHASESHIFT_QWEN4EXP_MODEL_DIR");
    const std::string dir = (env != nullptr && env[0] != '\0')
                                ? env
                                : "/opt/zen/work/models/Qwen3.8-Flash-Next";
    if (!fs::exists(dir)) {
        printf("SKIP: model dir not found: %s\n", dir.c_str());
        return 77;
    }

    auto col = ps::weights::SafetensorsCollection::open(dir);
    if (!col.ok()) {
        printf("FAIL: open collection: %s\n", col.status().message().c_str());
        return 1;
    }

    std::vector<std::int64_t> actual_sizes;
    std::vector<std::int64_t> actual_offsets;
    std::vector<std::int64_t> actual_mults;
    std::string shard_sizes;
    std::string shard_offsets;
    std::string shard_mults;
    if (!read_i64(col.value(), "ngram_heads_vocab_sizes", actual_sizes, shard_sizes) ||
        !read_i64(col.value(), "ngram_heads_offsets", actual_offsets, shard_offsets) ||
        !read_i64(col.value(), "layer_multipliers", actual_mults, shard_mults)) {
        printf("FAIL: cannot read I64 tensors\n");
        return 1;
    }

    printf("  vocab_sizes shard: %s\n", shard_sizes.c_str());
    printf("  offsets shard: %s\n", shard_offsets.c_str());
    printf("  multipliers shard: %s\n", shard_mults.c_str());
    print_values("actual vocab_sizes", actual_sizes);
    print_values("actual offsets", actual_offsets);
    print_values("actual multipliers", actual_mults);

    check(actual_sizes.size() == ple::kNgramHeads, "actual: 16 vocab sizes");
    check(actual_offsets.size() == ple::kNgramHeads, "actual: 16 offsets");
    check(actual_mults.size() == ple::kNgramSize, "actual: 3 multipliers");

    bool cumulative = true;
    std::int64_t acc = 0;
    for (std::uint32_t h = 0; h < actual_sizes.size() && h < ple::kNgramHeads; ++h) {
        if (actual_offsets[h] != acc) cumulative = false;
        acc += actual_sizes[h];
    }
    check(cumulative, "actual: offsets are cumulative sums of sizes");
    printf("  total rows from actual tables: %lld\n", static_cast<long long>(acc));
    check(acc <= static_cast<std::int64_t>(128ull * 2500012ull),
          "actual: rows fit 128 shards of 2500012");

    struct Named {
        const char* name;
        ple::NgramHashParams params;
    };
    Named patterns[4];
    patterns[0].name = "slot=0 unigram=248320 (docs-corrected interpretation)";
    patterns[0].params.ple_layer_slot = 0;
    patterns[0].params.unigram_vocab_size = 248320;
    patterns[1].name = "slot=1 unigram=248320 (docs layer index reading)";
    patterns[1].params.ple_layer_slot = 1;
    patterns[1].params.unigram_vocab_size = 248320;
    patterns[2].name = "slot=0 unigram=20000000 (ngram_vocab_size_base reading)";
    patterns[2].params.ple_layer_slot = 0;
    patterns[2].params.unigram_vocab_size = 20000000;
    patterns[3].name = "slot=1 unigram=20000000 (both docs readings)";
    patterns[3].params.ple_layer_slot = 1;
    patterns[3].params.unigram_vocab_size = 20000000;

    PatternResult results[4];
    for (int i = 0; i < 4; ++i) {
        results[i] = evaluate(patterns[i].params, actual_sizes, actual_offsets, actual_mults);
        printf("  pattern %d [%s]: sizes=%s offsets=%s multipliers=%s\n", i, patterns[i].name,
               results[i].sizes ? "MATCH" : "mismatch",
               results[i].offsets ? "MATCH" : "mismatch",
               results[i].multipliers ? "MATCH" : "mismatch");
    }

    check(results[0].sizes && results[0].offsets && results[0].multipliers,
          "checkpoint: slot=0 + vocab_size reproduces all three tensors");
    check(!results[1].sizes || !results[1].multipliers,
          "checkpoint: layer-index reading does not reproduce");
    check(!results[2].multipliers,
          "checkpoint: ngram_vocab_size_base reading does not reproduce multipliers");
    check(!results[3].sizes || !results[3].multipliers,
          "checkpoint: combined docs reading does not reproduce");

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
