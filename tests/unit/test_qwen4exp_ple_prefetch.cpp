#include <shelf/inference/scheduling/scheduled_batch.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/runtime/ple_input.h>
#include <shelf/models/qwen4exp/weights/ple_table.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

using shelf::bf16_t;
using shelf::qwen4exp::PleInputRuntime;
using shelf::qwen4exp::PleTable;
using shelf::qwen4exp::Qwen4ExpTextConfig;
using shelf::scheduling::ScheduledBatch;
using shelf::scheduling::ScheduledRequest;
using shelf::scheduling::TokenIdsLocation;
using shelf::state::PagedSequenceState;

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) {
        std::fprintf(stderr, "ok: %s\n", what);
    } else {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

struct TokenGen {
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    int32_t vocab_size = 0;
    int32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<int32_t>((state >> 33) % static_cast<std::uint64_t>(vocab_size));
    }
};

std::vector<int32_t> make_tokens(int len, int32_t vocab, int32_t eos,
                                 const std::vector<int>& eos_positions) {
    TokenGen gen;
    gen.vocab_size = vocab;
    std::vector<int32_t> tokens(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) tokens[static_cast<std::size_t>(i)] = gen.next();
    for (int pos : eos_positions) {
        if (pos >= 0 && pos < len) tokens[static_cast<std::size_t>(pos)] = eos;
    }
    return tokens;
}

std::vector<bf16_t> baseline_prepare(const std::shared_ptr<const PleTable>& table,
                                     std::uint32_t hidden,
                                     const std::vector<int32_t>& tokens,
                                     const std::vector<std::uint32_t>& chunk_sizes) {
    auto runtime = PleInputRuntime::create(table, hidden);
    if (runtime == nullptr) return {};
    PagedSequenceState seq;
    seq.slot = 0;
    seq.position = 0;
    std::vector<bf16_t> out;
    out.reserve(tokens.size() * hidden);
    std::uint32_t offset = 0;
    for (std::uint32_t n : chunk_sizes) {
        seq.position = offset;
        ScheduledRequest req;
        req.sequence = &seq;
        req.token_begin = offset;
        req.num_tokens = n;
        req.prefix_tokens = offset;
        std::vector<ScheduledRequest> reqs = {req};
        ScheduledBatch batch;
        batch.token_ids = tokens.data();
        batch.token_ids_location = TokenIdsLocation::Host;
        batch.requests = reqs;
        batch.num_tokens = n;
        batch.num_requests = 1;
        batch.num_prefill_requests = 1;
        std::vector<bf16_t> chunk_out;
        auto st = runtime->prepare(batch, chunk_out);
        if (!st.ok()) {
            std::fprintf(stderr, "baseline prepare failed: %s\n", st.message().c_str());
            return {};
        }
        out.insert(out.end(), chunk_out.begin(), chunk_out.end());
        offset += n;
    }
    return out;
}

std::vector<bf16_t> prefetch_stateless(const PleTable& table,
                                       const std::vector<int32_t>& tokens,
                                       const std::vector<std::uint32_t>& chunk_sizes,
                                       std::uint32_t ngram_size) {
    std::vector<bf16_t> out;
    out.reserve(tokens.size() * table.head_count() * table.row_width());
    std::vector<int32_t> prefix_tail;
    std::uint32_t offset = 0;
    for (std::uint32_t n : chunk_sizes) {
        std::span<const int32_t> chunk(tokens.data() + offset, n);
        auto rows = table.gather_bf16_stateless(prefix_tail, chunk);
        if (!rows.ok()) {
            std::fprintf(stderr, "prefetch stateless failed: %s\n",
                         rows.status().message().c_str());
            return {};
        }
        out.insert(out.end(), rows.value().begin(), rows.value().end());
        for (std::uint32_t i = 0; i < n; ++i) {
            prefix_tail.push_back(tokens[static_cast<std::size_t>(offset + i)]);
            while (prefix_tail.size() > ngram_size - 1u) prefix_tail.erase(prefix_tail.begin());
        }
        offset += n;
    }
    return out;
}

bool compare(const char* label, const std::vector<bf16_t>& a, const std::vector<bf16_t>& b) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "FAIL: %s size mismatch (%zu vs %zu)\n", label, a.size(), b.size());
        return false;
    }
    if (a.empty()) {
        std::fprintf(stderr, "ok: %s (empty)\n", label);
        return true;
    }
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(bf16_t)) != 0) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i].data != b[i].data) {
                std::fprintf(stderr,
                             "FAIL: %s first mismatch at element %zu (0x%04x vs 0x%04x)\n",
                             label, i, a[i].data, b[i].data);
                break;
            }
        }
        return false;
    }
    std::fprintf(stderr, "ok: %s (%zu elements bit-identical)\n", label, a.size());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir =
        "/opt/zen/work/shelf/models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) model_dir = argv[++i];
    }
    if (const char* env = std::getenv("SHELF_QWEN4EXP_MODEL_DIR")) model_dir = env;

    auto config_result = shelf::qwen4exp::read_qwen4exp_text_config(model_dir);
    if (!config_result.ok()) {
        std::fprintf(stderr, "SKIP: config not readable at %s: %s\n", model_dir.c_str(),
                     config_result.status().message().c_str());
        return 77;
    }
    const Qwen4ExpTextConfig& config = config_result.value();
    const std::uint32_t hidden = static_cast<std::uint32_t>(config.hidden_size);
    const std::uint32_t ngram_size = static_cast<std::uint32_t>(config.ple.ngram_size);
    const int32_t eos = config.ple.eos_token_id;
    const int32_t vocab = static_cast<int32_t>(config.vocab_size);

    auto ple_result = PleTable::load(model_dir, config, true);
    if (!ple_result.ok()) {
        std::fprintf(stderr, "SKIP: PLE table not loadable at %s: %s\n", model_dir.c_str(),
                     ple_result.status().message().c_str());
        return 77;
    }
    auto table = std::make_shared<const PleTable>(ple_result.release());
    std::fprintf(stderr, "PLE loaded: rows=%llu hidden=%u ngram=%u eos=%d vocab=%d\n",
                 static_cast<unsigned long long>(table->rows()), hidden, ngram_size, eos, vocab);

    {
        const int len = 4096;
        auto tokens = make_tokens(len, vocab, eos, {500, 1500, 2500, 3500});
        std::vector<std::uint32_t> chunks = {1, 2, 16, 256, 1024, 2048, 749};
        auto base = baseline_prepare(table, hidden, tokens, chunks);
        auto pref = prefetch_stateless(*table, tokens, chunks, ngram_size);
        check(compare("mixed-chunk prefetch==baseline", base, pref), "mixed-chunk");
    }
    {
        const int len = 2048;
        auto tokens = make_tokens(len, vocab, eos, {1000});
        std::vector<std::uint32_t> single = {static_cast<std::uint32_t>(len)};
        auto base = baseline_prepare(table, hidden, tokens, single);
        std::vector<std::uint32_t> sub = {1, 2, 16, 256, 1024, 749};
        auto pref = prefetch_stateless(*table, tokens, sub, ngram_size);
        check(compare("single-request==chunked-prefetch", base, pref), "single-vs-chunked");
    }
    {
        const int len = 5;
        auto tokens = make_tokens(len, vocab, eos, {});
        std::vector<std::uint32_t> chunks(static_cast<std::size_t>(len), 1u);
        auto base = baseline_prepare(table, hidden, tokens, chunks);
        auto pref = prefetch_stateless(*table, tokens, chunks, ngram_size);
        check(compare("short per-token prefetch==baseline", base, pref), "short");
    }
    {
        std::vector<int32_t> tokens = {100, 200, eos, 300, 400, 500, 600, 700};
        std::vector<std::uint32_t> chunks = {3, 3, 2};
        auto base = baseline_prepare(table, hidden, tokens, chunks);
        auto pref = prefetch_stateless(*table, tokens, chunks, ngram_size);
        check(compare("eos-in-tail prefetch==baseline", base, pref), "eos-tail");
    }
    {
        TokenGen gen;
        gen.vocab_size = vocab;
        const int rounds = 12;
        std::vector<std::vector<int32_t>> batches;
        std::vector<std::uint32_t> commits;
        for (int r = 0; r < rounds; ++r) {
            const std::uint32_t drafts = 1u + static_cast<std::uint32_t>(r % 6);
            std::vector<int32_t> batch(drafts + 1u);
            for (auto& t : batch) t = gen.next();
            const std::uint32_t accept = 1u + static_cast<std::uint32_t>(r % (drafts + 1u));
            batches.push_back(std::move(batch));
            commits.push_back(accept);
        }

        auto runtime = PleInputRuntime::create(table, hidden);
        PagedSequenceState seq;
        seq.slot = 3;
        seq.position = 0;
        std::vector<bf16_t> base;
        std::uint32_t pos = 0;
        for (std::size_t r = 0; r < batches.size(); ++r) {
            seq.position = pos;
            ScheduledRequest req;
            req.sequence = &seq;
            req.token_begin = 0;
            req.num_tokens = static_cast<std::uint32_t>(batches[r].size());
            req.prefix_tokens = pos;
            std::vector<ScheduledRequest> reqs = {req};
            ScheduledBatch batch;
            batch.token_ids = batches[r].data();
            batch.token_ids_location = TokenIdsLocation::Host;
            batch.requests = reqs;
            batch.num_tokens = static_cast<std::uint32_t>(batches[r].size());
            batch.num_requests = 1;
            batch.num_prefill_requests = 1;
            std::vector<bf16_t> out;
            auto st = runtime->prepare(batch, out);
            if (!st.ok()) {
                std::fprintf(stderr, "decode baseline prepare failed: %s\n",
                             st.message().c_str());
                ++failures;
                break;
            }
            base.insert(base.end(), out.begin(), out.end());
            pos += commits[r];
        }

        std::vector<bf16_t> pref;
        std::vector<int32_t> tail;
        for (std::size_t r = 0; r < batches.size(); ++r) {
            std::span<const int32_t> batch(batches[r]);
            auto rows = table->gather_bf16_stateless(tail, batch);
            if (!rows.ok()) {
                std::fprintf(stderr, "decode prefetch gather failed: %s\n",
                             rows.status().message().c_str());
                ++failures;
                break;
            }
            pref.insert(pref.end(), rows.value().begin(), rows.value().end());
            for (std::uint32_t i = 0; i < commits[r]; ++i) {
                tail.push_back(batches[r][i]);
                while (tail.size() > ngram_size - 1u) tail.erase(tail.begin());
            }
        }
        check(compare("decode partial-accept prefetch==baseline", base, pref),
              "decode-partial");
    }

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}