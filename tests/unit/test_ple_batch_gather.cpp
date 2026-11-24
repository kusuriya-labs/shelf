#include <shelf/io/safetensors_writer.h>
#include <shelf/models/qwen4exp/weights/ple_table.h>
#include <shelf/quantization/ngram/ngram.h>

#include <bit>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
namespace nq = shelf::quantization::ngram;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::vector<shelf::bf16_t> sequential(const shelf::qwen4exp::PleTable& table,
                                      const std::vector<std::int32_t>& prefix,
                                      const std::vector<std::int32_t>& tokens) {
    std::vector<std::int32_t> history(prefix);
    std::vector<shelf::bf16_t> out;
    for (const std::int32_t token : tokens) {
        history.push_back(token);
        auto rows = table.gather_bf16(history, 1u);
        if (!rows.ok()) return {};
        out.insert(out.end(), rows.value().begin(), rows.value().end());
    }
    return out;
}

}  // namespace

int main() {
    using shelf::qwen4exp::PleTable;
    using shelf::qwen4exp::Qwen4ExpTextConfig;

    Qwen4ExpTextConfig config;
    config.vocab_size = 96;
    config.ple.layer_ids = {2};
    config.ple.embed_dim = 2560;
    config.ple.ngram_size = 3;
    config.ple.heads_per_ngram = 8;
    config.ple.ngram_vocab_size_base = 2;

    auto sizes = shelf::qwen4exp::qwen4exp_ple_head_vocab_sizes(config).value();
    const fs::path root = fs::temp_directory_path() /
                          ("shelf-ple-batch-" + std::to_string(getpid()));
    fs::create_directories(root);

    nq::Manifest manifest;
    manifest.profile = "ngram4";
    manifest.source_model = "test";
    for (std::size_t h = 0; h < sizes.size(); ++h) {
        nq::Table t;
        t.id = "head_" + std::to_string(h);
        t.name = t.id;
        t.shard = t.id + ".safetensors";
        t.codes_tensor = t.id + ".codes";
        t.scale_tensor = t.id + ".scale";
        t.codebook_tensor = t.id + ".codebook";
        t.layer = 1;
        t.head = h;
        t.ngram_order = 2 + static_cast<int>(h / 8u);
        t.rows = sizes[h];
        t.dim = 160;
        t.bits = 4;
        const auto width = nq::row_bytes(t.dim, t.bits).value();
        std::vector<std::uint8_t> codes(t.rows * width, 0);
        std::vector<std::uint8_t> indices(t.dim);
        for (std::size_t r = 0; r < t.rows; ++r) {
            for (std::size_t j = 0; j < t.dim; ++j)
                indices[j] = static_cast<std::uint8_t>((h + r + j) % 16u);
            check(nq::pack(indices, t.bits, {codes.data() + r * width, width}).ok(),
                  "pack");
        }
        std::vector<std::uint16_t> scales(t.rows,
                                          std::bit_cast<std::uint16_t>(_Float16(1)));
        std::vector<std::uint16_t> book(16);
        for (std::size_t i = 0; i < book.size(); ++i)
            book[i] = std::bit_cast<std::uint16_t>(
                _Float16(-1.0f + 2.0f * static_cast<float>(i) / 15.0f));
        auto writer = shelf::io::SafetensorsWriter::create((root / t.shard).string());
        check(writer.ok(), "writer");
        auto& w = writer.value();
        check(w.plan_tensor(t.codes_tensor, shelf::io::SType::U8,
                            {t.rows, width})
                  .ok(),
              "codes plan");
        check(w.plan_tensor(t.scale_tensor, shelf::io::SType::F16, {t.rows}).ok(),
              "scale plan");
        check(w.plan_tensor(t.codebook_tensor, shelf::io::SType::F16, {book.size()})
                  .ok(),
              "book plan");
        check(w.write_header().ok(), "header");
        check(w.write_tensor(t.codes_tensor, codes.data(), codes.size()).ok(),
              "codes write");
        check(w.write_tensor(t.scale_tensor, scales.data(), scales.size() * 2).ok(),
              "scales write");
        check(w.write_tensor(t.codebook_tensor, book.data(), book.size() * 2).ok(),
              "book write");
        check(w.finish().ok(), "finish");
        manifest.tables.push_back(t);
    }
    auto json = nq::serialize_manifest(manifest);
    check(json.ok(), "manifest serialize");
    std::ofstream(root / "ngram_manifest.json") << json.value();

    auto loaded = PleTable::load(root.string(), config);
    check(loaded.ok(), "table load");
    if (loaded.ok()) {
        const PleTable& table = loaded.value();
        const std::uint32_t width = table.head_count() * table.row_width();
        const std::int32_t eos = config.ple.eos_token_id;
        struct BatchCase {
            std::vector<std::int32_t> prefix;
            std::vector<std::int32_t> tokens;
        };
        const std::vector<BatchCase> cases = {
            {{1, 2}, {3}},
            {{1, 2, 3}, {4, 5, 6, 7, 8}},
            {{}, {10, 20, eos, 30, 40, 50, 60, 70}},
            {{7, 8, 9}, {}},
            {{5}, {eos, eos, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
        };
        for (std::size_t c = 0; c < cases.size(); ++c) {
            auto got = table.gather_bf16_batch(cases[c].prefix, cases[c].tokens);
            check(got.ok(), "batch gather ok");
            if (!got.ok()) continue;
            check(got.value().size() == cases[c].tokens.size() * width,
                  "batch gather size");
            auto want = sequential(table, cases[c].prefix, cases[c].tokens);
            check(got.value().size() == want.size(), "batch size matches sequential");
            bool bit_equal = got.value().size() == want.size();
            for (std::size_t i = 0; bit_equal && i < want.size(); ++i)
                bit_equal = got.value()[i].data == want[i].data;
            check(bit_equal, "batch gather is bit-identical to sequential gather");
            auto stateless =
                table.gather_bf16_stateless(cases[c].prefix, cases[c].tokens);
            bool wrapper_equal = stateless.ok() &&
                                 stateless.value().size() == got.value().size();
            for (std::size_t i = 0; wrapper_equal && i < got.value().size(); ++i)
                wrapper_equal =
                    stateless.value()[i].data == got.value()[i].data;
            check(wrapper_equal, "stateless wrapper matches batch gather");
        }
    }

    fs::remove_all(root);
    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
