#include <shelf/models/qwen4exp/weights/ple_table.h>
#include <shelf/quantization/ngram/ngram.h>
#include <shelf/io/safetensors_writer.h>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;
namespace nq = shelf::quantization::ngram;
using shelf::qwen4exp::PleTable;
using shelf::qwen4exp::Qwen4ExpTextConfig;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* name) { if (!ok) { std::printf("FAIL: %s\n", name); ++failures; } };
    Qwen4ExpTextConfig config;
    config.vocab_size = 128;
    config.ple.layer_ids = {2};
    config.ple.embed_dim = 2560;
    config.ple.ngram_size = 3;
    config.ple.heads_per_ngram = 8;
    config.ple.ngram_vocab_size_base = 2;
    auto sizes = shelf::qwen4exp::qwen4exp_ple_head_vocab_sizes(config).value();
    auto root = fs::temp_directory_path() / ("shelf-qwen-ngram-" + std::to_string(getpid()));
    fs::create_directories(root);
    for (int profile = 0; profile < 3; ++profile) {
        nq::Manifest manifest;
        manifest.profile = profile == 0 ? "ngram4" : profile == 1 ? "ngram5" : "mixed";
        manifest.source_model = "test";
        for (std::size_t h = 0; h < 16; ++h) {
            nq::Table t;
            t.id = "head_" + std::to_string(h);
            t.name = t.id;
            t.shard = t.id + ".safetensors";
            t.codes_tensor = t.id + ".codes";
            t.scale_tensor = t.id + ".scale";
            t.codebook_tensor = t.id + ".codebook";
            t.layer = 1;
            t.head = h;
            t.ngram_order = 2 + h / 8;
            t.rows = sizes[h];
            t.dim = 160;
            t.bits = profile == 0 ? 4 : profile == 1 ? 5 : h < 8 ? 4 : 5;
            auto width = nq::row_bytes(t.dim, t.bits).value();
            std::vector<std::uint8_t> codes(t.rows * width, 0);
            std::vector<std::uint8_t> indices(t.dim);
            for (std::size_t r = 0; r < t.rows; ++r) {
                for (std::size_t j = 0; j < t.dim; ++j)
                    indices[j] = static_cast<std::uint8_t>((h + r + j) % (1u << t.bits));
                check(nq::pack(indices, t.bits, {codes.data() + r * width, width}).ok(), "pack");
            }
            std::vector<std::uint16_t> scales(t.rows, std::bit_cast<std::uint16_t>(_Float16(1)));
            std::vector<std::uint16_t> book(1u << t.bits);
            for (std::size_t i = 0; i < book.size(); ++i) book[i] = std::bit_cast<std::uint16_t>(_Float16(-1.0f + 2.0f * i / (book.size() - 1)));
            auto writer = shelf::io::SafetensorsWriter::create((root / t.shard).string());
            check(writer.ok(), "writer");
            if (!writer.ok()) return 1;
            auto& w = writer.value();
            check(w.plan_tensor(t.codes_tensor, shelf::io::SType::U8, {t.rows, width}).ok(), "codes plan");
            check(w.plan_tensor(t.scale_tensor, shelf::io::SType::F16, {t.rows}).ok(), "scale plan");
            check(w.plan_tensor(t.codebook_tensor, shelf::io::SType::F16, {book.size()}).ok(), "book plan");
            check(w.write_header().ok(), "header");
            check(w.write_tensor(t.codes_tensor, codes.data(), codes.size()).ok(), "codes write");
            check(w.write_tensor(t.scale_tensor, scales.data(), scales.size() * 2).ok(), "scales write");
            check(w.write_tensor(t.codebook_tensor, book.data(), book.size() * 2).ok(), "book write");
            check(w.finish().ok(), "finish");
            manifest.tables.push_back(t);
        }
        auto json = nq::serialize_manifest(manifest);
        check(json.ok(), "manifest serialize");
        std::ofstream(root / "ngram_manifest.json") << json.value();
        for (const char* cache : {"0", "0.001"}) {
            setenv("SHELF_PLE_HOT_CACHE_GIB", cache, 1);
            auto table = PleTable::load(root.string(), config);
            check(table.ok(), "load");
            if (!table.ok()) continue;
            auto rows = table.value().ngram_rows({1, 2, 3}, 1);
            auto gathered = table.value().gather_bf16({1, 2, 3}, 1);
            check(rows.ok() && gathered.ok(), "gather");
            if (!rows.ok() || !gathered.ok()) continue;
            for (std::size_t h = 0; h < 16; ++h) {
                auto reader = nq::TableReader::open(root, manifest.tables[h]);
                std::vector<float> direct(160);
                auto local = rows.value()[h] - table.value().head_offsets()[h];
                check(reader.ok() && reader.value().decode_row(local, direct).ok(), "direct decode");
                for (std::size_t j = 0; j < 160; ++j)
                    check(gathered.value()[h * 160 + j].data == shelf::f32_to_bf16_rne(direct[j]), "bitwise gather");
            }
            check((table.value().cached_blocks() > 0) == (cache[1] == '.'), "cache path");
            std::uint64_t payload = 0;
            for (const auto& t : manifest.tables)
                payload += t.rows * (nq::row_bytes(t.dim, t.bits).value() + 2u) + (2u << t.bits);
            check(table.value().payload_bytes() == payload, "payload size");
            check(table.value().host_bytes() <= payload, "host size");
        }
        auto bad = manifest;
        bad.tables[0].head = 16;
        std::ofstream(root / "ngram_manifest.json") << nq::serialize_manifest(bad).value();
        check(!PleTable::load(root.string(), config).ok(), "reject geometry");
        std::ofstream(root / "ngram_manifest.json") << json.value();
        auto source = shelf::io::SafetensorsReader::open((root / manifest.tables[0].shard).string());
        check(source.ok(), "corrupt source open");
        auto scale_span = source.value().tensor_span(manifest.tables[0].scale_tensor);
        check(scale_span.ok(), "corrupt scale span");
        {
            std::fstream file(root / manifest.tables[0].shard, std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(static_cast<std::streamoff>(scale_span.value().absolute_offset));
            const char invalid_scale[2] = {0, static_cast<char>(0xfc)};
            file.write(invalid_scale, sizeof(invalid_scale));
        }
        check(!PleTable::load(root.string(), config).ok(), "reject corrupt payload");
        std::ofstream(root / "ngram_manifest.json") << "{";
        check(!PleTable::load(root.string(), config).ok(), "reject corrupt manifest");
    }
    fs::remove(root / "ngram_manifest.json");
    std::ofstream(root / "ngram_embedding.safetensors") << "exl3";
    auto legacy = PleTable::load(root.string(), config);
    check(!legacy.ok() && legacy.status().code() == shelf::Status::Code::unsupported,
          "reject EXL3 PLE");
    unsetenv("SHELF_PLE_HOT_CACHE_GIB");
    fs::remove_all(root);
    return failures ? 1 : 0;
}
