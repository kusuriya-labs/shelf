#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/quantization/ngram/ngram.h>
#include <shelf/io/safetensors_writer.h>
#include <hip/hip_runtime.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <vector>

using namespace shelf::hy4;
using namespace shelf::quantization::ngram;

uint16_t half(float v) { return std::bit_cast<uint16_t>(static_cast<_Float16>(v)); }
uint16_t bf(float v) {
    auto u = std::bit_cast<uint32_t>(v);
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float fbf(uint16_t v) { return std::bit_cast<float>(uint32_t(v) << 16); }
void check(hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); }
}
bool test_load() {
    namespace fs = std::filesystem;
    auto root = fs::temp_directory_path() / ("shelf-hy4-ngram-" + std::to_string(getpid()));
    fs::create_directories(root);
    Manifest manifest;
    manifest.profile = "mixed";
    manifest.source_model = "test";
    auto writer_result = shelf::io::SafetensorsWriter::create((root / "tables.safetensors").string());
    if (!writer_result.ok()) return false;
    auto &writer = writer_result.value();
    Checkpoint model;
    for (int g = 0; g < 4; ++g) for (int h = 0; h < 16; ++h) {
        Table t;
        t.id = "t" + std::to_string(g * 16 + h);
        t.name = t.id; t.shard = "tables.safetensors";
        t.codes_tensor = t.id + ".codes"; t.scale_tensor = t.id + ".scale"; t.codebook_tensor = t.id + ".codebook";
        t.layer = 1 + 10 * g; t.ngram_order = 2 + h / 8; t.head = h;
        t.rows = 1; t.dim = 96; t.bits = h % 2 ? 5 : 4;
        model.moduli[g][h] = 1;
        if (!writer.plan_tensor(t.codes_tensor, shelf::io::SType::U8, {1, 12 * t.bits}).ok() ||
            !writer.plan_tensor(t.scale_tensor, shelf::io::SType::F16, {1}).ok() ||
            !writer.plan_tensor(t.codebook_tensor, shelf::io::SType::F16, {size_t(1u << t.bits)}).ok()) return false;
        manifest.tables.push_back(t);
    }
    if (!writer.write_header().ok()) return false;
    for (const auto &t : manifest.tables) {
        std::vector<uint8_t> codes(12 * t.bits);
        uint16_t scale = 0;
        std::vector<uint16_t> book(1u << t.bits, half(1));
        if (!writer.write_tensor(t.codes_tensor, codes.data(), codes.size()).ok() ||
            !writer.write_tensor(t.scale_tensor, &scale, 2).ok() ||
            !writer.write_tensor(t.codebook_tensor, book.data(), book.size() * 2).ok()) return false;
    }
    if (!writer.finish().ok()) return false;
    auto serialized = serialize_manifest(manifest);
    if (!serialized.ok()) return false;
    std::ofstream(root / "ngram_manifest.json") << serialized.value();
    bool ok = true;
    try { model.load_ngram(root); } catch (...) { ok = false; }
    ok &= model.ngram_heads[3][15].bits == 5 && model.engram_files.size() == 1;
    manifest.tables[0].dim = 95;
    auto altered = serialize_manifest(manifest);
    if (!altered.ok()) return false;
    std::ofstream(root / "ngram_manifest.json") << altered.value();
    try { Checkpoint bad; bad.load_ngram(root); ok = false; } catch (const std::exception &) {}
    std::ofstream(root / "ngram_manifest.json") << "{broken";
    try { Checkpoint bad; bad.load_ngram(root); ok = false; } catch (const std::exception &) {}
    fs::remove_all(root);
    return ok;
}
int main() {
    if (!test_load()) { std::fprintf(stderr, "hy4 Ngram loader: FAIL\n"); return 1; }
    constexpr int rows = 2;
    NgramLayout layout{};
    EngramNorms norms{};
    std::array<std::vector<uint16_t>, 16> books;
    std::array<std::vector<uint8_t>, 16> codes;
    std::vector<uint8_t> packed;
    std::array<std::array<float, 96>, 16> expected{};
    for (int h = 0; h < 16; ++h) {
        unsigned bits = h % 2 ? 5 : 4, width = 12 * bits;
        layout.bits[h] = bits;
        layout.offsets[h] = layout.stride;
        layout.stride += width + 2;
        for (unsigned i = 0; i < (1u << bits); ++i)
            books[h].push_back(half(float(int(i) - int(1u << (bits - 1))) / float(1u << (bits - 1))));
        std::vector<uint8_t> indices(96);
        for (int c = 0; c < 96; ++c) indices[c] = (c * 7 + h) % (1u << bits);
        codes[h].resize(width);
        if (!pack(indices, bits, codes[h]).ok() || !decode(codes[h], half(1.25f), books[h], bits, expected[h]).ok()) return 1;
    }
    packed.resize(rows * layout.stride);
    for (int h = 0; h < 16; ++h) {
        auto dst = packed.data() + layout.offsets[h];
        std::copy(codes[h].begin(), codes[h].end(), dst);
        auto scale = half(1.25f);
        dst[codes[h].size()] = scale & 255;
        dst[codes[h].size() + 1] = scale >> 8;
        uint16_t *device_book;
        check(hipMalloc(&device_book, books[h].size() * 2));
        check(hipMemcpy(device_book, books[h].data(), books[h].size() * 2, hipMemcpyHostToDevice));
        layout.books[h] = device_book;
    }
    std::array<uint16_t, 96> ones; ones.fill(bf(1));
    uint16_t *device_norm, *device_out;
    uint8_t *device_packed;
    check(hipMalloc(&device_norm, sizeof(ones)));
    check(hipMemcpy(device_norm, ones.data(), sizeof(ones), hipMemcpyHostToDevice));
    for (auto &norm : norms.head) norm = device_norm;
    check(hipMalloc(&device_packed, packed.size()));
    check(hipMemcpy(device_packed, packed.data(), packed.size(), hipMemcpyHostToDevice));
    check(hipMalloc(&device_out, rows * 1536 * 2));
    launch_decode_heads(device_packed, norms, Mat{device_out, rows, 1536}, layout, true);
    std::vector<uint16_t> out(rows * 1536);
    check(hipMemcpy(out.data(), device_out, out.size() * 2, hipMemcpyDeviceToHost));
    bool ok = true;
    for (int h = 0; h < 16; ++h) {
        float sum = 0;
        for (auto &v : expected[h]) { v = fbf(bf(v)); sum += v * v; }
        float inv = 1.f / std::sqrt(sum / 96.f + 1.1920928955e-7f);
        for (int c = 0; c < 96; ++c) {
            float want = fbf(bf(expected[h][c] * inv));
            ok &= std::abs(fbf(out[h * 96 + c]) - want) <= 0.016f;
            ok &= out[1536 + h * 96 + c] == 0;
        }
    }
    for (auto book : layout.books) check(hipFree(const_cast<uint16_t *>(book)));
    check(hipFree(device_norm)); check(hipFree(device_out)); check(hipFree(device_packed));
    std::printf("hy4 Ngram GPU decode: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
