#include <shelf/quantization/ngram/ngram.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

namespace shelf::quantization::ngram {
namespace {
Status bad(const char* s) { return Status::invalid_argument(s, __FILE__, __LINE__); }
float f16(uint16_t x) { return static_cast<float>(std::bit_cast<_Float16>(x)); }
uint16_t h16(float x) { return std::bit_cast<uint16_t>(static_cast<_Float16>(x)); }
uint16_t load16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
bool bits_ok(unsigned bits) { return bits == 4 || bits == 5; }
unsigned index_at(std::span<const uint8_t> c, std::size_t i, unsigned bits) {
    if (bits == 4) return (c[i / 2] >> ((i % 2) * 4)) & 15;
    const auto pos = i * 5;
    unsigned v = c[pos / 8] >> (pos % 8);
    if (pos % 8 > 3) v |= unsigned(c[pos / 8 + 1]) << (8 - pos % 8);
    return v & 31;
}
unsigned nearest(float v, std::span<const uint16_t> cb, float scale) {
    unsigned best = 0;
    float error = std::abs(v - scale * f16(cb[0]));
    for (unsigned i = 1; i < cb.size(); ++i) {
        float e = std::abs(v - scale * f16(cb[i]));
        if (e < error) { error = e; best = i; }
    }
    return best;
}
Status check_cb(std::span<const uint16_t> cb, unsigned bits) {
    if (!bits_ok(bits) || cb.size() != (1u << bits)) return bad("invalid codebook size");
    float max_abs = 0, prev = -std::numeric_limits<float>::infinity();
    for (auto h : cb) { float x = f16(h); if (!std::isfinite(x) || x < prev) return bad("invalid codebook"); prev = x; max_abs = std::max(max_abs, std::abs(x)); }
    if (max_abs != 1.0f) return bad("codebook is not normalized");
    return Status::make_ok();
}
Result<std::filesystem::path> safe_path(const std::filesystem::path& root, const std::string& shard) {
    namespace fs = std::filesystem;
    fs::path p(shard);
    if (p.empty() || p.is_absolute()) return bad("invalid shard path");
    for (const auto& part : p) if (part == "..") return bad("invalid shard path");
    std::error_code ec;
    auto base = fs::canonical(root, ec); if (ec) return bad("invalid bundle root");
    auto path = fs::canonical(base / p, ec); if (ec) return bad("invalid shard path");
    auto b = base.begin(), q = path.begin();
    for (; b != base.end(); ++b, ++q) if (q == path.end() || *b != *q) return bad("shard escapes bundle root");
    return path;
}
}
Result<std::size_t> row_bytes(std::size_t dim, unsigned bits) {
    if (!bits_ok(bits) || dim == 0 || dim > (std::numeric_limits<std::size_t>::max() - 7) / bits) return bad("invalid row size or bits");
    return (dim * bits + 7) / 8;
}
Status pack(std::span<const uint8_t> indices, unsigned bits, std::span<uint8_t> codes) {
    auto n = row_bytes(indices.size(), bits); if (!n.ok() || codes.size() != n.value()) return bad("invalid packed row size");
    std::fill(codes.begin(), codes.end(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= (1u << bits)) return bad("index out of range");
        auto pos = i * bits; codes[pos / 8] |= uint8_t(indices[i] << (pos % 8));
        if (pos % 8 + bits > 8) codes[pos / 8 + 1] |= uint8_t(indices[i] >> (8 - pos % 8));
    }
    return Status::make_ok();
}
Status unpack(std::span<const uint8_t> codes, unsigned bits, std::span<uint8_t> indices) {
    auto n = row_bytes(indices.size(), bits); if (!n.ok() || codes.size() != n.value()) return bad("invalid packed row size");
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = index_at(codes, i, bits);
    return Status::make_ok();
}
Status decode(std::span<const uint8_t> codes, uint16_t scale, std::span<const uint16_t> cb, unsigned bits, std::span<float> out) {
    auto n = row_bytes(out.size(), bits); if (!n.ok() || codes.size() != n.value() || cb.size() != (1u << bits)) return bad("invalid decode input");
    float s = f16(scale); if (s == 0) { std::fill(out.begin(), out.end(), 0.0f); return Status::make_ok(); }
    float lut[32]; for (std::size_t i = 0; i < cb.size(); ++i) lut[i] = s * f16(cb[i]);
    std::size_t i = 0;
    if (bits == 4) {
        for (; i + 1 < out.size(); i += 2) { auto b = codes[i/2]; out[i] = lut[b & 15]; out[i+1] = lut[b >> 4]; }
    } else {
        for (; i + 8 <= out.size(); i += 8) {
            auto p = codes.data() + i * 5 / 8;
            uint64_t v = uint64_t(p[0]) | uint64_t(p[1]) << 8 | uint64_t(p[2]) << 16 | uint64_t(p[3]) << 24 | uint64_t(p[4]) << 32;
            for (unsigned j = 0; j < 8; ++j) out[i+j] = lut[(v >> (j*5)) & 31];
        }
    }
    for (; i < out.size(); ++i) out[i] = lut[index_at(codes, i, bits)];
    return Status::make_ok();
}
Status decode_bf16(std::span<const uint8_t> codes, uint16_t scale, std::span<const uint16_t> cb, unsigned bits, std::span<uint16_t> out) {
    auto n = row_bytes(out.size(), bits); if (!n.ok() || codes.size() != n.value() || cb.size() != (1u << bits)) return bad("invalid decode input");
    float s = f16(scale); uint16_t lut[32]{};
    if (s != 0) for (std::size_t i = 0; i < cb.size(); ++i) { auto u = std::bit_cast<uint32_t>(s * f16(cb[i])); lut[i] = uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = lut[index_at(codes, i, bits)];
    return Status::make_ok();
}
Status quantize(std::span<const float> row, std::span<const uint16_t> cb, unsigned bits, uint16_t& scale, std::span<uint8_t> codes) {
    auto n = row_bytes(row.size(), bits); if (!n.ok() || codes.size() != n.value()) return bad("invalid row");
    auto st = check_cb(cb, bits); if (!st.ok()) return st;
    float s = 0; for (float x : row) { if (!std::isfinite(x)) return bad("nonfinite sample"); s = std::max(s, std::abs(x)); }
    for (unsigned iter = 0; iter < 3 && s > 0; ++iter) {
        double num = 0, den = 0;
        for (float x : row) { float c = f16(cb[nearest(x, cb, s)]); num += double(x) * c; den += double(c) * c; }
        s = den ? float(std::max(0.0, num / den)) : 0;
    }
    scale = h16(s); if (!std::isfinite(f16(scale))) return bad("scale out of range");
    std::fill(codes.begin(), codes.end(), 0); s = f16(scale); if (s == 0) { scale = 0; return Status::make_ok(); }
    for (std::size_t i = 0; i < row.size(); ++i) { unsigned idx = nearest(row[i], cb, s); auto pos = i * bits; codes[pos/8] |= uint8_t(idx << (pos%8)); if (pos%8 + bits > 8) codes[pos/8+1] |= uint8_t(idx >> (8-pos%8)); }
    return Status::make_ok();
}
Result<std::vector<uint16_t>> train(std::span<const float> samples, unsigned bits) {
    if (!bits_ok(bits) || samples.empty()) return bad("invalid training input");
    std::vector<float> sorted(samples.begin(), samples.end());
    for (float v : sorted) if (!std::isfinite(v)) return bad("nonfinite sample");
    std::sort(sorted.begin(), sorted.end());
    unsigned k = 1u << bits; std::vector<float> centers(k);
    for (unsigned i = 0; i < k; ++i) centers[i] = sorted[(uint64_t(2*i+1) * sorted.size()) / (2*k)];
    for (unsigned it = 0; it < 100; ++it) {
        std::vector<double> sum(k); std::vector<std::size_t> count(k); unsigned j = 0;
        for (float x : sorted) { while (j+1 < k && double(x)*2 > double(centers[j])+centers[j+1]) ++j; sum[j] += x; ++count[j]; }
        for (unsigned i = 0; i < k; ++i) if (count[i]) centers[i] = float(sum[i] / count[i]);
    }
    float m = std::max(std::abs(centers.front()), std::abs(centers.back())); if (!(m > 0) || !std::isfinite(m)) return bad("cannot normalize codebook");
    std::vector<uint16_t> result; for (float c : centers) result.push_back(h16(c / m));
    auto st = check_cb(result, bits); if (!st.ok()) return st; return result;
}
Status validate_manifest(const Manifest& m) {
    if (m.format != "ngram-q" || m.version != 1 || m.codec != "lloyd-row-f16" || (m.profile != "ngram4" && m.profile != "ngram5" && m.profile != "mixed") || m.source_model.empty() || m.tables.empty()) return bad("invalid manifest identity");
    std::set<std::string> ids; bool four = false, five = false;
    for (const auto& t : m.tables) {
        if (t.id.empty() || t.name.empty() || t.shard.empty() || !ids.insert(t.id).second || !t.ngram_order || !t.rows || !t.dim || !bits_ok(t.bits) || t.codes_tensor != t.id + ".codes" || t.scale_tensor != t.id + ".scale" || t.codebook_tensor != t.id + ".codebook") return bad("invalid table");
        auto n = row_bytes(t.dim, t.bits); if (!n.ok() || t.rows > std::numeric_limits<std::size_t>::max() / n.value() || t.rows > std::numeric_limits<std::size_t>::max() / 2) return bad("table size overflow");
        std::filesystem::path p(t.shard); if (p.is_absolute()) return bad("invalid shard path"); for (const auto& part : p) if (part == "..") return bad("invalid shard path");
        four |= t.bits == 4; five |= t.bits == 5;
    }
    if ((m.profile == "ngram4" && five) || (m.profile == "ngram5" && four) || (m.profile == "mixed" && !(four && five))) return bad("profile mismatch");
    return Status::make_ok();
}
Result<Manifest> parse_manifest(const std::string& text) {
    try {
        auto j = nlohmann::json::parse(text); Manifest m;
        auto str = [](const auto& v, const char* key) -> std::string { return v.at(key).template get<std::string>(); };
        auto integer = [](const auto& v, const char* key) -> std::size_t { const auto& x = v.at(key); if (!x.is_number_unsigned() && !(x.is_number_integer() && x.template get<int64_t>() >= 0)) throw std::runtime_error("bad integer"); auto n = x.template get<uint64_t>(); if (n > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("integer overflow"); return std::size_t(n); };
        m.format = str(j,"format"); auto version = integer(j,"version"); if (version != 1) return bad("invalid version"); m.version = unsigned(version);
        m.profile = str(j,"profile"); m.codec = str(j,"codec"); m.source_model = str(j,"source_model");
        if (!j.at("tables").is_array()) return bad("invalid tables");
        for (const auto& v : j.at("tables")) {
            Table t; t.id = str(v,"id"); t.name = str(v,"name"); t.shard = str(v,"shard"); t.codes_tensor = str(v,"codes_tensor"); t.scale_tensor = str(v,"scale_tensor"); t.codebook_tensor = str(v,"codebook_tensor");
            t.layer = integer(v,"layer"); t.ngram_order = integer(v,"ngram_order"); t.head = integer(v,"head"); t.rows = integer(v,"rows"); t.dim = integer(v,"dim"); auto bits = integer(v,"bits"); if (bits != 4 && bits != 5) return bad("invalid bits"); t.bits = unsigned(bits); m.tables.push_back(std::move(t));
        }
        auto st = validate_manifest(m); if (!st.ok()) return st; return m;
    } catch (const std::exception&) { return bad("invalid manifest JSON"); }
}
Result<std::string> serialize_manifest(const Manifest& m) {
    auto st = validate_manifest(m); if (!st.ok()) return st;
    nlohmann::json j = {{"format",m.format},{"version",m.version},{"profile",m.profile},{"codec",m.codec},{"source_model",m.source_model},{"tables",nlohmann::json::array()}};
    for (const auto& t : m.tables) j["tables"].push_back({{"id",t.id},{"name",t.name},{"layer",t.layer},{"ngram_order",t.ngram_order},{"head",t.head},{"rows",t.rows},{"dim",t.dim},{"bits",t.bits},{"shard",t.shard},{"codes_tensor",t.codes_tensor},{"scale_tensor",t.scale_tensor},{"codebook_tensor",t.codebook_tensor}});
    return j.dump(2);
}
Result<TableReader> TableReader::open(const std::filesystem::path& root, const Table& t) {
    Manifest m; m.profile = t.bits == 4 ? "ngram4" : "ngram5"; m.source_model = "table"; m.tables.push_back(t);
    auto valid = validate_manifest(m); if (!valid.ok()) return valid;
    auto path = safe_path(root, t.shard); if (!path.ok()) return path.status();
    auto reader = io::SafetensorsReader::open(path.value().string()); if (!reader.ok()) return reader.status();
    TableReader result(t, reader.release()); auto st = result.validate(); if (!st.ok()) return st; return result;
}
Status TableReader::validate() const {
    auto n = row_bytes(table_.dim, table_.bits); if (!n.ok() || !table_.rows || table_.rows > std::numeric_limits<std::size_t>::max()/n.value() || table_.rows > std::numeric_limits<std::size_t>::max()/2) return bad("invalid table size");
    const auto width = n.value();
    auto cs = reader_.tensor_spec(table_.codes_tensor), ss = reader_.tensor_spec(table_.scale_tensor), bs = reader_.tensor_spec(table_.codebook_tensor);
    if (!cs.ok() || !ss.ok() || !bs.ok()) return bad("missing tensor");
    if (cs.value().dtype != io::SType::U8 || cs.value().shape != std::vector<std::size_t>{table_.rows,width} || ss.value().dtype != io::SType::F16 || ss.value().shape != std::vector<std::size_t>{table_.rows} || bs.value().dtype != io::SType::F16 || bs.value().shape != std::vector<std::size_t>{std::size_t(1u << table_.bits)}) return bad("tensor shape or dtype mismatch");
    std::size_t cbn = 0, sn = 0, cn = 0; auto cb = reader_.tensor_data(table_.codebook_tensor, cbn), sc = reader_.tensor_data(table_.scale_tensor, sn), co = reader_.tensor_data(table_.codes_tensor, cn);
    if (!cb.ok() || !sc.ok() || !co.ok() || cbn != (2u << table_.bits) || sn != table_.rows*2 || cn != table_.rows*width) return bad("tensor size mismatch");
    auto b = static_cast<const uint8_t*>(cb.value()), s = static_cast<const uint8_t*>(sc.value()), c = static_cast<const uint8_t*>(co.value());
    std::vector<uint16_t> codebook(1u << table_.bits); for (std::size_t i=0;i<codebook.size();++i) codebook[i] = load16(b+2*i);
    auto st = check_cb(codebook, table_.bits); if (!st.ok()) return st;
    unsigned used = unsigned((table_.dim * table_.bits) % 8);
    for (std::size_t r = 0; r < table_.rows; ++r) {
        uint16_t h = load16(s + 2*r); float scale = f16(h); const auto* row = c + r*width;
        if (!std::isfinite(scale) || (h & 0x8000)) return bad("invalid scale");
        if (used && (row[width-1] >> used)) return bad("nonzero padding bits");
        if (h == 0 && std::any_of(row, row+width, [](uint8_t v){ return v != 0; })) return bad("invalid zero row");
    }
    return Status::make_ok();
}
Status TableReader::decode_row(std::size_t row, std::span<float> out) const {
    if (row >= table_.rows || out.size() != table_.dim) return bad("invalid row index or output");
    auto width = row_bytes(table_.dim, table_.bits); if (!width.ok()) return width.status();
    std::size_t cn=0,sn=0,bn=0; auto co=reader_.tensor_data(table_.codes_tensor,cn), sc=reader_.tensor_data(table_.scale_tensor,sn), cb=reader_.tensor_data(table_.codebook_tensor,bn);
    if (!co.ok() || !sc.ok() || !cb.ok()) return bad("missing tensor");
    auto b = static_cast<const uint8_t*>(cb.value()); uint16_t book[32]; for (unsigned i=0;i<(1u<<table_.bits);++i) book[i]=load16(b+2*i);
    return decode({static_cast<const uint8_t*>(co.value()) + row*width.value(), width.value()}, load16(static_cast<const uint8_t*>(sc.value()) + 2*row), {book, std::size_t(1u<<table_.bits)}, table_.bits, out);
}
Status validate_bundle(const std::filesystem::path& root, const Manifest& m) {
    auto st = validate_manifest(m); if (!st.ok()) return st;
    for (const auto& table : m.tables) {
        auto reader = TableReader::open(root, table);
        if (!reader.ok()) return reader.status().with_context((root / table.shard).string());
    }
    return Status::make_ok();
}
}
