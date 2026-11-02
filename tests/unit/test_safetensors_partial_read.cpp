#include <shelf/io/safetensors_reader.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/weights/weight_loader.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using shelf::io::SType;
using shelf::io::StAxisSpan;

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

static std::string temp_path(const char* name) {
    return "/tmp/shelf_partial_" + std::string(name) + "_" +
           std::to_string(getpid()) + ".safetensors";
}

static void remove_file(const std::string& p) { std::remove(p.c_str()); }

static std::vector<std::uint8_t> pattern(std::size_t n, std::uint8_t seed) {
    std::vector<std::uint8_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<std::uint8_t>((seed + i * 31u + (i >> 8) * 17u) & 0xFFu);
    }
    return out;
}

static void write_file(const std::string& path, const std::string& rel,
                       const std::vector<std::uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    (void)rel;
}

static void test_i64_roundtrip() {
    const std::string path = temp_path("i64");
    remove_file(path);

    const std::vector<std::int64_t> multipliers = {1234567, -987654321, 42};
    auto w = shelf::io::SafetensorsWriter::create(path);
    check(w.ok(), "i64: create writer");
    if (!w.ok()) return;
    check(w.value().plan_tensor("m", SType::I64, {3}).ok(), "i64: plan I64");
    auto base = w.value().write_header();
    check(base.ok(), "i64: write header");
    if (!base.ok()) return;
    check(w.value()
              .write_tensor("m", multipliers.data(),
                            multipliers.size() * sizeof(std::int64_t))
              .ok(),
          "i64: write payload");
    check(w.value().finish().ok(), "i64: finish");

    auto r = shelf::io::SafetensorsReader::open(path);
    check(r.ok(), "i64: reopen");
    if (!r.ok()) {
        remove_file(path);
        return;
    }
    auto spec = r.value().tensor_spec("m");
    check(spec.ok(), "i64: tensor_spec");
    if (spec.ok()) {
        check(spec.value().dtype == SType::I64, "i64: dtype is I64");
        check(spec.value().shape == std::vector<std::size_t>{3}, "i64: shape [3]");
        check(spec.value().data_end - spec.value().data_begin == 24, "i64: 24 bytes");
    }
    check(shelf::io::stype_elem_size(SType::I64) == 8, "i64: elem size 8");
    check(std::string(shelf::io::stype_name(SType::I64)) == "I64", "i64: name I64");

    std::int64_t got[3] = {0, 0, 0};
    check(r.value().read_tensor("m", got, sizeof(got)).ok(), "i64: read back");
    check(std::memcmp(got, multipliers.data(), sizeof(got)) == 0,
          "i64: payload roundtrip");

    auto span = r.value().tensor_span("m");
    check(span.ok(), "i64: tensor_span");
    if (span.ok()) {
        check(span.value().absolute_offset == r.value().data_base_offset(),
              "i64: absolute offset = data_base");
        check(span.value().byte_length == 24, "i64: span bytes 24");
    }
    remove_file(path);
}

static void test_i64_unsupported_dtype_still_rejected() {
    const std::string path = temp_path("baddtype");
    remove_file(path);
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        std::string h = R"({"a":{"dtype":"I16","shape":[2],"data_offsets":[0,4]}})";
        std::uint64_t len = h.size();
        std::fwrite(&len, 1, 8, f);
        std::fwrite(h.data(), 1, h.size(), f);
        std::fwrite("abcd", 1, 4, f);
        std::fclose(f);
    }
    auto r = shelf::io::SafetensorsReader::open(path);
    check(!r.ok(), "unknown dtype still rejected");
    remove_file(path);
}

static void test_rank2_row_spans() {
    const std::string path = temp_path("rows");
    remove_file(path);

    constexpr std::size_t kRows = 7;
    constexpr std::size_t kCols = 5;
    const auto payload = pattern(kRows * kCols * 2, 0x11);

    auto w = shelf::io::SafetensorsWriter::create(path);
    check(w.ok(), "rows: create writer");
    if (!w.ok()) return;
    check(w.value().plan_tensor("w", SType::BF16, {kRows, kCols}).ok(),
          "rows: plan rank-2");
    auto base = w.value().write_header();
    if (!base.ok()) return;
    check(w.value().write_tensor("w", payload.data(), payload.size()).ok(),
          "rows: write payload");
    check(w.value().finish().ok(), "rows: finish");

    auto r = shelf::io::SafetensorsReader::open(path);
    check(r.ok(), "rows: reopen");
    if (!r.ok()) {
        remove_file(path);
        return;
    }
    const std::uint64_t base_off = r.value().data_base_offset();

    auto first = r.value().axis_span("w", 0, 1);
    check(first.ok(), "rows: first row span");
    if (first.ok()) {
        check(first.value().absolute_offset == base_off, "rows: first row offset");
        check(first.value().byte_length == kCols * 2, "rows: first row bytes");
        check(first.value().unit_bytes == kCols * 2, "rows: unit stride");
    }

    auto last = r.value().axis_span("w", kRows - 1, 1);
    check(last.ok(), "rows: last row span");
    if (last.ok()) {
        check(last.value().absolute_offset == base_off + (kRows - 1) * kCols * 2,
              "rows: last row offset");
    }

    auto mid = r.value().axis_span("w", 2, 3);
    check(mid.ok(), "rows: mid range span");
    if (mid.ok()) {
        check(mid.value().byte_length == 3 * kCols * 2, "rows: mid range bytes");
        check(mid.value().absolute_offset == base_off + 2 * kCols * 2,
              "rows: mid range offset");
    }

    check(!r.value().axis_span("w", kRows, 1).ok(), "rows: out-of-range begin rejected");
    check(!r.value().axis_span("w", 0, kRows + 1).ok(), "rows: overlong range rejected");
    check(!r.value().axis_span("w", 0, 0).ok(), "rows: empty range rejected");
    check(!r.value().axis_span("w", 0, ~std::uint64_t{0}).ok(),
          "rows: overflowing count rejected");
    check(!r.value().axis_span("missing", 0, 1).ok(), "rows: missing tensor rejected");

    auto spec = r.value().tensor_spec("w");
    check(spec.ok() && spec.value().shape.size() == 2, "rows: rank-2 spec");
    remove_file(path);
}

static void test_rank3_expert_spans() {
    const std::string path = temp_path("expert");
    remove_file(path);

    constexpr std::size_t kExperts = 4;
    constexpr std::size_t kRows = 3;
    constexpr std::size_t kCols = 2;
    const auto payload = pattern(kExperts * kRows * kCols * 2, 0x5A);

    auto w = shelf::io::SafetensorsWriter::create(path);
    check(w.ok(), "expert: create writer");
    if (!w.ok()) return;
    check(w.value().plan_tensor("e", SType::BF16, {kExperts, kRows, kCols}).ok(),
          "expert: plan rank-3");
    auto base = w.value().write_header();
    if (!base.ok()) return;
    check(w.value().write_tensor("e", payload.data(), payload.size()).ok(),
          "expert: write payload");
    check(w.value().finish().ok(), "expert: finish");

    auto r = shelf::io::SafetensorsReader::open(path);
    check(r.ok(), "expert: reopen");
    if (!r.ok()) {
        remove_file(path);
        return;
    }
    const std::uint64_t base_off = r.value().data_base_offset();
    const std::uint64_t expert_bytes = kRows * kCols * 2;

    for (std::uint64_t e = 0; e < kExperts; ++e) {
        auto span = r.value().axis_span("e", e, 1);
        check(span.ok(), "expert: slice span");
        if (!span.ok()) continue;
        check(span.value().absolute_offset == base_off + e * expert_bytes,
              "expert: slice offset");
        check(span.value().byte_length == expert_bytes, "expert: slice bytes");

        std::vector<std::uint8_t> buf(payload.size());
        const auto expected = payload.data() + e * expert_bytes;
        auto rd = r.value().read_tensor("e", buf.data(), buf.size());
        check(rd.ok(), "expert: whole read");
        if (rd.ok()) {
            check(std::memcmp(buf.data() + e * expert_bytes, expected,
                              expert_bytes) == 0,
                  "expert: slice bytes match source");
        }
    }

    auto whole = r.value().axis_span("e", 0, kExperts);
    check(whole.ok(), "expert: all experts span");
    if (whole.ok()) {
        check(whole.value().byte_length == payload.size(), "expert: all bytes");
    }
    check(!r.value().axis_span("e", kExperts, 1).ok(),
          "expert: invalid expert id rejected");
    check(!r.value().axis_span("e", 0, kExperts + 1).ok(),
          "expert: overlong expert range rejected");
    remove_file(path);
}

static void test_rank0_rejected() {
    const std::string path = temp_path("rank0");
    remove_file(path);
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        std::string h = R"({"s":{"dtype":"U8","shape":[],"data_offsets":[0,1]}})";
        std::uint64_t len = h.size();
        std::fwrite(&len, 1, 8, f);
        std::fwrite(h.data(), 1, h.size(), f);
        std::fwrite("x", 1, 1, f);
        std::fclose(f);
    }
    auto r = shelf::io::SafetensorsReader::open(path);
    check(r.ok(), "rank0: opens as scalar tensor");
    if (r.ok()) {
        check(!r.value().axis_span("s", 0, 1).ok(), "rank0: axis_span rejected");
    }
    remove_file(path);
}

static void test_truncated_and_overflow() {
    const std::string path = temp_path("trunc");
    remove_file(path);
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        std::string h =
            R"({"a":{"dtype":"BF16","shape":[4,4],"data_offsets":[0,32]}})";
        std::uint64_t len = h.size();
        std::fwrite(&len, 1, 8, f);
        std::fwrite(h.data(), 1, h.size(), f);
        std::fwrite("short", 1, 5, f);
        std::fclose(f);
    }
    auto r = shelf::io::SafetensorsReader::open(path);
    check(!r.ok(), "truncated payload rejected at open");
    remove_file(path);

    const std::string path2 = temp_path("ovf");
    remove_file(path2);
    {
        FILE* f = std::fopen(path2.c_str(), "wb");
        std::string h =
            R"({"a":{"dtype":"BF16","shape":[18446744073709551615,3],"data_offsets":[0,8]}})";
        std::uint64_t len = h.size();
        std::fwrite(&len, 1, 8, f);
        std::fwrite(h.data(), 1, h.size(), f);
        std::fwrite("12345678", 1, 8, f);
        std::fclose(f);
    }
    auto r2 = shelf::io::SafetensorsReader::open(path2);
    check(!r2.ok(), "shape overflow rejected");
    remove_file(path2);
}

static void write_shard(const std::string& path, const std::string& name,
                        SType dtype, const std::vector<std::size_t>& shape,
                        const std::vector<std::uint8_t>& bytes) {
    auto w = shelf::io::SafetensorsWriter::create(path);
    if (!w.ok()) return;
    if (!w.value().plan_tensor(name, dtype, shape).ok()) return;
    if (!w.value().write_header().ok()) return;
    if (!w.value().write_tensor(name, bytes.data(), bytes.size()).ok()) return;
    (void)w.value().finish();
}

static void test_collection_multi_shard() {
    const std::string dir = "/tmp/shelf_partial_col_" + std::to_string(getpid());
    fs::remove_all(dir);
    fs::create_directories(dir);

    const auto b0 = pattern(4 * 4 * 2, 0x21);
    const auto b1 = pattern(3 * 8, 0x43);
    write_shard(dir + "/model-00001-of-00002.safetensors", "alpha", SType::BF16,
                {4, 4}, b0);
    write_shard(dir + "/model-00002-of-00002.safetensors", "beta", SType::I64,
                {3}, b1);

    {
        std::ofstream idx(dir + "/model.safetensors.index.json");
        idx << R"({"weight_map":{"alpha":"model-00001-of-00002.safetensors",)"
            << R"("beta":"model-00002-of-00002.safetensors"}})";
    }

    auto col = shelf::weights::SafetensorsCollection::open(dir);
    check(col.ok(), "collection: open");
    if (!col.ok()) {
        fs::remove_all(dir);
        return;
    }

    auto shard_a = col.value().tensor_shard("alpha");
    check(shard_a.ok() && shard_a.value() == "model-00001-of-00002.safetensors",
          "collection: shard resolve alpha");
    auto shard_b = col.value().tensor_shard("beta");
    check(shard_b.ok() && shard_b.value() == "model-00002-of-00002.safetensors",
          "collection: shard resolve beta");
    check(!col.value().tensor_shard("gamma").ok(), "collection: unknown tensor rejected");

    auto span_a = col.value().tensor_span("alpha");
    check(span_a.ok(), "collection: alpha span");
    if (span_a.ok()) {
        check(span_a.value().byte_length == 32, "collection: alpha bytes");
        check(span_a.value().absolute_offset >= 8, "collection: alpha offset past header");
    }

    auto row = col.value().axis_span("alpha", 1, 1);
    check(row.ok(), "collection: alpha row span");
    if (row.ok() && span_a.ok()) {
        check(row.value().byte_length == 8, "collection: alpha row bytes");
        check(row.value().absolute_offset == span_a.value().absolute_offset + 8,
              "collection: alpha row offset");
    }

    auto spec_b = col.value().tensor_spec("beta");
    check(spec_b.ok() && spec_b.value().dtype == SType::I64,
          "collection: beta I64 spec");

    auto fsz = col.value().shard_file_size("model-00001-of-00002.safetensors");
    check(fsz.ok() && fsz.value() > 64, "collection: shard file size");

    check(!col.value().shard_file_size("../escape.safetensors").ok(),
          "collection: path traversal rejected");
    check(!col.value().shard_file_size("/etc/passwd").ok(),
          "collection: absolute path rejected");

    fs::remove_all(dir);
}

static void test_python_safetensors_compatibility() {
    const std::string path = temp_path("pycompat");
    remove_file(path);

    const std::vector<std::int64_t> vals = {11, 22, 33, 44};
    auto w = shelf::io::SafetensorsWriter::create(path);
    if (!w.ok()) return;
    if (!w.value().plan_tensor("t", SType::I64, {4}).ok()) return;
    if (!w.value().write_header().ok()) return;
    if (!w.value().write_tensor("t", vals.data(), vals.size() * 8).ok()) return;
    if (!w.value().finish().ok()) return;

    const std::string py_path = "/tmp/shelf_partial_check_" + std::to_string(getpid()) + ".py";
    {
        std::ofstream pf(py_path, std::ios::trunc);
        pf << "import json,struct,sys\n"
              "p=sys.argv[1]\n"
              "with open(p,'rb') as f:\n"
              "    n=struct.unpack('<Q',f.read(8))[0]\n"
              "    h=json.loads(f.read(n))\n"
              "    e=h['t']\n"
              "    assert e['dtype']=='I64', e['dtype']\n"
              "    assert e['shape']==[4], e['shape']\n"
              "    assert e['data_offsets']==[0,32], e['data_offsets']\n"
              "    data=f.read()\n"
              "    assert data==struct.pack('<4q',11,22,33,44), data.hex()\n"
              "print('PYOK')\n";
    }
    const std::string cmd =
        "python3 '" + py_path + "' '" + path + "' 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    std::string out;
    if (pipe) {
        char buf[256];
        while (std::fgets(buf, sizeof(buf), pipe)) out += buf;
        pclose(pipe);
    }
    check(out.find("PYOK") != std::string::npos, "python safetensors header compat");
    if (out.find("PYOK") == std::string::npos) printf("  py: %s\n", out.c_str());
    std::remove(py_path.c_str());
    remove_file(path);
}

int main() {
    printf("=== Safetensors partial read / metadata ===\n");
    test_i64_roundtrip();
    test_i64_unsupported_dtype_still_rejected();
    test_rank2_row_spans();
    test_rank3_expert_spans();
    test_rank0_rejected();
    test_truncated_and_overflow();
    test_collection_multi_shard();
    test_python_safetensors_compatibility();
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
