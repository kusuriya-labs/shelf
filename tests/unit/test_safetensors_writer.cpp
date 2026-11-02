#include <phaseshift/io/safetensors_writer.h>
#include <phaseshift/io/safetensors_reader.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) { passed++; printf("PASS: %s\n", msg); }
    else { failed++; printf("FAIL: %s\n", msg); }
}

static std::string temp_path(const char* name) {
    return std::string("/tmp/ps_writer_") + name + "_" + std::to_string(getpid()) + ".safetensors";
}

static void remove_file(const std::string& p) {
    std::remove(p.c_str());
}

static void roundtrip_dtype(ps::io::SType dt, const char* dtname) {
    const std::string path = temp_path(dtname);
    remove_file(path);

    auto w = ps::io::SafetensorsWriter::create(path);
    check(w.ok(), (std::string("create ") + dtname).c_str());
    if (!w.ok()) return;

    std::vector<std::size_t> shape = {2, 4};
    check(w.value().plan_tensor("t", dt, shape).ok(), (std::string("plan ") + dtname).c_str());
    check(w.value().set_metadata("phaseshift.format", "phaseshift-fpx-safetensors").ok(), (std::string("metadata ") + dtname).c_str());

    auto base = w.value().write_header();
    check(base.ok(), (std::string("write_header ") + dtname).c_str());
    if (!base.ok()) return;

    std::size_t esz = (dt == ps::io::SType::F32) ? 4 : (dt == ps::io::SType::BF16 || dt == ps::io::SType::F16) ? 2 : 1;
    std::size_t nbytes = 8 * esz;
    std::vector<std::uint8_t> payload(nbytes);
    for (std::size_t i = 0; i < nbytes; ++i) payload[i] = static_cast<std::uint8_t>((i * 7 + 3) & 0xFF);
    check(w.value().write_tensor("t", payload.data(), payload.size()).ok(), (std::string("write_tensor ") + dtname).c_str());
    auto fin = w.value().finish();
    check(fin.ok(), (std::string("finish ") + dtname).c_str());

    auto r = ps::io::SafetensorsReader::open(path);
    check(r.ok(), (std::string("reopen ") + dtname).c_str());
    if (!r.ok()) return;

    auto specs = r.value().tensor_spec("t");
    check(specs.ok(), (std::string("tensor_spec ") + dtname).c_str());
    if (!specs.ok()) return;
    check(specs.value().dtype == dt, (std::string("dtype preserved ") + dtname).c_str());
    check(specs.value().shape == shape, (std::string("shape preserved ") + dtname).c_str());

    auto list = r.value().list_tensors();
    bool metadata_excluded = true;
    for (const auto& n : list.value()) {
        if (n == "__metadata__") metadata_excluded = false;
    }
    check(metadata_excluded, (std::string("__metadata__ not a tensor ") + dtname).c_str());
    check(r.value().metadata().at("phaseshift.format") == "phaseshift-fpx-safetensors",
          (std::string("metadata readable ") + dtname).c_str());

    std::vector<std::uint8_t> buf(payload.size());
    check(r.value().read_tensor("t", buf.data(), buf.size()).ok(), (std::string("read_tensor ") + dtname).c_str());
    check(std::memcmp(buf.data(), payload.data(), payload.size()) == 0,
          (std::string("payload roundtrip ") + dtname).c_str());

    remove_file(path);
}

static void test_size_mismatch_rejected() {
    const std::string path = temp_path("mismatch");
    remove_file(path);
    auto w = ps::io::SafetensorsWriter::create(path);
    if (!w.ok()) return;
    check(w.value().plan_tensor("t", ps::io::SType::U8, {4}).ok(), "plan size mismatch");
    auto base = w.value().write_header();
    if (!base.ok()) return;
    std::vector<std::uint8_t> wrong(3);
    check(!w.value().write_tensor("t", wrong.data(), wrong.size()).ok(), "write size mismatch rejected");
    remove_file(path);
}

static void test_duplicate_rejected() {
    const std::string path = temp_path("dup");
    remove_file(path);
    auto w = ps::io::SafetensorsWriter::create(path);
    if (!w.ok()) return;
    check(w.value().plan_tensor("t", ps::io::SType::U8, {4}).ok(), "plan dup first");
    check(!w.value().plan_tensor("t", ps::io::SType::U8, {4}).ok(), "plan dup rejected");
    remove_file(path);
}

static void test_reserved_name_rejected() {
    const std::string path = temp_path("reserved");
    remove_file(path);
    auto w = ps::io::SafetensorsWriter::create(path);
    if (!w.ok()) return;
    check(!w.value().plan_tensor("__metadata__", ps::io::SType::U8, {4}).ok(), "reserved name rejected");
    remove_file(path);
}

static void test_zero_dim_rejected() {
    const std::string path = temp_path("zerodim");
    remove_file(path);
    auto w = ps::io::SafetensorsWriter::create(path);
    if (!w.ok()) return;
    check(!w.value().plan_tensor("t", ps::io::SType::U8, {0}).ok(), "zero dim rejected");
    remove_file(path);
}

static void test_malformed_reader_rejection() {
    const std::string path = temp_path("malformed");
    remove_file(path);

    {
        FILE* f = std::fopen(path.c_str(), "wb");
        std::uint64_t len = 5;
        std::fwrite(&len, 1, 8, f);
        std::fwrite("{\"t\"", 1, 5, f);
        std::fclose(f);
    }
    auto r = ps::io::SafetensorsReader::open(path);
    check(!r.ok(), "malformed header rejected");
    remove_file(path);
}

static void test_offset_validation() {
    const std::string path = temp_path("offset");
    remove_file(path);
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        std::string h = R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"__metadata__":{"k":"v"}})";
        std::uint64_t len = h.size();
        std::fwrite(&len, 1, 8, f);
        std::fwrite(h.data(), 1, h.size(), f);
        std::fwrite("\x01\x02\x03\x04", 1, 4, f);
        std::fclose(f);
    }
    auto r = ps::io::SafetensorsReader::open(path);
    check(r.ok(), "valid minimal safetensors opens");
    if (r.ok()) {
        check(r.value().metadata().at("k") == "v", "metadata parsed");
        std::vector<std::uint8_t> buf(4);
        check(r.value().read_tensor("a", buf.data(), 4).ok(), "offset read ok");
        check(buf[0] == 1 && buf[3] == 4, "offset data correct");
    }
    remove_file(path);
}

static void test_out_of_file_rejected() {
    const std::string path = temp_path("oob");
    remove_file(path);
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        std::string h = R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,400]}})";
        std::uint64_t len = h.size();
        std::fwrite(&len, 1, 8, f);
        std::fwrite(h.data(), 1, h.size(), f);
        std::fwrite("\x01\x02\x03\x04", 1, 4, f);
        std::fclose(f);
    }
    auto r = ps::io::SafetensorsReader::open(path);
    check(!r.ok(), "out-of-file offsets rejected");
    remove_file(path);
}

int main() {
    printf("=== SafetensorsWriter Roundtrip / Validation ===\n");
    roundtrip_dtype(ps::io::SType::BF16, "bf16");
    roundtrip_dtype(ps::io::SType::F16, "f16");
    roundtrip_dtype(ps::io::SType::F32, "f32");
    roundtrip_dtype(ps::io::SType::I8, "i8");
    roundtrip_dtype(ps::io::SType::U8, "u8");
    test_size_mismatch_rejected();
    test_duplicate_rejected();
    test_reserved_name_rejected();
    test_zero_dim_rejected();
    test_malformed_reader_rejection();
    test_offset_validation();
    test_out_of_file_rejected();
    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return (failed > 0) ? 1 : 0;
}
