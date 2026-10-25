// shelf unit test — safetensors reader / Raw Tensor Catalog (plan PR-004).
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <shelf/artifact/safetensors/reader.hpp>

#include "../support/safetensors_fixture.hpp"

namespace {

using namespace shelf;
using namespace shelf::artifact;

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

std::vector<std::uint8_t> bf16_payload(std::size_t count) {
    std::vector<std::uint8_t> bytes(count * 2);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t v = static_cast<std::uint16_t>(0x3F80 + (i % 16));  // ~1.0 family
        bytes[2 * i] = static_cast<std::uint8_t>(v & 0xFF);
        bytes[2 * i + 1] = static_cast<std::uint8_t>(v >> 8);
    }
    return bytes;
}

void test_valid_file() {
    const auto path = st_fixture::tmp_path("valid");
    st_fixture::Tensor embed{"model.embed_tokens.weight", "BF16", {4, 8}, bf16_payload(32)};
    st_fixture::Tensor w1{"layer.mlp.w1", "F32", {3, 3}, std::vector<std::uint8_t>(36, 0x3C)};
    check(st_fixture::write_file(path, {embed, w1}, "{\"format\":\"pt\"}"), "fixture written");

    auto reader = safetensors::Reader::open(path);
    check(reader.ok(), "open valid file");
    if (!reader.ok()) return;

    check(reader.value().catalog().size() == 2, "catalog size 2");
    auto names = reader.value().list_tensors();
    check(names.size() == 2, "list_tensors size");
    check(reader.value().metadata().count("format") == 1 &&
              reader.value().metadata().at("format") == "pt",
          "metadata parsed");
    check(reader.value().data_base_offset() > 8, "data base offset sane");

    const auto* e = reader.value().catalog().find(embed.name);
    check(e != nullptr, "find embed");
    if (e != nullptr) {
        check(e->dtype == DataType::bf16, "dtype bf16");
        check(e->shape == TensorShape::create({4, 8}).value(), "shape 4x8");
        check(e->end - e->begin == 64, "byte range 64");
    }

    // parity: bytes read back must match bytes written
    std::vector<bf16_t> got;
    check(reader.value().read_tensor_bf16(embed.name, got).ok(), "read bf16 tensor");
    check(got.size() == 32, "bf16 element count");
    if (got.size() == 32) {
        check(std::memcmp(got.data(), embed.payload.data(), embed.payload.size()) == 0,
              "bf16 payload parity");
    }

    std::vector<std::uint8_t> f32_buf(36);
    check(reader.value().read_tensor(w1.name, f32_buf.data(), f32_buf.size()).ok(), "read f32");
    check(f32_buf == w1.payload, "f32 payload parity");

    // exact-size contract + typed guard + missing tensor
    check(reader.value().read_tensor(w1.name, f32_buf.data(), 35).code() ==
              Status::Code::insufficient_memory,
          "short destination rejected");
    check(reader.value().read_tensor_bf16(w1.name, got).code() == Status::Code::invalid_state,
          "bf16 read on f32 rejected");
    check(reader.value().read_tensor("nope", f32_buf.data(), 36).code() == Status::Code::not_found,
          "missing tensor -> not_found");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void test_error_probes() {
    {   // header not JSON
        const auto path = st_fixture::tmp_path("notjson");
        st_fixture::write_raw(path, "not json {{", "abc");
        check(safetensors::Reader::open(path).code() == Status::Code::invalid_argument,
              "non-JSON header rejected");
        std::error_code ec; std::filesystem::remove(path, ec);
    }
    {   // payload range beyond data region
        const auto path = st_fixture::tmp_path("oversize");
        st_fixture::write_raw(path,
            "{\"t\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,999]}}",
            std::string(8, '\0'));
        check(safetensors::Reader::open(path).code() == Status::Code::invalid_argument,
              "oversized offset rejected");
        std::error_code ec; std::filesystem::remove(path, ec);
    }
    {   // overlapping ranges
        const auto path = st_fixture::tmp_path("overlap");
        st_fixture::write_raw(path,
            "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},"
            "\"b\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[2,8]}}",
            std::string(8, '\0'));
        check(safetensors::Reader::open(path).code() == Status::Code::invalid_argument,
              "overlapping ranges rejected");
        std::error_code ec; std::filesystem::remove(path, ec);
    }
    {   // shape*dtype vs range mismatch (4 elements * 2 bytes != 3 bytes)
        const auto path = st_fixture::tmp_path("sizemismatch");
        st_fixture::write_raw(path,
            "{\"t\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,3]}}",
            std::string(3, '\0'));
        check(safetensors::Reader::open(path).code() == Status::Code::invalid_argument,
              "range/shape mismatch rejected");
        std::error_code ec; std::filesystem::remove(path, ec);
    }
    {   // unsupported dtype (still unimplemented codec family)
        const auto path = st_fixture::tmp_path("dtype");
        st_fixture::write_raw(path,
            "{\"t\":{\"dtype\":\"F8_E4M3\",\"shape\":[4],\"data_offsets\":[0,4]}}",
            std::string(4, '\0'));
        check(safetensors::Reader::open(path).code() == Status::Code::unsupported,
              "unknown dtype -> unsupported");
        std::error_code ec; std::filesystem::remove(path, ec);
    }
    {   // missing file
        check(safetensors::Reader::open("/tmp/shelf_st_absent.safetensors").code() ==
                  Status::Code::not_found,
              "missing file -> not_found");
    }
    {   // zero tensors is legal
        const auto path = st_fixture::tmp_path("empty");
        st_fixture::write_raw(path, "{}", "");
        auto r = safetensors::Reader::open(path);
        check(r.ok() && r.value().catalog().size() == 0, "empty catalog legal");
        std::error_code ec; std::filesystem::remove(path, ec);
    }
}

} // namespace

int main() {
    test_valid_file();
    test_error_probes();
    std::printf("test_artifact_safetensors: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
