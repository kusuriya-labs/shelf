// shelf unit test — Canonical Tensor Catalog / Weight Binding front (PR-004).
#include <cstdio>
#include <filesystem>
#include <vector>

#include <shelf/artifact/safetensors/reader.hpp>
#include <shelf/weights/catalog/tensor_catalog.hpp>

#include "../support/safetensors_fixture.hpp"

namespace {

using namespace shelf;
using namespace shelf::artifact;
using namespace shelf::weights;

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

std::vector<RawTensorEntry> shard_entries(const std::string& path) {
    auto reader = safetensors::Reader::open(path);
    if (!reader.ok()) return {};
    return reader.value().catalog().entries();
}

void test_multi_shard_resolution() {
    const auto p1 = st_fixture::tmp_path("shard1");
    const auto p2 = st_fixture::tmp_path("shard2");
    st_fixture::write_file(p1, {{"model.embed_tokens.weight", "BF16", {4, 8},
                                 std::vector<std::uint8_t>(64, 1)}});
    st_fixture::write_file(p2, {{"layer.attn.q_proj.weight", "BF16", {8, 8},
                                 std::vector<std::uint8_t>(128, 2)},
                                {"layer.attn.k_norm.weight", "F32", {8},
                                 std::vector<std::uint8_t>(32, 3)}});

    std::vector<RawTensorEntry> merged;
    for (auto& e : shard_entries(p1)) merged.push_back(e);
    for (auto& e : shard_entries(p2)) merged.push_back(e);

    auto catalog = TensorCatalog::from_raw(merged);
    check(catalog.ok(), "multi-shard catalog built");
    if (!catalog.ok()) return;
    check(catalog.value().size() == 3, "3 logical tensors resolved");

    const auto* spec = catalog.value().find("layer.attn.q_proj.weight");
    check(spec != nullptr, "cross-shard name resolved");
    if (spec != nullptr) {
        check(spec->role == TensorRole::weight, "role is weight");
        check(spec->logical_dtype == DataType::bf16, "logical dtype");
        check(spec->shape == TensorShape::create({8, 8}).value(), "shape mapped");
    }
    // ids are stable, distinct, valid
    auto a = catalog.value().id_of("model.embed_tokens.weight");
    auto b = catalog.value().id_of("layer.attn.k_norm.weight");
    check(a.valid() && b.valid() && a != b, "distinct valid ids across shards");
    check(!catalog.value().id_of("does.not.exist").valid(), "unknown name -> invalid id");

    std::error_code ec;
    std::filesystem::remove(p1, ec);
    std::filesystem::remove(p2, ec);
}

void test_consistency_rejections() {
    RawTensorEntry ok{"good.weight", DataType::bf16, TensorShape::create({4}).value(), 0, 8};

    RawTensorEntry dup{"good.weight", DataType::bf16, TensorShape::create({4}).value(), 0, 8};
    auto dup_cat = TensorCatalog::from_raw({ok, dup});
    check(dup_cat.code() == Status::Code::invalid_state, "duplicate name across shards rejected");

    RawTensorEntry bad_size{"bad.weight", DataType::bf16, TensorShape::create({4}).value(), 0, 7};
    check(TensorCatalog::from_raw({bad_size}).code() == Status::Code::invalid_argument,
          "shape/size mismatch rejected");

    RawTensorEntry no_name{"", DataType::f32, TensorShape::create({1}).value(), 0, 4};
    check(TensorCatalog::from_raw({no_name}).code() == Status::Code::invalid_argument,
          "empty name rejected");
}

} // namespace

int main() {
    test_multi_shard_resolution();
    test_consistency_rejections();
    std::printf("test_weights_catalog: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
