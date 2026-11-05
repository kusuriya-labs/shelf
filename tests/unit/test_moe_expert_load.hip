#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/models/qwen35moe/weights/moe_expert_load.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include "moe_expert_fixture.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace moe_expert_fixture;
using shelf::gpu::GpuArena;
using shelf::qwen35moe::build_moe_expert_group_plan;
using shelf::qwen35moe::MoeExpertGroupPlan;
using shelf::qwen35moe::MoeExpertGroupWeights;
using shelf::quantization::bundle::QuantizedModelReader;
using shelf::quantization::mxfp4::Mxfp4CanonicalView;
using shelf::quantization::mxfp4::Mxfp4NativeHost;
using shelf::weights::MatrixEncoding;
using shelf::weights::MatrixWeight;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

std::vector<std::uint8_t> d2h(const void* dev, std::size_t bytes) {
    std::vector<std::uint8_t> host(bytes);
    if (bytes == 0)
        return host;
    if (hipMemcpy(host.data(), dev, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        check(false, "device to host copy");
        host.assign(bytes, 0);
    }
    return host;
}

// Independent assembly of one plane's canonical payload: gate rows first, then up rows,
// experts in ascending order. Written from the fixture maps, not from the loader.
struct CanonicalPlane {
    std::uint32_t rows = 0;
    std::uint64_t k_padded = 0;
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> scales;

    std::uint64_t packed_k() const noexcept { return k_padded / 2u; }
    std::uint64_t blocks() const noexcept { return k_padded / 32u; }

    std::uint8_t code_byte(std::uint32_t expert, std::uint32_t row, std::uint64_t k) const {
        const std::size_t idx = static_cast<std::size_t>((expert * rows + row) * packed_k() + k / 2u);
        return codes[idx];
    }
    std::uint8_t scale_byte(std::uint32_t expert, std::uint32_t row, std::uint64_t block) const {
        return scales[static_cast<std::size_t>((expert * rows + row) * blocks() + block)];
    }
};

CanonicalPlane build_w13(const Fixture& f, const MoeExpertGroupPlan& plan) {
    CanonicalPlane p;
    p.rows = plan.w13_rows;
    p.k_padded = plan.w13_k_padded;
    p.codes.assign(static_cast<std::size_t>(plan.experts) * p.rows * (p.k_padded / 2u), 0u);
    p.scales.assign(static_cast<std::size_t>(plan.experts) * p.rows * (p.k_padded / 32u), 0u);
    const std::uint32_t inter = plan.w13_gate_rows;
    for (std::uint32_t e = 0; e < plan.experts; ++e) {
        for (std::uint32_t r = 0; r < p.rows; ++r) {
            const bool gate = r < inter;
            const std::string base = shelf::qwen35moe::moe_expert_tensor_name(
                plan.layer, e, gate ? "gate_proj" : "up_proj", false);
            const std::uint32_t local = gate ? r : r - inter;
            const auto& codes = f.codes.at(base);
            const auto& scales = f.scales.at(base + "_scale");
            const std::size_t packed_k = static_cast<std::size_t>(p.k_padded / 2u);
            std::memcpy(&p.codes[(static_cast<std::size_t>(e) * p.rows + r) * packed_k],
                        codes.data() + static_cast<std::size_t>(local) * packed_k, packed_k);
            const std::size_t blocks = static_cast<std::size_t>(p.k_padded / 32u);
            std::memcpy(&p.scales[(static_cast<std::size_t>(e) * p.rows + r) * blocks],
                        scales.data() + static_cast<std::size_t>(local) * blocks, blocks);
        }
    }
    return p;
}

CanonicalPlane build_w2(const Fixture& f, const MoeExpertGroupPlan& plan) {
    CanonicalPlane p;
    p.rows = plan.w2_rows;
    p.k_padded = plan.w2_k_padded;
    p.codes.assign(static_cast<std::size_t>(plan.experts) * p.rows * (p.k_padded / 2u), 0u);
    p.scales.assign(static_cast<std::size_t>(plan.experts) * p.rows * (p.k_padded / 32u), 0u);
    for (std::uint32_t e = 0; e < plan.experts; ++e) {
        const std::string base =
            shelf::qwen35moe::moe_expert_tensor_name(plan.layer, e, "down_proj", false);
        const std::size_t packed_k = static_cast<std::size_t>(p.k_padded / 2u);
        const std::size_t blocks = static_cast<std::size_t>(p.k_padded / 32u);
        for (std::uint32_t r = 0; r < p.rows; ++r) {
            std::memcpy(&p.codes[(static_cast<std::size_t>(e) * p.rows + r) * packed_k],
                        f.codes.at(base).data() + static_cast<std::size_t>(r) * packed_k, packed_k);
            std::memcpy(&p.scales[(static_cast<std::size_t>(e) * p.rows + r) * blocks],
                        f.scales.at(base + "_scale").data() + static_cast<std::size_t>(r) * blocks,
                        blocks);
        }
    }
    return p;
}

// The GPU-resident plane must be the documented native layout: codes tiled in
// 16-output x 32-K blocks of 256 bytes, E8M0 scales row-major, one row reference byte
// per output row.
void test_native_layout(const CanonicalPlane& canon, const MatrixWeight& w,
                        const std::string& tag) {
    const std::uint64_t blocks = canon.blocks();
    const std::uint64_t tiles = w.rows / 16u;
    const std::vector<std::uint8_t> codes = d2h(w.codes.data(), w.codes_batch_stride_bytes * w.experts);
    const std::vector<std::uint8_t> scales =
        d2h(w.scales.data(), w.scales_batch_stride_bytes * w.experts);
    const std::vector<std::uint8_t> row_ref = d2h(w.row_ref.data(), std::size_t{w.experts} * w.rows);

    bool codes_ok = codes.size() == static_cast<std::size_t>(w.codes_batch_stride_bytes * w.experts);
    bool scales_ok = true;
    bool ref_ok = true;
    for (std::uint32_t e = 0; e < w.experts && codes_ok; ++e) {
        for (std::uint64_t t = 0; t < tiles; ++t) {
            for (std::uint64_t ib = 0; ib < blocks; ++ib) {
                const std::size_t block_base =
                    static_cast<std::size_t>(e * w.codes_batch_stride_bytes + (t * blocks + ib) * 256u);
                for (std::uint32_t ol = 0; ol < 16u; ++ol) {
                    const std::uint32_t row = static_cast<std::uint32_t>(t * 16u + ol);
                    for (std::uint32_t j = 0; j < 16u; ++j) {
                        const std::uint64_t k_lo = ib * 32u + j;
                        const std::uint64_t k_hi = ib * 32u + 16u + j;
                        const std::uint8_t lo = (canon.code_byte(e, row, k_lo) >> (4u * (k_lo & 1u))) & 0x0Fu;
                        const std::uint8_t hi = (canon.code_byte(e, row, k_hi) >> (4u * (k_hi & 1u))) & 0x0Fu;
                        if (codes[block_base + ol * 16u + j] != static_cast<std::uint8_t>(lo | (hi << 4u)))
                            codes_ok = false;
                    }
                }
            }
        }
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            std::uint8_t ref = 0u;
            for (std::uint64_t ib = 0; ib < blocks; ++ib) {
                const std::size_t idx = static_cast<std::size_t>(e * w.scales_batch_stride_bytes +
                                                                 std::size_t{r} * w.scale_row_stride_bytes + ib);
                if (scales[idx] != canon.scale_byte(e, r, ib))
                    scales_ok = false;
                if (canon.scale_byte(e, r, ib) > ref)
                    ref = canon.scale_byte(e, r, ib);
            }
            if (row_ref[static_cast<std::size_t>(e) * w.rows + r] != ref)
                ref_ok = false;
        }
    }
    check(codes_ok, tag + ": codes hold the 16x32 native block layout");
    check(scales_ok, tag + ": scales hold the row-major E8M0 plane");
    check(ref_ok, tag + ": row reference is the row's maximum E8M0 byte");
}

// Resident expert e must be byte-identical to what the dense rank-2 path stores for the
// same expert: preshuffle one expert's canonical payload and compare.
void test_per_expert_equivalence(const CanonicalPlane& canon, const MatrixWeight& w,
                                 const std::string& tag) {
    for (std::uint32_t e = 0; e < w.experts; ++e) {
        Mxfp4CanonicalView cv;
        cv.batch = 1;
        cv.n = w.rows;
        cv.padded_k = w.k_padded;
        cv.codes = canon.codes.data() + static_cast<std::size_t>(e) * w.rows * canon.packed_k();
        cv.codes_bytes = static_cast<std::size_t>(w.rows) * canon.packed_k();
        cv.scales = canon.scales.data() + static_cast<std::size_t>(e) * w.rows * canon.blocks();
        cv.scales_bytes = static_cast<std::size_t>(w.rows) * canon.blocks();
        if (!cv.validate()) {
            check(false, tag + ": expert canonical view validates e=" + std::to_string(e));
            continue;
        }
        Mxfp4NativeHost native;
        if (!shelf::quantization::mxfp4::preshuffle_mxfp4_native(cv, native)) {
            check(false, tag + ": expert preshuffle e=" + std::to_string(e));
            continue;
        }
        const std::size_t codes_at = static_cast<std::size_t>(e) * w.codes_batch_stride_bytes;
        const std::vector<std::uint8_t> codes = d2h(w.codes.data<std::uint8_t>() + codes_at,
                                                    native.codes.size());
        check(codes.size() == native.codes.size() &&
                  std::memcmp(codes.data(), native.codes.data(), native.codes.size()) == 0,
              tag + ": expert codes match the dense path e=" + std::to_string(e));
        check(std::memcmp(w.scales.data<std::uint8_t>() +
                              static_cast<std::size_t>(e) * w.scales_batch_stride_bytes,
                          native.scales.data(), native.scales.size()) == 0,
              tag + ": expert scales match the dense path e=" + std::to_string(e));
        check(std::memcmp(w.row_ref.data<std::uint8_t>() + static_cast<std::size_t>(e) * w.rows,
                          native.row_ref.data(), native.row_ref.size()) == 0,
              tag + ": expert row reference matches the dense path e=" + std::to_string(e));
        check(native.row_ref_exact == w.row_ref_exact,
              tag + ": exact fold band flag e=" + std::to_string(e));
    }
}

// One fused rank-3 plane's canonical payload, assembled from the checkpoint's bf16
// values with the Shelf codec: gate rows first, experts in ascending order.
CanonicalPlane build_fused_plane(const Fixture& f, const MoeExpertGroupPlan& plan,
                                 bool gate_up) {
    CanonicalPlane p;
    p.rows = gate_up ? plan.w13_rows : plan.w2_rows;
    p.k_padded = gate_up ? plan.w13_k_padded : plan.w2_k_padded;
    p.codes.assign(static_cast<std::size_t>(plan.experts) * p.rows * (p.k_padded / 2u), 0u);
    p.scales.assign(static_cast<std::size_t>(plan.experts) * p.rows * (p.k_padded / 32u), 0u);
    const std::string name = gate_up ? plan.w13_name : plan.w2_name;
    const std::vector<std::uint16_t>& src = f.fused.at(name);
    const std::uint32_t cols = gate_up ? plan.w13_cols : plan.w2_cols;
    std::vector<float> row(p.k_padded, 0.0f);
    for (std::uint32_t e = 0; e < plan.experts; ++e) {
        for (std::uint32_t r = 0; r < p.rows; ++r) {
            for (std::uint32_t c = 0; c < cols; ++c) {
                const std::uint16_t bits = src[(static_cast<std::size_t>(e) * p.rows + r) * cols + c];
                const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
                float value = 0.0f;
                std::memcpy(&value, &word, sizeof(value));
                row[c] = value;
            }
            shelf::quantization::mxfp4::quantize_mxfp4_matrix(
                row.data(), cols, 1u, cols, p.k_padded,
                p.codes.data() + (static_cast<std::size_t>(e) * p.rows + r) * (p.k_padded / 2u),
                p.scales.data() + (static_cast<std::size_t>(e) * p.rows + r) * (p.k_padded / 32u));
        }
    }
    return p;
}

void test_geometry(const MoeExpertGroupPlan& plan, const MoeExpertGroupWeights& g) {
    const auto report = [](const MatrixWeight& w, const char* tag) {
        std::printf("%s: experts=%u rows=%u cols=%u k_padded=%u codes_row=%u scale_row=%u "
                    "codes_batch=%llu scales_batch=%llu row_ref_exact=%d\n",
                    tag, w.experts, w.rows, w.cols, w.k_padded, w.codes_row_stride_bytes,
                    w.scale_row_stride_bytes,
                    static_cast<unsigned long long>(w.codes_batch_stride_bytes),
                    static_cast<unsigned long long>(w.scales_batch_stride_bytes),
                    w.row_ref_exact ? 1 : 0);
    };
    report(g.w13, "w13");
    report(g.w2, "w2");

    check(g.w13.experts == plan.experts && g.w2.experts == plan.experts, "experts per plane");
    check(g.w13.rows == plan.w13_rows && g.w13.cols == plan.w13_cols &&
              g.w13.k_padded == plan.w13_k_padded,
          "w13 geometry");
    check(g.w2.rows == plan.w2_rows && g.w2.cols == plan.w2_cols &&
              g.w2.k_padded == plan.w2_k_padded,
          "w2 geometry");
    check(g.w13.codes_row_stride_bytes == plan.w13_codes_row_stride_bytes &&
              g.w13.scale_row_stride_bytes == plan.w13_scales_row_stride_bytes,
          "w13 native row strides");
    check(g.w2.codes_row_stride_bytes == plan.w2_codes_row_stride_bytes &&
              g.w2.scale_row_stride_bytes == plan.w2_scales_row_stride_bytes,
          "w2 native row strides");
    check(g.w13.codes_batch_stride_bytes == plan.w13_codes_batch_stride_bytes &&
              g.w13.scales_batch_stride_bytes == plan.w13_scales_batch_stride_bytes,
          "w13 batch strides match the plan");
    check(g.w2.codes_batch_stride_bytes == plan.w2_codes_batch_stride_bytes &&
              g.w2.scales_batch_stride_bytes == plan.w2_scales_batch_stride_bytes,
          "w2 batch strides match the plan");
    check(g.w13.encoding == MatrixEncoding::Mxfp4 && g.w2.encoding == MatrixEncoding::Mxfp4,
          "planes are Mxfp4");
    check(g.w13.preshuffled && g.w2.preshuffled, "planes are native");
    check(g.w13.weight_scale_group == plan.scale_block_k && g.w2.weight_scale_group == plan.scale_block_k,
          "weight scale group is 32");
    check(g.w13.row_ref_exact && !g.w2.row_ref_exact,
          "w13 is inside the exact fold band, w2 is not");

    // Dead rows (zero codes, E8M0 byte 1) must stay representable: the row reference is
    // that byte, so no infinity or NaN exponent appears anywhere in the plane.
    bool dead_ok = true;
    for (const MatrixWeight* w : {&g.w13, &g.w2}) {
        const std::vector<std::uint8_t> ref = d2h(w->row_ref.data(), std::size_t{w->experts} * w->rows);
        for (std::uint32_t e = 0; e < w->experts; ++e) {
            if (ref[static_cast<std::size_t>(e) * w->rows] != kE8m0DeadByte)
                dead_ok = false;
        }
    }
    check(dead_ok, "dead rows keep a finite row reference");
}

void test_rejections(const QuantizedModelReader& reader, const MoeExpertGroupPlan& plan,
                     GpuArena& arena, hipStream_t stream) {
    auto bad = plan;
    bad.sources[0].name = "model.language_model.layers.0.mlp.experts.9.down_proj.weight";
    check(!shelf::qwen35moe::load_moe_expert_group(reader, bad, arena, stream).ok(),
          "a missing expert tensor is rejected");

    bad = plan;
    bad.sources[0].rows = plan.w2_rows + 16u;
    check(!shelf::qwen35moe::load_moe_expert_group(reader, bad, arena, stream).ok(),
          "expert rows that do not fit the plane are rejected");

    bad = plan;
    bad.sources[0].row_offset = 16u;
    check(!shelf::qwen35moe::load_moe_expert_group(reader, bad, arena, stream).ok(),
          "a source row range that runs past the plane is rejected");

    bad = plan;
    bad.experts = plan.experts + 1u;
    check(!shelf::qwen35moe::load_moe_expert_group(reader, bad, arena, stream).ok(),
          "a plan whose expert count disagrees with its sources is rejected");
}

}  // namespace

int main() {
    std::printf("=== MoE expert group load (GPU) ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess)
        return 1;

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess)
        return 1;

    auto ar = GpuArena::create(0, 256ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        std::printf("FAIL arena\n");
        return 1;
    }
    GpuArena arena = ar.release();

    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path(ec) / ("shelf_moe_expert_load_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    Fixture f;
    if (!write_fixture(f, root / "mxfp4")) {
        std::printf("FAIL fixture write\n");
        return 1;
    }

    std::vector<std::string> names;
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        for (const auto& name : expert_names(layer))
            names.push_back(name);
        for (const auto& name : mlp_aux_names(layer))
            names.push_back(name);
    }
    auto source = shelf::quantization::bundle::synthesize_hf_quantized_source(root / "mxfp4", names, false);
    if (!source.ok()) {
        std::printf("FAIL hf source: %s\n", source.status().message().c_str());
        return 1;
    }
    auto reader = QuantizedModelReader::open_hf_source(source.release(), false);
    if (!reader.ok()) {
        std::printf("FAIL reader: %s\n", reader.status().message().c_str());
        return 1;
    }

    auto plan = build_moe_expert_group_plan(f.config, 0);
    if (!plan.ok()) {
        std::printf("FAIL plan: %s\n", plan.status().message().c_str());
        return 1;
    }
    const MoeExpertGroupPlan& p = plan.value();

    const std::size_t used_before = arena.used();
    auto loaded = shelf::qwen35moe::load_moe_expert_group(reader.value(), p, arena, stream);
    if (!loaded.ok()) {
        std::printf("FAIL load: %s\n", loaded.status().message().c_str());
        return 1;
    }
    MoeExpertGroupWeights& g = loaded.value();

    test_geometry(p, g);

    const CanonicalPlane w13_canon = build_w13(f, p);
    const CanonicalPlane w2_canon = build_w2(f, p);
    test_native_layout(w13_canon, g.w13, "w13");
    test_native_layout(w2_canon, g.w2, "w2");
    test_per_expert_equivalence(w13_canon, g.w13, "w13");
    test_per_expert_equivalence(w2_canon, g.w2, "w2");

    const std::uint64_t expect_bytes = p.w13_codes_bytes + p.w13_scales_bytes + p.w2_codes_bytes +
                                       p.w2_scales_bytes +
                                       std::uint64_t{p.experts} * (p.w13_row_ref_bytes_per_expert +
                                                                   p.w2_row_ref_bytes_per_expert);
    const std::size_t used = arena.used() - used_before;
    std::printf("arena use: %zu bytes (planes %llu bytes)\n", used,
                static_cast<unsigned long long>(expect_bytes));
    check(used == static_cast<std::size_t>(expect_bytes),
          "the group occupies exactly the planned planes in one arena bump each");

    test_rejections(reader.value(), p, arena, stream);

    {
        // The same checkpoint read with a fused plan asks for tensor names that do not
        // exist there, so it fails instead of assembling an empty plane.
        auto fused_plan = shelf::qwen35moe::build_moe_mtp_expert_group_plan(f.config);
        check(fused_plan.ok(), "the fused expert group plan builds");
        if (fused_plan.ok()) {
            const MoeExpertGroupPlan& mp = fused_plan.value();
            check(mp.fused_rank3 && mp.sources.empty() && !mp.w13_name.empty() &&
                      !mp.w2_name.empty(),
                  "a fused plan names two planes and no per-expert source");
            check(mp.mlp_prefix == shelf::qwen35moe::moe_mtp_mlp_prefix() &&
                      mp.w13_name == mp.mlp_prefix + "experts.gate_up_proj" &&
                      mp.w2_name == mp.mlp_prefix + "experts.down_proj",
                  "a fused plan names the checkpoint's fused tensors");
            check(mp.w13_gate_rows == f.config.ffn.moe_intermediate_size &&
                      mp.w13_rows == 2u * mp.w13_gate_rows && mp.w2_rows == kHidden &&
                      mp.experts == kExperts,
                  "a fused plan keeps the layer's expert geometry");
            auto st = shelf::qwen35moe::load_moe_expert_group(reader.value(), mp, arena, stream);
            check(!st.ok(), "a fused plan over a checkpoint without fused planes fails");
        }
    }

    Fixture fused;
    if (write_fixture(fused, root / "mxfp4_fused", false, false, kInter, kInter, true)) {
        auto fused_plan = shelf::qwen35moe::build_moe_mtp_expert_group_plan(fused.config);
        if (fused_plan.ok()) {
            const MoeExpertGroupPlan& mp = fused_plan.value();
            auto fused_source = shelf::quantization::bundle::synthesize_hf_quantized_source(
                root / "mxfp4_fused", {mp.w13_name, mp.w2_name}, false);
            check(fused_source.ok(), "fused fixture synthesizes");
            if (fused_source.ok()) {
                auto fused_reader =
                    QuantizedModelReader::open_hf_source(fused_source.release(), false);
                check(fused_reader.ok(), "fused fixture reader opens");
                if (fused_reader.ok()) {
                    auto fused_group = shelf::qwen35moe::load_moe_expert_group(
                        fused_reader.value(), mp, arena, stream);
                    check(fused_group.ok(), "the fused planes load");
                    if (fused_group.ok()) {
                        const MoeExpertGroupWeights& g = fused_group.value();
                        check(g.w13.rows == mp.w13_rows && g.w2.rows == mp.w2_rows &&
                                  g.w13.experts == mp.experts,
                              "the fused planes carry the plan geometry");
                        const CanonicalPlane w13_canon = build_fused_plane(fused, mp, true);
                        const CanonicalPlane w2_canon = build_fused_plane(fused, mp, false);
                        test_native_layout(w13_canon, g.w13, "fused w13");
                        test_native_layout(w2_canon, g.w2, "fused w2");
                        test_per_expert_equivalence(w13_canon, g.w13, "fused w13");
                        test_per_expert_equivalence(w2_canon, g.w2, "fused w2");
                    }
                }
            }
        } else {
            check(false, "fused expert group plan builds");
        }
    } else {
        check(false, "fused fixture written");
    }

    Fixture poisoned;
    if (write_fixture(poisoned, root / "mxfp4_nan_byte", true)) {
        auto poison_source = shelf::quantization::bundle::synthesize_hf_quantized_source(
            root / "mxfp4_nan_byte", names, false);
        check(poison_source.ok(), "poisoned fixture synthesizes");
        if (poison_source.ok()) {
            auto poison_reader = QuantizedModelReader::open_hf_source(poison_source.release(), false);
            check(poison_reader.ok(), "poisoned fixture reader opens");
            if (poison_reader.ok()) {
                auto st = shelf::qwen35moe::load_moe_expert_group(poison_reader.value(), p, arena, stream);
                check(!st.ok(), "an E8M0 NaN byte in an expert scale is rejected");
                if (!st.ok())
                    check(st.status().code() == shelf::Status::Code::invalid_argument,
                          "the E8M0 NaN byte rejection is invalid_argument");
            }
        }
    } else {
        check(false, "poisoned fixture written");
    }

    fs::remove_all(root, ec);
    arena.shutdown();
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    std::printf("test_moe_expert_load: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
