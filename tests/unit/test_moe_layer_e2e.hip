#include <shelf/core/memory/types.h>
#include <shelf/io/safetensors_reader.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/optimized/linear/mxfp4.h>
#include <shelf/kernels/optimized/moe/combine.h>
#include <shelf/kernels/optimized/moe/router.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/moe_expert_load.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/quantization/codec/e4m3.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/weights/matrix_weight.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef SHELF_MOE_FIXTURE_DIR
#error "SHELF_MOE_FIXTURE_DIR is not set"
#endif

namespace quant = shelf::quantization;
namespace rt = shelf::runtime;
namespace moe = shelf::kernel;

namespace {

constexpr uint32_t kRowTile = moe::kMxfp4GemmRowTile;
const std::string kFixtureDir = SHELF_MOE_FIXTURE_DIR;

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

float bf16_to_f32(shelf::bf16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

template <typename T>
T* alloc(std::size_t count) {
    T* p = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)) != hipSuccess) {
        std::printf("FAIL hipMalloc %zu\n", count * sizeof(T));
        std::exit(1);
    }
    return p;
}

template <typename T>
std::vector<T> d2h(const T* dev, std::size_t count) {
    std::vector<T> host(count);
    if (count == 0)
        return host;
    if (hipMemcpy(host.data(), dev, count * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL d2h %zu\n", count);
        std::exit(1);
    }
    return host;
}

template <typename T>
void h2d(T* dev, const std::vector<T>& host) {
    if (hipMemcpy(dev, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL h2d %zu\n", host.size());
        std::exit(1);
    }
}

template <typename T>
std::vector<T> read_tensor(const shelf::io::SafetensorsReader& reader, const std::string& name,
                           shelf::io::SType expect) {
    auto spec = reader.tensor_spec(name);
    if (!spec.ok()) {
        std::printf("FAIL fixture tensor %s: %s\n", name.c_str(), spec.status().message().c_str());
        std::exit(1);
    }
    if (spec.value().dtype != expect) {
        std::printf("FAIL fixture tensor %s has the wrong dtype\n", name.c_str());
        std::exit(1);
    }
    std::size_t bytes = 0;
    auto payload = reader.tensor_data(name, bytes);
    if (!payload.ok() || payload.value() == nullptr) {
        std::printf("FAIL fixture tensor %s payload\n", name.c_str());
        std::exit(1);
    }
    const void* data = payload.value();
    std::vector<T> out(bytes / sizeof(T));
    std::memcpy(out.data(), data, bytes);
    return out;
}

// Canonical MXFP4 payload of one expert projection, read from the checkpoint bytes.
struct CanonicalWeights {
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
    uint32_t rows = 0;
    uint32_t k_padded = 0;

    double at(uint32_t row, uint32_t k) const {
        const uint8_t byte = codes[static_cast<std::size_t>(row) * (k_padded / 2u) + (k >> 1)];
        const uint8_t code = (k & 1u) == 0u ? static_cast<uint8_t>(byte & 0x0Fu)
                                            : static_cast<uint8_t>(byte >> 4);
        const uint8_t scale_byte = scales[static_cast<std::size_t>(row) * (k_padded / 32u) +
                                          (k / quant::kMxfp4ScaleBlockK)];
        return static_cast<double>(quant::mxfp4::e2m1_decode_f32(code)) *
               static_cast<double>(quant::mxfp4::e8m0_decode_f32(scale_byte));
    }
};

CanonicalWeights read_canonical(const shelf::io::SafetensorsReader& reader,
                                const std::string& base, uint32_t rows, uint32_t k_padded) {
    CanonicalWeights w;
    w.rows = rows;
    w.k_padded = k_padded;
    w.codes = read_tensor<uint8_t>(reader, base, shelf::io::SType::U8);
    w.scales = read_tensor<uint8_t>(reader, base + "_scale", shelf::io::SType::U8);
    if (w.codes.size() != static_cast<std::size_t>(rows) * (k_padded / 2u) ||
        w.scales.size() != static_cast<std::size_t>(rows) * (k_padded / 32u)) {
        std::printf("FAIL fixture payload size for %s\n", base.c_str());
        std::exit(1);
    }
    return w;
}

// The activation operand values the GPU actually multiplies: E4M3 codes in fragment
// order, folded against the row reference in scale column 0.
std::vector<double> decode_activation_plane(const std::vector<uint8_t>& codes,
                                            const std::vector<float>& scales,
                                            const rt::Int8ActivationWorkspaceLayout& layout,
                                            uint32_t rows) {
    const uint32_t scale_floats = layout.scale_row_stride_bytes / sizeof(float);
    std::vector<double> out(static_cast<std::size_t>(rows) * layout.k_padded, 0.0);
    for (uint32_t row = 0; row < rows; ++row) {
        const double ref = static_cast<double>(
            scales[static_cast<std::size_t>(row) * scale_floats]);
        const size_t base = static_cast<size_t>(row >> 4) *
            (static_cast<size_t>(layout.code_row_stride_bytes) * 16u);
        for (uint32_t k = 0; k < layout.k_padded; ++k) {
            const size_t off = static_cast<size_t>(k >> 6) * quant::fp8::kFragBlockBytes +
                               quant::fp8::fp8_frag_offset(row & 15u, k & 63u);
            out[static_cast<std::size_t>(row) * layout.k_padded + k] =
                static_cast<double>(quant::codec::e4m3_decode_f32(codes[base + off])) * ref;
        }
    }
    return out;
}

void compare_stage(const std::vector<shelf::bf16_t>& got, const std::vector<double>& want,
                   uint32_t rows, uint32_t features, const std::string& tag) {
    double l2_want = 0.0;
    double l2_diff = 0.0;
    double worst = 0.0;
    std::size_t exact = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t f = 0; f < features; ++f) {
            const std::size_t at = static_cast<std::size_t>(r) * features + f;
            const double a = want[at];
            const double b = bf16_to_f32(got[at]);
            l2_want += a * a;
            l2_diff += (a - b) * (a - b);
            worst = std::max(worst, std::fabs(a - b));
            if (got[at].data == shelf::f32_to_bf16_rne(static_cast<float>(a)))
                ++exact;
        }
    }
    const double rel = std::sqrt(l2_diff) / std::max(std::sqrt(l2_want), 1e-9);
    const double ratio = rows ? static_cast<double>(exact) / static_cast<double>(
                                    static_cast<std::size_t>(rows) * features)
                              : 0.0;
    check(ratio >= 0.999, tag + ": bf16 exact >= 99.9% of the f64 chain (" +
                               std::to_string(ratio) + ")");
    check(rel < 4.0e-3, tag + ": rel_l2 < 4e-3 of the f64 chain (" + std::to_string(rel) + ")");
    std::printf("%s: rows %u, bf16 exact %.4f, rel_l2 %.3e, max_abs %.3e\n", tag.c_str(), rows,
                ratio, rel, worst);
}

}  // namespace

int main() {
    std::printf("=== MoE 1 layer end to end against the HF f64 fixture (GPU) ===\n");
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

    auto cfg = shelf::qwen35moe::read_qwen35_moe_text_config(kFixtureDir);
    if (!cfg.ok()) {
        std::printf("FAIL fixture config: %s\n", cfg.status().message().c_str());
        return 1;
    }
    const shelf::qwen35moe::Qwen35MoeTextConfig& config = cfg.value();

    auto oracle_open = shelf::io::SafetensorsReader::open(kFixtureDir + "/oracle.safetensors");
    if (!oracle_open.ok()) {
        std::printf("FAIL open oracle.safetensors: %s\n", oracle_open.status().message().c_str());
        return 1;
    }
    const shelf::io::SafetensorsReader& oracle_reader = oracle_open.value();

    const auto xn = read_tensor<shelf::bf16_t>(oracle_reader, "xn", shelf::io::SType::BF16);
    const auto hf_expert_out = read_tensor<float>(oracle_reader, "expected_routed",
                                                  shelf::io::SType::F32);
    const auto hf_shared = read_tensor<float>(oracle_reader, "shared_expert_out",
                                              shelf::io::SType::F32);
    const auto hf_gate = read_tensor<float>(oracle_reader, "shared_gate_sigmoid",
                                            shelf::io::SType::F32);
    const auto gate_logit = read_tensor<shelf::bf16_t>(oracle_reader, "shared_gate_logit",
                                                       shelf::io::SType::BF16);
    const auto hf_selected = read_tensor<int32_t>(oracle_reader, "hf_selected_experts",
                                                  shelf::io::SType::I32);
    const auto hf_weights = read_tensor<float>(oracle_reader, "hf_routing_weights",
                                               shelf::io::SType::F32);
    const uint32_t tokens = static_cast<uint32_t>(hf_gate.size());
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t experts = static_cast<uint32_t>(config.ffn.num_experts);
    const uint32_t top_k = static_cast<uint32_t>(config.ffn.num_experts_per_tok);
    const uint32_t inter = static_cast<uint32_t>(config.ffn.moe_intermediate_size);
    if (xn.size() != static_cast<std::size_t>(tokens) * hidden ||
        hf_expert_out.size() != static_cast<std::size_t>(tokens) * hidden) {
        std::printf("FAIL fixture shapes\n");
        return 1;
    }
    std::printf("fixture: tokens %u hidden %u inter %u experts %u top_k %u\n", tokens, hidden,
                inter, experts, top_k);

    std::vector<std::string> names;
    const std::string mlp = "model.language_model.layers.0.mlp.";
    for (uint32_t e = 0; e < experts; ++e)
        for (const char* proj : {"down_proj", "gate_proj", "up_proj"})
            names.push_back(shelf::qwen35moe::moe_expert_tensor_name(0, e, proj, false));
    names.push_back(mlp + "gate.weight");
    names.push_back(mlp + "shared_expert_gate.weight");
    for (const char* proj : {"gate_proj", "up_proj", "down_proj"})
        names.push_back(mlp + "shared_expert." + proj + ".weight");

    auto source = quant::bundle::synthesize_hf_quantized_source(kFixtureDir, names, false);
    if (!source.ok()) {
        std::printf("FAIL hf source: %s\n", source.status().message().c_str());
        return 1;
    }
    auto reader = quant::bundle::QuantizedModelReader::open_hf_source(source.release(), false);
    if (!reader.ok()) {
        std::printf("FAIL reader: %s\n", reader.status().message().c_str());
        return 1;
    }
    auto plan = shelf::qwen35moe::build_moe_expert_group_plan(config, 0);
    if (!plan.ok()) {
        std::printf("FAIL plan: %s\n", plan.status().message().c_str());
        return 1;
    }
    const shelf::qwen35moe::MoeExpertGroupPlan& p = plan.value();
    check(p.w13_rows == 2u * inter && p.w13_cols == hidden && p.w2_rows == hidden &&
              p.w2_cols == inter,
          "fixture plan geometry");

    auto arena = shelf::gpu::GpuArena::create(0, 64ull * 1024ull * 1024ull);
    if (!arena.ok()) {
        std::printf("FAIL arena\n");
        return 1;
    }
    shelf::gpu::GpuArena gpu_arena = arena.release();
    auto loaded = shelf::qwen35moe::load_moe_expert_group(reader.value(), p, gpu_arena, stream);
    if (!loaded.ok()) {
        std::printf("FAIL expert load: %s\n", loaded.status().message().c_str());
        return 1;
    }
    const shelf::weights::MatrixWeight& w13 = loaded.value().w13;
    const shelf::weights::MatrixWeight& w2 = loaded.value().w2;
    check(w13.experts == experts && w2.experts == experts, "loaded planes carry the expert count");

    auto shard_open = shelf::io::SafetensorsReader::open(kFixtureDir + "/model.safetensors");
    if (!shard_open.ok()) {
        std::printf("FAIL open model.safetensors\n");
        return 1;
    }
    const shelf::io::SafetensorsReader& shard_reader = shard_open.value();

    std::vector<CanonicalWeights> w13_canon(experts);
    std::vector<CanonicalWeights> w2_canon(experts);
    for (uint32_t e = 0; e < experts; ++e) {
        const std::vector<uint8_t> gate_codes = read_tensor<uint8_t>(
            shard_reader, mlp + "experts." + std::to_string(e) + ".gate_proj.weight",
            shelf::io::SType::U8);
        const std::vector<uint8_t> gate_scales = read_tensor<uint8_t>(
            shard_reader, mlp + "experts." + std::to_string(e) + ".gate_proj.weight_scale",
            shelf::io::SType::U8);
        const std::vector<uint8_t> up_codes = read_tensor<uint8_t>(
            shard_reader, mlp + "experts." + std::to_string(e) + ".up_proj.weight",
            shelf::io::SType::U8);
        const std::vector<uint8_t> up_scales = read_tensor<uint8_t>(
            shard_reader, mlp + "experts." + std::to_string(e) + ".up_proj.weight_scale",
            shelf::io::SType::U8);
        CanonicalWeights& fused = w13_canon[e];
        fused.rows = p.w13_rows;
        fused.k_padded = p.w13_k_padded;
        fused.codes = gate_codes;
        fused.codes.insert(fused.codes.end(), up_codes.begin(), up_codes.end());
        fused.scales = gate_scales;
        fused.scales.insert(fused.scales.end(), up_scales.begin(), up_scales.end());
        w2_canon[e] = read_canonical(shard_reader,
                                     mlp + "experts." + std::to_string(e) + ".down_proj.weight",
                                     p.w2_rows, p.w2_k_padded);
    }

    {
        const auto ref_gate = read_tensor<shelf::bf16_t>(
            oracle_reader, "reference_bf16.experts.0.gate_proj.weight", shelf::io::SType::BF16);
        std::vector<float> src(ref_gate.size());
        for (std::size_t i = 0; i < ref_gate.size(); ++i)
            src[i] = bf16_to_f32(ref_gate[i]);
        std::vector<uint8_t> codes(src.size() / 2u), scales(src.size() / 32u);
        quant::mxfp4::quantize_mxfp4_matrix(src.data(), p.w13_cols, inter, p.w13_cols,
                                            p.w13_k_padded, codes.data(), scales.data());
        const std::vector<uint8_t> fixture_codes = read_tensor<uint8_t>(
            shard_reader, mlp + "experts.0.gate_proj.weight", shelf::io::SType::U8);
        const std::vector<uint8_t> fixture_scales = read_tensor<uint8_t>(
            shard_reader, mlp + "experts.0.gate_proj.weight_scale", shelf::io::SType::U8);
        check(codes == fixture_codes && scales == fixture_scales,
              "the Shelf codec reproduces the fixture payload from its BF16 weights");
    }

    const uint32_t group_capacity = moe::moe_group_count_upper_bound(tokens, experts, top_k);
    const uint32_t rows_cap = moe::moe_group_row_capacity(tokens, experts, top_k);

    auto router_weight = reader.value().resolve(mlp + "gate.weight");
    if (!router_weight.ok()) {
        std::printf("FAIL resolve router weight\n");
        return 1;
    }
    std::vector<float> logits(static_cast<std::size_t>(tokens) * experts, 0.0f);
    {
        const auto* gate_bf16 = static_cast<const shelf::bf16_t*>(router_weight.value().data.data);
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t e = 0; e < experts; ++e) {
                float acc = 0.0f;
                for (uint32_t h = 0; h < hidden; ++h)
                    acc += bf16_to_f32(xn[static_cast<std::size_t>(t) * hidden + h]) *
                           bf16_to_f32(gate_bf16[static_cast<std::size_t>(e) * hidden + h]);
                logits[static_cast<std::size_t>(t) * experts + e] = acc;
            }
        }
    }

    float* d_logits = alloc<float>(logits.size());
    int32_t* d_sel_expert = alloc<int32_t>(static_cast<std::size_t>(tokens) * top_k);
    float* d_sel_weight = alloc<float>(static_cast<std::size_t>(tokens) * top_k);
    uint32_t* d_tile_counts = alloc<uint32_t>(
        static_cast<std::size_t>(experts) * moe::moe_slot_tile_count(tokens));
    uint32_t* d_row_begin = alloc<uint32_t>(experts + 1u);
    uint32_t* d_tile_base = alloc<uint32_t>(
        static_cast<std::size_t>(experts) * moe::moe_slot_tile_count(tokens));
    uint32_t* d_group_begin = alloc<uint32_t>(experts + 1u);
    uint32_t* d_group_expert = alloc<uint32_t>(group_capacity);
    uint32_t* d_group_rows = alloc<uint32_t>(group_capacity);
    uint32_t* d_group_count = alloc<uint32_t>(1);
    int32_t* d_row_token = alloc<int32_t>(rows_cap);
    int32_t* d_slot_row_of = alloc<int32_t>(static_cast<std::size_t>(tokens) * top_k);
    const auto plane_layout13 = rt::Int8ActivationWorkspaceLayout::make(p.w13_k_padded, rows_cap);
    const auto plane_layout2 = rt::Int8ActivationWorkspaceLayout::make(p.w2_k_padded, rows_cap);
    const std::size_t plane_bytes = std::max(plane_layout13.total_bytes, plane_layout2.total_bytes);
    uint8_t* d_plane = alloc<uint8_t>(plane_bytes);
    uint8_t* d_plane_codes = d_plane;
    float* d_plane_scales = reinterpret_cast<float*>(d_plane +
        std::max(plane_layout13.scales_offset, plane_layout2.scales_offset));
    shelf::bf16_t* d_hidden13 = alloc<shelf::bf16_t>(static_cast<std::size_t>(tokens) * hidden);
    shelf::bf16_t* d_hidden2 = alloc<shelf::bf16_t>(static_cast<std::size_t>(rows_cap) * inter);
    shelf::bf16_t* d_out13 = alloc<shelf::bf16_t>(static_cast<std::size_t>(rows_cap) * p.w13_rows);
    shelf::bf16_t* d_out2 = alloc<shelf::bf16_t>(static_cast<std::size_t>(rows_cap) * p.w2_rows);
    shelf::bf16_t* d_shared = alloc<shelf::bf16_t>(static_cast<std::size_t>(tokens) * hidden);
    shelf::bf16_t* d_gate_bf16 = alloc<shelf::bf16_t>(tokens);
    float* d_gate = alloc<float>(tokens);
    shelf::bf16_t* d_combined = alloc<shelf::bf16_t>(static_cast<std::size_t>(tokens) * hidden);
    shelf::bf16_t* d_routed = alloc<shelf::bf16_t>(static_cast<std::size_t>(tokens) * hidden);

    h2d(d_logits, logits);
    h2d(d_hidden13, xn);
    std::vector<shelf::bf16_t> shared_bf16(hf_shared.size());
    for (std::size_t i = 0; i < hf_shared.size(); ++i)
        shared_bf16[i] = shelf::bf16_t{shelf::f32_to_bf16_rne(hf_shared[i])};
    h2d(d_shared, shared_bf16);
    // In production shared_expert_gate is a [1, H] linear whose bf16 output is activated by
    // the sigmoid elementwise kernel, so the fixture hands us that bf16 output directly.
    h2d(d_gate_bf16, gate_logit);
    check(shelf::kernel::launch_sigmoid_bf16_f32(d_gate_bf16, 1u, d_gate, 1u, tokens, 1u, stream) ==
              hipSuccess,
          "shared expert gate sigmoid launch");

    if (hipMemset(d_tile_counts, 0,
                  static_cast<std::size_t>(experts) * moe::moe_slot_tile_count(tokens) *
                      sizeof(uint32_t)) != hipSuccess)
        check(false, "counts memset");
    check(moe::launch_moe_router(d_logits, experts, d_sel_expert, d_sel_weight, d_tile_counts,
                                 tokens, experts, top_k, nullptr, stream) == hipSuccess,
          "router launch");
    check(moe::launch_moe_slot_scan(d_tile_counts, d_row_begin, d_tile_base, d_group_begin,
                                    d_group_expert, d_group_rows, d_group_count, d_row_token,
                                    tokens, experts, top_k, stream) == hipSuccess,
          "scan launch");
    check(moe::launch_moe_slot_scatter(d_sel_expert, d_tile_base, d_row_begin, d_group_begin,
                                       d_row_token, d_slot_row_of, tokens, experts, top_k,
                                       stream) == hipSuccess,
          "scatter launch");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL plan sync\n");
        return 1;
    }

    const auto sel_expert = d2h(d_sel_expert, static_cast<std::size_t>(tokens) * top_k);
    const auto sel_weight = d2h(d_sel_weight, static_cast<std::size_t>(tokens) * top_k);
    const auto group_expert = d2h(d_group_expert, group_capacity);
    const auto group_rows = d2h(d_group_rows, group_capacity);
    const auto row_token = d2h(d_row_token, rows_cap);
    const auto slot_row_of = d2h(d_slot_row_of, static_cast<std::size_t>(tokens) * top_k);
    uint32_t group_count = 0;
    group_count = d2h(d_group_count, 1).at(0);

    bool selection_matches_hf = true;
    double worst_weight = 0.0;
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<int32_t> ours, theirs;
        for (uint32_t j = 0; j < top_k; ++j) {
            ours.push_back(sel_expert[static_cast<std::size_t>(t) * top_k + j]);
            theirs.push_back(hf_selected[static_cast<std::size_t>(t) * top_k + j]);
        }
        if (ours != theirs)
            selection_matches_hf = false;
        for (uint32_t j = 0; j < top_k; ++j) {
            const double want = hf_weights[static_cast<std::size_t>(t) * top_k + j];
            const double got = sel_weight[static_cast<std::size_t>(t) * top_k + j];
            worst_weight = std::max(worst_weight, std::fabs(want - got) / std::max(want, 1e-9));
        }
    }
    check(selection_matches_hf, "the device router selects the same experts as HF float64");
    check(worst_weight < 1.0e-5, "routing weights match HF within 1e-5 (" +
                                     std::to_string(worst_weight) + ")");

    const auto layout13 = rt::Int8ActivationWorkspaceLayout::make(p.w13_k_padded, rows_cap);
    const auto layout2 = rt::Int8ActivationWorkspaceLayout::make(p.w2_k_padded, rows_cap);
    check(moe::launch_activation_quantize_mxfp4_rowref(
              d_hidden13, d_plane_codes, d_plane_scales, rows_cap, p.w13_cols, p.w13_k_padded,
              p.w13_cols, layout13.code_row_stride_bytes, layout13.scale_row_stride_bytes,
              d_row_token, stream) == hipSuccess,
          "quantize the routed activation in group row order");
    const auto plane13_codes = d2h(d_plane_codes, layout13.codes_bytes());
    const auto plane13_scales = d2h(d_plane_scales, layout13.scales_bytes() / 4u);

    check(moe::launch_gemm_mxfp4_w4a8_grouped(
              w13.codes.data<uint8_t>(), w13.scales.data<uint8_t>(), nullptr,
              reinterpret_cast<const int8_t*>(d_plane_codes),
              d_plane_scales, d_out13, moe::Mxfp4GemmOutputDType::BF16, p.w13_rows,
              p.w13_k_padded, p.w13_rows, w13.codes_batch_stride_bytes,
              w13.scales_batch_stride_bytes, w13.scale_row_stride_bytes,
              layout13.code_row_stride_bytes, layout13.scale_row_stride_bytes, p.w13_rows,
              d_group_expert, d_group_rows, d_group_count, group_capacity,
              stream) == hipSuccess,
          "grouped w13 launch");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL grouped w13 sync\n");
        return 1;
    }
    const auto y13 = d2h(d_out13, static_cast<std::size_t>(rows_cap) * p.w13_rows);

    const auto act13 = decode_activation_plane(plane13_codes, plane13_scales, layout13, rows_cap);
    std::vector<double> want13(static_cast<std::size_t>(rows_cap) * p.w13_rows, 0.0);
    for (uint32_t g = 0; g < group_count; ++g) {
        const CanonicalWeights& cw = w13_canon[group_expert[g]];
        for (uint32_t i = 0; i < group_rows[g]; ++i) {
            const uint32_t row = g * kRowTile + i;
            for (uint32_t o = 0; o < p.w13_rows; ++o) {
                double acc = 0.0;
                for (uint32_t k = 0; k < p.w13_k_padded; ++k)
                    acc += act13[static_cast<std::size_t>(row) * p.w13_k_padded + k] *
                           cw.at(o, k);
                want13[static_cast<std::size_t>(row) * p.w13_rows + o] = acc;
            }
        }
    }
    compare_stage(y13, want13, group_count * kRowTile, p.w13_rows, "layer/w13");

    // Activation quantization is the numerical class difference of this path, so it is
    // measured on its own: the same GEMM on the exact bf16 activations.
    {
        double l2_want = 0.0;
        double l2_diff = 0.0;
        std::size_t compared = 0;
        for (uint32_t g = 0; g < group_count; ++g) {
            const CanonicalWeights& cw = w13_canon[group_expert[g]];
            for (uint32_t i = 0; i < group_rows[g]; ++i) {
                const uint32_t row = g * kRowTile + i;
                const uint32_t token = static_cast<uint32_t>(row_token[row]);
                for (uint32_t o = 0; o < p.w13_rows; ++o) {
                    double acc = 0.0;
                    for (uint32_t k = 0; k < p.w13_k_padded; ++k)
                        acc += static_cast<double>(bf16_to_f32(
                                   xn[static_cast<std::size_t>(token) * hidden + k])) *
                               cw.at(o, k);
                    const std::size_t at = static_cast<std::size_t>(row) * p.w13_rows + o;
                    l2_want += acc * acc;
                    l2_diff += (acc - bf16_to_f32(y13[at])) * (acc - bf16_to_f32(y13[at]));
                    ++compared;
                }
            }
        }
        std::printf("layer/w13 vs exact bf16 activations (E4M3 activation class): rel_l2 %.3e\n",
                    std::sqrt(l2_diff) / std::max(std::sqrt(l2_want), 1e-9));
    }

    // Production reaches this point through the SWIGLU primitive; here it is applied on the
    // host over the grouped w13 output, which is the same bf16 operands the kernel would
    // read. Padding rows stay zero and are re-pointed at -1 by the row index below.
    std::vector<shelf::bf16_t> swiglu(static_cast<std::size_t>(rows_cap) * inter, shelf::bf16_t{0});
    for (uint32_t g = 0; g < group_count; ++g) {
        for (uint32_t i = 0; i < group_rows[g]; ++i) {
            const uint32_t row = g * kRowTile + i;
            for (uint32_t c = 0; c < inter; ++c) {
                const double gate = bf16_to_f32(y13[static_cast<std::size_t>(row) * p.w13_rows + c]);
                const double up =
                    bf16_to_f32(y13[static_cast<std::size_t>(row) * p.w13_rows + inter + c]);
                swiglu[static_cast<std::size_t>(row) * inter + c] = shelf::bf16_t{shelf::f32_to_bf16_rne(
                    static_cast<float>((gate / (1.0 + std::exp(-gate))) * up))};
            }
        }
    }

    // The swiglu plane is already in group row order, so the second quantizer maps each
    // row to itself; only the padding rows are re-pointed at -1, which keeps them zero
    // instead of quantizing the bytes the grouped GEMM never stored.
    std::vector<int32_t> swiglu_row_index(rows_cap, -1);
    for (uint32_t g = 0; g < group_count; ++g)
        for (uint32_t i = 0; i < group_rows[g]; ++i)
            swiglu_row_index[g * kRowTile + i] = static_cast<int32_t>(g * kRowTile + i);
    int32_t* d_swiglu_row_index = alloc<int32_t>(rows_cap);
    h2d(d_swiglu_row_index, swiglu_row_index);

    h2d(d_hidden2, swiglu);
    check(moe::launch_activation_quantize_mxfp4_rowref(
              d_hidden2, d_plane_codes, d_plane_scales, rows_cap, p.w2_cols, p.w2_k_padded,
              p.w2_cols, layout2.code_row_stride_bytes, layout2.scale_row_stride_bytes,
              d_swiglu_row_index, stream) == hipSuccess,
          "quantize the swiglu activation in group row order");
    const auto plane2_codes = d2h(d_plane_codes, layout2.codes_bytes());
    const auto plane2_scales = d2h(d_plane_scales, layout2.scales_bytes() / 4u);

    check(moe::launch_gemm_mxfp4_w4a8_grouped(
              w2.codes.data<uint8_t>(), w2.scales.data<uint8_t>(), nullptr,
              reinterpret_cast<const int8_t*>(d_plane_codes),
              d_plane_scales, d_out2, moe::Mxfp4GemmOutputDType::BF16, p.w2_rows, p.w2_k_padded,
              p.w2_rows, w2.codes_batch_stride_bytes, w2.scales_batch_stride_bytes,
              w2.scale_row_stride_bytes, layout2.code_row_stride_bytes,
              layout2.scale_row_stride_bytes, p.w2_rows, d_group_expert, d_group_rows,
              d_group_count, group_capacity, stream) == hipSuccess,
          "grouped w2 launch");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL grouped w2 sync\n");
        return 1;
    }
    const auto y2 = d2h(d_out2, static_cast<std::size_t>(rows_cap) * p.w2_rows);

    const auto act2 = decode_activation_plane(plane2_codes, plane2_scales, layout2, rows_cap);
    std::vector<double> want2(static_cast<std::size_t>(rows_cap) * p.w2_rows, 0.0);
    for (uint32_t g = 0; g < group_count; ++g) {
        const CanonicalWeights& cw = w2_canon[group_expert[g]];
        for (uint32_t i = 0; i < group_rows[g]; ++i) {
            const uint32_t row = g * kRowTile + i;
            for (uint32_t o = 0; o < p.w2_rows; ++o) {
                double acc = 0.0;
                for (uint32_t k = 0; k < p.w2_k_padded; ++k)
                    acc += act2[static_cast<std::size_t>(row) * p.w2_k_padded + k] * cw.at(o, k);
                want2[static_cast<std::size_t>(row) * p.w2_rows + o] = acc;
            }
        }
    }
    compare_stage(y2, want2, group_count * kRowTile, p.w2_rows, "layer/w2");

    // The routed chain alone is what this path's numerical class decides (E4M3 group 128
    // activations, MXFP4 weights), so it is measured by itself. The layer total adds the
    // shared expert term, whose magnitude changes a relative measure of the sum without
    // changing the per-element error.
    check(moe::launch_moe_combine(d_routed, d_out2, d_sel_weight, d_slot_row_of, nullptr, nullptr,
                                  tokens, top_k, hidden, p.w2_rows, hidden, 1u, stream) == hipSuccess,
          "combine launch without the shared expert");
    check(moe::launch_moe_combine(d_combined, d_out2, d_sel_weight, d_slot_row_of, d_shared,
                                  d_gate, tokens, top_k, hidden, p.w2_rows, hidden, 1u,
                                  stream) == hipSuccess,
          "combine launch with the shared expert");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL combine sync\n");
        return 1;
    }
    const auto combined = d2h(d_combined, static_cast<std::size_t>(tokens) * hidden);
    const auto routed_only = d2h(d_routed, static_cast<std::size_t>(tokens) * hidden);
    const auto device_gate = d2h(d_gate, tokens);

    {
        double l2_hf = 0.0;
        double l2_diff = 0.0;
        double l2_ours = 0.0;
        std::size_t exact = 0;
        for (std::size_t i = 0; i < hf_expert_out.size(); ++i) {
            const double want = hf_expert_out[i];
            const double got = bf16_to_f32(routed_only[i]);
            l2_hf += want * want;
            l2_ours += got * got;
            l2_diff += (want - got) * (want - got);
            if (routed_only[i].data == shelf::f32_to_bf16_rne(static_cast<float>(want)))
                ++exact;
        }
        const double rel = std::sqrt(l2_diff) / std::max(std::sqrt(l2_hf), 1e-9);
        std::printf("routed vs HF float64: rel_l2 %.3e, rms ours %.4f rms HF %.4f, bf16 exact %.4f\n",
                    rel, std::sqrt(l2_ours / static_cast<double>(hf_expert_out.size())),
                    std::sqrt(l2_hf / static_cast<double>(hf_expert_out.size())),
                    static_cast<double>(exact) / static_cast<double>(hf_expert_out.size()));
        // The E4M3 activation class on this fixture measures 5.0e-2 (and a numpy model of
        // the same pipeline reproduces it), so 8e-2 is the gate.
        check(rel < 8.0e-2, "the routed expert chain stays within 8e-2 rel_l2 of HF float64 (" +
                                std::to_string(rel) + ")");
    }

    std::vector<double> want_combined(static_cast<std::size_t>(tokens) * hidden, 0.0);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<std::pair<int32_t, float>> slots;
        for (uint32_t j = 0; j < top_k; ++j) {
            const std::size_t at = static_cast<std::size_t>(t) * top_k + j;
            slots.push_back({slot_row_of[at], sel_weight[at]});
        }
        std::sort(slots.begin(), slots.end());
        const float gate = device_gate[t];
        for (uint32_t f = 0; f < hidden; ++f) {
            float acc = 0.0f;
            for (const auto& slot : slots)
                acc = std::fmaf(slot.second,
                                bf16_to_f32(y2[static_cast<std::size_t>(slot.first) * p.w2_rows +
                                                   f]),
                                acc);
            acc = std::fmaf(gate, bf16_to_f32(shared_bf16[static_cast<std::size_t>(t) * hidden +
                                                          f]),
                            acc);
            want_combined[static_cast<std::size_t>(t) * hidden + f] = static_cast<double>(acc);
        }
    }
    bool combine_bit_exact = true;
    for (std::size_t i = 0; i < want_combined.size(); ++i)
        if (combined[i].data != shelf::f32_to_bf16_rne(static_cast<float>(want_combined[i])))
            combine_bit_exact = false;
    check(combine_bit_exact, "the combined layer output is bit identical to the f64 chain");

    double l2_hf = 0.0;
    double l2_diff_hf = 0.0;
    double worst_hf = 0.0;
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t f = 0; f < hidden; ++f) {
            const std::size_t at = static_cast<std::size_t>(t) * hidden + f;
            const double want = hf_expert_out[at] + static_cast<double>(hf_gate[t]) *
                                                        static_cast<double>(hf_shared[at]);
            const double got = bf16_to_f32(combined[at]);
            l2_hf += want * want;
            l2_diff_hf += (want - got) * (want - got);
            worst_hf = std::max(worst_hf, std::fabs(want - got));
        }
    }
    const double rel_hf = std::sqrt(l2_diff_hf) / std::max(std::sqrt(l2_hf), 1e-9);
    std::printf("layer total vs HF float64: rel_l2 %.3e, max_abs %.3e\n", rel_hf, worst_hf);
    // Measured 3.7e-2 here: the shared expert term adds signal, so the total is closer to
    // HF than the routed chain alone. Gate 8e-2 keeps headroom for the E4M3 class.
    check(rel_hf < 8.0e-2, "the layer total stays within 8e-2 rel_l2 of the HF f64 layer (" +
                               std::to_string(rel_hf) + ")");

    hipFree(d_swiglu_row_index);
    hipFree(d_logits);
    hipFree(d_sel_expert);
    hipFree(d_sel_weight);
    hipFree(d_tile_counts);
    hipFree(d_row_begin);
    hipFree(d_tile_base);
    hipFree(d_group_begin);
    hipFree(d_group_expert);
    hipFree(d_group_rows);
    hipFree(d_group_count);
    hipFree(d_row_token);
    hipFree(d_slot_row_of);
    hipFree(d_plane);
    hipFree(d_hidden13);
    hipFree(d_hidden2);
    hipFree(d_out13);
    hipFree(d_out2);
    hipFree(d_shared);
    hipFree(d_gate_bf16);
    hipFree(d_gate);
    hipFree(d_combined);
    hipFree(d_routed);
    gpu_arena.shutdown();
    if (hipStreamDestroy(stream) != hipSuccess)
        check(false, "stream destroy");

    std::printf("test_moe_layer_e2e: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
