#include <shelf/models/qwen35moe/weights/moe_expert_load.h>

#include <shelf/core/memory/types.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace shelf::qwen35moe {
namespace {

using shelf::gpu::GpuArena;
using shelf::gpu::Tensor;
using shelf::quantization::bundle::QuantizedModelReader;
using shelf::quantization::bundle::QuantizedTensorView;
using shelf::quantization::manifest::QuantizedEncoding;
using shelf::quantization::mxfp4::Mxfp4CanonicalView;
using shelf::quantization::mxfp4::Mxfp4NativeHost;
using shelf::weights::MatrixEncoding;
using shelf::weights::MatrixWeight;

constexpr uint32_t kScaleBlockK = shelf::quantization::kMxfp4ScaleBlockK;
constexpr uint32_t kNativeOutputTile = shelf::quantization::mxfp4::mxfp4_native_output_tile;

bool fuses_into_w13(MoeExpertProjection projection) noexcept {
    return projection == MoeExpertProjection::Gate || projection == MoeExpertProjection::Up;
}

Tensor alloc_u8(std::size_t bytes, const std::vector<std::size_t>& shape,
                const std::vector<std::size_t>& strides, GpuArena& arena) {
    auto view = arena.allocate_aligned(bytes, 16);
    if (!view.ok())
        return {};
    auto tensor = Tensor::create<uint8_t>(view.release(), shape, strides);
    if (!tensor.ok())
        return {};
    return tensor.release();
}

Status upload(Tensor& dst, const void* src, std::size_t bytes, hipStream_t stream,
              const std::string& what) {
    const hipError_t err = hipMemcpyAsync(dst.data<void>(), src, bytes, hipMemcpyHostToDevice,
                                          stream);
    if (err != hipSuccess)
        return Status::hip_error(("moe expert " + what + " upload").c_str(),
                                 hipGetErrorString(err), __FILE__, __LINE__);
    return Status::make_ok();
}

// One plane's canonical payload: [experts][rows][k_padded / 2] codes and
// [experts][rows][k_padded / 32] E8M0 scales, experts and rows in plane order.
struct PlaneDraft {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t k_padded = 0;
    uint64_t scale_k = 0;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
};

Status gather_plane(const QuantizedModelReader& reader, const MoeExpertGroupPlan& plan,
                    bool fused_gate_up, PlaneDraft& out) {
    out.rows = fused_gate_up ? plan.w13_rows : plan.w2_rows;
    out.cols = fused_gate_up ? plan.w13_cols : plan.w2_cols;
    out.k_padded = fused_gate_up ? plan.w13_k_padded : plan.w2_k_padded;
    out.scale_k = out.k_padded / kScaleBlockK;
    const uint64_t packed_k = out.k_padded / 2u;
    const uint64_t expert_codes = static_cast<uint64_t>(out.rows) * packed_k;
    const uint64_t expert_scales = static_cast<uint64_t>(out.rows) * out.scale_k;
    out.codes.assign(static_cast<std::size_t>(expert_codes * plan.experts), 0u);
    out.scales.assign(static_cast<std::size_t>(expert_scales * plan.experts), 0u);

    for (const auto& src : plan.sources) {
        if (fuses_into_w13(src.projection) != fused_gate_up)
            continue;
        auto resolved = reader.resolve(src.name);
        if (!resolved.ok())
            return resolved.status();
        const QuantizedTensorView& v = resolved.value();
        if (v.encoding != QuantizedEncoding::Mxfp4) {
            return Status::invalid_argument(
                ("moe expert tensor is not mxfp4: " + src.name).c_str(), __FILE__, __LINE__);
        }
        if (static_cast<uint64_t>(src.row_offset) + src.rows > out.rows) {
            return Status::invalid_argument(
                ("moe expert source rows out of plane: " + src.name).c_str(), __FILE__, __LINE__);
        }
        if (v.logical_shape.size() != 2u || static_cast<uint64_t>(v.logical_shape[0]) != src.rows ||
            static_cast<uint64_t>(v.logical_shape[1]) != out.cols ||
            static_cast<uint64_t>(v.k_padded) != out.k_padded) {
            return Status::invalid_argument(
                ("moe expert logical shape mismatch: " + src.name).c_str(), __FILE__, __LINE__);
        }
        const uint64_t codes_want = static_cast<uint64_t>(src.rows) * packed_k;
        const uint64_t scales_want = static_cast<uint64_t>(src.rows) * out.scale_k;
        if (v.codes.data == nullptr || v.codes.size != codes_want) {
            return Status::invalid_argument(
                ("moe expert codes payload size mismatch: " + src.name).c_str(), __FILE__, __LINE__);
        }
        if (v.metadata1.data == nullptr || v.metadata1.size != scales_want) {
            return Status::invalid_argument(
                ("moe expert scale payload size mismatch: " + src.name).c_str(), __FILE__, __LINE__);
        }
        const uint64_t row_base = static_cast<uint64_t>(src.expert) * out.rows + src.row_offset;
        std::memcpy(out.codes.data() + row_base * packed_k, v.codes.data,
                    static_cast<std::size_t>(codes_want));
        std::memcpy(out.scales.data() + row_base * out.scale_k, v.metadata1.data,
                    static_cast<std::size_t>(scales_want));
    }
    return Status::make_ok();
}

Result<MatrixWeight> materialize_plane(const MoeExpertGroupPlan& plan, PlaneDraft& draft,
                                       GpuArena& arena, hipStream_t stream,
                                       const char* plane_name) {
    MatrixWeight w;
    for (const uint8_t byte : draft.scales) {
        if (byte == shelf::quantization::mxfp4::kE8m0NanByte) {
            return Status::invalid_argument(
                (std::string("mxfp4 scale stream carries an E8M0 NaN byte: ") + plane_name)
                    .c_str(),
                __FILE__, __LINE__);
        }
    }

    Mxfp4CanonicalView cv;
    cv.batch = plan.experts;
    cv.n = draft.rows;
    cv.padded_k = draft.k_padded;
    cv.codes = draft.codes.data();
    cv.codes_bytes = draft.codes.size();
    cv.scales = draft.scales.data();
    cv.scales_bytes = draft.scales.size();
    if (!cv.validate()) {
        return Status::invalid_argument(
            (std::string("mxfp4 canonical payload invalid: ") + plane_name).c_str(),
            __FILE__, __LINE__);
    }

    Mxfp4NativeHost native;
    if (!shelf::quantization::mxfp4::preshuffle_mxfp4_native(cv, native)) {
        return Status::invalid_argument(
            (std::string("mxfp4 native preshuffle failed: ") + plane_name).c_str(), __FILE__,
            __LINE__);
    }

    const uint64_t tiles = draft.rows / kNativeOutputTile;
    const uint64_t codes_batch =
        shelf::quantization::mxfp4::mxfp4_native_codes_batch_stride_bytes(draft.rows,
                                                                          draft.k_padded);
    const uint64_t scales_batch =
        shelf::quantization::mxfp4::mxfp4_native_scale_batch_stride_bytes(draft.rows,
                                                                          draft.k_padded);

    w.codes = alloc_u8(native.codes.size(),
                       {plan.experts, static_cast<std::size_t>(tiles),
                        static_cast<std::size_t>(native.codes_row_stride_bytes)},
                       {static_cast<std::size_t>(codes_batch),
                        static_cast<std::size_t>(native.codes_row_stride_bytes), 1u},
                       arena);
    if (w.codes.ndim() == 0) {
        return Status::insufficient_memory(
            (std::string("moe expert codes allocation failed: ") + plane_name).c_str(),
            __FILE__, __LINE__);
    }
    w.scales = alloc_u8(native.scales.size(),
                        {plan.experts, static_cast<std::size_t>(draft.rows),
                         static_cast<std::size_t>(native.scale_row_stride_bytes)},
                        {static_cast<std::size_t>(scales_batch),
                         static_cast<std::size_t>(native.scale_row_stride_bytes), 1u},
                        arena);
    if (w.scales.ndim() == 0) {
        return Status::insufficient_memory(
            (std::string("moe expert scales allocation failed: ") + plane_name).c_str(),
            __FILE__, __LINE__);
    }
    w.row_ref = alloc_u8(native.row_ref.size(),
                         {plan.experts, static_cast<std::size_t>(draft.rows)},
                         {static_cast<std::size_t>(draft.rows), 1u}, arena);
    if (w.row_ref.ndim() == 0) {
        return Status::insufficient_memory(
            (std::string("moe expert row reference allocation failed: ") + plane_name).c_str(),
            __FILE__, __LINE__);
    }

    Status st = upload(w.codes, native.codes.data(), native.codes.size(), stream,
                       std::string(plane_name) + " codes");
    if (!st.ok())
        return st;
    st = upload(w.scales, native.scales.data(), native.scales.size(), stream,
                std::string(plane_name) + " scales");
    if (!st.ok())
        return st;
    st = upload(w.row_ref, native.row_ref.data(), native.row_ref.size(), stream,
                std::string(plane_name) + " row_ref");
    if (!st.ok())
        return st;

    w.experts = plan.experts;
    w.rows = draft.rows;
    w.cols = draft.cols;
    w.k_padded = draft.k_padded;
    w.codes_row_stride_bytes = static_cast<uint32_t>(native.codes_row_stride_bytes);
    w.scale_row_stride_bytes = static_cast<uint32_t>(native.scale_row_stride_bytes);
    w.storage_scale_stride_bytes = static_cast<uint32_t>(native.scale_row_stride_bytes);
    w.codes_batch_stride_bytes = codes_batch;
    w.scales_batch_stride_bytes = scales_batch;
    w.weight_scale_group = kScaleBlockK;
    w.encoding = MatrixEncoding::Mxfp4;
    w.preshuffled = true;
    w.row_ref_exact = native.row_ref_exact;
    shelf::weights::apply_default_compute_spec(w);
    return w;
}

}  // namespace

Result<MoeExpertGroupWeights> load_moe_expert_group(const QuantizedModelReader& reader,
                                                    const MoeExpertGroupPlan& plan,
                                                    GpuArena& arena, hipStream_t stream) {
    if (plan.experts == 0u)
        return Status::invalid_argument("load_moe_expert_group: plan has no experts",
                                        __FILE__, __LINE__);
    if (plan.sources.size() != static_cast<std::size_t>(plan.experts) * 3u)
        return Status::invalid_argument("load_moe_expert_group: plan source count mismatch",
                                        __FILE__, __LINE__);

    PlaneDraft w13_draft;
    Status st = gather_plane(reader, plan, true, w13_draft);
    if (!st.ok())
        return st;
    PlaneDraft w2_draft;
    st = gather_plane(reader, plan, false, w2_draft);
    if (!st.ok())
        return st;

    MoeExpertGroupWeights out;
    {
        auto w13 = materialize_plane(plan, w13_draft, arena, stream, "w13");
        if (!w13.ok())
            return w13.status();
        out.w13 = w13.release();
    }
    {
        auto w2 = materialize_plane(plan, w2_draft, arena, stream, "w2");
        if (!w2.ok())
            return w2.status();
        out.w2 = w2.release();
    }
    const hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess)
        return Status::hip_error("moe expert group upload", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    return out;
}

}  // namespace shelf::qwen35moe
