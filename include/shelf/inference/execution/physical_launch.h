#pragma once
#include <shelf/core/status.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/kernels/optimized/gdn/conv1d.h>
#include <shelf/kernels/optimized/kv_append.h>
#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/kernels/optimized/l2_normalize.h>
#include <shelf/kernels/optimized/rmsnorm.h>
#include <shelf/kernels/kv_types.h>
#include <shelf/runtime/program/program.h>
#include <shelf/inference/execution/selectors/elementwise_selector.h>
#include <shelf/inference/execution/selectors/embedding_selector.h>
#include <shelf/inference/execution/selectors/gdn_conv_selector.h>
#include <shelf/inference/execution/selectors/gdn_recurrence_selector.h>
#include <shelf/inference/execution/selectors/kv_append_selector.h>
#include <shelf/inference/execution/selectors/l2_normalize_selector.h>
#include <shelf/inference/execution/selectors/linear_selector.h>
#include <shelf/inference/execution/selectors/paged_attention_selector.h>
#include <shelf/inference/execution/selectors/rmsnorm_selector.h>
#include <shelf/inference/execution/selectors/rope_selector.h>
#include <shelf/inference/execution/selectors/sampling_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>

#include <cstdint>

namespace shelf::execution {

struct HostExecutionContext;

::shelf::kernel::RmsNormDataType rmsnorm_kd(::shelf::runtime::ValueDType d);

enum class PhysicalResolveStatus : uint8_t {
    Applicable = 0,
    NotApplicable = 1,
    Skipped = 2,
};


enum class PhysicalRmsNormVariant : uint8_t {
    Bf16PfOnePlus = 0,
    Unsupported = 1,
    F32PgDirect = 2,
    Bf16F32PgOnePlus = 3,
    Bf16PgOnePlus = 4,
};

struct PhysicalRmsNormLaunch {
    uint64_t input = 0;
    uint64_t weight = 0;
    uint64_t output = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    float eps = 0.0f;
    float weight_offset = 0.0f;
    ::shelf::kernel::RmsNormDataType input_dtype =
        ::shelf::kernel::RmsNormDataType::BF16;
    ::shelf::kernel::RmsNormDataType weight_dtype =
        ::shelf::kernel::RmsNormDataType::BF16;
    ::shelf::kernel::RmsNormDataType output_dtype =
        ::shelf::kernel::RmsNormDataType::BF16;
    ::shelf::kernel::RmsNormWeightLayout weight_layout =
        ::shelf::kernel::RmsNormWeightLayout::NONE;
    ::shelf::kernel::RmsNormWeightMode weight_mode =
        ::shelf::kernel::RmsNormWeightMode::ONE_PLUS;
    PhysicalRmsNormVariant variant = PhysicalRmsNormVariant::Unsupported;
    uint32_t grid = 0;
    uint32_t workgroup = 0;
};

struct PhysicalActivationQuantizeLaunch {
    uint64_t input = 0;
    uint64_t codes = 0;
    uint64_t scales = 0;
    // Group-plan row indirection: output row m reads input row row_index[m], and a negative
    // entry writes a zero row. Zero when the rows are the input's own rows.
    uint64_t row_index = 0;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
    uint32_t input_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
    uint32_t grid = 0;
    uint32_t workgroup = 0;
};


struct PhysicalElementwiseLaunch {
    ElementwiseProfile profile = ElementwiseProfile::Count;
    uint64_t in0 = 0;
    uint64_t in1 = 0;
    uint64_t out0 = 0;
    uint64_t out1 = 0;
    uint32_t in0_stride = 0;
    uint32_t in1_stride = 0;
    uint32_t out0_stride = 0;
    uint32_t out1_stride = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t features_b = 0;
    uint32_t aux = 0;
    float scalar_a = 0.0f;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t workgroup = 0;
};

enum class PhysicalBf16Variant : uint8_t {
    ExactRows = 0,
    WmmaKPartition = 1,
    WmmaWide = 2,
    Wmma = 3,
    ExactRowsDot2 = 4,
    Unsupported = 255,
};

struct PhysicalBf16Launch {
    uint64_t weight = 0;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    ::shelf::kernel::Bf16GemmOutputDType output_dtype =
        ::shelf::kernel::Bf16GemmOutputDType::BF16;
    ::shelf::kernel::Bf16GemmConfig config{};
    PhysicalBf16Variant variant = PhysicalBf16Variant::Unsupported;
    uint32_t grid_x = 0;
    uint32_t grid_y = 1;
    uint32_t workgroup = 0;
    bool output_rows_domain = false;
};


struct PhysicalL2NormalizeLaunch {
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    float eps = 0.0f;
    uint32_t grid_x = 0;
    uint32_t grid_y = 1;
    uint32_t workgroup = 0;
};

enum class PhysicalGdnRecurrenceVariant : uint8_t {
    WmmaVerify = 0,
    Wmma = 4,
    F32 = 5,
    Unsupported = 255,
};

struct PhysicalGdnConv1dLaunch {
    ::shelf::kernel::GdnConv1dArgs args{};
    uint32_t row_tile = 0;
    uint32_t tiles = 1;
    uint32_t grid_x = 0;
    uint32_t grid_y = 1;
    uint32_t grid_z = 1;
    uint32_t workgroup = 0;
};

struct PhysicalGdnRecurrenceLaunch {
    ::shelf::kernel::GdnRecurrenceArgs kernel{};
    bool lossy = false;
    bool split_regions = false;
    uint32_t rows = 0;
    uint32_t num_requests = 0;
    uint32_t max_request_rows = 0;
    PhysicalGdnRecurrenceVariant variant = PhysicalGdnRecurrenceVariant::Unsupported;
    uint32_t grid_x = 0;
    uint32_t grid_y = 1;
    uint32_t grid_z = 1;
    uint32_t workgroup = 0;
};

struct PhysicalEmbeddingLaunch {
    EmbeddingStorage storage = EmbeddingStorage::Bf16;
    uint64_t table = 0;
    uint64_t codes = 0;
    uint64_t scales = 0;
    uint32_t codes_row_stride_bytes = 0;
    uint32_t scale_row_stride_bytes = 0;
    uint32_t k_padded = 0;
    uint64_t token_ids = 0;
    uint64_t output = 0;
    uint32_t rows = 0;
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t output_row_stride = 0;
};

Result<PhysicalResolveStatus> resolve_embedding_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalEmbeddingLaunch& out);

Result<PhysicalResolveStatus> resolve_rmsnorm_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalRmsNormLaunch& out);

Result<PhysicalResolveStatus> resolve_activation_quantize_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalActivationQuantizeLaunch& out);

Result<PhysicalResolveStatus> resolve_elementwise_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalElementwiseLaunch& out);

Result<PhysicalResolveStatus> resolve_bf16_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalBf16Launch& out);

Result<PhysicalResolveStatus> resolve_l2_normalize_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalL2NormalizeLaunch& out);

Result<PhysicalResolveStatus> resolve_gdn_conv1d_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalGdnConv1dLaunch& out);

Result<PhysicalResolveStatus> resolve_gdn_recurrence_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalGdnRecurrenceLaunch& out);

enum class PhysicalRopeVariant : uint8_t {
    F32Bf16Pair = 0,
    F32Bf16Generic = 1,
    Bf16Rounded = 2,
    Unsupported = 255,
};

struct PhysicalRopeLaunch {
    PhysicalRopeVariant variant = PhysicalRopeVariant::Unsupported;
    uint64_t input = 0;
    uint64_t output = 0;
    uint64_t positions = 0;
    uint64_t inv_freq = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    float theta = 0.0f;
    float attention_factor = 1.0f;
    uint32_t grid = 0;
    uint32_t workgroup = 0;
};

Result<PhysicalResolveStatus> resolve_rope_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalRopeLaunch& out);

hipError_t launch_rope_physical(const PhysicalRopeLaunch& d, hipStream_t stream);

enum class PhysicalKvAppendVariant : uint8_t {
    Bf16 = 0,
    Fp8E4M3 = 1,
    Octave = 2,
    Unsupported = 255,
};

struct PhysicalKvAppendLaunch {
    PhysicalKvAppendVariant variant = PhysicalKvAppendVariant::Unsupported;
    ::shelf::kernel::KvAppendCommonArgs common{};
    uint64_t k_pool = 0;
    uint64_t v_pool = 0;
    uint64_t k_scale_pool = 0;
    uint64_t v_scale_pool = 0;
    uint64_t k_extra_pool = 0;
    uint64_t v_extra_pool = 0;
    uint32_t blocks_per_head = 0;
    ::shelf::kernel::OctaveKVCacheFormat octave_format{};
    uint32_t grid = 0;
    uint32_t workgroup = 0;
};

Result<PhysicalResolveStatus> resolve_kv_append_physical(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx,
    PhysicalKvAppendLaunch& out);

hipError_t launch_kv_append_physical(const PhysicalKvAppendLaunch& d,
                                     hipStream_t stream);

enum class PhysicalPagedAttentionVariant : uint8_t {
    Bf16FullHead = 0,
    Bf16PartialHead = 1,
    Bf16SplitQ4Full = 2,
    Bf16SplitOther = 3,
    Bf16ReduceQ4FullQh1 = 4,
    Bf16ReduceQ4FullQh2 = 5,
    Bf16ReduceQ4FullBase = 6,
    Bf16ReduceOther = 7,
    Fp8E4M3 = 8,
    Fp8SplitQ4Full = 9,
    Fp8SplitOther = 10,
    Bf16Prefill = 14,
    Fp8E4M3Prefill = 15,
    OctaveSplit = 17,
    OctaveReduce = 18,
    OctavePrefill = 19,
    Unsupported = 255,
};

struct PhysicalPagedAttentionKernel {
    PhysicalPagedAttentionVariant variant = PhysicalPagedAttentionVariant::Unsupported;
    ::shelf::kernel::PagedAttentionCommonArgs common{};
    uint64_t k_pool = 0;
    uint64_t v_pool = 0;
    uint64_t k_scale_pool = 0;
    uint64_t v_scale_pool = 0;
    uint64_t k_extra_pool = 0;
    uint64_t v_extra_pool = 0;
    uint64_t partials = 0;
    uint32_t splits = 0;
    uint32_t q_per_kv = 0;
    uint32_t qh_per_wg = 0;
    uint32_t blocks_per_head = 0;
    uint32_t grid = 0;
    uint32_t workgroup = 0;
    uint32_t row_begin = 0;
    uint32_t source_layer = 0;
    ::shelf::kernel::OctaveKVCacheFormat octave_format{};
};

inline constexpr uint32_t kPhysicalPagedAttentionMaxPrefillRegions = 64u;

inline constexpr uint32_t kPhysicalPagedAttentionMaxKernels =
    4u + 2u * kPhysicalPagedAttentionMaxPrefillRegions;

struct PhysicalPagedAttentionPlan {
    bool optimized = false;
    bool use_prefill = false;
    ::shelf::kernel::KVCacheDType kv_dtype = ::shelf::kernel::KVCacheDType::BF16;
    uint32_t splits = 1u;
    uint32_t physical_count = 0u;
    PhysicalPagedAttentionKernel kernels[kPhysicalPagedAttentionMaxKernels];
};

Result<PhysicalResolveStatus> resolve_paged_attention_physical(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx,
    PhysicalPagedAttentionPlan& out);

Result<PhysicalResolveStatus> resolve_paged_attention_physical_static_plan(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx,
    PhysicalPagedAttentionPlan& out);

Result<PhysicalResolveStatus> resolve_paged_attention_physical_static_prefill(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx,
    PhysicalPagedAttentionPlan& out);

hipError_t launch_paged_attention_physical(const PhysicalPagedAttentionPlan& plan,
                                           hipStream_t stream);

}  // namespace shelf::execution
