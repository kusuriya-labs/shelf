#pragma once

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

namespace shelf {
namespace qwen35 {

namespace runtime {

using ::shelf::execution::Bf16GemmSelectorInput;
using ::shelf::execution::ElementwiseImplementation;
using ::shelf::execution::ElementwiseProfile;
using ::shelf::execution::ElementwiseSelectorInput;
using ::shelf::execution::EmbeddingImplementation;
using ::shelf::execution::EmbeddingRule;
using ::shelf::execution::EmbeddingSelectorInput;
using ::shelf::execution::EmbeddingStorage;
using ::shelf::execution::Fp8FinegrainedGemmSelectorInput;
using ::shelf::execution::GdnConvImplementation;
using ::shelf::execution::GdnConvSelectorInput;
using ::shelf::execution::GdnRecurrenceImplementation;
using ::shelf::execution::GdnRecurrenceSelectorInput;
using ::shelf::execution::KvAppendImplementation;
using ::shelf::execution::KvAppendSelectorInput;
using ::shelf::execution::L2NormalizeImplementation;
using ::shelf::execution::L2NormalizeSelectorInput;
using ::shelf::execution::LinearComputeFamily;
using ::shelf::execution::linear_shape_supported;
using ::shelf::execution::PagedAttentionImplementation;
using ::shelf::execution::PagedAttentionSelectorInput;
using ::shelf::execution::RmsNormImplementation;
using ::shelf::execution::RmsNormSelectorInput;
using ::shelf::execution::RmsNormSelectorWeightLayout;
using ::shelf::execution::RopeImplementation;
using ::shelf::execution::RopeSelectorInput;
using ::shelf::execution::SamplingBatchMode;
using ::shelf::execution::SamplingChoice;
using ::shelf::execution::SamplingImplementation;
using ::shelf::execution::SamplingRule;
using ::shelf::execution::SamplingSelectorInput;
using ::shelf::execution::kOptimizedMaxRows;
using ::shelf::execution::select_bf16_gemm_config;
using ::shelf::execution::select_elementwise_implementation;
using ::shelf::execution::select_embedding_implementation;
using ::shelf::execution::select_fp8_finegrained_gemm_config;
using ::shelf::execution::select_gdn_conv_implementation;
using ::shelf::execution::select_gdn_recurrence_implementation;
using ::shelf::execution::select_kv_append_implementation;
using ::shelf::execution::select_l2_normalize_implementation;
using ::shelf::execution::select_paged_attention_implementation;
using ::shelf::execution::select_rmsnorm_implementation;
using ::shelf::execution::select_rope_implementation;
using ::shelf::execution::select_sampling_implementation;

}

}
}
