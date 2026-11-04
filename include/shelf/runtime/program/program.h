#pragma once
#include <shelf/core/status.h>
#include <shelf/runtime/program/kernel_id.h>
#include <shelf/runtime/program/workspace_layout.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/graph/value_type.h>
#include <shelf/runtime/execution/execution_types.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace shelf::runtime {

constexpr uint32_t kMaxIoSlots = 6;
constexpr uint32_t kNoWeight = 0xFFFFFFFFu;
constexpr uint32_t kNoParameter = 0xFFFFFFFFu;
constexpr uint32_t kNoWorkspace = 0xFFFFFFFFu;

struct WeightSlot {
    uint32_t rows = 0;
    uint32_t cols = 0;
    const void* device_ptr = nullptr;
    uint64_t bytes = 0;
    uint32_t k_padded = 0;
    uint8_t encoding = 0;
    uint8_t compute_spec = 0;
    const void* codes = nullptr;
    const void* scales = nullptr;
    const void* row_ref = nullptr;
    bool row_ref_exact = false;
    uint32_t storage_scale_stride_bytes = 0;
    const void* compute_codes = nullptr;
    const void* compute_scales_bf16 = nullptr;
    uint32_t codes_row_stride_bytes = 0;
    uint32_t scale_row_stride_bytes = 0;
    uint32_t weight_scale_group = 0;
    bool preshuffled = false;
    // Rank-3 expert planes. experts == 1 is the 2D path; when experts > 1, rows / cols /
    // k_padded and the row strides describe ONE expert and the batch strides separate the
    // experts inside one arena allocation. row_ref is then a [experts][rows] byte plane
    // with row stride 1, so its batch stride is experts * rows.
    uint32_t experts = 1;
    uint64_t codes_batch_stride_bytes = 0;
    uint64_t scales_batch_stride_bytes = 0;
    uint32_t output_col_offset = 0;
    uint32_t output_col_total = 0;
};

struct WeightTableView {
    const WeightSlot* slots = nullptr;
    size_t count = 0;
    WeightSlot at(size_t index) const {
        return slots != nullptr ? slots[index] : WeightSlot{};
    }
};

struct StaticParameterSlot {
    const void* device_ptr = nullptr;
    uint32_t elements = 0;
    ValueDType dtype = ValueDType::BF16;
};

struct StaticParameterTableView {
    const StaticParameterSlot* slots = nullptr;
    size_t count = 0;
};

struct DispatchBinding {
    KernelId kernel_id = KernelId::EMBEDDING_LOOKUP;
    uint32_t workspace_slot = kNoWorkspace;
    uint32_t descriptor_slot = 0;
    uint32_t weight_index = kNoWeight;
    float scalar_a = 1.0f;
    float scalar_b = 0.0f;
    uint32_t rotary_dim = 0;
    uint32_t input_slots[kMaxIoSlots] = {};
    uint32_t input_count = 0;
    uint32_t output_slots[kMaxIoSlots] = {};
    uint32_t output_count = 0;
    uint32_t parameter_index0 = kNoParameter;
    uint32_t parameter_index1 = kNoParameter;
    uint32_t group_size = 0;
    uint32_t flags = 0;
    uint32_t compute_spec = 0;
    uint32_t activation_kp = 0;
    uint32_t activation_max_rows = 0;
    uint32_t activation_code_stride = 0;
    uint32_t activation_scale_stride = 0;
};

enum class ValueStorage : uint8_t {
    EXTERNAL_INPUT = 0,
    EXTERNAL_OUTPUT = 1,
    WORKSPACE = 2,
    WORKSPACE_VIEW = 3,
    WEIGHT = 4,
    STATE = 5,
};

struct ValueBinding {
    ValueId logical_value;
    ValueStorage storage = ValueStorage::WORKSPACE;
    ValueDType dtype = ValueDType::BF16;
    ValueRowDomain row_domain = ValueRowDomain::TOKEN_ROWS;
    uint32_t slot = 0;
    uint32_t feature_count = 0;
    uint32_t row_stride = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct StateBinding {
    StateId state;
    PrimitiveStateKind kind = PrimitiveStateKind::KV_CACHE;
    uint32_t state_index = 0;
    uint32_t slot = 0;
};

struct Program {
    RowBucket row_bucket = RowBucket::R16;
    ExecutionClass execution_class = ExecutionClass::DECODE;
    WorkspaceLayout workspace;
    std::vector<DispatchBinding> dispatches;
    std::vector<uint32_t> dispatch_source_node;
    std::vector<ValueBinding> values;
    std::vector<StateBinding> states;
    uint32_t weight_slot_count = 0;
    uint32_t parameter_slot_count = 0;
    uint32_t external_input_count = 0;
    uint32_t external_output_count = 0;

    Status validate() const;

    const ValueBinding* find_value(ValueId v) const noexcept {
        for (const auto& b : values) {
            if (b.logical_value.id == v.id) return &b;
        }
        return nullptr;
    }
};

struct ProgramSet {
    static constexpr size_t kRowBucketCount = 10;
    std::array<Program, kRowBucketCount> programs;

    Program* resolve(RowBucket bucket);
    const Program* resolve(RowBucket bucket) const;
};

struct ProgramBuildOptions {
    bool validate_workspace = true;
    bool no_workspace_reuse = false;
    uint32_t max_token_rows = 0;
    uint32_t max_output_rows = 0;
};

Result<Program> build_program(
    const PrimitiveGraph& graph,
    const WeightTableView& weights,
    RowBucket bucket,
    ExecutionClass execution_class,
    const ProgramBuildOptions& options);

Result<ProgramSet> build_program_set(
    const PrimitiveGraph& graph,
    const WeightTableView& weights,
    const StaticParameterTableView& parameters,
    ExecutionClass execution_class,
    const ProgramBuildOptions& options);

}
