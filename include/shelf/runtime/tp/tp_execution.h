#pragma once

#include <shelf/core/status.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/graph/value_type.h>
#include <shelf/runtime/program/program.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace shelf::runtime {

struct TpArgmaxInvocation;

enum class TpBarrierKind : uint8_t {
    None = 0,
    SumHidden = 1,
};

enum class TpCombineBackend : uint8_t {
    Event = 0,
    StreamSignal = 1,
};

enum class TpCombineFusion : uint8_t {
    None = 0,
    ResidualAdd = 1,
    HyperApply = 2,
};

struct TpProgramSegment {
    uint32_t dispatch_begin = 0;
    uint32_t dispatch_end = 0;
    uint32_t producer_end = 0;
    TpBarrierKind barrier_after = TpBarrierKind::None;
    // EP reduce runs after the TP reduce of the same combine value.
    bool ep_scope = false;
    ValueId combine_value{};
    uint32_t combine_local_cols = 0;
    TpCombineFusion fusion = TpCombineFusion::None;
    ValueId fusion_residual{};
    ValueId fusion_injection{};
    ValueId fusion_output{};
    ValueId argmax_output{};
    uint32_t argmax_dispatch = UINT32_MAX;
};

struct TpExecutionSchedule {
    std::vector<TpProgramSegment> segments;

    bool active() const noexcept { return !segments.empty(); }
};

Result<TpExecutionSchedule> build_tp_execution_schedule(
    const PrimitiveGraph& graph,
    const Program& program,
    std::uint32_t tp_size = 0u,
    std::uint32_t ep_size = 1u,
    bool ep_enabled = false);

struct TpSumTarget {
    void* ptr = nullptr;
    uint64_t rows = 0;
    uint32_t row_stride = 0;
    uint32_t feature_count = 0;
    ValueDType dtype = ValueDType::BF16;
    TpCombineFusion fusion = TpCombineFusion::None;
    const void* residual = nullptr;
    const void* injection = nullptr;
    void* fusion_output = nullptr;
    uint32_t residual_row_stride = 0;
    uint32_t injection_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t hidden = 0;
    uint32_t hc = 0;
};

struct TpSumInvocation {
    std::span<TpSumTarget> targets;
    std::span<hipStream_t> streams;
    std::span<const hipEvent_t> ready;
    bool peer_write_complete = false;
};

class TpTransport {
public:
    virtual ~TpTransport() = default;
    TpTransport(const TpTransport&) = delete;
    TpTransport& operator=(const TpTransport&) = delete;

    virtual const char* name() const noexcept = 0;

    virtual Status verify() { return Status::make_ok(); }
    virtual Status shutdown() { return Status::make_ok(); }

    virtual Status sum_hidden(const TpSumInvocation& invocation) = 0;

    virtual std::size_t inbox_capacity() const noexcept { return 0; }

    virtual void advance_barrier(int rank) { (void)rank; }

    virtual bool stream_signal_ready() const noexcept { return false; }

    virtual Status sum_hidden_stream_signal(int rank, const TpSumTarget& target,
                                            hipStream_t stream, bool peer_written) {
        (void)rank;
        (void)target;
        (void)stream;
        (void)peer_written;
        return Status::unsupported("stream signal barrier is not supported", __FILE__,
                                   __LINE__);
    }

    virtual Status reserve(std::size_t bytes) {
        (void)bytes;
        return Status::make_ok();
    }

    virtual bool supports_combine_fusion() const noexcept { return false; }

    virtual Status peer_write_target(int rank, std::size_t bytes, void*& peer_out) {
        (void)rank;
        (void)bytes;
        peer_out = nullptr;
        return Status::make_ok();
    }

    virtual bool supports_argmax_reduce() const noexcept { return false; }

    virtual Status begin_argmax(int rank, const TpArgmaxInvocation& invocation) {
        (void)rank;
        (void)invocation;
        return Status::unsupported("argmax reduce is not supported", __FILE__, __LINE__);
    }

    virtual Status finish_argmax(int rank, const TpArgmaxInvocation& invocation) {
        (void)rank;
        (void)invocation;
        return Status::unsupported("argmax reduce is not supported", __FILE__, __LINE__);
    }

protected:
    TpTransport() = default;
};

class TpBarrierHook {
public:
    virtual ~TpBarrierHook() = default;

    virtual Status arrive(int rank, const TpSumTarget& target, hipStream_t stream,
                          bool peer_written) = 0;

    virtual bool supports_combine_fusion() const noexcept { return false; }

    virtual Status reserve(std::size_t bytes) {
        (void)bytes;
        return Status::make_ok();
    }

    virtual Status peer_write_target(int rank, std::size_t bytes, void*& peer_out) {
        (void)rank;
        (void)bytes;
        peer_out = nullptr;
        return Status::make_ok();
    }

    virtual bool supports_argmax_reduce() const noexcept { return false; }

    virtual Status argmax_reduce(int rank, const TpArgmaxInvocation& invocation) {
        (void)rank;
        (void)invocation;
        return Status::unsupported("argmax reduce is not supported", __FILE__, __LINE__);
    }
};

}  // namespace shelf::runtime
