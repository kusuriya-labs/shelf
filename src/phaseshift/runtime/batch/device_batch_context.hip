#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdint>
namespace ps::runtime {
namespace {
template <typename T>
Result<gpu::Tensor> allocate_tensor(gpu::GpuArena& arena, std::size_t count) {
    std::size_t bytes = count * sizeof(T);
    auto view_res = arena.allocate_aligned(bytes, 128);
    if (!view_res.ok()) return view_res.status();
    auto view = view_res.release();
    std::vector<std::size_t> shape{count};
    std::vector<std::size_t> strides{1};
    auto t_res = gpu::Tensor::create<T>(view, shape, strides);
    if (!t_res.ok()) return t_res.status();
    return t_res.release();
}
}
Result<DeviceBatchContextStorage> create_device_batch_context(gpu::GpuArena& arena, uint32_t max_requests, uint32_t max_tokens) {
    if (max_requests == 0 || max_tokens == 0) return Status::invalid_argument("max_requests and max_tokens must be > 0", __FILE__, __LINE__);
    if (arena.is_shutdown()) return Status::invalid_state("arena shutdown", __FILE__, __LINE__);
    if (arena.device() < 0) return Status::invalid_state("arena device invalid", __FILE__, __LINE__);
    DeviceBatchContextStorage s;
    s.max_requests = max_requests; s.max_tokens = max_tokens;
    auto t_ids = allocate_tensor<int32_t>(arena, max_tokens);
    if (!t_ids.ok()) return t_ids.status();
    s.token_ids = t_ids.release();
    auto t_req = allocate_tensor<DeviceRequestDescriptor>(arena, max_requests);
    if (!t_req.ok()) return t_req.status();
    s.requests = t_req.release();
    auto t_slots = allocate_tensor<uint32_t>(arena, max_tokens);
    if (!t_slots.ok()) return t_slots.status();
    s.row_sequence_slots = t_slots.release();
    auto t_pos = allocate_tensor<uint32_t>(arena, max_tokens);
    if (!t_pos.ok()) return t_pos.status();
    s.row_positions = t_pos.release();
    const uint32_t out_cap = max_requests > max_tokens ? max_requests : max_tokens;
    auto t_out_rows = allocate_tensor<uint32_t>(arena, out_cap);
    if (!t_out_rows.ok()) return t_out_rows.status();
    s.output_rows = t_out_rows.release();
    auto t_flags = allocate_tensor<DeviceSamplingParams>(arena, out_cap);
    if (!t_flags.ok()) return t_flags.status();
    s.output_sampling_params = t_flags.release();
    auto t_dev = allocate_tensor<DeviceBatchContext>(arena, 1);
    if (!t_dev.ok()) return t_dev.status();
    s.device_struct = t_dev.release();
    DeviceBatchContext host_ctx{};
    host_ctx.actual_rows = 0; host_ctx.num_requests = 0; host_ctx.num_outputs = 0;
    host_ctx.num_decode_requests = 0; host_ctx.num_verify_requests = 0; host_ctx.num_prefill_requests = 0;
    host_ctx.token_ids = s.token_ids.data<int32_t>();
    host_ctx.requests = s.requests.data<DeviceRequestDescriptor>();
    host_ctx.row_sequence_slots = s.row_sequence_slots.data<uint32_t>();
    host_ctx.row_positions = s.row_positions.data<uint32_t>();
    host_ctx.output_rows = s.output_rows.data<uint32_t>();
    host_ctx.output_sampling_params = s.output_sampling_params.data<DeviceSamplingParams>();
    host_ctx.device_ptr = s.device_struct.data<DeviceBatchContext>();
    auto dev_guard = gpu::ScopedDevice::create(arena.device());
    if (!dev_guard.ok()) return dev_guard.status();
    hipError_t err = hipMemcpy(s.device_struct.data<DeviceBatchContext>(), &host_ctx, sizeof(DeviceBatchContext), hipMemcpyHostToDevice);
    if (err != hipSuccess) return Status::hip_error("hipMemcpy DeviceBatchContext", hipGetErrorString(err), __FILE__, __LINE__);
    return s;
}
Status destroy_device_batch_context(DeviceBatchContextStorage& storage) noexcept {
    storage = DeviceBatchContextStorage{};
    return Status::make_ok();
}
__global__ void prepare_batch_descriptor_kernel(DeviceBatchContext* ctx) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= ctx->num_requests) return;
    DeviceRequestDescriptor req = ctx->requests[r];
    for (uint32_t t = 0; t < req.row_count; ++t) {
        uint32_t row = req.row_begin + t;
        if (row >= ctx->actual_rows) continue;
        ctx->row_sequence_slots[row] = req.request_handle.slot;
        ctx->row_positions[row] = req.prefix_length + t;
    }
    if (req.compute_logits && req.output_index != 0xFFFFFFFFu) {
        uint32_t oc = req.output_count ? req.output_count : 1u;
        if (oc > req.row_count) oc = req.row_count;
        const uint32_t first = req.row_begin + req.row_count - oc;
        for (uint32_t j = 0; j < oc; ++j) {
            ctx->output_rows[req.output_index + j] = first + j;
            DeviceSamplingParams sp = req.sampling;
            sp.sample_index = req.sampling.sample_index + j;
            ctx->output_sampling_params[req.output_index + j] = sp;
        }
    }
}
Status launch_prepare_batch_descriptor(DeviceBatchContext* ctx, uint32_t max_requests, hipStream_t stream) {
    if (ctx == nullptr) return Status::invalid_argument("ctx null", __FILE__, __LINE__);
    if (max_requests == 0) return Status::invalid_argument("max_requests 0", __FILE__, __LINE__);
    constexpr uint32_t kBlock = 256;
    uint32_t grid = (max_requests + kBlock - 1) / kBlock;
    hipLaunchKernelGGL(prepare_batch_descriptor_kernel, dim3(grid), dim3(kBlock), 0, stream, ctx);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return Status::hip_error("prepare_batch_descriptor", hipGetErrorString(err), __FILE__, __LINE__);
    return Status::make_ok();
}
}
