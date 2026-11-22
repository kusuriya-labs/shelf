#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/models/hy4_flash/routes.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <cstring>
#include <rocprim/block/block_scan.hpp>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
__global__ void router(Mat x, const uint16_t *weight, float *scores) {
    int expert = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32, row = blockIdx.y;
    float dot = 0;
    for (int c = lane; c < 3072; c += 32)
        dot += bf16_bits_to_f32(x.p[row * 3072 + c]) * bf16_bits_to_f32(weight[expert * 3072 + c]);
    for (int d = 16; d; d /= 2)
        dot += __shfl_down(dot, d, 32);
    if (!lane)
        scores[row * 512 + expert] = 1.f / (1.f + expf(-dot));
}
__global__ void choose(const float *scores, const float *bias, Hy4Route *out) {
    __shared__ float values[512], weight_sum;
    __shared__ int ids[512];
    int t = threadIdx.x, row = blockIdx.x;
    scores += row * 512;
    for (int i = t; i < 512; i += 256) {
        ids[i] = i;
        values[i] = scores[i] + bias[i];
    }
    __syncthreads();
    for (int width = 2; width <= 512; width *= 2)
        for (int step = width / 2; step; step /= 2) {
            int a = (t / step) * 2 * step + t % step, b = a + step;
            bool order = values[a] > values[b] || (values[a] == values[b] && ids[a] < ids[b]);
            if (order == bool(a & width)) {
                float va = values[a];
                int ia = ids[a];
                values[a] = values[b];
                ids[a] = ids[b];
                values[b] = va;
                ids[b] = ia;
            }
            __syncthreads();
        }
    if (!t) {
        float sum = 1e-20f;
        for (int j = 0; j < 16; ++j)
            sum += scores[ids[j]];
        weight_sum = sum;
    }
    __syncthreads();
    if (t < 16)
        out[row * 16 + t] = {row, ids[t], scores[ids[t]] / weight_sum * 3.998f};
}
__global__ void gather_route(Mat x, const Hy4Route *route, Mat out) {
    int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    out.p[r * 1536 + c] = x.p[route[r].row * 1536 + c];
}
__global__ void weighted(Mat x, const Hy4Route *route, const int *offsets, const int *inverse, Mat out) {
    int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    float sum = 0;
    for (int i = offsets[r]; i < offsets[r + 1]; ++i) {
        int j = inverse ? inverse[i] : i;
        sum = round_bf16(sum + round_bf16(route[j].weight * bf16_bits_to_f32(x.p[j * 1536 + c])));
    }
    out.p[r * 1536 + c] = f32_to_bf16_rne_device(sum);
}
__global__ void make_decode_plans(const Hy4Route *picks, Mat latent, const int *owners, uint8_t *packets,
                                  size_t stride, int first_group, bool scatter, uint64_t *totals) {
    int group = first_group + blockIdx.x;
    auto plan = reinterpret_cast<Hy4MoePlan *>(packets + blockIdx.x * stride);
    if (!threadIdx.x) {
        plan->partition(picks, latent.rows, owners, group);
        totals[group] += plan->count;
    }
    if (scatter && group != 0) {
        auto dst = reinterpret_cast<uint4 *>(plan + 1);
        auto src = reinterpret_cast<const uint4 *>(latent.p);
        for (int i = threadIdx.x; i < latent.rows * 1536 / 8; i += blockDim.x)
            dst[i] = src[i];
    }
}

__global__ void make_grouped_plans(const Hy4Route *picks, Mat latent, const int *owners, uint8_t *packets,
                                   size_t stride, uint64_t *totals) {
    using Scan = rocprim::block_scan<unsigned, 512>;
    __shared__ typename Scan::storage_type scan;
    __shared__ unsigned masks[3][512];
    __shared__ int starts[512], row_counts[Hy4GroupedPlan::max_rows];
    const int t = threadIdx.x, group = blockIdx.x, rows = latent.rows;
    auto plan = reinterpret_cast<Hy4GroupedPlan *>(packets + group * stride);
    for (auto &word : masks)
        word[t] = 0;
    if (t < rows)
        row_counts[t] = 0;
    __syncthreads();
    for (int i = t; i < rows * 16; i += 512) {
        const int expert = picks[i].expert, row = i / 16;
        if (owners[expert] == group) {
            atomicOr(&masks[row / 32][expert], 1u << (row % 32));
            atomicAdd(&row_counts[row], 1);
        }
    }
    __syncthreads();
    if (t == 0) {
        int total = 0;
        for (int row = 0; row < rows; ++row) {
            plan->offsets[row] = total;
            total += row_counts[row];
        }
        plan->count = plan->offsets[rows] = total;
        totals[group] += total;
    }
    int count = 0;
    for (auto &word : masks)
        count += __popc(word[t]);
    const unsigned size = count + (((count + 15) / 16) << 16);
    unsigned prefix;
    Scan().exclusive_scan(size, prefix, 0u, scan);
    starts[t] = prefix & 65535;
    int batch = prefix >> 16;
    for (int i = 0; i < count; i += 16)
        plan->batches[batch++] = {starts[t] + i, min(16, count - i)};
    if (t == 511)
        plan->num_batches = batch;
    __syncthreads();
    for (int i = t; i < rows * 16; i += 512) {
        const auto route = picks[i];
        const int e = route.expert, row = route.row;
        if (owners[e] != group)
            continue;
        plan->place(route, picks + row * 16, owners, masks, starts[e]);
    }
    if (group) {
        auto dst = reinterpret_cast<uint4 *>(plan + 1);
        auto src = reinterpret_cast<const uint4 *>(latent.p);
        for (int i = t; i < rows * 1536 / 8; i += 512)
            dst[i] = src[i];
    }
}

template<class Plan>
Mat decode_experts(Ops &op, int layer, Mat latent, const Plan *plan, hipStream_t stream) {
    const int capacity = Hy4MoePlan::capacity(latent.rows);
    const int intermediate = 1536 / op.gpu.tp_size;
    Mat output = op.mat(latent.rows, 1536);
    auto codes = static_cast<int8_t *>(op.arena.next(size_t(capacity) * 2 * 1536));
    auto scales = static_cast<float *>(op.arena.next(size_t(capacity) * 2 * 12 * sizeof(float)));
    auto partial = static_cast<float *>(op.arena.next(size_t(capacity) * 2 * intermediate * sizeof(float)));
    Mat values = op.mat(capacity, 1536);
    auto matrices = op.gpu.matrices[op.rank] + (layer - 1) * 1536;
    if (op.gpu.expert_begin[op.rank])
        checked(hipEventRecord(op.gpu.expert_begin[op.rank], stream));
    const int *inverse = nullptr, *num_batches = nullptr;
    const Hy4ExpertBatch *batches = nullptr;
    if constexpr (std::is_same_v<Plan, Hy4GroupedPlan>) {
        inverse = plan->inverse;
        batches = plan->batches;
        num_batches = &plan->num_batches;
    }
    checked(shelf_hy4_expert_decode(latent.p, matrices, plan->routes, &plan->count, capacity, intermediate,
                                    codes, scales, partial, values.p, stream, batches, num_batches));
    weighted<<<dim3(6, latent.rows), 256, 0, stream>>>(values, plan->routes, plan->offsets, inverse, output);
    checked(hipGetLastError());
    if (op.gpu.expert_end[op.rank])
        checked(hipEventRecord(op.gpu.expert_end[op.rank], stream));
    return output;
}

template<class Plan>
Mat decode_moe(Ops &op, int layer, const std::string &prefix, Mat x, Mat latent, const Hy4Route *picks) {
    auto &gpu = op.gpu;
    const int rank = op.rank;
    const int groups = gpu.tp_size == 1 ? gpu.ep_size : 1;
    const size_t stride = groups > 1 ? Plan::packet_bytes(x.rows) : sizeof(Plan);
    auto packets = static_cast<uint8_t *>(op.arena.next(groups * stride));
    if constexpr (std::is_same_v<Plan, Hy4GroupedPlan>)
        make_grouped_plans<<<groups, 512, 0, op.stream>>>(picks, latent, gpu.moe_owners[rank] + (layer - 1) * 512,
                                                         packets, stride, gpu.moe_counts[rank]);
    else
        make_decode_plans<<<groups, 128, 0, op.stream>>>(picks, latent, gpu.moe_owners[rank] + (layer - 1) * 512,
                                                   packets, stride, rank / gpu.tp_size, groups > 1,
                                                   gpu.moe_counts[rank]);
    checked(hipGetLastError());
    std::array<Plan *, 4> received{};
    std::array<Mat, 4> outputs{}, returns{};
    auto capture = op.capture;
    auto segment = capture && (groups > 1 || !op.batch_decode) ? &capture->plan.layers[layer - 1] : nullptr;
    for (int r = 1; r < groups; ++r) {
        auto &arena = gpu.workspace[r];
        arena.cursor = 0;
        received[r] = static_cast<Plan *>(arena.next(stride));
        if (capture) {
            peer_copy(received[r], packets + r * stride, stride, op.stream);
            returns[r] = op.mat(x.rows, 1536);
        } else
            gpu.transfer(received[r], r, packets + r * stride, rank, stride);
    }
    if (segment)
        capture->finish(segment->scatter);
    for (int r = 1; r < groups; ++r) {
        if (capture)
            capture->begin(r);
        checked(hipSetDevice(gpu.physical[r]));
        Ops remote(gpu, r);
        Mat input{reinterpret_cast<uint16_t *>(received[r] + 1), x.rows, 1536};
        outputs[r] = decode_experts(remote, layer, input, received[r], remote.stream);
        if (capture) {
            peer_copy(returns[r].p, outputs[r].p, outputs[r].bytes(), remote.stream);
            capture->finish(segment->expert[r]);
        }
    }
    checked(hipSetDevice(gpu.physical[rank]));
    auto local_stream = gpu.local_expert[rank];
    if (capture) {
        if (segment) capture->begin(rank);
        local_stream = op.stream;
    } else {
        checked(hipEventRecord(gpu.local_ready[rank], op.stream));
        checked(hipStreamWaitEvent(local_stream, gpu.local_ready[rank], 0));
    }
    Mat result = decode_experts(op, layer, latent, reinterpret_cast<Plan *>(packets), local_stream);
    if (!capture)
        checked(hipEventRecord(gpu.local_done[rank], local_stream));
    if (segment && op.batch_decode) {
        capture->finish(segment->local);
        capture->begin(rank);
    }
    Mat shared = feedforward(op, prefix + "shared_experts", x);
    if (segment) {
        capture->finish(segment->expert[rank]);
        capture->begin(rank);
    } else if (!capture)
        checked(hipStreamWaitEvent(op.stream, gpu.local_done[rank], 0));
    for (int r = 1; r < groups; ++r) {
        Mat local = returns[r];
        if (!capture) {
            local = op.mat(x.rows, 1536);
            gpu.transfer(local.p, rank, outputs[r].p, r, local.bytes());
        }
        op.add_into(result, local, result);
    }
    if (gpu.tp_size > 1) {
        gpu.tp_sum(result.p, rank, result.bytes());
        gpu.ep_reduce(result.p, rank, result.bytes());
    }
    if (gpu.expert_begin[rank]) {
        for (int i = 0; i < groups; ++i) {
            int r = rank + i;
            checked(hipSetDevice(gpu.physical[r]));
            checked(hipEventSynchronize(gpu.expert_end[r]));
            float ms;
            checked(hipEventElapsedTime(&ms, gpu.expert_begin[r], gpu.expert_end[r]));
            gpu.expert_ms[r] += ms;
        }
        checked(hipSetDevice(gpu.physical[rank]));
    }
    return op.norm(op.add(shared, op.linear(prefix + "routed_expert_up_proj", result)),
                   prefix + "combine_output_norm.weight");
}
Mat grouped(Ops &op, Mat x, const Hy4Route *routes, int layer, int kind, const Hy4ExpertBatch *batches,
            int count) {
    const int k = x.cols;
    const int n = kind == 2 ? 1536 : (op.gpu.tp_size > 1 ? 768 : 1536);
    Mat y = op.mat(x.rows, n);
    size_t mark = op.arena.cursor;
    auto codes = static_cast<int8_t *>(op.arena.next(size_t(x.rows) * k));
    auto scales = static_cast<float *>(op.arena.next(size_t(x.rows) * (k / 128) * 4));
    auto partial = static_cast<float *>(op.arena.next(size_t(x.rows) * n * 4));
    auto desc = op.gpu.matrices[op.rank] + (layer - 1) * 1536 + kind;
    checked(shelf_hy4_expert_linear(x.p, desc, 3, codes, scales, partial, y.p, x.rows, routes, batches, count,
                                    op.gpu.model.mixed, k, n, op.stream));
    op.arena.cursor = mark;
    return y;
}
}
Mat feedforward(Ops &op, const std::string &p, Mat x, bool clip) {
    if (!clip && op.gpu.tp_size == 1 && !op.gpu.model.mixed && x.rows <= 16 && x.cols == 3072 &&
        p.ends_with(".shared_experts")) {
        auto projection = [&](const std::string &name) {
            const auto *w = op.gpu.model.find(name + ".trellis", op.rank);
            const auto *shape = op.gpu.model.shape_of(name, op.rank);
            if (!w || !shape || *shape != std::array<int, 2>{3072, 1536} ||
                w->spec.shape.size() != 3 || w->spec.shape[0] != 192 ||
                w->spec.shape[1] != 96 || w->spec.shape[2] != 64)
                return Hy4ExpertProjection{};
            return Hy4ExpertProjection{static_cast<const uint16_t *>(op.param(name + ".suh").p),
                                       static_cast<const uint16_t *>(op.param(name + ".svh").p),
                                       static_cast<const uint32_t *>(op.gpu.device(*w)), 4, 3072, 1536};
        };
        auto gate = projection(p + ".gate_proj"), up = projection(p + ".up_proj");
        if (gate.trellis && up.trellis) {
            Mat activation = op.mat(x.rows, 1536);
            size_t mark = op.arena.cursor;
            auto partial = static_cast<float *>(op.arena.next(size_t(2 * 8 * x.rows * 1536) * sizeof(float)));
            checked(shelf_hy4_shared_gate_up(x.p, gate, up, partial, activation.p, x.rows, op.stream));
            op.arena.cursor = mark;
            return op.linear(p + ".down_proj", activation);
        }
    }
    Mat gate = op.linear(p + ".gate_proj", x);
    Mat up = op.linear(p + ".up_proj", x);
    return op.linear(p + ".down_proj", op.swiglu(gate, up, clip));
}
static Mat moe_ep(Ops &op, int layer, const std::string &p, Mat x, Mat latent, const Hy4Route *picks) {
    std::vector<Hy4Route> host(size_t(x.rows) * 16);
    op.copy(host.data(), picks, host.size() * sizeof(Hy4Route), hipMemcpyDeviceToHost);
    std::array<std::vector<Hy4Route>, 4> routes;
    std::array<std::vector<int>, 4> offsets;
    partition_routes(host, x.rows, op.gpu.model.owners[layer - 1], op.gpu.ranks, routes, offsets);
    std::array<Mat, 4> result{};
    for (int iteration = 0; iteration < op.gpu.ranks; ++iteration) {
        int r = (iteration + 1) % op.gpu.ranks;
        if (routes[r].empty())
            continue;
        checked(hipSetDevice(op.gpu.physical[r]));
        Ops rank(op.gpu, r);
        if (r)
            rank.arena.cursor = 0;
        Mat input = latent;
        if (r) {
            input = rank.mat(x.rows, 1536);
            op.gpu.transfer(input.p, r, latent.p, 0, latent.bytes());
        }
        int count = routes[r].size();
        op.gpu.routed_rows[r] += count;
        std::vector<Hy4ExpertBatch> groups;
        std::vector<int> undo;
        group_routes(routes[r], undo, groups);
        offsets[r].push_back(count);
        size_t route_bytes = count * sizeof(Hy4Route), offset_bytes = offsets[r].size() * sizeof(int);
        size_t undo_bytes = undo.size() * sizeof(int), group_bytes = groups.size() * sizeof(Hy4ExpertBatch);
        size_t total = route_bytes + offset_bytes + undo_bytes + group_bytes;
        auto host = static_cast<uint8_t *>(op.gpu.route_buffer(r, total));
        std::memcpy(host, routes[r].data(), route_bytes);
        std::memcpy(host + route_bytes, offsets[r].data(), offset_bytes);
        if (undo_bytes)
            std::memcpy(host + route_bytes + offset_bytes, undo.data(), undo_bytes);
        if (group_bytes)
            std::memcpy(host + route_bytes + offset_bytes + undo_bytes, groups.data(), group_bytes);
        auto dev = static_cast<uint8_t *>(rank.arena.next(total));
        rank.copy(dev, host, total, hipMemcpyHostToDevice);
        auto gpu_routes = reinterpret_cast<const Hy4Route *>(dev);
        auto offsets_gpu = reinterpret_cast<const int *>(dev + route_bytes);
        auto inverse = undo_bytes ? reinterpret_cast<const int *>(dev + route_bytes + offset_bytes) : nullptr;
        auto batches =
            group_bytes
                ? reinterpret_cast<const Hy4ExpertBatch *>(dev + route_bytes + offset_bytes + undo_bytes)
                : nullptr;
        int num_batches = groups.size();
        Mat a = rank.mat(count, 1536);
        if (op.gpu.expert_begin[r])
            checked(hipEventRecord(op.gpu.expert_begin[r], rank.stream));
        gather_route<<<dim3(6, count), 256, 0, rank.stream>>>(input, gpu_routes, a);
        checked(hipGetLastError());
        Mat g = grouped(rank, a, gpu_routes, layer, 0, batches, num_batches),
            u = grouped(rank, a, gpu_routes, layer, 1, batches, num_batches);
        Mat v = grouped(rank, rank.swiglu(g, u, true), gpu_routes, layer, 2, batches, num_batches);
        result[r] = rank.mat(x.rows, 1536);
        weighted<<<dim3(6, x.rows), 256, 0, rank.stream>>>(v, gpu_routes, offsets_gpu, inverse, result[r]);
        checked(hipGetLastError());
        if (op.gpu.expert_end[r])
            checked(hipEventRecord(op.gpu.expert_end[r], rank.stream));
    }
    checked(hipSetDevice(op.gpu.physical[0]));
    Mat shared = feedforward(op, p + "shared_experts", x), combined{};
    for (int r = 0; r < op.gpu.ranks; ++r)
        if (result[r].p) {
            Mat local = result[r];
            if (r) {
                local = op.mat(x.rows, 1536);
                op.gpu.transfer(local.p, 0, result[r].p, r, local.bytes());
            }
            combined = combined.p ? op.add(combined, local) : local;
        }
    if (op.gpu.expert_begin[0]) {
        for (int r = 0; r < op.gpu.ranks; ++r)
            if (result[r].p) {
                checked(hipSetDevice(op.gpu.physical[r]));
                checked(hipEventSynchronize(op.gpu.expert_end[r]));
                float ms;
                checked(hipEventElapsedTime(&ms, op.gpu.expert_begin[r], op.gpu.expert_end[r]));
                op.gpu.expert_ms[r] += ms;
            }
        checked(hipSetDevice(op.gpu.physical[0]));
    }
    return op.norm(op.add(shared, op.linear(p + "routed_expert_up_proj", combined)),
                   p + "combine_output_norm.weight");
}

Mat moe(Ops &op, int layer, Mat x) {
    std::string p = "model.layers." + std::to_string(layer) + ".mlp.";
    Mat latent = op.linear(p + "routed_expert_down_proj", x);
    auto scores = static_cast<float *>(op.arena.next(size_t(x.rows) * 512 * 4));
    router<<<dim3(128, x.rows), 128, 0, op.stream>>>(x, static_cast<const uint16_t *>(op.param(p + "gate.weight").p), scores);
    checked(hipGetLastError());
    auto picks = static_cast<Hy4Route *>(op.arena.next(size_t(x.rows) * 16 * sizeof(Hy4Route)));
    choose<<<x.rows, 256, 0, op.stream>>>(scores, static_cast<const float *>(op.param(p + "gate.e_score_correction_bias").p),
                            picks);
    checked(hipGetLastError());
    if (op.batch_decode && x.rows > 1 && x.rows <= Hy4GroupedPlan::max_rows)
        return decode_moe<Hy4GroupedPlan>(op, layer, p, x, latent, picks);
    if (Hy4MoePlan::supports(x.rows))
        return decode_moe<Hy4MoePlan>(op, layer, p, x, latent, picks);
    if (op.gpu.tp_size == 1)
        return moe_ep(op, layer, p, x, latent, picks);
    std::vector<Hy4Route> host(size_t(x.rows) * 16);
    op.copy(host.data(), picks, host.size() * sizeof(Hy4Route), hipMemcpyDeviceToHost);
    std::array<std::vector<Hy4Route>, 4> routes;
    std::array<std::vector<int>, 4> offsets;
    partition_routes(host, x.rows, op.gpu.model.owners[layer - 1], op.gpu.ep_size, routes, offsets);
    const int group = op.rank / op.gpu.tp_size;
    Mat result = op.mat(x.rows, 1536);
    if (routes[group].empty()) {
        checked(hipMemsetAsync(result.p, 0, result.bytes(), op.stream));
    } else {
        int count = routes[group].size();
        op.gpu.routed_rows[op.rank] += count;
        std::vector<Hy4ExpertBatch> groups;
        std::vector<int> undo;
        group_routes(routes[group], undo, groups);
        offsets[group].push_back(count);
        size_t route_bytes = count * sizeof(Hy4Route), offset_bytes = offsets[group].size() * sizeof(int);
        size_t undo_bytes = undo.size() * sizeof(int), group_bytes = groups.size() * sizeof(Hy4ExpertBatch);
        size_t total = route_bytes + offset_bytes + undo_bytes + group_bytes;
        auto host_routes = static_cast<uint8_t *>(op.gpu.route_buffer(op.rank, total));
        std::memcpy(host_routes, routes[group].data(), route_bytes);
        std::memcpy(host_routes + route_bytes, offsets[group].data(), offset_bytes);
        if (undo_bytes)
            std::memcpy(host_routes + route_bytes + offset_bytes, undo.data(), undo_bytes);
        if (group_bytes)
            std::memcpy(host_routes + route_bytes + offset_bytes + undo_bytes, groups.data(), group_bytes);
        auto dev = static_cast<uint8_t *>(op.arena.next(total));
        op.copy(dev, host_routes, total, hipMemcpyHostToDevice);
        auto gpu_routes = reinterpret_cast<const Hy4Route *>(dev);
        auto offsets_gpu = reinterpret_cast<const int *>(dev + route_bytes);
        auto inverse = undo_bytes ? reinterpret_cast<const int *>(dev + route_bytes + offset_bytes) : nullptr;
        auto batches =
            group_bytes
                ? reinterpret_cast<const Hy4ExpertBatch *>(dev + route_bytes + offset_bytes + undo_bytes)
                : nullptr;
        int num_batches = groups.size();
        Mat a = op.mat(count, 1536);
        if (op.gpu.expert_begin[op.rank])
            checked(hipEventRecord(op.gpu.expert_begin[op.rank], op.stream));
        gather_route<<<dim3(6, count), 256, 0, op.stream>>>(latent, gpu_routes, a);
        checked(hipGetLastError());
        Mat g = grouped(op, a, gpu_routes, layer, 0, batches, num_batches),
            u = grouped(op, a, gpu_routes, layer, 1, batches, num_batches);
        Mat v = grouped(op, op.swiglu(g, u, true), gpu_routes, layer, 2, batches, num_batches);
        weighted<<<dim3(6, x.rows), 256, 0, op.stream>>>(v, gpu_routes, offsets_gpu, inverse, result);
        checked(hipGetLastError());
    }
    op.gpu.tp_sum(result.p, op.rank, result.bytes());
    if (!routes[group].empty() && op.gpu.expert_end[op.rank])
        checked(hipEventRecord(op.gpu.expert_end[op.rank], op.stream));
    if (op.gpu.ep_size > 1)
        op.gpu.ep_reduce(result.p, op.rank, result.bytes());
    if (op.rank == 0 && op.gpu.expert_begin[0] && !routes[group].empty()) {
        checked(hipEventSynchronize(op.gpu.expert_end[0]));
        float ms;
        checked(hipEventElapsedTime(&ms, op.gpu.expert_begin[0], op.gpu.expert_end[0]));
        op.gpu.expert_ms[0] += ms;
    }
    Mat shared = feedforward(op, p + "shared_experts", x);
    return op.norm(op.add(shared, op.linear(p + "routed_expert_up_proj", result)),
                   p + "combine_output_norm.weight");
}
}
