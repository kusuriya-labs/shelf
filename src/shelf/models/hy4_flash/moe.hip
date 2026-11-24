#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <rocprim/block/block_scan.hpp>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
__global__ void router_single(Mat x, const uint16_t *weight, float *scores) {
    int expert = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32, row = blockIdx.y;
    float dot = 0;
    for (int c = lane; c < 3072; c += 32)
        dot += bf16_bits_to_f32(x.p[row * 3072 + c]) * bf16_bits_to_f32(weight[expert * 3072 + c]);
    for (int d = 16; d; d /= 2)
        dot += __shfl_down(dot, d, 32);
    if (!lane)
        scores[row * 512 + expert] = 1.f / (1.f + expf(-dot));
}
__global__ void router(Mat x, const uint16_t *weight, float *scores) {
    int expert = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32, row = blockIdx.y * 4;
    float dot[4] = {};
    for (int c = lane; c < 3072; c += 32) {
        float w = bf16_bits_to_f32(weight[expert * 3072 + c]);
        for (int i = 0; i < 4; ++i)
            if (row + i < x.rows)
                dot[i] += bf16_bits_to_f32(x.p[(row + i) * 3072 + c]) * w;
    }
    for (int d = 16; d; d /= 2)
        for (int i = 0; i < 4; ++i)
            dot[i] += __shfl_down(dot[i], d, 32);
    if (!lane)
        for (int i = 0; i < 4; ++i)
            if (row + i < x.rows)
                scores[(row + i) * 512 + expert] = 1.f / (1.f + expf(-dot[i]));
}
__device__ bool route_better(float a, int ai, float b, int bi) {
    return a > b || (a == b && ai < bi);
}
__global__ void choose(const float *scores, const float *bias, Hy4Route *out) {
    __shared__ float values[2][256], weight_sum;
    __shared__ int ids[2][256];
    int t = threadIdx.x, row = blockIdx.x, lane = t % 32, wave = t / 32;
    scores += row * 512;
    float value = scores[t] + bias[t];
    int id = t;
    for (int width = 2; width <= 32; width *= 2)
        for (int step = width / 2; step; step /= 2) {
            float other = __shfl_xor(value, step, 32);
            int other_id = __shfl_xor(id, step, 32);
            bool keep_best = ((lane & step) == 0) == ((lane & width) == 0);
            if (route_better(value, id, other, other_id) != keep_best) {
                value = other;
                id = other_id;
            }
        }
    if (lane < 16) {
        values[0][wave * 16 + lane] = value;
        ids[0][wave * 16 + lane] = id;
    }
    __syncthreads();
    for (int round = 0; round < 4; ++round) {
        int from = round & 1, to = from ^ 1;
        if (t < (8 >> round)) {
            int left = t * 32, right = left + 16, i = 0, j = 0;
            for (int k = 0; k < 16; ++k) {
                bool take_left = j == 16 || (i < 16 && route_better(values[from][left + i], ids[from][left + i],
                                                                     values[from][right + j], ids[from][right + j]));
                int source = take_left ? left + i++ : right + j++;
                values[to][t * 16 + k] = values[from][source];
                ids[to][t * 16 + k] = ids[from][source];
            }
        }
        __syncthreads();
    }
    if (!t) {
        float sum = 1e-20f;
        for (int j = 0; j < 16; ++j)
            sum += scores[ids[0][j]];
        weight_sum = sum;
    }
    __syncthreads();
    if (t < 16)
        out[row * 16 + t] = {row, ids[0][t], scores[ids[0][t]] / weight_sum * 3.998f};
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
        checked(hipEventSynchronize(gpu.local_done[rank]));
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
__global__ void pack_prefill(Mat latent, const Hy4Route *routes, const int *owners, int groups, unsigned peers,
                             PrefillRoutes *choices, uint4 *copy, uint64_t *totals) {
    int row = blockIdx.x, t = threadIdx.x;
    __shared__ int sorted_groups[16];
    if (t < 16) {
        auto route = routes[row * 16 + t];
        int order = 0;
        for (int j = 0; j < 16; ++j)
            order += __shfl(route.expert, j, 16) < route.expert;
        choices[row].choices[order] = {route.expert, route.weight};
        sorted_groups[order] = owners[route.expert];
    }
    __syncthreads();
    if (t < 16)
        for (int g = 0; g < groups; ++g) {
            unsigned mask = __ballot(sorted_groups[t] == g);
            if (t == 0) {
                choices[row].masks[g] = mask;
                unsigned count = __popc(mask);
                if (count)
                    atomicAdd(reinterpret_cast<unsigned long long *>(totals + g), count);
                if (peers & (1u << g)) {
                    unsigned absent = mask ? 0 : 3072;
                    unsigned saved = sizeof(PrefillRoutes) - sizeof(unsigned) - count * sizeof(PrefillRoutes::Choice);
                    atomicAdd(reinterpret_cast<unsigned long long *>(totals + 4 + g), saved + absent);
                    if (absent) atomicAdd(reinterpret_cast<unsigned long long *>(totals + 8 + g), absent);
                }
            }
        }
    if (copy) {
        auto src = reinterpret_cast<const uint4 *>(latent.p) + row * 192;
        for (int i = t; i < 192; i += blockDim.x)
            copy[row * 192 + i] = src[i];
    }
}
__global__ void copy_prefill(PrefillRoutes *dst, const PrefillRoutes *src, int group) {
    int row = blockIdx.x, t = threadIdx.x;
    unsigned mask = src[row].masks[group];
    if (!t) dst[row].masks[group] = mask;
    if (!mask) return;
    if (t < 16 && (mask & (1u << t)))
        dst[row].choices[t] = src[row].choices[t];
    auto input = reinterpret_cast<const uint4 *>(src + gridDim.x) + row * 192;
    auto output = reinterpret_cast<uint4 *>(dst + gridDim.x) + row * 192;
    for (int i = t; i < 192; i += blockDim.x)
        output[i] = input[i];
}
__global__ void count_prefill(const PrefillRoutes *choices, int size, int group, int *histogram) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && (choices[i / 16].masks[group] & (1u << (i % 16))))
        atomicAdd(histogram + choices[i / 16].choices[i % 16].expert, 1);
}
__global__ void plan_prefill(int *histogram, Hy4ExpertBatch *batches, int *sizes) {
    using Scan = rocprim::block_scan<unsigned, 512>;
    __shared__ typename Scan::storage_type storage;
    int expert = threadIdx.x, count = histogram[expert];
    unsigned prefix;
    Scan().exclusive_scan(unsigned(count + (((count + 31) / 32) << 16)), prefix, 0u, storage);
    int start = prefix & 65535, batch = prefix >> 16;
    histogram[expert] = start;
    for (int i = 0; i < count; i += 32)
        batches[batch++] = {start + i, min(32, count - i)};
    if (expert == 511) {
        sizes[0] = start + count;
        sizes[1] = batch;
    }
}
__global__ void scatter_prefill(const PrefillRoutes *choices, int size, int group, int *cursor,
                                Hy4Route *routes, int *inverse) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    int j = -1;
    if (choices[i / 16].masks[group] & (1u << (i % 16))) {
        auto choice = choices[i / 16].choices[i % 16];
        int expert = choice.expert;
        j = atomicAdd(cursor + expert, 1);
        routes[j] = {i / 16, expert, choice.weight};
    }
    inverse[i] = j;
}
__global__ void sum_prefill(Mat x, const Hy4Route *routes, const int *inverse, Mat out, const PrefillRoutes *choices, int group) {
    int row = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (choices && !choices[row].masks[group]) return;
    float sum = 0;
    for (int i = 0; i < 16; ++i) {
        int j = inverse[row * 16 + i];
        if (j >= 0)
            sum = round_bf16(sum + round_bf16(routes[j].weight * bf16_bits_to_f32(x.p[j * 1536 + c])));
    }
    out.p[row * 1536 + c] = f32_to_bf16_rne_device(sum);
}
__global__ void add_prefill(Mat sum, Mat partial, const PrefillRoutes *choices, int group) {
    int row = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (choices[row].masks[group]) {
        size_t i = size_t(row) * 1536 + c;
        sum.p[i] = f32_to_bf16_rne_device(bf16_bits_to_f32(sum.p[i]) + bf16_bits_to_f32(partial.p[i]));
    }
}
void prefill_experts(Ops &op, int layer, Mat latent, const PrefillRoutes *choices, Mat out, bool sparse_output = false) {
    const int group = op.rank / op.gpu.tp_size, capacity = latent.rows * 16;
    const int intermediate = 1536 / op.gpu.tp_size;
    auto mark = op.arena.cursor;
    auto histogram = static_cast<int *>(op.arena.next(512 * sizeof(int)));
    auto sizes = static_cast<int *>(op.arena.next(2 * sizeof(int)));
    auto inverse = static_cast<int *>(op.arena.next(capacity * sizeof(int)));
    auto routes = static_cast<Hy4Route *>(op.arena.next(capacity * sizeof(Hy4Route)));
    auto batches = static_cast<Hy4ExpertBatch *>(op.arena.next((512 + capacity / 32) * sizeof(Hy4ExpertBatch)));
    checked(hipMemsetAsync(histogram, 0, 512 * sizeof(int), op.stream));
    int blocks = (capacity + 255) / 256;
    count_prefill<<<blocks, 256, 0, op.stream>>>(choices, capacity, group, histogram);
    plan_prefill<<<1, 512, 0, op.stream>>>(histogram, batches, sizes);
    scatter_prefill<<<blocks, 256, 0, op.stream>>>(choices, capacity, group, histogram, routes, inverse);
    checked(hipGetLastError());
    auto codes = static_cast<int8_t *>(op.arena.next(size_t(capacity) * 2 * 1536));
    auto scales = static_cast<float *>(op.arena.next(size_t(capacity) * 2 * 12 * sizeof(float)));
    auto partial = static_cast<float *>(op.arena.next(size_t(capacity) * 2 * intermediate * sizeof(float)));
    Mat values = op.mat(capacity, 1536);
    auto matrices = op.gpu.matrices[op.rank] + (layer - 1) * 1536;
    checked(shelf_hy4_expert_decode(latent.p, matrices, routes, sizes, capacity, intermediate, codes, scales,
                                    partial, values.p, op.stream, batches, sizes + 1, true));
    sum_prefill<<<dim3(6, latent.rows), 256, 0, op.stream>>>(values, routes, inverse, out, sparse_output ? choices : nullptr, group);
    checked(hipGetLastError());
    op.arena.cursor = mark;
}
}
void peer_prefill_packet(void *dst, const void *src, int rows, int group, hipStream_t stream) {
    copy_prefill<<<rows, 128, 0, stream>>>(static_cast<PrefillRoutes *>(dst), static_cast<const PrefillRoutes *>(src), group);
    checked(hipGetLastError());
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
static Mat prefill_moe(Ops &op, int layer, const std::string &p, Mat x, Mat latent, const Hy4Route *picks) {
    auto &gpu = op.gpu;
    int groups = gpu.tp_size == 1 ? gpu.ep_size : 1;
    unsigned active = 0;
    if (groups > 1)
        for (int owner : gpu.model.owners[layer - 1])
            active |= 1u << owner;
    Mat result = op.mat(x.rows, 1536);
    auto mark = op.arena.cursor;
    for (int first = 0; first < x.rows; first += expert_tile_rows) {
        Mat tile = latent.row(first, std::min(expert_tile_rows, x.rows - first));
        size_t choice_bytes = tile.rows * sizeof(PrefillRoutes);
        size_t bytes = choice_bytes + (groups > 1 ? tile.bytes() : 0);
        auto packet = static_cast<uint8_t *>(op.arena.next(bytes));
        auto choices = reinterpret_cast<PrefillRoutes *>(packet);
        auto copy = groups > 1 ? reinterpret_cast<uint4 *>(packet + choice_bytes) : nullptr;
        pack_prefill<<<tile.rows, 128, 0, op.stream>>>(tile, picks + first * 16, gpu.moe_owners[op.rank] + (layer - 1) * 512,
                                                      gpu.ep_size, active & ~1u, choices, copy, gpu.moe_counts[op.rank]);
        checked(hipGetLastError());
        std::array<Mat, 4> remote{};
        std::array<bool, 4> peer_result{};
        for (int r = 1; r < groups; ++r) {
            if (!(active & (1u << r))) continue;
            peer_result[r] = gpu.push ? gpu.peer[r][0] : gpu.peer[0][r];
            bool direct = gpu.push && peer_result[r];
            Ops rank(gpu, r);
            rank.arena.cursor = 0;
            auto recv = static_cast<uint8_t *>(rank.arena.next(bytes));
            gpu.transfer(recv, r, packet, 0, bytes, tile.rows);
            Mat input{reinterpret_cast<uint16_t *>(recv + choice_bytes), tile.rows, tile.cols};
            if (gpu.expert_begin[r]) checked(hipEventRecord(gpu.expert_begin[r], rank.stream));
            remote[r] = direct ? op.mat(tile.rows, 1536) : rank.mat(tile.rows, 1536);
            prefill_experts(rank, layer, input, reinterpret_cast<PrefillRoutes *>(recv), remote[r], peer_result[r]);
            if (peer_result[r]) {
                checked(hipEventSynchronize(gpu.ready[r][0][0]));
                checked(hipEventRecord(gpu.ready[r][0][0], rank.stream));
            }
            if (gpu.expert_end[r]) checked(hipEventRecord(gpu.expert_end[r], rank.stream));
        }
        checked(hipSetDevice(gpu.physical[op.rank]));
        if (gpu.expert_begin[op.rank]) checked(hipEventRecord(gpu.expert_begin[op.rank], op.stream));
        Mat sum = result.row(first, tile.rows);
        prefill_experts(op, layer, tile, choices, sum);
        if (gpu.expert_end[op.rank]) checked(hipEventRecord(gpu.expert_end[op.rank], op.stream));
        for (int r = 1; r < groups; ++r) {
            if (!(active & (1u << r))) continue;
            if (peer_result[r]) {
                checked(hipStreamWaitEvent(op.stream, gpu.ready[r][0][0], 0));
                add_prefill<<<dim3(6, tile.rows), 256, 0, op.stream>>>(sum, remote[r], choices, r);
                checked(hipGetLastError());
                gpu.transfer_bytes[r][0] += sum.bytes();
                if (!gpu.push) {
                    checked(hipEventSynchronize(gpu.ack[0][r][0]));
                    checked(hipEventRecord(gpu.ack[0][r][0], op.stream));
                    checked(hipSetDevice(gpu.physical[r]));
                    checked(hipStreamWaitEvent(gpu.compute[r], gpu.ack[0][r][0], 0));
                    checked(hipSetDevice(gpu.physical[0]));
                }
            } else {
                Mat local = op.mat(tile.rows, 1536);
                gpu.transfer(local.p, 0, remote[r].p, r, local.bytes());
                op.add_into(sum, local, sum);
            }
        }
        if (gpu.expert_begin[op.rank]) {
            for (int r = op.rank; r < op.rank + groups; ++r) {
                if (r != op.rank && !(active & (1u << r))) continue;
                checked(hipSetDevice(gpu.physical[r]));
                checked(hipEventSynchronize(gpu.expert_end[r]));
                float ms;
                checked(hipEventElapsedTime(&ms, gpu.expert_begin[r], gpu.expert_end[r]));
                gpu.expert_ms[r] += ms;
            }
            checked(hipSetDevice(gpu.physical[op.rank]));
        }
        op.arena.cursor = mark;
    }
    if (gpu.tp_size > 1) {
        gpu.tp_sum(result.p, op.rank, result.bytes());
        gpu.ep_reduce(result.p, op.rank, result.bytes());
    }
    Mat shared = feedforward(op, p + "shared_experts", x);
    return op.norm(op.add(shared, op.linear(p + "routed_expert_up_proj", result)), p + "combine_output_norm.weight");
}

Mat moe(Ops &op, int layer, Mat x) {
    std::string p = "model.layers." + std::to_string(layer) + ".mlp.";
    Mat latent = op.linear(p + "routed_expert_down_proj", x);
    auto scores = static_cast<float *>(op.arena.next(size_t(x.rows) * 512 * 4));
    if (x.rows <= 16)
        router_single<<<dim3(128, x.rows), 128, 0, op.stream>>>(x, static_cast<const uint16_t *>(op.param(p + "gate.weight").p), scores);
    else
        router<<<dim3(128, (x.rows + 3) / 4), 128, 0, op.stream>>>(x, static_cast<const uint16_t *>(op.param(p + "gate.weight").p), scores);
    checked(hipGetLastError());
    auto picks = static_cast<Hy4Route *>(op.arena.next(size_t(x.rows) * 16 * sizeof(Hy4Route)));
    choose<<<x.rows, 512, 0, op.stream>>>(scores, static_cast<const float *>(op.param(p + "gate.e_score_correction_bias").p),
                            picks);
    checked(hipGetLastError());
    if (op.batch_decode && x.rows > 1 && x.rows <= Hy4GroupedPlan::max_rows)
        return decode_moe<Hy4GroupedPlan>(op, layer, p, x, latent, picks);
    if (Hy4MoePlan::supports(x.rows))
        return decode_moe<Hy4MoePlan>(op, layer, p, x, latent, picks);
    return prefill_moe(op, layer, p, x, latent, picks);
}
}
