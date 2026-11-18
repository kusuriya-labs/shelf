#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <cstring>
#include <numeric>

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
Mat grouped(Ops &op, Mat x, const Hy4Route *routes, int layer, int kind, const Hy4ExpertBatch *batches,
            int count, bool wmma) {
    Mat y = op.mat(x.rows, 1536);
    size_t mark = op.arena.cursor;
    auto codes = static_cast<int8_t *>(op.arena.next(size_t(x.rows) * 1536));
    auto scales = static_cast<float *>(op.arena.next(size_t(x.rows) * 12 * 4));
    auto partial = static_cast<float *>(op.arena.next(size_t(x.rows) * 1536 * 4));
    auto desc = op.gpu.matrices[op.rank] + (layer - 1) * 1536 + kind;
    checked(shelf_hy4_expert_linear(x.p, desc, 3, codes, scales, partial, y.p, x.rows, routes, batches, count,
                                    wmma));
    op.arena.cursor = mark;
    return y;
}
}
Mat feedforward(Ops &op, const std::string &p, Mat x, bool clip) {
    Mat gate = op.linear(p + ".gate_proj", x);
    Mat up = op.linear(p + ".up_proj", x);
    return op.linear(p + ".down_proj", op.swiglu(gate, up, clip));
}
Mat moe(Ops &op, int layer, Mat x) {
    std::string p = "model.layers." + std::to_string(layer) + ".mlp.";
    Mat latent = op.linear(p + "routed_expert_down_proj", x);
    auto scores = static_cast<float *>(op.arena.next(size_t(x.rows) * 512 * 4));
    router<<<dim3(128, x.rows), 128>>>(x, static_cast<const uint16_t *>(op.param(p + "gate.weight").p), scores);
    checked(hipGetLastError());
    auto picks = static_cast<Hy4Route *>(op.arena.next(size_t(x.rows) * 16 * sizeof(Hy4Route)));
    choose<<<x.rows, 256>>>(scores, static_cast<const float *>(op.param(p + "gate.e_score_correction_bias").p),
                            picks);
    checked(hipGetLastError());
    std::vector<Hy4Route> host(size_t(x.rows) * 16);
    op.copy(host.data(), picks, host.size() * sizeof(Hy4Route), hipMemcpyDeviceToHost);
    std::array<std::vector<Hy4Route>, 3> routes;
    std::array<std::vector<int>, 3> offsets;
    for (int row = 0; row < x.rows; ++row) {
        for (int r = 0; r < 3; ++r)
            offsets[r].push_back(routes[r].size());
        auto first = host.begin() + row * 16;
        std::sort(first, first + 16, [](const auto &a, const auto &b) { return a.expert < b.expert; });
        for (auto it = first; it != first + 16; ++it)
            routes[op.gpu.model.owners[layer - 1][it->expert]].push_back(*it);
    }
    std::array<Mat, 3> result{};
    for (int r : {1, 2, 0}) {
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
        bool use_wmma = r == 0;
        if (x.rows >= 8) {
            auto &route = routes[r];
            std::vector<int> order(count);
            undo.resize(count);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int i, int j) { return route[i].expert < route[j].expert; });
            std::vector<Hy4Route> sorted(count);
            for (int j = 0; j < count; ++j) {
                sorted[j] = route[order[j]];
                undo[order[j]] = j;
                if (!j || sorted[j - 1].expert != sorted[j].expert || groups.back().count == 16)
                    groups.push_back({j, 0});
                ++groups.back().count;
            }
            route.swap(sorted);
        }
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
        checked(hipMemcpyAsync(dev, host, total, hipMemcpyHostToDevice));
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
            checked(hipEventRecord(op.gpu.expert_begin[r]));
        gather_route<<<dim3(6, count), 256>>>(input, gpu_routes, a);
        checked(hipGetLastError());
        Mat g = grouped(rank, a, gpu_routes, layer, 0, batches, num_batches, use_wmma),
            u = grouped(rank, a, gpu_routes, layer, 1, batches, num_batches, use_wmma);
        Mat v = grouped(rank, rank.swiglu(g, u, true), gpu_routes, layer, 2, batches, num_batches, use_wmma);
        result[r] = rank.mat(x.rows, 1536);
        weighted<<<dim3(6, x.rows), 256>>>(v, gpu_routes, offsets_gpu, inverse, result[r]);
        checked(hipGetLastError());
        if (op.gpu.expert_end[r])
            checked(hipEventRecord(op.gpu.expert_end[r]));
    }
    checked(hipSetDevice(op.gpu.physical[0]));
    Mat shared = feedforward(op, p + "shared_experts", x), combined{};
    for (int r = 0; r < 3; ++r)
        if (result[r].p) {
            Mat local = result[r];
            if (r) {
                local = op.mat(x.rows, 1536);
                op.gpu.transfer(local.p, 0, result[r].p, r, local.bytes());
            }
            combined = combined.p ? op.add(combined, local) : local;
        }
    if (op.gpu.expert_begin[0]) {
        for (int r = 0; r < 3; ++r)
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
}
