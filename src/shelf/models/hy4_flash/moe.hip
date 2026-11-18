#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <numeric>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
struct Route {
    int row;
    float weight;
    Hy4ExpertProjection proj[3];
};
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
__global__ void gather_route(Mat x, const Route *route, Mat out) {
    int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    out.p[r * 1536 + c] = x.p[route[r].row * 1536 + c];
}
__global__ void weighted(Mat x, const Route *route, const int *offsets, Mat out) {
    int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    float sum = 0;
    for (int j = offsets[r]; j < offsets[r + 1]; ++j)
        sum = round_bf16(sum + round_bf16(route[j].weight * bf16_bits_to_f32(x.p[j * 1536 + c])));
    out.p[r * 1536 + c] = f32_to_bf16_rne_device(sum);
}
Mat grouped(Ops &op, Mat x, const Route *routes, int kind) {
    Mat y = op.mat(x.rows, 1536);
    size_t mark = op.arena.cursor;
    auto codes = static_cast<int8_t *>(op.arena.next(size_t(x.rows) * 1536));
    auto scales = static_cast<float *>(op.arena.next(size_t(x.rows) * 12 * 4));
    auto partial = static_cast<float *>(op.arena.next(size_t(x.rows) * 1536 * 4));
    auto desc = static_cast<Hy4ExpertProjection *>(op.arena.next(sizeof(Hy4ExpertProjection) * x.rows));
    checked(hipMemcpy2DAsync(desc, sizeof(*desc), &routes->proj[kind], sizeof(Route), sizeof(*desc), x.rows,
                             hipMemcpyDeviceToDevice));
    checked(shelf_hy4_expert_linear(x.p, desc, 1, codes, scales, partial, y.p, x.rows));
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
    std::vector<float> host(size_t(x.rows) * 512);
    op.copy(host.data(), scores, host.size() * 4, hipMemcpyDeviceToHost);
    auto bias = static_cast<const float *>(op.gpu.model.at(p + "gate.e_score_correction_bias").host);
    std::array<std::vector<Route>, 3> routes;
    std::array<std::vector<int>, 3> offsets;
    for (int i = 0; i < x.rows; ++i) {
        for (int r = 0; r < 3; ++r)
            offsets[r].push_back(routes[r].size());
        int ids[512];
        std::iota(ids, ids + 512, 0);
        const auto prob = host.data() + i * 512;
        std::partial_sort(ids, ids + 16, ids + 512, [&](int a, int b) {
            float av = prob[a] + bias[a], bv = prob[b] + bias[b];
            return av == bv ? a < b : av > bv;
        });
        float sum = 1e-20f;
        for (int j = 0; j < 16; ++j)
            sum += prob[ids[j]];
        std::sort(ids, ids + 16);
        for (int j = 0; j < 16; ++j) {
            int e = ids[j], r = op.gpu.model.owners[layer - 1][e];
            Route route{i, prob[e] * 3.998f / sum, {}};
            std::string ep = p + "experts." + std::to_string(e) + ".";
            int k = 0;
            for (auto suffix : {"gate_proj", "up_proj", "down_proj"}) {
                auto matrix = ep + suffix;
                auto ptr = [&](const char *part) { return op.gpu.device(op.gpu.model.at(matrix + part)); };
                route.proj[k++] = {static_cast<const uint16_t *>(ptr(".suh")),
                                   static_cast<const uint16_t *>(ptr(".svh")),
                                   static_cast<const uint32_t *>(ptr(".trellis"))};
            }
            routes[r].push_back(route);
        }
    }
    std::array<Mat, 3> result{};
    for (int r : {1, 2, 0}) {
        if (routes[r].empty())
            continue;
        checked(hipSetDevice(r));
        Ops rank(op.gpu, r);
        if (r)
            rank.arena.cursor = 0;
        Mat input = latent;
        if (r) {
            input = rank.mat(x.rows, 1536);
            op.gpu.transfer(input.p, r, latent.p, 0, latent.bytes());
        }
        int count = routes[r].size();
        auto gpu_routes = static_cast<Route *>(rank.arena.next(sizeof(Route) * count));
        rank.copy(gpu_routes, routes[r].data(), sizeof(Route) * count, hipMemcpyHostToDevice);
        offsets[r].push_back(count);
        int *offsets_gpu = rank.indices(offsets[r]);
        Mat a = rank.mat(count, 1536);
        gather_route<<<dim3(6, count), 256>>>(input, gpu_routes, a);
        checked(hipGetLastError());
        Mat g = grouped(rank, a, gpu_routes, 0), u = grouped(rank, a, gpu_routes, 1);
        Mat v = grouped(rank, rank.swiglu(g, u, true), gpu_routes, 2);
        result[r] = rank.mat(x.rows, 1536);
        weighted<<<dim3(6, x.rows), 256>>>(v, gpu_routes, offsets_gpu, result[r]);
        checked(hipGetLastError());
    }
    checked(hipSetDevice(0));
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
    return op.norm(op.add(shared, op.linear(p + "routed_expert_up_proj", combined)),
                   p + "combine_output_norm.weight");
}
}
