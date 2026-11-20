#include <shelf/kernels/hy4_flash/linear.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

void check(hipError_t e, const char *what) {
    if (e != hipSuccess) {
        fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

uint32_t next_rand(uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

struct Args {
    int rows = 6;
    int groups = 0;
    int warmup = 5;
    int iters = 30;
    std::string save;
    std::string reference;
    bool pair = false;
    bool twice = false;
};

Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto value = [&]() -> int {
            if (i + 1 >= argc)
                std::exit(2);
            return std::atoi(argv[++i]);
        };
        if (s == "--rows")
            a.rows = value();
        else if (s == "--groups")
            a.groups = value();
        else if (s == "--warmup")
            a.warmup = value();
        else if (s == "--iters")
            a.iters = value();
        else if (s == "--save")
            a.save = argv[++i];
        else if (s == "--ref")
            a.reference = argv[++i];
        else if (s == "--pair")
            a.pair = true;
        else if (s == "--twice")
            a.twice = true;
        else
            std::exit(2);
    }
    if (a.rows < 1 || a.rows > 64 || a.groups < 0 || a.groups > 4 || a.iters < 1)
        std::exit(2);
    return a;
}

}  // namespace

int main(int argc, char **argv) {
    Args a = parse(argc, argv);
    check(hipSetDevice(0), "hipSetDevice");
    constexpr int kExperts = 4;
    constexpr size_t kWords = 96 * 96 * 32;
    constexpr size_t kCodes = 1536;
    constexpr size_t kScales = 12;
    constexpr size_t kCols = 1536;

    uint32_t seed = 0x12345678u;
    std::vector<uint32_t> trellis_host(kWords);
    for (auto &w : trellis_host)
        w = next_rand(seed);
    std::vector<uint16_t> ones_all(kExperts * 3 * 2 * 1536, 0x3c00);
    std::vector<Hy4ExpertProjection> matrices_host(kExperts * 3);
    std::vector<uint32_t> *trellis_dev = new std::vector<uint32_t>[kExperts * 3];
    for (int e = 0; e < kExperts; ++e) {
        for (int k = 0; k < 3; ++k) {
            int id = e * 3 + k;
            auto &td = trellis_dev[id] = trellis_host;
            matrices_host[id] = {nullptr, nullptr, td.data(), 4};
        }
    }

    std::vector<uint16_t> x_host(size_t(a.rows) * 1536);
    for (auto &v : x_host) {
        float f = 2.f * float(int(next_rand(seed) & 65535)) / 65535.f - 1.f;
        v = uint16_t(std::bit_cast<uint32_t>(f) >> 16);
    }
    std::vector<int8_t> codes_host(size_t(a.rows) * kCodes);
    for (auto &v : codes_host)
        v = int8_t(int(next_rand(seed) & 127) - 64);
    std::vector<float> scales_host(size_t(a.rows) * kScales);
    for (auto &v : scales_host)
        v = 0.01f * float(int(next_rand(seed) & 1023)) / 1023.f;
    std::vector<Hy4Route> routes_host(a.rows);
    for (int r = 0; r < a.rows; ++r)
        routes_host[r] = {r, r % kExperts, 1.f};
    std::vector<Hy4ExpertBatch> batches_host;
    if (a.groups)
        for (int g = 0; g < a.groups; ++g) {
            int first = g * a.rows / a.groups;
            int last = (g + 1) * a.rows / a.groups;
            batches_host.push_back({first, last - first});
        }

    uint32_t *trellis = nullptr;
    Hy4ExpertProjection *matrices = nullptr;
    uint16_t *suha = nullptr, *x = nullptr, *y = nullptr;
    int8_t *codes = nullptr;
    float *scales = nullptr, *partial = nullptr, *partial2 = nullptr;
    Hy4Route *routes = nullptr;
    Hy4ExpertBatch *batches = nullptr;
    check(hipMalloc(&trellis, kExperts * 3 * kWords * sizeof(uint32_t)), "trellis alloc");
    check(hipMalloc(&matrices, kExperts * 3 * sizeof(Hy4ExpertProjection)), "matrices alloc");
    check(hipMalloc(&suha, kExperts * 3 * 2 * 1536 * sizeof(uint16_t)), "suh alloc");
    check(hipMalloc(&x, x_host.size() * sizeof(uint16_t)), "x alloc");
    check(hipMalloc(&y, size_t(a.rows) * kCols * sizeof(uint16_t)), "y alloc");
    check(hipMalloc(&codes, codes_host.size()), "codes alloc");
    check(hipMalloc(&scales, scales_host.size() * sizeof(float)), "scales alloc");
    check(hipMalloc(&partial, size_t(a.rows) * kCols * sizeof(float)), "partial alloc");
    if (a.pair)
        check(hipMalloc(&partial2, size_t(a.rows) * kCols * sizeof(float)), "partial2 alloc");
    check(hipMalloc(&routes, routes_host.size() * sizeof(Hy4Route)), "routes alloc");
    if (a.groups)
        check(hipMalloc(&batches, batches_host.size() * sizeof(Hy4ExpertBatch)), "batches alloc");

    for (int id = 0; id < kExperts * 3; ++id) {
        check(hipMemcpy(trellis + size_t(id) * kWords, trellis_dev[id].data(), kWords * sizeof(uint32_t),
                        hipMemcpyHostToDevice), "trellis copy");
        matrices_host[id].trellis = trellis + size_t(id) * kWords;
        matrices_host[id].suh = suha + size_t(id) * 2 * 1536;
        matrices_host[id].svh = suha + size_t(id) * 2 * 1536 + 1536;
    }
    check(hipMemcpy(suha, ones_all.data(), ones_all.size() * sizeof(uint16_t), hipMemcpyHostToDevice), "suh copy");
    check(hipMemcpy(matrices, matrices_host.data(), matrices_host.size() * sizeof(Hy4ExpertProjection),
                    hipMemcpyHostToDevice), "matrices copy");
    check(hipMemcpy(x, x_host.data(), x_host.size() * sizeof(uint16_t), hipMemcpyHostToDevice), "x copy");
    check(hipMemcpy(codes, codes_host.data(), codes_host.size(), hipMemcpyHostToDevice), "codes copy");
    check(hipMemcpy(scales, scales_host.data(), scales_host.size() * sizeof(float), hipMemcpyHostToDevice),
          "scales copy");
    check(hipMemcpy(routes, routes_host.data(), routes_host.size() * sizeof(Hy4Route), hipMemcpyHostToDevice),
          "routes copy");
    if (a.groups)
        check(hipMemcpy(batches, batches_host.data(), batches_host.size() * sizeof(Hy4ExpertBatch),
                        hipMemcpyHostToDevice), "batches copy");

    auto launch = [&]() {
        hipError_t e;
        if (a.pair && a.groups)
            e = shelf_hy4_expert_batch_gate_up_swiglu(x, matrices, routes, batches, codes, scales, partial,
                                                      partial2, y, a.rows, a.groups, 0);
        else if (a.pair)
            e = shelf_hy4_expert_gate_up_swiglu(x, matrices, 3, codes, scales, partial, partial2, y, a.rows,
                                               routes, 0);
        else
            e = shelf_hy4_expert_linear(x, matrices, 3, codes, scales, partial, y, a.rows, routes,
                                        a.groups ? batches : nullptr, a.groups, 0, 0);
        check(e, "expert linear");
        if (a.twice) {
            e = shelf_hy4_expert_linear(x, matrices, 3, codes, scales, partial, y, a.rows, routes,
                                        a.groups ? batches : nullptr, a.groups, 0, 0);
            check(e, "expert linear second");
        }
    };
    for (int i = 0; i < a.warmup; ++i)
        launch();
    check(hipDeviceSynchronize(), "warmup sync");

    hipEvent_t begin, end;
    check(hipEventCreate(&begin), "event create");
    check(hipEventCreate(&end), "event create");
    std::vector<float> ms;
    for (int i = 0; i < a.iters; ++i) {
        check(hipEventRecord(begin, 0), "event record begin");
        launch();
        check(hipEventRecord(end, 0), "event record end");
        check(hipEventSynchronize(end), "event sync");
        float t = 0;
        check(hipEventElapsedTime(&t, begin, end), "elapsed");
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    double sum = 0;
    for (float t : ms)
        sum += t;
    printf("rows=%d groups=%d median_ms=%.6f min_ms=%.6f max_ms=%.6f mean_ms=%.6f\n", a.rows, a.groups,
           ms[ms.size() / 2], ms.front(), ms.back(), sum / ms.size());
    auto verify = [](const std::vector<uint16_t> &actual, const std::string &path) {
        if (path.empty())
            return true;
        std::ifstream in(path, std::ios::binary);
        std::vector<uint16_t> ref(actual.size());
        in.read(reinterpret_cast<char *>(ref.data()), ref.size() * sizeof(uint16_t));
        if (!in || size_t(in.gcount()) != ref.size() * sizeof(uint16_t)) {
            fprintf(stderr, "参照ファイルが不正です: %s\n", path.c_str());
            return false;
        }
        size_t diff = 0, nan = 0;
        double ss = 0, se = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (actual[i] != ref[i])
                ++diff;
            float r = std::bit_cast<float>(uint32_t(ref[i]) << 16);
            float v = std::bit_cast<float>(uint32_t(actual[i]) << 16);
            if (!std::isfinite(r) || !std::isfinite(v))
                ++nan;
            ss += double(r) * r;
            se += double(v - r) * (v - r);
        }
        printf("verify elements=%zu diff=%zu nan=%zu bit_exact=%s relative_l2=%.9g\n", ref.size(), diff, nan,
               diff ? "no" : "yes", ss > 0 && std::isfinite(ss) ? std::sqrt(se / ss) : -1.0);
        return diff == 0 && nan == 0;
    };
    std::vector<uint16_t> y_host(size_t(a.rows) * kCols);
    check(hipMemcpy(y_host.data(), y, y_host.size() * sizeof(uint16_t), hipMemcpyDeviceToHost), "y copy");
    if (!a.save.empty()) {
        std::ofstream out(a.save, std::ios::binary);
        out.write(reinterpret_cast<const char *>(y_host.data()), y_host.size() * sizeof(uint16_t));
    }
    if (!verify(y_host, a.reference))
        return 1;
    check(hipEventDestroy(begin), "event destroy begin");
    check(hipEventDestroy(end), "event destroy end");
    return 0;
}
