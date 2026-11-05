#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/kernels/optimized/moe/combine.h>
#include <shelf/kernels/optimized/moe/router.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace moe = shelf::kernel;

namespace {

constexpr uint32_t kRowTile = moe::kMxfp4GemmRowTile;
// Production feature width (Qwen3.6-35B-A3B hidden 2048). The combine kernel splits features
// over grid.y in kCombineThreads-sized blocks, so this width uses 16 blocks per row.
constexpr uint32_t kFeatures = 2048u;
constexpr uint32_t kExperts = 256u;
constexpr uint32_t kMaxRows = 64u;
constexpr uint32_t kMaxTopK = 8u;
constexpr uint32_t kMaxGroupCapacity = moe::moe_group_count_upper_bound(kMaxRows, kExperts,
                                                                        kMaxTopK);
constexpr uint32_t kMaxPlaneRows = kMaxGroupCapacity * kRowTile;

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

template <typename T>
T* alloc(std::size_t count) {
    T* p = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)) != hipSuccess) {
        std::printf("FAIL hipMalloc %zu\n", count * sizeof(T));
        std::exit(1);
    }
    return p;
}

template <typename T>
std::vector<T> d2h(const T* dev, std::size_t count) {
    std::vector<T> host(count);
    if (count == 0)
        return host;
    if (hipMemcpy(host.data(), dev, count * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL d2h %zu\n", count);
        std::exit(1);
    }
    return host;
}

template <typename T>
void h2d(T* dev, const std::vector<T>& host) {
    if (host.empty())
        return;
    if (hipMemcpy(dev, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL h2d %zu\n", host.size());
        std::exit(1);
    }
}

float bf16_to_f32(shelf::bf16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

struct Device {
    float* logits = nullptr;
    int32_t* sel_expert = nullptr;
    float* sel_weight = nullptr;
    uint32_t* tile_counts = nullptr;
    uint32_t* row_begin = nullptr;
    uint32_t* tile_base = nullptr;
    uint32_t* group_begin = nullptr;
    uint32_t* group_expert = nullptr;
    uint32_t* group_rows = nullptr;
    uint32_t* group_count = nullptr;
    int32_t* row_token = nullptr;
    int32_t* slot_row_of = nullptr;
    shelf::bf16_t* expert_out = nullptr;
    shelf::bf16_t* shared_out = nullptr;
    float* shared_gate = nullptr;
    shelf::bf16_t* out = nullptr;
};

struct Case {
    uint32_t rows = 1;
    uint32_t top_k = 2;
    bool shared = false;
    uint32_t group_capacity = 0;
    // The gate plane is a padded row-major value in the program workspace, so a row's gate
    // is not necessarily the next float. The kernel has to be told the stride.
    uint32_t gate_stride = 1u;
    std::vector<float> logits;
    std::vector<shelf::bf16_t> expert_payload;
    std::vector<shelf::bf16_t> shared_payload;
    std::vector<float> gate;
    std::vector<int32_t> slot_row_of;
    std::vector<float> sel_weight;
    std::vector<uint32_t> sel_expert;
};

Case make_case(uint32_t rows, uint32_t top_k, bool shared,
                 uint32_t gate_stride = 1u) {
    Case c;
    c.gate_stride = gate_stride;
    c.rows = rows;
    c.top_k = top_k;
    c.shared = shared;
    c.group_capacity = moe::moe_group_count_upper_bound(rows, kExperts, top_k);

    uint32_t s = 12345u + rows * 7u + top_k;
    auto rnd = [&s]() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xffffu) / 32768.0f - 4.0f;
    };
    c.logits.assign(static_cast<std::size_t>(rows) * kExperts, 0.0f);
    for (auto& v : c.logits)
        v = rnd();

    c.expert_payload.assign(static_cast<std::size_t>(kMaxPlaneRows) * kFeatures,
                            shelf::bf16_t{0});
    for (uint32_t row = 0; row < kMaxPlaneRows; ++row) {
        for (uint32_t f = 0; f < kFeatures; ++f) {
            const float value = rnd();
            c.expert_payload[static_cast<std::size_t>(row) * kFeatures + f] =
                shelf::bf16_t{shelf::f32_to_bf16_rne(value)};
        }
    }
    c.shared_payload.assign(static_cast<std::size_t>(kMaxRows) * kFeatures, shelf::bf16_t{0});
    for (auto& v : c.shared_payload)
        v = shelf::bf16_t{shelf::f32_to_bf16_rne(rnd())};
    // The kernel receives the gate already activated, so the fixture stores sigmoid
    // values directly and the reference uses the same numbers.
    c.gate.assign(static_cast<std::size_t>(kMaxRows) * c.gate_stride, 0.0f);
    for (uint32_t t = 0; t < kMaxRows; ++t)
        c.gate[static_cast<std::size_t>(t) * c.gate_stride] = 1.0f / (1.0f + std::exp(-rnd()));
    return c;
}

// The contract order: expert index ascending, which is group row ascending.
std::vector<shelf::bf16_t> reference_combine(const Case& c, bool expert_ascending) {
    std::vector<shelf::bf16_t> out(static_cast<std::size_t>(c.rows) * kFeatures, shelf::bf16_t{0});
    for (uint32_t t = 0; t < c.rows; ++t) {
        std::vector<std::pair<int32_t, float>> slots;
        for (uint32_t j = 0; j < c.top_k; ++j) {
            const std::size_t at = static_cast<std::size_t>(t) * c.top_k + j;
            slots.push_back({c.slot_row_of[at], c.sel_weight[at]});
        }
        if (expert_ascending)
            std::sort(slots.begin(), slots.end(),
                      [](const std::pair<int32_t, float>& a, const std::pair<int32_t, float>& b) {
                          return a.first < b.first;
                      });
        else
            std::sort(slots.begin(), slots.end(),
                      [](const std::pair<int32_t, float>& a, const std::pair<int32_t, float>& b) {
                          return a.first > b.first;
                      });
        const float gate = c.gate[static_cast<std::size_t>(t) * c.gate_stride];
        for (uint32_t f = 0; f < kFeatures; ++f) {
            float acc = 0.0f;
            for (const auto& slot : slots)
                acc = std::fmaf(slot.second,
                                bf16_to_f32(c.expert_payload[static_cast<std::size_t>(slot.first) *
                                                                 kFeatures + f]),
                                acc);
            if (c.shared)
                acc = std::fmaf(gate,
                                bf16_to_f32(c.shared_payload[static_cast<std::size_t>(t) *
                                                                 kFeatures + f]),
                                acc);
            out[static_cast<std::size_t>(t) * kFeatures + f] =
                shelf::bf16_t{shelf::f32_to_bf16_rne(acc)};
        }
    }
    return out;
}

void run_case(const Case& spec, Device& d, hipStream_t stream) {
    Case c = spec;
    const std::string tag = "rows" + std::to_string(c.rows) + "/top" + std::to_string(c.top_k) +
                            (c.shared ? "/shared" : "");

    h2d(d.logits, c.logits);
    h2d(d.expert_out, c.expert_payload);
    h2d(d.shared_out, c.shared_payload);
    h2d(d.shared_gate, c.gate);
    if (hipMemset(d.tile_counts, 0,
                  kExperts * moe::moe_slot_tile_count(c.rows) * sizeof(uint32_t)) != hipSuccess)
        check(false, tag + ": counts memset");
    if (moe::launch_moe_router(d.logits, kExperts, d.sel_expert, d.sel_weight, d.tile_counts,
                               c.rows, kExperts, c.top_k, nullptr, stream) != hipSuccess)
        check(false, tag + ": router");
    if (moe::launch_moe_slot_scan(d.tile_counts, d.row_begin, d.tile_base, d.group_begin,
                                  d.group_expert, d.group_rows, d.group_count, d.row_token, c.rows,
                                  kExperts, c.top_k, stream) != hipSuccess)
        check(false, tag + ": scan");
    if (moe::launch_moe_slot_scatter(d.sel_expert, d.tile_base, d.row_begin, d.group_begin,
                                     d.row_token, d.slot_row_of, c.rows, kExperts, c.top_k,
                                     stream) != hipSuccess)
        check(false, tag + ": scatter");
    if (hipStreamSynchronize(stream) != hipSuccess)
        check(false, tag + ": plan sync");

    c.slot_row_of = d2h(d.slot_row_of, static_cast<std::size_t>(c.rows) * c.top_k);
    c.sel_weight = d2h(d.sel_weight, static_cast<std::size_t>(c.rows) * c.top_k);
    auto sel = d2h(d.sel_expert, static_cast<std::size_t>(c.rows) * c.top_k);
    c.sel_expert.assign(sel.begin(), sel.end());

    // The plan arrays are sized by grouped_row_tile_count, so the group count the scan
    // produced has to stay within it for every routing the router can emit.
    const uint32_t groups = d2h(d.group_count, 1u)[0];
    if (groups > c.group_capacity)
        std::printf("  (%s: groups %u over capacity %u)\n", tag.c_str(), groups,
                    c.group_capacity);
    check(groups <= c.group_capacity, tag + ": group count stays within the host capacity");

    const hipError_t err = moe::launch_moe_combine(
        d.out, d.expert_out, d.sel_weight, d.slot_row_of, c.shared ? d.shared_out : nullptr,
        c.shared ? d.shared_gate : nullptr, c.rows, c.top_k, kFeatures, kFeatures, kFeatures,
        c.gate_stride, stream);
    check(err == hipSuccess, tag + ": launch_moe_combine");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        check(false, tag + ": combine sync");
        return;
    }
    const std::vector<shelf::bf16_t> got = d2h(d.out, static_cast<std::size_t>(c.rows) * kFeatures);
    const std::vector<shelf::bf16_t> want = reference_combine(c, true);
    bool exact = true;
    for (std::size_t i = 0; i < want.size(); ++i)
        if (got[i].data != want[i].data)
            exact = false;
    check(exact, tag + ": combine is bit identical to the expert-ascending f32 reference");

}

// A crafted plan whose expert-ascending order and the reverse order give different f32
// sums, so the order gate is not vacuous: rows 0 and 1 hold 1.0, row 2 holds 1e8 and
// row 3 holds -1e8. Ascending gives (1 + 1 + 1e8) - 1e8 == 0 in f32 (the 2 is absorbed
// by the 1e8 term), while the reverse order gives 2.
void run_order_discriminating_case(Device& d, hipStream_t stream) {
    constexpr uint32_t rows = 1u;
    constexpr uint32_t top_k = 4u;
    const std::vector<int32_t> slot_row_of = {3, 1, 0, 2};
    const std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<shelf::bf16_t> payload(static_cast<std::size_t>(kMaxPlaneRows) * kFeatures,
                                       shelf::bf16_t{0});
    const float per_row[4] = {1.0f, 1.0f, 1.0e8f, -1.0e8f};
    for (uint32_t row = 0; row < 4u; ++row)
        for (uint32_t f = 0; f < kFeatures; ++f)
            payload[static_cast<std::size_t>(row) * kFeatures + f] =
                shelf::bf16_t{shelf::f32_to_bf16_rne(per_row[row])};

    h2d(d.slot_row_of, slot_row_of);
    h2d(d.sel_weight, weight);
    h2d(d.expert_out, payload);
    const hipError_t err = moe::launch_moe_combine(d.out, d.expert_out, d.sel_weight,
                                                   d.slot_row_of, nullptr, nullptr, rows, top_k,
                                                   kFeatures, kFeatures, kFeatures, 1u, stream);
    check(err == hipSuccess, "order case: launch_moe_combine");
    const std::vector<shelf::bf16_t> got = d2h(d.out, static_cast<std::size_t>(rows) * kFeatures);

    float ascending = 0.0f;
    for (const int32_t row : {0, 1, 2, 3})
        ascending = std::fmaf(1.0f, bf16_to_f32(payload[static_cast<std::size_t>(row) * kFeatures]),
                              ascending);
    float descending = 0.0f;
    for (const int32_t row : {3, 2, 1, 0})
        descending = std::fmaf(1.0f, bf16_to_f32(payload[static_cast<std::size_t>(row) * kFeatures]),
                               descending);
    std::printf("order case: ascending %.4g, descending %.4g, device %.4g\n", ascending,
                descending, bf16_to_f32(got[0]));
    check(ascending != descending, "order case: the two orders are numerically different");
    check(bf16_to_f32(got[0]) == ascending,
          "order case: the kernel accumulates expert ascending");
    for (uint32_t f = 1; f < kFeatures; ++f)
        if (got[f].data != got[0].data)
            check(false, "order case: every feature column agrees");
}

// features is not a multiple of the block width: the last block of a row is partial, and the
// rows of the feature domain beyond `features` must stay untouched.
void run_partial_feature_block_case(Device& d, hipStream_t stream) {
    constexpr uint32_t rows = 2u;
    constexpr uint32_t top_k = 2u;
    constexpr uint32_t feats = 200u;
    constexpr uint32_t stride = 256u;
    constexpr uint16_t kSentinel = 0x0001u;

    const std::vector<int32_t> slot_row_of = {5, 2, 7, 3};
    const std::vector<float> weight = {2.0f, 0.5f, 1.0f, 4.0f};
    auto payload_value = [](uint32_t row, uint32_t f) {
        return static_cast<uint16_t>(100u + ((row * 397u + f * 7u) % 500u));
    };

    std::vector<shelf::bf16_t> payload(static_cast<std::size_t>(kMaxPlaneRows) * stride);
    for (uint32_t row = 0; row < kMaxPlaneRows; ++row)
        for (uint32_t f = 0; f < stride; ++f)
            payload[static_cast<std::size_t>(row) * stride + f] =
                shelf::bf16_t{payload_value(row, f)};

    std::vector<shelf::bf16_t> poison(static_cast<std::size_t>(kMaxRows) * kFeatures,
                                      shelf::bf16_t{kSentinel});
    h2d(d.slot_row_of, slot_row_of);
    h2d(d.sel_weight, weight);
    h2d(d.expert_out, payload);
    h2d(d.out, poison);

    const hipError_t err = moe::launch_moe_combine(d.out, d.expert_out, d.sel_weight,
                                                   d.slot_row_of, nullptr, nullptr, rows, top_k,
                                                   feats, stride, stride, 1u, stream);
    check(err == hipSuccess, "partial feature case: launch_moe_combine");
    const std::vector<shelf::bf16_t> got =
        d2h(d.out, static_cast<std::size_t>(kMaxRows) * kFeatures);

    // The kernel walks the slots in ascending group row.
    const uint32_t order[2][2] = {{2u, 5u}, {3u, 7u}};
    const float order_weight[2][2] = {{0.5f, 2.0f}, {4.0f, 1.0f}};
    uint32_t mismatches = 0;
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t f = 0; f < feats; ++f) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < top_k; ++j)
                acc = std::fmaf(order_weight[t][j],
                                bf16_to_f32(payload[static_cast<std::size_t>(order[t][j]) * stride +
                                                    f]),
                                acc);
            const shelf::bf16_t want{shelf::f32_to_bf16_rne(acc)};
            if (got[static_cast<std::size_t>(t) * feats + f].data != want.data)
                ++mismatches;
        }
    }
    check(mismatches == 0, "partial feature case: every feature column matches the reference");

    uint32_t touched = 0;
    for (std::size_t i = static_cast<std::size_t>(rows) * feats; i < poison.size(); ++i)
        if (got[i].data != kSentinel)
            ++touched;
    check(touched == 0, "partial feature case: nothing outside the feature domain is written");
}

}  // namespace

int main() {
    std::printf("=== MoE combine (GPU) ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess)
        return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess)
        return 1;

    Device d;
    d.logits = alloc<float>(static_cast<std::size_t>(kMaxRows) * kExperts);
    d.sel_expert = alloc<int32_t>(static_cast<std::size_t>(kMaxRows) * kMaxTopK);
    d.sel_weight = alloc<float>(static_cast<std::size_t>(kMaxRows) * kMaxTopK);
    d.tile_counts = alloc<uint32_t>(
        static_cast<std::size_t>(kExperts) * moe::moe_slot_tile_count(kMaxRows));
    d.row_begin = alloc<uint32_t>(kExperts + 1u);
    d.tile_base = alloc<uint32_t>(
        static_cast<std::size_t>(kExperts) * moe::moe_slot_tile_count(kMaxRows));
    d.group_begin = alloc<uint32_t>(kExperts + 1u);
    d.group_expert = alloc<uint32_t>(kMaxGroupCapacity);
    d.group_rows = alloc<uint32_t>(kMaxGroupCapacity);
    d.group_count = alloc<uint32_t>(1);
    d.row_token = alloc<int32_t>(kMaxPlaneRows);
    d.slot_row_of = alloc<int32_t>(static_cast<std::size_t>(kMaxRows) * kMaxTopK);
    d.expert_out = alloc<shelf::bf16_t>(static_cast<std::size_t>(kMaxPlaneRows) * kFeatures);
    d.shared_out = alloc<shelf::bf16_t>(static_cast<std::size_t>(kMaxRows) * kFeatures);
    d.shared_gate = alloc<float>(static_cast<std::size_t>(kMaxRows) * 8u);
    d.out = alloc<shelf::bf16_t>(static_cast<std::size_t>(kMaxRows) * kFeatures);

    run_case(make_case(1, 8, false), d, stream);
    run_case(make_case(8, 2, false), d, stream);
    run_case(make_case(8, 8, true), d, stream);
    run_case(make_case(64, 8, true), d, stream);
    run_case(make_case(64, 4, false), d, stream);
    // A padded gate plane: every token after the first would read another row's bytes (or
    // padding) if the gate were indexed packed, which silently drops the shared expert.
    run_case(make_case(8, 8, true, 4u), d, stream);
    run_order_discriminating_case(d, stream);
    run_partial_feature_block_case(d, stream);

    check(moe::launch_moe_combine(d.out, d.expert_out, d.sel_weight, d.slot_row_of, d.shared_out,
                                  nullptr, 1, 2, kFeatures, kFeatures, kFeatures, 1u,
                                  stream) != hipSuccess,
          "shared output without a gate is rejected");
    check(moe::launch_moe_combine(d.out, d.expert_out, d.sel_weight, d.slot_row_of, nullptr,
                                  d.shared_gate, 1, 2, kFeatures, kFeatures, kFeatures, 1u,
                                  stream) != hipSuccess,
          "shared gate without output is rejected");
    check(moe::launch_moe_combine(d.out, d.expert_out, d.sel_weight, d.slot_row_of, nullptr,
                                  nullptr, 1, 0, kFeatures, kFeatures, kFeatures, 1u,
                                  stream) != hipSuccess,
          "zero top_k is rejected");
    check(moe::launch_moe_combine(d.out, d.expert_out, d.sel_weight, d.slot_row_of, nullptr,
                                  nullptr, 1, moe::kMoeRouterMaxTopK + 1u, kFeatures, kFeatures,
                                  kFeatures, 1u, stream) != hipSuccess,
          "top_k above the kernel bound is rejected");
    check(moe::launch_moe_combine(d.out, d.expert_out, d.sel_weight, d.slot_row_of, nullptr,
                                  nullptr, 1, 2, kFeatures, kFeatures - 1u, kFeatures, 1u,
                                  stream) != hipSuccess,
          "an expert row stride below features is rejected");
    check(moe::launch_moe_combine(d.out, d.expert_out, d.sel_weight, d.slot_row_of, d.shared_out,
                                  d.shared_gate, 1, 2, kFeatures, kFeatures, kFeatures, 0u,
                                  stream) != hipSuccess,
          "a shared gate plane with no row stride is rejected");

    hipFree(d.logits);
    hipFree(d.sel_expert);
    hipFree(d.sel_weight);
    hipFree(d.tile_counts);
    hipFree(d.row_begin);
    hipFree(d.tile_base);
    hipFree(d.group_begin);
    hipFree(d.group_expert);
    hipFree(d.group_rows);
    hipFree(d.group_count);
    hipFree(d.row_token);
    hipFree(d.slot_row_of);
    hipFree(d.expert_out);
    hipFree(d.shared_out);
    hipFree(d.shared_gate);
    hipFree(d.out);
    if (hipStreamDestroy(stream) != hipSuccess)
        check(false, "stream destroy");

    std::printf("test_moe_combine: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
