#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/kernels/optimized/moe/router.h>
#include <shelf/runtime/program/device_program.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using shelf::kernel::kMoeRouterMaxExperts;
using shelf::kernel::kMoeSlotNotOwned;
using shelf::kernel::kMxfp4GemmRowTile;
using shelf::kernel::launch_moe_router;
using shelf::kernel::launch_moe_slot_scan;
using shelf::kernel::launch_moe_slot_scatter;
using shelf::kernel::moe_group_count_upper_bound;
using shelf::kernel::moe_group_row_capacity;
using shelf::kernel::moe_slot_tile_count;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

template <typename T>
T* dalloc(std::size_t count) {
    T* p = nullptr;
    if (hipMalloc(&p, count * sizeof(T)) != hipSuccess) {
        std::printf("FAIL hipMalloc %zu\n", count);
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

struct Buffers {
    float* logits = nullptr;
    uint32_t* error_word = nullptr;
    int32_t* sel_expert = nullptr;
    float* sel_weight = nullptr;
    uint32_t* tile_counts = nullptr;
    uint32_t* expert_row_begin = nullptr;
    uint32_t* tile_base = nullptr;
    uint32_t* expert_group_begin = nullptr;
    uint32_t* group_expert = nullptr;
    uint32_t* group_rows = nullptr;
    uint32_t* group_count = nullptr;
    int32_t* row_token = nullptr;
    int32_t* slot_row_of = nullptr;
};

struct Reference {
    std::vector<int32_t> expert;
    std::vector<double> weight;
};

// The decision order contract, written directly from its definition.
Reference reference_route(const std::vector<float>& logits, uint32_t rows, uint32_t experts,
                          uint32_t top_k) {
    Reference r;
    r.expert.assign(static_cast<std::size_t>(rows) * top_k, -1);
    r.weight.assign(static_cast<std::size_t>(rows) * top_k, 0.0);
    for (uint32_t t = 0; t < rows; ++t) {
        const float* row = logits.data() + static_cast<std::size_t>(t) * experts;
        double max = -1e300;
        for (uint32_t e = 0; e < experts; ++e)
            max = std::max(max, static_cast<double>(row[e]));
        std::vector<double> p(experts);
        double sum = 0.0;
        for (uint32_t e = 0; e < experts; ++e) {
            p[e] = std::exp(static_cast<double>(row[e]) - max);
            sum += p[e];
        }
        for (uint32_t e = 0; e < experts; ++e)
            p[e] /= sum;
        std::vector<char> taken(experts, 0);
        std::vector<uint32_t> picked;
        double sel_sum = 0.0;
        for (uint32_t j = 0; j < top_k; ++j) {
            uint32_t best = experts;
            for (uint32_t e = 0; e < experts; ++e) {
                if (taken[e])
                    continue;
                if (best == experts || p[e] > p[best])
                    best = e;
            }
            taken[best] = 1;
            picked.push_back(best);
            sel_sum += p[best];
        }
        for (uint32_t j = 0; j < top_k; ++j) {
            r.expert[static_cast<std::size_t>(t) * top_k + j] = static_cast<int32_t>(picked[j]);
            r.weight[static_cast<std::size_t>(t) * top_k + j] = p[picked[j]] / sel_sum;
        }
    }
    return r;
}

std::vector<float> make_logits(uint32_t rows, uint32_t experts, uint32_t top_k, uint32_t seed,
                               bool with_ties) {
    std::vector<float> out(static_cast<std::size_t>(rows) * experts);
    uint32_t s = seed;
    auto rnd = [&s]() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xffffu) / 4096.0f - 8.0f;
    };
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t e = 0; e < experts; ++e)
            out[static_cast<std::size_t>(t) * experts + e] = rnd();
        if (with_ties && experts >= 20u) {
            const float peak = 9.0f;
            out[static_cast<std::size_t>(t) * experts + 9u] = peak;
            out[static_cast<std::size_t>(t) * experts + 5u] = peak;
            if (top_k > 2u)
                out[static_cast<std::size_t>(t) * experts + 17u] = peak;
        }
    }
    return out;
}

struct Plan {
    std::vector<int32_t> sel_expert;
    std::vector<float> sel_weight;
    std::vector<uint32_t> expert_row_begin;
    std::vector<uint32_t> expert_group_begin;
    std::vector<uint32_t> group_expert;
    std::vector<uint32_t> group_rows;
    std::vector<int32_t> row_token;
    std::vector<int32_t> slot_row_of;
    uint32_t group_count = 0;
    uint32_t status_word = 0;
};

Plan run_plan(const std::vector<float>& logits, uint32_t rows, uint32_t experts, uint32_t top_k,
              const Buffers& b, hipStream_t stream, const std::string& tag,
              bool zero_counts = true, uint32_t expert_begin = 0, uint32_t owned_experts = 0) {
    // The logits are always as wide as the model's experts. A split plans and materializes
    // only the owned range, so the plan arrays are sized by owned_experts.
    const uint32_t owned = owned_experts != 0u ? owned_experts : experts;
    const uint32_t tiles = moe_slot_tile_count(rows);
    const uint32_t slots = rows * top_k;
    const uint32_t group_bound = moe_group_count_upper_bound(rows, owned, top_k);
    const uint32_t row_capacity = moe_group_row_capacity(rows, owned, top_k);

    h2d(b.logits, logits);
    if (hipMemset(b.error_word, 0, sizeof(uint32_t)) != hipSuccess) {
        std::printf("FAIL memset error word %s\n", tag.c_str());
        std::exit(1);
    }
    if (zero_counts &&
        hipMemset(b.tile_counts, 0, static_cast<std::size_t>(owned) * tiles * sizeof(uint32_t)) !=
            hipSuccess) {
        std::printf("FAIL memset %s\n", tag.c_str());
        std::exit(1);
    }
    if (launch_moe_router(b.logits, experts, b.sel_expert, b.sel_weight, b.tile_counts, rows,
                          experts, top_k, expert_begin, owned, b.error_word,
                          stream) != hipSuccess)
        check(false, tag + ": launch_moe_router");
    if (launch_moe_slot_scan(b.tile_counts, b.expert_row_begin, b.tile_base, b.expert_group_begin,
                             b.group_expert, b.group_rows, b.group_count, b.row_token, rows, owned,
                             top_k, stream) != hipSuccess)
        check(false, tag + ": launch_moe_slot_scan");
    if (launch_moe_slot_scatter(b.sel_expert, b.tile_base, b.expert_row_begin,
                                b.expert_group_begin, b.row_token, b.slot_row_of, rows, owned,
                                top_k, stream) != hipSuccess)
        check(false, tag + ": launch_moe_slot_scatter");
    if (hipStreamSynchronize(stream) != hipSuccess)
        check(false, tag + ": sync");

    Plan p;
    p.sel_expert = d2h(b.sel_expert, slots);
    p.sel_weight = d2h(b.sel_weight, slots);
    p.expert_row_begin = d2h(b.expert_row_begin, owned + 1u);
    p.expert_group_begin = d2h(b.expert_group_begin, owned + 1u);
    p.group_expert = d2h(b.group_expert, group_bound);
    p.group_rows = d2h(b.group_rows, group_bound);
    p.row_token = d2h(b.row_token, row_capacity);
    p.slot_row_of = d2h(b.slot_row_of, slots);
    p.group_count = d2h(b.group_count, 1).at(0);
    p.status_word = d2h(b.error_word, 1).at(0);
    return p;
}

void compare_selection(const Plan& p, const Reference& ref, uint32_t rows, uint32_t experts,
                       uint32_t top_k, const std::string& tag) {
    bool index_ok = true;
    bool weight_ok = true;
    double worst = 0.0;
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t j = 0; j < top_k; ++j) {
            const std::size_t at = static_cast<std::size_t>(t) * top_k + j;
            if (p.sel_expert[at] != ref.expert[at])
                index_ok = false;
            const double want = ref.weight[at];
            const double got = p.sel_weight[at];
            const double err = std::fabs(want - got) / (want > 1e-12 ? want : 1e-12);
            if (err > worst)
                worst = err;
            if (err > 1.0e-5)
                weight_ok = false;
        }
    }
    std::printf("%s: worst weight rel err %.3e\n", tag.c_str(), worst);
    check(index_ok, tag + ": selected expert indices match the contract");
    check(weight_ok, tag + ": renormalized weights match within 1e-5 (worst " +
                         std::to_string(worst) + ")");

    bool distinct = true;
    bool descending = true;
    bool sums_one = true;
    for (uint32_t t = 0; t < rows; ++t) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < top_k; ++j) {
            const std::size_t at = static_cast<std::size_t>(t) * top_k + j;
            const int32_t e = p.sel_expert[at];
            if (e < 0 || static_cast<uint32_t>(e) >= experts)
                distinct = false;
            for (uint32_t k = 0; k < j; ++k)
                if (p.sel_expert[static_cast<std::size_t>(t) * top_k + k] == e)
                    distinct = false;
            if (j > 0u && p.sel_weight[at] > p.sel_weight[at - 1u])
                descending = false;
            sum += p.sel_weight[at];
        }
        if (std::fabs(sum - 1.0f) > 1.0e-5f)
            sums_one = false;
    }
    check(distinct, tag + ": each token selects top_k distinct experts in range");
    check(descending, tag + ": selections are probability descending");
    check(sums_one, tag + ": selected weights renormalize to 1");
}

bool selects(const Plan& p, uint32_t token, uint32_t expert, uint32_t top_k) {
    for (uint32_t j = 0; j < top_k; ++j)
        if (p.sel_expert[static_cast<std::size_t>(token) * top_k + j] ==
            static_cast<int32_t>(expert))
            return true;
    return false;
}

void compare_plan(const Plan& p, uint32_t rows, uint32_t experts, uint32_t top_k,
                  const std::string& tag, uint32_t slot_total = 0) {
    // A rank of an expert-axis split plans only its owned selections, so the slot total it has
    // to partition is smaller than rows * top_k.
    const uint32_t slots = slot_total != 0u ? slot_total : rows * top_k;
    check(p.expert_row_begin[0] == 0u && p.expert_row_begin[experts] == slots,
          tag + ": expert slot ranges partition every slot");
    check(p.expert_group_begin[0] == 0u, tag + ": group ranges start at 0");
    bool monotone = true;
    for (uint32_t e = 0; e < experts; ++e) {
        if (p.expert_row_begin[e] > p.expert_row_begin[e + 1u])
            monotone = false;
        if (p.expert_group_begin[e] > p.expert_group_begin[e + 1u])
            monotone = false;
    }
    check(monotone, tag + ": expert ranges are ordered and disjoint");
    check(p.group_count == p.expert_group_begin[experts],
          tag + ": group total matches the expert group prefix");
    check(p.group_count <= moe_group_count_upper_bound(rows, experts, top_k),
          tag + ": group count stays inside the host capacity bound");

    bool tile_shape = true;
    bool expert_of_group = true;
    bool covers = true;
    for (uint32_t e = 0; e < experts; ++e) {
        const uint32_t slots_e = p.expert_row_begin[e + 1u] - p.expert_row_begin[e];
        const uint32_t groups_e = p.expert_group_begin[e + 1u] - p.expert_group_begin[e];
        if (groups_e != (slots_e + kMxfp4GemmRowTile - 1u) / kMxfp4GemmRowTile)
            covers = false;
        uint32_t seen = 0u;
        for (uint32_t g = p.expert_group_begin[e]; g < p.expert_group_begin[e + 1u]; ++g) {
            if (p.group_expert[g] != e)
                expert_of_group = false;
            const uint32_t cnt = p.group_rows[g];
            if (cnt == 0u || cnt > kMxfp4GemmRowTile)
                tile_shape = false;
            if (g + 1u < p.expert_group_begin[e + 1u] && cnt != kMxfp4GemmRowTile)
                tile_shape = false;
            seen += cnt;
        }
        if (seen != slots_e)
            covers = false;
    }
    check(expert_of_group, tag + ": every group names its expert");
    check(tile_shape, tag + ": groups are 16 slots except the last group of an expert");
    check(covers, tag + ": groups cover an expert's slots exactly");

    bool rows_ok = true;
    bool token_ascending = true;
    for (uint32_t g = 0; g < p.group_count; ++g) {
        const uint32_t e = p.group_expert[g];
        const uint32_t cnt = p.group_rows[g];
        int32_t previous = -1;
        for (uint32_t i = 0; i < cnt; ++i) {
            const int32_t token = p.row_token[g * kMxfp4GemmRowTile + i];
            if (token < 0 || static_cast<uint32_t>(token) >= rows)
                rows_ok = false;
            else if (!selects(p, static_cast<uint32_t>(token), e, top_k))
                rows_ok = false;
            if (token <= previous)
                token_ascending = false;
            previous = token;
        }
        for (uint32_t i = cnt; i < kMxfp4GemmRowTile; ++i) {
            if (p.row_token[g * kMxfp4GemmRowTile + i] != -1)
                rows_ok = false;
        }
    }
    bool beyond_capacity_rows = true;
    for (uint32_t r = p.group_count * kMxfp4GemmRowTile; r < p.row_token.size(); ++r)
        if (p.row_token[r] != -1)
            beyond_capacity_rows = false;
    check(rows_ok, tag + ": every group row is a token that selected that expert, padding is -1");
    check(beyond_capacity_rows, tag + ": rows past the active groups are -1 as well");
    check(token_ascending, tag + ": rows inside one expert are token ascending");

    bool row_map_ok = true;
    bool expert_ascending_for_token = true;
    std::vector<char> used(p.row_token.size(), 0);
    for (uint32_t t = 0; t < rows; ++t) {
        std::vector<std::pair<int32_t, int32_t>> by_row;
        for (uint32_t j = 0; j < top_k; ++j) {
            const std::size_t at = static_cast<std::size_t>(t) * top_k + j;
            const int32_t r = p.slot_row_of[at];
            const int32_t e = p.sel_expert[at];
            // A rank of an expert-axis split leaves the selections it does not own marked in
            // both arrays; they claim no group row and are not part of the ordering.
            if (e == kMoeSlotNotOwned) {
                if (r != kMoeSlotNotOwned) row_map_ok = false;
                continue;
            }
            if (r < 0 || static_cast<uint32_t>(r) >= p.row_token.size() || used[r] ||
                p.row_token[r] != static_cast<int32_t>(t))
                row_map_ok = false;
            else
                used[r] = 1;
            by_row.push_back({r, e});
        }
        std::sort(by_row.begin(), by_row.end());
        for (std::size_t i = 1; i < by_row.size(); ++i)
            if (by_row[i].second <= by_row[i - 1].second)
                expert_ascending_for_token = false;
    }
    check(row_map_ok, tag + ": slot_row_of is a bijection onto the claimed group rows");
    check(expert_ascending_for_token,
          tag + ": walking a token's group rows in order visits experts ascending");

    uint32_t claimed = 0u;
    for (uint32_t g = 0; g < p.group_count; ++g)
        claimed += p.group_rows[g];
    uint32_t mapped = 0u;
    for (char u : used)
        mapped += u ? 1u : 0u;
    check(claimed == slots && mapped == slots, tag + ": slot count, group rows and map agree");
}

void run_case(uint32_t rows, uint32_t experts, uint32_t top_k, bool with_ties,
              const Buffers& b, hipStream_t stream) {
    const std::string tag = "E" + std::to_string(experts) + "/top" + std::to_string(top_k) +
                            "/rows" + std::to_string(rows) + (with_ties ? "/ties" : "");
    const std::vector<float> logits = make_logits(rows, experts, top_k, 7u + rows, with_ties);
    const Reference ref = reference_route(logits, rows, experts, top_k);

    const Plan p = run_plan(logits, rows, experts, top_k, b, stream, tag);
    check(p.status_word == static_cast<uint32_t>(::shelf::runtime::ProgramStatus::COMPLETE),
          tag + ": finite logits leave the error word alone");
    compare_selection(p, ref, rows, experts, top_k, tag);
    compare_plan(p, rows, experts, top_k, tag);

    if (with_ties && experts >= 20u) {
        bool tie_small_index = true;
        for (uint32_t t = 0; t < rows; ++t) {
            if (p.sel_expert[static_cast<std::size_t>(t) * top_k] != 5)
                tie_small_index = false;
            if (p.sel_expert[static_cast<std::size_t>(t) * top_k + 1u] != 9)
                tie_small_index = false;
            if (top_k > 2u && p.sel_expert[static_cast<std::size_t>(t) * top_k + 2u] != 17)
                tie_small_index = false;
        }
        check(tie_small_index, tag + ": exact ties pick the smaller expert index first");
    }

    const Plan again = run_plan(logits, rows, experts, top_k, b, stream, tag + "/repeat");
    check(again.sel_expert == p.sel_expert && again.sel_weight == p.sel_weight &&
              again.expert_row_begin == p.expert_row_begin &&
              again.expert_group_begin == p.expert_group_begin &&
              again.group_expert == p.group_expert && again.group_rows == p.group_rows &&
              again.row_token == p.row_token && again.slot_row_of == p.slot_row_of &&
              again.group_count == p.group_count,
          tag + ": the plan is bit identical across runs");

    // A model with more than one MoE layer shares one plan scratch inside a single program
    // run, so the counters a routing round consumes have to be left ready for the next
    // round. Zeroing them from the host per layer would cost one launch per layer.
    const Plan chained =
        run_plan(logits, rows, experts, top_k, b, stream, tag + "/chained", false);
    check(chained.group_count == p.group_count &&
              chained.expert_row_begin == p.expert_row_begin &&
              chained.expert_group_begin == p.expert_group_begin &&
              chained.group_expert == p.group_expert && chained.group_rows == p.group_rows &&
              chained.row_token == p.row_token && chained.slot_row_of == p.slot_row_of,
          tag + ": a second round without an external zeroing gives the same plan");
    std::printf("%s: groups %u / slot rows %u\n", tag.c_str(), p.group_count,
                p.group_count * kMxfp4GemmRowTile);
}

void run_expert_axis_split_case(const Buffers& b, hipStream_t stream) {
    // The same logits routed by two ranks that own half the experts each. The selection and
    // the renormalized weights have to be identical to the unsplit round on every slot; only
    // the plan arrays and the group rows are restricted to the owned range, and a slot the
    // rank does not own carries the not-owned marker in both sel_expert and slot_row_of.
    constexpr uint32_t rows = 64u;
    constexpr uint32_t experts = 256u;
    constexpr uint32_t top_k = 8u;
    constexpr uint32_t half = experts / 2u;
    const std::vector<float> logits = make_logits(rows, experts, top_k, 20261105u, false);

    const Plan whole = run_plan(logits, rows, experts, top_k, b, stream, "split whole");
    const Plan r0 = run_plan(logits, rows, experts, top_k, b, stream, "split rank 0", true, 0u,
                             half);
    const Plan r1 = run_plan(logits, rows, experts, top_k, b, stream, "split rank 1", true, half,
                             half);
    const Reference ref = reference_route(logits, rows, experts, top_k);

    bool weight_ok = true;
    bool index_ok = true;
    bool marker_ok = true;
    bool partition_ok = true;
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t j = 0; j < top_k; ++j) {
            const std::size_t at = static_cast<std::size_t>(t) * top_k + j;
            if (whole.sel_expert[at] != ref.expert[at]) index_ok = false;
            const int32_t e = whole.sel_expert[at];
            const bool low = static_cast<uint32_t>(e) < half;
            if (r0.sel_weight[at] != whole.sel_weight[at] ||
                r1.sel_weight[at] != whole.sel_weight[at])
                weight_ok = false;
            if (r0.sel_expert[at] != (low ? e : kMoeSlotNotOwned) ||
                r1.sel_expert[at] != (low ? kMoeSlotNotOwned : e - static_cast<int32_t>(half)))
                marker_ok = false;
            if ((r0.slot_row_of[at] < 0) == (r1.slot_row_of[at] < 0)) partition_ok = false;
            if (r0.slot_row_of[at] < 0 && r1.slot_row_of[at] >= 0) {
                if (whole.sel_expert[at] < static_cast<int32_t>(half)) partition_ok = false;
            }
            if (r1.slot_row_of[at] < 0 && r0.slot_row_of[at] >= 0) {
                if (whole.sel_expert[at] >= static_cast<int32_t>(half)) partition_ok = false;
            }
        }
    }
    check(weight_ok, "split: every rank carries the whole selection's renormalized weight");
    check(marker_ok, "split: sel_expert is the local plane index or kMoeSlotNotOwned");
    check(partition_ok, "split: exactly one rank materializes each selection slot");

    // Each rank's plan is a plan over its own expert count, and the two ranks together cover
    // every slot the unsplit round planned.
    const uint32_t whole_slots = rows * top_k;
    uint32_t owned0 = 0;
    uint32_t owned1 = 0;
    for (uint32_t at = 0; at < whole_slots; ++at) {
        if (r0.slot_row_of[at] >= 0) ++owned0;
        if (r1.slot_row_of[at] >= 0) ++owned1;
    }
    compare_plan(r0, rows, half, top_k, "split rank 0 plan", owned0);
    compare_plan(r1, rows, half, top_k, "split rank 1 plan", owned1);
    check(owned0 + owned1 == whole_slots, "split: the two ranks cover every slot");
    check(owned0 == r0.expert_row_begin[half],
          "split: rank 0's plan counts exactly its owned slots");
    check(owned1 == r1.expert_row_begin[half],
          "split: rank 1's plan counts exactly its owned slots");
    bool local_experts_ok = true;
    for (uint32_t g = 0; g < r1.group_count; ++g)
        if (r1.group_expert[g] >= half) local_experts_ok = false;
    check(local_experts_ok, "split: group_expert indexes the local plane, not the global expert");
}

void run_unsafe_logits_case(const Buffers& b, hipStream_t stream) {
    // A row whose logits are not finite must still give a usable plan: an out-of-range expert
    // index makes the scatter walk past its arrays and fault the context.
    constexpr uint32_t rows = 4u;
    constexpr uint32_t experts = 8u;
    constexpr uint32_t top_k = 2u;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> logits = {
        nan, 5.0f, 1.0f, inf, -inf, 2.0f, nan, 0.5f,   // only the finite entries compete
        -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, // nothing finite at all
        nan, nan, nan, nan, nan, nan, nan, nan,        // nothing finite at all
        inf, inf, 1.0f, 0.0f, -1.0f, nan, inf, -inf,   // +inf is not a winner either
    };
    const std::string tag = "non-finite logits";

    const Plan p = run_plan(logits, rows, experts, top_k, b, stream, tag);

    check(p.status_word == static_cast<uint32_t>(::shelf::runtime::ProgramStatus::NUMERIC_ERROR),
          tag + ": the round is reported as a numeric error so the run stops");
    bool in_range = true;
    for (const int32_t e : p.sel_expert)
        if (e < 0 || static_cast<uint32_t>(e) >= experts) in_range = false;
    check(in_range, tag + ": every selected expert is in range");

    const std::vector<int32_t> want = {1, 5, 0, 1, 0, 1, 2, 3};
    check(p.sel_expert == std::vector<int32_t>(want),
          tag + ": non-finite entries lose to finite ones, an all-non-finite row takes the "
                "lowest indices");

    bool weights_finite = true;
    for (const float w : p.sel_weight)
        if (!std::isfinite(w)) weights_finite = false;
    check(weights_finite, tag + ": every routing weight is finite");
    check(p.sel_weight[0] > 0.0f && p.sel_weight[0] + p.sel_weight[1] > 0.999f &&
              p.sel_weight[0] + p.sel_weight[1] < 1.001f,
          tag + ": the finite row renormalizes to 1 per token");
    check(p.sel_weight[4] == 0.0f && p.sel_weight[5] == 0.0f,
          tag + ": an all-non-finite row routes with zero weight");
    check(p.expert_row_begin[experts] == rows * top_k,
          tag + ": every slot is counted into the plan");
    compare_plan(p, rows, experts, top_k, tag);
}

}  // namespace

int main() {
    std::printf("=== MoE router / group plan (GPU) ===\n");
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

    constexpr uint32_t kMaxRows = 512u;
    constexpr uint32_t kMaxExperts = 256u;
    constexpr uint32_t kMaxTopK = 8u;
    const uint32_t max_tiles = moe_slot_tile_count(kMaxRows);
    const uint32_t max_groups = moe_group_count_upper_bound(kMaxRows, kMaxExperts, kMaxTopK);

    Buffers b;
    b.logits = dalloc<float>(static_cast<std::size_t>(kMaxRows) * kMaxExperts);
    b.error_word = dalloc<uint32_t>(1);
    b.sel_expert = dalloc<int32_t>(static_cast<std::size_t>(kMaxRows) * kMaxTopK);
    b.sel_weight = dalloc<float>(static_cast<std::size_t>(kMaxRows) * kMaxTopK);
    b.tile_counts = dalloc<uint32_t>(static_cast<std::size_t>(kMaxExperts) * max_tiles);
    b.expert_row_begin = dalloc<uint32_t>(kMaxExperts + 1u);
    b.tile_base = dalloc<uint32_t>(static_cast<std::size_t>(kMaxExperts) * max_tiles);
    b.expert_group_begin = dalloc<uint32_t>(kMaxExperts + 1u);
    b.group_expert = dalloc<uint32_t>(max_groups);
    b.group_rows = dalloc<uint32_t>(max_groups);
    b.group_count = dalloc<uint32_t>(1);
    b.row_token = dalloc<int32_t>(static_cast<std::size_t>(max_groups) * kMxfp4GemmRowTile);
    b.slot_row_of = dalloc<int32_t>(static_cast<std::size_t>(kMaxRows) * kMaxTopK);

    run_case(1, 8, 2, false, b, stream);
    run_case(1, 256, 8, true, b, stream);
    run_case(7, 256, 8, false, b, stream);
    run_case(64, 256, 8, true, b, stream);
    run_case(65, 256, 8, false, b, stream);
    run_case(512, 256, 8, false, b, stream);

    check(launch_moe_router(nullptr, 0, nullptr, nullptr, nullptr, 1, 8, 2, 0u, 8u, nullptr,
                            stream) != hipSuccess,
          "null logits rejected");
    check(launch_moe_router(b.logits, 8, b.sel_expert, b.sel_weight, b.tile_counts, 1, 8, 9, 0u, 8u,
                            b.error_word, stream) != hipSuccess,
          "top_k greater than experts rejected");
    check(launch_moe_router(b.logits, 8, b.sel_expert, b.sel_weight, b.tile_counts, 1,
                            kMoeRouterMaxExperts + 1u, 2, 0u, kMoeRouterMaxExperts + 1u, b.error_word,
                            stream) != hipSuccess,
          "experts above the kernel bound rejected");
    check(launch_moe_slot_scatter(b.sel_expert, b.tile_base, b.expert_row_begin,
                                  b.expert_group_begin, b.row_token, b.slot_row_of, 0, 8, 2,
                                  stream) != hipSuccess,
          "zero rows rejected");
    check(launch_moe_router(b.logits, 8, b.sel_expert, b.sel_weight, b.tile_counts, 1, 8, 2, 0u, 0u,
                            b.error_word, stream) != hipSuccess,
          "an empty owned expert range is rejected");
    check(launch_moe_router(b.logits, 8, b.sel_expert, b.sel_weight, b.tile_counts, 1, 8, 2, 0u, 9u,
                            b.error_word, stream) != hipSuccess,
          "owning more experts than the logits hold is rejected");
    check(launch_moe_router(b.logits, 8, b.sel_expert, b.sel_weight, b.tile_counts, 1, 8, 2, 5u, 4u,
                            b.error_word, stream) != hipSuccess,
          "an owned expert range that runs past the logits is rejected");

    run_unsafe_logits_case(b, stream);
    run_expert_axis_split_case(b, stream);

    hipFree(b.logits);
    hipFree(b.error_word);
    hipFree(b.sel_expert);
    hipFree(b.sel_weight);
    hipFree(b.tile_counts);
    hipFree(b.expert_row_begin);
    hipFree(b.tile_base);
    hipFree(b.expert_group_begin);
    hipFree(b.group_expert);
    hipFree(b.group_rows);
    hipFree(b.group_count);
    hipFree(b.row_token);
    hipFree(b.slot_row_of);
    hipStreamDestroy(stream);

    std::printf("test_moe_router: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
