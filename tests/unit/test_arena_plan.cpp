// shelf unit test — declare 相の offsets（D4）。GPU 不要。
//
// 見るもの:
//   ① 全 primitive 出力に置き場が付き、寿命の重なる値どうしは重ならない
//   ② 寿命の重ならない値は同じ offset を再利用している（peak が下がる）
//   ③ 現行の自動算出（64 MiB + 2×Σ出力(f32)）を超えない
//   ④ reshape（view）は元の値と同じ offset を指す
//   ⑤ 同じグラフからは同じ表が出る（決定論的）
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

#include <shelf/compiler/arena_offsets.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}
bool overlaps(std::size_t a, std::size_t ab, std::size_t b, std::size_t bb) {
    return a < b + bb && b < a + ab;
}
}  // namespace

int main() {
    using namespace shelf;
    model::qwen35::TextConfig cfg;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.vocab_size = 128;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 64;
    cfg.rotary_dim = 16;
    cfg.rope_theta = 1e7;
    cfg.lin_key_heads = 1;
    cfg.lin_value_heads = 2;
    cfg.lin_key_head_dim = 32;
    cfg.lin_value_head_dim = 32;
    cfg.conv_kernel = 4;
    cfg.intermediate_size = 128;
    cfg.rms_eps = 1e-6;
    cfg.tie_word_embeddings = false;
    cfg.eos_token_id = 0;
    cfg.layer_kinds = {model::qwen35::LayerKind::full_attention,
                       model::qwen35::LayerKind::linear_attention};

    auto g = model::qwen35::build_step_graph(cfg, 16, 1);
    if (!g.ok()) {
        std::printf("FAIL: グラフを作れない\n");
        return 1;
    }
    const auto& graph = g.value().graph;

    const auto plan = compiler::plan_arena(graph);
    check(plan.slots.size() == graph.primitives().size(), "全出力に置き場が付く");
    check(plan.peak_bytes >= plan.values_bytes, "peak が値の領域以上");
    check(plan.scratch_base == (plan.values_bytes + plan.alignment - 1) /
                                    plan.alignment * plan.alignment,
          "一時領域が境界に揃う");
    check(plan.peak_bytes == plan.scratch_base + plan.scratch_bytes, "peak = 値 + 一時");

    bool aligned = true, inside = true;
    for (const auto& s : plan.slots) {
        if (s.offset % plan.alignment != 0) aligned = false;
        if (s.offset + s.bytes > plan.values_bytes) inside = false;
    }
    check(aligned, "offset が境界に揃う");
    check(inside, "置き場が値の領域に収まる");

    // ① 寿命の重なる値どうしは重ならない（同一の置き場を指す view を除く）。
    bool disjoint = true;
    for (std::size_t i = 0; i < plan.slots.size(); ++i) {
        for (std::size_t j = i + 1; j < plan.slots.size(); ++j) {
            const auto& a = plan.slots[i];
            const auto& b = plan.slots[j];
            // 寿命が重ならないものは同じ offset を共有してよい（それが再利用）。
            const bool live_together = !(a.last_use < b.first_use || b.last_use < a.first_use);
            if (!live_together) continue;
            if (a.offset == b.offset) {
                // 同時に生きているのに同じ置き場を指してよいのは view（reshape）だけ。
                if (!a.aliased && !b.aliased) disjoint = false;
                continue;
            }
            if (overlaps(a.offset, a.bytes, b.offset, b.bytes)) disjoint = false;
        }
    }
    check(disjoint, "寿命の重なる値は重ならない");

    // ② 再利用が起きている（重複 offset がある＝peak が「全部を積む」より小さい）。
    std::set<std::size_t> offsets;
    std::size_t sum_bytes = 0;
    for (const auto& s : plan.slots) {
        if (!s.aliased) offsets.insert(s.offset);
        sum_bytes += s.aliased ? 0 : s.bytes;
    }
    check(offsets.size() < plan.slots.size(), "offset が再利用されている");
    check(plan.values_bytes < sum_bytes, "値の領域が「全部を積む」より小さい");
    std::printf("  values=%zu B / 積んだ場合=%zu B（%.2f 倍）scratch=%zu B\n",
                plan.values_bytes, sum_bytes,
                static_cast<double>(plan.values_bytes) / static_cast<double>(sum_bytes),
                plan.scratch_bytes);

    // ④ reshape は元の値と同じ置き場を指す。
    std::size_t aliases = 0;
    for (const auto& s : plan.slots) {
        if (!s.aliased) continue;
        ++aliases;
        bool found = false;
        for (const auto& o : plan.slots) {
            if (&o != &s && o.offset == s.offset && o.bytes == s.bytes) found = true;
        }
        check(found, "view の置き場が元の値と一致する");
    }
    std::printf("  view（reshape）=%zu 件\n", aliases);

    // 行数を増やすと peak が増える（アリーナ必要量の性質）。
    auto g4 = model::qwen35::build_step_graph(cfg, 16, 4);
    if (g4.ok()) {
        const auto p4 = compiler::plan_arena(g4.value().graph);
        check(p4.peak_bytes > plan.peak_bytes, "行数で peak が増える");
        check(p4.slots.size() == g4.value().graph.primitives().size(), "行数違いでも全出力に置き場");
    }

    // 前段 hidden を受けるステージ（非先頭）は入力の受け口が先頭に付く。
    {
        model::qwen35::StageRange r{1, 2};
        auto gs = model::qwen35::build_step_graph(cfg, 16, 1, r);
        if (gs.ok()) {
            const auto ps = compiler::plan_arena(gs.value().graph);
            check(ps.input_bytes > 0, "中間ステージは hidden の受け口を持つ");
            check(ps.values_bytes > ps.input_bytes, "値の領域が受け口の後ろに続く");
        } else {
            check(false, "中間ステージのグラフを作れる");
        }
    }

    // ⑤ 決定論的。
    {
        const auto again = compiler::plan_arena(graph);
        bool same = again.slots.size() == plan.slots.size();
        for (std::size_t i = 0; same && i < plan.slots.size(); ++i) {
            same = again.slots[i].offset == plan.slots[i].offset &&
                   again.slots[i].bytes == plan.slots[i].bytes;
        }
        check(same, "同じグラフからは同じ表");
    }

    // ③ 現行の自動算出に対する上限（切替でメモリが増えないこと）。
    {
        std::size_t f32_sum = 0;
        for (const auto& prim : graph.primitives())
            f32_sum += compiler::arena_numel(prim.output.shape) * sizeof(float);
        const std::size_t current_formula = (64u << 20) + 2u * f32_sum;
        check(plan.peak_bytes <= current_formula, "offsets 版は現行の自動算出を超えない");
        std::printf("  peak=%zu B / 現行式=%zu B（比 %.3f）\n", plan.peak_bytes, current_formula,
                    static_cast<double>(plan.peak_bytes) /
                        static_cast<double>(current_formula));
    }

    // 再利用を切ると「全部を積む」に戻る（デバッグ読み出し用の経路）。
    {
        const auto no_reuse = compiler::plan_arena(graph, false);
        check(!no_reuse.reuse, "reuse=false が反映される");
        check(no_reuse.values_bytes >= sum_bytes, "再利用なしでは全部を積む");
        check(no_reuse.values_bytes >= plan.values_bytes, "再利用なしの方が広い");
    }

    std::printf("test_arena_plan: slots=%zu peak=%zu B / failures=%d\n", plan.slots.size(),
                plan.peak_bytes, failures);
    return failures == 0 ? 0 : 1;
}
