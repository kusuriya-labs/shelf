// shelf compiler — declare 相: 活性化アリーナの offsets（D4）。
//
// 目的: run 相から割当 API を外す。グラフから offsets を**事前に**決め、実行器は
//       その表に従って書くだけにする（radiance の `arena.h` 規約と同じ形）。
//
// 決め方:
//   * 値（primitive の出力）は SSA なので、寿命は「生成された位置 〜 最後に読まれた
//     位置」の閉区間で決まる。重ならない区間どうしは同じ offset を再利用できる。
//   * `reshape` は view なので、元の値と同じ offset を指す（新しい領域を取らない）。
//   * 一時領域（rope の f32 展開、elem_mul の変換、GDN の準備バッファ等）は
//     **1 つの primitive の内側でしか生きない**。ストリームは直列なので全 primitive で
//     同じ領域を共有でき、必要量は「1 primitive 分の最大」で足りる。
//
// この表は GPU 無しで作れる（純粋関数）。実行器は offsets をそのまま使う。
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <shelf/foundation/numeric.hpp>
#include <shelf/foundation/shape.hpp>
#include <shelf/ir/graph.hpp>

namespace shelf::compiler {

// 形状の要素数（rank 0 は 0 とみなす）。
inline std::size_t arena_numel(const TensorShape& s) {
    if (s.rank() == 0) return 0;
    std::size_t n = 1;
    for (std::size_t i = 0; i < s.rank(); ++i) n *= s.dim(i);
    return n;
}

// 実行器の方針: この primitive の出力を f32 で持つか。
// bf16 の活性化を基本とし、f32 を要求するカーネル（logits / silu / GDN 出力）だけ f32。
// **方針を変えるときはここ 1 か所と実行器の `alloc()` の引数を同時に直す**
// （食い違えば実行器が `insufficient_memory` で止まる。黙って壊れない）。
inline bool arena_output_is_f32(ir::Kind kind) noexcept {
    switch (kind) {
    case ir::Kind::output_projection:   // logits は f32 で読む
    case ir::Kind::silu:                // bf16 入力・f32 出力のカーネル
    case ir::Kind::gdn_step:            // 再帰カーネルが f32 出力
        return true;
    default:
        return false;
    }
}

// 1 つの値の置き場。
struct ArenaSlot {
    ir::ValueId value;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    bool f32 = false;        // true = f32、false = bf16
    bool aliased = false;    // reshape: 元の値と同じ置き場を指す
    std::uint32_t first_use = 0;   // 生成位置（primitive 添字）
    std::uint32_t last_use = 0;    // 最後に読まれる位置（閉区間）
};

// primitive 内の一時領域。`slot` はその primitive の中での通し番号。
struct ArenaScratch {
    std::size_t primitive = 0;
    std::uint8_t slot = 0;
    std::size_t offset = 0;   // 一時領域の先頭からの相対
    std::size_t bytes = 0;
};

struct ArenaPlan {
    std::vector<ArenaSlot> slots;       // primitive 出力 1 件につき 1 行（生成順）
    std::vector<ArenaScratch> scratch;  // 一時領域（primitive 内でのみ使う）
    std::size_t alignment = 256;        // WMMA のロードに十分な境界
    std::size_t input_offset = 0;       // 前段 hidden の受け口（非先頭ステージ）
    std::size_t input_bytes = 0;
    std::size_t values_bytes = 0;       // 値の領域の終端
    std::size_t scratch_base = 0;       // 一時領域の開始（values_bytes を境界に揃えた値）
    std::size_t scratch_bytes = 0;      // 一時領域の大きさ（1 primitive 分の最大）
    std::size_t peak_bytes = 0;         // 確保すべき総量
    bool reuse = true;                  // false = 寿命再利用なし（デバッグ読み出し用）

    const ArenaSlot* slot_of(ir::ValueId v) const {
        for (const auto& s : slots) {
            if (s.value == v) return &s;
        }
        return nullptr;
    }
    const ArenaScratch* scratch_of(std::size_t primitive, std::uint8_t slot) const {
        for (const auto& s : scratch) {
            if (s.primitive == primitive && s.slot == slot) return &s;
        }
        return nullptr;
    }
};

namespace detail {

inline std::size_t arena_align_up(std::size_t v, std::size_t a) {
    return a ? (v + a - 1) / a * a : v;
}

// primitive が要求する一時領域（実行器の実装と同じ式）。
struct ScratchNeed {
    std::uint8_t slot;
    std::size_t bytes;
};

inline std::vector<ScratchNeed> scratch_needs(
    const ir::Graph& g, const ir::Primitive& p,
    const std::unordered_map<std::uint32_t, bool>& is_f32) {
    std::vector<ScratchNeed> out;
    const std::size_t out_n = arena_numel(p.output.shape);
    switch (p.kind) {
    case ir::Kind::rope:
        // bf16 → f32 に広げてから回す（vendored rope は f32 入力）。
        out.push_back({0, out_n * sizeof(float)});
        break;
    case ir::Kind::elem_mul: {
        // 入力が bf16 のときだけ f32 へ広げる（実行器の分岐と同じ条件）。
        // 番号は**入力の位置**で固定する（実行器が a=0 / b=1 で引くため）。
        for (std::size_t k = 0; k < p.inputs.size(); ++k) {
            const auto it = is_f32.find(p.inputs[k].value());
            const bool is32 = it != is_f32.end() && it->second;
            if (!is32) out.push_back({static_cast<std::uint8_t>(k), out_n * sizeof(float)});
        }
        break;
    }
    case ir::Kind::causal_conv_step: {
        // 畳み込みは f32 の中間を作る（活性化は bf16 のまま）。
        const auto* x = p.inputs.empty() ? nullptr : g.find_value(p.inputs[0]);
        if (x) out.push_back({0, arena_numel(x->shape) * sizeof(float)});
        break;
    }
    case ir::Kind::gdn_step: {
        // q/k の正規化に 2 本 + スケール済み q に 1 本（どれも 1 トークン分の幅）。
        const auto* q = p.inputs.empty() ? nullptr : g.find_value(p.inputs[0]);
        if (q) {
            const std::size_t bytes = arena_numel(q->shape) * sizeof(float);
            out.push_back({0, bytes});
            out.push_back({1, bytes});
            out.push_back({2, bytes});
        }
        break;
    }
    case ir::Kind::rmsnorm_gated: {
        // 入力が f32（GDN 出力）のときは f32 → bf16 の変換を挟む。
        const auto it = p.inputs.empty() ? is_f32.end() : is_f32.find(p.inputs[0].value());
        const bool x32 = it != is_f32.end() && it->second;
        if (x32) {
            out.push_back({0, out_n * sizeof(float)});
            out.push_back({1, out_n * sizeof(bf16_t)});
        } else {
            out.push_back({0, out_n * sizeof(bf16_t)});
        }
        break;
    }
    default:
        break;
    }
    return out;
}

}  // namespace detail

// グラフから offsets を決める。
//
// `reuse == true` で寿命の重ならない値を同じ offset に載せる（既定）。
// `reuse == false` は値をすべて別の offset に置く（ステップ後に中間値を読み出す
// デバッグ経路用。`fetch_value_debug` はステップ終了後に読むため、再利用があると
// 別のテンソルが入っている）。
inline ArenaPlan plan_arena(const ir::Graph& g, bool reuse = true) {
    ArenaPlan p;
    p.reuse = reuse;

    const auto& prims = g.primitives();
    const std::size_t n = prims.size();

    // グラフ入力のうち、データ入力（i32 でない先頭入力 = 前段 hidden）は arena に置く。
    const auto& ins = g.all_inputs();
    if (!ins.empty() && ins[0].dtype != DataType::i32) {
        p.input_offset = 0;
        p.input_bytes = arena_numel(ins[0].shape) * sizeof(bf16_t);
    }

    // 値 →（生成位置, 最後に読まれる位置, f32 か）
    std::unordered_map<std::uint32_t, std::uint32_t> first, last;
    std::unordered_map<std::uint32_t, bool> is_f32;
    std::unordered_map<std::uint32_t, bool> consumed;
    for (const auto& v : ins) {
        first[v.id.value()] = 0;
        last[v.id.value()] = 0;
        is_f32[v.id.value()] = false;   // 入力は bf16 / i32
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto& pr = prims[i];
        const std::uint32_t oid = pr.output.id.value();
        first[oid] = static_cast<std::uint32_t>(i);
        last[oid] = static_cast<std::uint32_t>(i);
        is_f32[oid] = arena_output_is_f32(pr.kind);
        consumed[oid] = false;
        for (const auto in : pr.inputs) {
            const std::uint32_t iid = in.value();
            auto it = last.find(iid);
            if (it == last.end()) {
                first[iid] = 0;
                last[iid] = static_cast<std::uint32_t>(i);
            } else if (it->second < i) {
                it->second = static_cast<std::uint32_t>(i);
            }
            consumed[iid] = true;
        }
    }
    // 誰にも読まれない値（logits / ステージ出力）は取り出されるので終端まで生かす。
    if (n > 0) {
        const std::uint32_t tail = static_cast<std::uint32_t>(n - 1);
        for (const auto& kv : consumed) {
            if (!kv.second) {
                auto it = last.find(kv.first);
                if (it != last.end()) it->second = std::max(it->second, tail);
            }
        }
    }

    // reshape は view。根（元の値）を求めて寿命を併合する。
    std::unordered_map<std::uint32_t, std::uint32_t> alias_root;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& pr = prims[i];
        if (pr.kind != ir::Kind::reshape || pr.inputs.empty()) continue;
        std::uint32_t root = pr.inputs[0].value();
        const auto it = alias_root.find(root);
        if (it != alias_root.end()) root = it->second;
        alias_root[pr.output.id.value()] = root;
        // view は元の値の型を引き継ぐ（f32 の値を reshape しても f32 のまま）。
        const auto fr = is_f32.find(root);
        if (fr != is_f32.end()) is_f32[pr.output.id.value()] = fr->second;
        auto rl = last.find(root);
        const auto ol = last.find(pr.output.id.value());
        if (rl != last.end() && ol != last.end())
            rl->second = std::max(rl->second, ol->second);
    }

    // 割当: 期限切れの区間を再利用し、足りなければ後ろへ伸ばす。
    struct Block {
        std::size_t offset;
        std::size_t bytes;
        std::uint32_t end;   // この区間の最後の使用位置
    };
    std::vector<Block> blocks;
    std::unordered_map<std::uint32_t, Block> root_block;   // 根の値 → 割当

    std::size_t top = detail::arena_align_up(p.input_bytes, p.alignment);
    std::size_t values_end = top;
    p.slots.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        const auto& pr = prims[i];
        const std::uint32_t oid = pr.output.id.value();
        ArenaSlot s;
        s.value = pr.output.id;
        s.f32 = arena_output_is_f32(pr.kind);
        s.bytes = arena_numel(pr.output.shape) * (s.f32 ? sizeof(float) : sizeof(bf16_t));
        s.first_use = first[oid];
        s.last_use = last[oid];

        const auto ai = alias_root.find(oid);
        if (ai != alias_root.end()) {
            s.aliased = true;
            const auto fr = is_f32.find(ai->second);
            if (fr != is_f32.end()) s.f32 = fr->second;
            const auto rb = root_block.find(ai->second);
            if (rb != root_block.end()) {
                s.offset = rb->second.offset;
                s.bytes = rb->second.bytes;
            } else {
                // グラフ入力の view（領域は入力の置き場そのもの）。
                s.offset = p.input_offset;
                s.bytes = p.input_bytes;
            }
            p.slots.push_back(s);
            continue;
        }

        Block chosen{0, 0, 0};
        bool reused = false;
        if (reuse) {
            // 一番若い offset の空きを使う（決定論的で、詰め方が安定する）。
            for (auto& b : blocks) {
                if (b.end < s.first_use && b.bytes >= s.bytes) {
                    chosen = b;
                    b.end = s.last_use;   // 新しい住人の寿命に更新
                    b.bytes = s.bytes;
                    reused = true;
                    break;
                }
            }
        }
        if (!reused) {
            chosen.offset = top;
            chosen.bytes = s.bytes;
            chosen.end = s.last_use;
            top = detail::arena_align_up(chosen.offset + chosen.bytes, p.alignment);
            blocks.push_back(chosen);
            values_end = std::max(values_end, chosen.offset + chosen.bytes);
        }
        s.offset = chosen.offset;
        p.slots.push_back(s);
        root_block[oid] = chosen;
    }

    // 一時領域は 1 本を全 primitive で共有する（直列実行なので重ならない）。
    std::size_t scratch_max = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto needs = detail::scratch_needs(g, prims[i], is_f32);
        std::size_t off = 0;
        for (const auto& nd : needs) {
            ArenaScratch sc;
            sc.primitive = i;
            sc.slot = nd.slot;
            sc.offset = detail::arena_align_up(off, p.alignment);
            sc.bytes = nd.bytes;
            p.scratch.push_back(sc);
            off = sc.offset + sc.bytes;
        }
        scratch_max = std::max(scratch_max, off);
    }

    p.values_bytes = values_end;
    p.scratch_base = detail::arena_align_up(values_end, p.alignment);
    p.scratch_bytes = scratch_max;
    p.peak_bytes = p.scratch_base + scratch_max;
    return p;
}

}  // namespace shelf::compiler
