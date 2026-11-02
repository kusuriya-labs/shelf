// shelf unit test — checkpoint の `quantization_config`（FP8 の宣言）解析。
//
// 宣言を**そのまま**実行形（スケール粒度・スケールテンソル名・活性化粒度）へ
// 落とせているかを見る。推測で埋めた值是不正なので、未対応の宣言は明示的に
// 失敗することを確認する（サイレントに近似しない）。
#include <cstdio>
#include <string>

#include <shelf/quant/fp8_perchannel.hpp>
#include <shelf/quant/fp8_finegrained.hpp>

#include "common/app_support.hpp"

namespace {

using shelf::app::Fp8ModelFormat;
using shelf::app::read_fp8_format;
using shelf::quant::fp8::fine::WeightScaleGranularity;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

// 9B-FP8（ModelOpt FP8_DYNAMIC / compressed-tensors）の実形の要約。
const char* kCompressedChannelToken = R"({
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "kv_cache_scheme": null,
    "config_groups": {
      "group_0": {
        "targets": ["Linear"],
        "format": "float-quantized",
        "weights": {"num_bits": 8, "type": "float", "strategy": "channel",
                    "symmetric": true, "dynamic": false, "group_size": null,
                    "block_structure": null, "observer": "memoryless_minmax"},
        "input_activations": {"num_bits": 8, "type": "float", "strategy": "token",
                              "symmetric": true, "dynamic": true, "group_size": null,
                              "block_structure": null}
      }
    }
  }
})";

// 27B-FP8（HF blockwise fp8）の実形の要約。
const char* kBlockwiseFp8 = R"({
  "quantization_config": {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": ["lm_head"]
  }
})";

}  // namespace

int main() {
    {
        auto r = read_fp8_format(kCompressedChannelToken);
        check(r.ok(), "compressed-tensors channel/token が解析できる");
        if (!r.ok()) { std::printf("  %s\n", r.status().message().c_str()); }
        else {
            const auto& f = r.value();
            check(f.quantized, "量子化フラグ");
            check(f.weight == WeightScaleGranularity::PerChannel, "重み = 出力チャネル単位");
            check(f.act_group == 0u, "活性化 = トークン（行）単位");
            check(f.scale_suffix == "_scale", "スケール名は weight_scale");
            std::printf("  9B 形: %s\n", f.declared.c_str());
        }
    }
    {
        auto r = read_fp8_format(kBlockwiseFp8);
        check(r.ok(), "fp8 blockwise が解析できる");
        if (!r.ok()) { std::printf("  %s\n", r.status().message().c_str()); }
        else {
            const auto& f = r.value();
            check(f.quantized, "量子化フラグ（blockwise）");
            check(f.weight == WeightScaleGranularity::Block128, "重み = 128x128 ブロック");
            check(f.act_group == 128u, "活性化 = 1x128");
            check(f.scale_suffix == "_scale_inv", "スケール名は weight_scale_inv");
            std::printf("  27B 形: %s\n", f.declared.c_str());
        }
    }
    {
        auto r = read_fp8_format(R"({"model_type":"qwen3_5_text"})");
        check(r.ok() && !r.value().quantized, "quantization_config 無しは非量子化（bf16 正本）");
    }
    {
        auto r = read_fp8_format("");
        check(r.ok() && !r.value().quantized, "config が空でも落とさない");
    }
    {
        // compressed-tensors でも block 128 + group 128 を宣言できる（27B と同じ形）。
        auto r = read_fp8_format(R"({
          "quantization_config": {"quant_method":"compressed-tensors","kv_cache_scheme":null,
            "config_groups": {"group_0": {
              "weights": {"num_bits":8,"type":"float","strategy":"block","block_structure":[128,128]},
              "input_activations": {"num_bits":8,"type":"float","strategy":"group","group_size":128,"dynamic":true}
            }}}})");
        check(r.ok() && r.value().weight == WeightScaleGranularity::Block128 &&
                  r.value().act_group == 128u,
              "compressed-tensors の block128 / group128 も同じ実行形になる");
    }
    {
        // 未対応は明示的に失敗させる（近似しない）。
        auto a = read_fp8_format(R"({"quantization_config":{"quant_method":"gptq"}})");
        check(!a.ok(), "未対応の quant_method は失敗");
        auto b = read_fp8_format(R"({
          "quantization_config": {"quant_method":"fp8","activation_scheme":"dynamic","fmt":"e4m3",
            "weight_block_size":[64,64]}})");
        check(!b.ok(), "weight_block_size が 128x128 以外は失敗");
        auto c = read_fp8_format(R"({
          "quantization_config": {"quant_method":"fp8","activation_scheme":"static","fmt":"e4m3",
            "weight_block_size":[128,128]}})");
        check(!c.ok(), "活性化の静的スケールは失敗");
        auto d = read_fp8_format(R"({
          "quantization_config": {"quant_method":"compressed-tensors","kv_cache_scheme":{"num_bits":8},
            "config_groups": {"group_0": {
              "weights": {"num_bits":8,"type":"float","strategy":"channel"},
              "input_activations": {"num_bits":8,"type":"float","strategy":"token","dynamic":true}}}}})");
        check(!d.ok(), "KV 量子化の宣言は失敗（未対応を隠さない）");
        auto e = read_fp8_format(R"({
          "quantization_config": {"quant_method":"compressed-tensors","kv_cache_scheme":null,
            "config_groups": {"group_0": {
              "weights": {"num_bits":4,"type":"int","strategy":"channel"},
              "input_activations": {"num_bits":8,"type":"float","strategy":"token","dynamic":true}}}}})");
        check(!e.ok(), "int 4bit の宣言は失敗（EXL3 系に戻さない）");
        auto f = read_fp8_format(R"({
          "quantization_config": {"quant_method":"compressed-tensors","kv_cache_scheme":null,
            "config_groups": {"group_0": {
              "weights": {"num_bits":8,"type":"float","strategy":"channel"},
              "input_activations": {"num_bits":8,"type":"float","strategy":"group","group_size":64,"dynamic":true}}}}})");
        check(!f.ok(), "活性化 group_size=64 は失敗（128 のみ実装）");
    }

    {
        // MXFP4（amd-quark）: 実測の宣言。`weight` U8[out,K/2] / `weight_scale` U8[out,K/32]
        auto r = read_fp8_format(R"({
          "quantization_config": {"quant_method":"quark","version":"0.13+unknown",
            "global_quant_config": {"weight": {"dtype":"fp4","qscheme":"per_group",
              "group_size":32,"scale_format":"e8m0","scale_type":"float",
              "observer_cls":"PerBlockMXObserver"},
              "input_tensors": {"dtype":"fp4","qscheme":"per_group","group_size":32,
              "scale_format":"e8m0"}}}})");
        check(r.ok(), "quark(MXFP4) の宣言を解析できる");
        if (r.ok()) {
            const auto& f = r.value();
            check(f.quantized && f.mxfp4, "MXFP4 として印が付く");
            check(f.mxfp4_group == 32u, "group 32");
            check(f.scale_suffix == "_scale", "スケール名は weight_scale");
            check(f.act_group == 128u, "活性化は W4A8 group128（既定）");
            std::printf("  MXFP4 形: %s\n", f.declared.c_str());
        }

        // `input_tensors` の宣言（e4m3 / 無し）によらず W4A8 group128。
        const char* kCases[] = {
            R"({"quantization_config":{"quant_method":"quark","global_quant_config":{
                "weight":{"dtype":"fp4","qscheme":"per_group","group_size":32,
                          "scale_format":"e8m0"},
                "input_tensors":{"dtype":"e4m3","qscheme":"per_group","group_size":32}}}})",
            R"({"quantization_config":{"quant_method":"quark","global_quant_config":{
                "weight":{"dtype":"fp4","qscheme":"per_group","group_size":32,
                          "scale_format":"e8m0"}}}})",
        };
        for (const char* js : kCases) {
            auto r2 = read_fp8_format(js);
            check(r2.ok() && r2.value().act_group == 128u,
                  "input_tensors の宣言に依存せず W4A8 group128");
        }
        // 逸脱は明示的に失敗させる（近似しない）
        auto bad_dtype = read_fp8_format(R"({"quantization_config":{"quant_method":"quark",
            "global_quant_config":{"weight":{"dtype":"int4","qscheme":"per_group","group_size":32}}}})");
        check(!bad_dtype.ok(), "quark の int4 は拒否");
        auto bad_group = read_fp8_format(R"({"quantization_config":{"quant_method":"quark",
            "global_quant_config":{"weight":{"dtype":"fp4","qscheme":"per_group","group_size":16}}}})");
        check(!bad_group.ok(), "quark の group_size=16 は拒否");
        auto bad_scheme = read_fp8_format(R"({"quantization_config":{"quant_method":"quark",
            "global_quant_config":{"weight":{"dtype":"fp4","qscheme":"per_tensor","group_size":32}}}})");
        check(!bad_scheme.ok(), "quark の per_tensor は拒否");
    }

    std::printf("test_fp8_format: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
