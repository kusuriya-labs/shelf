// shelf unit test — checkpoint の `quantization_config`（FP8 の宣言）解析。
//
// 宣言を**そのまま**実行形（スケール粒度・スケールテンソル名・活性化粒度）へ
// 落とせているかを見る。推測で埋めた值是不正なので、未対応の宣言は明示的に
// 失敗することを確認する（サイレントに近似しない）。
#include <cstdio>
#include <string>

#include <shelf/quant/fp8_w8a8.hpp>

#include "common/app_support.hpp"

namespace {

using shelf::app::Fp8ModelFormat;
using shelf::app::read_fp8_format;
using shelf::quant::fp8::WeightScaleGranularity;

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

    std::printf("test_fp8_format: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
