// shelf api — 生成パラメータの再輸出（実装は `runtime/generate`）。
//
// 境界ルール #12 により `src/runtime` は `shelf/api/` を include できない。
// そのため抽選の実装は runtime 側に置き、Server 層（このヘッダ）は名前を再輸出する。
// 型・関数の意味は `shelf/runtime/generate/sampling.hpp` が正本。
#pragma once

#include <shelf/runtime/generate/sampling.hpp>

namespace shelf::api {

using runtime::generate::Draw;
using runtime::generate::Philox4x32;
using runtime::generate::Sampling;
using runtime::generate::assign_seed;
using runtime::generate::sample;
using runtime::generate::validate_sampling;

}  // namespace shelf::api
