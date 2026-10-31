#!/usr/bin/env bash
# shelf-serve のプロトコル検査（モデル・GPU を使わない）。
# 使い方: serve_protocol.sh <shelf-serve のパス>
set -u

BIN="${1:?usage: serve_protocol.sh <shelf-serve>}"
fail=0
out="$(printf '%s\n' \
  '{"op":"ping"}' \
  '{"op":"generate","request_id":"r1","input_ids":[1],"max_new_tokens":1}' \
  'not json' \
  | "$BIN" --device cpu 2>/dev/null)"

expect() {  # expect <説明> <grep パターン>
  if ! grep -qF -- "$2" <<<"$out"; then
    echo "FAIL: $1（'$2' が無い）"
    fail=1
  fi
}

expect "ping に pong が返る" '"event":"pong"'
expect "capabilities に backend が入る" '"capabilities":{'
expect "モデル無しの generate は model_not_found" '"code":"model_not_found"'
expect "壊れた行は invalid_argument" '"code":"invalid_argument"'
expect "EOF で shutdown を出す" '"event":"shutdown"'

# クライアント向けメッセージにソース位置を出さない（serve_protocol §1）
if grep -qE '"message":"[^"]*(\.cpp|/opt/)' <<<"$out"; then
  echo "FAIL: message にソース位置が混ざっている"
  fail=1
fi

# 起動時の usage は exit 2、--help は exit 0（GPU を触らない）
"$BIN" --help >/dev/null 2>&1 || { echo "FAIL: --help が exit 0 でない"; fail=1; }
if "$BIN" --bogus >/dev/null 2>&1; then
  echo "FAIL: 未知のオプションが受理された"
  fail=1
fi

exit "$fail"
