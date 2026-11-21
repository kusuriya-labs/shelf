# MTPDrafter の TP 分割

## 目的

Qwen3.8-Flash-Next（`qwen4_exp`）の `--serve-stdio --mtp` で、rank0 に集中していた
MTP block と drafter 用 full-vocabulary lm_head を TP rank に分割し、VRAM を均等化する。

## 結論

ACCEPT。TP=2 の serve で MTP block を全 rank に load し、attention と routed experts を
TP 軸で分割する。drafter は target の語彙分割 lm_head を共用し、draft の greedy は
logits の TP combine で全語彙を得る。

単発生成（`--serve-stdio` なし）は従来どおり rank0 が MTP block 全体を持つ。この経路は
`mtp_split` を立てない。

## 変更

- `Qwen4ExpLoadOptions::mtp_split` を追加した。true のときだけ MTP tensor を全 rank に
  load し、`mtp.layers.0.` の attention を TP partition plan に加える。
- routed experts は `load_moe` の ep_size / ep_rank で分割する。既定 EP=1 では
  `LegacyAcrossTP` として TP rank に expert 範囲を割り当てる。
- MTP lower は local query/kv head と local expert 範囲を使い、attention と MoE の
  `tp_combine` を有効にする。lm_head は `weights.lm_head` を使い、full head の複製を廃止した。
- `MtpDrafter` は peer の executor と sequence を保持する。draft forward は
  `TpCoordinator::on_execute_aux` で全 rank に同じ batch を流す。
- peer の MTP KV ring は local KV head 数で作る。leader と同じ position へ進め、
  reset / rewind も同時に適用する。
- peer の hidden 入力は leader の hidden を host 経由で配る。peer 間の
  `hipMemcpyPeerAsync` はこの環境で GPU hang を起こしたため使わない。

## 実測

条件: `Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`、TP=2、`device=2,3`、
`--mtp --mtp-drafts 3 --mtp-window 4096 --mtp-sink 64`、`max_seq_len=8192`、
`max_concurrent_requests=2`、BF16 KV。

| 項目 | 分割前 | 分割後 |
| --- | ---: | ---: |
| target weights rank0 | 26.32 GiB | 25.50 GiB |
| target weights rank1 | 25.00 GiB | 25.50 GiB |
| MTP batch reserve rank0 | 1.41 GiB | 0.35 GiB |
| MTP batch reserve rank1 | 0.00 GiB | 0.35 GiB |
| KV budget rank0 | 1.62 GiB | 4.41 GiB |
| KV budget rank1 | 4.35 GiB | 4.41 GiB |

分割前は 262144 の run、分割後は 8192 の run の値であり、MTP batch reserve は
`max_seq_len` に依存する。weight と rank 間の対称性は比較できる。

## 検証

- 2 request 同時、`max_new_tokens=12` の target-only と MTP の生成列が一致した。
- `max_seq_len=8192`、`mtp-window 4096`、2 request の MTP serve が完走した。
- ring 有効時の rollback 回帰テストを `test_prefix_cache` に追加した。

## verify 方式と decode

条件: TP=2、`max_seq_len=4096`、`mtp-drafts=3`、`mtp-window=4096`、単一 request、
同じ prompt・64 token。

| verify | decode | MTP_BATCH |
| --- | ---: | --- |
| target-only | 35.4 tok/s | - |
| safe serial | 33.1 tok/s | `chunks=63 emitted=63` |
| fused（`SHELF_MTP_FUSED_VERIFY=1`） | 68.0 tok/s | `chunks=23 emitted=63` |

safe serial は `accepted + rounds == emitted == chunks` であり、1 token あたり 1 回の
target decode になる。MTP の draft 分だけ余計なので target-only を超えない。
fused は 1 round 1 target forward で、この条件では約 2 倍になった。生成列は
3 方式で一致した。

fused verify では `create_mtp_spec_batch_context` が rank ごとの GDN history /
compact log を確保するため、TP serve の MTP reserve に
`mtp_batch_rank_state_reserve_bytes` を加算する必要がある。加算前は起動時に
arena が 0.05 GiB 不足した。

`SHELF_MTP_FUSED_VERIFY=1` で `test_compute_mtp_batch.py`（TP=2、window 4096、
2 request）は 10/10 PASS した。ただし near-tie 分岐の許容契約を含むため、
target-only との bit 一致を要求する用途では追加検証が必要である。

## prefix cache

MTP の admit は prefix cache を lookup していなかったため、`restored_tokens` は常に 0
だった。target の prefix state に加えて drafter の KV ring / hidden / last token /
position を同じ checkpoint に保存する `MtpDraftSnapshot` を追加した。

- target は通常の `PrefixCache::restore` と TP mirror restore を使う。
- drafter の ring は leader と peer の rank ごとに保存・復元する。
- snapshot の MTP position が prefix - 1 のときは、suffix の先頭 token を 1 回
  MTP forward して position を揃えてから suffix を prefill する。
- suffix が空の prompt は pending token を作れないため restore しない。

検証: 4096 token の prefix を持つ request を流した後、同じ prefix + 100 token の
request を流した。2 本目は `restored_tokens=4096`、`prefill_tokens=100` で完了した。

```text
PFXDBG mtp lookup hit=1 state=4096 in=4196 snapshots=2
PFXDBG mtp restored prefix=4096 target=4096 mtp=4095 last=4095
[req 2] prompt=4196 restored=4096 prefill=100
```

3162 token の request の後、同じ prefix + 3 token の request でも確認した。
2 本目は `restored_tokens=2048`、`prefill_tokens=1117` で、復元後も MTP decode が
継続した。

```text
[req 1] prompt=3162 restored=0 prefill=3162 decode_tok_s=59.5
[req 2] prompt=3165 restored=2048 prefill=1117 decode_tok_s=45.5
```

`tests/server/test_compute_mtp_prefix.py` を追加し、5/5 PASS を確認した。

## slot 再利用時の不具合

`MTP_SLOTS` の slot 順と `mtp.active` の並びが一致しない状態（短い request が先に
終わり、その slot を別の request が再利用する場合）で、batch draft が peer sequence を
`context.peer_drafters[rank][active index]` で選んでいた。peer sequence は drafter slot
ごとに独立しているため、別 slot の sequence を渡すと `prefix_tokens != sequence.position`
で失敗した。

leader の `MtpDrafter::peers` から同じ slot の peer sequence を選ぶよう修正した。
`mtp.active` の並びが drafter slot 順と一致しなくても、leader と peer の sequence が
同じ slot に対応する。

修正後、長い request と短い request を同時に流し、短い方が終わった直後に同じ slot へ
別の request を admit する条件で完走した。`test_compute_mtp_batch.py` にも slot 再利用の
確認を追加した。

## 未検証

- `--expert-parallel > 1` の MTP 経路。TP と EP が別軸になる場合の collectives は
  実測していない。
- 最大 context 262144 での分割後の MTP batch reserve と KV budget。
- decode 速度。MTP draft が 2 GPU に分かれるため、重みトラフィックは半減するが、
  draft ごとの collective が増える。target-only / MTP の比較測定が必要である。
