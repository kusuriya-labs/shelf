# N-gram 量子化 Step 3: PopQA 短答生成

## 実験と判定

実験 ID: NGRAM-Q-STEP3-POPQA。固定したチャット形式で 300 問を各モードについて greedy 生成し、対照との対応のある正誤を比較した。モデルは Hy4-Flash EXL3 4bpw、Engram の評価には Step 2 の codebook を用いた。

評価用 300 問は人気度 low / middle / high 各 100 問。探索用 10 問は別の calibration 集合を使用した。単純な few-shot 形式では質問の反復や無関係な文章が生じたため、モデル付属のチャット形式と `reasoning_effort:no_think` を使う形式に変更した。後者は探索用 10 問すべてで短答または短い不明回答を返した。3 個の固定例をプロンプトに含め、全モードで同一条件とした。

各問題・モードで KV と内部状態を初期化し、最大 16 トークン、EOS または改行で停止した。生成結果は `/tmp/ngram_q_step3_hy4/generations.jsonl`、集計は `scores.json`。モデルの出力や選択肢は小文字化し、前後の空白と句読点を除去した上で、正解候補を生成文に含む substring 一致と完全一致を別々に集計した。McNemar 検定は不一致数に基づく両側の正確二項検定。

## 最終判定

**候補選択は `lloyd4` を優先し、`lloyd5` を品質重視の代替候補とする。** 部分一致では `lloyd4` は 119/300、`lloyd5` は 122/300 であり、`lloyd8` の 123/300 との正答率差はそれぞれ -1.33、-0.33 percentage point だった。reference 正解 119 問のうち 111 問（93.3%）をいずれも維持した。差が有意でないことは同等性や劣化がないことの証明ではない。4 bit と 5 bit の選択は、[Step 2](ngram_q_step2_hy4.md) の teacher forcing 評価とビット数のトレードオフを踏まえた暫定判断であり、本評価だけで優劣を証明したものではない。

`mix43` と `lloyd3` の部分一致はどちらも 122/300 で、今回の短答正答率では `lloyd4` や `lloyd5` の優位を示せない。一方、Step 2 の低人気度層では `mix43` の保持率に有意な低下があり、`lloyd3` は回答区間の KL がより大きかった。このため両者を今回の第一候補にはしない。ただし本実験では両者の正答率低下を検出しておらず、Step 2 と指標が異なる結果として扱う。Step 2 と同じ評価問題を使っているため独立した再現ではない。表の p 値は多重比較の補正なしであり、300 問では小さな品質差を見落とす可能性がある。圧縮形式の実装後の速度、メモリ削減、他タスクの品質はこの評価では検証していない。

## 部分一致

正答数は全体 300 問、人気度別は各 100 問。維持率の分母は reference 正解数。差は lloyd8 に対する正答率差（percentage point）。

| モード | 全体 | low | middle | high | reference 正解の維持 | reference 勝ち/負け | p 対 reference | 差 対 lloyd8 | p 対 lloyd8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 119/300 | 32/100 | 31/100 | 56/100 | 119/119 | 0/0 | 1 | -1.33 | 0.424 |
| zero | 57/300 | 16/100 | 8/100 | 33/100 | 51/119 | 6/68 | 2.145e-14 | -22.00 | 5.249e-16 |
| lloyd8 | 123/300 | 35/100 | 35/100 | 53/100 | 114/119 | 9/5 | 0.424 | +0.00 | 1 |
| lloyd5 | 122/300 | 34/100 | 32/100 | 56/100 | 111/119 | 11/8 | 0.6476 | -0.33 | 1 |
| lloyd4 | 119/300 | 33/100 | 31/100 | 55/100 | 111/119 | 8/8 | 1 | -1.33 | 0.424 |
| mix43 | 122/300 | 33/100 | 33/100 | 56/100 | 111/119 | 11/8 | 0.6476 | -0.33 | 1 |
| lloyd3 | 122/300 | 34/100 | 32/100 | 56/100 | 111/119 | 11/8 | 0.6476 | -0.33 | 1 |

層別の対照比較と不一致件数は `scores.json` に記録した。

## 完全一致

正答数は全体 300 問、人気度別は各 100 問。維持率の分母は reference 正解数。差は lloyd8 に対する正答率差（percentage point）。

| モード | 全体 | low | middle | high | reference 正解の維持 | reference 勝ち/負け | p 対 reference | 差 対 lloyd8 | p 対 lloyd8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 112/300 | 31/100 | 28/100 | 53/100 | 112/112 | 0/0 | 1 | -0.33 | 1 |
| zero | 50/300 | 12/100 | 8/100 | 30/100 | 47/112 | 3/65 | 3.555e-16 | -21.00 | 3.089e-17 |
| lloyd8 | 113/300 | 33/100 | 32/100 | 48/100 | 105/112 | 8/7 | 1 | +0.00 | 1 |
| lloyd5 | 112/300 | 33/100 | 30/100 | 49/100 | 103/112 | 9/9 | 1 | -0.33 | 1 |
| lloyd4 | 111/300 | 31/100 | 29/100 | 51/100 | 105/112 | 6/7 | 1 | -0.67 | 0.7744 |
| mix43 | 113/300 | 30/100 | 31/100 | 52/100 | 104/112 | 9/8 | 1 | +0.00 | 1 |
| lloyd3 | 113/300 | 32/100 | 30/100 | 51/100 | 103/112 | 10/9 | 1 | +0.00 | 1 |

層別の対照比較と不一致件数は `scores.json` に記録した。

## 不一致例

### lloyd4: reference のみ正解

- Who was the screenwriter for A Busy Day? 正解候補: Charlie Chaplin, Charles Spencer Chaplin, Charles Chaplin, Sir Charles Spencer Chaplin。reference: Charlie Chaplin。lloyd4: Cecil B. DeMille。
- In what country is Ector? 正解候補: United States of America, the United States of America, America, U.S.A., USA, U.S., US, the US, the USA, US of A, the United States, U. S. A., U. S., the States, the U.S., 'Merica, U.S, United States, 'Murica。reference: United States。lloyd4: France。
- What is Friedrich Carl von Savigny's occupation? 正解候補: jurist。reference: Jurist。lloyd4: Lawyer。

### lloyd4: 候補のみ正解

- Who was the composer of Separado!? 正解候補: Gruff Rhys, Gruffudd Maredudd Bowen Rhys。reference: David Byrne。lloyd4: Gruff Rhys。
- What genre is Niall Toner? 正解候補: bluegrass music, bluegrass, Bluegrass。reference: Country。lloyd4: Bluegrass/ Irish folk/ Country.。
- Who is the author of The Rain Before It Falls? 正解候補: Jonathan Coe。reference: Joyce Carol Oates。lloyd4: Jonathan Coe。

### lloyd5: reference のみ正解

- Who was the screenwriter for A Busy Day? 正解候補: Charlie Chaplin, Charles Spencer Chaplin, Charles Chaplin, Sir Charles Spencer Chaplin。reference: Charlie Chaplin。lloyd5: Anita Loos。
- Who was the composer of Bad Things? 正解候補: Jace Everett。reference: Jace Everett。lloyd5: Jaden Smith and Justin Bieber。
- What genre is GeneRally? 正解候補: racing video game, racing game, racing video games, racer。reference: Racing game。lloyd5: Racing。

### lloyd5: 候補のみ正解

- Who was the director of The Woman on the Jury? 正解候補: Harry O. Hoyt, Harry Orlando Hoyt。reference: Bernard D. H. Hyman。lloyd5: Harry O. Hoyt。
- What genre is Niall Toner? 正解候補: bluegrass music, bluegrass, Bluegrass。reference: Country。lloyd5: Bluegrass。
- Who is the author of The Rain Before It Falls? 正解候補: Jonathan Coe。reference: Joyce Carol Oates。lloyd5: Jonathan Coe。

## 再現方法

GPU は gfx1201 1 枚と gfx1030 2 枚。評価用コードは指定 worktree の `build-hy4-eval/shelf-compute` に差分ビルドした。入力生成と採点は以下で再現する。

```bash
python3 /tmp/ngram_q_step3_hy4/prepare.py
ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55,GPU-08b2ddcbd6e6b36c,GPU-76a08c022586fed6 \
  build-hy4-eval/shelf-compute hy4 eval-engram \
  --model-dir /home/homelab1/llm-fast-load/Hy4-flash-exl3-4bpw-mul24 \
  --engram-dir /home/homelab1/datapool/rocm-exl3-rdna2/hy4-flash-pilot/engram-e4m3fn \
  --placement /home/homelab1/datapool/rocm-exl3-rdna2/hy4-flash-pilot/shelf-placement/owners-4bpw.json \
  --max-seq-len 192 --chunk 64 \
  --sequences /tmp/ngram_q_step3_hy4/eval.jsonl \
  --codebooks /tmp/ngram_q_step2_hy4/codebooks.json \
  --out /tmp/ngram_q_step3_hy4/generations_raw.jsonl \
  --modes reference,zero,lloyd8,lloyd5,lloyd4,mix43,lloyd3 \
  --eval-generate 1 --max-new-tokens 16
python3 /tmp/ngram_q_step3_hy4/score_popqa.py
```
