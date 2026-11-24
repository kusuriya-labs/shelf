#!/bin/bash
# 27B decode の TP / HIP graph グリッド計測。
# boot ごとに disable_acs_redir の状態を記録するので、ACS 有効・無効の両方で実行して比較する。
#
# 例:
#   tools/bench/tp_graph_grid.sh
#   TP_LIST="2" GRAPH_LIST="on off" RUNS=5 tools/bench/tp_graph_grid.sh
#
# 出力: 1 行 1 計測の CSV と、セルごとの中央値。場所は最後に表示する。
set -u

REPO=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$REPO/build/shelf-bench}
MODEL=${MODEL:-$REPO/models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw}
TP_LIST=${TP_LIST:-"1 2"}
GRAPH_LIST=${GRAPH_LIST:-"on off"}
REDUCE_LIST=${REDUCE_LIST:-"default legacy"}
RUNS=${RUNS:-3}
CONTEXT=${CONTEXT:-64}
TOKENS=${TOKENS:-8}
WARMUP=${WARMUP:-2}
DEVICES=${DEVICES:-0,1}
LOCK_FILE=${LOCK_FILE:-/tmp/r9700.lock}
OUT=${OUT:-$(mktemp -d /tmp/shelf-tp-grid.XXXXXX)}

if [ ! -x "$BIN" ]; then
  echo "shelf-bench が見つかりません: $BIN" >&2
  exit 1
fi
if [ ! -d "$MODEL" ]; then
  echo "model dir が見つかりません: $MODEL" >&2
  exit 1
fi

BOOT=default
if grep -q "disable_acs_redir" /proc/cmdline 2>/dev/null; then
  BOOT=acs_redir_off
fi

mkdir -p "$OUT" || exit 1
CSV=$OUT/grid.csv
echo "boot,tp,graph,reduce,run,wall_tok_s,gpu_tok_s,argmax,logits_sum,status" > "$CSV"
echo "boot=$BOOT"
echo "model=$MODEL"
echo "context=$CONTEXT tokens=$TOKENS warmup=$WARMUP devices=$DEVICES runs=$RUNS"
echo "tp=$TP_LIST graph=$GRAPH_LIST reduce=$REDUCE_LIST"
echo "csv=$CSV"

run_cell() {
  local tp=$1 graph=$2 reduce=$3 run=$4
  local flag=1 devs=$DEVICES
  if [ "$graph" = "off" ]; then flag=0; fi
  if [ "$tp" = "1" ]; then devs=${DEVICES%%,*}; fi
  local log=$OUT/tp${tp}_${graph}_${reduce}_${run}.log
  flock -x "$LOCK_FILE" env HIP_VISIBLE_DEVICES="$DEVICES" SHELF_TP_HIP_GRAPH="$flag" \
    SHELF_TP_REDUCE="$reduce" \
    "$BIN" tg --model-dir "$MODEL" --device "$devs" --tensor-parallel "$tp" \
    --context "$CONTEXT" --tokens "$TOKENS" --warmup "$WARMUP" > "$log" 2>&1
  local status=$?
  local wall gpu argmax sum
  wall=$(grep -o "wall_tokens_per_sec=[0-9.]*" "$log" | tail -1 | cut -d= -f2)
  gpu=$(grep -o "gpu_tokens_per_sec=[0-9.]*" "$log" | tail -1 | cut -d= -f2)
  argmax=$(grep -o "LOGITS argmax=[-0-9]*" "$log" | tail -1 | cut -d= -f2)
  sum=$(grep -o "LOGITS argmax=[-0-9]* max=[-0-9.e]* sum=[-0-9.]*" "$log" | tail -1 |
    sed -n 's/.*sum=//p')
  if [ -z "$wall" ]; then status=1; wall=; gpu=; argmax=; sum=; fi
  echo "$BOOT,$tp,$graph,$reduce,$run,${wall:-},${gpu:-},${argmax:-},${sum:-},$status" >> "$CSV"
  printf "tp=%s graph=%-3s reduce=%-7s run=%s wall=%s gpu=%s status=%s\n" \
    "$tp" "$graph" "$reduce" "$run" "${wall:--}" "${gpu:--}" "$status"
}

for run in $(seq 1 "$RUNS"); do
  for tp in $TP_LIST; do
    for graph in $GRAPH_LIST; do
      for reduce in $REDUCE_LIST; do
        if [ "$tp" = "1" ] && [ "$reduce" = "legacy" ]; then continue; fi
        run_cell "$tp" "$graph" "$reduce" "$run"
      done
    done
  done
done

python3 - "$CSV" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
cells = {}
for r in rows:
    if r["status"] != "0" or not r["wall_tok_s"]:
        continue
    cells.setdefault((r["tp"], r["graph"], r.get("reduce", "default")), []).append(
        float(r["wall_tok_s"]))
print("\ntp graph reduce   n  median_wall_tok_s  min  max")
for (tp, graph, reduce), values in sorted(cells.items()):
    print(f"{tp}  {graph:<3} {reduce:<7} {len(values)}  {statistics.median(values):.2f}  "
          f"{min(values):.2f}  {max(values):.2f}")
PY

echo "csv=$CSV"
