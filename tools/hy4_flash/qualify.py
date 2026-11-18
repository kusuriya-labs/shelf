import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open


def execute(command, log):
    with log.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"実機検査に失敗しました: {log}")
    return log.read_text()


def sequence(log):
    return list(map(int, re.search(r"^GENERATED_IDS=([0-9,]+)$", log, re.MULTILINE).group(1).split(",")))


def main():
    p = argparse.ArgumentParser(description="Hy4-flash の EP=3 一致・予約・実効速度を検証する")
    for field in ("executable", "body", "engram", "draft", "placement", "reference", "calibration", "results"):
        p.add_argument(field, type=Path)
    args = p.parse_args()
    visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    if visible != "GPU-a8e9ddefa2d60f55,GPU-08b2ddcbd6e6b36c,GPU-76a08c022586fed6":
        raise ValueError("locked.py ep3 経由で実行してください")
    args.results.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="ep3-", dir=args.results))
    common = ["--model-dir", args.body, "--engram-dir", args.engram, "--placement", args.placement]
    def run(mode, name, options, dfly):
        if max(os.getloadavg()) >= 8:
            raise RuntimeError("load average が 8 以上です")
        return execute([args.executable.resolve(), "hy4", mode, *common, *options,
                        *(["--dfly-dir", args.draft] if dfly else [])], output / (name + ".log"))
    with safe_open(args.reference / "logits.safetensors", framework="np") as f:
        expected = f.get_tensor("logits")[-1].astype(np.float32)
        tokens = f.get_tensor("input_ids").tolist()
    prompt = output/"ids.txt"
    prompt.write_text(" ".join(map(str,tokens)))
    dump = output/"prefill.f32"
    flags = ["--input-ids-file",prompt,"--max-seq-len","128","--max-new-tokens","24"]
    base = sequence(run("generate","target",[*flags,"--dump-logits",dump],False))
    actual = np.fromfile(dump,dtype="<f4")
    error = float(np.linalg.norm(actual-expected) / np.linalg.norm(expected))
    if not np.isfinite(error) or error > 0.1 or base[0] != int(expected.argmax()):
        raise ValueError(f"CPU logits と不一致: 相対 L2={error}, 保存先={output}")
    if sequence(run("generate","dfly",flags,True)) != base:
        raise ValueError("DFly と通常生成の greedy 列が一致しません")
    sample = [*flags,"--temperature","0.8","--seed","20261119"]
    if sequence(run("generate","sample-target",sample,False)) != sequence(run("generate","sample-dfly",sample,True)):
        raise ValueError("同一 seed の target サンプルと DFly 生成列が一致しません")
    allocation = run("reserve","reserve",["--max-seq-len","2097152"],True)
    reserved = int(re.search(r"^reserved_tokens=(\d+)$",allocation,re.MULTILINE).group(1))
    if reserved not in (1048576,2097152):
        raise ValueError("1Mi / 2Mi の実予約が失敗しました")
    with safe_open(args.calibration,framework="np") as f:
        ids = f.get_tensor("input_ids").reshape(-1)[:2048].tolist()
    if len(ids) != 2048:
        raise ValueError("pp2048 の入力が不足しています")
    prompt.write_text(" ".join(map(str,ids)))
    bench = run("generate","bench",["--input-ids-file",prompt,"--max-seq-len","2304","--max-new-tokens","128",
                                    "--repeats","4"],True)
    rows = re.findall(r"prefill_tokens=(\d+) prefill_s=([\d.]+) decode_tokens=(\d+) decode_s=([\d.]+)",bench)
    if len(rows) != 4 or any(int(n)!=2048 or int(t)<1 for n,_,t,_ in rows):
        raise ValueError("ベンチ結果の反復数または token 数が不正です")
    pp = [int(n)/float(s) for n,s,_,_ in rows[1:]]
    tg = [int(t)/float(s) for _,_,t,s in rows[1:]]
    result = dict(logits_relative_l2=error,reserved_tokens=reserved,pp2048=pp,decode=tg,
                  prefill_median=statistics.median(pp),decode_median=statistics.median(tg))
    (output/"results.json").write_text(json.dumps(result,indent=2))
    print(output,result,flush=True)
    if result["prefill_median"] < 500 or result["decode_median"] < 40:
        raise SystemExit("スループット目標未達")


if __name__ == "__main__":
    main()
