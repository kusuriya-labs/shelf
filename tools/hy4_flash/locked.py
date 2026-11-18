import argparse
import fcntl
import os
import subprocess
from contextlib import ExitStack


DEVICES = {
    "r9700": ("/tmp/r9700.lock", "GPU-a8e9ddefa2d60f55"),
    "v620": ("/tmp/v620.lock", "GPU-08b2ddcbd6e6b36c"),
    "v620-76": ("/tmp/v620-76a08.lock", "GPU-76a08c022586fed6"),
}


def run(ranks, command):
    if not ranks or len(set(ranks)) != len(ranks) or not command:
        raise ValueError("重複のない GPU 一覧と実行コマンドが必要です")
    if max(os.getloadavg()) >= 8:
        raise RuntimeError("load average が 8 以上です")
    with ExitStack() as stack:
        for name in DEVICES:
            if name in ranks:
                lock = stack.enter_context(open(DEVICES[name][0], "a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if max(os.getloadavg()) >= 8:
            raise RuntimeError("lock 取得時の load average が 8 以上です")
        env = os.environ.copy()
        env["ROCR_VISIBLE_DEVICES"] = ",".join(DEVICES[r][1] for r in ranks)
        return subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="共有 lock を保持し、指定 GPU で検証を実行する")
    parser.add_argument("devices", choices=["ep3", "v620-pair", *DEVICES])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ranks = {"ep3": list(DEVICES), "v620-pair": ["v620", "v620-76"]}.get(args.devices, [args.devices])
    run(ranks, args.command)
