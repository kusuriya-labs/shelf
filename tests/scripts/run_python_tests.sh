#!/usr/bin/env bash
# Python 面（apps/server）の pytest を ctest から呼ぶ。
#
# 依存（aiohttp / pytest）が無い環境では **skip**（exit 77）にして、
# C++ の CI を Python の有無で赤くしない。使い方:
#   run_python_tests.sh <python> <tests-dir>
set -u

PY="${1:-python3}"
DIR="${2:?usage: run_python_tests.sh <python> <tests-dir>}"

if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
    echo "SKIP: pytest が無い（$PY）"
    exit 77
fi
if ! "$PY" -c "import aiohttp" >/dev/null 2>&1; then
    echo "SKIP: aiohttp が無い（$PY）"
    exit 77
fi

exec "$PY" -m pytest "$DIR" -q
