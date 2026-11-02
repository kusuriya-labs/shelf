#!/usr/bin/env python3
"""Generate the option tables in docs/user from the entrypoint --help output."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Source:
    name: str
    document: Path
    argv: tuple[str, ...]
    python_entry: bool


def sources(build_dir: Path) -> list[Source]:
    compute = str(build_dir / "shelf-compute")
    bench = str(build_dir / "shelf-bench")
    return [
        Source("compute", REPO_ROOT / "docs/user/compute.md", (compute, "--help"), False),
        Source("server", REPO_ROOT / "docs/user/server.md",
               (sys.executable, str(REPO_ROOT / "src/apps/server/shelf_server.py"),
                "--help"), True),
        Source("bench-pp", REPO_ROOT / "docs/user/bench.md", (bench, "pp", "--help"), False),
        Source("bench-tg", REPO_ROOT / "docs/user/bench.md", (bench, "tg", "--help"), False),
    ]


def parse_help(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    current: list[str] | None = None
    current_indent = 0
    for raw in text.splitlines():
        if not raw.strip():
            current = None
            continue
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        if re.match(r"^[A-Za-z0-9][A-Za-z0-9 _/()|-]*:$", stripped):
            current = None
            continue
        if stripped.startswith(("usage:", "Usage:", "run 'shelf-bench")):
            current = None
            continue
        if stripped.startswith("--") or re.match(r"^-h\b", stripped):
            match = re.match(r"^(\S.*?)(?:\s{2,}(.*))?$", stripped)
            if match is None:
                continue
            spec = match.group(1).strip()
            description = (match.group(2) or "").strip()
            row = [spec, description]
            rows.append((spec, description))
            current = row
            current_indent = indent
            continue
        if current is not None and indent > current_indent:
            current[1] = (current[1] + " " + stripped).strip()
            rows[-1] = (current[0], current[1])
    return rows


def render_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| option | 説明 |", "| --- | --- |"]
    for spec, description in rows:
        safe_spec = spec.replace("|", r"\|")
        safe_description = description.replace("|", r"\|")
        lines.append(f"| `{safe_spec}` | {safe_description} |")
    return "\n".join(lines)


def marker_block(name: str, table: str) -> str:
    return f"<!-- cli:{name} -->\n{table}\n<!-- /cli:{name} -->"


def update_document(path: Path, blocks: dict[str, str], check: bool) -> list[str]:
    text = path.read_text()
    failures: list[str] = []
    for name, block in blocks.items():
        pattern = re.compile(
            rf"(?s)<!-- cli:{re.escape(name)} -->.*?<!-- /cli:{re.escape(name)} -->")
        if pattern.search(text) is None:
            failures.append(f"{path}: missing marker cli:{name}")
            continue
        replacement = marker_block(name, block)
        updated = pattern.sub(lambda _: replacement, text, count=1)
        if check:
            if updated != text:
                failures.append(f"{path}: cli:{name} is out of date")
        else:
            text = updated
    if not check and not failures:
        path.write_text(text)
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate docs option tables")
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "build"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sources", choices=("all", "cpp"), default="all")
    args = parser.parse_args(argv)

    build_dir = Path(args.build_dir).resolve()
    blocks_by_document: dict[Path, dict[str, str]] = {}
    failures: list[str] = []
    for source in sources(build_dir):
        if args.sources == "cpp" and source.python_entry:
            continue
        try:
            result = subprocess.run(
                source.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except FileNotFoundError:
            failures.append(f"{source.name}: entrypoint not found: {source.argv[0]}")
            continue
        if result.returncode != 0:
            failures.append(
                f"{source.name}: --help failed with exit {result.returncode}")
            continue
        rows = parse_help(result.stdout)
        if not rows:
            failures.append(f"{source.name}: no options parsed from --help")
            continue
        blocks_by_document.setdefault(source.document, {})[source.name] = render_table(rows)

    for document, blocks in blocks_by_document.items():
        failures.extend(update_document(document, blocks, args.check))

    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print("cli docs are in sync" if args.check else "cli docs updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
