"""Scan what the public repository would contain for private material, or copy it to a release tree.

    python tools/release_tree.py --tracked            # scan the files git would commit (tracked + untracked, not ignored)
    python tools/release_tree.py <target dir> [--force]   # also copy exactly those files to <target dir>

The public tree is what `.gitignore` (an allowlist) lets through; `private/`, `runs/`, `own-footage/`,
`third_party/` and `.venv/` never are.  Every text file is scanned for the development footage, local absolute
paths, personal names, e-mail addresses, credentials and verbatim review notes; every file for size.  Exit 0: CLEAN.
Exit 1: the findings are listed (and written to RELEASE_SCAN.json in the target); fix the files or add a reviewed
exception below, then rerun.  Nothing is published or pushed.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEVER = ("private/", "runs/", "own-footage/", "third_party/", "vendor/", ".venv/")
MAX_BYTES = 2_000_000
PATTERNS = {
    "development footage": r"own-footage[\\/]\w|plate\.mp4|r39_fix|mixkit|reporter[-_ ]?(plate|test|full|footage|a\b)",
    "local absolute path": r"[A-Za-z]:[\\/]+Users[\\/]+|[A-Za-z]:[\\/]+Developer[\\/]|/home/[a-z]+/|/Users/[A-Za-z]+/",
    "personal": r"frederik|wessberg|ideanote|@gmail\.|Frede\b",
    "user quote marker": r"\bthe user:|the user \(verbatim\)",
    "e-mail address": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "credential": r"ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}"
                  r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[abp]-[A-Za-z0-9-]{10,}",
    "private folder reference": r"private[\\/](docs|tools|BUILD_STATUS|snapshots|spec)",
}
# reviewed exceptions: (file glob, pattern name, why it is fine)
EXCEPTIONS = [
    ("src/eyeclean/inference/process.py", "development footage",
     "PRIVATE_DIRS = (\"inputs\", \"own-footage\"): a generic guard that refuses to write into user-footage folders"),
    ("tools/release_tree.py", "*", "the scanner's own patterns"),
    ("docs/RELEASE.md", "*", "the checklist names what to scan for"),
    ("uv.lock", "e-mail address", "package metadata"),
    ("*", "e-mail address:noreply@anthropic.com", "commit attribution address, not personal"),
]
TEXT = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ps1", ".sh", ".svg", ".cfg", ".ini", ".lock", ""}


def selected() -> list[Path]:
    if not (ROOT / ".git").exists():
        sys.exit("not a git repository: the public tree is what git would commit (run `git init` first)")
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    files = []
    for rel in sorted(set(filter(None, out.split("\0")))):
        p = ROOT / rel
        if p.is_file():
            files.append(p)
    return files


def excepted(rel: str, what: str, match: str) -> bool:
    for g, w, _ in EXCEPTIONS:
        kind, _, literal = w.partition(":")
        if fnmatch.fnmatch(rel, g) and (kind == "*" or kind == what) and (not literal or literal == match):
            return True
    return False


def scan(files: list[Path]) -> list[dict]:
    findings = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(NEVER):
            findings.append({"file": rel, "line": 0, "kind": "never-public folder", "match": rel.split("/")[0]})
        size = p.stat().st_size
        if size > MAX_BYTES:
            findings.append({"file": rel, "line": 0, "kind": "large file", "match": f"{size / 1e6:.1f} MB"})
        if p.suffix.lower() not in TEXT:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for what, rx in PATTERNS.items():
            for m in re.finditer(rx, text, flags=re.IGNORECASE):
                if excepted(rel, what, m.group(0)):
                    continue
                line = text.count("\n", 0, m.start()) + 1
                findings.append({"file": rel, "line": line, "kind": what, "match": m.group(0)})
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="copy the public tree here (must be outside the working tree)")
    ap.add_argument("--tracked", action="store_true", help="scan only, no copy")
    ap.add_argument("--force", action="store_true", help="replace an existing target directory")
    a = ap.parse_args()
    if not a.tracked and not a.target:
        ap.error("pass --tracked or a target directory")
    files = selected()
    findings = scan(files)
    report = {"files": len(files), "bytes": sum(p.stat().st_size for p in files), "findings": findings,
              "exceptions": [{"glob": g, "kind": w, "why": y} for g, w, y in EXCEPTIONS]}
    where = "(scan only)"
    if a.target:
        target = Path(a.target).resolve()
        if target == ROOT or ROOT in target.parents:
            sys.exit("the target must be outside the working tree")
        if target.exists():
            if not a.force:
                sys.exit(f"{target} exists (use --force to replace it)")
            shutil.rmtree(target)
        for p in files:
            dst = target / p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        (target / "RELEASE_SCAN.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        where = f"-> {target}"
    print(f"{len(files)} files, {report['bytes'] / 1e6:.1f} MB {where}")
    by = {}
    for f in findings:
        by.setdefault((f["file"], f["kind"]), []).append(f"{f['line']}:{f['match']}")
    for (f, k), hits in sorted(by.items()):
        print(f"  {k:24} {f}: {hits[:6]}{' ...' if len(hits) > 6 else ''}")
    print("CLEAN" if not findings else f"{len(findings)} findings in {len({f['file'] for f in findings})} files")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
