"""
Run every experiment and print one verdict table.

  python run_all.py           # full runs
  python run_all.py --quick   # smoke runs (seconds each)
  python run_all.py --clean   # delete caches and downloaded data; keeps code and results/
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
EXPS = sorted(p for p in (ROOT / "experiments").iterdir() if (p / "run.py").exists())
CLEAN = ["__pycache__", ".pytest_cache", "data", ".cache", "checkpoints"]


def clean():
    freed = 0
    for pattern in CLEAN:
        for p in ROOT.rglob(pattern):
            if p.is_dir() and ".git" not in p.parts:
                freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p)
    print(f"cleaned {freed / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    if a.clean:
        return clean()
    rows, ok = [], True
    for exp in EXPS:
        print(f"── {exp.name}")
        r = subprocess.run([sys.executable, "run.py"] + (["--quick"] if a.quick else []), cwd=exp)
        ok &= r.returncode == 0
        vf = exp / "results" / ("quick/verdict.json" if a.quick else "verdict.json")
        v = json.loads(vf.read_text()) if vf.exists() and r.returncode == 0 else {"verdict": "ERROR"}
        rows.append((exp.name, v.get("verdict"), f"{v.get('passed', '-')}/{v.get('total', '-')}",
                     "✨" if v.get("intriguing") else ""))
    print("\nexperiment                         verdict        criteria")
    for name, verdict, pc, star in rows:
        print(f"{name:<34} {verdict:<14} {pc:<8} {star}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
