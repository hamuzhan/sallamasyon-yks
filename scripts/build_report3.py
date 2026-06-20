#!/usr/bin/env python3
"""Rapor 3 (sıfırdan tahmin) için LaTeX tablosu + arşivleme."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "reports" / "generated"
RES = ROOT / "reports" / "results_coldstart.json"


def result_table(res) -> str:
    rnd = res["results"]["Rastgele"]["acc"]
    order = sorted(res["results"].items(), key=lambda x: -x[1]["acc"])
    best = order[0][0]
    rows = []
    for name, d in order:
        acc = d["acc"] * 100
        std = d["std"] * 100
        accs = f"{acc:.2f}" + (f"$\\pm${std:.1f}" if std else "")
        gain = (d["acc"] - rnd) * 100
        gs = f"\\textcolor{{red}}{{{gain:.2f}}}" if gain < 0 else f"+{gain:.2f}"
        nm = name
        if name == best:
            nm = f"\\textbf{{{name}}}"
            accs = f"\\textbf{{{accs}}}"
        rows.append(f"{nm} & {accs} & {gs} \\\\")
    return "\n".join(rows)


def drop_last(s: str) -> str:
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    if lines and lines[-1].endswith("\\\\"):
        lines[-1] = lines[-1][:-2].rstrip()
    return "\n".join(lines) + "\n"


def archive():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / "reports" / "archive" / f"report3_{ts}"
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    for f in ["report3.pdf", "report3.tex"]:
        src = ROOT / "reports" / f
        if src.exists():
            shutil.copy(src, dest / f)
    for fig in (ROOT / "reports" / "figures").glob("coldstart_*.png"):
        shutil.copy(fig, dest / "figures" / fig.name)
    if RES.exists():
        shutil.copy(RES, dest / "results_coldstart.json")
    return dest


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    res = json.loads(RES.read_text(encoding="utf-8"))
    (GEN / "coldstart_table.tex").write_text(
        drop_last(result_table(res)), encoding="utf-8")
    facts = {
        "best": max(res["results"], key=lambda m: res["results"][m]["acc"]),
        "best_acc": round(max(v["acc"] for v in res["results"].values()) * 100, 2),
        "oracle": round(res["oracle"]["acc"] * 100, 2),
        "oracle_perm": res["oracle"]["perm"],
        "random": round(res["results"]["Rastgele"]["acc"] * 100, 2),
    }
    (GEN / "facts3.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Rapor 3 tablosu üretildi.")
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
