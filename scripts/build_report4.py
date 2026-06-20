#!/usr/bin/env python3
"""Rapor 4 (Atlas) için LaTeX tabloları + arşivleme."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "reports" / "generated"
STATS = ROOT / "data" / "processed" / "stats" / "atlas.json"
CHOICES = list("ABCDE")
SUBJECT_ORDER = ["Türkçe", "Sosyal", "Matematik", "Fen",
                 "Edebiyat-Sosyal1", "Sosyal2"]


def drop_last(s):
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    if lines and lines[-1].endswith("\\\\"):
        lines[-1] = lines[-1][:-2].rstrip()
    return "\n".join(lines) + "\n"


def kunye_table(st):
    # yıl-sınav soru sayıları (sabit yapı)
    rows = [
        "TYT & 40 & 25 & 40 & 20 & --- & --- & 125 \\\\",
        "AYT & --- & --- & 40 & 40 & 40 & 46 & 166 \\\\",
    ]
    return "\n".join(rows)


def overall_table(st):
    d = st["genel_dagilim"]
    cells = " & ".join(f"{d[c]*100:.2f}" for c in CHOICES)
    return f"Genel (8 yıl) & {cells} \\\\"


def subject_table(st):
    by = st["ders_dagilim"]
    rows = []
    for s in SUBJECT_ORDER:
        if s in by:
            cells = " & ".join(f"{by[s][c]*100:.1f}" for c in CHOICES)
            # en çok/en az vurgusu
            rows.append(f"{s} & {cells} \\\\")
    return "\n".join(rows)


def archive():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / "reports" / "archive" / f"report4_{ts}"
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    for f in ["report4.pdf", "report4.tex"]:
        src = ROOT / "reports" / f
        if src.exists():
            shutil.copy(src, dest / f)
    for fig in (ROOT / "reports" / "figures").glob("atlas_*.png"):
        shutil.copy(fig, dest / "figures" / fig.name)
    shutil.copy(STATS, dest / "atlas.json")
    return dest


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATS.read_text("utf-8"))
    (GEN / "atlas_kunye.tex").write_text(drop_last(kunye_table(st)), "utf-8")
    (GEN / "atlas_overall.tex").write_text(drop_last(overall_table(st)), "utf-8")
    (GEN / "atlas_subject.tex").write_text(drop_last(subject_table(st)), "utf-8")
    facts = {
        "n": st["n"], "n_iptal": st["n_iptal"],
        "en_cok": st["en_cok"], "en_az": st["en_az"],
        "chi": round(st["ki_kare"]["stat"], 2),
        "kosegen": round(st["ardisik_kosegen_ort"] * 100, 2),
    }
    (GEN / "facts4.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), "utf-8")
    print("Rapor 4 tabloları üretildi.")
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
