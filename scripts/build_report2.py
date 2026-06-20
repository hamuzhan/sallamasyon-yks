#!/usr/bin/env python3
"""Rapor 2 için LaTeX tablo parçaları + arşivleme."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / "data" / "processed" / "stats"
GEN = ROOT / "reports" / "generated"
KNOWN = [0, 50, 70, 85, 90]


def imput_table(res) -> str:
    order = ["Rastgele", "GlobalFreq", "Markov", "Dengeleme", "PencereKaçın",
             "Hybrid", "SmartHybrid", "Unified", "GRU", "MaskedBERT"]
    best_per_k = {}
    for k in KNOWN:
        best_per_k[k] = max(
            res["results"], key=lambda m: res["results"][m][f"known_{k}"]["acc"]
        )
    rows = []
    for name in order:
        if name not in res["results"]:
            continue
        cells = []
        for k in KNOWN:
            d = res["results"][name][f"known_{k}"]
            cell = f"{d['acc']*100:.2f}$\\pm${d['std']*100:.1f}"
            if best_per_k[k] == name:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def gain_table(res) -> str:
    rows = []
    for name in ["Markov", "Dengeleme", "PencereKaçın", "Hybrid",
                 "SmartHybrid", "Unified", "GRU", "MaskedBERT"]:
        if name not in res["results"]:
            continue
        cells = []
        for k in KNOWN:
            g = (res["results"][name][f"known_{k}"]["acc"]
                 - res["results"]["Rastgele"][f"known_{k}"]["acc"]) * 100
            # negatif değerleri kırmızı, pozitifleri + ile göster
            if g < 0:
                cells.append(f"\\textcolor{{red}}{{{g:.2f}}}")
            else:
                cells.append(f"+{g:.2f}")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def signal_table(sig) -> str:
    b = sig["dengeleme"]
    g = sig["gap_profili"]
    m2 = sig["markov2"]
    wd = sig["yerel_cesitlilik"]
    rl = sig.get("run_length", {})
    rows = [
        f"Şık dengeleme (std oranı) & {b['gercek_std_ort']:.2f} / {b['rastgele_std_ort']:.2f} = {b['oran']:.2f} & Güçlü \\\\",
        f"Tekrar gap=1 & {g['1']*100:.1f}\\% (rastgele 20\\%) & Güçlü \\\\",
        f"Tekrar gap=2 & {g['2']*100:.1f}\\% & Orta \\\\",
        f"Tekrar gap=3 & {g['3']*100:.1f}\\% & Orta \\\\",
        f"Run-length: 2 üst üste sonrası & {rl.get('2',0)*100:.1f}\\% aynı (rastgele 20\\%) & Güçlü \\\\",
        f"Run-length: 3 üst üste sonrası & {rl.get('3',0)*100:.1f}\\% aynı (asla 4'lü yok) & Güçlü \\\\",
        f"2.\\ derece Markov & in-sample {m2['in_sample_acc']*100:.1f}\\% & Zayıf (ezber) \\\\",
        f"Yerel çeşitlilik (5'li) & {wd['hepsi_farkli_orani']*100:.1f}\\% (rastgele {wd['rastgele_beklenen']*100:.1f}\\%) & Orta \\\\",
    ]
    return "\n".join(rows)


def drop_last(s: str) -> str:
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    if lines and lines[-1].endswith("\\\\"):
        lines[-1] = lines[-1][:-2].rstrip()
    return "\n".join(lines) + "\n"


def archive():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / "reports" / "archive" / f"report2_{ts}"
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    for f in ["report2.pdf", "report2.tex"]:
        src = ROOT / "reports" / f
        if src.exists():
            shutil.copy(src, dest / f)
    for fig in (ROOT / "reports" / "figures").glob("*.png"):
        shutil.copy(fig, dest / "figures" / fig.name)
    for j in ["results_impute.json"]:
        src = ROOT / "reports" / j
        if src.exists():
            shutil.copy(src, dest / j)
    shutil.copytree(GEN, dest / "generated", dirs_exist_ok=True)
    return dest


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    res = json.loads((ROOT / "reports" / "results_impute.json").read_text(encoding="utf-8"))
    sig = json.loads((STATS / "signals.json").read_text(encoding="utf-8"))

    (GEN / "imput_table.tex").write_text(drop_last(imput_table(res)), encoding="utf-8")
    (GEN / "gain_table.tex").write_text(drop_last(gain_table(res)), encoding="utf-8")
    (GEN / "signal_table.tex").write_text(drop_last(signal_table(sig)), encoding="utf-8")

    rnd = res["results"]["Rastgele"]
    best_each = {
        k: max(res["results"], key=lambda m: res["results"][m][f"known_{k}"]["acc"])
        for k in KNOWN
    }
    facts = {
        "trials": res["trials"],
        "best_per_known": best_each,
        "best90_model": best_each[90],
        "best90_gain": round(
            (res["results"][best_each[90]]["known_90"]["acc"]
             - rnd["known_90"]["acc"]) * 100, 2),
        "best50_model": best_each[50],
        "best50_gain": round(
            (res["results"][best_each[50]]["known_50"]["acc"]
             - rnd["known_50"]["acc"]) * 100, 2),
        "balance_ratio": round(sig["dengeleme"]["oran"], 3),
        "runlength_2": round(sig.get("run_length", {}).get("2", 0) * 100, 1)
        if "run_length" in sig else None,
    }
    (GEN / "facts2.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print("Rapor 2 tabloları üretildi.")
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
