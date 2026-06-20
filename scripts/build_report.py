#!/usr/bin/env python3
"""Sonuç grafiği + LaTeX tablo parçaları üretir (rapor için)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results.json"
STATS = ROOT / "data" / "processed" / "stats" / "analysis.json"
FIG_DIR = ROOT / "reports" / "figures"
GEN_DIR = ROOT / "reports" / "generated"
CHOICES = list("ABCDE")


def fig_results(results):
    names = [r["name"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    stds = [r.get("std", 0) * 100 for r in results]
    colors = ["#999999" if r["type"] == "baseline" else "#4C72B0" for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(names, accs, yerr=stds, capsize=4, color=colors)
    ax.axhline(20, color="red", ls="--", lw=1, label="Rastgele (%20)")
    ax.set_ylabel("2025 Doğruluk (%)")
    ax.set_title("Model Karşılaştırması (2025 Evaluation)")
    ax.legend()
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.4, f"{a:.1f}",
                ha="center", fontsize=8)
    ax.set_ylim(0, max(accs + stds) + 6)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sonuc_karsilastirma.png", dpi=130)
    plt.close(fig)


def latex_results_table(results) -> str:
    rows = []
    best = max(r["accuracy"] for r in results)
    for r in results:
        acc = r["accuracy"] * 100
        std = r.get("std", 0) * 100
        accstr = f"{acc:.2f}" + (f"$\\pm${std:.2f}" if std else "")
        tr = r.get("train_accuracy")
        trstr = f"{tr*100:.1f}" if tr is not None else "--"
        name = r["name"]
        if abs(r["accuracy"] - best) < 1e-9:
            name = f"\\textbf{{{name}}}"
            accstr = f"\\textbf{{{accstr}}}"
        typ = "Baseline" if r["type"] == "baseline" else "Sinir Ağı"
        rows.append(f"{name} & {typ} & {accstr} & {trstr} \\\\")
    return "\n".join(rows)


def latex_dist_table(stats) -> str:
    d = stats["genel_dagilim"]
    cells = " & ".join(f"{d[c]*100:.2f}" for c in CHOICES)
    return f"Genel & {cells} \\\\"


def latex_subject_dist(stats) -> str:
    rows = []
    for ders, d in sorted(stats["ders_dagilim"].items()):
        cells = " & ".join(f"{d[c]*100:.1f}" for c in CHOICES)
        rows.append(f"{ders} & {cells} \\\\")
    return "\n".join(rows)


def main():
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))

    fig_results(results)

    # \input edilen tablo gövdelerinde SON satırın `\\`'i kaldırılır.
    # report.tex tarafında \input sonrasında ayrı satırda \\ ve \bottomrule
    # bulunur (booktabs ile \noalign hatasını önleyen tek sağlam düzen).
    def drop_last_rowsep(s: str) -> str:
        lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
        if lines and lines[-1].endswith("\\\\"):
            lines[-1] = lines[-1][:-2].rstrip()
        return "\n".join(lines) + "\n"

    (GEN_DIR / "results_table.tex").write_text(drop_last_rowsep(latex_results_table(results)), encoding="utf-8")
    (GEN_DIR / "dist_table.tex").write_text(drop_last_rowsep(latex_dist_table(stats)), encoding="utf-8")
    (GEN_DIR / "subject_dist_table.tex").write_text(drop_last_rowsep(latex_subject_dist(stats)), encoding="utf-8")

    # rapora gömülecek anahtar sayılar
    facts = {
        "n_train": stats["n"],
        "chi_stat": round(stats["ki_kare"]["stat"], 2),
        "chi_anlamli": stats["ki_kare"]["anlamli"],
        "ardisik": round(stats["ardisik"]["ardisik_ayni_orani"] * 100, 2),
        "max_run": stats["ardisik"]["max_ardisik_tekrar"],
        "best_model": max(results, key=lambda r: r["accuracy"])["name"],
        "best_acc": round(max(r["accuracy"] for r in results) * 100, 2),
        "random_acc": round(
            next(r["accuracy"] for r in results if r["name"] == "Rastgele") * 100, 2
        ),
    }
    (GEN_DIR / "facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Rapor parçaları üretildi:", GEN_DIR)
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
