#!/usr/bin/env python3
"""
Sıfırdan tahmin (cold-start) analiz figürleri (Rapor 3).

Üretilenler:
  1. Model karşılaştırma bar grafiği (rastgele + oracle tavan çizgileri).
  2. "En sık şık" yıllar-arası kararsızlık grafiği (frekansın neden
     güvenilmez olduğunu gösterir): train vs 2025 ders-bazlı en sık şık.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_eval, load_train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
RES = ROOT / "reports" / "results_coldstart.json"
CHOICES = list("ABCDE")


def fig_compare(res):
    items = sorted(res["results"].items(), key=lambda x: x[1]["acc"])
    names = [k for k, _ in items]
    accs = [v["acc"] * 100 for _, v in items]
    stds = [v["std"] * 100 for _, v in items]
    colors = []
    for k in names:
        if k == "PermCycle":
            colors.append("#2ca02c")        # yeşil: en iyi
        elif k in ("Dengeleme", "GRU", "Pozisyonel"):
            colors.append("#d62728")        # kırmızı: çökenler
        else:
            colors.append("#4C72B0")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, accs, xerr=stds, color=colors, capsize=3)
    ax.axvline(20, color="gray", ls="--", lw=1, label="Rastgele (%20)")
    ax.axvline(res["oracle"]["acc"] * 100, color="orange", ls=":", lw=1.5,
               label=f"Oracle tavan (%{res['oracle']['acc']*100:.1f})")
    ax.set_xlabel("2025 sıfırdan tahmin doğruluğu (%)")
    ax.set_title("Sıfırdan Tahmin (k=0) Model Karşılaştırması")
    ax.legend()
    for i, a in enumerate(accs):
        ax.text(a + 0.3, i, f"{a:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "coldstart_karsilastirma.png", dpi=130)
    plt.close(fig)


def fig_freq_instability(train, eval_recs):
    """Ders bazlı en sık şık: train vs 2025 — kararsızlığı gösterir."""
    def best_by_subject(recs):
        c = defaultdict(Counter)
        for r in recs:
            c[r["ders"]][r["cevap"]] += 1
        return {d: cc.most_common(1)[0][0] for d, cc in c.items()}

    tb = best_by_subject(train)
    eb = best_by_subject(eval_recs)
    subs = sorted(tb)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(subs))
    # her ders için train ve 2025 en sık şıkkı metin olarak göster
    for i, s in enumerate(subs):
        match = tb[s] == eb.get(s)
        ax.scatter(i, 0, s=300, color="#4C72B0")
        ax.scatter(i, 1, s=300, color="#2ca02c" if match else "#d62728")
        ax.text(i, 0, tb[s], ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, 1, eb.get(s, "-"), ha="center", va="center", color="white", fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Train (2018-24)\nen sık şık", "2025\nen sık şık"])
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=20, ha="right")
    ax.set_title("En Sık Şık Yıllar Arası Kararsız (yeşil=tutar, kırmızı=tutmaz)")
    ax.set_ylim(-0.5, 1.5)
    fig.tight_layout()
    fig.savefig(FIG / "coldstart_frekans_kararsiz.png", dpi=130)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = json.loads(RES.read_text(encoding="utf-8"))
    train = load_train()
    eval_recs = load_eval()
    fig_compare(res)
    fig_freq_instability(train, eval_recs)
    # kararsızlık özeti
    def best_by_subject(recs):
        c = defaultdict(Counter)
        for r in recs:
            c[r["ders"]][r["cevap"]] += 1
        return {d: cc.most_common(1)[0][0] for d, cc in c.items()}
    tb = best_by_subject(train)
    eb = best_by_subject(eval_recs)
    matches = sum(1 for s in tb if tb[s] == eb.get(s))
    print(f"En sık şık tutarlılığı: {matches}/{len(tb)} ders "
          f"(train ile 2025 aynı en sık şıkka sahip)")
    print(f"Figürler -> {FIG}/")


if __name__ == "__main__":
    main()
