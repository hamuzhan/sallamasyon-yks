#!/usr/bin/env python3
"""
YKS cevap anahtarı keşif analizi.

Üretilenler:
  - Genel / ders bazlı şık dağılımı (A-E %).
  - Ardışık tekrar oranları (aynı şık peş peşe gelme).
  - Pozisyonel olasılık tabloları: P(şık | ders, soru_no).
  - Markov geçiş matrisleri: P(sonraki | önceki), ders bazlı ve genel.
  - matplotlib figürleri (reports/figures/).
  - Sayısal özetler (data/processed/stats/*.json).

İPTAL kayıtları analizden çıkarılır (5-sınıf A-E üzerinde çalışırız).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data" / "processed" / "all_answers.json"
FIG_DIR = ROOT / "reports" / "figures"
STATS_DIR = ROOT / "data" / "processed" / "stats"
CHOICES = list("ABCDE")
CIDX = {c: i for i, c in enumerate(CHOICES)}


def load(path: Path) -> list[dict]:
    recs = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in recs if r["cevap"] in CIDX]


def dist_overall(recs: list[dict]) -> dict[str, float]:
    c = Counter(r["cevap"] for r in recs)
    n = sum(c.values())
    return {ch: c.get(ch, 0) / n for ch in CHOICES}


def dist_by(recs: list[dict], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        groups[str(r[key])][r["cevap"]] += 1
    out = {}
    for g, c in groups.items():
        n = sum(c.values())
        out[g] = {ch: c.get(ch, 0) / n for ch in CHOICES}
    return out


def sequences(recs: list[dict]) -> list[list[str]]:
    """Her (yil, sinav, ders) için soru_no sırasına göre şık dizisi."""
    g: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    for r in recs:
        g[(r["yil"], r["sinav"], r["ders"])].append((r["soru_no"], r["cevap"]))
    seqs = []
    for k, items in g.items():
        items.sort()
        seqs.append([c for _, c in items])
    return seqs


def repeat_stats(seqs: list[list[str]]) -> dict:
    total_adj = 0
    same_adj = 0
    max_run = 0
    run_hist = Counter()
    for s in seqs:
        if not s:
            continue
        run = 1
        for i in range(1, len(s)):
            total_adj += 1
            if s[i] == s[i - 1]:
                same_adj += 1
                run += 1
            else:
                run_hist[run] += 1
                max_run = max(max_run, run)
                run = 1
        run_hist[run] += 1
        max_run = max(max_run, run)
    return {
        "ardisik_ayni_orani": same_adj / total_adj if total_adj else 0.0,
        "max_ardisik_tekrar": max_run,
        "kosu_dagilimi": dict(sorted(run_hist.items())),
        "rastgele_beklenen_ardisik": 0.20,
    }


def markov(seqs: list[list[str]]) -> np.ndarray:
    M = np.zeros((5, 5))
    for s in seqs:
        for a, b in zip(s, s[1:]):
            M[CIDX[a], CIDX[b]] += 1
    row = M.sum(axis=1, keepdims=True)
    row[row == 0] = 1
    return M / row


def positional_table(recs: list[dict]) -> dict:
    """P(şık | ders, soru_no). Döndürür: {ders: {soru_no: [pA..pE]}}."""
    g: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in recs:
        g[r["ders"]][r["soru_no"]][r["cevap"]] += 1
    out = {}
    for ders, qmap in g.items():
        out[ders] = {}
        for q, c in qmap.items():
            n = sum(c.values())
            out[ders][q] = [c.get(ch, 0) / n for ch in CHOICES]
    return out


# ---------- figürler ----------
def fig_overall(d: dict[str, float]):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(CHOICES, [d[c] * 100 for c in CHOICES], color="#4C72B0")
    ax.axhline(20, color="red", ls="--", lw=1, label="Rastgele (%20)")
    ax.set_ylabel("Yüzde (%)")
    ax.set_title("Genel Şık Dağılımı (2018-2024)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dagilim_genel.png", dpi=130)
    plt.close(fig)


def fig_by_subject(dbs: dict[str, dict[str, float]]):
    subs = sorted(dbs)
    x = np.arange(len(subs))
    w = 0.16
    fig, ax = plt.subplots(figsize=(max(7, len(subs) * 1.3), 4))
    for i, ch in enumerate(CHOICES):
        ax.bar(x + (i - 2) * w, [dbs[s][ch] * 100 for s in subs], w, label=ch)
    ax.axhline(20, color="red", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=20, ha="right")
    ax.set_ylabel("Yüzde (%)")
    ax.set_title("Ders Bazlı Şık Dağılımı")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dagilim_ders.png", dpi=130)
    plt.close(fig)


def fig_markov(M: np.ndarray, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(0.35, M.max()))
    ax.set_xticks(range(5))
    ax.set_xticklabels(CHOICES)
    ax.set_yticks(range(5))
    ax.set_yticklabels(CHOICES)
    ax.set_xlabel("Sonraki şık")
    ax.set_ylabel("Önceki şık")
    ax.set_title(title)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.25 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130)
    plt.close(fig)


def fig_dist_by_year(dby: dict[str, dict[str, float]]):
    years = sorted(dby, key=int)
    fig, ax = plt.subplots(figsize=(7, 4))
    for ch in CHOICES:
        ax.plot(years, [dby[y][ch] * 100 for y in years], marker="o", label=ch)
    ax.axhline(20, color="gray", ls="--", lw=1)
    ax.set_ylabel("Yüzde (%)")
    ax.set_xlabel("Yıl")
    ax.set_title("Yıllara Göre Şık Dağılımı")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dagilim_yil.png", dpi=130)
    plt.close(fig)


def chi_square(d: dict[str, float], n: int) -> tuple[float, float]:
    """Düzgün dağılıma karşı ki-kare; (stat, kabaca p<0.05 eşiği 9.49)."""
    exp = n / 5
    stat = sum((d[c] * n - exp) ** 2 / exp for c in CHOICES)
    return stat, 9.488  # df=4, alpha=0.05 kritik değer


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    recs = load(MASTER)
    n = len(recs)
    print(f"Analiz edilen kayıt (İPTAL hariç): {n}")

    overall = dist_overall(recs)
    by_subject = dist_by(recs, "ders")
    by_year = dist_by(recs, "yil")
    by_exam = dist_by(recs, "sinav")
    seqs = sequences(recs)
    reps = repeat_stats(seqs)
    M_all = markov(seqs)
    M_by_subject = {}
    # ders bazlı markov
    g: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        g[r["ders"]][(r["yil"], r["sinav"])].append((r["soru_no"], r["cevap"]))
    for ders, ymap in g.items():
        ss = []
        for _, items in ymap.items():
            items.sort()
            ss.append([c for _, c in items])
        M_by_subject[ders] = markov(ss)

    postable = positional_table(recs)
    chi_stat, chi_crit = chi_square(overall, n)

    # ---- figürler ----
    fig_overall(overall)
    fig_by_subject(by_subject)
    fig_dist_by_year(by_year)
    fig_markov(M_all, "Markov Geçiş (Genel)", "markov_genel.png")
    for ders, M in M_by_subject.items():
        safe = ders.replace("/", "_")
        fig_markov(M, f"Markov Geçiş ({ders})", f"markov_{safe}.png")

    # ---- konsol özeti ----
    print("\nGenel dağılım (%):", {c: round(overall[c] * 100, 2) for c in CHOICES})
    print(f"Ki-kare (düzgün dağılıma karşı): {chi_stat:.2f} "
          f"(kritik {chi_crit:.2f}, df=4) -> "
          f"{'ANLAMLI sapma' if chi_stat > chi_crit else 'sapma anlamsız'}")
    print(f"Ardışık aynı şık oranı: {reps['ardisik_ayni_orani']*100:.2f}% "
          f"(rastgele beklenen %20)")
    print(f"Maksimum ardışık tekrar: {reps['max_ardisik_tekrar']}")

    # ---- stats kaydet ----
    stats = {
        "n": n,
        "genel_dagilim": overall,
        "ders_dagilim": by_subject,
        "yil_dagilim": by_year,
        "sinav_dagilim": by_exam,
        "ardisik": reps,
        "ki_kare": {"stat": chi_stat, "kritik_0.05_df4": chi_crit,
                    "anlamli": chi_stat > chi_crit},
        "markov_genel": M_all.tolist(),
        "markov_ders": {k: v.tolist() for k, v in M_by_subject.items()},
    }
    (STATS_DIR / "analysis.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (STATS_DIR / "positional.json").write_text(
        json.dumps(postable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nİstatistikler -> {STATS_DIR}/")
    print(f"Figürler      -> {FIG_DIR}/")


if __name__ == "__main__":
    main()
