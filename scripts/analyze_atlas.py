#!/usr/bin/env python3
"""
YKS Cevap Anahtarı Atlası — betimsel istatistik + infografikler (Rapor 4).

Model/tahmin YOK. Tamamen keşifsel/betimsel. TÜM yıllar (2018-2025) birlikte
kullanılır: bu rapor öğrenme/tahmin yapmadığından 2025'i dışarıda tutmak için
bir sebep yoktur; tersine, en eksiksiz betimsel resmi vermek için dahil edilir.

Üretilenler:
  - data/processed/stats/atlas.json (tüm sayısal özetler)
  - reports/figures/atlas_*.png (infografikler)
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
from data_utils import CHOICES, CIDX  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
STATS = ROOT / "data" / "processed" / "stats"

# Tutarlı renk paleti: her şıkka sabit renk
CHOICE_COLORS = {
    "A": "#4C72B0",  # mavi
    "B": "#DD8452",  # turuncu
    "C": "#55A868",  # yeşil
    "D": "#C44E52",  # kırmızı
    "E": "#8172B3",  # mor
}
SUBJECT_ORDER = ["Türkçe", "Sosyal", "Matematik", "Fen",
                 "Edebiyat-Sosyal1", "Sosyal2"]


def load_all() -> list[dict]:
    recs = json.loads(
        (ROOT / "data" / "processed" / "all_answers.json").read_text("utf-8"))
    for p in [ROOT / "data" / "eval" / "tyt" / "processed" / "tyt_2025.json",
              ROOT / "data" / "eval" / "ayt" / "processed" / "ayt_2025.json"]:
        recs += json.loads(p.read_text("utf-8"))
    return [r for r in recs if r["cevap"] in CIDX]


def dist(recs) -> dict:
    c = Counter(r["cevap"] for r in recs)
    n = sum(c.values())
    return {ch: c.get(ch, 0) / n for ch in CHOICES}


def dist_by(recs, key) -> dict:
    g = defaultdict(Counter)
    for r in recs:
        g[str(r[key])][r["cevap"]] += 1
    out = {}
    for k, c in g.items():
        n = sum(c.values())
        out[k] = {ch: c.get(ch, 0) / n for ch in CHOICES}
    return out


# ---------------- infografikler ----------------
def fig_overall(d):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    vals = [d[c] * 100 for c in CHOICES]
    bars = ax.bar(CHOICES, vals, color=[CHOICE_COLORS[c] for c in CHOICES])
    ax.axhline(20, color="black", ls="--", lw=1, label="Düzgün dağılım (%20)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}",
                ha="center", fontweight="bold")
    ax.set_ylabel("Yüzde (%)")
    ax.set_title("Genel Şık Dağılımı (2018–2025, tüm sınavlar)")
    ax.set_ylim(0, max(vals) + 3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "atlas_genel.png", dpi=140)
    plt.close(fig)


def fig_heatmap(by, title, fname, row_order=None):
    rows = row_order or sorted(by)
    M = np.array([[by[r][c] * 100 for c in CHOICES] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 0.55 * len(rows) + 1.5))
    im = ax.imshow(M, cmap="RdYlBu_r", aspect="auto", vmin=15, vmax=25)
    ax.set_xticks(range(5))
    ax.set_xticklabels(CHOICES)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(5):
            ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                    color="black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Yüzde (%)")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=140)
    plt.close(fig)


def fig_year_trend(by_year):
    years = sorted(by_year, key=int)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for c in CHOICES:
        ax.plot(years, [by_year[y][c] * 100 for y in years], marker="o",
                color=CHOICE_COLORS[c], label=c, lw=2)
    ax.axhline(20, color="gray", ls="--", lw=1)
    ax.set_ylabel("Yüzde (%)")
    ax.set_xlabel("Yıl")
    ax.set_title("Şıkların Yıllara Göre Seyri")
    ax.legend(ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "atlas_yil_trend.png", dpi=140)
    plt.close(fig)


def fig_tyt_ayt(by_exam):
    x = np.arange(5)
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - w / 2, [by_exam["TYT"][c] * 100 for c in CHOICES], w,
           label="TYT", color="#4C72B0")
    ax.bar(x + w / 2, [by_exam["AYT"][c] * 100 for c in CHOICES], w,
           label="AYT", color="#C44E52")
    ax.axhline(20, color="black", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CHOICES)
    ax.set_ylabel("Yüzde (%)")
    ax.set_title("TYT ve AYT Şık Dağılımı Karşılaştırması")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "atlas_tyt_ayt.png", dpi=140)
    plt.close(fig)


def fig_markov(recs):
    g = defaultdict(list)
    for r in recs:
        g[(r["yil"], r["sinav"], r["ders"])].append((r["soru_no"], r["cevap"]))
    M = np.zeros((5, 5))
    for k, items in g.items():
        s = [c for _, c in sorted(items)]
        for a, b in zip(s, s[1:]):
            M[CIDX[a], CIDX[b]] += 1
    M = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=0.3)
    ax.set_xticks(range(5)); ax.set_xticklabels(CHOICES)
    ax.set_yticks(range(5)); ax.set_yticklabels(CHOICES)
    ax.set_xlabel("Sonraki şık"); ax.set_ylabel("Önceki şık")
    ax.set_title("Ardışık Geçiş Matrisi (tüm veri)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.22 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "atlas_markov.png", dpi=140)
    plt.close(fig)
    return float(np.trace(M) / 5)  # köşegen ortalaması


def fig_gap_run(recs):
    g = defaultdict(list)
    for r in recs:
        g[(r["yil"], r["sinav"], r["ders"])].append((r["soru_no"], r["cevap"]))
    seqs = [[c for _, c in sorted(v)] for v in g.values()]
    # gap profili
    gap = {}
    for d in range(1, 7):
        same = tot = 0
        for s in seqs:
            for i in range(d, len(s)):
                same += int(s[i] == s[i - d]); tot += 1
        gap[d] = same / tot
    # run-length
    cont = defaultdict(lambda: [0, 0])
    for s in seqs:
        run = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cont[run][0] += 1; cont[run][1] += 1; run += 1
            else:
                cont[run][1] += 1; run = 1
    runs = {rl: (a / t if t else 0) for rl, (a, t) in sorted(cont.items()) if t >= 5}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(list(gap), [gap[d] * 100 for d in gap], marker="o",
            color="#4C72B0", lw=2)
    a1.axhline(20, color="red", ls="--", lw=1, label="Rastgele (%20)")
    a1.set_xlabel("Mesafe (gap)"); a1.set_ylabel("Aynı şık (%)")
    a1.set_title("Pencere-Bazlı Tekrar Kaçınması"); a1.set_ylim(0, 25); a1.legend()
    rk = list(runs)
    a2.bar([str(k) for k in rk], [runs[k] * 100 for k in rk], color="#C44E52")
    a2.axhline(20, color="black", ls="--", lw=1, label="Rastgele (%20)")
    a2.set_xlabel("Üst üste aynı şık sayısı"); a2.set_ylabel("Sonraki de aynı (%)")
    a2.set_title("Run-Length: Tekrar Olasılığı Düşüşü"); a2.legend()
    fig.tight_layout()
    fig.savefig(FIG / "atlas_gap_run.png", dpi=140)
    plt.close(fig)
    return gap, runs


def fig_pca_embedding(by_subject, by_year):
    """
    Ders ve yıl 'şık parmak izlerini' (5B vektör) 2B'ye indir (numpy SVD/PCA).
    Benzer dağılımlı ders/yıllar birbirine yakın düşer.
    """
    labels, vecs, kinds = [], [], []
    for s in SUBJECT_ORDER:
        if s in by_subject:
            labels.append(s); kinds.append("ders")
            vecs.append([by_subject[s][c] for c in CHOICES])
    for y in sorted(by_year, key=int):
        labels.append(y); kinds.append("yıl")
        vecs.append([by_year[y][c] for c in CHOICES])
    X = np.array(vecs)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T  # 2B izdüşüm

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for (x, y), lab, kind in zip(P, labels, kinds):
        col = "#C44E52" if kind == "ders" else "#4C72B0"
        mk = "s" if kind == "ders" else "o"
        ax.scatter(x, y, color=col, marker=mk, s=90,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(str(lab), (x, y), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    var = (S ** 2 / (S ** 2).sum())[:2] * 100
    ax.set_xlabel(f"Bileşen 1 (%{var[0]:.0f} varyans)")
    ax.set_ylabel(f"Bileşen 2 (%{var[1]:.0f} varyans)")
    ax.set_title("Şık Dağılımı 'Embedding' Haritası (PCA, 5B→2B)\n"
                 "kırmızı kare=ders, mavi daire=yıl; yakınlık=benzer dağılım")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "atlas_embedding.png", dpi=140)
    plt.close(fig)


def fig_balance(recs):
    """Test-içi denge: gerçek vs rastgele şık-sayısı std (yıl bazlı)."""
    g = defaultdict(list)
    for r in recs:
        g[(r["yil"], r["sinav"], r["ders"])].append(r["cevap"])
    by_year = defaultdict(list)
    rng = np.random.default_rng(0)
    rand_all = []
    for k, cevs in g.items():
        c = Counter(cevs)
        by_year[k[0]].append(np.std([c.get(ch, 0) for ch in CHOICES]))
        n = len(cevs)
        rr = [np.std(list(Counter(rng.integers(0, 5, n).tolist()).get(i, 0)
                              for i in range(5))) for _ in range(20)]
        rand_all.append(np.mean(rr))
    years = sorted(by_year, key=int)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(years, [np.mean(by_year[y]) for y in years], color="#55A868",
           label="Gerçek")
    ax.axhline(np.mean(rand_all), color="red", ls="--", lw=1.5,
               label=f"Rastgele beklenti (~{np.mean(rand_all):.2f})")
    ax.set_ylabel("Test-içi şık sayısı std")
    ax.set_xlabel("Yıl")
    ax.set_title("Şık Dengesi: Gerçek Sınavlar Düzgüne Çok Yakın")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "atlas_denge.png", dpi=140)
    plt.close(fig)


def chi_square(d, n):
    exp = n / 5
    stat = sum((d[c] * n - exp) ** 2 / exp for c in CHOICES)
    return stat, 9.488


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    recs = load_all()
    n = len(recs)
    overall = dist(recs)
    by_year = dist_by(recs, "yil")
    by_subject = dist_by(recs, "ders")
    by_exam = dist_by(recs, "sinav")
    chi, crit = chi_square(overall, n)

    fig_overall(overall)
    fig_heatmap(by_year, "Yıl × Şık Dağılımı (%)", "atlas_heatmap_yil.png",
                row_order=sorted(by_year, key=int))
    fig_heatmap(by_subject, "Ders × Şık Dağılımı (%) — 'Şık Parmak İzleri'",
                "atlas_heatmap_ders.png",
                row_order=[s for s in SUBJECT_ORDER if s in by_subject])
    fig_year_trend(by_year)
    fig_tyt_ayt(by_exam)
    diag = fig_markov(recs)
    gap, runs = fig_gap_run(recs)
    fig_pca_embedding(by_subject, by_year)
    fig_balance(recs)

    # künye sayıları
    counts = defaultdict(int)
    for r in recs:
        counts[(r["yil"], r["sinav"])] += 1
    ipt = json.loads(
        (ROOT / "data" / "processed" / "all_answers.json").read_text("utf-8"))
    ipt += json.loads((ROOT / "data" / "eval" / "tyt" / "processed"
                       / "tyt_2025.json").read_text("utf-8"))
    n_iptal = sum(1 for r in ipt if r["cevap"] not in CIDX)

    stats = {
        "kapsam": "2018-2025 (8 yil) TYT+AYT, model yok, salt betimsel",
        "n": n,
        "n_iptal": n_iptal,
        "genel_dagilim": overall,
        "en_cok": max(overall, key=overall.get),
        "en_az": min(overall, key=overall.get),
        "ki_kare": {"stat": chi, "kritik": crit, "anlamli": chi > crit},
        "yil_dagilim": by_year,
        "ders_dagilim": by_subject,
        "sinav_dagilim": by_exam,
        "ardisik_kosegen_ort": diag,
        "gap_profili": gap,
        "run_length": runs,
    }
    (STATS / "atlas.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), "utf-8")
    print(f"Atlas: {n} cevap (8 yıl), İPTAL={n_iptal}")
    print(f"Genel: en çok {stats['en_cok']} ({overall[stats['en_cok']]*100:.2f}%), "
          f"en az {stats['en_az']} ({overall[stats['en_az']]*100:.2f}%)")
    print(f"Ki-kare: {chi:.2f} ({'anlamlı' if chi>crit else 'anlamsız'})")
    print(f"Ardışık köşegen ort: {diag*100:.2f}%")
    print(f"Figürler -> {FIG}/atlas_*.png")


if __name__ == "__main__":
    main()
