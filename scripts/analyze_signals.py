#!/usr/bin/env python3
"""
Genişletilmiş sinyal analizi (Rapor 2).

TÜM analizler YALNIZCA train (2018-2024) üzerinde yapılır. 2025'e dokunulmaz.

Aranan sinyaller:
  1. Şık dengeleme: test-içi şık dağılımının düzgüne yakınlığı
     (gerçek std vs rastgele Monte Carlo std).
  2. Pencere-bazlı tekrar kaçınması: gap=1..6 için aynı-şık oranı.
  3. 2. derece Markov: P(next | prev2, prev1) entropisi ve in-sample isabet.
  4. Yerel çeşitlilik: 5'li kayan pencerede benzersiz şık oranı.

Çıktı: data/processed/stats/signals.json + reports/figures/*.png
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
from data_utils import CHOICES, CIDX, group_sequences, load_train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
STATS_DIR = ROOT / "data" / "processed" / "stats"
RNG = np.random.default_rng(0)


def seq_labels(train):
    """Her (yil,sinav,ders) için şık idx dizisi listesi."""
    return [s["labels"] for s in group_sequences(train)]


# ---------- 1. Şık dengeleme ----------
def balance_signal(seqs, n_mc=500):
    real_stds, rand_stds = [], []
    for s in seqs:
        n = len(s)
        c = Counter(s)
        real_stds.append(float(np.std([c.get(i, 0) for i in range(5)])))
        # aynı uzunlukta rastgele dizilerin ortalama std'i
        acc = []
        for _ in range(max(1, n_mc // 50)):
            r = RNG.integers(0, 5, size=n)
            rc = Counter(r.tolist())
            acc.append(np.std([rc.get(i, 0) for i in range(5)]))
        rand_stds.append(float(np.mean(acc)))
    return {
        "gercek_std_ort": float(np.mean(real_stds)),
        "rastgele_std_ort": float(np.mean(rand_stds)),
        "oran": float(np.mean(real_stds) / np.mean(rand_stds)),
        "gercek_std_list": real_stds,
    }


def balance_by_year(train):
    by_year = defaultdict(list)
    g = defaultdict(list)
    for r in train:
        g[(r["yil"], r["sinav"], r["ders"])].append((r["soru_no"], r["cevap"]))
    for k, items in g.items():
        s = [CIDX[c] for _, c in sorted(items)]
        c = Counter(s)
        by_year[k[0]].append(np.std([c.get(i, 0) for i in range(5)]))
    return {str(y): float(np.mean(v)) for y, v in sorted(by_year.items())}


# ---------- 2. Pencere-bazlı tekrar ----------
def gap_profile(seqs, max_gap=6):
    out = {}
    for gap in range(1, max_gap + 1):
        same = tot = 0
        for s in seqs:
            for i in range(gap, len(s)):
                same += int(s[i] == s[i - gap])
                tot += 1
        out[gap] = same / tot if tot else 0.0
    return out


# ---------- 3. 2. derece Markov ----------
def markov2(seqs):
    ctx = defaultdict(Counter)
    for s in seqs:
        for i in range(2, len(s)):
            ctx[(s[i - 2], s[i - 1])][s[i]] += 1
    # in-sample argmax isabet
    corr = tot = 0
    ent = []
    for s in seqs:
        for i in range(2, len(s)):
            c = ctx[(s[i - 2], s[i - 1])]
            pred = c.most_common(1)[0][0]
            corr += int(pred == s[i])
            tot += 1
    # ortalama koşullu entropi
    for c in ctx.values():
        n = sum(c.values())
        p = np.array([c.get(i, 0) / n for i in range(5)])
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return {
        "in_sample_acc": corr / tot if tot else 0.0,
        "ort_kosullu_entropi": float(np.mean(ent)) if ent else 0.0,
        "max_entropi_log2_5": float(np.log2(5)),
        "n_baglam": len(ctx),
    }


# ---------- 3b. Run-length (iso'nun fikri) ----------
def run_length_signal(seqs):
    """k tane üst üste aynı şık sonrası, bir sonrakinin de aynı olma oranı."""
    from collections import defaultdict as dd
    cont = dd(lambda: [0, 0])
    for s in seqs:
        run = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cont[run][0] += 1
                cont[run][1] += 1
                run += 1
            else:
                cont[run][1] += 1
                run = 1
    return {str(rl): (a / t if t else 0.0)
            for rl, (a, t) in sorted(cont.items()) if t >= 5}


# ---------- 4. Yerel çeşitlilik ----------
def window_diversity(seqs, w=5):
    distinct = tot = 0
    for s in seqs:
        for i in range(w - 1, len(s)):
            win = s[i - w + 1 : i + 1]
            distinct += int(len(set(win)) == w)
            tot += 1
    rand = 1.0
    for k in range(5, 5 - w, -1):
        rand *= k / 5
    return {"hepsi_farkli_orani": distinct / tot if tot else 0.0,
            "rastgele_beklenen": rand, "pencere": w}


# ---------- figürler ----------
def fig_balance(by_year, bal):
    years = list(by_year)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(years, [by_year[y] for y in years], color="#4C72B0", label="Gerçek")
    ax.axhline(bal["rastgele_std_ort"], color="red", ls="--",
               label=f"Rastgele (~{bal['rastgele_std_ort']:.2f})")
    ax.set_ylabel("Test-içi şık sayısı std")
    ax.set_xlabel("Yıl")
    ax.set_title("Şık Dengeleme Sinyali (düşük std = dengeli dağılım)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dengeleme_std.png", dpi=130)
    plt.close(fig)


def fig_gap(gap):
    gaps = list(gap)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gaps, [gap[g] * 100 for g in gaps], marker="o", color="#4C72B0")
    ax.axhline(20, color="red", ls="--", label="Rastgele (%20)")
    ax.set_xlabel("Mesafe (gap)")
    ax.set_ylabel("Aynı şık olma oranı (%)")
    ax.set_title("Pencere-Bazlı Tekrar Kaçınması")
    ax.legend()
    ax.set_ylim(0, 25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gap_tekrar.png", dpi=130)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    train = load_train()
    seqs = seq_labels(train)
    print(f"Train dizi sayısı: {len(seqs)}, toplam cevap: {sum(len(s) for s in seqs)}")

    bal = balance_signal(seqs)
    by_year = balance_by_year(train)
    gap = gap_profile(seqs)
    m2 = markov2(seqs)
    rl = run_length_signal(seqs)
    wd = window_diversity(seqs)

    fig_balance(by_year, bal)
    fig_gap(gap)

    print("\n=== 1. ŞIK DENGELEME ===")
    print(f"  Gerçek std: {bal['gercek_std_ort']:.3f}  "
          f"Rastgele std: {bal['rastgele_std_ort']:.3f}  "
          f"Oran: {bal['oran']:.3f} (düşük = güçlü dengeleme)")
    print("\n=== 2. GAP PROFİLİ (aynı şık %) ===")
    for g, v in gap.items():
        print(f"  gap={g}: {v*100:5.2f}%")
    print("\n=== 3. 2. DERECE MARKOV ===")
    print(f"  In-sample acc: {m2['in_sample_acc']*100:.2f}%  "
          f"Koşullu entropi: {m2['ort_kosullu_entropi']:.3f}/{m2['max_entropi_log2_5']:.3f}")
    print("\n=== 3b. RUN-LENGTH (k üst üste sonrası aynı devam %) ===")
    for k, v in rl.items():
        print(f"  {k} üst üste -> aynı devam: {v*100:5.2f}% (rastgele 20%)")
    print("\n=== 4. YEREL ÇEŞİTLİLİK (5'li pencere) ===")
    print(f"  Hepsi farklı: {wd['hepsi_farkli_orani']*100:.2f}%  "
          f"Rastgele: {wd['rastgele_beklenen']*100:.2f}%")

    out = {
        "kaynak": "train (2018-2024) - 2025 dokunulmadi",
        "dengeleme": {k: v for k, v in bal.items() if k != "gercek_std_list"},
        "dengeleme_yil": by_year,
        "gap_profili": gap,
        "markov2": m2,
        "run_length": rl,
        "yerel_cesitlilik": wd,
    }
    (STATS_DIR / "signals.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nİstatistikler -> {STATS_DIR}/signals.json")
    print(f"Figürler      -> {FIG_DIR}/")


if __name__ == "__main__":
    main()
