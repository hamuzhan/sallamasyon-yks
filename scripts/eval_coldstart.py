#!/usr/bin/env python3
"""
Sıfırdan tahmin (cold-start, k=0) değerlendirmesi.

Senaryo: Hiçbir cevap bilinmiyor; her test baştan sona tahmin edilir ve
TÜM pozisyonlardaki isabet ölçülür. Sadece k=0'a odaklanır (Rapor 3).

Sızıntı garantileri:
  - Tüm modeller fit() ile yalnızca train (2018-2024) görür.
  - PermCycle/GlobalFreq vb. parametreleri yalnızca train'den seçilir.
  - 2025 yalnızca ölçüm için kullanılır.
  - Oracle üst-sınır AYRICA raporlanır (2025'e bakar; "ulaşılabilir tavan"
    referansıdır, model değildir, açıkça etiketlenir).

Çıktı: reports/results_coldstart.json
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_utils import CIDX, group_sequences, load_eval, load_train, subject_vocab  # noqa: E402
from imputers import (  # noqa: E402
    BalanceImputer, GlobalFreqImputer, MarkovImputer, RandomImputer,
)
from coldstart import COLDSTART_MODELS  # noqa: E402
from masked_net import MaskedGRUImputer, MaskedImputer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHOICES = list("ABCDE")
TRIALS = 1500


def eval_cold(model, seqs, trials, seed=0):
    """k=0: tüm pozisyonları tahmin et, isabeti ölç. trials tur ortalanır."""
    rng = np.random.default_rng(seed)
    # deterministik modeller için 1 tur yeter; rastgele içerenler için trials
    is_random = model.name in ("Rastgele", "BalancedShuffle")
    T = trials if is_random else 1
    per = []
    for _ in range(T):
        corr = tot = 0
        for s in seqs:
            n = len(s["labels"])
            preds = model.predict_masked(s["subject"], n, {})
            for p, pred in preds.items():
                corr += int(pred == s["labels"][p])
                tot += 1
        if tot:
            per.append(corr / tot)
    return float(np.mean(per)), float(np.std(per)) if len(per) > 1 else 0.0


def oracle_permutation(eval_seqs):
    """2025'e bakarak en iyi permutasyon döngüsü (ULAŞILABİLİR TAVAN, model değil)."""
    def score(perm):
        corr = tot = 0
        for s in eval_seqs:
            for i, lab in enumerate(s["labels"]):
                corr += int(CIDX[perm[i % 5]] == lab)
                tot += 1
        return corr / tot
    best = max(itertools.permutations(CHOICES), key=score)
    return score(best), "".join(best)


def main():
    train = load_train()
    eval_recs = load_eval()
    train_seqs = group_sequences(train)
    eval_seqs = group_sequences(eval_recs)
    print(f"Train dizi: {len(train_seqs)}  Eval(2025) dizi: {len(eval_seqs)}")
    print("Senaryo: SIFIRDAN TAHMİN (k=0, hiçbir cevap bilinmiyor)\n")

    svocab = subject_vocab(train)
    models = []

    # mevcut imputer'lar (k=0'da test)
    for Cls in (RandomImputer, GlobalFreqImputer, MarkovImputer, BalanceImputer):
        m = Cls()
        m.fit(train)
        models.append(m)
    # yeni cold-start modelleri
    for Cls in COLDSTART_MODELS:
        m = Cls()
        m.fit(train)
        models.append(m)
    # derin modeller
    mgru = MaskedGRUImputer(svocab)
    print("GRU eğitiliyor (GH200)...")
    mgru.fit(train)
    models.append(mgru)
    mbert = MaskedImputer(svocab)
    print("MaskedBERT eğitiliyor (GH200)...\n")
    mbert.fit(train)
    models.append(mbert)

    results = {}
    print(f"{'Model':16s} {'2025 (k=0)':>12s}")
    print("-" * 30)
    for m in models:
        mean, std = eval_cold(m, eval_seqs, TRIALS)
        results[m.name] = {"acc": mean, "std": std}
        extra = ""
        if m.name == "PermCycle":
            extra = f"  (döngü: {''.join(m.perm)}, train {m.train_acc*100:.1f}%)"
        print(f"{m.name:16s} {mean*100:9.2f}±{std*100:.1f}{extra}")

    # oracle üst sınır (model değil, referans)
    orc_acc, orc_perm = oracle_permutation(eval_seqs)
    print(f"\n[ORACLE üst-sınır] en iyi permutasyon '{orc_perm}': {orc_acc*100:.2f}% "
          f"(2025'e bakar; ulaşılabilir tavan referansı)")

    rnd = results["Rastgele"]["acc"]
    print("\nKAZANÇ (Rastgele'ye karşı, puan):")
    for name, d in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        print(f"  {name:16s} {(d['acc']-rnd)*100:+.2f}")

    out = {
        "senaryo": "sifirdan tahmin (k=0)",
        "kaynak": "fit=train(2018-2024), eval=2025; sizinti yok",
        "trials": TRIALS,
        "results": results,
        "oracle": {"acc": orc_acc, "perm": orc_perm,
                   "not": "2025'e bakar; model degil, ulasilabilir tavan"},
    }
    p = ROOT / "reports" / "results_coldstart.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSonuçlar -> {p}")


if __name__ == "__main__":
    main()
