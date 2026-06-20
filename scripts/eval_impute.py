#!/usr/bin/env python3
"""
Sızıntısız imputation (sallamasyon) değerlendirmesi.

Senaryo: Bir testte soruların %known'u BİLİNİR (öğrenci doğru bilir),
kalanı GİZLİDİR (boş -> sallanır). Model gizli pozisyonları tahmin eder.
Sadece gizli pozisyonlardaki doğruluk ölçülür (gerçek sallamasyon isabeti).

Sızıntı garantileri:
  - Modeller fit() ile YALNIZCA train (2018-2024) görür.
  - 2025'te model yalnızca bilinen pozisyonların gerçek şıkkını görür;
    gizli pozların şıkkına asla erişmez.
  - Hybrid alpha'sı TRAIN üzerinde iç-değerlendirme ile seçilir (2025 değil).

Çıktı: reports/results_impute.json + figür
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_utils import group_sequences, load_eval, load_train, subject_vocab  # noqa: E402
from imputers import (  # noqa: E402
    ALL_IMPUTERS, HybridImputer, SmartHybridImputer, UnifiedImputer,
)
from masked_net import MaskedGRUImputer, MaskedImputer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KNOWN_FRACS = [0.0, 0.5, 0.7, 0.85, 0.9]
TRIALS = 1500


def eval_model(model, seqs, known_frac, trials, seed=0):
    """
    Gizli pozlardaki doğruluk (ortalama ± std, `trials` tur üzerinden).

    ÖNEMLİ: `seed` her model için sabit (varsayılan 0) olduğundan, tüm
    modeller AYNI `trials` adet rastgele maskeleme desenini görür (paired
    karşılaştırma). Rastgele baseline dahil hiçbir sonuç tek-atış (one-shot)
    değildir; her ayar `trials` (=300) tur ortalanır.
    """
    rng = np.random.default_rng(seed)
    per_trial = []
    for _ in range(trials):
        corr = tot = 0
        for s in seqs:
            labels = s["labels"]
            n = len(labels)
            idx = np.arange(n)
            rng.shuffle(idx)
            k = int(round(n * known_frac))
            known_pos = set(idx[:k].tolist())
            known = {p: labels[p] for p in known_pos}
            preds = model.predict_masked(s["subject"], n, known)
            for p, pred in preds.items():
                corr += int(pred == labels[p])
                tot += 1
        if tot:
            per_trial.append(corr / tot)
    return float(np.mean(per_trial)), float(np.std(per_trial))


def tune_alpha(ModelCls, train_seqs):
    """
    Bir hibrit modelin alpha'sını TRAIN üzerinde seç (sızıntısız: 2025 yok).
    TRAIN dizilerinde iç-imputation isabetine bakar.
    """
    alphas = np.linspace(0, 1, 11)
    best_a, best_acc = 0.5, -1.0
    for a in alphas:
        m = ModelCls(alpha=a)
        m.fit(TRAIN_RECS)  # istatistik (sayım) train'den
        acc, _ = eval_model(m, train_seqs, known_frac=0.8, trials=40, seed=7)
        if acc > best_acc:
            best_acc, best_a = acc, float(a)
    return best_a, best_acc


def tune_unified(UnifiedCls, train_seqs):
    """
    Unified modelin (run, bal) ağırlıklarını TRAIN'de grid search ile seç
    (sızıntısız). Birden çok bilinen-oranın ortalamasını maksimize eder.
    (İsmail'in notu: Markov/pencere kaldırıldı; sadece run-length + dengeleme.)
    """
    grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    best_w, best_acc = (1.0, 1.0), -1.0
    for wr in grid:
        for wb in grid:
            m = UnifiedCls(w_run=wr, w_bal=wb)
            m.fit(TRAIN_RECS)
            accs = [
                eval_model(m, train_seqs, kf, trials=20, seed=7)[0]
                for kf in (0.6, 0.8)
            ]
            acc = float(np.mean(accs))
            if acc > best_acc:
                best_acc, best_w = acc, (wr, wb)
    return best_w, best_acc


def main():
    global TRAIN_RECS
    TRAIN_RECS = load_train()
    eval_recs = load_eval()
    train_seqs = group_sequences(TRAIN_RECS)
    eval_seqs = group_sequences(eval_recs)
    print(f"Train dizi: {len(train_seqs)}  Eval(2025) dizi: {len(eval_seqs)}")

    # Hibrit modellerin parametrelerini TRAIN'de seç (sızıntısız)
    best_a, best_a_acc = tune_alpha(HybridImputer, train_seqs)
    best_sa, best_sa_acc = tune_alpha(SmartHybridImputer, train_seqs)
    best_uw, best_uw_acc = tune_unified(UnifiedImputer, train_seqs)
    print(f"Hybrid alpha (train-seçimli):      {best_a:.2f} "
          f"(train iç-isabet {best_a_acc*100:.2f}%)")
    print(f"SmartHybrid alpha (train-seçimli): {best_sa:.2f} "
          f"(train iç-isabet {best_sa_acc*100:.2f}%)")
    print(f"Unified ağırlık (run,bal): {best_uw} "
          f"(train iç-isabet {best_uw_acc*100:.2f}%)\n")

    models = []
    for Cls in ALL_IMPUTERS:
        if Cls is UnifiedImputer:
            m = Cls(w_run=best_uw[0], w_bal=best_uw[1])
        else:
            m = Cls()
        m.fit(TRAIN_RECS)
        if isinstance(m, SmartHybridImputer):
            m.set_alpha(best_sa)
        elif isinstance(m, HybridImputer):
            m.set_alpha(best_a)
        models.append(m)

    # PyTorch derin modeller (GH200). svocab YALNIZCA train'den.
    svocab = subject_vocab(TRAIN_RECS)
    # Rapor 1'in GRU mimarisi, imputation versiyonu (çift yönlü)
    mgru = MaskedGRUImputer(svocab)
    print("GRU (imputation) eğitiliyor (GH200)...")
    mgru.fit(TRAIN_RECS)
    models.append(mgru)
    mbert = MaskedImputer(svocab)
    print("MaskedBERT eğitiliyor (GH200)...")
    mbert.fit(TRAIN_RECS)
    models.append(mbert)

    results = defaultdict(dict)
    header = "Model        " + "  ".join(f"k={int(f*100)}%" for f in KNOWN_FRACS)
    print(header)
    print("-" * len(header))
    for m in models:
        row = []
        for f in KNOWN_FRACS:
            mean, std = eval_model(m, eval_seqs, f, TRIALS)
            results[m.name][f"known_{int(f*100)}"] = {"acc": mean, "std": std}
            row.append(f"{mean*100:5.2f}±{std*100:.1f}")
        print(f"{m.name:12s} " + "  ".join(row))

    # kazanç tablosu (Rastgele'ye karşı)
    print("\nKAZANÇ (Rastgele'ye karşı, puan):")
    for name in ["Markov", "Dengeleme", "PencereKaçın", "Hybrid",
                 "SmartHybrid", "Unified", "GRU", "MaskedBERT"]:
        gains = []
        for f in KNOWN_FRACS:
            key = f"known_{int(f*100)}"
            g = (results[name][key]["acc"] - results["Rastgele"][key]["acc"]) * 100
            gains.append(f"k={int(f*100)}%:+{g:.2f}")
        print(f"  {name:10s} " + "  ".join(gains))

    out = {
        "kaynak": "fit=train(2018-2024), eval=2025; sizinti yok",
        "hybrid_alpha": best_a,
        "known_fracs": KNOWN_FRACS,
        "trials": TRIALS,
        "results": results,
    }
    out_path = ROOT / "reports" / "results_impute.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # figür
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [int(f * 100) for f in KNOWN_FRACS]
    for m in models:
        ys = [results[m.name][f"known_{int(f*100)}"]["acc"] * 100 for f in KNOWN_FRACS]
        ax.plot(xs, ys, marker="o", label=m.name)
    ax.axhline(20, color="gray", ls="--", lw=1, label="Rastgele teorik %20")
    ax.set_xlabel("Bilinen cevap oranı (%)")
    ax.set_ylabel("Gizli pozisyon doğruluğu (%)")
    ax.set_title("Sallamasyon (Imputation) Başarısı — 2025")
    ax.legend()
    fig.tight_layout()
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports" / "figures" / "imputation_basari.png", dpi=130)
    plt.close(fig)
    print(f"\nSonuçlar -> {out_path}")


if __name__ == "__main__":
    main()
