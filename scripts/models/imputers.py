#!/usr/bin/env python3
"""
Imputation (boşluk doldurma) modelleri — gerçek "sallamasyon" senaryosu.

Arayüz:
  fit(train_recs)                     # SADECE train'den öğrenir
  predict_masked(subject, n, known)   # known: {pos: label}, gizli pozları tahmin

Hiçbir model gizli (maskelenmiş) cevapların dağılımına erişemez; yalnızca
o testin bilinen cevaplarını ve train'den öğrendiği global istatistikleri
kullanır. Bu, sızıntısızdır.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import CIDX, group_sequences  # noqa: E402

NC = 5


class RandomImputer:
    name = "Rastgele"

    def fit(self, train):
        self.rng = np.random.default_rng(0)

    def predict_masked(self, subject, n, known):
        return {p: int(self.rng.integers(0, NC)) for p in range(n) if p not in known}


class GlobalFreqImputer:
    """Train'deki ders-bazlı genel şık frekansına göre argmax (sabit tahmin)."""

    name = "GlobalFreq"

    def fit(self, train):
        c = defaultdict(Counter)
        for r in train:
            c[r["ders"]][CIDX[r["cevap"]]] += 1
        self.best = {d: cc.most_common(1)[0][0] for d, cc in c.items()}
        self.glob = Counter(CIDX[r["cevap"]] for r in train).most_common(1)[0][0]

    def predict_masked(self, subject, n, known):
        b = self.best.get(subject, self.glob)
        return {p: b for p in range(n) if p not in known}


class BalanceImputer:
    """
    ŞIK DENGELEME modeli. Her testte 5 şıkkın ~eşit dağıldığı (train'de
    doğrulanmış) kısıtını sömürür: bilinen cevapların sayımından yola çıkıp,
    hedef bütçeye (n/5) göre en "eksik" şıkları gizli pozlara dağıtır.

    Train'den öğrenilen: yalnızca hedef oran vektörü (global şık dağılımı).
    Gizli cevaplara erişim: YOK.
    """

    name = "Dengeleme"

    def fit(self, train):
        # global şık oranı (neredeyse düzgün; yine de train'den)
        c = Counter(CIDX[r["cevap"]] for r in train)
        tot = sum(c.values())
        self.target_ratio = np.array([c[i] / tot for i in range(NC)])

    def predict_masked(self, subject, n, known):
        seen = Counter(known.values())
        # hedef sayı: oran * n
        target = self.target_ratio * n
        deficit = target - np.array([seen.get(i, 0) for i in range(NC)])
        preds = {}
        for p in sorted(set(range(n)) - set(known)):
            ch = int(np.argmax(deficit))
            preds[p] = ch
            deficit[ch] -= 1  # bu tahmini de bütçeden düş
        return preds


class MarkovImputer:
    """
    1. derece Markov, imputation modunda. Gizli pozun komşularını (varsa
    bilinen) kullanarak en olası şıkkı seçer. Hem önceki hem sonraki bilinen
    komşudan gelen olasılıkları çarpar.
    """

    name = "Markov"

    def fit(self, train):
        seqs = group_sequences(train)
        M = np.ones((NC, NC))  # Laplace
        for s in seqs:
            for a, b in zip(s["labels"], s["labels"][1:]):
                M[a, b] += 1
        self.M = M / M.sum(1, keepdims=True)
        c = Counter(l for s in seqs for l in s["labels"])
        tot = sum(c.values())
        self.prior = np.array([c[i] / tot for i in range(NC)])

    def predict_masked(self, subject, n, known):
        preds = {}
        for p in range(n):
            if p in known:
                continue
            score = self.prior.copy()
            if p - 1 in known:
                score = score * self.M[known[p - 1]]
            if p + 1 in known:
                score = score * self.M[:, known[p + 1]]
            preds[p] = int(np.argmax(score))
        return preds


class HybridImputer:
    """
    Dengeleme + Markov birleşimi. Markov komşu skoru ile dengeleme açığını
    ağırlıklı toplar. Ağırlık (alpha) TRAIN üzerinde iç-CV ile seçilir.
    """

    name = "Hybrid"

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, train):
        self.bal = BalanceImputer()
        self.bal.fit(train)
        self.mk = MarkovImputer()
        self.mk.fit(train)

    def set_alpha(self, a):
        self.alpha = a

    def predict_masked(self, subject, n, known):
        seen = Counter(known.values())
        target = self.bal.target_ratio * n
        deficit = target - np.array([seen.get(i, 0) for i in range(NC)])
        preds = {}
        for p in sorted(set(range(n)) - set(known)):
            mscore = self.mk.prior.copy()
            if p - 1 in known:
                mscore = mscore * self.mk.M[known[p - 1]]
            if p + 1 in known:
                mscore = mscore * self.mk.M[:, known[p + 1]]
            mscore = mscore / mscore.sum()
            # dengeleme skoru: açığı normalize et
            d = deficit - deficit.min()
            bscore = d / d.sum() if d.sum() > 0 else np.ones(NC) / NC
            score = self.alpha * bscore + (1 - self.alpha) * mscore
            ch = int(np.argmax(score))
            preds[p] = ch
            deficit[ch] -= 1
        return preds


class WindowAvoidImputer:
    """
    "Akıllı rastgele" (iso'nun fikri). Gizli poz p için, yakın penceredeki
    (gap=1,2,3) BİLİNEN komşu şıkları cezalandırır: mesafe arttıkça ceza azalır.
    Tekrar kaçınmasının komşuluğun ötesine yayıldığı (gap=1:%13, gap=2:%17,
    gap=3:%17) bulgusunu doğrudan sömürür.

    Ceza ağırlıkları TRAIN gap profilinden türetilir: w_g = (0.20 - p_g)/0.20,
    yani şıkın o mesafede ne kadar "az tekrarlandığı" kadar ceza.
    """

    name = "PencereKaçın"

    def fit(self, train):
        seqs = group_sequences(train)
        # gap profili (aynı şık olma oranı) -> ceza ağırlığı
        self.w = {}
        for gap in (1, 2, 3):
            same = tot = 0
            for s in seqs:
                lab = s["labels"]
                for i in range(gap, len(lab)):
                    same += int(lab[i] == lab[i - gap])
                    tot += 1
            p_same = same / tot if tot else 0.20
            # rastgele 0.20'nin ne kadar altındaysa o kadar güçlü ceza (>=0)
            self.w[gap] = max(0.0, (0.20 - p_same) / 0.20)
        c = Counter(l for s in seqs for l in s["labels"])
        tot = sum(c.values())
        self.prior = np.array([c[i] / tot for i in range(NC)])

    def _score(self, n, known, p):
        score = self.prior.copy()
        for gap in (1, 2, 3):
            for q in (p - gap, p + gap):
                if q in known:
                    score[known[q]] *= (1.0 - self.w[gap])
        s = score.sum()
        return score / s if s > 0 else np.ones(NC) / NC

    def predict_masked(self, subject, n, known):
        return {
            p: int(np.argmax(self._score(n, known, p)))
            for p in range(n) if p not in known
        }


class SmartHybridImputer:
    """
    Dengeleme + pencere-kaçınma birleşimi (en güçlü aday).
    Dengeleme bütçesini takip ederken, pencere-kaçınma yakın komşu şıkları
    cezalandırır. alpha TRAIN'de seçilir.
    """

    name = "SmartHybrid"

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, train):
        self.bal = BalanceImputer()
        self.bal.fit(train)
        self.win = WindowAvoidImputer()
        self.win.fit(train)

    def set_alpha(self, a):
        self.alpha = a

    def predict_masked(self, subject, n, known):
        seen = Counter(known.values())
        target = self.bal.target_ratio * n
        deficit = target - np.array([seen.get(i, 0) for i in range(NC)])
        preds = {}
        for p in sorted(set(range(n)) - set(known)):
            wscore = self.win._score(n, known, p)
            d = deficit - deficit.min()
            bscore = d / d.sum() if d.sum() > 0 else np.ones(NC) / NC
            score = self.alpha * bscore + (1 - self.alpha) * wscore
            ch = int(np.argmax(score))
            preds[p] = ch
            deficit[ch] -= 1
        return preds


class UnifiedImputer:
    """
    iso/İsmail'in sadeleştirilmiş "akıllı sallayıcı" fikri. İSMAİL'İN NOTU
    uyarınca Markov komşu-bakması ve pencere-kaçınma KALDIRILDI: bunların tek
    amacı "ardarda aynı şık gelmesin" idi; bu zaten run-length ile (iki yönlü)
    yakalanıyor. Geriye İKİ BAĞIMSIZ sinyal kalır:

      (1) Run-length (iki yönlü): bir gizli pozun solunda VE sağında kaç tane
          üst üste aynı şık var? Train'de öğrenilen "k tane üst üste sonrası
          aynı devam etme" oranı o şıkka uygulanır (2 üst üste -> %6, 3 -> %0;
          yani o şıkkı baskılar/yasaklar). Hem önceki hem sonraki bilinen
          komşuyu kapsadığı için eski Markov+pencere işlevini de görür.
      (2) Dengeleme bütçesi: o ana kadarki şık sayımına göre eksik şıkları
          öne çıkar (test-içi ~eşit dağılım kısıtı).

    Tüm istatistikler TRAIN'den öğrenilir; gizli cevaplara erişim yoktur.
    """

    name = "Unified"

    def __init__(self, w_run=1.0, w_bal=1.0):
        self.w = dict(run=w_run, bal=w_bal)

    def fit(self, train):
        seqs = group_sequences(train)
        labs_all = [s["labels"] for s in seqs]

        # global prior
        c = Counter(l for lab in labs_all for l in lab)
        tot = sum(c.values())
        self.prior = np.array([c[i] / tot for i in range(NC)])

        # Run-length: k üst üste sonrası "aynı devam" oranı
        cont = defaultdict(lambda: [0, 0])
        for lab in labs_all:
            run = 1
            for i in range(1, len(lab)):
                if lab[i] == lab[i - 1]:
                    cont[run][0] += 1
                    cont[run][1] += 1
                    run += 1
                else:
                    cont[run][1] += 1
                    run = 1
        self.run_same = {}  # run_len -> P(devam aynı)
        for rl, (a, t) in cont.items():
            self.run_same[rl] = a / t if t else 0.0

        # dengeleme hedef oranı
        self.target_ratio = self.prior.copy()

    def _run_side(self, known, preds, p, step):
        """p'den `step` yönünde (sol=-1, sağ=+1) ardışık aynı şık run'u."""
        def val(q):
            if q in known:
                return known[q]
            return preds.get(q)
        first = val(p + step)
        if first is None:
            return 0, None
        run = 1
        q = p + 2 * step
        while 0 <= q and val(q) == first:
            run += 1
            q += step
        return run, first

    def predict_masked(self, subject, n, known):
        seen = Counter(known.values())
        target = self.target_ratio * n
        deficit = target - np.array([seen.get(i, 0) for i in range(NC)])
        preds = {}
        for p in sorted(set(range(n)) - set(known)):
            score = np.ones(NC)

            # (1) Run-length (iki yönlü): sol ve sağdaki ardışık run'a göre
            #     ilgili şıkkı baskıla. Komşu-bakma işlevini de kapsar.
            run_factor = np.ones(NC)
            for step in (-1, +1):
                run, ch = self._run_side(known, preds, p, step)
                if ch is not None and run >= 1:
                    # bu yönde `run` tane varsa, o şıkka p_same(run)/0.20 ağırlık
                    p_same = self.run_same.get(run, 0.0)
                    run_factor[ch] *= max(1e-3, p_same / 0.20)
            score *= run_factor ** self.w["run"]

            # (2) Dengeleme bütçesi
            d = deficit - deficit.min()
            bal = (d / d.sum()) if d.sum() > 0 else np.ones(NC) / NC
            bal = bal + 1e-3
            score *= bal ** self.w["bal"]

            ch = int(np.argmax(score))
            preds[p] = ch
            deficit[ch] -= 1
        return preds


ALL_IMPUTERS = [
    RandomImputer, GlobalFreqImputer, MarkovImputer, BalanceImputer,
    WindowAvoidImputer, HybridImputer, SmartHybridImputer, UnifiedImputer,
]
