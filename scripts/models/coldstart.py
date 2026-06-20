#!/usr/bin/env python3
"""
Sıfırdan tahmin (cold-start, k=0) modelleri.

Senaryo: Hiçbir cevap bilinmiyor; model bir testin TÜM sorularını baştan
tahmin eder. İmputer arayüzüyle uyumludur: predict_masked(subject, n, known)
çağrısında known boştur.

Tüm istatistik/parametreler YALNIZCA train'den öğrenilir (sızıntısız).
"""
from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import CIDX, group_sequences  # noqa: E402

NC = 5
CHOICES = list("ABCDE")


class PositionalArgmax:
    """
    P(şık | ders, soru_no) argmax. Pozisyonel örüntü olup olmadığını test eder.
    (Bulgu: YOK — dağılım düzgüne yakın, bu model rastgelenin altında kalır.)
    """

    name = "Pozisyonel"

    def fit(self, train):
        g = defaultdict(Counter)
        for r in train:
            g[(r["ders"], r["soru_no"])][CIDX[r["cevap"]]] += 1
        self.best = {k: c.most_common(1)[0][0] for k, c in g.items()}
        self.glob = Counter(CIDX[r["cevap"]] for r in train).most_common(1)[0][0]

    def predict_masked(self, subject, n, known):
        # k=0'da qno'yu bilemeyiz; imputer arayüzü qno vermiyor, bu yüzden
        # pozisyon = dizi indeksi+1 varsayılır (testler 1..n sıralı).
        return {p: self.best.get((subject, p + 1), self.glob)
                for p in range(n) if p not in known}


class PermCycle:
    """
    Dengeli + ardarda-tekrarsız dizi üretici. Train'de en yüksek isabeti veren
    ABCDE-permutasyon DÖNGÜSÜNÜ seçer (ör. B,A,E,C,D,B,A,E,...) ve diziye
    pozisyon sırasına göre uygular. Hem dengeleme (her 5'te bir tüm şıklar)
    hem tekrar-kaçınma (ardarda asla aynı) kısıtını otomatik sağlar.

    Permutasyon SEÇİMİ yalnızca train isabetine göre yapılır (sızıntısız).
    """

    name = "PermCycle"

    def fit(self, train):
        seqs = group_sequences(train)

        def score(perm):
            corr = tot = 0
            for s in seqs:
                for i, lab in enumerate(s["labels"]):
                    corr += int(CIDX[perm[i % NC]] == lab)
                    tot += 1
            return corr / tot if tot else 0.0

        self.perm = max(itertools.permutations(CHOICES), key=score)
        self.train_acc = score(self.perm)

    def predict_masked(self, subject, n, known):
        return {p: CIDX[self.perm[p % NC]]
                for p in range(n) if p not in known}


class BalancedShuffle:
    """
    Dengeli ama deterministik-döngü olmayan üretici: her dizide şıkları
    train hedef oranına göre (n/5 adet) bir torbaya koyar ve pozisyon sırasına
    göre, ARDARDA TEKRARDAN kaçınarak yerleştirir. PermCycle'ın daha esnek hali.
    Rastgelelik tohuma bağlıdır; çoklu turda ortalanır.
    """

    name = "BalancedShuffle"

    def fit(self, train):
        c = Counter(CIDX[r["cevap"]] for r in train)
        tot = sum(c.values())
        self.ratio = np.array([c[i] / tot for i in range(NC)])
        self.rng = np.random.default_rng(0)

    def predict_masked(self, subject, n, known):
        # n soruyu hedef orana göre torbaya doldur
        counts = np.round(self.ratio * n).astype(int)
        # toplamı n'e ayarla
        while counts.sum() < n:
            counts[np.argmax(self.ratio)] += 1
        while counts.sum() > n:
            counts[np.argmax(counts)] -= 1
        bag = []
        for i in range(NC):
            bag += [i] * counts[i]
        self.rng.shuffle(bag)
        # ardarda tekrarı azaltacak basit yeniden düzenleme
        out = []
        for x in bag:
            if out and out[-1] == x:
                # swap ile tekrarı boz
                for j in range(len(out) - 1, -1, -1):
                    if out[j] != x and (j == 0 or out[j - 1] != x):
                        out.insert(j, x)
                        break
                else:
                    out.append(x)
            else:
                out.append(x)
        return {p: out[p] for p in range(n) if p not in known}


# Sadece cold-start'a özgü YENİ modeller (mevcutlar eval'de ayrıca eklenir)
COLDSTART_MODELS = [PositionalArgmax, PermCycle, BalancedShuffle]
