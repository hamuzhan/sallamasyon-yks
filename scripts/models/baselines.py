#!/usr/bin/env python3
"""İstatistiksel baseline modeller (GPU gerektirmez)."""
from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import CHOICES, CIDX, group_sequences  # noqa: E402


class RandomBaseline:
    """Düzgün rastgele tahmin (referans: %20)."""

    name = "Rastgele"

    def fit(self, train):
        self.rng = np.random.default_rng(0)

    def predict_seq(self, subject, qnos, prev_labels):
        return [int(self.rng.integers(0, 5)) for _ in qnos]


class MostFrequent:
    """Her ders için en sık görülen şıkkı her soruya ver."""

    name = "EnSıkŞık"

    def fit(self, train):
        from collections import Counter, defaultdict

        c = defaultdict(Counter)
        for r in train:
            c[r["ders"]][CIDX[r["cevap"]]] += 1
        self.best = {d: cc.most_common(1)[0][0] for d, cc in c.items()}
        self.glob = Counter(CIDX[r["cevap"]] for r in train).most_common(1)[0][0]

    def predict_seq(self, subject, qnos, prev_labels):
        b = self.best.get(subject, self.glob)
        return [b] * len(qnos)


class Positional:
    """P(şık | ders, soru_no) argmax."""

    name = "Pozisyonel"

    def fit(self, train):
        from collections import Counter, defaultdict

        g = defaultdict(lambda: defaultdict(Counter))
        for r in train:
            g[r["ders"]][r["soru_no"]][CIDX[r["cevap"]]] += 1
        self.table = {}
        for d, qm in g.items():
            self.table[d] = {q: cc.most_common(1)[0][0] for q, cc in qm.items()}
        self.glob = Counter(CIDX[r["cevap"]] for r in train).most_common(1)[0][0]

    def predict_seq(self, subject, qnos, prev_labels):
        t = self.table.get(subject, {})
        return [t.get(q, self.glob) for q in qnos]


class Markov:
    """
    P(sonraki | önceki) geçiş matrisi (ders bazlı + genel fallback).
    İlk soru için ders bazlı tekil dağılım argmax'ı kullanılır.
    Tahmin online: tahmin edilen değil GERÇEK önceki şık kullanılır
    (sallamasyon senaryosu: bazı cevaplar bilinir, sonraki tahmin edilir).
    """

    name = "Markov"

    def fit(self, train):
        seqs = group_sequences(train)
        self.M = {}
        self.start = {}
        from collections import defaultdict

        by_sub = defaultdict(list)
        for s in seqs:
            by_sub[s["subject"]].append(s["labels"])
        glob_M = np.ones((5, 5))
        glob_s = np.ones(5)
        for sub, labs in by_sub.items():
            M = np.ones((5, 5))  # Laplace smoothing
            st = np.ones(5)
            for seq in labs:
                if seq:
                    st[seq[0]] += 1
                for a, b in zip(seq, seq[1:]):
                    M[a, b] += 1
                    glob_M[a, b] += 1
            for seq in labs:
                if seq:
                    glob_s[seq[0]] += 1
            self.M[sub] = M / M.sum(1, keepdims=True)
            self.start[sub] = st / st.sum()
        self.glob_M = glob_M / glob_M.sum(1, keepdims=True)
        self.glob_s = glob_s / glob_s.sum()

    def predict_seq(self, subject, qnos, prev_labels):
        """prev_labels[i] = i. sorunun gerçek önceki şıkkı (yoksa None)."""
        M = self.M.get(subject, self.glob_M)
        st = self.start.get(subject, self.glob_s)
        preds = []
        for i in range(len(qnos)):
            if i == 0 or prev_labels[i] is None:
                preds.append(int(st.argmax()))
            else:
                preds.append(int(M[prev_labels[i]].argmax()))
        return preds


ALL_BASELINES = [RandomBaseline, MostFrequent, Positional, Markov]
