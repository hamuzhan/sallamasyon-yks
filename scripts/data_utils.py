#!/usr/bin/env python3
"""Ortak veri yükleme ve özellik üretimi (modeller için)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHOICES = list("ABCDE")
CIDX = {c: i for i, c in enumerate(CHOICES)}


def _load(path: Path) -> list[dict]:
    recs = json.loads(path.read_text(encoding="utf-8"))
    # İPTAL kayıtlarını çıkar (5-sınıf A-E problemi)
    return [r for r in recs if r["cevap"] in CIDX]


def load_train() -> list[dict]:
    return _load(ROOT / "data" / "processed" / "all_answers.json")


def load_eval() -> list[dict]:
    """2025 TYT + AYT (eval seti)."""
    recs = []
    for p in [
        ROOT / "data" / "eval" / "tyt" / "processed" / "tyt_2025.json",
        ROOT / "data" / "eval" / "ayt" / "processed" / "ayt_2025.json",
    ]:
        recs += _load(p)
    return recs


def group_sequences(recs: list[dict]) -> list[dict]:
    """
    (yil, sinav, ders) -> soru_no sıralı şık dizisi.
    Döndürür: [{"key":(yil,sinav,ders), "subject":..., "labels":[idx...]}].
    """
    g: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    for r in recs:
        g[(r["yil"], r["sinav"], r["ders"])].append((r["soru_no"], r["cevap"]))
    out = []
    for k, items in g.items():
        items.sort()
        out.append(
            {
                "key": k,
                "subject": k[2],
                "qnos": [q for q, _ in items],
                "labels": [CIDX[c] for _, c in items],
            }
        )
    return out


# Ders adı -> indeks (özellik için)
def subject_vocab(*recsets: list[dict]) -> dict[str, int]:
    subs = set()
    for recs in recsets:
        subs |= {r["ders"] for r in recs}
    return {s: i for i, s in enumerate(sorted(subs))}


PAD = 5  # önceki-şık özelliği için "yok" değeri (0..4 = A..E, 5 = baş)


def make_tabular(recs: list[dict], svocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    MLP için tablo özellikleri:
      [subject_id, qno_norm, prev1, prev2, prev3]  (prev: önceki şık idx veya PAD)
    Hedef: bu sorunun şık idx'i.
    """
    seqs = group_sequences(recs)
    X, y = [], []
    for s in seqs:
        sid = svocab[s["subject"]]
        labels = s["labels"]
        qnos = s["qnos"]
        maxq = max(qnos) if qnos else 1
        for i, lab in enumerate(labels):
            prev1 = labels[i - 1] if i >= 1 else PAD
            prev2 = labels[i - 2] if i >= 2 else PAD
            prev3 = labels[i - 3] if i >= 3 else PAD
            X.append([sid, qnos[i] / maxq, prev1, prev2, prev3])
            y.append(lab)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
