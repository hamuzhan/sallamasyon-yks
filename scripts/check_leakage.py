#!/usr/bin/env python3
"""
Veri sızıntısı (data leakage) doğrulama scripti.

Garantiler:
  1. TRAIN (all_answers.json) yalnızca 2018-2024 içerir, 2025 İÇERMEZ.
  2. EVAL yalnızca 2025 içerir.
  3. subject_vocab YALNIZCA train'den türetilir; eval'e bakmaz.
  4. Train ve eval kayıtları (yil,sinav,ders,soru_no) düzeyinde kesişmez.

Çıkış kodu 0 = temiz, 1 = sızıntı tespit edildi.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_eval, load_train, subject_vocab  # noqa: E402


def key(r):
    return (r["yil"], r["sinav"], r["ders"], r["soru_no"])


def main() -> int:
    train = load_train()
    eval_ = load_eval()
    ok = True

    train_years = sorted({r["yil"] for r in train})
    eval_years = sorted({r["yil"] for r in eval_})
    print(f"TRAIN yılları: {train_years}")
    print(f"EVAL  yılları: {eval_years}")

    # 1. 2025 train'de olmamalı
    if 2025 in train_years:
        print("! SIZINTI: 2025 train setinde bulundu")
        ok = False
    else:
        print("✓ 2025 train'de YOK")

    # 2. eval sadece 2025
    if eval_years != [2025]:
        print(f"! UYARI: eval beklenmedik yıllar içeriyor: {eval_years}")
        ok = False
    else:
        print("✓ EVAL sadece 2025")

    # 3. kayıt kesişimi
    tk = {key(r) for r in train}
    ek = {key(r) for r in eval_}
    inter = tk & ek
    if inter:
        print(f"! SIZINTI: {len(inter)} ortak (yil,sinav,ders,soru) kaydı")
        ok = False
    else:
        print("✓ Train/Eval kayıt kesişimi BOŞ")

    # 4. subject_vocab sadece train'den (eval'e özel ders sızmamalı)
    sv = subject_vocab(train)
    train_subs = {r["ders"] for r in train}
    eval_subs = {r["ders"] for r in eval_}
    leaked = eval_subs - train_subs
    print(f"Train dersleri: {sorted(train_subs)}")
    print(f"Vocab anahtarları: {sorted(sv)}")
    if leaked:
        print(f"  Not: eval'de train'de olmayan ders(ler): {sorted(leaked)} "
              f"-> bunlar <UNK>'e düşer (sızıntı değil).")
    if set(sv) - {"<UNK>"} != train_subs:
        print("! SIZINTI: vocab train ders kümesiyle uyuşmuyor")
        ok = False
    else:
        print("✓ subject_vocab yalnızca train derslerinden (+<UNK>)")

    print("\n" + ("SONUÇ: TEMİZ (sızıntı yok)" if ok else "SONUÇ: SIZINTI VAR!"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
