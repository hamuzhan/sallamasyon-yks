#!/usr/bin/env python3
"""
Tüm modelleri eğit (2018-2024) ve 2025'te değerlendir.

Değerlendirme senaryosu (sallamasyon):
  Her (ders) dizisinde, modelin her pozisyondaki şıkkı tahmin etmesi istenir.
  Dizisel modeller pozisyon i'yi tahmin ederken gerçek <i şıklarını görür
  (yani "önceki cevaplar biliniyor" varsayımı). Bu, baseline Markov ile
  adil karşılaştırma sağlar.

Sonuçlar: reports/results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_utils import (  # noqa: E402
    CIDX, group_sequences, load_eval, load_train, make_tabular, subject_vocab,
)
from baselines import ALL_BASELINES  # noqa: E402
from nets import BOS, MLP, SEQ_MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def set_seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------- baseline değerlendirme ----------------
def eval_baseline(model, eval_recs) -> dict:
    seqs = group_sequences(eval_recs)
    correct = total = 0
    per_sub = {}
    for s in seqs:
        labels = s["labels"]
        prev = [None] + labels[:-1]  # gerçek önceki şık
        preds = model.predict_seq(s["subject"], s["qnos"], prev)
        c = sum(int(p == l) for p, l in zip(preds, labels))
        correct += c
        total += len(labels)
        ps = per_sub.setdefault(s["subject"], [0, 0])
        ps[0] += c
        ps[1] += len(labels)
    return {
        "name": model.name,
        "type": "baseline",
        "accuracy": correct / total,
        "n": total,
        "per_subject": {k: v[0] / v[1] for k, v in per_sub.items()},
    }


# ---------------- MLP (tabular) ----------------
def train_mlp(train_recs, eval_recs, svocab, epochs=200) -> dict:
    Xtr, ytr = make_tabular(train_recs, svocab)
    Xev, yev = make_tabular(eval_recs, svocab)
    Xtr = torch.tensor(Xtr, device=DEVICE)
    ytr = torch.tensor(ytr, device=DEVICE)
    Xev = torch.tensor(Xev, device=DEVICE)
    yev = torch.tensor(yev, device=DEVICE)

    model = MLP(len(svocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xtr)
        loss = lossf(out, ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xev).argmax(1) == yev).float().mean().item()
                tracc = (out.argmax(1) == ytr).float().mean().item()
            best = max(best, acc)
    model.eval()
    with torch.no_grad():
        pred = model(Xev).argmax(1)
        acc = (pred == yev).float().mean().item()
        tracc = (model(Xtr).argmax(1) == ytr).float().mean().item()
    return {
        "name": "MLP", "type": "nn", "accuracy": acc,
        "train_accuracy": tracc, "best_eval": best, "n": len(yev),
    }


# ---------------- causal sequence modelleri ----------------
def pad_batch(seqs, max_len):
    """tokens girişi: [BOS, l0, l1, ...]; hedef: [l0, l1, ...]. Sol-align, pad=-100."""
    B = len(seqs)
    toks = torch.full((B, max_len), BOS, dtype=torch.long)
    tgts = torch.full((B, max_len), -100, dtype=torch.long)
    subs = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(seqs):
        labs = s["labels"]
        L = len(labs)
        # input token at pos j = label[j-1] (BOS at j=0)
        toks[i, 0] = BOS
        for j in range(1, L):
            toks[i, j] = labs[j - 1]
        for j in range(L):
            tgts[i, j] = labs[j]
        subs[i] = s["_sid"]
    return toks, tgts, subs


def train_seq(ModelCls, train_recs, eval_recs, svocab, epochs=300) -> dict:
    tr = group_sequences(train_recs)
    ev = group_sequences(eval_recs)
    for s in tr + ev:
        s["_sid"] = svocab[s["subject"]]
    max_len = max(len(s["labels"]) for s in tr + ev)

    toks, tgts, subs = pad_batch(tr, max_len)
    etoks, etgts, esubs = pad_batch(ev, max_len)
    toks, tgts, subs = toks.to(DEVICE), tgts.to(DEVICE), subs.to(DEVICE)
    etoks, etgts, esubs = etoks.to(DEVICE), etgts.to(DEVICE), esubs.to(DEVICE)

    model = ModelCls(len(svocab), max_len=max_len).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    def accuracy(tk, tg, sb):
        with torch.no_grad():
            logits = model(tk, sb)
            pred = logits.argmax(-1)
            mask = tg != -100
            return ((pred == tg) & mask).sum().item() / mask.sum().item()

    best = 0.0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(toks, subs)
        loss = lossf(logits.reshape(-1, 5), tgts.reshape(-1))
        loss.backward()
        opt.step()
        if (ep + 1) % 25 == 0:
            model.eval()
            best = max(best, accuracy(etoks, etgts, esubs))
    model.eval()
    acc = accuracy(etoks, etgts, esubs)
    tracc = accuracy(toks, tgts, subs)
    # per-subject
    with torch.no_grad():
        pred = model(etoks, esubs).argmax(-1)
    per = {}
    for i, s in enumerate(ev):
        L = len(s["labels"])
        c = (pred[i, :L].cpu().numpy() == np.array(s["labels"])).sum()
        ps = per.setdefault(s["subject"], [0, 0])
        ps[0] += int(c)
        ps[1] += L
    return {
        "name": ModelCls.name, "type": "nn", "accuracy": acc,
        "train_accuracy": tracc, "best_eval": best, "n": int((etgts != -100).sum()),
        "per_subject": {k: v[0] / v[1] for k, v in per.items()},
    }


def main():
    set_seed()
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_recs = load_train()
    eval_recs = load_eval()
    svocab = subject_vocab(train_recs, eval_recs)
    print(f"Train kayıt: {len(train_recs)}  Eval(2025) kayıt: {len(eval_recs)}")
    print(f"Dersler: {list(svocab)}\n")

    results = []
    N_SEEDS = 10  # NN modelleri için çok-tohumlu (istatistiksel güven)

    # baselines (deterministik; Rastgele için tek tohum yeterli)
    for B in ALL_BASELINES:
        m = B()
        m.fit(train_recs)
        r = eval_baseline(m, eval_recs)
        r["std"] = 0.0
        results.append(r)
        print(f"[baseline] {r['name']:12s} acc={r['accuracy']*100:5.2f}%")

    # NN modelleri: N_SEEDS tohum, ortalama±std
    def run_multi(label, fn):
        accs, tracc, best = [], [], []
        per_sub_acc = {}
        t0 = time.time()
        for seed in range(N_SEEDS):
            set_seed(seed)
            r = fn()
            accs.append(r["accuracy"])
            tracc.append(r["train_accuracy"])
            best.append(r["best_eval"])
            for k, v in r.get("per_subject", {}).items():
                per_sub_acc.setdefault(k, []).append(v)
        res = {
            "name": label, "type": "nn",
            "accuracy": float(np.mean(accs)), "std": float(np.std(accs)),
            "accuracy_max": float(np.max(accs)), "accuracy_min": float(np.min(accs)),
            "train_accuracy": float(np.mean(tracc)),
            "best_eval": float(np.mean(best)),
            "n": r["n"], "seeds": N_SEEDS,
            "per_subject": {k: float(np.mean(v)) for k, v in per_sub_acc.items()},
            "time_s": round(time.time() - t0, 1),
        }
        results.append(res)
        print(f"[nn]       {label:12s} acc={res['accuracy']*100:5.2f}%"
              f"±{res['std']*100:.2f} (max {res['accuracy_max']*100:.1f}, "
              f"train {res['train_accuracy']*100:.1f}%, {res['time_s']}s)")
        return res

    run_multi("MLP", lambda: train_mlp(train_recs, eval_recs, svocab))
    for name, Cls in SEQ_MODELS.items():
        run_multi(name, lambda Cls=Cls: train_seq(Cls, train_recs, eval_recs, svocab))

    out = ROOT / "reports" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSonuçlar -> {out}")


if __name__ == "__main__":
    main()
