#!/usr/bin/env python3
"""
BERT-tarzı maskeli imputation modeli (PyTorch, GH200).

Bir testin tüm şık dizisini alır; bilinen pozisyonlar gerçek şık token'ı,
gizli pozisyonlar [MASK] token'ı olur. Bidirectional Transformer gizli
pozisyonları tahmin eder. Hem dengeleme (global) hem komşuluk (sıralı)
sinyalini birlikte öğrenebilir.

Eğitim: train dizilerinde rastgele maskeleme (MLM). 2025'e dokunulmaz.
Imputation arayüzü diğer imputer'larla uyumludur: predict_masked(...).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import group_sequences, sid_of  # noqa: E402

NC = 5
MASK = 5          # [MASK] token
VOCAB = 6         # A,B,C,D,E,MASK
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _MaskedTransformer(nn.Module):
    def __init__(self, n_subjects, d_model=64, nhead=4, layers=3, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.sub = nn.Embedding(n_subjects, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model,
            dropout=0.2, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, NC)
        self.max_len = max_len

    def forward(self, tokens, subj, pad_mask):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.tok(tokens) + self.pos(pos) + self.sub(subj).unsqueeze(1)
        h = self.enc(x, src_key_padding_mask=pad_mask)
        return self.head(h)


class _MaskedGRU(nn.Module):
    """
    Rapor 1'deki GRU mimarisinin imputation (boşluk doldurma) versiyonu.
    Imputation'da bir gizli pozun HEM solunda HEM sağında bilinen cevap
    olabildiği için ÇİFT YÖNLÜ (bidirectional) GRU kullanılır; böylece her
    iki komşuluk da bilgi sağlar. (Rapor 1'deki sıralı tahminde GRU tek yönlü
    /causal idi; orada gelecek bilinmiyordu.)
    """

    def __init__(self, n_subjects, d_model=64, layers=2, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.sub = nn.Embedding(n_subjects, d_model)
        self.rnn = nn.GRU(
            d_model, d_model, num_layers=layers, batch_first=True,
            dropout=0.2, bidirectional=True,
        )
        self.head = nn.Linear(2 * d_model, NC)  # çift yönlü -> 2*d_model
        self.max_len = max_len

    def forward(self, tokens, subj, pad_mask):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.tok(tokens) + self.pos(pos) + self.sub(subj).unsqueeze(1)
        h, _ = self.rnn(x)  # pad_mask GRU'da kullanılmaz; pad pozları zaten
                            # kayıpta -100 ile maskeleniyor
        return self.head(h)


class MaskedImputer:
    """
    Diğer imputer'larla aynı arayüz: fit(train) + predict_masked(...).
    `backbone`: "transformer" (BERT-tarzı) veya "gru" (çift yönlü GRU).
    """

    name = "MaskedBERT"

    def __init__(self, svocab, epochs=400, mask_prob=0.3, seed=0,
                 backbone="transformer"):
        self.svocab = svocab
        self.epochs = epochs
        self.mask_prob = mask_prob
        self.seed = seed
        self.backbone = backbone

    def _build(self):
        if self.backbone == "gru":
            return _MaskedGRU(len(self.svocab), max_len=self.max_len)
        return _MaskedTransformer(len(self.svocab), max_len=self.max_len)

    def fit(self, train_recs):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        seqs = group_sequences(train_recs)
        self.max_len = max(len(s["labels"]) for s in seqs)
        self.model = self._build().to(DEVICE)

        # sabit padlenmiş tensörler
        B = len(seqs)
        labels = torch.full((B, self.max_len), -100, dtype=torch.long)
        subj = torch.zeros(B, dtype=torch.long)
        pad = torch.ones(B, self.max_len, dtype=torch.bool)  # True=pad
        for i, s in enumerate(seqs):
            L = len(s["labels"])
            labels[i, :L] = torch.tensor(s["labels"])
            subj[i] = sid_of(self.svocab, s["subject"])
            pad[i, :L] = False
        labels, subj, pad = labels.to(DEVICE), subj.to(DEVICE), pad.to(DEVICE)

        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-2)
        lossf = nn.CrossEntropyLoss(ignore_index=-100)
        valid = labels != -100
        g = torch.Generator(device=DEVICE).manual_seed(self.seed)
        for ep in range(self.epochs):
            self.model.train()
            # rastgele maske: geçerli pozların mask_prob'u
            mask = (torch.rand(labels.shape, generator=g, device=DEVICE) < self.mask_prob) & valid
            # en az 1 maske garantisi gerekmez; boşsa loss 0 olur
            tokens = labels.clone()
            tokens[~valid] = MASK            # pad -> MASK token (pad_mask zaten gizler)
            tokens[mask] = MASK              # maskelenen pozlar
            tokens_in = tokens.clone()
            tokens_in[valid & ~mask] = labels[valid & ~mask]  # bilinenler gerçek token
            target = torch.full_like(labels, -100)
            target[mask] = labels[mask]
            opt.zero_grad()
            logits = self.model(tokens_in, subj, pad)
            loss = lossf(logits.reshape(-1, NC), target.reshape(-1))
            if torch.isnan(loss):
                continue
            loss.backward()
            opt.step()
        self.model.eval()

    @torch.no_grad()
    def predict_masked(self, subject, n, known):
        tokens = torch.full((1, self.max_len), MASK, dtype=torch.long, device=DEVICE)
        pad = torch.ones(1, self.max_len, dtype=torch.bool, device=DEVICE)
        pad[0, :n] = False
        for p in range(n):
            if p in known:
                tokens[0, p] = known[p]
            else:
                tokens[0, p] = MASK
        subj = torch.tensor([sid_of(self.svocab, subject)], device=DEVICE)
        logits = self.model(tokens, subj, pad)[0]  # [L, NC]
        preds = {}
        for p in range(n):
            if p not in known:
                preds[p] = int(logits[p].argmax().item())
        return preds


class MaskedGRUImputer(MaskedImputer):
    """Rapor 1'in GRU mimarisinin imputation versiyonu (çift yönlü GRU)."""

    name = "GRU"

    def __init__(self, svocab, epochs=400, mask_prob=0.3, seed=0):
        super().__init__(svocab, epochs, mask_prob, seed, backbone="gru")
