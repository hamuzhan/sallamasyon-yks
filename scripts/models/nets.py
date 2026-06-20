#!/usr/bin/env python3
"""PyTorch model zoo: MLP, CNN, GRU, Transformer.

Görev: bir şık dizisinde, her pozisyondaki şıkkı geçmiş bağlamdan tahmin et.
Dizisel modeller (CNN/GRU/Transformer) nedensel (causal) çalışır:
pozisyon i, yalnızca <i pozisyonlarını görür. Böylece "önceki cevaplar
biliniyor, sonraki tahmin ediliyor" sallamasyon senaryosuna uyar.

MLP ise tablo özelliklerini (ders, soru_no, son 3 şık) kullanır.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 5
BOS = 5  # dizi başı tokeni (causal modeller için)
VOCAB = 6  # A,B,C,D,E,BOS


# ----------------- MLP (tabular) -----------------
class MLP(nn.Module):
    name = "MLP"
    seq_model = False

    def __init__(self, n_subjects: int, hidden: int = 128):
        super().__init__()
        self.sub_emb = nn.Embedding(n_subjects, 8)
        self.prev_emb = nn.Embedding(VOCAB, 8)  # 0..4 + PAD(5)
        in_dim = 8 + 1 + 8 * 3  # subject + qno_norm + prev1,2,3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x):
        # x: [B,5] = [sid, qno_norm, prev1, prev2, prev3]
        sid = x[:, 0].long()
        qno = x[:, 1:2]
        p1, p2, p3 = x[:, 2].long(), x[:, 3].long(), x[:, 4].long()
        feat = torch.cat(
            [self.sub_emb(sid), qno, self.prev_emb(p1),
             self.prev_emb(p2), self.prev_emb(p3)],
            dim=1,
        )
        return self.net(feat)


# ----------------- Causal sequence base -----------------
class _SeqBase(nn.Module):
    seq_model = True

    def __init__(self, n_subjects, d_model=64, max_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, d_model)
        self.sub_emb = nn.Embedding(n_subjects, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.d_model = d_model
        self.head = nn.Linear(d_model, NUM_CLASSES)

    def embed(self, tokens, subject_id):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        e = self.tok_emb(tokens) + self.pos_emb(pos) + self.sub_emb(subject_id).unsqueeze(1)
        return e


class CNN(_SeqBase):
    name = "CNN"

    def __init__(self, n_subjects, d_model=64, max_len=64):
        super().__init__(n_subjects, d_model, max_len)
        # nedensel conv: sol-pad ile gelecek sızıntısını engelle
        self.k = 3
        self.conv1 = nn.Conv1d(d_model, d_model, self.k)
        self.conv2 = nn.Conv1d(d_model, d_model, self.k)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def _causal(self, x, conv):
        x = nn.functional.pad(x, (self.k - 1, 0))  # sadece sol pad
        return conv(x)

    def forward(self, tokens, subject_id):
        e = self.embed(tokens, subject_id).transpose(1, 2)  # [B,d,L]
        h = self.act(self._causal(e, self.conv1))
        h = self.drop(h)
        h = self.act(self._causal(h, self.conv2))
        h = h.transpose(1, 2)  # [B,L,d]
        return self.head(h)


class GRU(_SeqBase):
    name = "GRU"

    def __init__(self, n_subjects, d_model=64, max_len=64):
        super().__init__(n_subjects, d_model, max_len)
        self.rnn = nn.GRU(d_model, d_model, num_layers=2, batch_first=True, dropout=0.3)

    def forward(self, tokens, subject_id):
        e = self.embed(tokens, subject_id)
        h, _ = self.rnn(e)  # nedensel: GRU doğal olarak geçmişe bakar
        return self.head(h)


class Transformer(_SeqBase):
    name = "Transformer"

    def __init__(self, n_subjects, d_model=64, max_len=64, nhead=4, layers=2):
        super().__init__(n_subjects, d_model, max_len)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model,
            dropout=0.3, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc, layers)

    def forward(self, tokens, subject_id):
        e = self.embed(tokens, subject_id)
        L = tokens.size(1)
        mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        h = self.enc(e, mask=mask)  # causal mask
        return self.head(h)


SEQ_MODELS = {"CNN": CNN, "GRU": GRU, "Transformer": Transformer}
