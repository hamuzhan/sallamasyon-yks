#!/usr/bin/env python3
"""
YKS cevap anahtarı çıkarıcı (PDF -> yapılandırılmış veri).

Strateji:
  - pdftotext -bbox-layout ile her kelimenin gerçek (x, y) koordinatını al.
  - Cevap anahtarı PDF'in SON sayfasındadır.
  - Son sayfadaki tek harfli A-E (ve 'İPTAL') hücrelerinin x-koordinatlarını
    kümeleyerek ders sütunlarını tespit et (watermark gürültüsünden bağımsız).
  - Her sütundaki soru numaralarını (N.) y-koordinatına göre cevaplarla eşle.
  - Sütun başlığını header kelimelerinden (ders adı) türet.

Bu yöntem, -layout tabanlı naif sütun bölmenin başarısız olduğu
AYT 2022 (sütun dağılması) ve watermark çakışması (TYT 2020 q18) gibi
durumları çözer. İptal edilen sorular ('İPTAL') ayrıca işaretlenir.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

VALID = set("ABCDE")
CANCELLED = "İPTAL"

# Her sütundaki ilk header kelimesine göre ders adı eşlemesi.
# (anahtar kelimeler büyük harfe çevrilmiş metinde aranır)
SUBJECT_KEYWORDS = [
    ("TÜRK DİLİ", "Edebiyat-Sosyal1"),
    ("EDEBİYAT", "Edebiyat-Sosyal1"),
    ("TÜRKÇE", "Türkçe"),
    ("SOSYAL BİLİMLER-2", "Sosyal2"),
    ("SOSYAL BİLİMLER-1", "Edebiyat-Sosyal1"),
    ("SOSYAL", "Sosyal"),
    ("TEMEL MATEMATİK", "Matematik"),
    ("MATEMATİK", "Matematik"),
    ("FEN", "Fen"),
]

WORD_RE = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>'
)
QNUM_RE = re.compile(r"^(\d{1,2})\.$")


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


def run_pdftotext_bbox(pdf: Path) -> str:
    res = subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf), "-"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"pdftotext failed for {pdf}: {res.stderr}")
    return res.stdout


def last_page_words(xml: str) -> list[Word]:
    pages = re.split(r"<page ", xml)
    if len(pages) < 2:
        raise RuntimeError("no pages found")
    last = pages[-1]
    out: list[Word] = []
    for m in WORD_RE.finditer(last):
        x0, y0, x1, y1, t = m.groups()
        out.append(Word(float(x0), float(y0), float(x1), float(y1), t.strip()))
    return out


def cluster_columns(xs: list[float], gap: float = 40.0) -> list[float]:
    """x-merkezlerini kümele, her kümenin ortalama merkezini döndür."""
    if not xs:
        return []
    xs = sorted(xs)
    clusters: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] <= gap:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [sum(c) / len(c) for c in clusters]


def subject_for_column(words: list[Word], col_x: float, first_answer_y: float) -> str:
    """Header bölgesindeki (y < ilk cevap) kelimelerden ders adını türet."""
    hdr = [
        w
        for w in words
        if w.y0 < first_answer_y - 2 and abs(w.xc - col_x) < 80 and w.text
    ]
    hdr_text = " ".join(w.text for w in sorted(hdr, key=lambda w: (w.y0, w.x0))).upper()
    for kw, name in SUBJECT_KEYWORDS:
        if kw in hdr_text:
            return name
    return f"Bilinmeyen(x={col_x:.0f})"


def extract(pdf: Path) -> tuple[list[dict], list[str]]:
    """Bir PDF'ten cevap kayıtlarını çıkar. (records, warnings) döndürür."""
    warnings: list[str] = []
    words = last_page_words(run_pdftotext_bbox(pdf))

    # Cevap hücreleri: tek başına A-E harfi olan kelimeler.
    answer_cells = [w for w in words if w.text in VALID]
    if not answer_cells:
        raise RuntimeError(f"{pdf.name}: cevap hücresi bulunamadı")

    first_answer_y = min(w.y0 for w in answer_cells)
    col_centers = cluster_columns([w.xc for w in answer_cells])
    if len(col_centers) not in (4,):
        warnings.append(
            f"{pdf.name}: beklenmeyen sütun sayısı {len(col_centers)} (beklenen 4)"
        )

    # Soru numaraları (N.) -> bir sütuna ata (en yakın merkez).
    # Sadece cevap bölgesindeki (y >= ilk cevap y'si - tolerans) numaraları al;
    # header/yönerge bölgesindeki "1. OTURUM", "2." gibi gürültüyü ele.
    qnums = []
    for w in words:
        m = QNUM_RE.match(w.text)
        if m and w.y0 >= first_answer_y - 6:
            qnums.append((int(m.group(1)), w))

    def nearest_col(x: float) -> int:
        return min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - x))

    # Sütun başına ders adı.
    subjects = [subject_for_column(words, cx, first_answer_y) for cx in col_centers]

    records: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for qno, qw in qnums:
        col = nearest_col(qw.xc)
        col_x = col_centers[col]
        # Bu soru numarasının y'sine ve sütun x'ine en yakın cevabı bul.
        # Önce İPTAL kontrolü (aynı satırda 'İPTAL' kelimesi).
        same_row = [w for w in words if abs(w.yc - qw.yc) < 5 and w.xc > qw.xc - 5]
        cancelled = any(w.text.upper().replace("I", "İ") == CANCELLED for w in same_row)
        # cevap harfi: bu sütunda, bu y'ye en yakın A-E
        cands = [
            w
            for w in answer_cells
            if abs(w.xc - col_x) < 40 and abs(w.yc - qw.yc) < 6
        ]
        if cancelled:
            ans = CANCELLED
        elif cands:
            # y'ye en yakın
            ans = min(cands, key=lambda w: abs(w.yc - qw.yc)).text
        else:
            warnings.append(
                f"{pdf.name}: {subjects[col]} q{qno} için cevap bulunamadı "
                f"(y={qw.yc:.0f}, x={col_x:.0f})"
            )
            continue

        key = (col, qno)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "sinav": "",  # caller doldurur
                "yil": 0,
                "ders": subjects[col],
                "soru_no": qno,
                "cevap": ans,
            }
        )

    # Doğrulama: her ders için soru no 1..N kesintisiz mi?
    by_subject: dict[str, list[int]] = {}
    for r in records:
        by_subject.setdefault(r["ders"], []).append(r["soru_no"])
    for ders, nums in by_subject.items():
        nums_sorted = sorted(nums)
        expected = list(range(1, max(nums_sorted) + 1))
        missing = sorted(set(expected) - set(nums_sorted))
        dup = len(nums) != len(set(nums))
        if missing:
            warnings.append(f"{pdf.name}: {ders} eksik sorular {missing}")
        if dup:
            warnings.append(f"{pdf.name}: {ders} tekrar eden soru no")

    return records, warnings


def parse_filename(pdf: Path) -> tuple[str, int]:
    m = re.match(r"(tyt|ayt)_(\d{4})", pdf.stem)
    if not m:
        raise RuntimeError(f"dosya adı şemaya uymuyor: {pdf.name}")
    return m.group(1).upper(), int(m.group(2))


def write_outputs(records: list[dict], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["yil", "sinav", "ders", "soru_no", "cevap"]
    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fields})
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", default=str(Path(__file__).resolve().parent.parent / "data")
    )
    ap.add_argument("--strict", action="store_true", help="uyarı varsa hata ver")
    args = ap.parse_args()

    root = Path(args.root)
    pdfs = sorted(root.glob("**/raw/*.pdf"))
    if not pdfs:
        print("PDF bulunamadı", file=sys.stderr)
        return 1

    all_records: list[dict] = []
    total_warnings = 0
    print(f"{'DOSYA':22s} {'KAYIT':>6s}  DERS DAĞILIMI")
    for pdf in pdfs:
        sinav, yil = parse_filename(pdf)
        records, warnings = extract(pdf)
        for r in records:
            r["sinav"] = sinav
            r["yil"] = yil

        # processed dizini: raw'ın kardeşi
        out_dir = pdf.parent.parent / "processed"
        write_outputs(records, out_dir, pdf.stem)

        is_eval = "eval" in pdf.parts
        if not is_eval:
            all_records.extend(records)

        dist = {}
        for r in records:
            dist[r["ders"]] = dist.get(r["ders"], 0) + 1
        dist_str = ", ".join(f"{k}:{v}" for k, v in dist.items())
        tag = " [EVAL]" if is_eval else ""
        print(f"{pdf.name:22s} {len(records):>6d}  {dist_str}{tag}")
        for w in warnings:
            print(f"    ! {w}")
            total_warnings += 1

    # master (eval hariç)
    master_dir = root / "processed"
    write_outputs(all_records, master_dir, "all_answers")
    print(f"\nMaster: {len(all_records)} kayıt -> {master_dir}/all_answers.{{csv,json}}")
    print(f"Toplam uyarı: {total_warnings}")

    if args.strict and total_warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
