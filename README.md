# sallamasyon-yks

YKS (Yükseköğretim Kurumları Sınavı) TYT ve AYT sınavlarının cevap
anahtarlarındaki yapısal örüntülerin analizi ve "sallamasyon" (boş soruları
istatistiksel olarak işaretleme) stratejilerinin değerlendirilmesi.

**Yazarlar:** Hamza Yiğit Kültür, İsmail Güler

Proje, 2018–2025 arası resmî YKS sınav kitapçıklarının cevap anahtarlarını
PDF'lerden çıkarır, istatistiksel olarak inceler ve hem klasik hem derin
öğrenme (PyTorch, NVIDIA GH200) modelleriyle tahmin denemeleri yapar. Tüm
çalışma dört rapor halinde belgelenmiştir.

## Raporlar

İşlenmiş PDF raporlar [`releases/`](releases/) altındadır:

1. **Rapor 1 — Model Zoo ve Şık Tahmini:** Veri çıkarımı, keşifsel analiz,
   sıralı tahmin için baseline'lar + MLP/CNN/GRU/Transformer. En iyi: GRU.
2. **Rapor 2 — Şık Dengeleme ve Imputation:** Asıl güçlü sinyal olan *şık
   dengeleme*nin keşfi ve sızıntısız boşluk-doldurma değerlendirmesi.
   Bilinen oran %90'da rastgeleye karşı +13 puana kadar kazanç.
3. **Rapor 3 — Sıfırdan Tahmin:** Hiçbir cevap bilinmeden (k=0) tüm cevap
   anahtarının kestirimi. En iyi: dengeli-tekrarsız permutasyon döngüsü
   (PermCycle).
4. **Rapor 4 — Cevap Anahtarı Atlası:** Model içermeyen, salt betimsel
   atlas: sekiz yılın şık dağılımı, ısı haritaları, PCA embedding ve yapı
   analizi.

## Temel Bulgular

- Şık dağılımı düzgüne çok yakındır (en çok D ~%21, en az E ~%19; ki-kare
  anlamsız) — "hep bir şık" işe yaramaz.
- ÖSYM aynı şıkkı ardarda nadiren kullanır (~%13, rastgele %20); 4'lü tekrar
  hiç yoktur.
- Her test içinde 5 şık olağanüstü dengeli dağıtılır (std ~1.2 vs rastgele
  ~2.2). Bu **dengeleme** sinyali, bilinen cevaplar arttıkça boş soruları
  doldurmada en büyük avantajı sağlar.
- Sıfırdan tahmin neredeyse imkansızdır (tavan ~%25); asıl kazanç en az birkaç
  cevabın bilindiği durumlardan gelir.

## Klasör Yapısı

```
data/
├── ayt/{raw,processed}/      # AYT ham PDF (2018-2024) + işlenmiş CSV/JSON
├── tyt/{raw,processed}/      # TYT ham PDF (2018-2024) + işlenmiş CSV/JSON
├── eval/{ayt,tyt}/...        # 2025 sınavları (bağımsız değerlendirme seti)
└── processed/
    ├── all_answers.{csv,json}   # birleşik cevap anahtarı (2018-2024)
    └── stats/                   # analiz çıktıları (analysis, signals, atlas)

scripts/
├── extract_answers.py        # PDF -> yapılandırılmış cevap anahtarı (bbox)
├── check_leakage.py          # veri sızıntısı doğrulaması (2025 izolasyonu)
├── analyze.py                # genel keşifsel analiz
├── analyze_signals.py        # dengeleme/gap/run-length sinyalleri
├── analyze_atlas.py          # betimsel atlas + infografikler (Rapor 4)
├── analyze_coldstart.py      # sıfırdan tahmin analizi (Rapor 3)
├── train.py                  # sıralı tahmin model zoo eğitimi (Rapor 1)
├── eval_impute.py            # imputation değerlendirmesi (Rapor 2)
├── eval_coldstart.py         # sıfırdan tahmin değerlendirmesi (Rapor 3)
├── build_report{,2,3,4}.py   # LaTeX tabloları + arşivleme
└── models/
    ├── baselines.py          # Rastgele, EnSıkŞık, Pozisyonel, Markov
    ├── nets.py               # MLP, CNN, GRU, Transformer (sıralı)
    ├── imputers.py           # Dengeleme, Hybrid, SmartHybrid, Unified
    ├── coldstart.py          # PermCycle, BalancedShuffle, Pozisyonel
    └── masked_net.py         # BERT-tarzı / GRU maskeli imputer (PyTorch)

reports/
├── report{,2,3,4}.tex        # LaTeX kaynakları
├── figures/                  # üretilen grafikler (PNG)
├── generated/                # otomatik üretilen LaTeX tabloları
└── archive/                  # zaman damgalı rapor arşivleri

releases/                     # adlandırılmış, yayımlanmış PDF raporlar
checkpoints/                  # eğitilmiş model ağırlıkları (.pt)
```

## Veri Çıkarımı

Cevap anahtarları, her PDF'in son sayfasındaki çok sütunlu ve üzeri filigranlı
tablodan `pdftotext -bbox-layout` ile koordinat-tabanlı olarak çıkarılır. Bu
yöntem, naif metin çıkarımının başarısız olduğu sütun dağılması ve filigran
çakışması durumlarını çözer. Çıkarılan veri şeması: `yil, sinav, ders,
soru_no, cevap`. İptal edilen sorular ayrıca işaretlenir.

## Yeniden Üretim

```bash
# 1. PDF'lerden cevap anahtarlarını çıkar
python3 scripts/extract_answers.py

# 2. Sızıntı kontrolü (2025'in eğitime karışmadığını doğrula)
python3 scripts/check_leakage.py

# 3. Analizler
python3 scripts/analyze.py            # Rapor 1 keşif
python3 scripts/analyze_signals.py    # Rapor 2 sinyaller
python3 scripts/analyze_coldstart.py  # Rapor 3
python3 scripts/analyze_atlas.py      # Rapor 4 atlas

# 4. Model eğitimi/değerlendirmesi (GPU önerilir)
python3 scripts/train.py              # Rapor 1 model zoo
python3 scripts/eval_impute.py        # Rapor 2 imputation
python3 scripts/eval_coldstart.py     # Rapor 3 sıfırdan tahmin

# 5. Raporları derle (LaTeX tabloları + xelatex)
python3 scripts/build_report.py  && (cd reports && xelatex report.tex)
python3 scripts/build_report2.py && (cd reports && xelatex report2.tex)
python3 scripts/build_report3.py && (cd reports && xelatex report3.tex)
python3 scripts/build_report4.py && (cd reports && xelatex report4.tex)
```

### Bağımlılıklar

- Python 3 (`numpy`, `pandas`, `matplotlib`, `torch`)
- `pdftotext` (poppler-utils)
- `xelatex` (TeX Live; Türkçe karakter desteği için)

## Veri Adlandırma

Tüm ham PDF dosyaları `{sınav}_{yıl}.pdf` şemasını izler
(`tyt_2018.pdf` … `ayt_2025.pdf`). 2025 sınavları, model değerlendirmesinde
bağımsız test seti olarak `eval/` altında ayrılmıştır.

## Metodoloji Notları

- **Sızıntısızlık:** Tahmin yapan tüm raporlarda (1–3) modeller yalnızca
  2018–2024 ile eğitilir; 2025 dokunulmamış test setidir
  (`check_leakage.py` ile doğrulanır). Rapor 4 betimsel olduğundan 2025 dahil
  edilir.
- **Tur sayısı:** İmputation/sıfırdan değerlendirmelerde rastgele maskeleme
  1500 tur ortalanır; sinir ağları 10 tohumla eğitilir. Sonuçlar ortalama ±
  standart sapma olarak raporlanır.

## Lisans

Bu proje [GNU General Public License v3.0](LICENSE) ile lisanslanmıştır.
