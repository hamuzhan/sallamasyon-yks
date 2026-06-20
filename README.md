# sallamasyon-yks

YKS (Yükseköğretim Kurumları Sınavı) TYT ve AYT sınavlarına ait ham sınav
PDF'lerinden oluşan veri seti.

## Klasör Yapısı

```
data/
├── ayt/
│   ├── raw/         # AYT ham sınav PDF'leri (2018-2024)
│   └── processed/   # İşlenmiş AYT verisi
├── tyt/
│   ├── raw/         # TYT ham sınav PDF'leri (2018-2024)
│   └── processed/   # İşlenmiş TYT verisi
└── eval/
    ├── ayt/
    │   ├── raw/         # Değerlendirme AYT PDF'i (2025)
    │   └── processed/
    └── tyt/
        ├── raw/         # Değerlendirme TYT PDF'i (2025)
        └── processed/
```

## Dosya Adlandırma

Tüm PDF dosyaları `{sınav}_{yıl}.pdf` şemasını izler:

- `ayt_2018.pdf` … `ayt_2024.pdf`
- `tyt_2018.pdf` … `tyt_2024.pdf`
- `eval/` altında 2025 yılı sınavları değerlendirme amaçlı ayrılmıştır.

## Lisans

Bu proje [GNU General Public License v3.0](LICENSE) ile lisanslanmıştır.
