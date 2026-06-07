# Basic Python
Basic knowledge python using notebook

--------------
Several notebook files I used for telling peope about basic python and data science. Created in Bahasa Indonesia.

Still maintain to get it better and can be used by other people.

# Table of content
- Logical
- Looping
- Function and method
- Class
- Array, list, and arithmatic operation
- Linear regression algorithm example
- Decision tree algorithm example
- SKLearn library for data science

# OCR Leaderboard

Program `ocr_leaderboard.py` membaca semua gambar di folder `datasets/`, melakukan OCR, lalu menulis hasil gabungan ke `output.json`.

## Persiapan

Install Tesseract OCR di sistem:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ind
```

Install dependency Python opsional untuk preprocessing gambar:

```bash
python3 -m pip install -r requirements.txt
```

## Cara menjalankan

1. Simpan gambar leaderboard ke folder `datasets/`.
2. Jalankan program:

```bash
python3 ocr_leaderboard.py
```

Output akan tersimpan di `output.json` dengan struktur `images` untuk hasil per gambar dan `entries` sebagai gabungan semua baris leaderboard. Setiap item di `entries` berisi `rank`, `nama_pemain`, dan `point`.
