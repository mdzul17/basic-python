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

Program `ocr_leaderboard.py` membaca semua gambar di folder `datasets/`, melakukan OCR, lalu menulis hasil gabungan ke `output.json`. Dataset juga bisa di-fetch langsung dari Google Drive.

## Persiapan

Install Tesseract OCR di sistem:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ind
```

Install dependency Python untuk preprocessing gambar dan download Google Drive:

```bash
python3 -m pip install -r requirements.txt
```

## Cara menjalankan

1. Simpan gambar leaderboard ke folder `datasets/`.
2. Jalankan program:

```bash
python3 ocr_leaderboard.py
```

Untuk mengambil dataset dari Google Drive terlebih dahulu:

```bash
python3 ocr_leaderboard.py --fetch-gdrive
```

Default Google Drive folder:

```text
https://drive.google.com/drive/folders/1e8cx33AYnAZOehlh7FnJWZNiFEmDJ6R7
```

Untuk percobaan awal agar tidak memproses semua gambar sekaligus:

```bash
python3 ocr_leaderboard.py --fetch-gdrive --limit 10 --workers 2
```

Untuk debugging akurasi OCR, tambahkan `--debug-ocr`. Output JSON akan menyertakan:

- `raw_text`: raw OCR dari kandidat terbaik per gambar.
- `raw_ocr_attempts`: raw OCR dari semua crop/preprocessing yang dicoba.

```bash
python3 ocr_leaderboard.py --fetch-gdrive --limit 5 --workers 2 -l eng --debug-ocr
```

## Google Colab / Google Cloud Shell

Jalankan command berikut di cell Colab atau terminal Cloud Shell:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ind
python3 -m pip install -r requirements.txt
python3 ocr_leaderboard.py --fetch-gdrive --limit 10 --workers 2
```

Jika hasil sample sudah sesuai, hapus `--limit 10` untuk memproses semua gambar:

```bash
python3 ocr_leaderboard.py --fetch-gdrive --workers 2
```

Output akan tersimpan di `output.json` dengan struktur `images` untuk hasil per gambar dan `entries` sebagai gabungan semua baris leaderboard. Setiap item di `entries` berisi `rank`, `nama_pemain`, dan `point`.
