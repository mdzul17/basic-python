Tempatkan gambar leaderboard yang ingin diproses di folder ini, atau isi otomatis dari Google Drive dengan `--fetch-gdrive`.

Format yang didukung: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, dan `.tiff`.

Contoh:

```bash
cp /path/ke/leaderboard.jpg datasets/leaderboard.jpg
python3 ocr_leaderboard.py
```

Fetch dari Google Drive:

```bash
python3 ocr_leaderboard.py --fetch-gdrive --limit 10
```
