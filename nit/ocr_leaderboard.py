"""
OCR Leaderboard Extractor
=========================
Ambil peringkat, nama player, dan power dari screenshot leaderboard MLBB.

Pipeline:
  1. Deteksi panel tabel (tepi + warna biru semi-transparan)
  2. Deteksi baris dari angka Power di kolom kanan
  3. OCR per kolom (Peringkat | Player | Power) per baris
  4. Batch: proses semua gambar di datasets/ -> hasil.json

Cara pakai:
    python ocr_leaderboard.py
    python ocr_leaderboard.py --input datasets --json hasil.json --debug
    python ocr_leaderboard.py path/to/satu-gambar.jpeg
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import easyocr
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_INPUT_DIR = Path("datasets")
DEFAULT_OUTPUT_JSON = Path("hasil.json")


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def box_metrics(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return {
        "cx": sum(xs) / 4.0,
        "cy": sum(ys) / 4.0,
        "x_left": min(xs),
        "x_right": max(xs),
        "y_top": min(ys),
        "y_bottom": max(ys),
        "height": max(ys) - min(ys),
        "width": max(xs) - min(xs),
    }


def is_power_number(text):
    cleaned = re.sub(r"[.,\s]", "", text)
    return cleaned.isdigit() and 3 <= len(cleaned) <= 6


def parse_int(text):
    cleaned = re.sub(r"[^0-9]", "", text)
    return int(cleaned) if cleaned else None


def preprocess_for_ocr(bgr, scale=2):
    """Tingkatkan kontras teks putih/kuning di background gelap."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(up)


def preprocess_player_text(bgr, scale=3):
    """
    Preprocessing khusus kolom player: mask avatar/flag/rank bleed di kiri,
    upscale lebih besar agar simbol kecil (™, ©, ツ, ❄) terbaca.
    """
    h, w = bgr.shape[:2]
    masked = bgr.copy()
    masked[:, : max(1, int(w * 0.28))] = 0

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(up)


# ---------------------------------------------------------------------------
# Deteksi panel tabel (tepi + warna)
# ---------------------------------------------------------------------------
def detect_table_panel(img):
    """
    Temukan ROI panel leaderboard di sisi kanan gambar.
    Kombinasi: mask warna biru + Canny edge + kontur terbesar.
    """
    h, w = img.shape[:2]
    right = img[:, int(w * 0.38):]
    rh, rw = right.shape[:2]

    hsv = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)
    # biru gelap semi-transparan panel tabel
    mask_blue = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([130, 255, 200]))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(mask_blue, edges)
    combined = cv2.dilate(combined, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        x0 = int(w * 0.38)
        return img[:, x0:], (x0, 0, w - x0, h)

    best = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(best)

    # pastikan ROI cukup lebar (panel tabel)
    if bw < rw * 0.5:
        x, y, bw, bh = 0, 0, rw, rh

    pad_x, pad_y = 4, 4
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    bw = min(rw - x, bw + 2 * pad_x)
    bh = min(rh - y, bh + 2 * pad_y)

    panel = right[y : y + bh, x : x + bw]
    offset_x = int(w * 0.38) + x
    return panel, (offset_x, y, bw, bh)


# ---------------------------------------------------------------------------
# Deteksi baris dari anchor Power
# ---------------------------------------------------------------------------
def find_power_anchors(detections, panel_width):
    """Angka power di kolom kanan (~60% lebar panel) jadi anchor tiap baris."""
    power_x_min = panel_width * 0.58
    anchors = []
    for d in detections:
        if is_power_number(d["text"]) and d["cx"] >= power_x_min:
            anchors.append(d)
    anchors.sort(key=lambda d: d["cy"])
    return anchors


def cluster_power_anchors(anchors, min_gap_ratio=0.35):
    """Gabung deteksi power ganda dalam satu baris (mis. digit terpisah)."""
    if not anchors:
        return []

    heights = [a["height"] for a in anchors]
    median_h = sorted(heights)[len(heights) // 2]
    min_gap = median_h * min_gap_ratio

    clusters = [[anchors[0]]]
    for a in anchors[1:]:
        if abs(a["cy"] - clusters[-1][-1]["cy"]) <= min_gap:
            clusters[-1].append(a)
        else:
            clusters.append([a])

    merged = []
    for cluster in clusters:
        best = max(cluster, key=lambda d: len(re.sub(r"\D", "", d["text"])))
        merged.append(best)
    return merged


def row_bands_from_anchors(anchors, panel_height):
    """Bagi panel jadi strip horizontal tanpa overlap, di antara anchor power."""
    if not anchors:
        return []
    ys = [a["cy"] for a in anchors]
    bands = []
    for i, cy in enumerate(ys):
        if i == 0:
            y1 = max(0, int(cy - (ys[1] - ys[0]) * 0.55) if len(ys) > 1 else int(cy - 40))
        else:
            y1 = int((ys[i - 1] + cy) / 2)
        if i == len(ys) - 1:
            y2 = min(panel_height, int(cy + 45))
        else:
            y2 = int((cy + ys[i + 1]) / 2)
        bands.append((y1, y2))
    return bands


def row_band_from_anchor(anchor, panel_height, band_ratio=1.8):
    """Legacy helper untuk debug overlay."""
    half = max(anchor["height"] * band_ratio, 28)
    y1 = max(0, int(anchor["cy"] - half))
    y2 = min(panel_height, int(anchor["cy"] + half))
    return y1, y2


# ---------------------------------------------------------------------------
# Ekstraksi per kolom dalam satu baris
# ---------------------------------------------------------------------------
RANK_COL = (0.0, 0.22)
PLAYER_COL = (0.22, 0.58)
POWER_COL = (0.58, 1.0)

NO_RANK_LABEL = "Tidak Ada Peringkat"


def ocr_region(reader, bgr, allowlist=None, mag_ratio=1.5, preprocess_fn=None, scale=2):
    preprocess_fn = preprocess_fn or preprocess_for_ocr
    proc = preprocess_fn(bgr, scale=scale)
    kwargs = {"detail": 1, "mag_ratio": mag_ratio}
    if allowlist:
        kwargs["allowlist"] = allowlist
    return reader.readtext(proc, **kwargs)


def tokens_in_column(detections, x_min, x_max):
    return [d for d in detections if x_min <= d["cx"] < x_max]


def detections_from_ocr(raw, y_offset=0, x_offset=0, scale=2):
    out = []
    for box, text, conf in raw:
        m = box_metrics(box)
        m["cx"] = m["cx"] / scale + x_offset
        m["cy"] = m["cy"] / scale + y_offset
        m["x_left"] = m["x_left"] / scale + x_offset
        m["x_right"] = m["x_right"] / scale + x_offset
        m["y_top"] = m["y_top"] / scale + y_offset
        m["y_bottom"] = m["y_bottom"] / scale + y_offset
        m["height"] /= scale
        m["width"] /= scale
        m["text"] = text.strip()
        m["conf"] = conf
        out.append(m)
    return out


def parse_rank_from_tokens(tokens, row_index):
    """Rank: angka 1-5, atau label 'Tidak Ada Peringkat'."""
    joined = " ".join(t["text"] for t in tokens).lower()

    if "tidak" in joined and "ada" in joined:
        return NO_RANK_LABEL
    if "peringkat" in joined and "tidak" in joined:
        return NO_RANK_LABEL

    for t in tokens:
        digits = re.sub(r"\D", "", t["text"])
        if digits in {"1", "2", "3", "4", "5"}:
            return int(digits)

    # badge rank 1-3 kadang cuma terbaca sebagai angka kecil
    for t in tokens:
        if re.fullmatch(r"[1-5]", t["text"].strip()):
            return int(t["text"])

    return row_index


NAME_NOISE = {
    "player", "power", "peringkat", "bonus", "aktif", "tidak", "ada",
    "skor", "estimasi", "eneld", "aanni", "anri", "global", "indonesia",
    "belum", "diperoleh",
}

# simbol umum di username game
USERNAME_SYMBOLS = re.compile(r"[©™®•@*_\-.'`~^|\\/<>#&+=\[\]{}()]")

UI_PHRASES = ("tidak ada", "bonus", "aktif", "skor", "estimasi", "belum diperoleh")


def collapse_whitespace(name):
    return re.sub(r"\s+", " ", name).strip()


def is_meaningful_player_token(text):
    """Token nama valid: huruf, angka, unicode, atau simbol username — bukan noise UI."""
    t = text.strip()
    if not t or is_power_number(t):
        return False
    low = t.lower()
    if low in NAME_NOISE:
        return False
    if any(phrase in low for phrase in UI_PHRASES):
        return False
    # huruf/angka unicode
    if re.search(r"[^\W_]", t, re.UNICODE):
        return True
    # simbol khusus / karakter non-ascii (™, ©, ツ, ❄, dll.)
    if USERNAME_SYMBOLS.search(t):
        return True
    return any(ord(c) > 127 for c in t)


def player_col_bounds(panel_width):
    x_start = panel_width * PLAYER_COL[0]
    x_end = panel_width * PLAYER_COL[1]
    col_w = x_end - x_start
    # abaikan area avatar/flag di awal kolom player
    min_cx = x_start + col_w * 0.20
    return x_start, x_end, min_cx


AVATAR_NOISE_WORDS = {
    "incys", "flayel", "fidyei", "taiyci", "dyei", "iyci", "player", "il", "pt",
}


def is_avatar_frame_noise(text, cx, min_cx):
    """Token pendek dari tepi kiri kolom (bingkai avatar/flag)."""
    t = text.strip()
    if cx > min_cx + 35:
        return False
    low = t.lower()
    if low in AVATAR_NOISE_WORDS:
        return True
    if len(t) <= 6 and re.fullmatch(r"[A-Za-z]+", t) and cx < min_cx + 30:
        return True
    return False


def is_rank_badge_leak(text, row_index):
    """Buang token yang hampir pasti bocor dari badge peringkat."""
    t = text.strip()
    if re.fullmatch(r"\d{1,2}", t) and int(t) == row_index:
        return True
    if re.fullmatch(r"[\d\(\)~]{1,3}", t):
        return True
    return False


def filter_player_tokens(tokens, panel_width, row_index):
    _, _, min_cx = player_col_bounds(panel_width)
    out = []
    for t in tokens:
        text = t["text"].strip()
        if t["cx"] < min_cx:
            continue
        if is_avatar_frame_noise(text, t["cx"], min_cx):
            continue
        if is_rank_badge_leak(text, row_index):
            continue
        if is_meaningful_player_token(text):
            out.append(t)
    return out


def merge_player_tokens(tokens):
    """
    Gabung token OCR jadi satu nama.
    Spasi hanya ditambah jika jarak horizontal antar token cukup jauh.
    """
    usable = [
        t for t in sorted(tokens, key=lambda d: d["cx"])
        if is_meaningful_player_token(t["text"])
    ]
    if not usable:
        return ""

    name = usable[0]["text"].strip()
    for prev, curr in zip(usable, usable[1:]):
        gap = curr["x_left"] - prev["x_right"]
        avg_h = (curr["height"] + prev["height"]) / 2
        sep = " " if gap > avg_h * 0.35 else ""
        name += sep + curr["text"].strip()
    return collapse_whitespace(name)


def score_player_name(name, row_index):
    """Skor kualitas nama — pilih hasil OCR terbaik tanpa normalisasi."""
    if not name:
        return -999
    score = min(len(name), 48)
    if re.match(rf"^{row_index}\s", name) or re.match(rf"^{row_index}\D", name):
        score -= 35
    if re.match(r"^[\d\(\)~'\"]+\s*", name):
        score -= 30
    if re.search(r"[^\x00-\x7F]", name):
        score += 12
    if re.search(r"[©™®•@*_]", name):
        score += 10
    if re.search(r"(incys|flayel|fidyei|taiyci|dyei|iyci)", name, re.I):
        score -= 25
    if re.match(r"^Pt\s?", name):
        score -= 20
    return score


def symbol_density(name):
    """Hitung simbol/unicode — prefer hasil yang menangkap karakter khusus."""
    return sum(
        1 for c in name
        if ord(c) > 127 or c in "©™®•@*_-"
    )


def choose_best_name(panel_name, row_name, row_index):
    """Pilih hasil OCR terbaik; buang versi yang punya junk prefix."""
    if not panel_name:
        return row_name
    if not row_name:
        return panel_name

    if panel_name in row_name and row_name != panel_name:
        prefix = row_name[: row_name.index(panel_name)]
        if prefix.strip():
            return panel_name

    p_score = score_player_name(panel_name, row_index)
    r_score = score_player_name(row_name, row_index)

    # prefer row OCR jika menangkap lebih banyak simbol asli
    if symbol_density(row_name) > symbol_density(panel_name) and r_score >= p_score - 5:
        return row_name

    return row_name if r_score > p_score else panel_name


def ocr_player_column(reader, panel, y1, y2, panel_width):
    """OCR khusus kolom player per baris — fokus simbol & unicode."""
    x_rank_end = int(panel_width * RANK_COL[1])
    x_player_end = int(panel_width * PLAYER_COL[1])
    player_img = panel[y1:y2, x_rank_end:x_player_end]
    if player_img.size == 0:
        return []

    scale = 3
    raw = ocr_region(
        reader,
        player_img,
        mag_ratio=2.0,
        preprocess_fn=preprocess_player_text,
        scale=scale,
    )
    return detections_from_ocr(raw, y_offset=y1, x_offset=x_rank_end, scale=scale)


def build_player_name(reader, panel, panel_player_dets, y1, y2, panel_width, row_index):
    """
    Hybrid: deteksi panel + OCR baris khusus player.
    Pilih hasil terbaik berdasarkan skor, tanpa normalisasi alphabetic.
    """
    panel_name = merge_player_tokens(
        filter_player_tokens(panel_player_dets, panel_width, row_index)
    )
    row_dets = filter_player_tokens(
        ocr_player_column(reader, panel, y1, y2, panel_width),
        panel_width,
        row_index,
    )
    row_name = merge_player_tokens(row_dets)

    return choose_best_name(panel_name, row_name, row_index)


def split_row_detections(detections, y1, y2, panel_width):
    """Pisahkan deteksi panel ke kolom rank / player / power dalam satu baris."""
    rank_x = panel_width * RANK_COL[1]
    player_x_end = panel_width * PLAYER_COL[1]
    power_x = panel_width * POWER_COL[0]

    row_dets = [d for d in detections if y1 <= d["cy"] <= y2]
    rank_dets = [d for d in row_dets if d["cx"] < rank_x]
    player_dets = [d for d in row_dets if rank_x <= d["cx"] < player_x_end]
    power_dets = [d for d in row_dets if d["cx"] >= power_x]
    return rank_dets, player_dets, power_dets


def extract_row(reader, panel, panel_detections, y1, y2, panel_width, row_index, anchor):
    """Ekstrak rank, name, power dari deteksi panel + OCR rank jika perlu."""
    rank_dets, player_dets, power_dets = split_row_detections(
        panel_detections, y1, y2, panel_width
    )

    pw = parse_int(anchor["text"])
    if pw is None:
        return None

    row_text = " ".join(d["text"] for d in rank_dets + player_dets + power_dets).lower()
    if "estimasi" in row_text or "skor" in row_text:
        return None

    rank = parse_rank_from_tokens(rank_dets, row_index)
    if rank != NO_RANK_LABEL and not isinstance(rank, int):
        rank = row_index

    name = build_player_name(
        reader, panel, player_dets, y1, y2, panel_width, row_index
    )

    if not name or len(name) < 2:
        return None

    return {"rank": rank, "name": name, "power": pw}


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------
def extract_leaderboard(img, reader):
    panel, (ox, oy, pw, ph) = detect_table_panel(img)

    # OCR cepat di panel penuh untuk cari anchor power
    panel_gray = preprocess_for_ocr(panel)
    raw = reader.readtext(panel_gray, detail=1)
    detections = []
    for box, text, conf in raw:
        m = box_metrics(box)
        m["cx"] /= 2
        m["cy"] /= 2
        m["height"] /= 2
        m["text"] = text.strip()
        m["conf"] = conf
        detections.append(m)

    anchors = cluster_power_anchors(find_power_anchors(detections, pw))

    # buang footer estimasi skor (6275)
    filtered = []
    for a in anchors:
        near_footer = a["cy"] > ph * 0.82
        band_text = " ".join(
            d["text"] for d in detections if abs(d["cy"] - a["cy"]) < 35
        ).lower()
        if near_footer and ("estimasi" in band_text or "skor" in band_text):
            continue
        if parse_int(a["text"]) == 6275:
            continue
        filtered.append(a)
    anchors = filtered

    bands = row_bands_from_anchors(anchors, ph)
    results = []
    for i, (anchor, (y1, y2)) in enumerate(zip(anchors, bands), start=1):
        row = extract_row(
            reader, panel, detections, y1, y2, pw, i, anchor
        )
        if not row:
            continue
        if row["rank"] != NO_RANK_LABEL:
            row["rank"] = i
        results.append(row)

    ranked = [r for r in results if r["rank"] != NO_RANK_LABEL]
    unranked = [r for r in results if r["rank"] == NO_RANK_LABEL]
    final = sorted(ranked, key=lambda r: r["rank"]) + unranked
    return final, panel, (ox, oy, pw, ph), anchors


# ---------------------------------------------------------------------------
# Batch & CLI
# ---------------------------------------------------------------------------
def collect_images(input_dir: Path):
    """Kumpulkan semua file gambar di folder (non-recursive)."""
    if not input_dir.is_dir():
        return []
    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def save_debug_image(img, img_path, roi, anchors, ph):
    ox, oy, pw, _ = roi
    dbg = img.copy()
    cv2.rectangle(dbg, (ox, oy), (ox + pw, oy + ph), (255, 0, 0), 2)
    for a in anchors:
        y1, y2 = row_band_from_anchor(a, ph)
        cv2.rectangle(dbg, (ox, oy + y1), (ox + pw, oy + y2), (0, 255, 0), 2)
        cv2.circle(dbg, (int(ox + a["cx"]), int(oy + a["cy"])), 5, (0, 0, 255), -1)
    for frac in (RANK_COL[1], PLAYER_COL[1]):
        x = ox + int(pw * frac)
        cv2.line(dbg, (x, oy), (x, oy + ph), (0, 255, 255), 1)
    out = img_path.with_name(img_path.stem + "_debug.png")
    cv2.imwrite(str(out), dbg)
    return out


def print_results(results, title="HASIL LEADERBOARD"):
    print(f"\n================ {title} ================")
    print(f"{'Rank':<22}{'Name':<28}{'Power':>8}")
    print("-" * 58)
    for r in results:
        print(f"{str(r['rank']):<22}{r['name']:<28}{r['power']:>8}")
    print("=" * 58)
    print(f"Total entri: {len(results)}")


def process_image(img_path: Path, reader, debug=False):
    """Proses satu gambar, kembalikan entri leaderboard."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"gagal membaca gambar: {img_path}")

    results, panel, roi, anchors = extract_leaderboard(img, reader)
    ox, oy, pw, ph = roi

    if debug:
        out = save_debug_image(img, img_path, roi, anchors, ph)
        print(f"  [DEBUG] {out.name}")

    return results


def process_batch(input_dir: Path, reader, debug=False):
    """Proses semua gambar di folder, gabung jadi satu list output."""
    images = collect_images(input_dir)
    if not images:
        raise FileNotFoundError(f"tidak ada gambar di {input_dir}")

    batch_output = []
    for img_path in images:
        print(f"[INFO] Memproses: {img_path.name}")
        try:
            entries = process_image(img_path, reader, debug=debug)
            batch_output.append({
                "file": img_path.name,
                "leaderboard": entries,
            })
            print(f"  -> {len(entries)} entri")
        except Exception as exc:
            print(f"  [ERROR] {img_path.name}: {exc}", file=sys.stderr)
            batch_output.append({
                "file": img_path.name,
                "leaderboard": [],
                "error": str(exc),
            })

    return batch_output


def main():
    parser = argparse.ArgumentParser(description="OCR leaderboard MLBB (batch)")
    parser.add_argument(
        "image",
        nargs="?",
        help="path satu gambar (opsional; default: batch dari --input)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"folder gambar untuk batch (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"file JSON output (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument("--debug", action="store_true", help="simpan gambar debug per file")
    parser.add_argument(
        "--lang",
        default="en",
        help="bahasa OCR, comma-separated (contoh: en,ja untuk karakter Jepang)",
    )
    args = parser.parse_args()

    langs = [s.strip() for s in args.lang.split(",") if s.strip()]
    print(f"[INFO] Memuat EasyOCR ({', '.join(langs)})...")
    reader = easyocr.Reader(langs, gpu=False, verbose=False)

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[ERROR] File tidak ditemukan: {img_path}", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] Mode single: {img_path.name}")
        results = process_image(img_path, reader, debug=args.debug)
        output = [{"file": img_path.name, "leaderboard": results}]
        print_results(results, title=img_path.name)
    else:
        print(f"[INFO] Mode batch: {args.input}")
        output = process_batch(args.input, reader, debug=args.debug)
        total = sum(len(item["leaderboard"]) for item in output)
        print(f"\n[INFO] Selesai: {len(output)} gambar, {total} entri total")

    args.json_out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Disimpan ke {args.json_out}")


if __name__ == "__main__":
    main()
