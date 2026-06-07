#!/usr/bin/env python3
"""OCR leaderboard images and write a single JSON output file.

The script scans every image in the input folder, runs OCR with the Tesseract
CLI, parses leaderboard rows, and writes all extracted rows into output.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_DATASET_DIR = "datasets"
DEFAULT_OUTPUT_FILE = "output.json"
DEFAULT_GDRIVE_URL = "https://drive.google.com/drive/folders/1e8cx33AYnAZOehlh7FnJWZNiFEmDJ6R7"
MAX_OCR_IMAGE_WIDTH = 1024


@dataclass(frozen=True)
class LeaderboardEntry:
    """Single leaderboard row parsed from OCR text."""

    rank: int | None
    nama_pemain: str
    point: int | None
    source_image: str
    raw_line: str
    rank_label: str | None = None
    point_label: str | None = None


def iter_images(input_dir: Path) -> list[Path]:
    """Return supported image files in a deterministic order."""

    if not input_dir.exists():
        return []

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def download_gdrive_folder(url: str, output_dir: Path, force: bool = False) -> None:
    """Download a public Google Drive folder into the dataset directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_images = iter_images(output_dir)
    if existing_images and not force:
        print(
            f"Folder {output_dir} sudah berisi {len(existing_images)} gambar. "
            "Lewati download Google Drive. Pakai --force-download untuk download ulang.",
            flush=True,
        )
        return

    if force:
        for image_path in existing_images:
            image_path.unlink()

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Package gdown belum terinstall. Jalankan: python3 -m pip install -r requirements.txt"
        ) from exc

    print(f"Download dataset dari Google Drive ke {output_dir} ...", flush=True)
    downloaded_files = gdown.download_folder(
        url=url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    downloaded_count = len(downloaded_files or [])
    print(f"Download selesai: {downloaded_count} file.", flush=True)


def normalize_ocr_line(line: str) -> str:
    """Normalize whitespace and common OCR separators without changing names."""

    cleaned = line.replace("\u00a0", " ")
    cleaned = re.sub(r"[|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_number(value: str) -> int | None:
    """Convert OCR-ish numeric text into an int when it is reliable enough."""

    translation = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1",
            "S": "5",
            "s": "5",
            "B": "8",
        }
    )
    digits = re.sub(r"\D", "", value.translate(translation))
    if not digits:
        return None
    return int(digits)


def is_noise_line(line: str) -> bool:
    """Filter UI labels and instructions that are not player rows."""

    lowered = line.lower()
    noise_keywords = (
        "leaderboard",
        "peringkat player",
        "player power",
        "sisa waktu",
        "global",
        "indonesia",
        "jawa barat",
        "kota sukabumi",
        "server",
        "anda harus",
        "memasuki leaderboard",
    )
    return any(keyword in lowered for keyword in noise_keywords)


def is_leading_noise_token(token: str) -> bool:
    """Return whether a leading OCR token is likely from icons/flags, not names."""

    stripped = token.strip("`'\".,:;|[](){}<>+-_=\\/!@#$%^&*~? ")
    if not stripped:
        return True

    lowered = stripped.lower()
    if lowered in {"oy", "rai", "ine", "cy", "fa", "fe", "my", "ry", "as", "rt", "fast"}:
        return True

    if normalize_number(stripped) is not None and len(stripped) <= 2:
        return True

    return len(stripped) <= 2 and "@" not in token


def cleanup_player_name(value: str) -> str:
    """Remove common leading OCR artifacts from avatar, rank, and flag icons."""

    cleaned = value.strip(" -:\t|[](){}")
    tokens = cleaned.split()

    while tokens and is_leading_noise_token(tokens[0]):
        tokens.pop(0)

    cleaned = " ".join(tokens)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -:\t|[](){}")


def split_row_head_and_point(line: str) -> tuple[str, int] | None:
    """Split a row into text before the final score and the score value."""

    match = re.match(r"^(?P<head>.+?)\s+(?P<point>[0-9oOIlSBs]{3,6})(?:\D{0,20})$", line)
    if not match:
        return None

    point = normalize_number(match.group("point"))
    if point is None:
        return None

    return match.group("head"), point


def parse_unranked_line(line: str, source_image: str) -> LeaderboardEntry | None:
    """Parse the local player's unranked row when OCR keeps it on one line."""

    match = re.search(
        r"(?i)(tidak\s+ada\s+peringkat)\s+(.+?)\s+(belum\s+diperoleh|[0-9oOIlSBs]{3,6})$",
        line,
    )
    if not match:
        return None

    point_text = match.group(3)
    point = normalize_number(point_text) if re.search(r"\d", point_text) else None
    return LeaderboardEntry(
        rank=None,
        rank_label="Tidak Ada Peringkat",
        nama_pemain=match.group(2).strip(),
        point=point,
        point_label=None if point is not None else point_text.title(),
        source_image=source_image,
        raw_line=line,
    )


def parse_ranked_line(line: str, source_image: str) -> LeaderboardEntry | None:
    """Parse rows shaped like: ``6 Player Name 8061``."""

    match = re.match(r"^(?P<rank>[0-9oOIlS]{1,3})\s+(?P<row>.+)$", line)
    if not match:
        return None

    rank = normalize_number(match.group("rank"))
    row_parts = split_row_head_and_point(match.group("row"))
    if rank is None or row_parts is None:
        return None

    head, point = row_parts
    nama_pemain = cleanup_player_name(head)
    if not nama_pemain:
        return None

    return LeaderboardEntry(
        rank=rank,
        nama_pemain=nama_pemain,
        point=point,
        source_image=source_image,
        raw_line=line,
    )


def parse_inferred_rank_line(line: str, source_image: str, rank: int) -> LeaderboardEntry | None:
    """Parse a row without visible rank text and infer rank from row order."""

    row_parts = split_row_head_and_point(line)
    if row_parts is None:
        return None

    head, point = row_parts
    nama_pemain = cleanup_player_name(head)
    if len(nama_pemain) < 3 or not re.search(r"[A-Za-z0-9@]", nama_pemain):
        return None

    return LeaderboardEntry(
        rank=rank,
        nama_pemain=nama_pemain,
        point=point,
        source_image=source_image,
        raw_line=line,
    )


def parse_leaderboard_text(raw_text: str, source_image: str = "") -> list[LeaderboardEntry]:
    """Extract leaderboard entries from OCR text."""

    entries: list[LeaderboardEntry] = []
    seen: set[tuple[int | None, str, int | None]] = set()
    next_inferred_rank = 1

    for raw_line in raw_text.splitlines():
        line = normalize_ocr_line(raw_line)
        if not line or is_noise_line(line):
            continue

        entry = parse_ranked_line(line, source_image)
        if entry is not None and entry.rank is not None:
            next_inferred_rank = entry.rank + 1

        if entry is None:
            entry = parse_unranked_line(line, source_image)

        if entry is None:
            entry = parse_inferred_rank_line(line, source_image, next_inferred_rank)
            if entry is not None:
                next_inferred_rank += 1

        if entry is None:
            continue

        key = (entry.rank, entry.nama_pemain.casefold(), entry.point)
        if key in seen:
            continue

        entries.append(entry)
        seen.add(key)

    return entries


def build_ocr_candidates(image_path: Path, tmp_dir: Path) -> list[Path]:
    """Build original and optional preprocessed image variants for OCR."""

    try:
        from PIL import Image, ImageOps
    except ImportError:
        return [image_path]

    candidates: list[Path] = []

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width > MAX_OCR_IMAGE_WIDTH:
            resized_height = int(height * MAX_OCR_IMAGE_WIDTH / width)
            image = image.resize((MAX_OCR_IMAGE_WIDTH, resized_height))
            width, height = image.size

        variants = []

        # Mobile Legends leaderboard rows are normally on the right panel.
        if width > 1 and height > 1:
            variants.append(
                (
                    "leaderboard-panel",
                    image.crop((int(width * 0.43), int(height * 0.16), int(width * 0.90), int(height * 0.91))),
                    "gray",
                )
            )
            variants.append(
                (
                    "leaderboard-panel-binary",
                    image.crop((int(width * 0.43), int(height * 0.16), int(width * 0.90), int(height * 0.91))),
                    "binary",
                )
            )
            variants.append(
                (
                    "name-power-panel",
                    image.crop((int(width * 0.59), int(height * 0.22), int(width * 0.82), int(height * 0.90))),
                    "gray",
                )
            )
        variants.append(("full", image, "gray"))

        for name, variant, mode in variants:
            gray = ImageOps.grayscale(variant)
            enhanced = ImageOps.autocontrast(gray)
            if mode == "binary":
                enhanced = enhanced.point(lambda pixel: 0 if pixel > 135 else 255, mode="1").convert("L")
            output_path = tmp_dir / f"{image_path.stem}-{name}.png"
            enhanced.save(output_path)
            candidates.append(output_path)

    candidates.append(image_path)
    return candidates


def tesseract_languages(primary_language: str) -> list[str | None]:
    """Return language attempts, falling back when Indonesian data is absent."""

    languages: list[str | None] = []
    for language in (primary_language, "eng", None):
        if language not in languages:
            languages.append(language)
    return languages


def run_tesseract(image_path: Path, language: str, timeout: int) -> tuple[str, list[str]]:
    """Run Tesseract over image variants and return the best raw OCR text."""

    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "Tesseract CLI tidak ditemukan. Install tesseract-ocr, lalu jalankan ulang program."
        )

    errors: list[str] = []
    best_text = ""
    best_score = (-1, -1)

    with tempfile.TemporaryDirectory(prefix="ocr-leaderboard-") as tmp:
        candidates = build_ocr_candidates(image_path, Path(tmp))

        for candidate in candidates:
            for lang in tesseract_languages(language):
                command = ["tesseract", str(candidate), "stdout", "--oem", "3", "--psm", "6"]
                if lang:
                    command.extend(["-l", lang])

                env = os.environ.copy()
                env.setdefault("OMP_THREAD_LIMIT", "1")

                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                        env=env,
                    )
                except subprocess.TimeoutExpired:
                    lang_label = lang or "default"
                    errors.append(f"{candidate.name} ({lang_label}): timeout setelah {timeout} detik")
                    continue

                if result.returncode != 0:
                    lang_label = lang or "default"
                    errors.append(f"{candidate.name} ({lang_label}): {result.stderr.strip()}")
                    continue

                raw_text = result.stdout.strip()
                parsed_entries = parse_leaderboard_text(raw_text, str(image_path))
                score = (len(parsed_entries), len(raw_text))
                if score > best_score:
                    best_score = score
                    best_text = raw_text
                if len(parsed_entries) >= 5:
                    return raw_text, errors

    return best_text, errors


def process_image(image_path: Path, language: str, timeout: int) -> dict:
    """OCR one image and return a serializable result payload."""

    source_image = str(image_path.relative_to(Path.cwd())) if image_path.is_relative_to(Path.cwd()) else str(image_path)

    try:
        raw_text, errors = run_tesseract(image_path, language, timeout)
        entries = parse_leaderboard_text(raw_text, source_image)
        error = None
    except Exception as exc:  # The per-image error is reported in output.json.
        raw_text = ""
        errors = []
        entries = []
        error = str(exc)

    return {
        "image": source_image,
        "entries": [asdict(entry) for entry in entries],
        "raw_text": raw_text,
        "error": error,
        "ocr_warnings": errors,
    }


def write_output(output_path: Path, input_dir: Path, image_results: list[dict]) -> None:
    """Write the single JSON source of OCR results."""

    all_entries = [
        entry
        for image_result in image_results
        for entry in image_result["entries"]
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "images": image_results,
        "entries": all_entries,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_images(image_paths: list[Path], language: str, timeout: int, workers: int) -> list[dict]:
    """Process images concurrently while preserving dataset order."""

    if not image_paths:
        return []

    workers = max(1, workers)
    results: list[dict | None] = [None] * len(image_paths)

    if workers == 1:
        for index, image_path in enumerate(image_paths, start=1):
            print(f"[{index}/{len(image_paths)}] OCR {image_path}", flush=True)
            results[index - 1] = process_image(image_path, language, timeout)
        return [result for result in results if result is not None]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_image, image_path, language, timeout): (index, image_path)
            for index, image_path in enumerate(image_paths)
        }
        completed = 0
        for future in as_completed(futures):
            index, image_path = futures[future]
            results[index] = future.result()
            completed += 1
            print(f"[{completed}/{len(image_paths)}] OCR selesai: {image_path}", flush=True)

    return [result for result in results if result is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR gambar leaderboard dari folder datasets dan simpan hasilnya ke output.json."
    )
    parser.add_argument("-i", "--input", default=DEFAULT_DATASET_DIR, help="Folder berisi gambar input.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE, help="File JSON output.")
    parser.add_argument(
        "--fetch-gdrive",
        action="store_true",
        help="Download dataset dari Google Drive sebelum OCR.",
    )
    parser.add_argument(
        "--gdrive-url",
        default=DEFAULT_GDRIVE_URL,
        help="URL folder Google Drive dataset.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Hapus gambar yang sudah ada di folder input lalu download ulang dari Google Drive.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah gambar yang diproses, berguna untuk percobaan awal di Colab/Cloud Shell.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="eng+ind",
        help="Bahasa Tesseract yang dicoba pertama kali, default: eng+ind.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Timeout OCR per percobaan dalam detik.")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Jumlah proses OCR paralel, default maksimal 4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)
    input_dir.mkdir(parents=True, exist_ok=True)

    if args.fetch_gdrive:
        download_gdrive_folder(args.gdrive_url, input_dir, force=args.force_download)

    all_image_paths = iter_images(input_dir)
    image_paths = all_image_paths
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit harus bernilai 0 atau lebih.")
        image_paths = all_image_paths[: args.limit]

    if not image_paths:
        if all_image_paths and args.limit == 0:
            print("Tidak ada gambar diproses karena --limit 0.", flush=True)
        else:
            print(
                f"Tidak ada gambar di {input_dir}. "
                "Tambahkan gambar manual atau jalankan dengan --fetch-gdrive.",
                flush=True,
            )

    image_results = process_images(image_paths, args.language, args.timeout, args.workers)
    write_output(output_path, input_dir, image_results)

    print(f"Processed {len(image_results)} image(s). Output written to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
