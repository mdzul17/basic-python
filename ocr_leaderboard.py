#!/usr/bin/env python3
"""OCR leaderboard images and write a single JSON output file.

The script scans every image in the input folder, runs OCR with the Tesseract
CLI, parses leaderboard rows, and writes all extracted rows into output.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_DATASET_DIR = "datasets"
DEFAULT_OUTPUT_FILE = "output.json"


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

    match = re.match(r"^(?P<rank>[0-9oOIlS]{1,3})\s+(?P<player>.+?)\s+(?P<point>[0-9oOIlSBs]{3,6})$", line)
    if not match:
        return None

    rank = normalize_number(match.group("rank"))
    point = normalize_number(match.group("point"))
    nama_pemain = match.group("player").strip(" -:\t")
    if rank is None or point is None or not nama_pemain:
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

    for raw_line in raw_text.splitlines():
        line = normalize_ocr_line(raw_line)
        if not line or is_noise_line(line):
            continue

        entry = parse_ranked_line(line, source_image) or parse_unranked_line(line, source_image)
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

    candidates = [image_path]

    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        return candidates

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        variants = [("full", image)]

        # Mobile Legends leaderboard rows are normally on the right panel.
        if width > 1 and height > 1:
            variants.append(("leaderboard-panel", image.crop((int(width * 0.42), 0, width, height))))

        for name, variant in variants:
            gray = ImageOps.grayscale(variant)
            scaled = gray.resize((gray.width * 2, gray.height * 2))
            enhanced = ImageOps.autocontrast(scaled).filter(ImageFilter.SHARPEN)
            output_path = tmp_dir / f"{image_path.stem}-{name}.png"
            enhanced.save(output_path)
            candidates.append(output_path)

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

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR gambar leaderboard dari folder datasets dan simpan hasilnya ke output.json."
    )
    parser.add_argument("-i", "--input", default=DEFAULT_DATASET_DIR, help="Folder berisi gambar input.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE, help="File JSON output.")
    parser.add_argument(
        "-l",
        "--language",
        default="eng+ind",
        help="Bahasa Tesseract yang dicoba pertama kali, default: eng+ind.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Timeout OCR per percobaan dalam detik.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)
    input_dir.mkdir(parents=True, exist_ok=True)

    image_results = [
        process_image(image_path, args.language, args.timeout)
        for image_path in iter_images(input_dir)
    ]
    write_output(output_path, input_dir, image_results)

    print(f"Processed {len(image_results)} image(s). Output written to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
