"""Build the Mel-Spectrogram dataset from raw GTZAN audio.

Walks the extracted GTZAN genre folders, chunks each 30s track into 3s
segments, computes normalized Mel-Spectrograms, and saves them as .npy
arrays under data/processed/mel_spectrograms/<genre>/. Also writes a
manifest CSV (data/processed/manifest.csv) used later to build a
leakage-free train/val/test split grouped by source track.

Usage:
    venv/bin/python -m src.preprocessing.build_dataset
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.audio_utils import load_audio, chunk_audio
from src.preprocessing.spectrogram import audio_chunk_to_spectrogram_image

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/mel_spectrograms")
MANIFEST_PATH = Path("data/processed/manifest.csv")

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

AUDIO_EXTS = {".wav", ".au"}


def find_genre_root(raw_dir: Path) -> Path:
    """Locate the folder containing the 10 genre subfolders, whatever the
    archive's top-level layout turned out to be."""
    candidates = [
        raw_dir / "genres",
        raw_dir / "genres_original",
        raw_dir / "Data" / "genres_original",
    ]
    for c in candidates:
        if c.exists() and any((c / g).exists() for g in GENRES):
            return c
    # fall back to a recursive search for a dir that directly contains genre subdirs
    for p in raw_dir.rglob("*"):
        if p.is_dir() and all((p / g).exists() for g in GENRES):
            return p
    raise FileNotFoundError(
        f"Could not locate GTZAN genre folders under {raw_dir}. "
        f"Expected one of: {[str(c) for c in candidates]}"
    )


def main():
    genre_root = find_genre_root(RAW_DIR)
    print(f"Using genre root: {genre_root}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    skipped = []
    t0 = time.time()

    for genre in GENRES:
        genre_dir = genre_root / genre
        if not genre_dir.exists():
            print(f"  [warn] missing genre dir: {genre_dir}", file=sys.stderr)
            continue

        out_genre_dir = OUT_DIR / genre
        out_genre_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            f for f in genre_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS
        )
        print(f"{genre}: {len(files)} tracks")

        for f in files:
            track_id = f.stem  # e.g. "jazz.00054" -- used to group splits, avoid leakage
            try:
                y = load_audio(f)
            except Exception as e:
                print(f"  [skip] failed to load {f}: {e}", file=sys.stderr)
                skipped.append(str(f))
                continue

            chunks = chunk_audio(y)
            for i, chunk in enumerate(chunks):
                try:
                    spec = audio_chunk_to_spectrogram_image(chunk, sr=22050)
                except Exception as e:
                    print(f"  [skip] failed spectrogram for {f} chunk {i}: {e}", file=sys.stderr)
                    continue

                out_name = f"{track_id}_chunk{i}.npy"
                out_path = out_genre_dir / out_name
                np.save(out_path, spec)

                records.append({
                    "filepath": str(out_path),
                    "genre": genre,
                    "track_id": track_id,
                    "chunk_idx": i,
                })

    df = pd.DataFrame.from_records(records)
    df.to_csv(MANIFEST_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"Total spectrogram chunks: {len(df)}")
    print(df.groupby("genre").size())
    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable files:")
        for s in skipped:
            print(f"  - {s}")
    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
