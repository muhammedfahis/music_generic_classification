"""Quick EDA: sample waveforms + Mel-Spectrograms per genre, and a
sample-rate/duration consistency check across the raw dataset.

Usage:
    venv/bin/python -m src.preprocessing.eda
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

from src.preprocessing.audio_utils import SAMPLE_RATE, load_audio
from src.preprocessing.spectrogram import compute_mel_spectrogram
from src.preprocessing.build_dataset import GENRES, find_genre_root, RAW_DIR

FIGURES_DIR = Path("reports/figures")


def check_consistency(genre_root):
    print("Sample rate / duration consistency check:")
    srs, durations, bad = set(), [], []
    for genre in GENRES:
        for f in sorted((genre_root / genre).glob("*.wav")):
            try:
                info = sf.info(str(f))
            except Exception as e:
                bad.append((str(f), str(e)))
                continue
            srs.add(info.samplerate)
            durations.append(info.duration)
    print(f"  Unique sample rates found: {sorted(srs)}")
    print(f"  Duration range: {min(durations):.2f}s - {max(durations):.2f}s "
          f"(mean {sum(durations)/len(durations):.2f}s)")
    if bad:
        print(f"  Unreadable files: {len(bad)}")
        for path, err in bad:
            print(f"    - {path}: {err}")


def plot_waveforms_and_spectrograms(genre_root, out_path):
    fig, axes = plt.subplots(len(GENRES), 2, figsize=(10, 2.2 * len(GENRES)))
    for i, genre in enumerate(GENRES):
        sample_file = sorted((genre_root / genre).glob("*.wav"))[0]
        try:
            y = load_audio(sample_file)
        except Exception:
            # fall back to the second file if the first is unreadable
            sample_file = sorted((genre_root / genre).glob("*.wav"))[1]
            y = load_audio(sample_file)

        axes[i, 0].plot(y, linewidth=0.4)
        axes[i, 0].set_ylabel(genre, rotation=0, ha="right", va="center")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        mel_db = compute_mel_spectrogram(y[: SAMPLE_RATE * 3], SAMPLE_RATE)
        axes[i, 1].imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    axes[0, 0].set_title("Waveform (full clip)")
    axes[0, 1].set_title("Mel-Spectrogram (first 3s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved EDA figure to {out_path}")


def main():
    genre_root = find_genre_root(RAW_DIR)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    check_consistency(genre_root)
    plot_waveforms_and_spectrograms(genre_root, FIGURES_DIR / "eda_waveforms_spectrograms.png")


if __name__ == "__main__":
    main()
