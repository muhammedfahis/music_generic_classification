"""Audio loading and chunking utilities built on Librosa."""

import numpy as np
import librosa

SAMPLE_RATE = 22050
CHUNK_DURATION = 3.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


def load_audio(path, sr=SAMPLE_RATE):
    """Load an audio file as a mono waveform at the target sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def chunk_audio(y, chunk_samples=CHUNK_SAMPLES, drop_last=True):
    """Split a waveform into fixed-length, non-overlapping chunks.

    GTZAN clips are ~30s; chunking into 3s segments both fits typical CNN
    input sizes and multiplies the number of training examples (small
    dataset otherwise). `drop_last` discards a trailing partial chunk so
    every chunk has identical shape.
    """
    n_chunks = len(y) // chunk_samples
    chunks = [
        y[i * chunk_samples : (i + 1) * chunk_samples] for i in range(n_chunks)
    ]
    if not drop_last:
        remainder = y[n_chunks * chunk_samples :]
        if len(remainder) > 0:
            padded = np.pad(remainder, (0, chunk_samples - len(remainder)))
            chunks.append(padded)
    return chunks
