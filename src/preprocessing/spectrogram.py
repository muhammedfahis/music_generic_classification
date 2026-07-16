"""Mel-Spectrogram generation and normalization for CNN input."""

import numpy as np
import cv2
import librosa

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
IMG_SIZE = (128, 128)  # (height, width) fed to the CNN


def compute_mel_spectrogram(y, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Compute a log-scaled (dB) Mel-Spectrogram for a waveform chunk."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def normalize_spectrogram(mel_db):
    """Scale a dB spectrogram to [0, 1] using its own min/max."""
    mn, mx = mel_db.min(), mel_db.max()
    if mx - mn < 1e-6:
        return np.zeros_like(mel_db, dtype=np.float32)
    return ((mel_db - mn) / (mx - mn)).astype(np.float32)


def resize_spectrogram(mel_norm, size=IMG_SIZE):
    """Resize a normalized spectrogram to a fixed CNN input size."""
    return cv2.resize(mel_norm, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def audio_chunk_to_spectrogram_image(y, sr, size=IMG_SIZE):
    """Full chunk -> model-ready spectrogram pipeline (dB -> normalize -> resize)."""
    mel_db = compute_mel_spectrogram(y, sr)
    mel_norm = normalize_spectrogram(mel_db)
    return resize_spectrogram(mel_norm, size)
