"""Streamlit demo: upload an audio clip, get a genre prediction.

Usage:
    venv/bin/streamlit run app/streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import load_trained_model
from src.preprocessing.audio_utils import CHUNK_SAMPLES, SAMPLE_RATE, chunk_audio, load_audio
from src.preprocessing.spectrogram import audio_chunk_to_spectrogram_image, compute_mel_spectrogram
from src.training.data_loader import GENRES

# Preferred first: whichever model scored higher test accuracy in
# reports/*_metrics.json (see reports/final_report/results.md). Currently
# the from-scratch custom CNN (76.4%) outperforms the VGG16 transfer model
# (71.1%, fine-tuning cut short) on this dataset.
MODEL_CANDIDATES = [
    Path("models/checkpoints/custom_cnn_best.keras"),
    Path("models/checkpoints/vgg16_transfer_best.keras"),
]

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")
st.title("🎵 Music Genre Classifier")
st.write(
    "Upload a music clip (WAV/MP3, a few seconds or more) and the model will "
    "predict its genre from its Mel-Spectrogram, trained on the GTZAN dataset."
)


@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return load_trained_model(path), path.name
    return None, None


model, model_name = load_model()
if model is None:
    st.error(
        "No trained model found under models/checkpoints/. "
        "Train the baseline CNN or the transfer learning model first."
    )
    st.stop()

st.caption(f"Using model: `{model_name}`")

uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    try:
        y = load_audio(tmp_path, sr=SAMPLE_RATE)
    except Exception as e:
        st.error(f"Could not read this audio file: {e}")
        st.stop()

    if len(y) < CHUNK_SAMPLES:
        st.error(
            f"Clip is too short — need at least {CHUNK_SAMPLES / SAMPLE_RATE:.0f}s of audio."
        )
        st.stop()

    chunks = chunk_audio(y)
    specs = np.stack([audio_chunk_to_spectrogram_image(c, SAMPLE_RATE) for c in chunks])
    specs = specs[..., np.newaxis]  # (n_chunks, 128, 128, 1)

    with st.spinner("Predicting..."):
        probs = model.predict(specs, verbose=0)
    mean_probs = probs.mean(axis=0)  # average over all 3s chunks in the clip

    top_idx = int(np.argmax(mean_probs))
    st.subheader(f"Predicted genre: **{GENRES[top_idx].upper()}**")
    st.write(f"Confidence: {mean_probs[top_idx] * 100:.1f}%")

    st.bar_chart({genre: float(p) for genre, p in zip(GENRES, mean_probs)})

    st.subheader("Mel-Spectrogram (first 3s segment)")
    mel_db = compute_mel_spectrogram(chunks[0], SAMPLE_RATE)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("time frames")
    ax.set_ylabel("mel bands")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
