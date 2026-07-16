# Music Genre Classification (Minor Project 21CSA697A)

Deep learning-based music genre classification using Mel-Spectrograms and CNNs,
trained on the GTZAN dataset, with a Streamlit demo app.

See:
- [`Music_Genre_Classification_Proposal.pdf`](./Music_Genre_Classification_Proposal.pdf) — original proposal
- [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) — detailed technical plan and repo structure
- [`PROGRESS_TRACKER.md`](./PROGRESS_TRACKER.md) — phase-by-phase checklist, update as work progresses

## Run the trained app (already built in this repo)

A trained model already exists at `models/checkpoints/custom_cnn_best.keras`.
To launch the demo:

```bash
venv/bin/streamlit run app/streamlit_app.py
```

This opens a browser tab at `http://localhost:8501` (or prints the URL to
visit). Upload a WAV/MP3/OGG/FLAC clip and it predicts the genre.

## Full setup from scratch

```bash
# 1. Create the venv (Python 3.11; TensorFlow doesn't yet support 3.13 on macOS)
/opt/homebrew/bin/python3.11 -m venv venv   # or: python3.11 -m venv venv

# 2. Install dependencies
venv/bin/pip install -r requirements.txt
```

`requirements.txt` is pinned for Apple Silicon (`tensorflow-macos` +
`tensorflow-metal`). On Linux/Windows, swap those two lines for `tensorflow`.

Get the GTZAN dataset (10 genres × 100 clips) into `data/raw/genres/<genre>/*.wav`.
Kaggle requires an API key; this project instead used the
[`marsyas/gtzan`](https://huggingface.co/datasets/marsyas/gtzan) mirror on
Hugging Face, which needs no auth:

```bash
curl -L -o data/raw/genres.tar.gz \
  "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
tar -xzf data/raw/genres.tar.gz -C data/raw/
rm data/raw/genres.tar.gz   # ~1.2GB, safe to delete once extracted
```

## Reproduce the full pipeline

Run from the repo root, in order:

```bash
venv/bin/python -m src.preprocessing.build_dataset   # audio -> mel-spectrogram .npy files (~1 min)
venv/bin/python -m src.preprocessing.make_splits     # leakage-free train/val/test split
venv/bin/python -m src.preprocessing.eda             # optional: sample waveform/spectrogram plots

venv/bin/python -m src.training.train_baseline       # trains the custom CNN (~10-15 min on M1)
venv/bin/python -m src.evaluation.metrics \
  --model models/checkpoints/custom_cnn_best.keras --name custom_cnn

venv/bin/python -m src.training.train_transfer       # trains VGG16 transfer learning (slower, ~1-2 hrs)
venv/bin/python -m src.evaluation.metrics \
  --model models/checkpoints/vgg16_transfer_best.keras --name vgg16_transfer

venv/bin/streamlit run app/streamlit_app.py           # launch the demo app
```

Results, confusion matrices, and training curves land in `reports/figures/`
and `reports/*_metrics.json`; the write-up is in `reports/final_report/`.
