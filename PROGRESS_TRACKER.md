# Progress Tracker — Music Genre Classification Project

Update the **Status** and **Date Done** columns as you go. Status values:
`Not Started` · `In Progress` · `Blocked` · `Done`

Fill in **Start Date** once you actually begin — the Week numbers below are
relative to that date (per the proposal's 12-week plan).

**Start Date:** 2026-07-16

## Overview

| Phase | Weeks | Status | Date Done | Notes |
|---|---|---|---|---|
| 1. Literature Review & Dataset Collection | 1–2 | Done | 2026-07-16 | |
| 2. Preprocessing & Mel-Spectrogram Generation | 3–4 | Done | 2026-07-16 | 9,981 spectrogram chunks, 0 leaked tracks |
| 3. Custom CNN Design & Training | 5–7 | Done | 2026-07-16 | Test accuracy 76.40% |
| 4. Transfer Learning (VGG16/ResNet50) | 8–9 | Done | 2026-07-16 | Test accuracy 71.07%; fine-tuning cut short (time) |
| 5. Model Evaluation | 10 | Done | 2026-07-16 | Baseline CNN selected as final model |
| 6. Streamlit Web App | 11 | Done | 2026-07-16 | Inference path + server boot verified |
| 7. Testing, Bug Fixes, Final Report | 12 | In Progress | 2026-07-16 | Report written; final repo cleanup pass still open |

---

## Phase 1 — Literature Review & Dataset Collection (Weeks 1–2)
- [x] Read papers/articles on music genre classification / MIR with CNNs (Tzanetakis & Cook 2002 + several recent CNN/transfer-learning-on-GTZAN studies)
- [x] Write short literature review summary for final report → `reports/final_report/literature_review.md`
- [x] Download GTZAN dataset (via HuggingFace `marsyas/gtzan` mirror — no Kaggle credentials configured) → `data/raw/`
- [x] Verify all 10 genre folders present with ~100 clips each (confirmed: 100 each)
- [x] Check for known corrupt file (`jazz.00054.wav`) — confirmed corrupt (`NoBackendError`), auto-skipped by the pipeline
- [x] Quick EDA: plot sample waveforms per genre, check sample rate/duration consistency — all clips 22050Hz, ~30s (`src/preprocessing/eda.py` → `reports/figures/eda_waveforms_spectrograms.png`)

## Phase 2 — Preprocessing & Mel-Spectrogram Generation (Weeks 3–4)
- [x] Decide segment strategy — chunked into 3s segments (10 chunks/track) to multiply training examples
- [x] Decide storage format — `.npy` arrays, 128×128 normalized log-mel spectrograms
- [x] Implement `src/preprocessing/audio_utils.py` (Librosa audio loading + chunking)
- [x] Implement `src/preprocessing/spectrogram.py` (Mel-spectrogram + dB + normalize + resize)
- [x] Batch-generate spectrograms for full dataset → `data/processed/mel_spectrograms/` (9,981 chunks, `src/preprocessing/build_dataset.py`)
- [x] Create stratified train/val/test split, grouped by `track_id` to prevent leakage (`src/preprocessing/make_splits.py` → `data/processed/splits.csv`; verified 0 leaking tracks). Split: 7183 train / 1298 val / 1500 test
- [x] Sanity-check: visualize a few spectrograms per genre (same EDA figure as above)

## Phase 3 — Custom CNN Design & Training (Weeks 5–7)
- [x] Design baseline CNN architecture (`src/models/custom_cnn.py`) — 4 conv blocks + GAP + dense, 423K params
- [x] Build `tf.data` pipeline for spectrogram inputs (`src/training/data_loader.py`)
- [x] Implement training script with checkpointing + early stopping (`src/training/train_baseline.py`)
- [x] Train baseline model, log accuracy/loss curves — 19 epochs (early stopped), best val_accuracy 71.65% at epoch 17 → `reports/figures/baseline_cnn_training_curves.png`
- [x] Save baseline model to `models/checkpoints/custom_cnn_best.keras` (4.9MB)
- [x] Record baseline test accuracy — **76.40%** test accuracy, macro F1 0.75 (`reports/custom_cnn_metrics.json`). Weakest class: rock (F1 0.44, confused across many genres — a known hard class in GTZAN); strongest: classical/metal/pop (F1 ~0.89-0.90 area)

## Phase 4 — Transfer Learning: VGG16 / ResNet50 (Weeks 8–9)
- [x] Implement `src/models/transfer_model.py` (VGG16 base + custom head)
- [x] Handle input adaptation (1-channel spectrogram → 3-channel, resize to 224×224, VGG16 preprocessing as a proper serializable layer)
- [x] Train head-only (frozen base) — 15 epochs, best val_accuracy 69.95%
- [x] Fine-tune top layers (unfrozen from layer 15, low LR 1e-5) — only 1 epoch completed (val_accuracy 69.72%) before being stopped deliberately due to training time on available hardware (~880-940ms/step on M1 CPU-bound steps; head phase alone took ~55 min)
- [x] Save best transfer-learning model to `models/checkpoints/vgg16_transfer_best.keras` (117MB)
- [x] Record transfer-learning test accuracy — **71.07%**, macro F1 0.71 (`reports/vgg16_transfer_metrics.json`)
- Note: hit a Keras 3 deserialization issue (`Lambda` layer with a Python closure can't be safely reloaded) — fixed by rewriting the VGG16 preprocessing step as a proper serializable custom `Layer` subclass (`VGG16Preprocess` in `transfer_model.py`) and adding a `load_trained_model()` fallback in `src/evaluation/metrics.py` that rebuilds the architecture + loads weights for the already-saved checkpoint.

## Phase 5 — Model Evaluation (Week 10)
- [x] Compute accuracy, precision, recall, F1 (macro + per-class) for both models
- [x] Generate confusion matrices (Seaborn heatmaps) for both models
- [x] Build comparison table: custom CNN vs. transfer learning → `reports/final_report/results.md`
- [x] Identify commonly confused genre pairs, note in report — rock is hardest for both models (confused with disco/country/metal/blues); reggae/hiphop confusable in the transfer model
- [x] Select final model for deployment — **custom CNN** (76.40% > 71.07%); Streamlit app now prefers it

## Phase 6 — Streamlit Web App (Week 11)
- [x] Implement `app/streamlit_app.py` (upload → preprocess → predict → display)
- [x] Load final trained model into app (prefers `custom_cnn_best.keras`, the better-performing model — see `results.md`)
- [x] Display prediction confidence (bar chart) + spectrogram of uploaded clip
- [x] Handle bad input (wrong format, too short/long, corrupt file)
- [x] Local run/test of the app end-to-end — inference path (chunk → spectrogram → predict → average) verified against a real held-out test-split file (`blues.00090.wav`, correctly predicted "blues" at 74.8% confidence); Streamlit server confirmed to boot and serve HTTP 200 before being stopped

## Phase 7 — Testing, Bug Fixes, Final Report (Week 12)
- [x] Test app with a fresh held-out audio clip (see above) — a true browser UI click-through wasn't run (no interactive browser session in this environment) but the full inference pipeline the UI calls was exercised directly
- [x] Fix edge cases found during testing — fixed a Keras 3 model-loading bug (`Lambda` layer deserialization) discovered while evaluating the transfer model; see Phase 4 note
- [x] Write final report (methodology, results, literature review) → `reports/final_report/{methodology,results,literature_review}.md`
- [x] Add `README.md` with setup/run instructions
- [ ] Final repo cleanup

---

## Deliverables Checklist (from proposal)
- [x] Trained CNN model classifying 10 genres — custom CNN, 76.40% test accuracy
- [x] Comparative evaluation: custom CNN vs. Transfer Learning (VGG16/ResNet50) — `reports/final_report/results.md`
- [x] Mel-Spectrogram visualization pipeline (Librosa) — `src/preprocessing/spectrogram.py`, `eda.py`
- [x] Streamlit web app for genre prediction — `app/streamlit_app.py`, inference path verified
- [x] Final project report — `reports/final_report/{literature_review,methodology,results}.md` (a short intro/conclusion + PDF export is the only remaining polish item)
