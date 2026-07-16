# Project Plan — Deep Learning-Based Music Genre Classification

**Student:** Mohammed Fayis CC (AA.SC.P2MCA25010058)
**Course:** Minor Project (21CSA697A)
**Source:** `Music_Genre_Classification_Proposal.pdf`

## 1. Goal

Classify 30-second audio clips into one of 10 genres (blues, classical, country,
disco, hip-hop, jazz, metal, pop, reggae, rock) by converting audio to
Mel-Spectrogram images and training CNNs on them, comparing a custom CNN
against transfer learning (VGG16 / ResNet50), then shipping a Streamlit demo.

## 2. Architecture / Pipeline

```
GTZAN audio (.wav, 30s clips, 10 genres x 100 clips)
        │
        ▼
Librosa: load audio → compute Mel-Spectrogram → convert to dB scale
        │
        ▼
OpenCV: resize / normalize spectrogram images → save as arrays or PNGs
        │
        ▼
Train/val/test split (stratified by genre)
        │
        ├──► Custom CNN (baseline, trained from scratch)
        │
        └──► Transfer Learning (VGG16 / ResNet50, frozen base + fine-tuned head)
        │
        ▼
Evaluation: accuracy, precision, recall, F1, confusion matrix
        │
        ▼
Best model exported (.h5 / SavedModel)
        │
        ▼
Streamlit app: upload audio → live spectrogram → genre prediction
```

## 3. Repo Structure

```
pg-minor-project/
├── data/
│   ├── raw/                     # GTZAN audio files (not committed — see .gitignore)
│   └── processed/
│       └── mel_spectrograms/    # generated spectrogram images/arrays
├── notebooks/                   # exploration, EDA, prototyping
├── src/
│   ├── preprocessing/           # audio loading, spectrogram generation
│   ├── models/                  # CNN architectures (custom + transfer learning)
│   ├── training/                # training loops, config, callbacks
│   └── evaluation/               # metrics, confusion matrix, reports
├── models/
│   └── checkpoints/             # saved trained models
├── reports/
│   ├── figures/                 # plots (accuracy/loss curves, confusion matrices)
│   └── final_report/            # final written report
├── app/                          # Streamlit web application
├── PROJECT_PLAN.md
├── PROGRESS_TRACKER.md
└── requirements.txt
```

## 4. Phased Plan (maps to proposal's 12-week timeline)

### Phase 1 — Literature Review & Dataset Collection (Weeks 1–2)
- Review MIR / genre classification literature (short summary, 1–2 pages, for final report).
- Download GTZAN dataset from Kaggle into `data/raw/`.
- Verify integrity: 10 genre folders × 100 clips × 30s, check for the known corrupt
  file (`jazz.00054.wav` is a common GTZAN gotcha — confirm it loads with Librosa).
- Quick EDA: waveform plots, duration/sample-rate check per genre.

### Phase 2 — Preprocessing & Mel-Spectrogram Generation (Weeks 3–4)
- `src/preprocessing/audio_utils.py`: load audio with Librosa, standard sample rate.
- `src/preprocessing/spectrogram.py`: compute Mel-Spectrogram (`librosa.feature.melspectrogram`),
  convert to dB (`librosa.power_to_db`).
- Decide: fixed-size spectrogram (resize via OpenCV) vs. chunking each 30s clip
  into shorter segments (e.g. 3s) to multiply training samples — recommended,
  since 1,000 clips is small for CNN training.
- Save processed spectrograms as `.npy` arrays or PNGs to `data/processed/mel_spectrograms/`.
- Build stratified train/val/test split (e.g. 70/15/15), save split indices/manifest.

### Phase 3 — Custom CNN Design & Training (Weeks 5–7)
- `src/models/custom_cnn.py`: baseline CNN (Conv2D/MaxPool blocks → Dense → softmax(10)).
- `src/training/train_baseline.py`: training loop with data generators/`tf.data`,
  augmentation (optional: SpecAugment-style time/freq masking), early stopping, checkpointing.
- Track training curves (loss/accuracy) via Matplotlib, log to `reports/figures/`.
- Establish baseline accuracy as reference point for transfer learning comparison.

### Phase 4 — Transfer Learning: VGG16 / ResNet50 (Weeks 8–9)
- `src/models/transfer_model.py`: load pre-trained VGG16/ResNet50 (ImageNet weights,
  `include_top=False`), add custom classification head, adapt input shape/channels
  (spectrograms are single-channel — replicate to 3 channels or adapt first layer).
- Phase A: freeze base, train head only.
- Phase B: unfreeze top N layers, fine-tune with low learning rate.
- Compare VGG16 vs ResNet50 if time allows; otherwise pick one based on Phase A results.

### Phase 5 — Model Evaluation (Week 10)
- `src/evaluation/metrics.py`: accuracy, precision, recall, F1 (macro + per-class),
  confusion matrix (Scikit-learn + Seaborn heatmap).
- Compare custom CNN vs. transfer learning model side by side — this comparison
  table is an explicit deliverable in the proposal.
- Identify commonly confused genre pairs (e.g. rock/metal, disco/pop) for discussion in report.
- Select final model to ship in the app.

### Phase 6 — Streamlit Web App (Week 11)
- `app/streamlit_app.py`: file upload → run same preprocessing pipeline → predict → display
  genre + confidence scores (bar chart) + spectrogram visualization of the uploaded clip.
- Load the exported best model from `models/checkpoints/`.
- Basic input validation (file type/duration) and error handling for bad uploads.

### Phase 7 — Testing, Bug Fixes, Final Report (Week 12)
- End-to-end manual test: fresh audio clips (not in GTZAN) through the app.
- Fix edge cases (very short clips, non-wav formats, mono/stereo).
- Write final report: methodology, architecture diagrams, experiments, results
  tables, confusion matrices, screenshots of the app, conclusion & future work.
- Clean up repo, add README with setup/run instructions.

## 5. Key Technical Decisions to Make Early
- **Segment length**: whole 30s clips vs. 3s chunks (chunking is standard practice
  for GTZAN + CNNs and strongly recommended given the small dataset size).
- **Spectrogram format**: save as `.npy` (faster, more precise) vs. `.png` (easier
  to inspect, directly reusable with Keras `ImageDataGenerator`/`image_dataset_from_directory`).
- **Input shape** for transfer learning models (fixed 224×224×3 for VGG16/ResNet50
  vs. custom shape for the baseline CNN).
- **Framework**: TensorFlow/Keras (as specified in proposal tools list).

## 6. Risks / Watch-outs
- GTZAN has a known corrupted file and near-duplicate tracks across "genres" in
  some copies — validate the dataset copy used.
- 1,000 clips total is small; without chunking/augmentation, overfitting is likely,
  especially for transfer learning models with many parameters.
- Train/val/test leakage: if chunking clips into segments, make sure segments from
  the same original track don't span across split boundaries.

## 7. Tools (from proposal)
Python 3.10+, TensorFlow/Keras, Librosa, OpenCV, NumPy/Pandas, Matplotlib/Seaborn,
Scikit-learn, Streamlit, Jupyter Notebook, GTZAN Dataset (Kaggle).
