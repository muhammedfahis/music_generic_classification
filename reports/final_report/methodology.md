# Methodology

## 1. Dataset

GTZAN (Tzanetakis & Cook, 2002): 1,000 audio clips, 10 genres (blues,
classical, country, disco, hiphop, jazz, metal, pop, reggae, rock), 100
clips/genre, ~30s each, 22050 Hz mono WAV. Obtained via the `marsyas/gtzan`
mirror on Hugging Face (the Kaggle copy was not accessible in this
environment; the underlying audio is identical to the canonical GTZAN
release). One file, `jazz.00054.wav`, is corrupted in the source archive
(fails to decode under both `soundfile` and `audioread` backends) and was
excluded, leaving 999 usable tracks.

An EDA pass (`src/preprocessing/eda.py`) confirmed all usable clips share a
consistent 22050 Hz sample rate and 29.9–30.7s duration, and produced sample
waveform/spectrogram plots per genre (`reports/figures/eda_waveforms_spectrograms.png`)
showing visibly distinct time-domain and spectral characteristics across
genres (e.g. classical's wide dynamic range and silences vs. metal's
consistently high energy).

## 2. Preprocessing

Implemented in `src/preprocessing/`:

1. **Chunking** (`audio_utils.py`): each ~30s track is split into
   non-overlapping 3-second segments (10 chunks/track), discarding any
   trailing partial segment. This serves two purposes: it multiplies the
   number of training examples nearly 10x (999 tracks → 9,981 chunks total
   after excluding a handful of preprocessing failures on the same corrupt
   file), which is important given how small GTZAN is for CNN training; and
   it forces the model to classify genre from short local context rather
   than memorizing whole-track structure.

2. **Mel-Spectrogram generation** (`spectrogram.py`): for each 3s chunk,
   `librosa.feature.melspectrogram` computes a 128-band Mel-Spectrogram
   (`n_fft=2048`, `hop_length=512`), converted to a log (dB) scale via
   `librosa.power_to_db`. Each spectrogram is then min-max normalized to
   [0, 1] and resized to a fixed 128×128 via OpenCV (`cv2.resize`,
   `INTER_AREA`) for uniform CNN input.

3. **Storage**: each processed spectrogram is saved as an individual
   `.npy` array under `data/processed/mel_spectrograms/<genre>/`, with a
   manifest CSV (`data/processed/manifest.csv`) recording the filepath,
   genre, source `track_id`, and chunk index for every example.

## 3. Train / Validation / Test Split

A naive random split at the *chunk* level would leak information: multiple
3s chunks from the same 30s track would end up in different splits,
letting the model partly "recognize" a track rather than generalize across
genres. To avoid this, `src/preprocessing/make_splits.py` groups by
`track_id` using `sklearn.model_selection.GroupShuffleSplit`, applied
per-genre to preserve class balance, producing:

- Train: 7,183 chunks
- Validation: 1,298 chunks
- Test: 1,500 chunks

A leakage check (no `track_id` appears in more than one split) was run and
confirmed 0 leaking tracks.

## 4. Model Architectures

### 4.1 Custom baseline CNN (`src/models/custom_cnn.py`)

A CNN trained from scratch: 4 convolutional blocks (32→64→128→256 filters,
3×3 kernels, same padding, batch normalization, 2×2 max pooling after each),
followed by global average pooling, a 128-unit dense layer with 40% dropout,
and a 10-way softmax output. ~424K parameters. Trained with the Adam
optimizer and sparse categorical cross-entropy loss.

### 4.2 Transfer learning: VGG16 (`src/models/transfer_model.py`)

An ImageNet-pretrained VGG16 (`include_top=False`) is used as a frozen (or
partially fine-tuned) feature extractor. Since spectrograms are
single-channel and VGG16 expects 3-channel 224×224 input, the pipeline
resizes each 128×128 spectrogram to 224×224, replicates it across 3
channels, and applies VGG16's standard `preprocess_input` normalization
before the frozen convolutional base. A global average pooling layer, a
256-unit dense layer with 50% dropout, and a 10-way softmax head sit on top.
~14.8M parameters (mostly frozen in the base during head training).

Training proceeds in two phases:
- **Phase A (head-only):** VGG16 base fully frozen, train only the new
  head, learning rate 1e-3.
- **Phase B (fine-tuning):** unfreeze the base's later layers (from layer
  index 15 onward), recompile with a much lower learning rate (1e-5), and
  continue training so the pretrained low-level filters are preserved while
  higher-level filters adapt to spectrogram texture.

## 5. Training Setup

Both models use:
- `tf.data` pipelines (`src/training/data_loader.py`) reading `.npy`
  spectrograms directly from the split manifest, batched (32) and
  prefetched.
- `EarlyStopping` on validation loss (patience 8 for the baseline; patience
  5 per phase for transfer learning) to prevent overfitting on this
  comparatively small dataset.
- `ModelCheckpoint` saving only the best validation-accuracy epoch to
  `models/checkpoints/`.
- Training/validation loss and accuracy curves logged to
  `reports/figures/`.

Hardware: Apple M1 (8GB), TensorFlow 2.16 with the `tensorflow-metal` GPU
plugin.

## 6. Evaluation

`src/evaluation/metrics.py` evaluates a trained model on the held-out test
split: overall accuracy, macro-averaged precision/recall/F1, a per-class
classification report, and a confusion matrix (saved as a heatmap). The
same script is run once per model (baseline CNN, VGG16 transfer) to produce
a direct, apples-to-apples comparison — this comparison is one of the
project's explicit deliverables.

## 7. Deployment

The best-performing model (by test accuracy) is loaded into a Streamlit app
(`app/streamlit_app.py`) that accepts an uploaded audio file, runs it
through the same preprocessing pipeline (chunking → Mel-Spectrogram →
normalize → resize), averages the model's per-chunk genre probabilities
across the whole clip, and displays the predicted genre, a confidence bar
chart across all 10 genres, and a visualization of the clip's spectrogram.
