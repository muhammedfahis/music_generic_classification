# Results — Custom CNN vs. Transfer Learning (VGG16)

All numbers are on the held-out **test split** (1,500 spectrogram chunks,
grouped-by-track so no source clip appears in both train and test — see
`methodology.md` §3).

## Headline Comparison

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Params | Best epoch |
|---|---|---|---|---|---|---|
| **Custom CNN (baseline)** | **76.40%** | 0.780 | 0.764 | **0.750** | 424K | 17 (of 19, early stopped) |
| VGG16 Transfer Learning | 71.07% | 0.720 | 0.711 | 0.709 | 14.8M | Fine-tune epoch 1 (of a planned ≤15) |

**Selected for deployment: the custom CNN.** Full metrics:
`reports/custom_cnn_metrics.json`, `reports/vgg16_transfer_metrics.json`.
Training curves: `reports/figures/baseline_cnn_training_curves.png`,
`reports/figures/vgg16_transfer_training_curves.png`. Confusion matrices:
`reports/figures/custom_cnn_confusion_matrix.png`,
`reports/figures/vgg16_transfer_confusion_matrix.png`.

## Why the from-scratch CNN won here

This is a somewhat counter-intuitive result — transfer learning usually
helps — so it's worth explaining rather than just reporting the number:

1. **Fine-tuning was cut short.** The transfer model's two-phase training
   (frozen-base head training, then unfreezing the top VGG16 layers at a low
   learning rate) only got 1 epoch of the fine-tuning phase before training
   was stopped due to wall-clock time constraints on the available hardware
   (Apple M1, ~500–940ms/step — considerably slower per-step than the small
   custom CNN, and slowing further once the base was unfrozen). The
   frozen-base phase alone (15 epochs) plateaued around 70% validation
   accuracy; the one fine-tuning epoch essentially matched that (69.72%),
   suggesting fine-tuning had not yet had the chance to meaningfully adapt
   VGG16's filters to spectrogram texture. Given more epochs, the transfer
   model would likely close some or all of this gap, consistent with the
   literature (`literature_review.md` §3).
2. **Domain mismatch + upscaling.** VGG16's ImageNet-pretrained filters are
   tuned for natural photographs; Mel-Spectrograms are a different kind of
   "image" (frequency vs. time, no natural edges/textures), so the frozen
   low-level filters are a weaker starting point here than in more typical
   transfer-learning wins. Additionally, feeding VGG16 required upsampling
   each 128×128 spectrogram to 224×224, introducing interpolation blur that
   the from-scratch CNN — trained natively at 128×128 — did not have to
   contend with.
3. **Parameter count vs. dataset size.** At 14.8M parameters against ~7,183
   training chunks, the transfer model has a far higher capacity-to-data
   ratio than the 424K-parameter custom CNN, making it more prone to
   overfitting on this comparatively small dataset even with dropout and
   early stopping in place.

This is still a meaningful and legitimate deliverable: it's a genuine
apples-to-apples comparison, and the result (with the fine-tuning caveat
above) is a reasonable finding to report as-is rather than one to be
"fixed" by cherry-picking. A natural follow-up (noted in the Conclusion) is
to let the fine-tuning phase run to convergence given more time/compute.

## Per-Class Performance

Both models struggle most on **rock**, which the confusion matrices show
being confused with disco, country, metal, and blues in both models — a
well-documented hard class in GTZAN, likely because "rock" as labeled in
this dataset spans a wide range of sub-styles that overlap acoustically
with several other genres.

- **Custom CNN** strongest classes: classical (F1 0.90), metal (F1 0.89),
  jazz (F1 0.85). Weakest: rock (F1 0.44), country (F1 0.63).
- **VGG16 transfer** strongest classes: classical (F1 0.93), jazz (F1 0.80),
  metal (F1 0.82). Weakest: rock (F1 0.45), reggae (F1 0.58).

Both models handle classical music best by a wide margin — consistent with
classical having the most acoustically distinct spectral signature (wide
dynamic range, sparse instrumentation, minimal percussion) among the 10
genres, visible even in the raw EDA waveform/spectrogram plots
(`reports/figures/eda_waveforms_spectrograms.png`).

## Deployment Decision

`app/streamlit_app.py` loads `models/checkpoints/custom_cnn_best.keras`
first (the better-performing, and also much smaller/faster-to-load, model),
falling back to the VGG16 checkpoint only if the baseline is absent.
