# Literature Review — Music Genre Classification

## 1. Foundational work: hand-engineered features

Tzanetakis and Cook (2002), *"Musical Genre Classification of Audio Signals,"*
IEEE Transactions on Speech and Audio Processing, 10(5), 293–302, is the
foundational paper for this problem and the origin of the GTZAN dataset used
in this project. They proposed three hand-engineered feature sets — timbral
texture (e.g. MFCCs, spectral centroid/rolloff), rhythmic content (beat
histograms), and pitch content — and classified genres with statistical
pattern recognition (Gaussian mixture models, k-NN), reporting around 61%
accuracy across 10 genres. This established both the benchmark dataset and
the core limitation this project addresses: hand-engineered features cap
how much of the audio's structure a classifier can exploit.

## 2. Shift to image-based deep learning (Mel-Spectrograms + CNNs)

The move from hand-engineered scalar features to treating audio as an
image — a Mel-Spectrogram — and applying Convolutional Neural Networks
directly to it is now the dominant approach in Music Information Retrieval
(MIR). By letting the CNN learn convolutional filters over frequency/time
directly, this approach captures local time-frequency patterns (onsets,
timbre texture, harmonic structure) that fixed hand-engineered statistics
tend to average away. This project follows that approach: raw audio to
Mel-Spectrogram (Librosa) to CNN, rather than replicating the original
GTZAN paper's feature-engineering pipeline.

## 3. Transfer learning for spectrogram classification

Several recent studies fine-tune ImageNet-pretrained CNNs (VGG16, ResNet50,
AlexNet) on Mel-Spectrograms for genre classification, treating the
spectrogram as a standard RGB-like image:

- Studies applying transfer learning with ResNet34/ResNet50, VGG16 and
  AlexNet on log-scaled Mel-Spectrograms from GTZAN report that pretrained
  ImageNet features transfer usefully to spectrogram classification, despite
  spectrograms looking nothing like natural photographs — the low-level
  filters (edges, textures) pretrained CNNs learn still pick up useful
  time-frequency structure.
- Hybrid architectures (e.g. VGG16 feature extraction followed by
  RNN/LSTM/GRU layers over the resulting feature sequence) have also been
  explored to additionally capture temporal dependencies across a clip,
  beyond what a plain CNN captures.
- Comparative studies of CNN vs. classical ML (e.g. XGBoost) on MFCCs vs.
  Mel-Spectrograms consistently find CNN-on-spectrogram approaches matching
  or exceeding classical hand-engineered-feature baselines.

This motivates this project's Phase 4 (Section 4 of the project plan):
comparing a custom CNN trained from scratch against a VGG16-based transfer
learning model, to quantify what pretrained ImageNet features add over a
small, task-specific CNN on this small (1,000-clip) dataset.

## 4. Known dataset limitations (GTZAN)

GTZAN, despite being the standard benchmark, has documented quality issues
that are relevant to interpreting this project's results: mislabeled tracks,
some exact or near-duplicate recordings across genre folders, and a small
overall size (100 clips/genre) that makes CNN training prone to overfitting
without chunking/augmentation. This project mitigates the small-sample-size
issue by chunking each 30s clip into 3s segments (Section 2 of the project
plan), and controls for the duplicate/leakage risk by splitting train/val/test
at the *track* level (not the chunk level) so segments of the same source
recording never cross a split boundary.

## References

1. Tzanetakis, G. and Cook, P. (2002). Musical Genre Classification of Audio
   Signals. *IEEE Transactions on Speech and Audio Processing*, 10(5),
   293–302. https://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf
2. Music Genre Classification using Transfer Learning on log-based Mel
   Spectrogram. https://www.researchgate.net/publication/351379331
3. Music Genre Recommendations Based on Spectrogram Analysis Using CNN with
   ResNet-50 and VGG-16 Architecture.
   https://www.researchgate.net/publication/361424096
4. Optimized Music Classification with a Hybrid VGG16-RNN Using
   Mel-Spectrogram and MFCC Features.
   https://www.researchgate.net/publication/388409534
5. Music Genre Classification: A Comparative Analysis of CNN and XGBoost
   Approaches with MFCCs and Mel-Spectrograms. https://arxiv.org/pdf/2401.04737
