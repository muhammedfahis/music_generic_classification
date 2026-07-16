"""Train the custom baseline CNN on preprocessed Mel-Spectrograms.

Usage:
    venv/bin/python -m src.training.train_baseline
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models.custom_cnn import build_custom_cnn
from src.training.data_loader import make_dataset

CHECKPOINT_PATH = Path("models/checkpoints/custom_cnn_best.keras")
FIGURES_DIR = Path("reports/figures")
EPOCHS = 40
BATCH_SIZE = 32


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def main():
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, n_train = make_dataset("train", batch_size=BATCH_SIZE, shuffle=True)
    val_ds, n_val = make_dataset("val", batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train examples: {n_train}, Val examples: {n_val}")

    model = build_custom_cnn()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(str(CHECKPOINT_PATH), monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_history(history, FIGURES_DIR / "baseline_cnn_training_curves.png")
    print(f"Best model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
