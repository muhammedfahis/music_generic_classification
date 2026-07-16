"""Train the VGG16 transfer-learning model: frozen-base phase, then fine-tuning.

Usage:
    venv/bin/python -m src.training.train_transfer
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models.transfer_model import build_transfer_model
from src.training.data_loader import make_dataset

CHECKPOINT_PATH = Path("models/checkpoints/vgg16_transfer_best.keras")
FIGURES_DIR = Path("reports/figures")
BATCH_SIZE = 32
HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 15
FINE_TUNE_AT = 15  # unfreeze VGG16 layers from this index onward


def plot_history(history_head, history_ft, out_path):
    loss = history_head.history["loss"] + history_ft.history["loss"]
    val_loss = history_head.history["val_loss"] + history_ft.history["val_loss"]
    acc = history_head.history["accuracy"] + history_ft.history["accuracy"]
    val_acc = history_head.history["val_accuracy"] + history_ft.history["val_accuracy"]
    split_epoch = len(history_head.history["loss"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(loss, label="train")
    axes[0].plot(val_loss, label="val")
    axes[0].axvline(split_epoch, color="gray", linestyle="--", label="fine-tune start")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(acc, label="train")
    axes[1].plot(val_acc, label="val")
    axes[1].axvline(split_epoch, color="gray", linestyle="--", label="fine-tune start")
    axes[1].set_title("Accuracy")
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

    # Phase A: frozen base, train head only
    model = build_transfer_model(fine_tune_at=None)
    model.summary()

    callbacks_head = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    history_head = model.fit(
        train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, callbacks=callbacks_head
    )

    # Phase B: unfreeze top layers of the SAME model's base and continue
    # training (so the head weights learned in Phase A are preserved),
    # with a low learning rate for fine-tuning.
    base = model.get_layer("vgg16")
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_ft = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(str(CHECKPOINT_PATH), monitor="val_accuracy", save_best_only=True),
    ]
    history_ft = model.fit(
        train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks_ft
    )

    plot_history(history_head, history_ft, FIGURES_DIR / "vgg16_transfer_training_curves.png")
    print(f"Best model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
