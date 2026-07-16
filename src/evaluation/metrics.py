"""Evaluate a trained model on the held-out test split.

Usage:
    venv/bin/python -m src.evaluation.metrics --model models/checkpoints/custom_cnn_best.keras --name custom_cnn
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.training.data_loader import GENRES, make_dataset

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports")


def load_trained_model(model_path):
    try:
        return tf.keras.models.load_model(model_path, safe_mode=False)
    except (ValueError, NotImplementedError):
        # Checkpoints saved by an earlier version of transfer_model.py used a
        # Lambda layer that Keras 3 cannot safely reconstruct from config.
        # The weights themselves are unaffected, so rebuild the (now-fixed)
        # architecture in code and load just the weights.
        from src.models.transfer_model import build_transfer_model

        model = build_transfer_model(fine_tune_at=15)
        model.load_weights(model_path)
        return model


def evaluate(model_path, name, split="test"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(model_path)
    ds, n = make_dataset(split, batch_size=32, shuffle=False)

    y_true, y_pred = [], []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(y.numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=GENRES, zero_division=0)
    print(f"\n=== {name} ({split} split, n={n}) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1 (macro):        {f1:.4f}\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=GENRES, yticklabels=GENRES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    cm_path = FIGURES_DIR / f"{name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix to {cm_path}")

    metrics = {
        "model": name,
        "split": split,
        "n_examples": int(n),
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    metrics_path = RESULTS_DIR / f"{name}_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    evaluate(args.model, args.name, args.split)


if __name__ == "__main__":
    main()
