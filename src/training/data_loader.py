"""tf.data pipelines built from the splits.csv manifest of .npy spectrograms."""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

SPLIT_PATH = Path("data/processed/splits.csv")

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}


def _load_npy(path, label):
    def _load(p):
        arr = np.load(p.numpy().decode("utf-8"))
        return arr.astype(np.float32)

    spec = tf.py_function(_load, [path], tf.float32)
    spec.set_shape((128, 128))
    spec = tf.expand_dims(spec, axis=-1)  # add channel dim -> (128,128,1)
    return spec, label


def make_dataset(split, batch_size=32, shuffle=False, splits_path=SPLIT_PATH):
    df = pd.read_csv(splits_path)
    df = df[df["split"] == split].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No rows found for split={split!r} in {splits_path}")

    paths = df["filepath"].tolist()
    labels = [GENRE_TO_IDX[g] for g in df["genre"]]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_load_npy, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(df)
