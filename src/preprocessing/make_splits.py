"""Build a leakage-free, stratified train/val/test split over the manifest.

Splitting is grouped by `track_id` so that chunks from the same original
30s clip never appear in more than one split (otherwise the model could
"cheat" by recognizing the same song across splits).

Usage:
    venv/bin/python -m src.preprocessing.make_splits
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

MANIFEST_PATH = Path("data/processed/manifest.csv")
SPLIT_PATH = Path("data/processed/splits.csv")

TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the remaining train+val pool
RANDOM_STATE = 42


def split_group(df, group_col, test_size, random_state):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    return df.iloc[train_idx], df.iloc[test_idx]


def main():
    df = pd.read_csv(MANIFEST_PATH)

    # Split per-genre so genre balance is preserved in each split, but the
    # grouping is still by track_id within each genre.
    train_parts, val_parts, test_parts = [], [], []
    for genre, gdf in df.groupby("genre"):
        trainval, test = split_group(gdf, "track_id", TEST_SIZE, RANDOM_STATE)
        train, val = split_group(trainval, "track_id", VAL_SIZE, RANDOM_STATE)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = pd.concat(train_parts).assign(split="train")
    val_df = pd.concat(val_parts).assign(split="val")
    test_df = pd.concat(test_parts).assign(split="test")

    out = pd.concat([train_df, val_df, test_df]).sort_values(["genre", "track_id", "chunk_idx"])
    out.to_csv(SPLIT_PATH, index=False)

    print(out.groupby(["split", "genre"]).size().unstack(fill_value=0))
    print(f"\nSplit sizes: {out.groupby('split').size().to_dict()}")

    # sanity check: no track_id appears in more than one split
    leak = (
        out.groupby("track_id")["split"].nunique().gt(1).sum()
    )
    print(f"Tracks leaking across splits: {leak} (should be 0)")

    print(f"\nSplits written to {SPLIT_PATH}")


if __name__ == "__main__":
    main()
