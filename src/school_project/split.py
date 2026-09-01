"""Stratified real-data train/test split.

Restricted to Institution Type == University (56 of 61 cleaned rows).
Polytechnic (3 rows) and unknown institution type (2 rows) are too few to
model or evaluate separately — kept in data/cleaned_students.csv for
transparency but excluded here.

test_real.csv is never touched by any synthesis step downstream — it exists
purely to give an honest, all-real accuracy number later.
"""

import pandas as pd

IN_PATH = "data/cleaned_students.csv"
TRAIN_OUT = "data/train_real.csv"
TEST_OUT = "data/test_real.csv"

TEST_FRACTION = 0.2
RNG_SEED = 42


def split(in_path: str = IN_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(in_path)
    uni = df[df["institution_type"] == "University"].copy()
    uni = uni.dropna(subset=["target_tier"])
    uni["target_tier"] = uni["target_tier"].astype(int)

    rng = __import__("numpy").random.default_rng(RNG_SEED)
    train_idx, test_idx = [], []
    for tier, group in uni.groupby("target_tier"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        if len(idx) < 2:
            # can't put a class in both train and test with <2 examples
            train_idx.extend(idx)
            continue
        n_test = max(1, round(len(idx) * TEST_FRACTION))
        n_test = min(n_test, len(idx) - 1)  # always leave >=1 in train
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train = uni.loc[train_idx].reset_index(drop=True)
    test = uni.loc[test_idx].reset_index(drop=True)
    return train, test


def main() -> None:
    train, test = split()
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    print(f"train: {len(train)} rows, test: {len(test)} rows")
    print("\ntrain target_tier counts:")
    print(train["target_tier"].value_counts().sort_index())
    print("\ntest target_tier counts:")
    print(test["target_tier"].value_counts().sort_index())


if __name__ == "__main__":
    main()
