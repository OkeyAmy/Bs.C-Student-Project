"""Turn the normalized training dataset into modeling-ready X/y matrices.

Input: data/train_clean.csv — the consolidated file the model is actually
trained on. It holds real rows (is_synthetic=False) plus LLM and SMOTE-NC
synthetic rows, but NO test rows: data/test_real.csv is kept 100% real and
untouched, so it stays a valid honest holdout.

What this module does:
  * Drop non-feature columns (IDs, audit flags, near-constant columns,
    binned CGPA duplicates) and the current-semester outcome columns so a
    model can only see information available BEFORE the semester begins.
  * Separate the two targets (target_tier, at_risk) from the predictors.
  * Offer a drop-in sklearn transformer for training/evaluation.
"""

OUT_PATH = "data/train_matrix.csv"

ID_AND_AUDIT = [
    "student_id_hash",
    "institution_name_suspect",   # audit flag, True on 3 rows
    "carryover_prev_ambiguous",   # audit flag, True on 3 rows
    "passed_curr_corrected",      # audit flag, True on 1 row
]

# Constant (all values identical) -> no predictive signal.
CONSTANT = ["institution_type"]

# Binned duplicates of the CGPA tier they describe. Keep the tier columns.
BINNED_DUPES = [
    "prior_gpa_bin",
    "prior_cgpa_bin",
    "target_gpa_bin",
    "target_cgpa_bin",
]

PROVENANCE = ["is_synthetic", "generation_method"]

# Semester being predicted — outcomes, not inputs. registered_curr (courses
# enrolled this term) is known at semester start, so it stays a predictor.
OUTCOMES = ["target_tier", "at_risk"]
CURRENT_SEM_RESULTS = ["passed_curr", "failed_curr"]

# Everything that is known BEFORE the semester's results exist.
NUMERIC = ["prior_tier", "registered_prev", "passed_prev", "carryover_prev", "registered_curr"]
CATEGORICAL = [
    "institution_name",
    "faculty",
    # department and programme__course_of_study are intentionally EXCLUDED:
    # department (40) and programme (64) categories are near instance-level.
    # On real test they add zero accuracy (0.636 with vs 0.636 without) while
    # tripling the one-hot feature count (160 -> 56), so they only add noise.
    # faculty/institution/qualification already carry the grouping signal.
    "qualification_type",
    "programme_category",
    "current_academic_level",
    "avg_grade_prev",
]

DROP = ID_AND_AUDIT + CONSTANT + BINNED_DUPES + PROVENANCE + OUTCOMES + CURRENT_SEM_RESULTS


def load(path: str = "data/train_clean.csv") -> pd.DataFrame:
    import pandas as pd
    return pd.read_csv(path)


def build_frames(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (X, y_tier, y_at_risk) with all preprocessing applied."""
    missing = [c for c in NUMERIC + CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"expected predictor columns missing: {missing}")

    # Drop rows with no target label. The raw real file contains empty
    # placeholder rows (institution_name "Unknown", all fields NaN) that are
    # not students; keeping them crashes astype(int) on NaN labels.
    df = df.dropna(subset=["target_tier", "at_risk"])

    X = df[NUMERIC + CATEGORICAL].copy()
    y_tier = df["target_tier"].astype(int)
    y_at_risk = df["at_risk"].astype(int)
    return X, y_tier, y_at_risk


def build_clean_frames(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize categorical labels, then build X/y. Apply to train AND test
    before encoding so label casing/spacing matches between the two."""
    from school_project.normalize import normalize

    return build_frames(normalize(df))


def make_transformer(seed: int = 42):
    """sklearn ColumnTransformer for the feature matrix (no fitting)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


def main() -> None:
    import pandas as pd

    df = load()
    X, y_tier, y_at_risk = build_frames(df)

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("MISSING VALUES (not expected for this dataset):")
        print(missing)
    else:
        print(f"no missing values across {len(df)} rows")

    print(f"\ninput shape  : {df.shape}")
    print(f"feature X    : {X.shape}  ({len(NUMERIC)} numeric, {len(CATEGORICAL)} categorical)")
    print(f"predictors dropped as non-features")
    for name in sorted(set(DROP)):
        print(f"  - {name}")
    print(f"\ntarget 1 — target_tier (5-class CGPA tier)   n={len(y_tier)}")
    print(y_tier.value_counts().sort_index().to_string())
    print(f"\ntarget 2 — at_risk (binary)                 n={len(y_at_risk)}")
    print(y_at_risk.value_counts().sort_index().to_string())

    print("\nclasses per predictor (categorical cardinality):")
    for c in CATEGORICAL:
        print(f"  {c}: {df[c].nunique()}")

    out = df[OUTCOMES].copy()
    out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote targets to {OUT_PATH} ({out.shape})")


if __name__ == "__main__":
    main()
