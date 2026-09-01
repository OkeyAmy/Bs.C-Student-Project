"""Rebuild the smart-augmented training set.

Fix for the SMOTE mis-targeting that broke the test score:

  * REAL data is the source of truth: tier {0:1, 1:2, 2:10, 3:26, 4:6} —
    tier-3 is the majority (26), tier-0 (1) and tier-1 (2) are the
    undersampled minority (exactly the at-risk classes).
  * Old synth_stat.py skipped tier-0 entirely (SMOTE can't interpolate a
    class with 1 real example) and the ad-hoc "at-risk bridge" later bloated
    tier-1 to 110, which skewed the model toward false at-risk on the test
    set.
  * Fix: hand-authored rows (data/synthetic_hand.csv) provide the missing
    minority seeds (tier0=6, tier1=14). Blending real+hand gives enough
    same-class examples for SMOTE to interpolate, so tier-0 and tier-1 get a
    MODEST top-up — anchored to the majority tier instead of being blown up.

Composition:
  base = train_real (45) + synthetic_hand (40)     -> 85 rows
  base already has tier {0:7, 1:16, 2:19, 3:32, 4:11}
  SMOTE-NC tops up minority tiers:
      tier0 -> 14   (from 7)
      tier1 -> 24   (from 16)
      tier4 -> 20   (from 11)
  then + synthetic_llm (60)                        -> final

All rows live in the full 28-column schema (same as train_augmented.csv) so
train.py + evaluation.py (build_clean_frames) consume it directly.

Writes: data/train_smart_augmented.csv (overwrites the 21-col 206-row file)
"""

import pandas as pd
from imblearn.over_sampling import SMOTENC

REAL = "data/train_real.csv"
HAND = "data/synthetic_hand.csv"
LLM = "data/synthetic_llm.csv"
OUT = "data/train_smart_augmented.csv"

# modest per-tier ABSOLUTE target counts after SMOTE (anchored to majority 32)
SMOTE_TARGETS = {0: 14, 1: 24, 4: 20}

NUMERIC_COLS = [
    "prior_tier", "registered_prev", "passed_prev", "carryover_prev",
    "registered_curr", "passed_curr", "failed_curr",
]
CATEGORICAL_COLS = [
    "institution_name", "faculty", "department", "programme__course_of_study",
    "qualification_type", "programme_category", "current_academic_level",
    "prior_gpa_bin", "prior_cgpa_bin", "avg_grade_prev",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def smote_topup(base: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["target_tier"] = df["target_tier"].astype(int)
    X = df[FEATURE_COLS].copy()
    y = df["target_tier"].to_numpy()

    encoders = {}
    for col in CATEGORICAL_COLS:
        codes, uniques = pd.factorize(X[col])
        X[col] = codes
        encoders[col] = uniques
    for col in NUMERIC_COLS:
        X[col] = X[col].astype(float)

    cat_idx = [FEATURE_COLS.index(c) for c in CATEGORICAL_COLS]
    sm = SMOTENC(categorical_features=cat_idx,
                 sampling_strategy=SMOTE_TARGETS, k_neighbors=2, random_state=42)
    X_res, y_res = sm.fit_resample(X.to_numpy(), y)

    res = pd.DataFrame(X_res, columns=FEATURE_COLS)
    for col in NUMERIC_COLS:
        res[col] = res[col].astype(float).round().astype(int)
    for col in CATEGORICAL_COLS:
        res[col] = encoders[col][res[col].astype(int).to_numpy()]
    res["target_tier"] = y_res.astype(int)

    synthetic = res.iloc[len(df):].reset_index(drop=True)
    synthetic["passed_curr"] = synthetic[["passed_curr", "registered_curr"]].min(axis=1)
    synthetic["failed_curr"] = (synthetic["registered_curr"] - synthetic["passed_curr"]).clip(lower=0)
    synthetic["student_id_hash"] = [f"synth-smote-{i:03d}" for i in range(len(synthetic))]
    synthetic["institution_name_suspect"] = False
    synthetic["institution_type"] = "University"
    synthetic["carryover_prev_ambiguous"] = False
    synthetic["passed_curr_corrected"] = False
    synthetic["is_synthetic"] = True
    synthetic["generation_method"] = "smote_nc"
    synthetic["at_risk"] = synthetic["target_tier"] <= 1
    uni_bins = ["0.00 - 0.99", "1.50 - 2.39", "2.40 - 3.49", "3.50 - 4.49", "4.50 - 5.00"]
    synthetic["target_cgpa_bin"] = synthetic["target_tier"].map(dict(enumerate(uni_bins)))
    synthetic["target_gpa_bin"] = synthetic["target_cgpa_bin"]
    return synthetic


def main() -> None:
    real = pd.read_csv(REAL)
    hand = pd.read_csv(HAND)
    llm = pd.read_csv(LLM)

    # real rows carry generation_method='real'
    real["generation_method"] = "real"

    base = pd.concat([real, hand], ignore_index=True)
    print("base (real+hand):", base.shape,
          "tier", base["target_tier"].value_counts().sort_index().to_dict())

    smote = smote_topup(base)
    print("smote synthetic:", smote.shape,
          "tier", smote["target_tier"].value_counts().sort_index().to_dict())

    final = pd.concat([base, smote, llm], ignore_index=True)
    # canonical 28-col order (aligned to synthetic_llm.csv which has all 28)
    final = final[list(llm.columns)]
    final.to_csv(OUT, index=False)

    print("\nFINAL train_smart_augmented.csv:", final.shape)
    print("tier :", final["target_tier"].value_counts().sort_index().to_dict())
    print("at_risk :", final["at_risk"].value_counts().to_dict())
    print("columns:", len(final.columns))


if __name__ == "__main__":
    main()