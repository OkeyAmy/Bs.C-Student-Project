"""SMOTE-NC statistical synthesis of the minority-class portion of
train_real.csv. Fit on train only — data/test_real.csv is never touched.

Per-class target counts below are chosen deliberately, not to full class
parity: bringing tier1/tier2/tier4 fully up to the majority (26) would make
those classes mostly-fabricated. Instead each minority class gets a bounded
top-up (roughly 40% of the total top-up per class; the rest comes from the
LLM-generated pool in synth_llm.py), and the majority class (tier3) is left
untouched. tier0 has only 1 real example — SMOTE needs >=2 real examples of
a class to interpolate, so tier0 gets zero SMOTE rows here; it's covered
entirely by the LLM pool instead.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

IN_PATH = "data/train_real.csv"
OUT_PATH = "data/synthetic_stat.csv"
RNG_SEED = 42

# {target_tier: (real_count, smote_target_count)}
# smote_target_count is the ABSOLUTE count after SMOTE for that class.
SMOTE_TARGETS = {
    1: 5,  # real=2  -> +3
    2: 13,  # real=10 -> +3
    4: 9,  # real=6  -> +3
}

NUMERIC_COLS = [
    "prior_tier",
    "registered_prev",
    "passed_prev",
    "carryover_prev",
    "registered_curr",
    "passed_curr",
    "failed_curr",
]
CATEGORICAL_COLS = [
    "institution_name",
    "faculty",
    "department",
    "programme__course_of_study",
    "qualification_type",
    "programme_category",
    "current_academic_level",
    "prior_gpa_bin",
    "prior_cgpa_bin",
    "avg_grade_prev",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def run(in_path: str = IN_PATH) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    df["target_tier"] = df["target_tier"].astype(int)

    X = df[FEATURE_COLS].copy()
    y = df["target_tier"].to_numpy()

    # factorize categoricals to ints; keep the mapping to decode after
    encoders = {}
    for col in CATEGORICAL_COLS:
        codes, uniques = pd.factorize(X[col])
        X[col] = codes
        encoders[col] = uniques

    for col in NUMERIC_COLS:
        X[col] = X[col].astype(float)

    cat_idx = [FEATURE_COLS.index(c) for c in CATEGORICAL_COLS]

    sm = SMOTENC(
        categorical_features=cat_idx,
        sampling_strategy=SMOTE_TARGETS,
        k_neighbors=1,
        random_state=RNG_SEED,
    )
    X_res, y_res = sm.fit_resample(X.to_numpy(), y)

    res = pd.DataFrame(X_res, columns=FEATURE_COLS)
    for col in NUMERIC_COLS:
        res[col] = res[col].astype(float).round().astype(int)
    for col in CATEGORICAL_COLS:
        res[col] = encoders[col][res[col].astype(int).to_numpy()]
    res["target_tier"] = y_res

    # keep only the newly generated rows (original 45 are first in output)
    synthetic = res.iloc[len(df) :].reset_index(drop=True)

    # interpolating passed_curr/failed_curr independently can violate
    # passed + failed <= registered; re-derive failed from the other two.
    synthetic["passed_curr"] = synthetic[["passed_curr", "registered_curr"]].min(axis=1)
    synthetic["failed_curr"] = (
        synthetic["registered_curr"] - synthetic["passed_curr"]
    ).clip(lower=0)

    synthetic["institution_type"] = "University"
    synthetic["is_synthetic"] = True
    synthetic["generation_method"] = "smote_nc"
    synthetic["student_id_hash"] = [
        f"synth-smote-{i:03d}" for i in range(len(synthetic))
    ]

    uni_bins = ["0.00 - 0.99", "1.50 - 2.39", "2.40 - 3.49", "3.50 - 4.49", "4.50 - 5.00"]
    synthetic["target_cgpa_bin"] = synthetic["target_tier"].map(dict(enumerate(uni_bins)))
    synthetic["target_gpa_bin"] = synthetic["target_cgpa_bin"]
    synthetic["institution_name_suspect"] = False
    synthetic["carryover_prev_ambiguous"] = False
    synthetic["passed_curr_corrected"] = False
    synthetic["at_risk"] = synthetic["target_tier"] <= 1
    return synthetic


def main() -> None:
    synthetic = run()
    synthetic.to_csv(OUT_PATH, index=False)
    print(f"generated {len(synthetic)} SMOTE-NC synthetic rows -> {OUT_PATH}")
    print(synthetic["target_tier"].value_counts().sort_index())


if __name__ == "__main__":
    main()
