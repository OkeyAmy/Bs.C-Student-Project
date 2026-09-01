"""Does the augmentation actually help? Real-only vs augmented training,
both scored on the same untouched real test set.

This is the check that decides whether augmentation goes in the thesis as a
benefit or as a reported null result. Run it before writing any results up.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42

# Features available BEFORE the semester's results exist.
# passed_curr / failed_curr / target_* are outcomes of the semester being
# predicted — including them would be describing, not predicting.
NUMERIC = ["prior_tier", "registered_prev", "passed_prev", "carryover_prev", "registered_curr"]
CATEGORICAL = ["institution_name", "faculty", "qualification_type", "current_academic_level", "avg_grade_prev"]


def make_model(seed: int = SEED) -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced", random_state=seed)),
    ])


def score(train: pd.DataFrame, test: pd.DataFrame, target: str, seed: int) -> float:
    model = make_model(seed)
    model.fit(train[NUMERIC + CATEGORICAL], train[target])
    return (model.predict(test[NUMERIC + CATEGORICAL]) == test[target]).mean()


def main() -> None:
    train_real = pd.read_csv("data/train_real.csv")
    train_aug = pd.read_csv("data/train_augmented.csv")
    test = pd.read_csv("data/test_real.csv")

    print(f"train_real={len(train_real)}  train_augmented={len(train_aug)}  test_real={len(test)}\n")

    for target, label in [("target_tier", "OBJECTIVE 1 — 5-class CGPA tier"),
                          ("at_risk", "OBJECTIVE 2 — binary at-risk")]:
        print(f"=== {label} ===")

        persistence = (test["prior_tier"] == test["target_tier"]).mean()
        majority_cls = train_real[target].mode()[0]
        majority = (test[target] == majority_cls).mean()

        # average over seeds — with n=11 a single seed is noise
        seeds = range(10)
        real_scores = [score(train_real, test, target, s) for s in seeds]
        aug_scores = [score(train_aug, test, target, s) for s in seeds]

        print(f"  majority-class baseline : {majority:.3f}")
        if target == "target_tier":
            print(f"  persistence baseline    : {persistence:.3f}")
        print(f"  model, real-only train  : {np.mean(real_scores):.3f} (sd {np.std(real_scores):.3f})")
        print(f"  model, augmented train  : {np.mean(aug_scores):.3f} (sd {np.std(aug_scores):.3f})")
        delta = np.mean(aug_scores) - np.mean(real_scores)
        print(f"  augmentation effect     : {delta:+.3f}")
        if target == "at_risk":
            print(f"  NOTE: test set has only {int(test[target].sum())} real at-risk row(s) "
                  f"of {len(test)} — treat this number as indicative only")
        print()

    print("Reminder: n_test = 11. Differences below ~0.09 are within one test row.")

    # The 11-row test set is too small to trust on its own. Repeated
    # stratified CV over all 56 real University rows gives a stable estimate
    # of how much real signal exists beyond the persistence rule.
    # Augmented CV is deliberately NOT reported: the synthetic rows were
    # derived from real rows that would land in test folds, which leaks.
    from sklearn.model_selection import RepeatedStratifiedKFold

    print("\n=== Repeated stratified 5-fold CV, REAL data only (n=56) ===")
    allreal = pd.read_csv("data/cleaned_students.csv")
    allreal = allreal[(allreal.institution_type == "University") & allreal.target_tier.notna()].copy()
    allreal["target_tier"] = allreal["target_tier"].astype(int)
    allreal["at_risk"] = allreal["at_risk"].astype(bool)

    from sklearn.metrics import balanced_accuracy_score

    for target in ["target_tier", "at_risk"]:
        y = allreal[target]
        smallest = y.value_counts().min()
        if target == "target_tier":
            # tier 0 has 1 member — cannot be stratified into any fold
            sub = allreal[y.map(y.value_counts()) >= 5]
            n_splits, metric = 5, "accuracy"
        else:
            # keep all 4 at-risk students; fold count limited by that count
            sub = allreal
            n_splits, metric = min(4, smallest), "balanced accuracy"

        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=6, random_state=SEED)
        scores, pers = [], []
        for tr, te in cv.split(sub, sub[target]):
            trd, ted = sub.iloc[tr], sub.iloc[te]
            model = make_model(SEED)
            model.fit(trd[NUMERIC + CATEGORICAL], trd[target])
            pred = model.predict(ted[NUMERIC + CATEGORICAL])
            scores.append(balanced_accuracy_score(ted[target], pred)
                          if metric == "balanced accuracy"
                          else (pred == ted[target]).mean())
            pers.append((ted["prior_tier"] == ted["target_tier"]).mean())
        lo, hi = np.percentile(scores, [2.5, 97.5])
        print(f"  {target:12s} ({metric}) model {np.mean(scores):.3f} "
              f"[95% {lo:.3f}-{hi:.3f}] | persistence {np.mean(pers):.3f} | "
              f"n={len(sub)}, folds={n_splits}")
        if target == "at_risk":
            print(f"    (only {int(y.sum())} at-risk students exist in the real data; "
                  "0.5 = no better than chance)")

    # Does augmentation rescue at-risk detection? Same CV folds over real
    # rows, but each training fold is topped up with the hand-authored LLM
    # rows. SMOTE rows are excluded here: they interpolate specific real
    # students who may sit in the test fold, which would leak. The LLM rows
    # are authored from scratch against tier-level patterns, not copied from
    # individual respondents.
    print("\n=== At-risk detection: real-only vs +LLM-synthetic training ===")
    print("Three schemes, most to least conservative. Balanced accuracy; 0.5 = chance.\n")
    llm = pd.read_csv("data/synthetic_llm.csv")
    llm["at_risk"] = llm["at_risk"].astype(bool)
    train_real["at_risk"] = train_real["at_risk"].astype(bool)
    test["at_risk"] = test["at_risk"].astype(bool)

    def cv_compare(pool: pd.DataFrame, n_splits: int) -> tuple[float, float]:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=8, random_state=SEED)
        plain, boosted = [], []
        for tr, te in cv.split(pool, pool["at_risk"]):
            trd, ted = pool.iloc[tr], pool.iloc[te]
            for bucket, ts in [(plain, trd), (boosted, pd.concat([trd, llm], ignore_index=True))]:
                model = make_model(SEED)
                model.fit(ts[NUMERIC + CATEGORICAL], ts["at_risk"])
                bucket.append(balanced_accuracy_score(
                    ted["at_risk"], model.predict(ted[NUMERIC + CATEGORICAL])))
        return float(np.mean(plain)), float(np.mean(boosted))

    def holdout_compare() -> tuple[float, float]:
        out = []
        for ts in [train_real, pd.concat([train_real, llm], ignore_index=True)]:
            model = make_model(SEED)
            model.fit(ts[NUMERIC + CATEGORICAL], ts["at_risk"])
            out.append(balanced_accuracy_score(
                test["at_risk"], model.predict(test[NUMERIC + CATEGORICAL])))
        return out[0], out[1]

    schemes = [
        ("CV over train_real only (45 rows, 3 at-risk)  [HEADLINE]", *cv_compare(train_real, 3)),
        ("CV over all 56 real rows (4 at-risk)", *cv_compare(allreal, 4)),
        (f"held-out test_real (11 rows, {int(test['at_risk'].sum())} at-risk)", *holdout_compare()),
    ]
    print(f"  {'scheme':<58} {'real':>6} {'+LLM':>6} {'effect':>8}")
    for name, a, b in schemes:
        print(f"  {name:<58} {a:>6.3f} {b:>6.3f} {b - a:>+8.3f}")
    print("\n  Headline = the most conservative scheme. The held-out row has a")
    print("  single positive example and is shown for completeness only.")


if __name__ == "__main__":
    main()
