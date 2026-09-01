"""Train the final model, save it for hosting (.pkl), and report a full set
of sklearn performance metrics on the fully-synthetic demo holdout.

Architecture (validated by analysis):
  * SINGLE model predicts target_tier (5 ordered CGPA tiers) via a tuned
    RBF SVM (C=3, gamma=0.03) over the full feature set.
  * at_risk is DERIVED from the prediction as `target_tier <= 1`; the
    corpus label matches that definition with a small irreducible noise
    term (LABEL_FLIP in build_corpus.py) so the task stays honest.
  * Trained and evaluated on data/corpus_train.csv / corpus_test.csv
    (fully-synthetic 560-row demo corpus, 80:20 split).

Outputs:
  * models/student_predictor.pkl        (hostable serialized model)
  * Metrics printed for BOTH targets:
      - target_tier (5-class): accuracy, balanced accuracy, macro/micro/
        weighted F1, precision, recall, Cohen's kappa, confusion matrix,
        per-class report.
      - at_risk (binary, derived): accuracy, precision, recall, F1,
        balanced accuracy, Matthews correlation, ROC-AUC from decision
        function, confusion matrix, classification report.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from school_project.preprocess import CATEGORICAL, NUMERIC, build_clean_frames

FEATURES = NUMERIC + CATEGORICAL

TRAIN_PATH = "data/corpus_train.csv"
TEST_PATH = "data/corpus_test.csv"
MODEL_PATH = "models/student_predictor.pkl"


def build_trainer(seed: int = 42) -> Pipeline:
    """Train pipeline: preprocess selected features + tuned RBF SVM."""
    transformer = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), [f for f in FEATURES if f in NUMERIC]),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), [f for f in FEATURES if f in CATEGORICAL]),
    ])
    return Pipeline([
        ("pre", transformer),
        ("clf", SVC(C=3.0, gamma=0.03, class_weight="balanced", random_state=seed)),
    ])


def collect_y(df, col):
    return np.asarray(df[col].astype(float))


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train, y_tier_train, _ = build_clean_frames(train)
    X_test, y_tier_test, y_at_risk_test = build_clean_frames(test)

    X_train = X_train[FEATURES]
    X_test = X_test[FEATURES]

    model = build_trainer()
    model.fit(X_train, y_tier_train)

    pred_tier = model.predict(X_test)
    pred_at_risk = (pred_tier <= 1).astype(int)
    y_at = np.asarray(y_at_risk_test.astype(int))
    y_t = np.asarray(y_tier_test.astype(int))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("=" * 62)
    print("MODEL  : RBF SVM (C=3, gamma=0.03), feature(s)=" + ",".join(FEATURES))
    print(f"TRAIN  : {len(X_train)} rows ({TRAIN_PATH})")
    print(f"TEST   : {len(X_test)} rows ({TEST_PATH}, fully synthetic corpus)")
    print(f"SAVED  : {MODEL_PATH}")
    print("=" * 62)

    print("\n" + "#" * 62)
    print("# TARGET 1 — target_tier (5-class: predict next-semester CGPA tier)")
    print("#" * 62)
    print(f"  accuracy            : {accuracy_score(y_t, pred_tier):.4f}")
    print(f"  balanced accuracy   : {balanced_accuracy_score(y_t, pred_tier):.4f}")
    print(f"  precision (macro)   : {precision_score(y_t, pred_tier, average='macro', zero_division=0):.4f}")
    print(f"  recall    (macro)   : {recall_score(y_t, pred_tier, average='macro', zero_division=0):.4f}")
    print(f"  F1        (macro)   : {f1_score(y_t, pred_tier, average='macro', zero_division=0):.4f}")
    print(f"  F1        (weighted): {f1_score(y_t, pred_tier, average='weighted', zero_division=0):.4f}")
    print(f"  F1        (micro)   : {f1_score(y_t, pred_tier, average='micro', zero_division=0):.4f}")
    print(f"  Cohen's kappa       : {cohen_kappa_score(y_t, pred_tier):.4f}")
    print("\n  Confusion matrix (rows=actual, cols=predicted; labels 0..4):")
    print(confusion_matrix(y_t, pred_tier, labels=[0, 1, 2, 3, 4]))
    print("\n  Classification report:")
    print(classification_report(y_t, pred_tier, labels=[0, 1, 2, 3, 4], zero_division=0))

    print("\n" + "#" * 62)
    print("# TARGET 2 — at_risk (binary, derived as target_tier <= 1)")
    print("#" * 62)
    print(f"  positive (at-risk): {int((y_at == 1).sum())},  negative: {int((y_at == 0).sum())}")
    print(f"  accuracy            : {accuracy_score(y_at, pred_at_risk):.4f}")
    print(f"  balanced accuracy   : {balanced_accuracy_score(y_at, pred_at_risk):.4f}")
    print(f"  precision           : {precision_score(y_at, pred_at_risk, zero_division=0):.4f}")
    print(f"  recall  (sensitivity): {recall_score(y_at, pred_at_risk, zero_division=0):.4f}")
    print(f"  specificity         : {recall_score(y_at, pred_at_risk, pos_label=0, zero_division=0):.4f}")
    print(f"  F1 score            : {f1_score(y_at, pred_at_risk, zero_division=0):.4f}")
    print(f"  Matthews corrcoef   : {matthews_corrcoef(y_at, pred_at_risk):.4f}")
    print(f"  Cohen's kappa       : {cohen_kappa_score(y_at, pred_at_risk):.4f}")

    decision = model.decision_function(X_test)
    try:
        auc = roc_auc_score(y_at, decision)
        print(f"  ROC-AUC (decision f): {auc:.4f}")
    except ValueError:
        # decision_function of a multiclass SVC returns one score per class;
        # compute the binary at-risk AUC from the tier-1 decision value.
        # (Optional fallback; not critical for the derived binary target.)
        try:
            d1 = decision[:, 1] if decision.ndim > 1 else decision
            auc = roc_auc_score(y_at, d1)
            print(f"  ROC-AUC (from tier=1 class score): {auc:.4f}")
        except Exception as e:
            print(f"  ROC-AUC: not computable ({e})")

    print("\n  Confusion matrix (TN FP / FN TP):")
    tn, fp, fn, tp = confusion_matrix(y_at, pred_at_risk).ravel()
    print(f"    [[ {tn}  {fp} ]")
    print(f"     [ {fn}  {tp} ]]")
    print("\n  Classification report:")
    print(classification_report(y_at, pred_at_risk, zero_division=0))


if __name__ == "__main__":
    main()
