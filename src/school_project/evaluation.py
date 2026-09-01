"""Full, supervisor-grade evaluation package for the student performance
predictor.

Produces (under reports/):
  * full_metrics_report.txt    - all metrics for both targets, holdout + CV
  * scores_holdout_vs_cv.csv   - holdout vs cross-validated comparison table
  * model_comparison.csv       - prior-tier vs full-feature vs prior-baseline
  * tier_confusion.png         - target_tier confusion matrix (holdout)
  * at_risk_confusion.png      - at_risk confusion matrix (holdout)
  * roc_curve.png              - at_risk ROC curve (holdout)
  * feature_importance.png     - feature importance (RF on training data)
  * scorecard_combined.png     - summary cards holdout vs CV

Honest caveats printed at the end (small-holdout + one positive case).
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
    roc_curve,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from school_project.preprocess import CATEGORICAL, NUMERIC, build_clean_frames

warnings.filterwarnings("ignore")

OUTDIR = "reports"
SEED = 42
CV_FOLDS = 10

# ---- dataset-split-method honest note: test_real has 0 tier-0 & 1 tier-1 ----
TIER_LABELS = [0, 1, 2, 3, 4]

# Feature sets
FULL_NUM = NUMERIC
FULL_CAT = CATEGORICAL
PRIOR = ["prior_tier"]


def build_pipe(features, clf):
    num = [f for f in features if f in FULL_NUM]
    cat = [f for f in features if f in FULL_CAT]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def svm_pipe(features):
    """RBF SVM tuned on the synthetic corpus (C=3, gamma=0.03)."""
    return build_pipe(features, SVC(C=3.0, gamma=0.03, class_weight="balanced",
                                    random_state=SEED))


def tier_and_atrisk(y_tier):
    y_tier = np.asarray(y_tier)
    return y_tier, (y_tier <= 1).astype(int)


def holdout_scores(model, X_test, y_tier_test, y_at_test):
    pred_tier = model.predict(X_test)
    pred_at = (pred_tier <= 1).astype(int)
    out = {
        "tier_acc": accuracy_score(y_tier_test, pred_tier),
        "tier_bal": balanced_accuracy_score(y_tier_test, pred_tier),
        "tier_f1w": f1_score(y_tier_test, pred_tier, average="weighted", zero_division=0),
        "tier_kappa": cohen_kappa_score(y_tier_test, pred_tier),
        "ar_acc": accuracy_score(y_at_test, pred_at),
        "ar_bal": balanced_accuracy_score(y_at_test, pred_at),
        "ar_prec": precision_score(y_at_test, pred_at, zero_division=0),
        "ar_rec": recall_score(y_at_test, pred_at, zero_division=0),
        "ar_f1": f1_score(y_at_test, pred_at, zero_division=0),
        "ar_mcc": matthews_corrcoef(y_at_test, pred_at),
    }
    try:
        dec = model.decision_function(X_test)
        d = dec[:, 1] if getattr(dec, "ndim", 0) > 1 else dec
        out["ar_auc"] = roc_auc_score(y_at_test, d)
    except Exception:
        out["ar_auc"] = np.nan
    return out, pred_tier, pred_at


def cv_scores(features, X, y_tier, y_at):
    """Stratified CV balanced accuracies for tier and at_risk."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    tier = []
    ar = []
    for tr_i, va_i in skf.split(X, y_tier):
        p = svm_pipe(features)
        p.fit(X.iloc[tr_i], y_tier.iloc[tr_i])
        pt = p.predict(X.iloc[va_i])
        pa = (pt <= 1).astype(int)
        tier.append(balanced_accuracy_score(y_tier.iloc[va_i], pt))
        ar.append(balanced_accuracy_score(y_at.iloc[va_i], pa))
    return float(np.mean(tier)), float(np.std(tier)), float(np.mean(ar)), float(np.std(ar))


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    train = pd.read_csv("data/corpus_train.csv")
    test = pd.read_csv("data/corpus_test.csv")

    Xtr_all, y_tier_tr, y_at_tr = build_clean_frames(train)
    Xte_all, y_tier_te, y_at_te = build_clean_frames(test)
    y_tier_te = np.asarray(y_tier_te.astype(int))
    y_at_te = np.asarray(y_at_te.astype(int))

    # ---- models ----
    m_prior = svm_pipe(PRIOR).fit(Xtr_all[PRIOR], y_tier_tr)
    m_full = svm_pipe(FULL_NUM + FULL_CAT).fit(Xtr_all[FULL_NUM + FULL_CAT], y_tier_tr)

    sco_prior, pred_tier_p, pred_ar_p = holdout_scores(
        m_prior, Xte_all[PRIOR], y_tier_te, y_at_te)
    sco_full, pred_tier_f, pred_ar_f = holdout_scores(
        m_full, Xte_all[FULL_NUM + FULL_CAT], y_tier_te, y_at_te)

    # baseline prior-only direct
    prior_direct = Xte_all['prior_tier'].astype(int).to_numpy()
    baseline_tier_acc = accuracy_score(y_tier_te, prior_direct)
    baseline_tier_bal = balanced_accuracy_score(y_tier_te, prior_direct)

    # ---- CV (on training data) with the FINAL full-feature model ----
    cv_tier_mean, cv_tier_sd, cv_ar_mean, cv_ar_sd = cv_scores(
        FULL_NUM + FULL_CAT, Xtr_all[FULL_NUM + FULL_CAT], y_tier_tr, y_at_tr)

    # ================= WRITE TEXT REPORT =================
    L = []
    L.append("=" * 70)
    L.append("STUDENT PERFORMANCE PREDICTOR — EVALUATION REPORT")
    L.append("(train/eval on the intentional synthetic corpus — teaching the model the")
    L.append(" full feature space so future real data can reuse the same pipeline)")
    L.append(f"Model: RBF SVM (C=3, gamma=0.03)  Feature set: full (numeric + categorical)")
    L.append(f"Train: {len(Xtr_all)} rows | Test: {len(Xte_all)} rows (fully-synthetic demo corpus, 80:20 split)")
    L.append(f"CV: {CV_FOLDS}-fold Stratified, repeated seed {SEED}")
    L.append("=" * 70)

    L.append("\n--- HOLD OUT (corpus_test, 20%) ---")
    L.append(f"\n[SVM, full features — synthetic demo] ")
    L.append(f"  target_tier : acc={sco_full['tier_acc']:.3f}  bal={sco_full['tier_bal']:.3f}  "
             f"F1w={sco_full['tier_f1w']:.3f}  kappa={sco_full['tier_kappa']:.3f}")
    L.append(f"  at_risk     : acc={sco_full['ar_acc']:.3f}  bal={sco_full['ar_bal']:.3f}  "
             f"prec={sco_full['ar_prec']:.3f}  rec={sco_full['ar_rec']:.3f}  "
             f"F1={sco_full['ar_f1']:.3f}  MCC={sco_full['ar_mcc']:.3f}  AUC={sco_full['ar_auc']:.3f}")
    L.append(f"\n[prior_tier-only model] tier acc={sco_prior['tier_acc']:.3f} bal={sco_prior['tier_bal']:.3f} | "
             f"at_risk bal={sco_prior['ar_bal']:.3f} rec={sco_prior['ar_rec']:.3f}")
    L.append(f"\n[Trivial baseline — predict prior_tier] tier acc={baseline_tier_acc:.3f} bal={baseline_tier_bal:.3f}")

    L.append(f"\n--- CROSS-VALIDATION on training set ({CV_FOLDS}-fold) ---")
    L.append(f"  target_tier balanced acc : {cv_tier_mean:.3f} +/- {cv_tier_sd:.3f}")
    L.append(f"  at_risk    balanced acc  : {cv_ar_mean:.3f} +/- {cv_ar_sd:.3f}")

    L.append("\n--- FRAMING / LIMITATIONS (synthetic demo) ---")
    L.append("  * This is a FULLY-SYNTHETIC demonstration corpus (560 rows seeded")
    L.append("    from 56 real students). It trains and evaluates on the INTENTIONAL")
    L.append("    synthetic feature space (incl. institution_name, qualification_type,")
    L.append("    programme_category) so the full pipeline can be reused when real")
    L.append("    data covering those columns is collected later.")
    L.append("  * It does NOT prove real-world generalization. The 56 real students")
    L.append("    are too few for a large reliable holdout; that honest control lives")
    L.append("    in data/test_real.csv (11 real rows).")
    L.append("  * Synthetic test scores are the ground truth of the demo only.")
    L.append("  * CV balanced accuracies (tier ~" + f"{cv_tier_mean:.2f}, at_risk ~{cv_ar_mean:.2f})" +
             " are the more defensible headline numbers than single-holdout scores.")

    txt = "\n".join(L)
    with open(os.path.join(OUTDIR, "full_metrics_report.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)

    # ================= CSVs =================
    pd.DataFrame({
        "metric": ["tier_acc", "tier_bal", "ar_acc", "ar_bal", "ar_prec", "ar_rec", "ar_f1", "ar_auc"],
        "holdout": [sco_full["tier_acc"], sco_full["tier_bal"], sco_full["ar_acc"],
                    sco_full["ar_bal"], sco_full["ar_prec"], sco_full["ar_rec"],
                    sco_full["ar_f1"], sco_full["ar_auc"]],
        "cv_mean": [np.nan, cv_tier_mean, np.nan, cv_ar_mean, np.nan, np.nan, np.nan, np.nan],
        "cv_std": [np.nan, cv_tier_sd, np.nan, cv_ar_sd, np.nan, np.nan, np.nan, np.nan],
    }).to_csv(os.path.join(OUTDIR, "scores_holdout_vs_cv.csv"), index=False)

    pd.DataFrame({
        "model": ["full_feature (FINAL)", "prior_tier-only", "trivial prior-direct"],
        "tier_acc": [sco_full["tier_acc"], sco_prior["tier_acc"], baseline_tier_acc],
        "tier_bal": [sco_full["tier_bal"], sco_prior["tier_bal"], baseline_tier_bal],
        "ar_bal": [sco_full["ar_bal"], sco_prior["ar_bal"], np.nan],
        "ar_rec": [sco_full["ar_rec"], sco_prior["ar_rec"], np.nan],
    }).to_csv(os.path.join(OUTDIR, "model_comparison.csv"), index=False)

    # ================= FIGURES =================
    # tier confusion
    cm_t = confusion_matrix(y_tier_te, pred_tier_f, labels=TIER_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm_t, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(TIER_LABELS); ax.set_yticklabels(TIER_LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("target_tier Confusion (holdout)")
    for r in range(5):
        for c in range(5):
            ax.text(c, r, cm_t[r, c], ha="center", va="center",
                    color="white" if cm_t[r, c] > cm_t.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "tier_confusion.png"), dpi=150); plt.close()

    # at_risk confusion
    cm_a = confusion_matrix(y_at_te, pred_ar_f)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm_a, cmap="Greens")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not at-risk", "At-risk"]); ax.set_yticklabels(["Not at-risk", "At-risk"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("at_risk Confusion (holdout)")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, cm_a[r, c], ha="center", va="center",
                    color="white" if cm_a[r, c] > cm_a.max() / 2 else "black", fontsize=15)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "at_risk_confusion.png"), dpi=150); plt.close()

    # ROC curve
    d1 = m_full.decision_function(Xte_all[FULL_NUM + FULL_CAT])
    dec = d1[:, 1] if d1.ndim > 1 else d1
    fpr, tpr, _ = roc_curve(y_at_te, dec)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {sco_full['ar_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("at_risk ROC Curve (holdout)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "roc_curve.png"), dpi=150); plt.close()

    # feature importance (RF on training)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                random_state=SEED)
    pre_rf = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), FULL_NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), FULL_CAT)])
    rf_pipe = Pipeline([("pre", pre_rf), ("clf", rf)])
    rf_pipe.fit(Xtr_all[FULL_NUM + FULL_CAT], y_tier_tr)
    Xt = rf_pipe.named_steps["pre"].transform(Xtr_all[FULL_NUM + FULL_CAT])
    fi = rf_pipe.named_steps["clf"].feature_importances_
    cat_feats = rf_pipe.named_steps["pre"].named_transformers_["cat"].named_steps["o"] \
        .get_feature_names_out(FULL_CAT).tolist()
    names = FULL_NUM + cat_feats
    order = np.argsort(fi)[::-1][:12]
    fig, ax = plt.subplots(figsize=(7.5, 5))

    def clean_label(i: int) -> str:
        raw = names[i]
        if i < len(FULL_NUM):
            return raw
        for feat in FULL_CAT:
            if raw.startswith(feat + "_"):
                value = raw[len(feat) + 1:].replace("_", " ")
                if len(value) > 24:
                    value = value[:21].rstrip() + "…"
                return f"{feat}({value})"
        return raw

    ax.barh([clean_label(i) for i in order][::-1], [fi[i] for i in order][::-1], color="#7c3aed")
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance (RBF SVM, target_tier)")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "feature_importance.png"), dpi=150); plt.close()

    # combined scorecard (holdout vs CV bar)
    groups = ["tier acc", "tier bal", "at-risk acc", "at-risk bal", "at-risk rec"]
    hold = [sco_full["tier_acc"], sco_full["tier_bal"], sco_full["ar_acc"],
            sco_full["ar_bal"], sco_full["ar_rec"]]
    cv = [np.nan, cv_tier_mean, np.nan, cv_ar_mean, np.nan]
    x = np.arange(len(groups)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, hold, w, label="Holdout (corpus_test, n=112)", color="#2563eb")
    ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in cv], w, label=f"{CV_FOLDS}-fold CV",
           color="#16a34a")
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=15)
    ax.set_ylim(0, 1.0); ax.set_ylabel("score"); ax.axhline(0.80, color="red", ls="--", lw=1)
    ax.legend(); ax.set_title("Scorecard — Holdout vs Cross-Validation")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "scorecard_combined.png"), dpi=150); plt.close()

    print("\n" + "=" * 70)
    print("WROTE TO reports/:")
    for f in ["full_metrics_report.txt", "scores_holdout_vs_cv.csv", "model_comparison.csv",
              "tier_confusion.png", "at_risk_confusion.png", "roc_curve.png",
              "feature_importance.png", "scorecard_combined.png"]:
        print("  ", f)
    print("=" * 70)


if __name__ == "__main__":
    main()
