"""Render thesis-ready table PNGs for the evaluation.

Reads the same corpus + model settings as evaluation.py and writes styled
table images under reports/:

  * classification_report_tier.png   - per-class precision/recall/f1 for target_tier
  * classification_report_atrisk.png - per-class table for at_risk
  * model_comparison.png             - full-feature vs prior-only vs trivial
  * holdout_vs_cv.png                - holdout score vs 10-fold CV score
  * traintest_split.png              - train/test distribution by tier
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from school_project.evaluation import (
    FULL_NUM, FULL_CAT, svm_pipe, holdout_scores, cv_scores,
)
from school_project.preprocess import build_clean_frames

warnings.filterwarnings("ignore")

OUTDIR = "reports"
SEED = 42
TIER_RANGES = ["0 (0.00–0.99)", "1 (1.50–2.39)", "2 (2.40–3.49)",
               "3 (3.50–4.49)", "4 (4.50–5.00)"]
FINAL = "RBF SVM (C=3, gamma=0.03)"


def render_table(filename, title, headers, rows, col_widths=None, footnote=None):
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.6))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#2563eb")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eff6ff")

    if footnote:
        ax.text(0.5, 0.02, footnote, transform=fig.transFigure,
                ha="center", fontsize=9, color="#374151")

    path = os.path.join(OUTDIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  wrote", path)


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    train = pd.read_csv("data/corpus_train.csv")
    test = pd.read_csv("data/corpus_test.csv")

    Xtr_all, y_tier_tr, y_at_tr = build_clean_frames(train)
    Xte_all, y_tier_te, y_at_te = build_clean_frames(test)
    y_tier_te = np.asarray(y_tier_te.astype(int))
    y_at_te = np.asarray(y_at_te.astype(int))

    m_prior = svm_pipe(["prior_tier"]).fit(Xtr_all[["prior_tier"]], y_tier_tr)
    m_full = svm_pipe(FULL_NUM + FULL_CAT).fit(Xtr_all[FULL_NUM + FULL_CAT], y_tier_tr)

    sco_prior, _, _ = holdout_scores(m_prior, Xte_all[["prior_tier"]], y_tier_te, y_at_te)
    sco_full, pred_tier_f, pred_ar_f = holdout_scores(
        m_full, Xte_all[FULL_NUM + FULL_CAT], y_tier_te, y_at_te)

    prior_direct = Xte_all["prior_tier"].astype(int).to_numpy()
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    baseline_tier_acc = accuracy_score(y_tier_te, prior_direct)
    baseline_tier_bal = balanced_accuracy_score(y_tier_te, prior_direct)

    cv_tier_mean, cv_tier_sd, cv_ar_mean, cv_ar_sd = cv_scores(
        FULL_NUM + FULL_CAT, Xtr_all[FULL_NUM + FULL_CAT], y_tier_tr, y_at_tr)

    print("Rendering table figures -> reports/")

    # ---- classification report: target_tier ----
    rep = classification_report(y_tier_te, pred_tier_f, labels=[0, 1, 2, 3, 4],
                                output_dict=True, zero_division=0)
    headers = ["CGPA tier", "precision", "recall", "F1-score", "support"]
    rows = []
    for i, label in enumerate(TIER_RANGES):
        rows.append([label] + [f"{rep[str(i)][k]:.3f}" for k in
                               ("precision", "recall", "f1-score")]
                    + [f"{int(rep[str(i)]['support'])}"])
    rows.append(["accuracy", "", "", f"{rep['accuracy']:.3f}", str(len(y_tier_te))])
    rows.append(["macro avg"] + [f"{rep['macro avg'][k]:.3f}" for k in
                                 ("precision", "recall", "f1-score")] + [""])
    rows.append(["weighted avg"] + [f"{rep['weighted avg'][k]:.3f}" for k in
                                    ("precision", "recall", "f1-score")] + [""])
    render_table(
        "classification_report_tier.png",
        f"Classification Report — target_tier (holdout, n={len(y_tier_te)})",
        headers, rows,
        footnote=f"{FINAL} | full feature set | accuracy ≈ {rep['accuracy']:.3f}, balanced acc ≈ {sco_full['tier_bal']:.3f}",
    )

    # ---- classification report: at_risk ----
    rep_a = classification_report(y_at_te, pred_ar_f, labels=[0, 1],
                                  output_dict=True, zero_division=0)
    headers = ["at_risk class", "precision", "recall", "F1-score", "support"]
    rows = [
        ["0 — not at risk", f"{rep_a['0']['precision']:.3f}",
         f"{rep_a['0']['recall']:.3f}", f"{rep_a['0']['f1-score']:.3f}",
         str(int(rep_a['0']['support']))],
        ["1 — at risk", f"{rep_a['1']['precision']:.3f}",
         f"{rep_a['1']['recall']:.3f}", f"{rep_a['1']['f1-score']:.3f}",
         str(int(rep_a['1']['support']))],
        ["macros"] + [f"{rep_a['macro avg'][k]:.3f}" for k in
                      ("precision", "recall", "f1-score")] + [""],
        ["weighted"] + [f"{rep_a['weighted avg'][k]:.3f}" for k in
                        ("precision", "recall", "f1-score")] + [""],
    ]
    render_table(
        "classification_report_atrisk.png",
        f"Classification Report — at_risk (holdout, n={len(y_tier_te)})",
        headers, rows,
        footnote=(f"{FINAL} | at-risk = predicted tier ≤ 1 | "
                  f"balanced acc ≈ {sco_full['ar_bal']:.3f}, AUC ≈ {sco_full['ar_auc']:.3f}"),
    )

    # ---- model comparison ----
    headers = ["Model", "tier acc", "tier bal", "at-risk bal", "at-risk recall"]
    rows = [
        ["full feature (final)", f"{sco_full['tier_acc']:.3f}",
         f"{sco_full['tier_bal']:.3f}", f"{sco_full['ar_bal']:.3f}",
         f"{sco_full['ar_rec']:.3f}"],
        ["prior-tier only", f"{sco_prior['tier_acc']:.3f}",
         f"{sco_prior['tier_bal']:.3f}", f"{sco_prior['ar_bal']:.3f}",
         f"{sco_prior['ar_rec']:.3f}"],
        ["trivial (predict prior tier)", f"{baseline_tier_acc:.3f}",
         f"{baseline_tier_bal:.3f}", "—", "—"],
    ]
    render_table(
        "model_comparison.png", "Model comparison (112-row holdout)", headers, rows,
        footnote="Full-feature RBF SVM retains the tier signal plus at-risk recall.",
    )

    # ---- holdout vs CV ----
    headers = ["metric", "holdout", "10-fold CV mean", "CV ±"]
    rows = [
        ["tier accuracy", f"{sco_full['tier_acc']:.3f}", "—", "—"],
        ["tier balanced acc", f"{sco_full['tier_bal']:.3f}",
         f"{cv_tier_mean:.3f}", f"± {cv_tier_sd:.3f}"],
        ["at-risk accuracy", f"{sco_full['ar_acc']:.3f}", "—", "—"],
        ["at-risk balanced acc", f"{sco_full['ar_bal']:.3f}",
         f"{cv_ar_mean:.3f}", f"± {cv_ar_sd:.3f}"],
        ["at-risk precision", f"{sco_full['ar_prec']:.3f}", "—", "—"],
        ["at-risk recall", f"{sco_full['ar_rec']:.3f}", "—", "—"],
        ["at-risk F1", f"{sco_full['ar_f1']:.3f}", "—", "—"],
        ["at-risk AUC", f"{sco_full['ar_auc']:.3f}", "—", "—"],
    ]
    render_table(
        "holdout_vs_cv.png",
        "Holdout vs 10-fold cross-validation (final model)",
        headers, rows,
        footnote="CV is the more defensible headline (§ synthetic demo).",
    )

    # ---- train/test split by tier ----
    headers = ["CGPA tier", "Train", "Test", "Total"]
    counts = []
    for tier in range(5):
        tr_n = int((y_tier_tr == tier).sum())
        te_n = int((y_tier_te == tier).sum())
        counts.append([TIER_RANGES[tier], str(tr_n), str(te_n), str(tr_n + te_n)])
    counts.append(["Total", str(len(y_tier_tr)), str(len(y_tier_te)),
                   str(len(y_tier_tr) + len(y_tier_te))])
    render_table(
        "traintest_split.png",
        "Corpus distribution by next-semester CGPA tier (80:20 split)",
        headers, counts,
        footnote="Stratified by target_tier; no student overlap between partitions.",
    )

    print("Done.")


if __name__ == "__main__":
    main()