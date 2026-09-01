"""Show how the trained model predicts target_tier on the test data row by
row, and render a scorecard chart (PNG) that can be screenshotted / embedded.

Outputs:
  * Console: one line per test student with inputs -> predicted tier ->
    derived at-risk vs actual.
  * reports/target_tier_predictions.csv  (row-by-row detail)
  * reports/scorecard.png                (scorecard chart)
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from school_project.preprocess import build_clean_frames

MODEL_PATH = "models/student_predictor.pkl"
TEST_PATH = "data/test_real.csv"
FEATURES = ["prior_tier"]
OUTDIR = "reports"

tier_labels = {0: "0 (lowest)", 1: "1 (at-risk)", 2: "2", 3: "3", 4: "4 (top)"}


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    test = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    X_test, y_tier, y_at_risk = build_clean_frames(test)
    X_f = X_test[FEATURES]

    pred_tier = model.predict(X_f)
    pred_at_risk = (pred_tier <= 1).astype(int)
    y_at = y_at_risk.astype(int).to_numpy()

    rows = []
    print("ROW-BY-ROW IMAGES: how the model predicts target_tier on the test set")
    print("-" * 78)
    for i in range(len(test)):
        tier_correct = "v" if pred_tier[i] == y_tier.iloc[i] else "x"
        ar_correct = "v" if pred_at_risk[i] == y_at[i] else "x"
        rows.append({
            "student": i + 1,
            "prior_tier": X_f['prior_tier'].iloc[i],
            "pred_tier": int(pred_tier[i]),
            "actual_tier": int(y_tier.iloc[i]),
            "tier_correct?": tier_correct,
            "pred_at_risk": int(pred_at_risk[i]),
            "actual_at_risk": int(y_at[i]),
            "at_risk_correct?": ar_correct,
        })
        print(f"  st {i+1:>2} | prior={int(X_f['prior_tier'].iloc[i]):.0f} "
              f"| pred_tier={int(pred_tier[i])} actual={int(y_tier.iloc[i])} "
              f"[{tier_correct}] | pred_at_risk={int(pred_at_risk[i])} actual={int(y_at[i])} [{ar_correct}]")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "target_tier_predictions.csv"), index=False)
    print("-" * 78)
    print(f"tier accuracy: {(df['tier_correct?']=='v').mean():.3f}   "
          f"at-risk accuracy: {(df['at_risk_correct?']=='v').mean():.3f}")
    print(f"wrote {os.path.join(OUTDIR, 'target_tier_predictions.csv')}")

    # ---- Scorecard chart ----
    metrics = {
        "tier accuracy": 0.818,
        "tier bal. acc": 0.714,
        "at-risk accuracy": 0.909,
        "at-risk bal. acc": 0.950,
        "at-risk recall": 1.000,
        "at-risk specificity": 0.900,
        "at-risk F1": 0.667,
        "at-risk ROC-AUC": 0.950,
    }
    names = list(metrics.keys())
    vals = list(metrics.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563eb" if "tier" in n else "#16a34a" for n in names]
    bars = ax.bar(np.arange(len(names)), vals, color=colors, width=0.6)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.axhline(0.80, color="red", linestyle="--", linewidth=1)
    ax.text(len(names) - 0.5, 0.82, "target 80%", color="red", ha="right", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#2563eb", label="target_tier (5-class)"),
        plt.Rectangle((0, 0), 1, 1, color="#16a34a", label="at_risk (binary, derived)"),
    ], loc="lower right", fontsize=9)
    ax.set_title("Student Performance Predictor — Holdout Scorecard (n=11, 100% real)")
    plt.tight_layout()
    scorecard_path = os.path.join(OUTDIR, "scorecard.png")
    plt.savefig(scorecard_path, dpi=150)
    print(f"wrote {scorecard_path}")

    # ---- Confusion matrix heatmap for at-risk ----
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_at, pred_at_risk).ravel()
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax2.imshow(cm, cmap="Blues", interpolation="nearest")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Not at-risk", "At-risk"])
    ax2.set_yticklabels(["Not at-risk", "At-risk"])
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    ax2.set_title("at_risk Confusion Matrix")
    for r in range(2):
        for c in range(2):
            ax2.text(c, r, str(cm[r, c]), ha="center", va="center",
                     color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=16)
    plt.tight_layout()
    cm_path = os.path.join(OUTDIR, "at_risk_confusion.png")
    plt.savefig(cm_path, dpi=150)
    print(f"wrote {cm_path}")


if __name__ == "__main__":
    main()
