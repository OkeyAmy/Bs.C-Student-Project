"""Generate Chapter 3 exploratory data analysis figures.

Reads the real cleaned training/held-out data and the combined dataset,
rendering publication-style PNGs into this directory.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DIROUT = "chapter3_plots"
TRAIN = "data/train_real.csv"
TEST = "data/test_real.csv"
CLEANED = "data/cleaned_students.csv"
COMBINED = "dataset_with_postgraduate_students.csv"

sns.set_theme(style="whitegrid", palette="deep")
sns.set(font_scale=1.15)

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
cleaned = pd.read_csv(CLEANED)
combined = pd.read_csv(COMBINED)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{DIROUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{DIROUT}/{name}")


TIER_LABELS = ["0 (0.00-0.99)", "1 (1.50-2.39)", "2 (2.40-3.49)", "3 (3.50-4.49)", "4 (4.50-5.00)"]
TIER_COLORS = ["#d62728", "#ff7f0e", "#ffd54f", "#9ccc65", "#2ca02c"]


# 1. Train/test target_tier class distribution (side by side)
fig, ax = plt.subplots(figsize=(9, 5))
tr_cnt = train["target_tier"].value_counts().sort_index()
te_cnt = test["target_tier"].value_counts().reindex(range(5), fill_value=0).sort_index()
x = range(5)
w = 0.38
ax.bar([i - w / 2 for i in x], tr_cnt.values, width=w, label="Train (n=45)", color="#2c3e50")
ax.bar([i + w / 2 for i in x], te_cnt.values, width=w, label="Test (n=11)", color="#e67e22")
ax.set_xticks(list(x))
ax.set_xticklabels(TIER_LABELS, rotation=20, ha="right")
ax.set_ylabel("Number of students")
ax.set_title("Target CGPA Tier Distribution: Training vs Held-Out Test")
ax.legend()
for i, v in enumerate(tr_cnt.values):
    ax.text(i - w / 2, v + 0.2, str(v), ha="center")
for i, v in enumerate(te_cnt.values):
    ax.text(i + w / 2, v + 0.2, str(v), ha="center")
save(fig, "01_train_test_tier_distribution.png")


# 2. Prior tier distribution (real cleaned data)
fig, ax = plt.subplots(figsize=(8, 5))
pc = cleaned["prior_tier"].value_counts().sort_index()
ax.bar(range(len(pc)), pc.values, color=TIER_COLORS)
ax.set_xticks(range(len(pc)))
ax.set_xticklabels(TIER_LABELS[: len(pc)], rotation=20, ha="right")
ax.set_ylabel("Number of students")
ax.set_title("Prior Semester CGPA Tier Distribution (n=61 real cleaned)")
for i, v in enumerate(pc.values):
    ax.text(i, v + 0.2, str(v), ha="center")
save(fig, "02_prior_tier_distribution.png")


# 3. Current academic level distribution (real cleaned)
fig, ax = plt.subplots(figsize=(8, 5))
order = ["100 level", "200 level", "300 level", "400 level", "500 level", "Done"]
present = [l for l in order if l in cleaned["current_academic_level"].unique()]
cnt = cleaned["current_academic_level"].value_counts().reindex(present)
ax.bar(range(len(cnt)), cnt.values, color="#7f8c8d")
ax.set_xticks(range(len(cnt)))
ax.set_xticklabels(cnt.index, rotation=20, ha="right")
ax.set_ylabel("Number of students")
ax.set_title("Current Academic Level Distribution (n=61 real cleaned)")
for i, v in enumerate(cnt.values):
    ax.text(i, v + 0.2, str(v), ha="center")
save(fig, "03_academic_level.png")


# 4. Correlation heatmap of numeric features vs target (real train)
fig, ax = plt.subplots(figsize=(9, 7))
num = ["prior_tier", "registered_prev", "passed_prev", "carryover_prev",
       "registered_curr", "passed_curr", "failed_curr", "target_tier"]
corr = train[num].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Matrix of Numeric Features (Real Training, n=45)")
save(fig, "04_correlation_heatmap.png")


# 5. At-risk vs not-at-risk (real cleaned University subset)
fig, ax = plt.subplots(figsize=(7, 5))
uni = cleaned[cleaned["institution_type"] == "University"]
ar = uni["at_risk"].map({True: "At risk", False: "Not at risk"}).value_counts()
ax.bar(ar.index, ar.values, color=["#e74c3c", "#2ecc71"])
ax.set_ylabel("Number of students")
ax.set_title("At-Risk Identification (Real University, n=56)")
for i, v in enumerate(ar.values):
    ax.text(i, v + 0.2, str(v), ha="center")
save(fig, "05_at_risk.png")


# 6. Faculty distribution (real cleaned University)
fig, ax = plt.subplots(figsize=(10, 6))
fc = uni["faculty"].value_counts()
ax.barh(range(len(fc)), fc.values[::-1], color="#16a085")
ax.set_yticks(range(len(fc)))
ax.set_yticklabels(fc.index[::-1], fontsize=9)
ax.set_xlabel("Number of students")
ax.set_title("Faculty Distribution (Real University, n=56)")
save(fig, "06_faculty_distribution.png")


# 7. Real vs synthetic mix in combined dataset
fig, ax = plt.subplots(figsize=(7, 5))
gm = combined["generation_method"].value_counts()
ax.bar(gm.index, gm.values, color=["#3498db", "#e67e22", "#9b59b6"])
ax.set_ylabel("Number of rows")
ax.set_title("Combined Dataset Composition (n=164)")
for i, v in enumerate(gm.values):
    ax.text(i, v + 1, str(v), ha="center")
save(fig, "07_dataset_composition.png")

print("done")
