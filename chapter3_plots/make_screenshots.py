"""Regenerate Chapter 3 screenshots with the correct data sources.

- shot1 df.head()          -> data/cleaned_students.csv          (61 real rows)
- all other shots          -> dataset_with_postgraduate_students.csv (164 combined)
"""

import io
import contextlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HEAD_SRC = "data/cleaned_students.csv"
COMBINED_SRC = "dataset_with_postgraduate_students.csv"
OUT = "chapter3_plots"

head = pd.read_csv(HEAD_SRC)
comb = pd.read_csv(COMBINED_SRC)


def render(title, text, fname, fontsize=9):
    lines = text.split("\n")
    width_chars = max([len(l) for l in lines] + [len(title)])
    fig_w = width_chars * fontsize * 1.05 / 72 + 1.4
    fig_h = (len(lines) * fontsize * 1.05 + 40) / 72 + 0.2
    fig_w = max(5, min(fig_w, 24))
    fig_h = max(2.5, min(fig_h, 34))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=10, loc="left")
    ax.text(0.01, 0.97, text, va="top", ha="left", family="monospace",
            fontsize=fontsize, transform=ax.transAxes)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)
    fig.savefig(f"{OUT}/{fname}", dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print("wrote", fname)


# 1. df.head()  -- CLEANED data (61 real rows)
render("df.head() -- first 5 rows (cleaned dataset, n=61 real)",
       head.head().to_string(),
       "shot1_df_head.png")

# 2. df.info()  -- COMBINED
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    comb.info()
render("df.info() -- column types & non-null counts (combined dataset, n=164)",
       buf.getvalue().strip(),
       "shot2_df_info.png", fontsize=8)

# 3. df.describe()  -- COMBINED
render("df.describe() -- descriptive statistics (combined dataset, n=164)",
       comb.describe().round(2).to_string(),
       "shot3_df_describe.png")

# 4. missing values  -- COMBINED
render("Missing values check -- df.isnull().sum() (combined dataset, n=164)",
       "Total missing values: " + str(comb.isnull().sum().sum()) + "\n\n" +
       comb.isnull().sum().to_string(),
       "shot4_missing.png")

# 5. target class distribution  -- COMBINED
render("Target class distribution -- target_tier.value_counts() (combined, n=164)",
       comb["target_tier"].value_counts().sort_index().to_string(),
       "shot5_target.png")

# 6. at-risk distribution  -- COMBINED
render("At-risk distribution -- at_risk.value_counts() (combined, n=164)",
       comb["at_risk"].value_counts().to_string(),
       "shot6_atrisk.png")

print("done")
