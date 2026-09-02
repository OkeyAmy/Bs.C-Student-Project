"""Real-data comparison: students with vs without prior-semester carryovers.

Uses ONLY the real records (data/cleaned_students.csv) — no synthetic rows — so
the comparison is what the survey data itself says. Response is computed once
and cached, since the source is static during the process lifetime.
"""

from functools import lru_cache

import pandas as pd
from scipy import stats

from api.predict import ROOT

CLEAN_PATH = str(ROOT / "data" / "cleaned_students.csv")


@lru_cache(maxsize=1)
def compute_comparison() -> dict:
    df = pd.read_csv(CLEAN_PATH)
    df = df.dropna(subset=["target_tier", "at_risk"]).copy()
    df["group"] = df["carryover_prev"].apply(lambda v: "with" if v >= 1 else "no")

    groups = {}
    for name, sub in df.groupby("group"):
        groups[name] = {
            "n": int(len(sub)),
            "at_risk_rate": float(sub["at_risk"].mean()),
            "at_risk_count": int(sub["at_risk"].sum()),
            "mean_tier": float(sub["target_tier"].mean()),
            "median_tier": float(sub["target_tier"].median()),
            "mean_prior_tier": float(sub["prior_tier"].mean()),
        }

    # Significance tests.
    table = pd.crosstab(df["group"], df["at_risk"]).reindex(columns=[0, 1], fill_value=0)
    chi2, chi_p, _, _ = stats.chi2_contingency(table)
    odds_ratio, fisher_p = stats.fisher_exact(table.values)

    no = df[df["group"] == "no"]["target_tier"]
    with_ = df[df["group"] == "with"]["target_tier"]
    _, mw_p = stats.mannwhitneyu(no, with_, alternative="two-sided")

    # Odds for the "any carryover" vs "none" at-risk ratio.
    p_no = groups["no"]["at_risk_rate"]
    p_with = groups["with"]["at_risk_rate"]
    risk_ratio = p_with / p_no if p_no > 0 else None

    return {
        "groups": groups,
        "tests": {
            "at_risk_chi2": float(chi2),
            "at_risk_chi2_p": float(chi_p),
            "at_risk_fisher_p": float(fisher_p),
            "at_risk_odds_ratio": float(odds_ratio),
            "at_risk_risk_ratio": risk_ratio,
            "performance_mannwhitney_p": float(mw_p),
        },
        "n_total": int(len(df)),
    }