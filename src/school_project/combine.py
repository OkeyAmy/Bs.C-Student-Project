"""Combine real train rows + SMOTE-NC synthetic + LLM synthetic into one
submittable CSV, and validate the result against the real data.

data/test_real.csv is deliberately left out — it stays 100% real and
untouched, so it remains a valid honest-evaluation holdout.

The validation below exists because of a real failure caught during
development: an earlier all-persistent synthetic batch (every student
staying in the same tier) would have taught a model that performance is
more predictable than it is.

WHAT WE VALIDATE, AND WHY NOT CORRELATION
-----------------------------------------
The quantity a model actually learns here is the transition distribution
P(target_tier | prior_tier) — how often, and how far, a student moves. That
is what we gate on: the synthetic rows must reproduce the real persistence
rate and the real per-step delta distribution.

We deliberately do NOT gate on the prior<->target Pearson correlation.
Correlation is not comparable between the real and augmented sets, because
rebalancing the classes (the whole point of the augmentation) raises the
variance of target_tier, which mechanically raises correlation even when
the conditional noise is identical. Observed here: real 0.66 vs augmented
0.84, while the delta distributions match to within ~2 percentage points
and the within-tier spread of prior_tier is equal or wider in the synthetic
rows. The correlation is still reported below as context, but the delta
distribution is the gate.
"""

import sys

import pandas as pd

OUT_PATH = "data/train_augmented.csv"

# Synthetic transition behaviour must match real within these tolerances.
PERSISTENCE_TOL = 0.06  # abs diff in P(target == prior)
DELTA_TOL = 0.06  # max abs diff in any per-step delta proportion


def corr(df: pd.DataFrame) -> float:
    return df[["prior_tier", "target_tier"]].corr().iloc[0, 1]


def delta_dist(df: pd.DataFrame) -> pd.Series:
    return (df["target_tier"] - df["prior_tier"]).value_counts(normalize=True)


def main() -> None:
    real = pd.read_csv("data/train_real.csv")
    real["generation_method"] = "real"
    stat = pd.read_csv("data/synthetic_stat.csv")
    llm = pd.read_csv("data/synthetic_llm.csv")

    combined = pd.concat([real, stat, llm], ignore_index=True, sort=False)
    combined.to_csv(OUT_PATH, index=False)

    n_synth = len(stat) + len(llm)
    print(f"train_augmented.csv: {len(combined)} rows")
    print(combined.groupby(["target_tier", "generation_method"]).size().unstack(fill_value=0))
    print(f"\nreal: {len(real)}, synthetic: {n_synth} "
          f"({n_synth / len(combined):.0%} of augmented train)")

    print("\nat-risk (tier <= 1) balance:")
    print(f"  real train:      {real['at_risk'].sum()}/{len(real)} ({real['at_risk'].mean():.0%})")
    print(f"  augmented train: {combined['at_risk'].sum()}/{len(combined)} ({combined['at_risk'].mean():.0%})")

    synth = pd.concat([stat, llm])

    print("\nprior_tier <-> target_tier correlation (context only, not a gate):")
    print(f"  real only: {corr(real):.3f}    augmented: {corr(combined):.3f}")
    print("  (higher in augmented by construction — class rebalancing raises "
          "target variance; see module docstring)")

    real_persist = (real["prior_tier"] == real["target_tier"]).mean()
    synth_persist = (synth["prior_tier"] == synth["target_tier"]).mean()
    print(f"\nGATE 1 — persistence rate: real {real_persist:.3f}, "
          f"synthetic {synth_persist:.3f} (tol {PERSISTENCE_TOL})")

    rd, sd = delta_dist(real), delta_dist(synth)
    steps = sorted(set(rd.index) | set(sd.index))
    print("\nGATE 2 — delta distribution (target_tier - prior_tier):")
    print("  step | real  | synth | diff")
    worst = 0.0
    for s in steps:
        r, v = rd.get(s, 0.0), sd.get(s, 0.0)
        worst = max(worst, abs(r - v))
        print(f"  {int(s):+d}   | {r:.3f} | {v:.3f} | {abs(r - v):.3f}")
    print(f"  worst per-step diff: {worst:.3f} (tol {DELTA_TOL})")

    problems = []
    if abs(real_persist - synth_persist) > PERSISTENCE_TOL:
        problems.append(
            f"synthetic persistence {synth_persist:.3f} differs from real "
            f"{real_persist:.3f} by more than {PERSISTENCE_TOL} — synthetic "
            "students move between tiers at the wrong rate"
        )
    if worst > DELTA_TOL:
        problems.append(
            f"delta distribution mismatch {worst:.3f} exceeds {DELTA_TOL} — "
            "synthetic tier transitions are the wrong shape"
        )
    bad = combined[combined["passed_curr"] > combined["registered_curr"]]
    if len(bad):
        problems.append(f"{len(bad)} rows have passed_curr > registered_curr")
    if combined["target_cgpa_bin"].isna().any():
        problems.append("blank target_cgpa_bin present")

    if problems:
        print("\nVALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nvalidation passed")


if __name__ == "__main__":
    main()
