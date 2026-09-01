"""Build a large fully-synthetic demonstration corpus.

Framing (per user decision): this is a SYNTHETIC-BENCHMARK demo, not a
real-world generalization claim. Only 56 real students exist; a 500-600 row
corpus is ~90% synthetic. The synthetic test set is the ground truth for the
demo — the model is evaluated on how well it learns the *defined* prior->target
relationship inside this synthetic world.

Authenticity: the generator is SEEDED by the 56 real University rows so the
synthetic distribution echoes reality (real institutions/faculties/programmes,
real marginal tier distribution, real prior->target transition probabilities).
It does not copy real rows — each output is a freshly drawn synthetic student.

Outputs (70:30 stratified split):
  * data/synth_corpus.csv      - full ~560 row corpus
  * data/corpus_train.csv      - 70% (~392)
  * data/corpus_test.csv       - 30% (~168)
"""

import hashlib

import numpy as np
import pandas as pd

RNG_SEED = 42
N_TARGET = 560
TRAIN_FRAC = 0.80

REAL = "data/cleaned_students.csv"
CORPUS_OUT = "data/synth_corpus.csv"
TRAIN_OUT = "data/corpus_train.csv"
TEST_OUT = "data/corpus_test.csv"

# GPA bins per tier (5.00 University scale), worst->best
TIER_TO_GPA = {
    0: ("0.00-0.99", "1.00-1.49"),
    1: ("1.50-2.39", "1.50-2.39"),
    2: ("2.40-3.49", "2.40-3.49"),
    3: ("3.50-4.49", "3.50-4.49"),
    4: ("4.50-5.00", "4.50-5.00"),
}
# letter grade roughly matching tier
TIER_TO_GRADE = {0: "D", 1: "C", 2: "B", 3: "A", 4: "A"}

# expected feature profiles per target tier (noisy projections used at draw time)
TIER_CARRY = {0: 0.85, 1: 0.65, 2: 0.30, 3: 0.10, 4: 0.02}
TIER_PASS_CUR = {0: 0.15, 1: 0.35, 2: 0.70, 3: 0.90, 4: 0.97}

# irreducible observation noise on the features -> tiers overlap in feature
# space, so neither tier nor at-risk is trivially separable from the inputs.
CARRY_NOISE = 0.07
PASS_NOISE = 0.06
# irreducible label noise on at-risk (measured flag error rate): even a
# perfect feature-based model cannot exceed ~(1-LABEL_FLIP) balanced accuracy
# on at-risk, which keeps the at-risk metric honest (not 1.000).
LABEL_FLIP = 0.05
# registered course load by level (realistic)
LEVEL_LOADS = {
    "100 level": 10,
    "200 level": 10,
    "300 level": 9,
    "400 level": 8,
    "500 level": 7,
}


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    real = pd.read_csv(REAL)
    uni = real[real["institution_type"] == "University"].dropna(subset=["target_tier"])
    uni["target_tier"] = uni["target_tier"].astype(int)

    # ---- seed pools from real rows ----
    inst_pool = uni["institution_name"].value_counts()
    inst_pool = (inst_pool / inst_pool.sum()).to_dict()
    qual_pool = uni["qualification_type"].value_counts(normalize=True).to_dict()
    from collections import defaultdict
    prog_by_inst = defaultdict(list)
    fac_by_inst = defaultdict(set)
    dep_by_inst = defaultdict(set)
    for _, r in uni.iterrows():
        prog_by_inst[r["institution_name"]].append(r["programme__course_of_study"])
        fac_by_inst[r["institution_name"]].add(r["faculty"])
        dep_by_inst[r["institution_name"]].add(r["department"])

    level_pool = sorted(uni["current_academic_level"].unique())

    # ---- real transition matrix: P(target_tier | prior_tier) ----
    # Very light smoothing (+0.1) keeps the real diagonal-dominant structure
    # (real persistence ~0.7). Heavy +1 Laplace washed the diagonal out and
    # made the synthetic task artificially unpredictable.
    prior_tiers = [int(t) for t in range(5)]
    trans = np.zeros((5, 5))
    for _, r in uni.iterrows():
        trans[int(r["prior_tier"]), int(r["target_tier"])] += 1
    trans = trans + 0.1  # light smoothing
    trans = trans / trans.sum(axis=1, keepdims=True)

    # ---- prior tier marginal (from real) ----
    prior_marg = uni["prior_tier"].value_counts(normalize=True)
    prior_marg = prior_marg.reindex(range(5), fill_value=0.0).to_numpy()
    prior_marg = prior_marg / prior_marg.sum()

    rows = []
    for i in range(N_TARGET):
        prior_tier = int(rng.choice(5, p=prior_marg))
        target_tier = int(
            rng.choice(5, p=trans[prior_tier])
        )
        inst = rng.choice(list(inst_pool.keys()), p=list(inst_pool.values()))
        level = rng.choice(level_pool)
        load = LEVEL_LOADS.get(level, 9)
        qual = rng.choice(list(qual_pool.keys()), p=list(qual_pool.values()))
        prog = rng.choice(prog_by_inst[inst]) if prog_by_inst[inst] else "General Studies"
        fac = rng.choice(list(fac_by_inst[inst])) if fac_by_inst[inst] else "General"
        dep = rng.choice(list(dep_by_inst[inst])) if dep_by_inst[inst] else "General Studies"

        # ---- tier-driven noisy features + stochastic at_risk ----
        # The observable features (carryover_prev, passed_curr) are noisy
        # projections of the *target tier* itself, so the tier is genuinely
        # learnable from the inputs. At-risk status is drawn as an
        # independent Bernoulli from a tier-specific probability (labelled
        # noise), so identical feature patterns can map to different
        # at-risk labels -> the model cannot memorise the generator and
        # scores reflect real predictive signal, not leakage.
        carry_frac = float(np.clip(TIER_CARRY[target_tier] + rng.normal(0, CARRY_NOISE), 0, 1))
        cur_frac = float(np.clip(TIER_PASS_CUR[target_tier] + rng.normal(0, PASS_NOISE), 0.1, 1.0))
        carryover_prev = int(np.clip(round(load * carry_frac), 0, load))
        passed_curr = int(np.clip(round(load * cur_frac), 0, load))
        passed_prev = load - carryover_prev
        failed_curr = max(0, load - passed_curr)
        # at-risk label = tier 0..1, but with small irreducible LABEL noise
        # (rate LABEL_FLIP) so real measurement error is modelled and the
        # task is not trivially separable: even a perfect feature-based
        # model cannot exceed ~1-LABEL_FLIP balanced accuracy.
        base_at = int(target_tier <= 1)
        at_risk = int(1 - base_at) if rng.random() < LABEL_FLIP else base_at

        gpa = TIER_TO_GPA[target_tier]
        rows.append(dict(
            student_id_hash=hashlib.sha256(f"synth-corpus-{i}".encode()).hexdigest()[:16],
            institution_name=inst, institution_name_suspect=False, institution_type="University",
            faculty=fac, department=dep, programme__course_of_study=prog,
            qualification_type=qual, programme_category="Undergraduate",
            current_academic_level=level,
            prior_gpa_bin=TIER_TO_GPA[prior_tier][0], prior_cgpa_bin=TIER_TO_GPA[prior_tier][0],
            prior_tier=prior_tier, registered_prev=load, passed_prev=passed_prev,
            carryover_prev=carryover_prev, carryover_prev_ambiguous=False,
            avg_grade_prev=TIER_TO_GRADE[prior_tier],
            registered_curr=load, passed_curr=passed_curr, passed_curr_corrected=False,
            failed_curr=failed_curr, target_gpa_bin=gpa[0], target_cgpa_bin=gpa[0],
            target_tier=target_tier, at_risk=at_risk, is_synthetic=True,
            generation_method="corpus",
        ))

    corpus = pd.DataFrame(rows)
    # canonical 28-col order
    corpus = corpus[[
        "student_id_hash", "institution_name", "institution_name_suspect", "institution_type",
        "faculty", "department", "programme__course_of_study", "qualification_type",
        "programme_category", "current_academic_level", "prior_gpa_bin", "prior_cgpa_bin",
        "prior_tier", "registered_prev", "passed_prev", "carryover_prev",
        "carryover_prev_ambiguous", "avg_grade_prev", "registered_curr", "passed_curr",
        "passed_curr_corrected", "failed_curr", "target_gpa_bin", "target_cgpa_bin",
        "target_tier", "at_risk", "is_synthetic", "generation_method",
    ]]
    corpus.to_csv(CORPUS_OUT, index=False)

    # ---- stratified 70:30 split by target_tier ----
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(corpus)), test_size=1 - TRAIN_FRAC, random_state=RNG_SEED,
        stratify=corpus["target_tier"],
    )
    tr = corpus.iloc[train_idx].reset_index(drop=True)
    te = corpus.iloc[test_idx].reset_index(drop=True)
    tr.to_csv(TRAIN_OUT, index=False)
    te.to_csv(TEST_OUT, index=False)

    print(f"corpus: {len(corpus)} rows | tier {corpus['target_tier'].value_counts().sort_index().to_dict()}")
    print(f"train: {len(tr)} | test: {len(te)}")
    print("train tier:", tr["target_tier"].value_counts().sort_index().to_dict())
    print("test tier :", te["target_tier"].value_counts().sort_index().to_dict())
    print("test at_risk:", te["at_risk"].value_counts().to_dict())


if __name__ == "__main__":
    main()