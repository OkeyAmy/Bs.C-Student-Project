"""LLM-authored synthetic rows — the hand-written share of the augmentation.

Authored directly by Claude (in-session), not via a live API call, so the
batch is fixed, inspectable and reproducible without an API key.

DESIGN NOTE — why these rows look the way they do
-------------------------------------------------
An earlier 13-row version of this file made every synthetic student
*persistent*: next-semester tier equal to previous tier. Measured against
the real data that was wrong, and measurably harmful: it pushed the
prior_tier<->target_tier correlation from 0.66 (real) to 0.79, i.e. it
taught a model that academic performance is more predictable than it
actually is.

The real 56 University respondents transition like this:

    delta = target_tier - prior_tier
    -2:  1   (1.8%)
    -1:  6  (10.7%)
     0: 40  (71.4%)
    +1:  7  (12.5%)
    +2:  2   (3.6%)
    non-persistent: 28.6%

This batch reproduces that shape: 60 rows, 42 persistent (70%) and 18
non-persistent (30%), moving in both directions, capped at +/-2 like the
real data. Every non-persistent row carries a *mechanism* in its course
counts rather than an unexplained jump — a declining student shows
carryover and failures appearing, an improving student shows a backlog
clearing. Course counts are also kept consistent with the per-tier real
averages (tier 4 has zero failures and grade A; tier 0 fails most of what
it registers; carryover peaks at tier 1).

Every row is grounded in a real institution/faculty/department/programme
combination drawn from train_real.csv (COMBOS below) — no invented
institutions or invented programme names.
"""

import pandas as pd

OUT_PATH = "data/synthetic_llm.csv"

UNI_BINS = ["0.00 - 0.99", "1.50 - 2.39", "2.40 - 3.49", "3.50 - 4.49", "4.50 - 5.00"]

# (institution, faculty, department, programme, qualification) — all taken
# verbatim (whitespace-stripped) from real rows in train_real.csv.
COMBOS = [
    ("Nnamdi Azikiwe University, Awka", "Art", "Modern European language", "BA", "BA"),
    ("Nnamdi Azikiwe University, Awka", "Education", "Health promotion and public health education", "BSc in education health promotion and public health education", "BSc"),
    ("Lagos State University", "Faculty Of Education", "Guidance and counseling psychology", "BSCE Guidance and counseling psychology", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Engineering", "Electrical and Electronic Engineering", "B.E/ENG", "BEng"),
    ("Lagos State University", "Education", "Guidance and counselling education", "B.sc(ed)", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Technology and Vocational Education", "Business Education", "Accounting", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Physical science", "Computer Science", "B.Sc Computer Science", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Education", "Educational Management and Policy", "B.Ed Educational Management and Policy", "B.Ed"),
    ("Enugu State University of Science and Technology", "Engineering", "Computer engineering", "Computer engineering", "BEng"),
    ("Nnamdi Azikiwe University, Awka", "Education", "Health Promotion and Public Health Education", "B. Ed Health Education", "B. Ed"),
    ("Nnamdi Azikiwe University, Awka", "Physical science", "Computer science", "B.Sc Computer science", "BSc"),
    ("Benue State University", "Social science", "Psychology", "B.sc Psychology", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Engineering", "Mechanical", "B.Sc Engineering", "BSc"),
    ("University of Port Harcourt", "Clinical Sciences", "Nursing", "B.Sc Nursing Science", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Physical science", "Software Engineering", "B.sc software engineering", "BSc"),
    ("Federal University Otuoke", "Science", "Biochemistry", "Biochemistry", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Law", "Civil law", "L.LB", "L.LB"),
    ("National Open University of Nigeria", "Social Science", "Tourism and hospitality management", "BSC Tourism", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Social sciences", "Mass communication", "B.Sc Mass communication", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Physical science", "Mathematics", "Mathematics", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Law", "Law", "Law", "LLB"),
    ("Lagos State University", "Agriculture", "Crop production", "Agriculture", "BSc"),
    ("Federal University Otuoke", "Management science", "Banking and finance", "B.Sc Banking and finance", "BSc"),
    ("Olabisi Onabanjo University", "Engineering", "Civil engineering", "B.sc engineering", "BEng"),
    ("Lagos State University", "Educational and human psychology", "Guidance and counselling education", "B. Sc/ed / Guidance and counselling education", "BSc"),
    ("University of Calabar", "Allied", "Public health science", "Bsc/public health science", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Agriculture", "Food science and technology", "B.Sc Food science and technology", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Social sciences", "Mass communication", "Mass communication", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Education", "Science education", "B.Sc", "BSc"),
    ("University of Lagos", "Education", "Educational Foundation", "B.A Guidance and Counseling", "BA"),
    ("Lagos State University", "Education", "Educational Foundation and Counselling Psychology", "B.ed Guidance and Counselling", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Law", "Civil law", "LLB Law", "LLB"),
    ("Nnamdi Azikiwe University, Awka", "Physical sciences", "Computer science", "Bsc Computer science", "BSc"),
    ("Nnamdi Azikiwe University, Awka", "Law", "Civil law", "LLB Bachelor of law", "LLB"),
]

# (combo, level, prior_tier, target_tier, reg_prev, pass_prev, carry_prev, grade_prev, reg_curr, pass_curr)
ROWS = [
    # ---- target tier 0 (0.00-0.99): severe failure, 9 rows ----
    (0, "100 level", 0, 0, 12, 6, 6, "D", 10, 3),
    (6, "300 level", 1, 0, 10, 6, 4, "D", 9, 3),    # -1: backlog compounds
    (2, "400 level", 0, 0, 8, 2, 6, "D", 8, 2),
    (20, "200 level", 1, 0, 6, 3, 3, "C", 6, 1),    # -1
    (3, "100 level", 0, 0, 11, 4, 7, "D", 11, 3),
    (11, "200 level", 0, 0, 7, 2, 5, "D", 7, 2),
    (23, "200 level", 2, 0, 10, 7, 3, "C", 10, 2),  # -2: sharp collapse
    (8, "300 level", 0, 0, 9, 3, 6, "D", 8, 2),
    (28, "100 level", 0, 0, 12, 5, 7, "D", 12, 4),
    # ---- target tier 1 (1.50-2.39): at-risk, 13 rows ----
    (22, "300 level", 1, 1, 6, 5, 1, "C", 6, 5),
    (1, "400 level", 1, 1, 11, 9, 2, "C", 12, 8),
    (29, "200 level", 1, 1, 9, 6, 3, "C", 9, 6),
    (17, "200 level", 1, 1, 9, 7, 2, "C", 8, 6),
    (5, "300 level", 2, 1, 8, 7, 1, "C", 8, 5),     # -1: failures appear
    (19, "200 level", 1, 1, 9, 6, 3, "D", 9, 6),
    (24, "400 level", 0, 1, 14, 8, 6, "D", 14, 9),  # +1: partial recovery
    (12, "300 level", 1, 1, 7, 5, 2, "C", 8, 5),
    (15, "100 level", 1, 1, 11, 8, 3, "C", 11, 8),
    (4, "400 level", 2, 1, 15, 12, 3, "C", 15, 10), # -1
    (27, "300 level", 1, 1, 8, 6, 2, "C", 8, 6),
    (10, "300 level", 0, 1, 10, 5, 5, "D", 9, 6),   # +1: backlog easing
    (9, "200 level", 1, 1, 12, 9, 3, "C", 12, 9),
    # ---- target tier 2 (2.40-3.49): 14 rows ----
    (19, "300 level", 2, 2, 9, 9, 1, "B", 8, 7),
    (22, "200 level", 3, 2, 9, 9, 0, "B", 9, 8),    # -1: slight slide
    (23, "300 level", 2, 2, 10, 9, 1, "C", 10, 9),
    (10, "400 level", 2, 2, 9, 9, 0, "C", 9, 8),
    (26, "400 level", 2, 2, 8, 8, 1, "C", 8, 7),
    (18, "200 level", 2, 2, 11, 10, 1, "C", 11, 10),
    (14, "300 level", 1, 2, 7, 5, 2, "C", 7, 6),    # +1: clearing carryover
    (21, "300 level", 2, 2, 8, 8, 0, "C", 8, 7),
    (25, "100 level", 2, 2, 10, 10, 0, "B", 10, 9),
    (7, "400 level", 3, 2, 10, 10, 0, "B", 10, 8),  # -1
    (6, "400 level", 2, 2, 10, 9, 1, "C", 9, 8),
    (11, "300 level", 2, 2, 6, 6, 0, "C", 6, 5),
    (30, "400 level", 1, 2, 15, 12, 3, "C", 15, 13),# +1
    (13, "200 level", 2, 2, 10, 9, 1, "C", 10, 9),
    # ---- target tier 3 (3.50-4.49): majority-class shape, 13 rows ----
    (32, "400 level", 3, 3, 9, 9, 0, "B", 9, 9),
    (16, "300 level", 3, 3, 5, 5, 0, "B", 5, 5),
    (26, "500 level", 3, 3, 8, 8, 0, "B", 8, 8),
    (15, "100 level", 3, 3, 11, 11, 0, "B", 11, 11),
    (2, "400 level", 2, 3, 15, 14, 1, "B", 15, 15), # +1
    (10, "400 level", 3, 3, 8, 8, 0, "B", 9, 9),
    (18, "100 level", 3, 3, 11, 11, 0, "C", 11, 11),
    (31, "300 level", 3, 3, 7, 7, 0, "B", 5, 5),
    (13, "300 level", 4, 3, 10, 10, 0, "A", 10, 9), # -1: small dip
    (28, "400 level", 3, 3, 8, 8, 0, "C", 8, 8),
    (1, "300 level", 2, 3, 11, 10, 1, "B", 11, 11), # +1
    (12, "300 level", 3, 3, 8, 8, 0, "B", 8, 8),
    (25, "200 level", 1, 3, 10, 8, 2, "C", 10, 10), # +2: strong turnaround
    # ---- target tier 4 (4.50-5.00): zero failures, 11 rows ----
    (10, "400 level", 4, 4, 8, 8, 0, "A", 8, 8),
    (20, "400 level", 4, 4, 5, 5, 0, "A", 5, 5),
    (13, "300 level", 4, 4, 10, 10, 0, "A", 10, 10),
    (32, "400 level", 4, 4, 9, 9, 0, "A", 9, 9),
    (31, "500 level", 4, 4, 5, 5, 0, "A", 5, 5),
    (32, "400 level", 3, 4, 9, 9, 0, "A", 9, 9),    # +1
    (33, "400 level", 4, 4, 6, 6, 0, "A", 6, 6),
    (14, "400 level", 4, 4, 7, 7, 0, "A", 7, 7),
    (27, "400 level", 3, 4, 8, 8, 0, "A", 8, 8),    # +1
    (6, "300 level", 2, 4, 9, 8, 1, "B", 9, 9),     # +2
    (20, "300 level", 4, 4, 7, 7, 0, "A", 7, 7),
]


def build() -> pd.DataFrame:
    rows = []
    for i, (combo, level, prior, target, rp, pp, cp, grade, rc, pc) in enumerate(ROWS):
        institution, faculty, department, programme, qualification = COMBOS[combo]
        rows.append({
            "student_id_hash": f"synth-llm-{i:03d}",
            "institution_name": institution,
            "institution_name_suspect": False,
            "institution_type": "University",
            "faculty": faculty,
            "department": department,
            "programme__course_of_study": programme,
            "qualification_type": qualification,
            "programme_category": "Undergraduate",
            "current_academic_level": level,
            "prior_gpa_bin": UNI_BINS[prior],
            "prior_cgpa_bin": UNI_BINS[prior],
            "prior_tier": prior,
            "registered_prev": rp,
            "passed_prev": pp,
            "carryover_prev": cp,
            "carryover_prev_ambiguous": False,
            "avg_grade_prev": grade,
            "registered_curr": rc,
            "passed_curr": pc,
            "passed_curr_corrected": False,
            "failed_curr": rc - pc,
            "target_gpa_bin": UNI_BINS[target],
            "target_cgpa_bin": UNI_BINS[target],
            "target_tier": target,
            "at_risk": target <= 1,
            "is_synthetic": True,
            "generation_method": "llm",
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = build()

    assert (df["passed_curr"] <= df["registered_curr"]).all()
    assert (df["passed_prev"] <= df["registered_prev"]).all()
    assert (df["passed_curr"] + df["failed_curr"] == df["registered_curr"]).all()

    df.to_csv(OUT_PATH, index=False)
    delta = df["target_tier"] - df["prior_tier"]
    print(f"generated {len(df)} LLM synthetic rows -> {OUT_PATH}")
    print("\ntarget_tier counts:")
    print(df["target_tier"].value_counts().sort_index())
    print(f"\nnon-persistent share: {(delta != 0).mean():.3f} (real: 0.286)")
    print("delta distribution:")
    print(delta.value_counts().sort_index())


if __name__ == "__main__":
    main()
