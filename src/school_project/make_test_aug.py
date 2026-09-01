"""Add FRESH hand-authored synthetic rows to the test set so it covers all
five tiers (currently test_real.csv has tier-0=0, tier-4=1).

Control discipline: the existing synthetic pools (LLM, hand, SMOTE) are all
part of the TRAINING set. Reusing any of them in test would be leakage (the
model memorized them), so the rows added here are newly authored and never
seen by the trainer.

Outputs:
  * data/test_augmented.csv  - real test (11) + fresh hand rows (recommended use)
  * data/test_real.csv       - left UNTOUCHED as the pure-real control
"""

import hashlib

import pandas as pd

REAL_TEST = "data/test_real.csv"
OUT = "data/test_augmented.csv"

# Fresh, never-seen-by-train hand rows. Two tier-0 (model must predict the
# worst at-risk class - currently untestable on the real set) and three
# tier-4 (real test has only 1).
NEW_ROWS = [
    # ---- tier 0 (at-risk) ----
    dict(institution_name="Nnamdi Azikiwe University, Awka", faculty="Social sciences",
         department="Mass communication", programme__course_of_study="B.Sc Mass communication",
         qualification_type="BSc", programme_category="Undergraduate",
         current_academic_level="200 level", prior_gpa_bin="1.00-1.49", prior_cgpa_bin="1.00-1.49",
         prior_tier=1, registered_prev=9, passed_prev=3, carryover_prev=6,
         carryover_prev_ambiguous=False, avg_grade_prev="D", registered_curr=9,
         passed_curr=3, passed_curr_corrected=False, failed_curr=6,
         target_gpa_bin="0.00-0.99", target_cgpa_bin="0.00-0.99", target_tier=0, at_risk=True),
    dict(institution_name="Lagos State University", faculty="Faculty Of Education",
         department="Guidance and counseling psychology", programme__course_of_study="BSCE Guidance and counseling psychology",
         qualification_type="BSc", programme_category="Undergraduate",
         current_academic_level="300 level", prior_gpa_bin="1.00-1.49", prior_cgpa_bin="1.00-1.49",
         prior_tier=0, registered_prev=8, passed_prev=2, carryover_prev=6,
         carryover_prev_ambiguous=False, avg_grade_prev="D", registered_curr=8,
         passed_curr=3, passed_curr_corrected=False, failed_curr=5,
         target_gpa_bin="0.00-0.99", target_cgpa_bin="0.00-0.99", target_tier=0, at_risk=True),
    # ---- tier 4 (top) ----
    dict(institution_name="Nnamdi Azikiwe University, Awka", faculty="Health Sciences",
         department="Medicine and Surgery", programme__course_of_study="MBBS",
         qualification_type="MBBS", programme_category="Undergraduate",
         current_academic_level="400 level", prior_gpa_bin="4.50-5.00", prior_cgpa_bin="4.50-5.00",
         prior_tier=4, registered_prev=10, passed_prev=10, carryover_prev=0,
         carryover_prev_ambiguous=False, avg_grade_prev="A", registered_curr=10,
         passed_curr=10, passed_curr_corrected=False, failed_curr=0,
         target_gpa_bin="4.50-5.00", target_cgpa_bin="4.50-5.00", target_tier=4, at_risk=False),
    dict(institution_name="University of Lagos", faculty="Management Sciences",
         department="Actuarial Science", programme__course_of_study="B.Sc Actuarial Science",
         qualification_type="BSc", programme_category="Undergraduate",
         current_academic_level="300 level", prior_gpa_bin="4.50-5.00", prior_cgpa_bin="4.50-5.00",
         prior_tier=4, registered_prev=8, passed_prev=8, carryover_prev=0,
         carryover_prev_ambiguous=False, avg_grade_prev="A", registered_curr=8,
         passed_curr=8, passed_curr_corrected=False, failed_curr=0,
         target_gpa_bin="4.50-5.00", target_cgpa_bin="4.50-5.00", target_tier=4, at_risk=False),
    dict(institution_name="University of Port Harcourt", faculty="Clinical Sciences",
         department="Dentistry", programme__course_of_study="BDS Dentistry",
         qualification_type="BDS", programme_category="Undergraduate",
         current_academic_level="200 level", prior_gpa_bin="4.50-5.00", prior_cgpa_bin="4.50-5.00",
         prior_tier=4, registered_prev=9, passed_prev=9, carryover_prev=0,
         carryover_prev_ambiguous=False, avg_grade_prev="A", registered_curr=9,
         passed_curr=9, passed_curr_corrected=False, failed_curr=0,
         target_gpa_bin="4.50-5.00", target_cgpa_bin="4.50-5.00", target_tier=4, at_risk=False),
]


def main() -> None:
    real = pd.read_csv(REAL_TEST)
    new = pd.DataFrame(NEW_ROWS)
    new["student_id_hash"] = [
        hashlib.sha256(f"test-hand-{i}".encode()).hexdigest()[:16] for i in range(len(new))
    ]
    new["institution_name_suspect"] = False
    new["institution_type"] = "University"
    new["is_synthetic"] = True
    new["generation_method"] = "hand"
    # align to the real test's 27-col schema (drop generation_method, real set lacks it)
    new = new[list(real.columns)]

    test_aug = pd.concat([real, new], ignore_index=True)
    test_aug.to_csv(OUT, index=False)

    print("test_real.csv  (control, untouched):", real.shape,
          "tier", real["target_tier"].value_counts().sort_index().to_dict())
    print("test_augmented.csv:", test_aug.shape,
          "tier", test_aug["target_tier"].value_counts().sort_index().to_dict())
    print("at_risk:", test_aug["at_risk"].value_counts().to_dict())
    print("synthetic rows added:", new.shape, "tier", new["target_tier"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()