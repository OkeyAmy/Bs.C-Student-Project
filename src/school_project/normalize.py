"""Normalize the master training file into a clean, modeling-ready dataset.

Reads data/train_master.csv (real + synthetic rows; the file the model is
trained on) and writes data/train_clean.csv.

Three kinds of intervention:
  1. FIX real-data entry errors that are impossible, not just noisy.
     passed_prev > registered_prev (3 rows, all carryover=0 and present in
     the source as respondent typos) -> passed_prev capped at registered_prev.
     This mirrors the correction clean.py already applies to passed_curr.
  2. NORMALIZE categorical label casing/spacing so one concept is not split
     into several categories (e.g. "Physical science" vs "Physical Sciences").
  3. DROP columns that carry no predictive signal: IDs, audit flags,
     constants, binned CGPA duplicates, provenance tags, and current-semester
     outcome columns (which would leak the answer into the features).

data/test_real.csv is deliberately NOT rewritten — it stays 100% real and
untouched so it remains a valid honest holdout.
"""

import pandas as pd

IN_PATH = "data/train_master.csv"
OUT_PATH = "data/train_clean.csv"

# Columns excluded from the model-ready output: no predictive signal (IDs,
# audit flags, constants, binned CGPA duplicates), provenance tags (real vs
# synthetic), or current-semester outcomes that would leak the answer.
MINIMAL_DROP = [
    "student_id_hash",
    "institution_name_suspect",
    "institution_type",
    "carryover_prev_ambiguous",
    "passed_curr_corrected",
    "prior_gpa_bin",
    "prior_cgpa_bin",
    "target_gpa_bin",
    "target_cgpa_bin",
    "is_synthetic",
    "generation_method",
    "passed_curr",
    "failed_curr",
    "department",
    "programme__course_of_study",
]

# Case/spacing normalization cannot be a simple .lower() because faculty names
# are proper-noun phrases; build explicit canonical maps.
FACULTY_MAP = {
    "physical science": "Physical Sciences",
    "physical sciences": "Physical Sciences",
    "social science": "Social Sciences",
    "social sciences": "Social Sciences",
    "management science": "Management Sciences",
    "management sciences": "Management Sciences",
    "faculty of education": "Education",
    "education": "Education",
    "art": "Art",
    "arts": "Art",
}
QUALIFICATION_MAP = {
    "l.lb": "LLB",
    "llb": "LLB",
    "b. ed": "B.Ed",
    "b.ed": "B.Ed",
    "b.a": "BA",
}


def _canonical(value: str, mapping: dict[str, str], default: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    key = value.strip().lower()
    return mapping.get(key, value.strip())


def _academic_level_canonical(value: str) -> str:
    if not isinstance(value, str):
        return "Unknown"
    v = value.strip()
    # "Done", "Graduate", "ND 1" are stray/ambiguous labels; the valid scheme
    # is "<number> level" for undergraduates and "Year <n>" for postgraduates.
    if not v or v.lower() in {"done", "graduate", "nd 1", "na", "-", "400l"}:
        return "Unknown"
    # "400L" style -> "400 level"
    if v.lower().rstrip("l").strip().isdigit() and not v.lower().endswith("level"):
        return f"{v.rstrip('l').strip()} level"
    return v


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bad_prev = df["passed_prev"] > df["registered_prev"]
    df.loc[bad_prev, "passed_prev"] = df.loc[bad_prev, "registered_prev"]

    df["faculty"] = df["faculty"].apply(
        lambda v: _canonical(v, FACULTY_MAP, "Unknown"))
    df["qualification_type"] = df["qualification_type"].apply(
        lambda v: _canonical(v, QUALIFICATION_MAP, "Unknown"))
    df["current_academic_level"] = df["current_academic_level"].apply(
        _academic_level_canonical)

    return df


def main() -> None:
    df = pd.read_csv(IN_PATH)
    before = len(df)
    fixed = int((df["passed_prev"] > df["registered_prev"]).sum())
    df = normalize(df)
    dropped = [c for c in MINIMAL_DROP if c in df.columns]
    df = df.drop(columns=dropped)
    df.to_csv(OUT_PATH, index=False)

    print(f"rows: {before} -> {len(df)}")
    print(f"passed_prev>registered_prev fixed: {fixed}")

    print("\nqualified current_academic_level:")
    print(df["current_academic_level"].value_counts().to_string())
    print("\nfaculty (normalized):")
    for k, v in df["faculty"].value_counts().items():
        print(f"  {k}: {v}")
    print("\nqualification_type (normalized):")
    for k, v in df["qualification_type"].value_counts().items():
        print(f"  {k}: {v}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
