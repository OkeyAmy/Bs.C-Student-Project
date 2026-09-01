"""Clean the raw merged survey export into one coalesced schema.

Source has two problems baked in:
1. Same 61 respondents appear twice (CSV export + XLSX export of the same
   Google Form, concatenated by merge.py). Timestamp format/timezone differs
   between exports so drop_duplicates() there missed it.
2. The form branches into two parallel column blocks (university vs
   polytechnic) with different GPA bin scales (max 5.00 vs max 4.00).
   These aren't missing values to impute — they're two schemas in one sheet.
"""

import hashlib
import re

import pandas as pd

RAW_PATH = "Student Programme Information (merged).csv"
OUT_PATH = "data/cleaned_students.csv"

# University GPA/CGPA bins, worst to best (max 5.00 scale)
UNI_BINS = ["0.00 - 0.99", "1.50 - 2.39", "2.40 - 3.49", "3.50 - 4.49", "4.50 - 5.00"]
# Polytechnic GPA/CGPA bins, worst to best (max 4.00 scale)
POLY_BINS = ["0.00 - 2.00", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"]

INSTITUTION_PATTERNS = [
    (r"nnamdi|azikiwe|unizik", "Nnamdi Azikiwe University, Awka"),
    (r"lagos state (university|poly)|\blasu\b", "Lagos State University"),
    (r"lagos state polytechnic", "Lagos State Polytechnic"),
    (r"federal university otuoke", "Federal University Otuoke"),
    (r"university of lagos|\bunilag\b", "University of Lagos"),
    (r"university of port harcourt", "University of Port Harcourt"),
    (r"university of nigeria|nsukka|\bunn\b", "University of Nigeria, Nsukka"),
    (r"benue state university", "Benue State University"),
    (r"olabisi onabanjo", "Olabisi Onabanjo University"),
    (r"unique university", "Unique University"),
    (r"enugu state university", "Enugu State University of Science and Technology"),
    (r"unical|university of calabar", "University of Calabar"),
    (r"yaba college", "Yaba College of Technology"),
    (r"khemsafe", "Khemsafe Polytechnic"),
    (r"noun|national open university", "National Open University of Nigeria"),
]

# Values that are clearly a person's name typed into the institution field,
# not an institution — seen directly in the raw data.
KNOWN_NAME_ARTIFACTS = {
    "Eze David",
    "Daniel Jonathan Yunana",
    "Eyibio Daniel uwem",
}


def normalize_institution(raw: str) -> tuple[str, bool]:
    """Return (canonical_name, is_suspect). is_suspect flags values that
    don't look like an institution at all (e.g. a person's name)."""
    if not isinstance(raw, str) or not raw.strip():
        return ("Unknown", True)
    val = raw.strip()
    if val in KNOWN_NAME_ARTIFACTS:
        return ("Unknown", True)
    low = val.lower()
    for pattern, canonical in INSTITUTION_PATTERNS:
        if re.search(pattern, low):
            return (canonical, False)
    looks_institutional = bool(
        re.search(r"university|college|polytechnic|institute|poly\b", low)
    )
    return (val, not looks_institutional)


def coalesce_carryover(raw) -> tuple[float, bool]:
    """Some respondents typed multiple numbers into a numeric field
    (e.g. '0; 1'). Take the max as the conservative (worse-case) reading
    and flag the row as having an ambiguous source value."""
    if pd.isna(raw):
        return (float("nan"), False)
    s = str(raw)
    if ";" in s:
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
        return (max(nums) if nums else float("nan"), True)
    try:
        return (float(s), False)
    except ValueError:
        return (float("nan"), True)


def hash_email(email: str) -> str:
    if not isinstance(email, str) or not email.strip():
        return ""
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]


def load_and_clean(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    df = df.drop(columns=["Timestamp"]).drop_duplicates().reset_index(drop=True)

    df["student_id_hash"] = df["Email"].apply(hash_email)
    df = df.drop(columns=["Email"])

    inst = df["Institution Name"].apply(normalize_institution)
    df["institution_name"] = [x[0] for x in inst]
    df["institution_name_suspect"] = [x[1] for x in inst]

    df["institution_type"] = df["Institution Type"].where(
        df["Institution Type"].isin(["University", "Polytechnic"]), other=pd.NA
    )

    is_uni = df["institution_type"] == "University"
    is_poly = df["institution_type"] == "Polytechnic"

    def pick(uni_col: str, poly_col: str) -> pd.Series:
        return df[uni_col].where(is_uni, df[poly_col].where(is_poly))

    df["prior_gpa_bin"] = pick(
        "Second to last semester GPA range",
        "Second to last semester GPA range (For Polytechnic Students)",
    )
    df["prior_cgpa_bin"] = pick(
        "CGPA range before  last completed semester  ",
        "CGPA range before  last completed semester  (For Polytechnic Students)",
    )
    df["registered_prev"] = pick(
        "Number of courses registered in the semester before your most recent completed semester ",
        "Number of courses registered in the semester before your most recent completed semester .1",
    )
    df["passed_prev"] = pick(
        "Number of courses passed in the semester before your most recent completed semester ",
        "Number of courses passed in the semester before your most recent completed semester .1",
    )
    carry_raw = pick(
        "Total carryover courses before your most recent completed semester ",
        "Total carryover courses before your most recent completed semester .1",
    )
    carry = carry_raw.apply(coalesce_carryover)
    df["carryover_prev"] = [x[0] for x in carry]
    df["carryover_prev_ambiguous"] = [x[1] for x in carry]

    df["avg_grade_prev"] = pick(
        "Average grade across most courses in the semester before your most recent completed semester ",
        "Average grade across most courses in the semester before your most recent completed semester .1",
    )
    df["registered_curr"] = pick(
        "Number of courses registered in your most recent completed semester ",
        "Number of courses registered in your most recent completed semester .1",
    )
    df["passed_curr"] = pick(
        "Number of courses passed in your most recent completed semester ",
        "Number of courses passed in your most recent completed semester .1",
    )
    df["failed_curr"] = pick(
        "Number of courses failed in your most recent completed semester ",
        "Number of courses failed in your most recent completed semester .1",
    )
    df["target_gpa_bin"] = pick(
        "GPA range for your most recent completed semester ",
        "GPA range for your most recent completed semester  (For Polytechnic Students)",
    )
    df["target_cgpa_bin"] = pick(
        "Current CGPA range after your most recent completed semester ",
        "Current CGPA range after your most recent completed semester  (For Polytechnic Students)",
    )

    # passed can't exceed registered; if failed is 0 the honest fix is
    # passed = registered (the passed figure was the typo).
    bad = df["passed_curr"] > df["registered_curr"]
    df["passed_curr_corrected"] = bad
    df.loc[bad & (df["failed_curr"] == 0), "passed_curr"] = df.loc[
        bad & (df["failed_curr"] == 0), "registered_curr"
    ]

    def tier(bin_series: pd.Series, is_uni_mask: pd.Series) -> pd.Series:
        out = pd.Series(pd.NA, index=bin_series.index, dtype="Int64")
        for scale, mask in [(UNI_BINS, is_uni_mask), (POLY_BINS, ~is_uni_mask)]:
            order = {b: i for i, b in enumerate(scale)}
            out.loc[mask] = bin_series.loc[mask].map(order).astype("Int64")
        return out

    df["target_tier"] = tier(df["target_cgpa_bin"], is_uni)
    df["prior_tier"] = tier(df["prior_cgpa_bin"], is_uni)

    keep = [
        "student_id_hash",
        "institution_name",
        "institution_name_suspect",
        "institution_type",
        "Faculty",
        "Department",
        "Programme/ Course of Study",
        "Qualification Type",
        "Programme Category ",
        "Current Academic Level",
        "prior_gpa_bin",
        "prior_cgpa_bin",
        "prior_tier",
        "registered_prev",
        "passed_prev",
        "carryover_prev",
        "carryover_prev_ambiguous",
        "avg_grade_prev",
        "registered_curr",
        "passed_curr",
        "passed_curr_corrected",
        "failed_curr",
        "target_gpa_bin",
        "target_cgpa_bin",
        "target_tier",
    ]
    out = df[keep].copy()
    out.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in out.columns]

    # Respondents typed trailing spaces, which otherwise split one category
    # into two ("Education" vs "Education ") in every downstream crosstab.
    for col in out.select_dtypes(include=["object", "str"]).columns:
        out[col] = out[col].str.strip()

    # Objective 2 target: at-risk = CGPA tier 0 or 1 (below 2.40).
    out["at_risk"] = out["target_tier"] <= 1

    out["is_synthetic"] = False
    return out


def main() -> None:
    import os

    df = load_and_clean()
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"cleaned rows: {len(df)} -> {OUT_PATH}")
    print(f"institution_name_suspect rows: {df['institution_name_suspect'].sum()}")
    print(f"carryover_prev_ambiguous rows: {df['carryover_prev_ambiguous'].sum()}")
    print(f"passed_curr_corrected rows: {df['passed_curr_corrected'].sum()}")
    print(df["institution_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
