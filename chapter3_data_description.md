# Chapter 3 — Data Description and Exploratory Analysis

## 3.1 Dataset Overview

The study rests on a Google-Form survey exported as
`Student Programme Information (merged).csv`. Running it through
`clean.py` yields `data/cleaned_students.csv`, and that cleaned file holds
**61 respondents** spread across **27 columns** — institution, faculty,
department, programme, academic level, course registration/pass/carryover
tallies, prior and current CGPA tiers, the predictive targets, and provenance
flags. Two of the 61 rows come through with every academic field blank (no
institution type, no tiers, no course counts). They are unusable and get
dropped, leaving **59 usable real records**: **56 University students** and 3
from Polytechnics. The Polytechnic rows stay in the dataset because they are
genuine survey answers, but there are too few of them to model on their own.

Fifty-nine records are far too few to train a dependable classifier on, so a
**synthetic demonstration corpus** is produced from the patterns in those real
rows (`build_corpus.py`). The generator takes the real data as a seed, which
keeps the synthetic distribution close to reality — real institutions, the
real marginal-tier spread, real prior→target transition probabilities — while
each output row is a freshly drawn synthetic student rather than a copy. The
result is a **560-row corpus** cut 80:20 into **448 training** and **112 test**
rows (`corpus_train.csv`, `corpus_test.csv`).

### The two datasets, honestly labelled

| Source | Rows | Role |
|---|---|---|
| Real survey, cleaned (`cleaned_students.csv`) | 61 (59 usable) | Ground truth seed; comparison analysis |
| Synthetic corpus (`corpus_train.csv` / `corpus_test.csv`) | 448 / 112 | Demo training and evaluation ground truth |

The fully-synthetic corpus is treated as ground truth *for the demo*: the
model is judged on how well it picks up the defined prior→target relationship
inside this synthetic world. It is **not** a claim of real-world generalisation
— see the honesty note in Section 3.3.

## 3.2 Data Preparation Pipeline

The real data moves through a reproducible pipeline:
`clean.py` → `build_corpus.py` → `train.py`.

1. **Deduplication** — repeated survey exports were merged; near-duplicate
   rows that differed only in timestamp format were collapsed before the
   timestamps themselves were discarded.
2. **Schema coalescing** — University and Polytechnic branches grade on
   different CGPA scales; both are folded into an ordinal `*_tier` (0 = worst,
   4 = best) evaluated within each row's own scale.
3. **Field correction** — merged carryover artifacts (`"0; 1"` → max),
   passed > registered corrections, and institution-name normalisation each get
   their own dedicated Boolean flag column
   (`carryover_prev_ambiguous`, `passed_curr_corrected`,
   `institution_name_suspect`).
4. **Privacy** — emails are swapped for a truncated SHA-256 hash
   (`student_id_hash`); no raw email is stored anywhere.
5. **Synthetic augmentation** — `build_corpus.py` draws a deterministic
   (seeded) 560-row corpus from the real records. It adds no SMOTE or
   LLM-generated rows; the single `corpus` generation method reproduces the
   real transition probabilities, course loads by level, and irreducible label
   noise so the model can never overfit to 1.000 metrics.

## 3.3 Train/Test Split

The model is trained and evaluated against the synthetic corpus using a
**stratified 80:20 split** (`build_corpus.py`, `TRAIN_FRAC = 0.80`) that gives
**448 training** and **112 test** rows. Stratification follows `target_tier`,
and `student_id_hash` differs between the partitions, so there is **no
leakage**.

### Train/test distribution across the corpus

| Tier (CGPA range) | Train | Test | Total |
|---|---|---|---|
| 0 (0.00–0.99) | 11 | 3 | 14 |
| 1 (1.50–2.39) | 33 | 8 | 41 |
| 2 (2.40–3.49) | 97 | 25 | 122 |
| 3 (3.50–4.49) | 245 | 61 | 306 |
| 4 (4.50–5.00) | 62 | 15 | 77 |
| **Total** | **448** | **112** | **560** |

**Honesty note.** The corpus test set is *synthetic*, so whatever the
112-row metrics say, they only measure how well the model learned the defined
relationship in the synthetic world — the reported ~0.86 accuracy and ~0.84
balanced accuracy are demo values, not real-world evidence. The real 59
records cannot sustain a reliable training/validation split on their own; they
are reserved for the descriptive comparisons in Section 3.6.

## 3.4 Feature Relevance and Correlation

On the **real usable records** (n = 59), the numeric features with the
strongest ties to next-semester CGPA tier are:

| Feature | Correlation with `target_tier` |
|---|---|
| `prior_tier` | **+0.685** |
| `failed_curr` | **−0.484** |
| `carryover_prev` | −0.261 |
| `registered_curr` | −0.196 |
| `passed_curr` | −0.140 |
| `registered_prev` | −0.076 |
| `passed_prev` | −0.056 |

The previous-semester CGPA tier is by a wide margin the strongest single
predictor; after it come the current-semester failure/carryover signals as the
strongest behavioural ones. Registration counts carry markedly less information
than the prior tier itself. The carryover relationship (−0.261) gets treated
as a comparison in Section 3.6.

## 3.5 Column Disposition (Keep vs Drop)

The 27 columns are trimmed down for modeling. The breakdown below backs the
feature-selection calls:

### 3.5.1 Dropped columns — identifiers and quasi-identifiers
| Column | Reason |
|---|---|
| `student_id_hash` | Identifier/hash with no predictive value |
| `department` | Too granular (40 unique values) — adds zero accuracy on tests |
| `programme__course_of_study` | Too granular (64 unique values), effectively an identifier |
| `institution_type` | Near-constant on the modeling set (56 University / 3 Polytechnic) |

### 3.5.2 Dropped columns — cleaning/provenance flags
| Column | Reason |
|---|---|
| `institution_name_suspect` | Cleaning flag, not a feature |
| `carryover_prev_ambiguous` | Cleaning flag, not a feature |
| `passed_curr_corrected` | Cleaning flag, not a feature |
| `is_synthetic` | Provenance tag (distinguishes real vs synthetic rows) |

### 3.5.3 Redundant/duplicate-information columns
| Column | Redundant with |
|---|---|
| `prior_gpa_bin`, `prior_cgpa_bin` | scaled as ordinal `prior_tier` |
| `target_gpa_bin`, `target_cgpa_bin` | scaled as ordinal `target_tier` |
| `at_risk` | `target_tier <= 1` (definitional) |
| `failed_curr` | `registered_curr − passed_curr` |
| `carryover_prev` | `registered_prev − passed_prev` |

### 3.5.4 Retained feature set
The retained modeling features (used by `train.py`) are:

- **Targets:** `target_tier` (5-class), `at_risk` (binary, derived as
  `target_tier <= 1`)
- **Numeric predictors:** `prior_tier`, `registered_prev`, `passed_prev`,
  `carryover_prev`, `registered_curr`
- **Categorical predictors:** `institution_name`, `faculty`,
  `qualification_type`, `programme_category`, `current_academic_level`,
  `avg_grade_prev`

`institution_name`, `qualification_type`, and `programme_category` are held
back on purpose: the generator is seeded from real institution/faculty/programme
combinations, so keeping them lets the model learn and round-trip those
grouping signals that exist in the synthetic world. `department` and
`programme__course_of_study` are left out because their many distinct values
make them effectively identifiers that add no generalizable signal.

That removes the identifiers, the cleaning flags, and the fully redundant
columns while keeping the academic and demographic features the model actually
learns from.

## 3.6 Carryover Comparison: Students With vs Without Previous-Semester Carryovers

This section answers the study objective of *comparing* the academic
performance and at-risk rates of students with and without previous-semester
carryover courses. Unlike the model evaluation, this analysis draws on **real
records only** (`cleaned_students.csv`, n = 59 usable — 56 University and 3
Polytechnic); no synthetic rows take part, so the figures describe what the
survey cohort itself reports.

Groups are defined by the previous-semester carryover count:

- **No carryovers** — `carryover_prev = 0`
- **With carryovers** — `carryover_prev >= 1`

### Table 3.2 — Carryover comparison on real records (n = 59)

| Measure | No carryovers | With carryovers |
|---|---|---|
| Sample size | 42 | 17 |
| At-risk rate (`target_tier <= 1`) | 4.8% (2/42) | 17.6% (3/17) |
| Mean next-semester tier | 2.83 | 2.41 |
| Median next-semester tier | 3.0 | 3.0 |
| Mean prior-tier (incoming) | 2.86 | 2.12 |

Students carrying courses into the semester run a **~3.7× higher at-risk rate**
(17.6% vs 4.8%; risk ratio ≈ 3.7, odds ratio ≈ 4.3) and finish with a lower
mean next-semester tier (2.41 vs 2.83). The direction matches the correlation
reported in Section 3.4 (`carryover_prev` vs `target_tier` ≈ −0.26 on the
usable records).

### Is the difference significant?

| Test | Statistic | p-value |
|---|---|---|
| Chi-square (at-risk × group) | χ² ≈ 1.20 | 0.274 |
| Fisher exact (one-sided) | odds ≈ 4.29 | 0.138 |
| Mann–Whitney U (tier × group) | — | 0.063 |

**Interpretation:** none of the tests clears the 0.05 threshold. At n = 59 the
spread between 4.8% and 17.6% is a **trend, not a statistically significant
difference**, and it must not be read as proof that carryovers *cause* the
outcome. Two cautions keep the reading honest:

1. The with-carryover group already arrived with a lower prior tier (mean 2.12
   vs 2.86), so part of the next-semester gap is **pre-existing** — it mirrors
   the academic standing these students were in when the semester began, not
   just what the carryovers themselves added.
2. The rate difference would only become testable with a larger real cohort
   (roughly n ≈ 200–300 at these base rates).

The comparison is reproducible from the same pipeline used elsewhere in this
project and appears live in the dashboard's carryover-comparison section
(read straight from `cleaned_students.csv`; see the API `GET /comparison`
route).