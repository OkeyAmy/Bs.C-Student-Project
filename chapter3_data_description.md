# Chapter 3 — Data Description and Exploratory Analysis

## 3.1 Dataset Overview

The study dataset, `dataset_with_postgraduate_students.csv`, contains **164 records**
comprising **45 real respondent records** and **119 synthetic (augmented) records**
generated to support the minority-class prediction tasks. After cleaning and
deduplication the underlying real cohort is **61 respondents**, of which **56 are
University students** used for modeling; the remaining 3 Polytechnic and 2
unrecognized records are retained for transparency but excluded from modeling
because they are too few to train or evaluate on.

The complete dataset is described by **28 columns** spanning institutional
attributes (institution, faculty, department, programme), student demographics at
the aggregate-semester level (academic level, prior and current CGPA tiers, course
registration/pass/carryover counts), predictive targets (next-semester CGPA tier,
at-risk flag), and provenance/metadata fields. **There are no missing values** in
any column.

### Dataset composition (Figure 7)

| Source | Count |
|---|---|
| Real survey records | 45 |
| SMOTE-NC synthetic | 9 |
| LLM-authored synthetic | 60 |
| **Total** | **164** |

## 3.2 Data Preparation Pipeline

The raw data went through a reproducible pipeline (`clean.py` → `split.py` →
`synth_stat.py` → `synth_llm.py` → `combine.py`):

1. **Deduplication** — the raw Google Form export contained each respondent
   twice (122 rows → 61 unique) due to differing timestamp formats; timestamps
   were dropped before deduplication.
2. **Schema coalescing** — University and Polytechnic branches carried different
   CGPA scales; these were unified into an ordinal `*_tier` (0 = worst, 4 = best)
   within each row's own scale.
3. **Field correction** — merged carryover artifacts (`"0; 1"` → max),
   passed > registered corrections, and institution-name normalization were each
   flagged with a dedicated Boolean column.
4. **Privacy** — emails were replaced with a truncated SHA-256 hash
   (`student_id_hash`); no raw email is stored anywhere.
5. **Synthetic augmentation** — 9 SMOTE-NC rows (interpolating between real
   minority rows) and 60 LLM-authored rows (grounded in real
   institution/faculty/department combinations) were added to the 45 real
   training rows, giving 114 augmented training rows. The held-out test set is
   **100% real and never sees synthetic data**, preserving an honest evaluation
   holdout.

## 3.3 Train/Test Split

A **stratified 80/20 split** by `target_tier` was applied to the 56 usable
University records, giving **45 training and 11 held-out test records**. Because
there is no student-ID overlap between the two partitions, there is **no data
leakage** between training and testing.

### Table 3.1 — Train/test distribution by target CGPA tier

| Tier (CGPA range) | Train | Test | Total |
|---|---|---|---|
| 0 (0.00–0.99) | 1 | 0 | 1 |
| 1 (1.50–2.39) | 2 | 1 | 3 |
| 2 (2.40–3.49) | 10 | 2 | 12 |
| 3 (3.50–4.49) | 26 | 7 | 33 |
| 4 (4.50–5.00) | 6 | 1 | 7 |
| **Total** | **45** | **11** | **56** |

**Note:** The dataset is highly imbalanced. Tier 3 is the majority class
(26/45 of training), while tiers 0 and 1 (the at-risk group) have only 3 real
training examples combined. Tier 0 (one real member) cannot appear in both
partitions and is forced entirely into the training set — consequently the test
set contains no tier-0 record and a single tier-1 record. Per-class metrics on
this 11-row test set are therefore illustrative rather than statistically
reliable; cross-validation over the real rows is the primary evaluation scheme.

## 3.4 Feature Relevance and Correlation

On the real training data (`train_real.csv`, n=45), the numeric features most
strongly associated with next-semester CGPA tier are:

| Feature | Correlation with `target_tier` |
|---|---|
| `prior_tier` | **+0.66** |
| `failed_curr` | **−0.55** |
| `carryover_prev` | −0.24 |
| `registered_curr` | −0.20 |
| `passed_curr` | −0.15 |
| `registered_prev` | −0.08 |
| `passed_prev` | −0.06 |

The previous-semester CGPA tier is by far the strongest single predictor, and
the current-semester failure count is the strongest behavioural signal. These
findings confirm that course-registration counts carry substantially less
information than the prior tier itself.

## 3.5 Column Disposition (Keep vs Drop)

The 28 columns are reduced for modeling. The following categorization supports
the feature-selection decisions:

### 3.5.1 Dropped columns — identifiers and quasi-identifiers
| Column | Reason |
|---|---|
| `student_id_hash` | Identifier/hash with no predictive value |
| `department` | Too granular (40 unique values, near instance-level) on the modeling set |
| `programme__course_of_study` | Too granular (64 unique values), effectively an identifier |
| `institution_type` | Constant (100% University) on the modeling set |

### 3.5.2 Dropped columns — cleaning/provenance flags
| Column | Reason |
|---|---|
| `institution_name_suspect` | Cleaning flag, not a feature |
| `carryover_prev_ambiguous` | Cleaning flag, not a feature |
| `passed_curr_corrected` | Cleaning flag, not a feature |
| `is_synthetic` | Provenance tag (used to separate real/synthetic rows) |
| `generation_method` | Provenance tag |

### 3.5.3 Redundant/duplicate-information columns
| Column | Redundant with |
|---|---|
| `prior_cgpa_bin`, `prior_gpa_bin` | scaled as ordinal `prior_tier` |
| `target_cgpa_bin`, `target_gpa_bin` | scaled as ordinal `target_tier` |
| `at_risk` | `target_tier <= 1` (94.5% derivation match) |
| `failed_curr` | `registered_curr − passed_curr` (95.7% match) |
| `carryover_prev` | `registered_prev − passed_prev` (84.8% match) |

### 3.5.4 Retained feature set
The retained modeling features reduce the schema to:

- **Targets:** `target_tier` (5-class), `at_risk` (binary)
- **Numeric predictors:** `prior_tier`, `registered_prev`, `passed_prev`,
  `carryover_prev`, `registered_curr`
- **Categorical predictors:** `institution_name`, `faculty`,
  `qualification_type`, `programme_category`, `current_academic_level`,
  `avg_grade_prev`

`institution_name`, `qualification_type` and `programme_category` are kept
deliberately: the model is trained on the augmented synthetic corpus that is
seeded from real institution/faculty/programme combinations, so retaining these
columns lets the model learn and round-trip those grouping signals. `department`
and `programme__course_of_study` are excluded because their many distinct values
make them effectively identifiers that add no generalizable signal.

This removes the identifiers, cleaning flags, and fully redundant columns while
retaining the academic and demographic features the model actually learns from.
