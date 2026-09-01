# Student Academic Performance Prediction

A machine-learning project that forecasts how a student will do next semester
from their past record: carried or failed courses, pass rate, prior CGPA tier,
faculty, institution, and qualification type.

One model produces two outputs:

- **target_tier** — the coming semester's CGPA tier on a 0–4 scale (0 = failing,
  4 = first class).
- **at_risk** — a yes/no flag pulled straight from the tier: a predicted tier
  of 0 or 1 counts as at-risk. It is not a second model; it is just the bottom
  two tiers read off the tier prediction.

## Layout

| Path | Purpose |
| --- | --- |
| `data/cleaned_students.csv` | The real student records this project rests on. |
| `data/corpus_train.csv` | 448 synthetic training rows the model learns from. |
| `data/corpus_test.csv` | 112 synthetic holdout rows used for evaluation. |
| `src/school_project/` | All the code, from cleaning through evaluation. |
| `reports/` | Metric tables, confusion matrices, ROC plot, feature importance. |
| `models/student_predictor.pkl` | The trained RBF SVM (`C=3, gamma=0.03`). |

Set up with `uv sync`, then:

```bash
python -m school_project.build_corpus   # rebuild the corpus and 80:20 split
python -m school_project.train          # fit + save the model, print metrics
python -m school_project.evaluation     # full metrics, CV, plots -> reports/
```

## A word on honesty

Only about sixty real student records were ever gathered. That is nowhere near
enough to train a dependable classifier by itself, so synthetic students
(generated from patterns in the real rows) make up most of the training data.
Because those generated rows are treated as ground truth, the numbers in
`reports/` are proof that the **pipeline runs** — they are not proof the model
would work on a fresh real cohort. The single finding that does rest on the
real records is the weakness profile: students who come in at tier 1–2 with a C
average are the most likely to be at-risk the following semester.

## What we actually tested along the way

### Algorithms tried

Not every model was kept. The experiments ran through several algorithm
families before settling on the final one:

- **Random Forest (`evaluate.py`)** — used early on to test whether the LLM
  augmentation actually helped, against majority-class and persistence
  baselines. It also scores each fold in the real-data CV schemes. Dropped as
  the final model because the SVM generalised better on the synthetic holdout.
- **Gradient Boosting** — a GBM (sklearn gradient boosting, per the pre-tuning
  pipeline) was tuned first but swapped out for the SVM after tuning; it never
  became the committed model.
- **RBF SVM (`SVC(C=3, gamma=0.03)`, final)** — the tuned winner in
  `train.py`/`evaluation.py`. Chosen because on the synthetic corpus it edged
  out the boosted tree model on tier accuracy and stayed stable on at-risk.
- **Baselines** — majority-class, persistence (prior tier == target tier),
  prior-tier-only SVM, and trivial prior-direct. These set the floor that the
  full model must beat.

The comparison table below is the head-to-head evidence for why the full
feature SVM is the final model.

### The overfitting problem and the fix

The at-risk label is easy to make trivially predictable if the generator is too
polished. An early version of the synthetic corpus separated at-risk perfectly,
and the model returned **at-risk = 1.000** on every metric. That is
memorisation, not learning. How we diagnosed it:

- Every SVM setting on the old corpus still gave at-risk 1.000, which pointed
  squarely at the **generator** rather than the model.
- The generator was rewritten so at-risk is genuinely hard to separate:
  carryover and pass rate are noisy projections of the tier, and a **5%
  label-flip** term means that even a flawless predictor tops out around 0.95
  balanced accuracy on at-risk.
- After the change the scores became honest: training ≈ test, CV ≈ holdout, and
  nothing sits at 1.000.

### Comparing models

From `reports/model_comparison.csv`, three models scored on the same 112-row
holdout:

| Model | tier acc | tier bal | at_risk bal | at_risk rec |
| --- | --- | --- | --- | --- |
| **Full-feature SVM (final)** | **0.857** | **0.838** | **0.852** | 0.714 |
| prior_tier-only SVM | 0.562 | 0.581 | 0.740 | 0.786 |
| Trivial baseline (reuse prior_tier) | 0.661 | 0.493 | — | — |

What this tells us:

- The full feature set beats the prior-tier-only model on tier accuracy
  (0.857 vs 0.562) and on at-risk balanced accuracy (0.852 vs 0.740). The extra
  features — pass rate, carryover, faculty, institution — add real signal
  beyond last semester's tier.
- The silly baseline of just repeating the prior tier manages only 0.661 tier
  accuracy and 0.493 balanced, so the model clearly does better than copying.
- The prior-tier-only model has higher at-risk recall (0.786) but a weaker
  overall balance — a trade-off worth mentioning in the write-up.

### Did augmentation actually help? (Random-Forest test)

Before the synthetic-corpus pipeline existed, a dedicated experiment in
`evaluate.py` asked whether LLM-generated rows help at all, using a Random
Forest scored on the untouched real test set:

- Baselines set the floor: majority-class and persistence
  (prior tier == target tier).
- Repeated CV over the **real rows only** (n=56) estimated how much signal
  exists beyond persistence, and a 3-way comparison ("real-only training" vs
  "+LLM-synthetic") measured the augmentation effect on at-risk detection.
- The result was a mostly **null-to-modest** augmentation effect on the tiny
  real test (11 rows, 1–4 at-risk) — differences below ~0.09 are within one
  test row. This is why the project pivoted from "augment the 45 real rows"
  to the fuller synthetic corpus approach.

### Holdout versus cross-validation

From `reports/scores_holdout_vs_cv.csv`, the single 20%-holdout against 10-fold
stratified CV on the training set:

| Metric | Holdout | CV mean (± std) |
| --- | --- | --- |
| tier balanced acc | 0.838 | 0.855 ± 0.046 |
| at_risk balanced acc | 0.852 | 0.810 ± 0.095 |

CV averages over many splits instead of one lucky draw, so it is the more
defensible headline. Holdout and CV line up closely, which supports the
no-overfit reading.

### Real-data control

We also ran the synthetic-trained model over the real records. It looks strong
on its own holdout (tier 0.857) but falls to roughly **tier 0.475** on the real
rows. That steep drop is the clearest sign that synthetic training does not
transfer to real students, and it is exactly why the thesis frames the demo
scores as a method check rather than a claim of generalisation.

### Data-quality checks

- Duplicates: none — no repeat rows and no repeat student hashes.
- Impossible values: three rows with `passed_prev > registered_prev` were
  capped; no out-of-range counts remain.
- Categorical hygiene: qualification typos merged (`B. Ed`/`B.Ed`, `L.LB`/
  `LLB`), faculty casing merged (`Physical science`→`Physical Sciences`), and
  level labels normalised (`400L`→`400 level`, `Done`/`Graduate`→`Unknown`).
- Label consistency: `at_risk` matches `target_tier <= 1` on every usable row.
- Two all-NaN placeholder rows are dropped so label casting does not crash.

## The synthetic paths we tried, and why each failed

- **Statistical-only (`synth_stat.py`)** — rows came from the marginal
  distributions with no link between features and label. Too noisy to separate
  tiers; abandoned.
- **LLM-based (`synth_llm.py`)** — LLM-invented students looked believable but
  had inconsistent ranges and over-clean class separation, so accuracy looked
  better than it was.
- **Hand-crafted / smart (`synth_hand.py`, `rebuild_smart.py`)** — strong
  training signal, but pushed too far it made at-risk trivially separable
  (1.000 scores, a textbook overfit).
- **Final (`build_corpus.py`)** — tier-driven noisy features plus a 5%
  label-flip, giving a corpus that stays honest and stable across
  train/test/CV. This is the generator behind the committed data.

## Chapter 3 plots

`chapter3_plots/` holds the data-exploration figures for the thesis: tier
distributions, prior-tier distribution, academic level, faculty, correlation
heatmap, at-risk rates, and dataset composition.
