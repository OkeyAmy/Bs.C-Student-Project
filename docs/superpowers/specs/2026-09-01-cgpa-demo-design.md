# Design: Streamlit + FastAPI CGPA Prediction Demo

Date: 2026-09-01
Status: Approved (architecture, components, error handling, testing/deployment all confirmed by user)

## Goal

A thesis-defense demo where an advisor types a student's prior-semester record and
immediately sees the predicted next-semester **CGPA tier** (0–4), the **at-risk flag**
(tier ≤ 1), and an explanation of which inputs drove the prediction. Everything is
surfaced in CGPA/GPA language (tier → CGPA band → Nigerian class name), not abstract
tier numbers alone.

## User choices (from brainstorming)

- Layout: **A — split layout** (form left, result card right), chosen from 3 visual mockups.
- Stack: **Streamlit (UI) + FastAPI (model API)** — two processes.
- Scope: **Prediction only** (no metrics tab, no batch upload).
- Result: **Tier + at-risk + explanation ("why" chips)**.
- Run during defense: **deployed to the cloud**.

## CGPA / tier mapping (5.00 scale, from build_corpus.py)

| Tier | CGPA band | Letter | Class (Nigerian) |
|---|---|---|---|
| 0 | 0.00–1.49 | D | Failing / probation |
| 1 | 1.50–2.39 | C | Third Class |
| 2 | 2.40–3.49 | B | Second Class Lower |
| 3 | 3.50–4.49 | A | Second Class Upper |
| 4 | 4.50–5.00 | A | First Class |

`at_risk = tier <= 1`. Source of truth: `TIER_TO_GPA` in `src/school_project/build_corpus.py:35`.

## Architecture

```
User (browser) ─► Streamlit UI (:8501) ── HTTP ──► FastAPI model API (:8000)
                                                    └─ loads models/student_predictor.pkl
```

### `api/main.py` — FastAPI service
- `POST /predict` accepts a student JSON (11 features), runs the model pipeline,
  returns `{tier, cgpa_band, class_name, at_risk, at_risk_reason, carryover_prev, prior_tier}`.
- Derives `at_risk` from `tier <= 1`.
- Derives `cgpa_band` + `class_name` from the tier→CGPA map.
- Builds `at_risk_reason` from the driving inputs with concrete rules:
  - `carryover_prev >= 3` → chip "N carryovers" (high carry count).
  - `prior_tier <= 1` → chip "prior tier {prior_tier}".
  - `avg_grade_prev == "C"` → chip "avg grade C".
  - If `at_risk` is False → `at_risk_reason = []`.
  - If `at_risk` is True and no chip fires → `at_risk_reason = ["prior performance signals"]`.
- On missing model file → HTTP 500 with a clear "run `python -m school_project.train`" message.

### `ui/app.py` — Streamlit page
- Split layout via `st.columns([3, 2])`:
  - Left: the input form.
  - Right: the result card + button.
- Sends the form to `POST /predict` via `requests`.
- On connection failure, shows a friendly hint instead of a stack trace.

### Shared helpers
- `api/schema.py` — Pydantic request/response models (validation → 422 with clear message).
- `api/predict.py` — loads `.pkl`, builds the 11-feature row, maps tier→CGPA.

## Components

### UI form inputs (from real feature set)
- Dropdowns: `current_academic_level` (100–500 level), `qualification_type`
  (BSc/BA/BEng/LLB/B.Ed/PhD), `avg_grade_prev` (A/B/C), `faculty`, `institution_name`.
- Numbers (sliders/steppers bound to valid ranges): `prior_tier` (1–4),
  `registered_prev` (1–15), `passed_prev` (1–15), `carryover_prev` (0–15),
  `registered_curr` (1–15).
- `programme_category` is fixed to "Undergraduate" (only value in training).

### Prediction flow
1. User fills the form → clicks **Predict**.
2. Streamlit posts JSON to `/predict`.
3. FastAPI builds the row, `model.predict()` → tier (0–4).
4. FastAPI derives at-risk, CGPA band, class name, reason.
5. Streamlit renders the result card.

### Result card (right column)
- Big predicted CGPA tier + "Second Class Upper · CGPA 3.50–4.49".
- At-risk flag (green "Not at risk" / red "AT RISK").
- Why-chips: e.g. "5 carryovers", "prior tier 1", "avg C".
- Reset button.

## Error handling & edge cases

- API not running → Streamlit shows "Start the API with `python -m api.main`".
- Numeric ranges enforced in the UI (sliders bound).
- FastAPI validates types via Pydantic → 422 with the missing/invalid key named.
- Unknown categorical values (e.g. avg grade "D") → `OneHotEncoder(handle_unknown="ignore")`
  contributes no signal (no crash).
- Model missing → API 500 with clear instruction.

## Testing

- `api/test_api.py` (pytest + FastAPI TestClient):
  - Known students → correct tier/at-risk/CGPA band.
  - Unknown category doesn't crash.
  - Bad payload → 422.
- Cross-check: 2 rows from `corpus_test.csv` → API prediction matches `train.py` metrics.
- Manual UI smoke test: run both processes, confirm result card renders.

## Deployment (cloud)

- FastAPI: container (`Dockerfile`) + `uvicorn api.main:app` on Railway/Render/Fly.io → `:8000`.
- Streamlit: second container on same platform (or Streamlit Community Cloud),
  `API_URL` env var pointing at the FastAPI service.
- `docker-compose.yml` for local combined runs; deployed Streamlit points at deployed API URL via env var.

## Out of scope

- Metrics tab, batch CSV upload, model retraining UI, real-time monitoring.
- Production auth (demo only).