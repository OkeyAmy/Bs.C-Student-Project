"""Model loading + single-student prediction logic (shared by API & tests)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from school_project.build_corpus import TIER_TO_GPA
from school_project.normalize import normalize
from school_project.preprocess import CATEGORICAL, NUMERIC
from school_project.train import FEATURES, MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]

# Tier (0..4) -> (Nigerian class name). Matches the CGPA bands in TIER_TO_GPA.
TIER_CLASS = {
    0: "Failing / probation",
    1: "Third Class",
    2: "Second Class Lower",
    3: "Second Class Upper",
    4: "First Class",
}

# Concrete driver rules for the "why this student is at risk" explanation.
AT_RISK_RULES = [
    ("carryover_prev", None, lambda v: v >= 3),
    ("prior_tier", None, lambda v: v <= 1),
    ("avg_grade_prev", None, lambda v: v == "C"),
]


def _format_reason(feat, value):
    if feat == "carryover_prev":
        return f"{value} carryovers"
    if feat == "prior_tier":
        return f"prior tier {value}"
    if feat == "avg_grade_prev":
        return f"avg grade {value}"
    return f"{feat}={value}"


def load_model():
    path = Path(ROOT) / MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"model not found at {path} — run `python -m school_project.train` first"
        )
    return joblib.load(path)


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def build_row(req) -> pd.DataFrame:
    """Build a 1-row feature frame, normalized exactly like training."""
    row = {
        **{c: getattr(req, c) for c in NUMERIC},
        **{c: getattr(req, c) for c in CATEGORICAL},
    }
    df = pd.DataFrame([row])
    return normalize(df)[FEATURES]


def predict(req) -> dict:
    """Run the model and return the CGPA-flavoured prediction dict."""
    model = get_model()
    X = build_row(req)

    tier = int(np.asarray(model.predict(X))[0])
    at_risk = tier <= 1
    band = TIER_TO_GPA[tier][0]

    reasons = []
    if at_risk:
        for feat, _label, rule in AT_RISK_RULES:
            value = getattr(req, feat)
            if rule(value):
                reasons.append(_format_reason(feat, value))
        if not reasons:
            reasons.append("prior performance signals")

    return {
        "tier": tier,
        "cgpa_band": band,
        "class_name": TIER_CLASS[tier],
        "at_risk": at_risk,
        "at_risk_reason": reasons,
        "numeric_input": {
            "prior_tier": req.prior_tier,
            "registered_prev": req.registered_prev,
            "passed_prev": req.passed_prev,
            "carryover_prev": req.carryover_prev,
            "registered_curr": req.registered_curr,
        },
    }