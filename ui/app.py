"""Streamlit UI for the CGPA prediction demo (split layout A).

Run (with the API up on :8000):
    streamlit run ui/app.py

The UI reads the API base URL from the API_URL env var so the same code runs
locally (default http://localhost:8000) and deployed.
"""

import os

import requests
import streamlit as st

st.set_page_config(page_title="CGPA Prediction Demo", layout="wide")

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

_TIERS = [
    ("0", "Failing / probation", "0.00–1.49"),
    ("1", "Third Class", "1.50–2.39"),
    ("2", "Second Class Lower", "2.40–3.49"),
    ("3", "Second Class Upper", "3.50–4.49"),
    ("4", "First Class", "4.50–5.00"),
]

_INSTITUTIONS = [
    "Benue State University",
    "Enugu State University of Science and Technology",
    "Federal University Otuoke",
    "Lagos State University",
    "National Open University of Nigeria",
    "Nnamdi Azikiwe University, Awka",
    "Olabisi Onabanjo University",
    "Unique University",
    "University of Calabar",
    "University of Lagos",
    "University of Nigeria, Nsukka",
    "University of Port Harcourt",
    "Other / not listed",
]

_FACULTIES = [
    "Agriculture",
    "Allied",
    "Art",
    "Biosciences",
    "Clinical Sciences",
    "Education",
    "Educational and human psychology",
    "Engineering",
    "Health",
    "Law",
    "Management Sciences",
    "Pharmaceutical sciences",
    "Physical Sciences",
    "Science",
    "Social Sciences",
    "Technology and Vocational Education",
]

_QUALIFICATIONS = ["BSc", "BA", "BEng", "LLB", "B.Ed", "PhD"]
_LEVELS = ["100 level", "200 level", "300 level", "400 level", "500 level"]
_GRADES = ["A", "B", "C"]


def call_api(payload: dict) -> dict:
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


st.title("Student CGPA Prediction Demo")
st.caption(
    "A thesis demo — the model predicts the next-semester CGPA tier (0–4, 5.00 "
    "scale) from the prior-semester record, and flags students at risk (tier ≤ 1)."
)
st.caption(f"Model API: {API_URL}")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Student record (previous semester)")

    institution = st.selectbox("Institution", _INSTITUTIONS, index=0)
    faculty = st.selectbox("Faculty", _FACULTIES, index=0)
    qualification = st.selectbox("Qualification type", _QUALIFICATIONS, index=0)
    level = st.selectbox("Current academic level", _LEVELS, index=2)

    prior_tier = st.selectbox(
        "Prior CGPA tier",
        options=[t[0] for t in _TIERS[1:]],
        index=2,
        help="The student's current CGPA band, as a tier from 1 (lowest) to 4 (highest).",
    )
    grade = st.selectbox("Average grade (previous semester)", _GRADES, index=0)

    st.caption("Course numbers (previous semester unless noted)")
    registered_prev = st.slider("Courses registered (prev)", 1, 15, 8)
    passed_prev = st.slider("Courses passed (prev)", 0, 15, 7)
    carryover_prev = st.slider("Carryovers (prev)", 0, 15, 1)
    registered_curr = st.slider("Courses registered (current)", 1, 15, 8)

    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")

    if predict_clicked:
        payload = {
            "prior_tier": int(prior_tier),
            "registered_prev": int(registered_prev),
            "passed_prev": int(passed_prev),
            "carryover_prev": int(carryover_prev),
            "registered_curr": int(registered_curr),
            "institution_name": "Unknown" if institution.startswith("Other") else institution,
            "faculty": faculty,
            "qualification_type": qualification,
            "programme_category": "Undergraduate",
            "current_academic_level": level,
            "avg_grade_prev": grade,
        }
        try:
            res = call_api(payload)
            tier = res["tier"]
            band = res["cgpa_band"]
            class_name = res["class_name"]
            at_risk = res["at_risk"]
            reasons = res["at_risk_reason"]

            st.markdown("#### Predicted next-semester CGPA tier")
            st.markdown(
                f"<div style='font-size:44px;font-weight:800;'>{tier} "
                f"<span style='font-size:16px;font-weight:400;color:#888;'>/ 4</span></div>"
                f"<div style='font-size:18px;'>CGPA {band} · {class_name}</div>",
                unsafe_allow_html=True,
            )

            if at_risk:
                st.error("AT RISK — predicted tier ≤ 1")
                if reasons:
                    st.markdown("**Why:** " + ", ".join(reasons))
            else:
                st.success("Not at risk")
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach the model API. Start it locally with "
                "`python -m api.main` (runs on :8000), or set API_URL."
            )
        except Exception as e:  # defensive
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Fill in the student record and click **Predict**.")
        st.caption(
            "The model is an RBF SVM trained on a fully-synthetic 560-row corpus "
            "(80:20 train/test) seeded from 59 real records. It demonstrates the "
            "pipeline; it is not evidence of real-world accuracy."
        )