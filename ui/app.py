"""Streamlit UI for the CGPA prediction demo.

Sidebar-driven navigation across three sections:
    1. Predict        — enter a student record, get the predicted tier + at-risk flag
    2. Comparison     — real-data at-risk/performance comparison (carryover vs none)
    3. About          — model card, data provenance, endpoint reference

Run (with the API up on :8000):
    streamlit run ui/app.py

The UI reads the API base URL from the API_URL env var so the same code runs
locally (default http://localhost:8000) and deployed.
"""

import os

import matplotlib.pyplot as plt
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


def get_comparison() -> dict:
    resp = requests.get(f"{API_URL}/comparison", timeout=30)
    resp.raise_for_status()
    return resp.json()


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("#### Student Gateway")
        st.caption("CGPA Prediction Thesis Demo")
        st.divider()
        page = st.radio(
            "Section",
            ["Predict a semester", "Carryover comparison", "About this demo"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Model API\n{API_URL}")
    return page


def render_predict(page: str) -> None:
    if page != "Predict a semester":
        return
    st.subheader("Predict a semester", anchor=False)
    st.caption(
        "Enter the prior-semester record below. The model returns the predicted "
        "next-semester CGPA tier (0–4 on the 5.00 scale) and flags the student as "
        "at risk when the tier is ≤ 1."
    )
    st.divider()

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("##### Student record (previous semester)")
        with st.container(border=True):
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

        st.markdown("##### Course numbers")
        with st.container(border=True):
            st.caption("Previous semester unless stated otherwise")
            registered_prev = st.slider("Courses registered (prev)", 1, 15, 8)
            passed_prev = st.slider("Courses passed (prev)", 0, 15, 7)
            carryover_prev = st.slider("Carryovers (prev)", 0, 15, 1)
            registered_curr = st.slider("Courses registered (current)", 1, 15, 8)

    with right:
        st.markdown("##### Result")
        with st.container(border=True):
            predict_clicked = st.button(
                "Predict", type="primary", use_container_width=True
            )

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

                    st.markdown("###### Predicted next-semester CGPA tier")
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


def render_comparison(page: str) -> None:
    if page != "Carryover comparison":
        return
    st.subheader("Carryover comparison", anchor=False)
    st.caption(
        "Real survey records only (cleaned_students.csv — no synthetic rows): "
        "at-risk rates and next-semester performance for students with vs without "
        "prior-semester carryovers."
    )
    st.divider()

    try:
        data = get_comparison()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the model API. Start it locally with "
            "`python -m api.main` (runs on :8000), or set API_URL."
        )
        return
    except Exception as e:  # defensive
        st.error(f"Comparison failed: {e}")
        return

    groups = data["groups"]
    tests = data["tests"]
    no, with_ = groups["no"], groups["with"]

    st.markdown("##### Key figures")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("At-risk — no carryover", f"{no['at_risk_rate'] * 100:.1f}%", f"n = {no['n']}")
    m2.metric("At-risk — with carryover", f"{with_['at_risk_rate'] * 100:.1f}%", f"n = {with_['n']}")
    m3.metric(
        "Rate ratio",
        f"×{tests['at_risk_risk_ratio']:.1f}",
        f"odds ≈ {tests['at_risk_odds_ratio']:.1f}",
    )
    m4.metric("Total records", f"{data['n_total']}", "real only")

    st.markdown("##### Charts")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    labels = ["No carryover", "With carryover"]
    at_rates = [no["at_risk_rate"] * 100, with_["at_risk_rate"] * 100]
    bars = axes[0].bar(labels, at_rates, color=["#4ade80", "#f87171"])
    for bar, val in zip(bars, at_rates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center",
            fontsize=11,
        )
    axes[0].set_ylabel("At-risk rate (%)")
    axes[0].set_ylim(0, max(at_rates) * 1.3)
    axes[0].set_title("At-risk flag (tier ≤ 1)")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    mean_tiers = [no["mean_tier"], with_["mean_tier"]]
    bars = axes[1].bar(labels, mean_tiers, color=["#22c55e", "#ef4444"])
    for bar, val in zip(bars, mean_tiers):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"{val:.2f}",
            ha="center",
            fontsize=11,
        )
    axes[1].set_ylabel("Mean CGPA tier (0–4)")
    axes[1].set_ylim(0, 4.3)
    axes[1].set_title("Next-semester performance")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("##### Significance")
    cols = st.columns(4)
    cols[0].metric("Chi-square p", f"{tests['at_risk_chi2_p']:.3f}", "not significant")
    cols[1].metric("Fisher (one-sided)", f"{tests['at_risk_fisher_p']:.3f}", "not significant")
    cols[2].metric(
        "Performance p (Mann–Whitney)", f"{tests['performance_mannwhitney_p']:.3f}", "borderline"
    )
    cols[3].metric("Mean prior tier", f"{with_['mean_prior_tier']:.2f} vs {no['mean_prior_tier']:.2f}", "pre-existing gap")

    st.warning(
        "**Interpretation:** students with carryovers show a ~3.7× higher at-risk rate "
        "(17.6% vs 4.8%), but with only 59 real records the difference is **not "
        f"statistically significant** (χ² p={tests['at_risk_chi2_p']:.3f}, Fisher "
        f"p={tests['at_risk_fisher_p']:.3f}). Treat it as a trend, not proof of cause. "
        "The with-carryover group also arrived with a lower prior tier "
        f"(mean {with_['mean_prior_tier']:.2f} vs {no['mean_prior_tier']:.2f}), so part "
        "of the next-semester gap is pre-existing rather than caused by carryover itself."
    )


def render_about(page: str) -> None:
    if page != "About this demo":
        return
    st.subheader("About this demo", anchor=False)
    st.divider()

    st.markdown("##### What it does")
    st.write(
        "A thesis demo of a full ML pipeline: collect → clean → train → serve → "
        "present. It predicts the next-semester CGPA tier (0–4 on a 5.00 scale) "
        "from a prior-semester record, and flags at-risk students (predicted tier ≤ 1)."
    )

    st.markdown("##### Data")
    st.write(
        "The corpus mixes 59 real survey records with a seeded synthetic "
        "expansion (560 rows, 80:20 train/test). The model is therefore a "
        "**pipeline demonstration — not evidence of real-world accuracy**."
    )
    st.code(
        "data/cleaned_students.csv   # 59 real records\n"
        "data/corpus_train.csv        # synthetic train split\n"
        "data/corpus_test.csv         # synthetic test split",
        language="text",
    )

    st.markdown("##### Model card")
    with st.container(border=True):
        st.markdown(
            "**Algorithm:** RBF SVM — `SVC(C=3.0, gamma=0.03, class_weight='balanced')`\n\n"
            "**Features (11):** prior_tier, registered_prev, passed_prev, carryover_prev, "
            "registered_curr, institution_name, faculty, qualification_type, "
            "programme_category, current_academic_level, avg_grade_prev\n\n"
            "**Tiers:** 0 Failing · 1 Third · 2 Lower Second · 3 Upper Second · 4 First "
            "(score cut-offs 1.50 / 2.40 / 3.50 / 4.50)."
        )

    st.markdown("##### Endpoints")
    st.code(
        f"GET  {API_URL}/health       → status: ok\n"
        f"POST {API_URL}/predict      → predict a student tier (JSON)\n"
        f"GET  {API_URL}/comparison   → real-data carryover comparison",
        language="text",
    )

    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
        st.success(f"API reachable — {health.get('status')}")
    except Exception:
        st.error("API not reachable. Start it locally with `python -m api.main`.")


PAGE = render_sidebar()

if PAGE == "Predict a semester":
    render_predict(PAGE)
elif PAGE == "Carryover comparison":
    render_comparison(PAGE)
else:
    render_about(PAGE)