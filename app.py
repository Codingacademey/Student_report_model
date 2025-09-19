import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ---------- App Config ----------
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Predictor")
st.caption("Enter a student's habits and well-being to estimate the exam score.")


# ---------- Model Loading ----------
@st.cache_resource(show_spinner=True)
def load_pipeline(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


MODEL_PATH = Path("pipe.pkl")
pipeline = None
model_load_error = None
try:
    pipeline = load_pipeline(MODEL_PATH)
except Exception as exc:
    model_load_error = exc


# ---------- Feature Schema (must match training) ----------
# After preprocessing in the notebook, the training features were numeric:
# [
#   'study_hours_per_day', 'social_media_hours', 'netflix_hours',
#   'sleep_hours', 'exercise_frequency', 'mental_health_rating',
#   'study_social_ratio'
# ]
FEATURE_ORDER = [
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "sleep_hours",
    "exercise_frequency",
    "mental_health_rating",
    "study_social_ratio",
]


with st.sidebar.expander("How to run", expanded=False):
    st.markdown(
        "1. Install requirements: `pip install streamlit scikit-learn xgboost joblib pandas numpy`\n"
        "2. Start the app from the project folder:\n\n"
        "   `streamlit run app.py`\n\n"
        "3. Ensure `pipe.pkl` is present in the same folder."
    )


if model_load_error is not None:
    st.error(
        "Could not load the trained pipeline. "
        "Make sure `pipe.pkl` exists and dependencies are installed.\n\n"
        f"Details: {model_load_error}"
    )
    st.stop()


# ---------- Input Form ----------
with st.form("input_form"):
    st.subheader("Inputs")

    col1, col2 = st.columns(2)
    with col1:
        study_hours_per_day = st.number_input(
            "Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5
        )
        social_media_hours = st.number_input(
            "Social media hours per day", min_value=0.0, max_value=24.0, value=2.0, step=0.5
        )
        netflix_hours = st.number_input(
            "Netflix/streaming hours per day", min_value=0.0, max_value=24.0, value=1.0, step=0.5
        )
    with col2:
        sleep_hours = st.number_input(
            "Sleep hours per day", min_value=0.0, max_value=24.0, value=7.0, step=0.5
        )
        exercise_frequency = st.number_input(
            "Exercise frequency (sessions per week)", min_value=0, max_value=14, value=3, step=1
        )
        mental_health_rating = st.slider(
            "Mental health rating (1-10)", min_value=1, max_value=10, value=6, step=1
        )

    submitted = st.form_submit_button("Predict Exam Score")


def compute_study_social_ratio(study_hours: float, social_media: float) -> float:
    # +1 in denominator to match the notebook logic and avoid divide-by-zero
    return float(study_hours) / (float(social_media) + 1.0)


def build_feature_frame(
    study_hours_per_day: float,
    social_media_hours: float,
    netflix_hours: float,
    sleep_hours: float,
    exercise_frequency: int,
    mental_health_rating: int,
) -> pd.DataFrame:
    study_social_ratio = compute_study_social_ratio(study_hours_per_day, social_media_hours)

    row = {
        "study_hours_per_day": float(study_hours_per_day),
        "social_media_hours": float(social_media_hours),
        "netflix_hours": float(netflix_hours),
        "sleep_hours": float(sleep_hours),
        "exercise_frequency": int(exercise_frequency),
        "mental_health_rating": int(mental_health_rating),
        "study_social_ratio": float(study_social_ratio),
    }
    # Ensure column order matches training
    df = pd.DataFrame([row], columns=FEATURE_ORDER)
    return df


if submitted:
    try:
        features_df = build_feature_frame(
            study_hours_per_day=study_hours_per_day,
            social_media_hours=social_media_hours,
            netflix_hours=netflix_hours,
            sleep_hours=sleep_hours,
            exercise_frequency=exercise_frequency,
            mental_health_rating=mental_health_rating,
        )

        with st.spinner("Predicting..."):
            pred = pipeline.predict(features_df)
        # Clip prediction to [0, 100]
        clipped = np.clip(pred, 0.0, 100.0)
        predicted_score = float(np.squeeze(clipped))

        st.success("Prediction complete")
        st.metric(label="Estimated Exam Score", value=f"{predicted_score:.2f}")

        with st.expander("View input features"):
            st.dataframe(features_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


