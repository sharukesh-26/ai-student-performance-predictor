"""
Streamlit website for the AI Student Performance Predictor.

What this file does:
- Loads the trained model from models/student_model.pkl
- Lets you type in study habits
- Shows a predicted exam score
- Explains the model's test-set accuracy honestly

Run it from the project folder:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "students.csv"
MODEL_PATH = ROOT / "models" / "student_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

NUMERIC_FEATURES = [
    "hours_studied",
    "attendance_percent",
    "previous_score",
    "sleep_hours",
    "assignments_completed",
]
CATEGORICAL_FEATURES = [
    "extra_curricular",
    "parental_support",
]


def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load the saved model: {exc}")
        return None


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    import json

    try:
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read model metrics: {exc}")
        return None
    if not isinstance(payload, dict):
        st.warning("The metrics file is not in the expected format.")
        return None
    return payload


def load_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as exc:
        st.warning(f"Could not read the sample dataset: {exc}")
        return None


def interpret_score(score: float) -> tuple[str, str]:
    if score >= 85:
        return "Excellent", "This predicted score is in a strong range."
    if score >= 70:
        return "Good", "This predicted score is solid, with room to grow."
    if score >= 55:
        return "Average", "This predicted score is around the middle. Extra study time may help."
    return "Needs support", "This predicted score is on the lower side. Focus on hours, attendance, and sleep."


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f3f7ff 0%, #eefaf6 100%);
            }
            .hero {
                background: linear-gradient(135deg, #1f4e79 0%, #2a9d8f 100%);
                color: white;
                padding: 1.4rem 1.6rem;
                border-radius: 18px;
                margin-bottom: 1.2rem;
            }
            .hero h1 {
                margin: 0 0 0.35rem 0;
                font-size: 1.8rem;
            }
            .hero p {
                margin: 0;
                opacity: 0.95;
            }
            .card {
                background: white;
                border: 1px solid #d9e4f5;
                border-radius: 16px;
                padding: 1rem 1.1rem;
                box-shadow: 0 8px 24px rgba(31, 78, 121, 0.08);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_missing_model() -> None:
    st.error("No trained model was found yet.")
    st.markdown(
        """
        Train the model first, then refresh this page.

        **In a terminal, from this project folder, run:**
        ```bash
        python train_model.py
        streamlit run app.py
        ```
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="AI Student Performance Predictor",
        page_icon="🎓",
        layout="wide",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>🎓 AI Student Performance Predictor</h1>
            <p>Enter study habits and see a predicted exam score. Built for learning with Python, pandas, scikit-learn, and Streamlit.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    metrics = load_metrics()
    dataset = load_dataset()

    if model is None:
        render_missing_model()
        st.stop()

    tab_predict, tab_quality, tab_data = st.tabs(
        ["Predict a score", "How accurate is the model?", "Sample dataset"]
    )

    with tab_predict:
        st.subheader("Student details")
        st.caption("These inputs match the columns used to train the model.")

        left, right = st.columns(2)
        with left:
            hours_studied = st.slider(
                "Hours studied per day",
                min_value=0.0,
                max_value=12.0,
                value=4.0,
                step=0.5,
            )
            attendance_percent = st.slider(
                "Attendance (%)",
                min_value=50,
                max_value=100,
                value=85,
                step=1,
            )
            previous_score = st.slider(
                "Previous exam score",
                min_value=0,
                max_value=100,
                value=72,
                step=1,
            )
        with right:
            sleep_hours = st.slider(
                "Sleep hours per night",
                min_value=3.0,
                max_value=12.0,
                value=7.0,
                step=0.5,
            )
            assignments_completed = st.slider(
                "Assignments completed (out of 10)",
                min_value=0,
                max_value=10,
                value=7,
                step=1,
            )
            extra_curricular = st.selectbox(
                "Extra-curricular activities",
                options=["Yes", "No"],
                index=1,
            )
            parental_support = st.selectbox(
                "Parental support",
                options=["Low", "Medium", "High"],
                index=1,
            )

        predict_clicked = st.button("Predict exam score", type="primary")

        if predict_clicked:
            row = pd.DataFrame(
                [
                    {
                        "hours_studied": hours_studied,
                        "attendance_percent": attendance_percent,
                        "previous_score": previous_score,
                        "sleep_hours": sleep_hours,
                        "assignments_completed": assignments_completed,
                        "extra_curricular": extra_curricular,
                        "parental_support": parental_support,
                    }
                ]
            )
            try:
                raw_prediction = float(model.predict(row)[0])
                prediction = min(100.0, max(0.0, raw_prediction))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                label, advice = interpret_score(prediction)
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted exam score", f"{prediction:.1f} / 100")
                m2.metric("Band", label)
                if metrics and "test" in metrics:
                    test_mae = metrics["test"].get("mae")
                    if test_mae is not None:
                        m3.metric("Typical error (test MAE)", f"{test_mae:.1f} pts")
                st.info(advice)
                if abs(prediction - raw_prediction) > 0.05:
                    st.caption(
                        f"The raw model output was {raw_prediction:.1f}. "
                        "It was clipped to the 0–100 exam range."
                    )
                st.caption(
                    "This is a teaching demo on synthetic data, not a grade for a real student."
                )

    with tab_quality:
        st.subheader("Honest model report")
        st.write(
            "The data was split into **80% training** and **20% testing**. "
            "The numbers below on the **test set** are the ones that matter, "
            "because those students were hidden during training."
        )

        if not metrics:
            st.warning(
                "Metrics file not found. Run `python train_model.py` to recreate it."
            )
        else:
            test = metrics.get("test", {})
            train = metrics.get("train", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Test MAE", f"{test.get('mae', float('nan')):.2f}")
            c2.metric("Test RMSE", f"{test.get('rmse', float('nan')):.2f}")
            c3.metric("Test R²", f"{test.get('r2', float('nan')):.3f}")

            st.markdown(
                """
                **What these numbers mean**
                - **MAE**: on average, the prediction is this many score points away from the true score.
                - **RMSE**: similar to MAE, but punishes bigger mistakes more.
                - **R²**: 1.0 would be a perfect fit. 0.0 means no better than guessing the average. Negative would mean worse than that average.
                """
            )

            st.write("Training-set scores (usually a little better; not the official result):")
            t1, t2, t3 = st.columns(3)
            t1.metric("Train MAE", f"{train.get('mae', float('nan')):.2f}")
            t2.metric("Train RMSE", f"{train.get('rmse', float('nan')):.2f}")
            t3.metric("Train R²", f"{train.get('r2', float('nan')):.3f}")

            st.caption(
                f"Model: {metrics.get('model_type', 'Linear Regression')} · "
                f"Rows: {metrics.get('n_rows', '?')} · "
                f"Train/test: {metrics.get('n_train', '?')}/{metrics.get('n_test', '?')}"
            )

        st.warning(
            "This project uses made-up learning data. Real school decisions need real data, "
            "fairness checks, and human judgment."
        )

    with tab_data:
        st.subheader("Sample students.csv")
        st.write(
            "Every row is a fictional student. The target column is **exam_score**. "
            "The model tries to estimate that score from the other columns."
        )
        if dataset is None:
            st.error("data/students.csv was not found.")
        else:
            st.dataframe(dataset, width="stretch", hide_index=True)
            st.caption(f"{len(dataset)} rows loaded from data/students.csv")

        with st.expander("Column meanings"):
            st.markdown(
                """
                | Column | Meaning |
                | --- | --- |
                | `student_id` | Fake ID, not used by the model |
                | `hours_studied` | Study hours per day |
                | `attendance_percent` | Class attendance from 50 to 100 |
                | `previous_score` | Score on a previous exam |
                | `sleep_hours` | Sleep per night |
                | `assignments_completed` | How many of 10 assignments were done |
                | `extra_curricular` | Yes or No |
                | `parental_support` | Low, Medium, or High |
                | `exam_score` | Target we want to predict (0–100) |
                """
            )


if __name__ == "__main__":
    main()
