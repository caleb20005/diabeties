from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DATA_PATH = "data/pima-indians-diabetes.csv"
FEATURE_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]
TARGET_COLUMN = "outcome"
INVALID_ZERO_COLUMNS = [
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f6f8f7;
                --panel: #ffffff;
                --panel-soft: #edf3f1;
                --text: #16302c;
                --muted: #5d726c;
                --primary: #1d8b7a;
                --primary-dark: #136456;
                --accent: #f39c4a;
                --border: rgba(22, 48, 44, 0.10);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(29, 139, 122, 0.09), transparent 30%),
                    radial-gradient(circle at top right, rgba(243, 156, 74, 0.11), transparent 24%),
                    linear-gradient(180deg, #fbfcfb 0%, var(--bg) 100%);
                color: var(--text);
            }

            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2rem;
            }

            .hero {
                padding: 1.4rem 1.5rem;
                border: 1px solid var(--border);
                border-radius: 14px;
                background: linear-gradient(135deg, rgba(29, 139, 122, 0.12), rgba(255, 255, 255, 0.95));
                box-shadow: 0 8px 28px rgba(22, 48, 44, 0.06);
                margin-bottom: 1rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 2rem;
                letter-spacing: 0;
                color: var(--text);
            }

            .hero p {
                margin: 0.45rem 0 0;
                color: var(--muted);
                font-size: 1rem;
                max-width: 900px;
            }

            .metric-card, .surface {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1rem;
                box-shadow: 0 6px 18px rgba(22, 48, 44, 0.04);
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.85rem;
                margin-bottom: 0.25rem;
            }

            .metric-value {
                color: var(--text);
                font-size: 1.7rem;
                font-weight: 700;
                line-height: 1.1;
            }

            .metric-note {
                color: var(--muted);
                font-size: 0.8rem;
                margin-top: 0.25rem;
            }

            .risk-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: var(--panel-soft);
                color: var(--text);
                font-size: 0.9rem;
                font-weight: 600;
            }

            .risk-high {
                background: rgba(217, 69, 52, 0.10);
                color: #9a2d24;
            }

            .risk-medium {
                background: rgba(243, 156, 74, 0.14);
                color: #8e5316;
            }

            .risk-low {
                background: rgba(29, 139, 122, 0.12);
                color: #16685b;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                padding: 0.45rem 0.9rem;
                background: rgba(255, 255, 255, 0.8);
            }

            .stTabs [aria-selected="true"] {
                background: rgba(29, 139, 122, 0.13) !important;
            }

            footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    local_path = Path(DATA_PATH)
    if local_path.exists():
        return pd.read_csv(local_path, header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN])
    return pd.read_csv(DATA_URL, header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN])


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    X[INVALID_ZERO_COLUMNS] = X[INVALID_ZERO_COLUMNS].astype(float).replace(0, np.nan)
    medians = X.median(numeric_only=True).to_dict()
    X = X.fillna(medians)
    return X, y, medians


@st.cache_resource(show_spinner=False)
def train_model() -> dict[str, object]:
    raw = load_data()
    X, y, medians = clean_data(raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

    return {
        "model": model,
        "medians": medians,
        "metrics": metrics,
        "feature_importance": pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False),
        "data": raw,
    }


def sanitize_input(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    frame = df.copy()
    frame = frame[FEATURE_COLUMNS]
    frame[INVALID_ZERO_COLUMNS] = frame[INVALID_ZERO_COLUMNS].astype(float).replace(0, np.nan)
    for column, median in medians.items():
        frame[column] = frame[column].fillna(median)
    return frame.astype(float)


def prediction_label(probability: float) -> tuple[str, str]:
    if probability >= 0.7:
        return "High risk", "risk-high"
    if probability >= 0.4:
        return "Moderate risk", "risk-medium"
    return "Lower risk", "risk-low"


def metric_tile(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_probability(model: RandomForestClassifier, row: pd.DataFrame) -> float:
    return float(model.predict_proba(row)[0, 1])


def main() -> None:
    inject_styles()

    bundle = train_model()
    model = bundle["model"]
    medians = bundle["medians"]
    metrics = bundle["metrics"]
    data = bundle["data"]
    importance = bundle["feature_importance"]

    st.markdown(
        """
        <div class="hero">
            <h1>Diabetes Prediction App</h1>
            <p>
                A Streamlit dashboard for screening diabetes risk using a trained machine-learning model.
                Use the single-patient form for instant predictions or upload a CSV for batch scoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_mid, top_right = st.columns(3)
    with top_left:
        metric_tile("Model accuracy", f"{metrics['accuracy']:.1%}", "Hold-out evaluation on the Pima dataset")
    with top_mid:
        metric_tile("ROC AUC", f"{metrics['roc_auc']:.3f}", "Probability ranking quality")
    with top_right:
        metric_tile("Training rows", f"{len(data):,}", "Public dataset loaded at runtime")

    tab_predict, tab_batch, tab_model = st.tabs(["Single prediction", "Batch upload", "Model insights"])

    with tab_predict:
        left, right = st.columns([1.2, 0.8], gap="large")
        with left:
            st.subheader("Patient details")
            with st.form("single_prediction_form"):
                c1, c2 = st.columns(2)
                with c1:
                    pregnancies = st.slider("Pregnancies", 0, 20, 2)
                    glucose = st.slider("Glucose", 0, 250, 120)
                    blood_pressure = st.slider("Blood pressure", 0, 140, 72)
                    skin_thickness = st.slider("Skin thickness", 0, 100, 23)
                with c2:
                    insulin = st.slider("Insulin", 0, 850, 79)
                    bmi = st.slider("BMI", 0.0, 70.0, 32.0, 0.1)
                    pedigree = st.slider("Diabetes pedigree", 0.0, 3.0, 0.5, 0.01)
                    age = st.slider("Age", 1, 100, 33)
                submitted = st.form_submit_button("Predict risk", use_container_width=True)

            input_frame = pd.DataFrame(
                [{
                    "pregnancies": pregnancies,
                    "glucose": glucose,
                    "blood_pressure": blood_pressure,
                    "skin_thickness": skin_thickness,
                    "insulin": insulin,
                    "bmi": bmi,
                    "diabetes_pedigree": pedigree,
                    "age": age,
                }]
            )
            prepared_input = sanitize_input(input_frame, medians)

        with right:
            st.subheader("Prediction")
            if submitted:
                probability = safe_probability(model, prepared_input)
                label, css_class = prediction_label(probability)
                st.markdown(
                    f"""
                    <div class="risk-chip {css_class}">
                        {label} - {probability:.1%} estimated probability
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(max(probability, 0.0), 1.0))
                st.write(
                    "This is a screening estimate, not a diagnosis. It should be interpreted with clinical context and professional care."
                )
            else:
                st.info("Adjust the patient details, then press Predict risk to generate an estimate.")

            st.markdown(
                """
                <div class="surface">
                    <strong>What the model uses</strong><br/>
                    The model looks at common clinical measurements from the Pima diabetes dataset and learns from patterns in the training data.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_batch:
        st.subheader("Batch scoring")
        st.write("Upload a CSV with the same feature columns to score multiple rows at once.")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            missing_columns = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
            if missing_columns:
                st.error(f"Missing columns: {', '.join(missing_columns)}")
            else:
                batch_input = sanitize_input(batch_df, medians)
                batch_predictions = model.predict_proba(batch_input)[:, 1]
                batch_labels = np.where(batch_predictions >= 0.7, "High risk", np.where(batch_predictions >= 0.4, "Moderate risk", "Lower risk"))
                results = batch_df.copy()
                results["predicted_probability"] = batch_predictions
                results["predicted_label"] = batch_labels
                st.dataframe(results, use_container_width=True)
                buffer = BytesIO()
                results.to_csv(buffer, index=False)
                st.download_button(
                    "Download predictions",
                    data=buffer.getvalue(),
                    file_name="diabetes_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with tab_model:
        st.subheader("Model insights")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_tile("Precision", f"{metrics['precision']:.1%}", "Positive prediction quality")
        with m2:
            metric_tile("Recall", f"{metrics['recall']:.1%}", "Sensitivity on the hold-out set")
        with m3:
            metric_tile("F1 score", f"{metrics['f1']:.1%}", "Balance between precision and recall")
        with m4:
            metric_tile("Dataset rows", f"{len(data):,}", "Original training data")

        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        heatmap = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale=["#edf3f1", "#1d8b7a"],
            aspect="auto",
        )
        heatmap.update_layout(
            margin=dict(l=0, r=0, t=24, b=0),
            coloraxis_showscale=False,
            title="Confusion matrix",
        )

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.plotly_chart(heatmap, use_container_width=True)
        with right:
            importance_df = importance.reset_index()
            importance_df.columns = ["feature", "importance"]
            fig = px.bar(
                importance_df,
                x="importance",
                y="feature",
                orientation="h",
                labels={"feature": "Feature", "importance": "Importance"},
                title="Feature importance",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=24, b=0), yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Dataset preview", expanded=False):
            st.dataframe(data.head(10), use_container_width=True)


if __name__ == "__main__":
    main()


