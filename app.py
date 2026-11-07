from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_models(classification_type: str) -> dict[str, object]:
    model_dir = ROOT / "models" / classification_type
    return {
        name: joblib.load(model_dir / filename)
        for name, filename in MODEL_FILES.items()
    }


def evaluate_model(model: object, data: pd.DataFrame, feature_names: list[str]) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    x_test = data[feature_names]
    y_test = data["target"]
    predictions = model.predict(x_test)
    classes = list(model.classes_)
    probabilities = model.predict_proba(x_test)
    present_labels = set(y_test)
    if len(classes) == 2:
        positive_label = "malignant" if "malignant" in classes else classes[-1]
        positive_index = classes.index(positive_label)
        auc = (
            roc_auc_score(y_test == positive_label, probabilities[:, positive_index])
            if len(present_labels) == 2
            else float("nan")
        )
        auc_label = "AUC"
    else:
        auc = (
            roc_auc_score(
                y_test,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
            if present_labels == set(classes)
            else float("nan")
        )
        auc_label = "AUC (macro OVR)"
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        auc_label: auc,
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    matrix = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=classes),
        index=pd.Index(classes, name="Actual"),
        columns=pd.Index(classes, name="Predicted"),
    )
    report = pd.DataFrame(
        classification_report(y_test, predictions, labels=classes, output_dict=True, zero_division=0)
    ).T
    return metrics, matrix, report


st.set_page_config(page_title="Classification Model Lab", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --bits-maroon: #8b1e3f;
        --bits-maroon-dark: #68152f;
        --bits-gold: #c69a35;
        --ink: #22262b;
        --muted-ink: #4b5560;
        --paper: #f8f9fa;
        --panel: #ffffff;
        --line: #d7dce1;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--paper) !important;
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f2f4 !important;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--ink) !important;
    }
    [data-testid="stHeader"] { background: rgba(248, 249, 250, 0.96); }
    .block-container { max-width: 1180px; padding-top: 3.5rem; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: var(--bits-maroon-dark) !important;
        font-family: Georgia, "Times New Roman", serif;
        letter-spacing: 0;
    }
    .stApp p, .stApp label, .stApp li,
    [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] {
        color: var(--ink) !important;
    }
    [data-testid="stCaptionContainer"] { color: var(--muted-ink) !important; }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h4 {
        color: var(--bits-maroon-dark) !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricValue"] { color: var(--bits-maroon-dark) !important; }
    [data-testid="stMetric"] {
        background: var(--panel) !important;
        border: 1px solid var(--line);
        border-top: 3px solid var(--bits-gold);
        padding: 14px 14px 12px;
        border-radius: 6px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 6px;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] * { color: var(--ink) !important; }
    [data-testid="stFileUploader"] section {
        background: var(--panel);
        border-color: #aab2bb;
    }
    .bits-masthead {
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid var(--line);
        padding: 0 0 13px;
        margin-bottom: 18px;
    }
    .bits-wordmark {
        color: var(--bits-maroon) !important;
        font-size: 13px;
        font-weight: 800;
        line-height: 1;
        padding-right: 12px;
        border-right: 3px solid var(--bits-gold);
    }
    .bits-program {
        color: var(--muted-ink) !important;
        font-size: 12px;
        font-weight: 600;
    }
    @media (max-width: 700px) {
        .block-container { padding-top: 3rem; }
        [data-testid="stMetric"] { padding: 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="bits-masthead"><span class="bits-wordmark">BITS PILANI</span>'
    '<span class="bits-program">M.TECH AIML · MACHINE LEARNING</span></div>',
    unsafe_allow_html=True,
)
st.title("Classification Model Lab")
classification_label = st.radio(
    "Classification type",
    ["Binary", "Multiclass"],
    horizontal=True,
    key="classification_type",
)
classification_type = classification_label.lower()
dataset_details = {
    "binary": "UCI Breast Cancer Wisconsin Diagnostic | 30 features | 2 classes",
    "multiclass": "UCI Dry Bean | 16 image-derived features | 7 classes",
}
st.caption(dataset_details[classification_type])
models = load_models(classification_type)
first_model = next(iter(models.values()))
feature_names = list(first_model.feature_names_in_)
class_names = list(first_model.classes_)

with st.sidebar:
    st.header(f"{classification_label} test data")
    test_data_source = st.radio(
        "Test data source",
        ["Bundled test split", "Upload CSV"],
        key=f"test_data_source_{classification_type}",
    )
    uploaded_file = None
    if test_data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a labeled test CSV",
            type=["csv"],
            key=f"test_csv_{classification_type}",
        )
    selected_model = st.selectbox("Model details", list(models))
    st.caption(f"Include all {len(feature_names)} feature columns and a target column.")
    st.caption(f"Target labels: {', '.join(class_names)}")

try:
    if test_data_source == "Bundled test split":
        sample_path = DATA_DIR / classification_type / "test_data.csv"
        test_data = pd.read_csv(sample_path)
        st.info(f"Showing the bundled {classification_label.lower()} test split.")
    elif uploaded_file is None:
        st.info("Upload a labeled CSV to evaluate the selected classification type.")
        st.stop()
    else:
        test_data = pd.read_csv(uploaded_file)

    required_columns = feature_names + ["target"]
    missing_columns = [column for column in required_columns if column not in test_data.columns]
    if missing_columns:
        st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
        st.stop()
    if test_data.empty:
        st.error("The uploaded CSV has no rows to evaluate.")
        st.stop()
    if test_data[required_columns].isna().any().any():
        st.error("Required feature and target columns cannot contain blank values.")
        st.stop()
    for feature_name in feature_names:
        test_data[feature_name] = pd.to_numeric(test_data[feature_name], errors="coerce")
    if test_data[feature_names].isna().any().any():
        st.error("Feature columns must contain numeric values only.")
        st.stop()

    observed_labels = set(test_data["target"].astype(str))
    expected_labels = set(class_names)
    unexpected_labels = sorted(observed_labels - expected_labels)
    if unexpected_labels:
        actual_type = "Binary" if len(observed_labels) == 2 else "Multiclass"
        st.error(
            f"This CSV contains {actual_type.lower()} target labels that do not match "
            f"the selected {classification_label.lower()} model suite: "
            f"{', '.join(unexpected_labels)}. Select the matching classification type "
            "and use its matching feature columns."
        )
        st.stop()
    if classification_type == "binary" and len(observed_labels) != 2:
        st.error("Binary mode requires exactly two target classes in the uploaded CSV.")
        st.stop()
    if classification_type == "multiclass" and len(observed_labels) < 3:
        st.error("Multiclass mode requires at least three target classes in the uploaded CSV.")
        st.stop()
    if len(observed_labels) < len(expected_labels):
        st.warning("AUC is unavailable because this CSV does not contain every model class.")

    all_results: dict[str, dict[str, float]] = {}
    selected_details = None
    for model_name, model in models.items():
        metrics, matrix, report = evaluate_model(model, test_data, feature_names)
        all_results[model_name] = metrics
        if model_name == selected_model:
            selected_details = (metrics, matrix, report)

    result_table = pd.DataFrame(all_results).T
    st.subheader("Model comparison")
    st.dataframe(result_table.style.format("{:.3f}", na_rep="N/A"), use_container_width=True)

    st.subheader(selected_model)
    selected_metrics, matrix, report = selected_details
    metric_columns = st.columns(6)
    for column, (metric_name, value) in zip(metric_columns, selected_metrics.items()):
        column.metric(metric_name, f"{value:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        st.dataframe(matrix, use_container_width=True)
    with right:
        st.markdown("#### Classification report")
        st.dataframe(report.style.format("{:.3f}"), use_container_width=True)
    st.caption(f"Evaluated {len(test_data)} rows. No model was trained on uploaded data.")
except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"Could not read the CSV or saved model files: {error}")