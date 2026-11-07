from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RANDOM_STATE = 42
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def build_models() -> dict[str, Pipeline | DecisionTreeClassifier | GaussianNB]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": Pipeline(
            [("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(probability=True, random_state=RANDOM_STATE)),
            ]
        ),
    }


def load_binary_dataset() -> pd.DataFrame:
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
        "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
        "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
        "symmetry_worst", "fractal_dimension_worst",
    ]
    columns = ["id", "diagnosis", *feature_names]
    source_path = DATA_DIR / "wdbc_source" / "wdbc.data"
    data = pd.read_csv(source_path, header=None, names=columns)
    data = data.drop(columns="id")
    data["diagnosis"] = data["diagnosis"].map({"B": "benign", "M": "malignant"})
    return data.rename(columns={"diagnosis": "target"})


def load_multiclass_dataset() -> pd.DataFrame:
    dataset_path = DATA_DIR / "Dry_Bean_Dataset.arff"
    records, _ = arff.loadarff(dataset_path)
    data = pd.DataFrame(records)
    data["Class"] = data["Class"].str.decode("utf-8")
    return data.rename(columns={"Class": "target"})


def train_dataset(dataset_type: str, data: pd.DataFrame) -> dict[str, dict[str, float]]:
    target = data["target"]
    features = data.drop(columns="target")
    mode_dir = ROOT / "models" / dataset_type
    output_dir = DATA_DIR / dataset_type
    mode_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "full_dataset.csv", index=False)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    test_data = x_test.copy()
    test_data["target"] = y_test
    test_data.to_csv(output_dir / "test_data.csv", index=False)

    metrics: dict[str, dict[str, float]] = {}
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        class_labels = list(model.classes_)
        class_probabilities = model.predict_proba(x_test)
        if len(class_labels) == 2:
            positive_index = class_labels.index("malignant") if "malignant" in class_labels else 1
            auc = roc_auc_score(
                y_test == class_labels[positive_index],
                class_probabilities[:, positive_index],
            )
        else:
            auc = roc_auc_score(
                y_test,
                class_probabilities,
                labels=class_labels,
                multi_class="ovr",
                average="macro",
            )
        metrics[model_name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "auc": float(auc),
            "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, predictions)),
        }
        joblib.dump(model, mode_dir / MODEL_FILES[model_name])

    print(
        f"{dataset_type}: {len(data)} rows, {len(features.columns)} features, "
        f"{target.nunique()} classes"
    )
    print(f"Train rows: {len(x_train)} | Test rows: {len(x_test)}")
    print(pd.DataFrame(metrics).T.round(4).to_string())
    print(f"Saved models to: {mode_dir}")
    print(f"Saved test data to: {output_dir / 'test_data.csv'}")
    return metrics


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    all_metrics = {
        "binary": train_dataset("binary", load_binary_dataset()),
        "multiclass": train_dataset("multiclass", load_multiclass_dataset()),
    }
    with (ROOT / "evaluation_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(all_metrics, metrics_file, indent=2)


if __name__ == "__main__":
    main()