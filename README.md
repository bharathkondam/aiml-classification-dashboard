# Classification Model Lab

## Problem Statement

This project supports both binary and multiclass classification. Choose the classification type in the Streamlit app, then use its bundled test split or upload a labeled CSV with the matching feature schema. Each type uses its own five trained models and evaluation results.

## Dataset Description

- **Binary dataset:** [UCI Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), 569 rows, 30 numeric features, two target classes (`benign`, `malignant`). The ID field is dropped before training.
- **Multiclass dataset:** [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), 13,611 rows, 16 numeric features, seven target classes (`BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER`, `SIRA`).
- **Citations:** W. Wolberg, O. Mangasarian, N. Street, and W. Street. Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Dry Bean [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C50S4B](https://doi.org/10.24432/C50S4B).
- **Split:** Both use an 80/20 stratified split with random seed 42.
- **Test rows:** 114 binary and 2,723 multiclass examples.
- **License:** Both UCI datasets are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Files:** Original UCI sources are bundled in `data/wdbc_source/` and `data/Dry_Bean_Dataset.arff`. Training generates full CSVs and held-out test CSVs under `data/binary/` and `data/multiclass/`.

The UCI sources are bundled with the project so training and deployment do not require Kaggle credentials or a dataset download at runtime. No API token is needed.

## Project Links

| Link type | Purpose | Link |
|---|---|---|
| **SOURCE REPOSITORY** | Source code, datasets, trained models, and documentation | [Open GitHub repository](https://github.com/bharathkondam/aiml-classification-dashboard) |
| **LIVE STREAMLIT APP** | Interactive Binary/Multiclass classification dashboard | [Launch Classification Model Lab](https://aiml-classification-dashboard.streamlit.app/) |

From this project folder, use these commands to publish future changes:

```powershell
git add app.py train_models.py requirements.txt README.md .gitignore .streamlit/config.toml evaluation_metrics.json data/ models/
git commit -m "Update classification model lab"
git push origin main
```

If `origin` is already configured, update it with `git remote set-url origin https://github.com/bharathkondam/aiml-classification-dashboard.git` instead of adding it again. The repository should include the source files, `requirements.txt`, this README, both UCI source datasets, the generated test data and metrics, and both five-model suites.

## Models Used

Scores use the fixed held-out split for each dataset. Precision, recall, and F1 are weighted averages. Binary AUC uses malignant as the positive class. Multiclass AUC is macro one-vs-rest. Scores are rounded to four decimal places.

### Binary: Breast Cancer Wisconsin Diagnostic

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9652 | 0.9649 | 0.9647 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| K-Nearest Neighbors | 0.9561 | 0.9823 | 0.9569 | 0.9561 | 0.9558 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 0.9440 | 0.9386 | 0.9372 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 0.9747 | 0.9737 | 0.9735 | 0.9442 |

### Multiclass: Dry Bean

| ML Model | Accuracy | Macro OVR AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8917 | 0.9448 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| K-Nearest Neighbors | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9672 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9949 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

### Model Performance Details

#### Binary Dataset Observations

| ML Model | Metric evidence | Performance interpretation |
|---|---|---|
| Logistic Regression | Accuracy 0.9649; AUC 0.9960; MCC 0.9245 | Strong separation of benign and malignant cases. Its highest AUC suggests the probability scores rank the two classes especially well, while accuracy indicates reliable predictions at the model's decision threshold. It is a competitive, relatively simple linear baseline. |
| Decision Tree | Accuracy 0.9298; AUC 0.9246; MCC 0.8492 | Lowest AUC and MCC among the binary models, so it is less consistent than the other approaches on this split. Its tree structure can be easier to inspect, but interpretability does not translate into the highest predictive performance here. |
| K-Nearest Neighbors | Accuracy 0.9561; AUC 0.9823; MCC 0.9058 | Performs well after feature scaling, with high accuracy and strong class agreement. Its AUC is below Logistic Regression, Gaussian Naive Bayes, and SVM; results can depend on the choice of neighbor count and distance behavior. |
| Gaussian Naive Bayes | Accuracy 0.9386; AUC 0.9934; MCC 0.8715 | The high AUC indicates strong ranking of cases, while the lower accuracy and MCC show that the final class assignments are less strong than its ranking suggests. The gap is a reason to consider calibration or decision-threshold analysis if the task requires a different error trade-off. |
| Support Vector Machine | Accuracy 0.9737; AUC 0.9947; F1 0.9735; MCC 0.9442 | Best binary accuracy, weighted F1, and MCC. It combines very strong class discrimination with the strongest overall agreement on this test split. Logistic Regression has a slightly higher AUC (0.9960), so SVM does not lead every metric. |

#### Multiclass Dataset Observations

| ML Model | Metric evidence | Performance interpretation |
|---|---|---|
| Logistic Regression | Accuracy 0.9214; macro OVR AUC 0.9948; MCC 0.9050 | Strong overall multi-class results. Its high macro OVR AUC indicates excellent average class ranking across the seven varieties, and its accuracy and MCC show that this also translates into strong hard-label predictions. |
| Decision Tree | Accuracy 0.8917; macro OVR AUC 0.9448; MCC 0.8691 | Lowest accuracy and macro OVR AUC among the multiclass models. The weaker average one-vs-rest ranking suggests a greater challenge separating several varieties; its rule-based structure may still be useful when inspectability is important. |
| K-Nearest Neighbors | Accuracy 0.9166; macro OVR AUC 0.9833; MCC 0.8992 | Competitive accuracy and MCC after scaling the 16 features. The AUC trails the linear and SVM models, suggesting slightly weaker probability ranking across classes on this split. |
| Gaussian Naive Bayes | Accuracy 0.7639; macro OVR AUC 0.9672; MCC 0.7154 | Lowest accuracy, weighted F1, and MCC, despite a substantially higher AUC. This indicates class-probability ranking is better than its final label decisions suggest, but its predictions are less reliable overall than the other models for this dataset. |
| Support Vector Machine | Accuracy 0.9221; macro OVR AUC 0.9949; weighted F1 0.9222; MCC 0.9059 | Best multiclass accuracy, weighted F1, MCC, and macro OVR AUC on this split. It is the strongest aggregate performer among the five evaluated models, although its accuracy advantage over Logistic Regression is small. |

#### Cross-Dataset Summary

| ML Model | Binary accuracy / AUC | Multiclass accuracy / macro OVR AUC | Summary |
|---|---:|---:|---|
| Logistic Regression | 0.9649 / 0.9960 | 0.9214 / 0.9948 | High-ranking performance on both tasks; the top binary AUC and second-highest multiclass accuracy. |
| Decision Tree | 0.9298 / 0.9246 | 0.8917 / 0.9448 | Lowest AUC in both comparisons, but offers a comparatively interpretable tree structure. |
| K-Nearest Neighbors | 0.9561 / 0.9823 | 0.9166 / 0.9833 | Strong accuracy on both datasets after feature scaling; AUC trails the strongest models. |
| Gaussian Naive Bayes | 0.9386 / 0.9934 | 0.7639 / 0.9672 | Much stronger accuracy on the binary dataset than the seven-class task; performance is dataset-dependent. |
| Support Vector Machine | 0.9737 / 0.9947 | 0.9221 / 0.9949 | Highest accuracy on both tasks and highest multiclass aggregate metrics; strongest overall choice for these fixed splits. |

**Overall observations:** Support Vector Machine is the accuracy leader on both held-out datasets and has the highest multiclass scores across the reported metrics. Logistic Regression attains the best binary AUC, showing that the winner depends on which metric matters. These results are from one stratified split per dataset; they should not be treated as a guarantee of performance on new data. No per-class precision/recall values are summarized here; use the app's classification report to inspect those for the selected test CSV.

## Run Locally

Use Python 3.11 or newer. From the project folder, run the following in PowerShell on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. The app opens at `http://localhost:8501`.

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

`train_models.py` reads both bundled UCI datasets and recreates the dataset CSVs, stratified test CSVs, both model suites, and combined evaluation metrics. The included model files and test CSVs can also be used directly by running `streamlit run app.py` after installing the requirements.

## Use the App

1. Run the local commands above or open the deployed app link below.
2. Select **Binary** or **Multiclass** at the top of the app.
3. In the sidebar, choose the bundled test split or **Upload CSV**.
4. For uploads, include the selected dataset's feature columns plus a `target` column. The app validates the feature schema and target classes against the selected model suite.
5. Review the comparison table and select a model to inspect its six metrics, confusion matrix, and classification report.

The app loads the saved models for the selected type and does not train on uploaded test data. Binary uploads must have exactly the two model target labels; multiclass uploads must use the matching Dry Bean feature columns and at least three known target labels.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub using the commands above.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, then select this repository and the `main` branch.
4. Set the app file path to `app.py`, then deploy.
5. Open the live app from the **Project Links** table above and confirm the deployed app loads before submitting.

## Project Files

Each mode has a separate test CSV and model directory so the evaluator can tell which saved classifiers match each test set. This provides the assignment's required test-data CSV and saved-model artifacts for both classification types.

```text
app.py
train_models.py
requirements.txt
README.md
.streamlit/config.toml
data/
	wdbc_source/
		wdbc.data
		wdbc.names
	Dry_Bean_Dataset.arff
	binary/
		full_dataset.csv
		test_data.csv
	multiclass/
		full_dataset.csv
		test_data.csv
models/
	binary/                 Five saved binary model pipelines
	multiclass/             Five saved multiclass model pipelines
evaluation_metrics.json
```

For the assignment submission PDF, include the final GitHub link, deployed app link, this README content, and the required screenshot of the assignment running in BITS Virtual Lab.