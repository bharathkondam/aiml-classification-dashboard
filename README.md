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

## GitHub Repository Link

**Repository:** [Open the project repository](https://github.com/bharathkondam/aiml-classification-dashboard).

To create the repository, open [github.com/new](https://github.com/new), create a public repository named `aiml-classification-dashboard`, and leave the options to add a README, `.gitignore`, or license unchecked. From this project folder, run these commands after replacing `YOUR-USERNAME`:

```powershell
git init
git add app.py train_models.py requirements.txt README.md SUBMISSION_CHECKLIST.md .gitignore .streamlit/config.toml evaluation_metrics.json data/ models/
git commit -m "Add classification model lab"
git branch -M main
git remote add origin https://github.com/bharathkondam/aiml-classification-dashboard.git
git push -u origin main
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

### Performance Observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Performs strongly on both tasks; it is competitive on binary and multiclass test splits. |
| Decision Tree | Lower AUC than other models on both tasks; remains the most interpretable tree-based baseline. |
| K-Nearest Neighbors | Strong results after feature scaling, especially on the binary test split. |
| Gaussian Naive Bayes | Performs better on the binary split than multiclass Dry Bean, reflecting dataset-dependent feature distributions. |
| Support Vector Machine | Best binary accuracy (0.9737) and the top multiclass accuracy (0.9221) on the generated splits. |
| **Overall winner** | **Support Vector Machine** leads by accuracy on both datasets. Results depend on the selected dataset and fixed split. |

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

**Live app:** Add the URL provided by Streamlit Community Cloud after deployment, for example `https://YOUR-APP-NAME.streamlit.app`.

1. Push this repository to GitHub using the commands above.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, then select this repository and the `main` branch.
4. Set the app file path to `app.py`, then deploy.
5. Open the deployed app and copy its public URL into the **Live app** line above and into the assignment submission PDF.

## Project Files

Each mode has a separate test CSV and model directory so the evaluator can tell which saved classifiers match each test set. This provides the assignment's required test-data CSV and saved-model artifacts for both classification types.

```text
app.py
train_models.py
requirements.txt
README.md
SUBMISSION_CHECKLIST.md
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

Before submitting, work through [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md). It distinguishes artifacts verified in this folder from the external links and screenshot that still need to be created.

For the assignment submission PDF, include the final GitHub link, deployed app link, this README content, and the required screenshot of the assignment running in BITS Virtual Lab.