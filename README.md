# Predictive Maintenance: Machine Failure Prediction

A machine-learning project that predicts machine failures from operating-condition measurements using the **AI4I 2020 Predictive Maintenance Dataset**.

## Project overview

The notebook:
- Loads and explores the AI4I 2020 dataset.
- Removes identifiers and failure-mode indicator columns to reduce target leakage.
- Preprocesses numeric and categorical features with a scikit-learn pipeline.
- Compares Logistic Regression, Random Forest, and Histogram Gradient Boosting.
- Uses stratified 5-fold cross-validation and F1-score for model selection.
- Tunes model hyperparameters with `GridSearchCV`.
- Evaluates tuned models on a held-out test set.
- Saves the selected fitted pipeline to `artifacts/best_model.joblib`.

The target is `Machine failure`. Since failures are relatively rare in this dataset, the notebook reports precision, recall, F1, and ROC AUC in addition to accuracy.

## Repository structure

```text
.
├── data/
│   └── .gitkeep
├── artifacts/
│   └── .gitkeep
├── machine_failure_prediction.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

`ai4i2020.csv`, trained model artifacts, and generated plots are intentionally not committed by default.

## Dataset

Download the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Place the CSV file at:

```text
data/ai4i2020.csv
```

Do not rename the dataset columns. The notebook expects the original column names, including `Machine failure`, `UDI`, `Product ID`, and the failure-mode columns.

## Setup

Python 3.10 or newer is recommended.

```bash
git clone <your-repository-url>
cd <your-repository-name>

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

Launch Jupyter:

```bash
jupyter lab
```

Open `machine_failure_prediction.ipynb`, then run the cells from top to bottom.

The notebook writes:
- `artifacts/best_model.joblib` — fitted preprocessing and prediction pipeline.
- `confusion_matrix.png` — final model confusion matrix.

## Prediction example

After running the notebook, use the saved model with a pandas DataFrame containing these features:

| Feature | Description | Example |
|---|---|---:|
| `Type` | Product quality/type (`L`, `M`, or `H`) | `M` |
| `Air_temperature` | Air temperature in kelvin | `300.5` |
| `Process_temperature` | Process temperature in kelvin | `310.2` |
| `Rotational_speed` | Rotational speed in rpm | `1500` |
| `Torque` | Torque in Nm | `40.0` |
| `Tool_wear` | Tool wear in minutes | `100` |

The output is a predicted class and an estimated failure probability. A probability is a model estimate, not a guarantee that a failure will occur.

## Methodology notes

- The data is split with stratification to preserve the target-class proportions.
- `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded because they encode failure-mode information that can leak the target.
- `UDI` and `Product ID` are excluded as identifiers.
- Scaling and one-hot encoding are performed inside the pipeline, so preprocessing is fitted within each cross-validation fold.
- F1 is used for model selection because accuracy alone can be misleading with an imbalanced target.

## Limitations

- Model performance depends on the dataset and the chosen train/test split.
- The notebook does not establish that the model is suitable for safety-critical or production use.
- Threshold selection, probability calibration, temporal validation, monitoring, and real-world deployment are not covered.

## License

Add a license before redistributing this repository. Check the dataset's terms separately before redistributing the CSV.
