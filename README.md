# ML Prediction Studio

> A complete, production-ready end-to-end Machine Learning deployment project.
> Train a model on any CSV dataset, serve it with a FastAPI backend, and let clients upload any CSV to get predictions instantly — through a professional analytics dashboard.

---

## Project Structure

```
ml_deployment_project/
│
├── server/
│   ├── train.py                ← Train, compare & save the best model
│   ├── main.py                 ← FastAPI REST API (prediction server)
│   ├── __init__.py
│   └── models/                 ← Auto-created after training
│       ├── best_model.pkl          Saved best pipeline
│       ├── model_metadata.json     Model stats, feature names, target col
│       └── plots/
│           ├── model_comparison.png
│           └── confusion_matrix.png
│
├── client/
│   └── index.html              ← Web dashboard (zero dependencies, open directly)
│
├── requirements.txt
└── README.md
```

---

## Quick Start (3 steps)

### Step 1 — Install dependencies

```powershell
pip install -r requirements.txt
```

> **Requires Python 3.9 – 3.13.** All versions are pinned with `>=` so pip picks the right wheels automatically.

---

### Step 2 — Train a model

**Option A — Built-in demo (Iris dataset, no data needed):**
```powershell
python server/train.py --demo
```

**Option B — Your own CSV dataset:**
```powershell
python server/train.py --data path\to\your_dataset.csv --target your_label_column
```

**Option C — Generate a ready-made dataset and train:**
```powershell
# Breast Cancer dataset (binary, SVM/RF usually wins ~96-98% accuracy)
python -c "
from sklearn.datasets import load_breast_cancer
import pandas as pd
bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df['diagnosis'] = ['malignant' if t==0 else 'benign' for t in bc.target]
df.to_csv('breast_cancer.csv', index=False)
print('Done!', len(df), 'rows saved.')
"
python server/train.py --data breast_cancer.csv --target diagnosis
```

**Training flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--data` | Path to your CSV file | — |
| `--target` | Name of the label/target column | — |
| `--demo` | Use the built-in Iris dataset | False |
| `--test_size` | Fraction of data held out for testing | `0.2` |

**What training does:**
- Loads and preprocesses your CSV (auto-encodes categorical columns)
- Splits data into train / test sets (stratified)
- Trains and compares **7 algorithms** using 5-fold cross-validation
- Selects the best model by **F1-Macro score**
- Prints a full classification report
- Saves the best model pipeline to `server/models/best_model.pkl`
- Saves metadata (feature names, classes, target column) to `server/models/model_metadata.json`
- Generates two plots in `server/models/plots/`

---

### Step 3 — Start the API server

```powershell
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
Model loaded from server/models/best_model.pkl
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open.** Open a second terminal or just open the browser.

---

### Step 4 — Open the dashboard

Double-click `client/index.html` or open it in your browser.

- The status badge turns 🟢 green automatically when connected
- Model name, accuracy, F1 score, and feature list load from the API
- Upload any CSV → predictions + 6 live charts appear instantly

---

## 🌐 API Reference

**Base URL:** `http://localhost:8000`

### `GET /health`
Returns model status and performance metrics.

```json
{
  "status": "ready",
  "model_name": "SVM (RBF)",
  "test_accuracy": 0.9737,
  "test_f1_macro": 0.9727,
  "features": ["mean radius", "mean texture", "..."],
  "classes": ["benign", "malignant"],
  "target_col": "diagnosis",
  "n_features": 30
}
```

### `GET /docs`
Interactive Swagger UI — test all endpoints directly in the browser.

---

### `POST /predict`
Predict from a JSON body.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8}
    ]
  }'
```

---

### `POST /predict/csv`
Upload any CSV file — even if it still contains the target/label column.

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@breast_cancer.csv"
```

**The backend handles everything automatically:**
- Detects and removes the target/label column if present
- Drops any extra column not used during training
- Encodes categorical columns
- Fills any missing feature columns with `0.0`
- Returns predictions for every row

---

### Response format (both predict endpoints):

```json
{
  "model_name": "SVM (RBF)",
  "n_samples": 569,
  "elapsed_ms": 43.2,
  "columns_used": ["mean radius", "mean texture", "..."],
  "columns_dropped": ["diagnosis"],
  "predictions": [
    {
      "row": 1,
      "prediction": "malignant",
      "confidence": 0.9821,
      "class_probabilities": {
        "benign": 0.0179,
        "malignant": 0.9821
      }
    }
  ]
}
```

---

## Models Compared During Training

| # | Algorithm | Best when… |
|---|-----------|-----------|
| 1 | **Logistic Regression** | Data is linearly separable, large dataset |
| 2 | **SVM (RBF kernel)** | Small–medium dataset, high-dimensional, clean data |
| 3 | **K-Nearest Neighbors** | Dense, well-clustered data |
| 4 | **Decision Tree** | Simple decision boundaries, interpretability needed |
| 5 | **Random Forest** | Noisy data, many features, general-purpose |
| 6 | **Gradient Boosting** | Complex patterns, imbalanced classes |
| 7 | **XGBoost** | Large datasets, mixed feature types, competitions |

**Selection criterion:** Highest 5-fold cross-validated **F1-Macro** score on the training set.

The winner is saved automatically. You can see the full leaderboard in the terminal after training and in `server/models/plots/model_comparison.png`.

---

## Dashboard Features

The `client/index.html` file is a standalone web dashboard with **zero external dependencies to install** — just open it in any browser.

| Feature | Description |
|---------|-------------|
| **Live status badge** | Auto-connects to API on load, shows model name |
| **4 stat cards** | Model name, accuracy, F1-score, session prediction count |
| **Model info panel** | All expected feature names numbered, output classes color-coded |
| **CSV upload** | Drag-and-drop or click to browse |
| **Column report** | Instantly shows which columns matched, which are extra, which are missing — before sending to server |
| **Auto-strip notice** | Warns user if target column is detected and will be removed |
| { } **JSON input** | Paste raw JSON for quick single-row testing |
| **Pie chart** | Class distribution of all predictions |
| **Radar chart** | Average confidence per class (spider/polygon chart) |
| **Line chart** | Confidence trend across all samples |
| **Bar chart** | Per-sample confidence (green=high, yellow=medium, red=low) |
| **Horizontal bar** | Total prediction count per class |
| **Doughnut chart** | Confidence quality buckets (Low / Medium / High) |
| **Results table** | Full row-by-row results with colored class chips, confidence bars, and per-class probability mini-bars |
| **Error box** | Clear explanation if something goes wrong with tips to fix it |

---

## Preparing Your Own Dataset

The only requirement is a **CSV file** where:
- One column is the **label/target** (you pass its name with `--target`)
- All other columns are **features** (numeric or categorical — auto-handled)
- There are no hard limits on rows or columns

**Example:**
```
age,income,education,loan_approved
25,45000,bachelors,yes
34,72000,masters,yes
22,21000,high_school,no
```

Train: `python server/train.py --data loans.csv --target loan_approved`

When clients upload this CSV to the dashboard, the system will automatically remove `loan_approved` and predict it.

---

## Recommended Datasets to Try

| Dataset | Classes | Command to generate |
|---------|---------|-------------------|
| **Iris** (demo) | 3 | `python server/train.py --demo` |
| **Breast Cancer** | 2 | See Quick Start Step 2 Option C |
| **Wine Quality** | 3 | `from sklearn.datasets import load_wine` |
| **Diabetes** | 2 | `from sklearn.datasets import load_diabetes` |
| **Digits** | 10 | `from sklearn.datasets import load_digits` |

---

## Full Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  TERMINAL 1                                                     │
│                                                                 │
│  python server/train.py --data dataset.csv --target label  →    │
│  server/models/best_model.pkl                                   │
│                                                                 │
│  uvicorn server.main:app --reload  →  http://localhost:8000     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER                                                        │
│                                                                 │
│  Open client/index.html                                         │
│  → Status turns green                                           │
│  → Upload dataset.csv  (with or without label column)           │
│  → Column report shows: All 30 features matched                 │
│  → Click Run Prediction                                         │
│  → 569 rows predicted · 6 charts appear · table fills in        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: breast_cancer.csv` | Generate the CSV first — see Quick Start Step 2 |
| `pip install` fails on Python 3.13 | Use `>=` versions in requirements.txt (already done) |
| Status badge stays red | Make sure `uvicorn server.main:app --reload` is running in a terminal |
| CSV upload gives column error | Check the column report in the dashboard — it shows exactly what's missing |
| Low confidence scores (30–40%) | Normal for multi-class SVM — try binary datasets for sharper scores |
| Port 8000 already in use | Run `uvicorn server.main:app --reload --port 8001` and update the API URL in the dashboard |

---

## Production Deployment Notes

For production, tighten CORS in `server/main.py`:
```python
allow_origins=["https://yourdomain.com"]
```

Use `gunicorn` with multiple workers:
```bash
gunicorn server.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.5.0 | Core ML algorithms |
| `xgboost` | ≥2.1.0 | XGBoost classifier |
| `pandas` | ≥2.2.0 | Data loading & processing |
| `numpy` | ≥1.26.0 | Numerical arrays |
| `joblib` | ≥1.4.0 | Model serialization |
| `matplotlib` | ≥3.9.0 | Training plots |
| `seaborn` | ≥0.13.2 | Confusion matrix heatmap |
| `fastapi` | ≥0.111.0 | REST API framework |
| `uvicorn` | ≥0.30.0 | ASGI server |
| `python-multipart` | ≥0.0.9 | File upload support |
| `pydantic` | ≥2.7.0 | Request/response validation |
| `loguru` | ≥0.7.2 | Structured logging |

---
