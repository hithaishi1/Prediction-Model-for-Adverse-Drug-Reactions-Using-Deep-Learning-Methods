# Prediction Model for Adverse Drug Reactions Using Deep Learning Methods

## Team Members

- Elizabeth Coquillette
- Hithaishi Reddy
- Ishan Chotalia

## Problem Statement

Drugs often have a list of side effects, and sometimes they include the prevalence of those side effects occurring in the general population. However, each patient is different, and patients are typically not given adjusted probabilities of different side effects based on their personal profile. This project develops a model that predicts the likelihood of an adverse drug reaction (ADR) for a specific patient based on their clinical profile. By providing patient-specific predictions, this tool can help clinicians make more informed prescribing decisions, reduce avoidable adverse reactions, and improve personalized healthcare outcomes.

## Objectives

- Clean, preprocess, and analyze the MIMIC-IV dataset.
- Explore and implement machine learning and deep learning methods on a large clinical dataset.
- Provide useful insights about drug side effects in the form of a personalized predictor model.
- Deploy an interactive prediction dashboard for clinical use.

---

## Dataset Information

### Primary Dataset
**MIMIC-IV Clinical Database (v2.2)**
- Source: PhysioNet
- Description: A large, de-identified electronic health record dataset containing longitudinal clinical data from patients admitted to Beth Israel Deaconess Medical Center.
- Data types include:
  - Patient demographics
  - Hospital admissions
  - Medication prescriptions
  - ICD-9 / ICD-10 diagnosis codes
  - Laboratory measurements (creatinine, ALT, AST)
  - Timestamps for all clinical events

**Dataset Link:** https://physionet.org/content/mimiciv/2.2/

---

## Data Access and Usage Policy

The MIMIC-IV dataset is a **restricted-access dataset** governed by a PhysioNet data use agreement. The dataset cannot be uploaded to this repository and must be downloaded separately following the steps below.

---

## How to Obtain the Dataset

### Step 1: Create a PhysioNet Account
Register at https://physionet.org/

### Step 2: Complete Required Training
Complete the **CITI "Data or Specimens Only Research"** course:
https://physionet.org/about/citi-course/

Upload your completion certificate to your PhysioNet profile.

### Step 3: Request Access to MIMIC-IV
Request access to **MIMIC-IV Clinical Database (v2.2)**. Approval typically takes 1–2 business days.

### Step 4: Download the Data

We recommend downloading using a Linux/macOS terminal or WSL:

```bash
wget -r -N -c -np --user=YOUR_PHYSIONET_USERNAME --ask-password \
  https://physionet.org/files/mimiciv/2.2/hosp/

wget -r -N -c -np --user=YOUR_PHYSIONET_USERNAME --ask-password \
  https://physionet.org/files/mimiciv/2.2/icu/
```

Place the downloaded files at:
- `data/hosp/` — hospital module CSVs (`patients.csv.gz`, `prescriptions.csv.gz`, `diagnoses_icd.csv.gz`, `admissions.csv.gz`, `labevents.csv.gz`)
- `data/icu/` — ICU module CSVs (`icustays.csv.gz`)

---

## Project Layout

```
.
├── data/
│   ├── hosp/          # Raw MIMIC-IV hospital CSVs (not tracked)
│   └── icu/           # Raw MIMIC-IV ICU CSVs (not tracked)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_adr_labeling.ipynb      # Generates notebooks/adr_labels.csv
│   ├── 03_baseline_models.ipynb   # Trains LR, RF, Gradient Boosting, XGBoost
│   └── 05_model_evaluation.ipynb  # Cross-model evaluation summary
├── src/
│   ├── models.py          # MLP, ResNet, Attention, EmbeddingMLP, DeepEnsemble
│   ├── train.py           # Deep learning training loop, early stopping, checkpointing
│   ├── train_EMLP.py      # EmbeddingMLP training script
│   ├── evaluate.py        # Evaluation, metrics, ROC/PR curves, temperature scaling
│   └── preprocessing.py   # Feature engineering + train/val/test split generation
├── tests/
│   └── test_pipeline.py   # pytest unit and integration tests
├── models/                # Saved model checkpoints (not tracked)
├── results/               # Evaluation plots and metrics (not tracked)
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── results_summary.csv    # Aggregated model metrics used by the dashboard
```

---

## Setup

**Requirements:** Python 3.11+, pip

```bash
git clone https://github.com/hithaishi1/Prediction-Model-for-Adverse-Drug-Reactions-Using-Deep-Learning-Methods.git
cd Prediction-Model-for-Adverse-Drug-Reactions-Using-Deep-Learning-Methods
pip install -r requirements.txt
```

---

## Reproducibility

All random seeds are fixed at **42** throughout the pipeline:

- `preprocessing.py`: `RANDOM_STATE = 42` (train/val/test split)
- `train.py`: `set_global_seed(42)` (NumPy, PyTorch, CUDA seeds)
- Baseline ML models: `random_state=42` passed to all scikit-learn estimators

All file paths are relative to the repository root and require no machine-specific edits.

---

## Run Pipeline

### 1. Generate ADR labels

Open and run all cells in `notebooks/02_adr_labeling.ipynb`. This produces `notebooks/adr_labels.csv`.

### 2. Preprocess data

```bash
python src/preprocessing.py
```

Outputs train/val/test CSVs to `processed_data_min1000_expanded/` (and other dataset variants).

### 3. Train deep learning models

```bash
python src/train.py
```

Trains MLP, ResNet, and Attention models. Saves best checkpoints to `models/`.

### 4. Train EmbeddingMLP

```bash
python src/train_EMLP.py
```

### 5. Evaluate deep learning models

```bash
# Evaluate a single model
python src/evaluate.py --model mlp --dataset min1000_expanded

# Evaluate all standard models
python src/evaluate.py --model all --dataset min1000_expanded
```

### 6. Train and evaluate baseline ML models

Open and run `notebooks/03_baseline_models.ipynb`.

### 7. Run the test suite

```bash
pytest tests/test_pipeline.py -v
```

All 19 tests should pass.

### 8. Launch the interactive dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The dashboard provides:
- **ADR Risk Prediction** — enter patient and prescription details to get a risk score
- **Metric Comparison** — compare AUROC/AUPRC across all models and feature sets
- **ROC / PR Curves** — visualize model discrimination performance

---

## Model Selection

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | ML baseline | Interpretable linear baseline |
| Random Forest | ML | Best leak-free AUROC (0.8815 on 19-feature set) |
| Gradient Boosting | ML | Strong tabular baseline |
| XGBoost | ML | High-performance gradient boosting |
| MLP | Deep Learning | 4-layer network with BatchNorm and Dropout |
| ResNet | Deep Learning | Residual connections for gradient flow |
| Attention | Deep Learning | Self-attention over feature representations |
| EmbeddingMLP | Deep Learning | Learned embeddings for drug/route/dose-unit |
| Deep Ensemble | Deep Learning | Weighted average of MLP + ResNet + Attention |

Primary evaluation metric: **AUROC** and **AUPRC** (preferred over accuracy due to ~17% class imbalance).

---

## Feature Sets

Three feature configurations were evaluated to assess the contribution of expanded clinical data and to identify data leakage:

| Features | Count | Notes |
|---|---|---|
| Original | 12 | Demographic + drug features only |
| Expanded (leak-free) | 19 | + lab values, comorbidity flags, polypharmacy |
| Expanded (with leaky features) | 21 | + `prior_adr`, `icu_stay` — inflates metrics, excluded from primary results |

---

## Assumptions and Limitations

- ADR labeling relies on ICD-9/ICD-10 diagnosis codes; not all ADRs may be coded.
- The repository does not include raw data (MIMIC-IV is restricted-access).
- The pipeline focuses on structured tabular features; clinical notes are not used.
- Some variation in deep learning results across hardware is possible despite fixed seeds.
