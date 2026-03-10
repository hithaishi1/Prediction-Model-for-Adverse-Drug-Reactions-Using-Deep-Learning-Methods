# Prediction Model for Adverse Drug Reactions Using Deep Learning Methods

## Team Members

- Elizabeth Coquillette
- Hithaishi Reddy
- Ishan Chotalia

Problem Statement

Drugs often have a list of side effects, and sometimes they include the prevalence of those side effects occurring in the general population. However, each patient is different, and patients are typically not given adjusted probabilities of different side effects based on their personal profile. This project will develop a model that predicts the likelihood of an adverse drug reaction for a specific patient based on other information about them. By providing patient-specific predictions, this tool can help clinicians make more informed prescribing decisions, reduce avoidable adverse reactions and improve personalized healthcare outcomes.

Objectives

Clean, preprocess, and analyze the FAERS dataset from the FDA and MIMIC-IV dataset.
Explore and implement machine learning and deep learning methods using a large dataset.
Provide useful insights about drug side effects, in the form of a personalized predictor model.

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
  - Laboratory measurements
  - Timestamps for all clinical events

The dataset enables patient-level, time-aware modeling of medication exposure and subsequent clinical outcomes, making it well suited for adverse drug reaction prediction.

**Dataset Link:**  
https://physionet.org/content/mimiciv/2.2/

---

## Data Access and Usage Policy

The MIMIC-IV dataset is a **restricted-access dataset** governed by a PhysioNet data use agreement.  
As a result:

- ❌ The dataset **cannot be redistributed**
- ❌ The dataset **is NOT uploaded to this GitHub repository**
- ✅ All code and analysis scripts are provided for authorized users to run locally after downloading the data directly from PhysioNet

This repository contains **only code, documentation, and instructions**.

---

## How to Obtain the Dataset

### Step 1: Create a PhysioNet Account
Register for a free account at:
https://physionet.org/

---

### Step 2: Complete Required Training
Complete the **CITI “Data or Specimens Only Research”** course as required by PhysioNet.

Instructions:
https://physionet.org/about/citi-course/

Upload your completion certificate to your PhysioNet profile.

---

### Step 3: Request Access to MIMIC-IV
Request access to:
**MIMIC-IV Clinical Database (v2.2)**

Approval typically takes 1–2 business days.

---

### Step 4: Download the Data (Recommended Method)

We recommend downloading the data using **WSL (Windows Subsystem for Linux)** or a Linux/macOS terminal.

Example command (downloads hospital tables only):

```bash
wget -r -N -c -np --user=YOUR_PHYSIONET_USERNAME --ask-password \
https://physionet.org/files/mimiciv/2.2/hosp/
```

## Project Layout

- `data/hosp/` (not tracked): raw MIMIC-IV hospital CSVs (`patients.csv.gz`, `prescriptions.csv.gz`, `diagnoses_icd.csv.gz`, etc.)
- `notebooks/`: EDA, ADR labeling, and baseline ML notebooks
- `src/preprocessing.py`: tabular feature engineering + train/val/test split generation
- `src/train.py`: deep learning training (MLP/ResNet/Attention)
- `src/evaluate.py`: model evaluation and plotting
- `processed_data/`: generated train/val/test files and preprocessing artifacts
- `models/`: saved model checkpoints and training histories
- `results/`: generated evaluation plots and metrics summaries

## Methods and Models

### Data Processing Workflow

1. Explore the raw data in `notebooks/01_data_exploration.ipynb`.
2. Generate ADR labels in `notebooks/02_adr_labeling.ipynb`.
3. Run `src/preprocessing.py` to merge the source tables, clean missing values, engineer features, encode variables, scale numeric columns, and create train/validation/test splits.
4. Train and compare models in notebooks `03` and `04`.
5. Summarize cross-model evaluation in notebook `05`.

### Model Selection Rationale

- Logistic Regression provides a simple interpretable baseline.
- Random Forest and Gradient Boosting provide stronger non-linear baselines for tabular data.
- XGBoost is included as a high-performing benchmark for structured features.
- MLP, ResNet, and Attention models test whether deeper architectures improve ADR prediction performance.

### How Results Are Generated, Interpreted, and Validated

- Validation and test AUROC/AUPRC are reported for ranking performance.
- The best model is selected by validation AUROC rather than test performance.
- Threshold-based metrics such as F1, precision, recall, specificity, and confusion matrices are used to interpret classification behavior.
- The combined evaluation notebook compares saved artifacts across baseline and deep learning models.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Pipeline

1. Generate ADR labels:

   - Open `notebooks/02_adr_labeling.ipynb`
   - Run all cells and save `adr_labels.csv` in `notebooks/`

2. Preprocess and create model-ready splits:

```bash
python src/preprocessing.py
```

3. Train deep models:

```bash
python src/train.py
```

4. Evaluate trained models:

```bash
python src/evaluate.py
```

5. Train/evaluate baseline ML models in notebook form:

   - Open `notebooks/03_baseline_models.ipynb`
   - Run all cells after preprocessing outputs are available

6. (Optional) Run deep learning workflow in notebook form:

   - Open `notebooks/04_deep_learning_models.ipynb`
   - Train/evaluate deep models (`mlp`, `resnet`, `attention`) interactively
   - Saves notebook-level deep artifacts in `results/`:
     - `deep_models_metrics.csv`
     - `deep_models_metrics.json`
     - `deep_models_test_predictions.json`
     - `deep_best_model_confusion_matrix.png`

7. Run final comparison notebook:

   - Open `notebooks/05_model_evaluation.ipynb`
   - Run all cells to compare baseline and deep learning model metrics

## Assumptions and Limitations

- The project depends on the ADR labeling logic defined in `notebooks/02_adr_labeling.ipynb`.
- The repository does not include raw datasets because MIMIC-IV is restricted-access.
- The current workflow focuses on structured tabular features rather than unstructured notes.
- Deep learning runs are more reproducible now, but some variation across hardware and software environments is still possible.

## Current Progress and Next Steps

### Current Progress

- Data exploration, ADR labeling, preprocessing, baseline modeling, and deep learning workflows are all implemented.
- A final notebook now compares evaluation metrics across the different model families.
- The project uses relative paths and pinned dependencies for easier reproduction.

### Next Steps

- Improve the ADR labeling logic and document the assumptions more formally.
- Expand feature engineering and model tuning.
- Save threshold-based artifacts for every baseline model, not just the selected best baseline.
- Incorporate feedback from the code walk into the documentation and modeling workflow.

## Notes

- Scripts now use repo-relative paths automatically, so no machine-specific path edits are required.
- If a required file is missing, scripts raise a clear `FileNotFoundError` listing missing paths.
- Notebooks auto-detect whether you launched Jupyter from repo root or from `notebooks/`.
- `src/preprocessing.py` uses a fixed split seed, and `src/train.py` now sets NumPy and PyTorch seeds for more reproducible runs.
