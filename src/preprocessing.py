import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from datetime import timedelta

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

# Configuration
BASE_DIR = "/Users/hithaishireddy/Desktop/ADR-project/Prediction-Model-for-Adverse-Drug-Reactions-Using-Deep-Learning-Methods"
DATA_DIR = f"{BASE_DIR}/data/hosp"
OUTPUT_DIR = f"{BASE_DIR}/processed_data"
ADR_LABELS_PATH = f"{BASE_DIR}/notebooks/adr_labels.csv"
RANDOM_STATE = 42

print("="*80)
print("MIMIC-IV ADR Prediction - Data Preprocessing Pipeline")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")

patients = pd.read_csv(f"{DATA_DIR}/patients.csv.gz")
prescriptions = pd.read_csv(
    f"{DATA_DIR}/prescriptions.csv.gz",
    usecols=["subject_id", "hadm_id", "drug", "starttime", "stoptime", 
             "dose_val_rx", "dose_unit_rx", "route"]
)
diagnoses = pd.read_csv(
    f"{DATA_DIR}/diagnoses_icd.csv.gz",
    usecols=["subject_id", "hadm_id", "icd_code", "icd_version"]
)
adr_labels = pd.read_csv(ADR_LABELS_PATH)

print(f"Patients: {patients.shape}")
print(f"Prescriptions: {prescriptions.shape}")
print(f"Diagnoses: {diagnoses.shape}")
print(f"ADR Labels: {adr_labels.shape}")

# ============================================================================
# 2. MERGE DATASETS
# ============================================================================
print("\n[2/8] Merging datasets...")

# Merge prescriptions with patient demographics
data = prescriptions.merge(
    patients[["subject_id", "gender", "anchor_age"]],
    on="subject_id",
    how="left"
)

# Merge with ADR labels
data = data.merge(
    adr_labels,
    on=["subject_id", "hadm_id", "drug"],
    how="inner"
)

print(f"Merged dataset shape: {data.shape}")
print(f"ADR distribution:\n{data['ADR'].value_counts(normalize=True)}")

# ============================================================================
# 3. HANDLE MISSING VALUES
# ============================================================================
print("\n[3/8] Handling missing values...")

print("\nMissing values before handling:")
print(data.isnull().sum())

# Convert dose_val_rx to numeric (handles mixed types)
print("\nConverting dose_val_rx to numeric...")
data["dose_val_rx"] = pd.to_numeric(data["dose_val_rx"], errors='coerce')

# Fill missing doses with median by drug
data["dose_val_rx"] = data.groupby("drug")["dose_val_rx"].transform(
    lambda x: x.fillna(x.median()) if x.notna().any() else x.fillna(0)
)

# Fill remaining missing doses with overall median
overall_median = data["dose_val_rx"].median()
if pd.notna(overall_median):
    data["dose_val_rx"] = data["dose_val_rx"].fillna(overall_median)
else:
    data["dose_val_rx"] = data["dose_val_rx"].fillna(0)

# Fill missing dose units and routes with 'Unknown'
data["dose_unit_rx"] = data["dose_unit_rx"].fillna("Unknown")
data["route"] = data["route"].fillna("Unknown")

# Handle time columns
data["starttime"] = pd.to_datetime(data["starttime"], errors='coerce')
data["stoptime"] = pd.to_datetime(data["stoptime"], errors='coerce')

# Calculate treatment duration
data["treatment_duration_hours"] = (
    (data["stoptime"] - data["starttime"]).dt.total_seconds() / 3600
)
data["treatment_duration_hours"] = data["treatment_duration_hours"].fillna(
    data["treatment_duration_hours"].median()
)

print("\nMissing values after handling:")
print(data.isnull().sum())

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n[4/8] Engineering features...")

# Age groups
data["age_group"] = pd.cut(
    data["anchor_age"],
    bins=[0, 30, 50, 65, 100],
    labels=["young", "middle", "senior", "elderly"]
)

# Dose features
data["log_dose"] = np.log1p(data["dose_val_rx"])

# Drug frequency (how common is this drug)
drug_freq = data["drug"].value_counts()
data["drug_frequency"] = data["drug"].map(drug_freq)

# Patient history features (number of admissions and prescriptions)
patient_admissions = data.groupby("subject_id")["hadm_id"].nunique()
data["patient_admission_count"] = data["subject_id"].map(patient_admissions)

patient_prescriptions = data.groupby("subject_id").size()
data["patient_prescription_count"] = data["subject_id"].map(patient_prescriptions)

# Risk score based on age and prescription count
data["risk_score"] = (
    data["anchor_age"] / 100 + 
    np.log1p(data["patient_prescription_count"]) / 10
)

print(f"\nFeatures created. New shape: {data.shape}")

# ============================================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n[5/8] Encoding categorical variables...")

# Initialize encoders
label_encoders = {}

categorical_cols = ["drug", "gender", "dose_unit_rx", "route", "age_group"]

for col in categorical_cols:
    le = LabelEncoder()
    data[f"{col}_encoded"] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# ============================================================================
# 6. PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[6/8] Preparing features and target...")

# Select features for modeling
feature_cols = [
    "drug_encoded",
    "gender_encoded",
    "anchor_age",
    "age_group_encoded",
    "dose_val_rx",
    "log_dose",
    "dose_unit_rx_encoded",
    "route_encoded",
    "treatment_duration_hours",
    "drug_frequency",
    "patient_admission_count",
    "patient_prescription_count",
    "risk_score"
]

X = data[feature_cols].copy()
y = data["ADR"].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns:\n{feature_cols}")

# Check for any remaining missing values
if X.isnull().any().any():
    print("\nWARNING: Missing values found in features!")
    print(X.isnull().sum())
    X.fillna(0, inplace=True)

# ============================================================================
# 7. SCALE NUMERICAL FEATURES
# ============================================================================
print("\n[7/8] Scaling numerical features...")

numerical_cols = [
    "anchor_age",
    "dose_val_rx",
    "log_dose",
    "treatment_duration_hours",
    "drug_frequency",
    "patient_admission_count",
    "patient_prescription_count",
    "risk_score"
]

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("Numerical features scaled using StandardScaler")

# ============================================================================
# 8. TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n[8/8] Splitting data into train/val/test sets...")

# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Second split: separate validation set (20% of remaining = 16% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"\nTrain set: {X_train.shape}, ADR rate: {y_train.mean():.3f}")
print(f"Val set:   {X_val.shape}, ADR rate: {y_val.mean():.3f}")
print(f"Test set:  {X_test.shape}, ADR rate: {y_test.mean():.3f}")

# ============================================================================
# 9. SAVE PROCESSED DATA
# ============================================================================
print("\n[9/9] Saving processed data...")

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save splits
X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
X_val.to_csv(f"{OUTPUT_DIR}/X_val.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)

y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_val.to_csv(f"{OUTPUT_DIR}/y_val.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

# Save preprocessors
with open(f"{OUTPUT_DIR}/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names
with open(f"{OUTPUT_DIR}/feature_names.txt", "w") as f:
    f.write("\n".join(feature_cols))

# Save full processed dataset for reference
data.to_csv(f"{OUTPUT_DIR}/full_processed_data.csv", index=False)

print(f"\nAll data saved to {OUTPUT_DIR}/")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)

# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nClass Distribution:")
print(f"Train - No ADR: {(y_train == 0).sum()}, ADR: {(y_train == 1).sum()}")
print(f"Val   - No ADR: {(y_val == 0).sum()}, ADR: {(y_val == 1).sum()}")
print(f"Test  - No ADR: {(y_test == 0).sum()}, ADR: {(y_test == 1).sum()}")

print("\nFeature Statistics (Training Set):")
print(X_train.describe())

print("\n" + "="*80)
print("Ready for modeling!")
print("="*80)