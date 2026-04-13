import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

# Configuration
# Resolve all paths relative to repository root so this script works on any machine.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "hosp"
OUTPUT_DIR = PROJECT_ROOT / "processed_data"
OUTPUT_DIR_MIN1000 = PROJECT_ROOT / "processed_data_min1000"
OUTPUT_DIR_EXPANDED = PROJECT_ROOT / "processed_data_expanded"
OUTPUT_DIR_MIN1000_EXPANDED = PROJECT_ROOT / "processed_data_min1000_expanded"
ICU_DIR = PROJECT_ROOT / "data" / "icu"
ADR_LABELS_PATH = PROJECT_ROOT / "notebooks" / "adr_labels.csv"
RANDOM_STATE = 42
MIN_DRUG_COUNT = 1000  # Minimum rows per drug to include in min1000 dataset

# MIMIC-IV labevents item IDs for expanded features
CREATININE_ITEMID = 50912
ALT_ITEMID = 50861
AST_ITEMID = 50878

print("="*80)
print("MIMIC-IV ADR Prediction - Data Preprocessing Pipeline")
print("="*80)

required_inputs = [
    DATA_DIR / "patients.csv.gz",
    DATA_DIR / "prescriptions.csv.gz",
    DATA_DIR / "diagnoses_icd.csv.gz",
    ADR_LABELS_PATH,
]
missing_inputs = [str(path) for path in required_inputs if not path.exists()]
if missing_inputs:
    # Fail early with explicit missing paths instead of raising during read_csv.
    raise FileNotFoundError(
        "Missing required input files:\n- " + "\n- ".join(missing_inputs)
    )

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")

# Load only columns needed for downstream features to reduce memory use.
patients = pd.read_csv(DATA_DIR / "patients.csv.gz")
prescriptions = pd.read_csv(
    DATA_DIR / "prescriptions.csv.gz",
    usecols=["subject_id", "hadm_id", "drug", "starttime", "stoptime", 
             "dose_val_rx", "dose_unit_rx", "route"]
)
diagnoses = pd.read_csv(
    DATA_DIR / "diagnoses_icd.csv.gz",
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
# Left join keeps all prescription rows while adding demographics when available.
data = prescriptions.merge(
    patients[["subject_id", "gender", "anchor_age"]],
    on="subject_id",
    how="left"
)

# Merge with ADR labels
# Inner join restricts rows to medication events that were labeled in notebook step.
data = data.merge(
    adr_labels,
    on=["subject_id", "hadm_id", "drug"],
    how="inner"
)

print(f"Merged dataset shape: {data.shape}")
print(f"ADR distribution:\n{data['ADR'].value_counts(normalize=True)}")

# ============================================================================
# 2b. LOAD SUPPLEMENTARY TABLES AND COMPUTE EXPANDED FEATURES
# ============================================================================
print("\n[2b] Computing expanded features...")

# --- Polypharmacy: distinct drugs per admission ---
# Count from the full prescriptions table (before the inner-join filter).
polypharmacy = (
    prescriptions.groupby(["subject_id", "hadm_id"])["drug"]
    .nunique()
    .reset_index(name="polypharmacy_count")
)

# --- Renal and liver disease flags from diagnoses_icd ---
renal_icd10 = ("N17", "N18", "N19")
renal_icd9  = ("584", "585", "586")
liver_icd10 = ("K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77")
liver_icd9  = ("571",)

diag_flags = diagnoses.copy()
diag_flags["renal_flag"] = (
    diag_flags["icd_code"].str.startswith(renal_icd10) |
    ((diag_flags["icd_version"] == 9) & diag_flags["icd_code"].str.startswith(renal_icd9))
)
diag_flags["liver_flag"] = (
    diag_flags["icd_code"].str.startswith(liver_icd10) |
    ((diag_flags["icd_version"] == 9) & diag_flags["icd_code"].str.startswith(liver_icd9))
)
renal_per_adm = (
    diag_flags.groupby(["subject_id", "hadm_id"])["renal_flag"].max().reset_index()
)
liver_per_adm = (
    diag_flags.groupby(["subject_id", "hadm_id"])["liver_flag"].max().reset_index()
)

# --- Admission type from admissions table ---
admissions_supp = pd.read_csv(
    DATA_DIR / "admissions.csv.gz",
    usecols=["subject_id", "hadm_id", "admission_type", "admittime"],
)
admissions_supp["admittime"] = pd.to_datetime(admissions_supp["admittime"])

# --- Prior ADR flag: did this patient have ADR=1 in any earlier admission? ---
prior_adr_src = adr_labels.merge(
    admissions_supp[["subject_id", "hadm_id", "admittime"]],
    on=["subject_id", "hadm_id"],
    how="left",
).sort_values(["subject_id", "admittime"])
prior_adr_src["prior_adr"] = (
    prior_adr_src.groupby("subject_id")["ADR"]
    .transform(lambda x: x.shift(1).expanding().max().fillna(0))
    .astype(int)
)
prior_adr_src = prior_adr_src[["subject_id", "hadm_id", "drug", "prior_adr"]]

# --- ICU stay flag ---
icu_path = ICU_DIR / "icustays.csv.gz"
if icu_path.exists():
    icustays = pd.read_csv(icu_path, usecols=["subject_id", "hadm_id"])
    icu_flag = icustays.drop_duplicates(subset=["subject_id", "hadm_id"]).copy()
    icu_flag["icu_stay"] = 1
else:
    print("  [warn] icustays.csv.gz not found — icu_stay will be 0 for all rows")
    icu_flag = pd.DataFrame(columns=["subject_id", "hadm_id", "icu_stay"])

# --- Lab values: creatinine, ALT, AST ---
# Take per-admission median to stay memory-efficient on this large table.
lab_path = DATA_DIR / "labevents.csv.gz"
lab_item_ids = [CREATININE_ITEMID, ALT_ITEMID, AST_ITEMID]
if lab_path.exists():
    labs_raw = pd.read_csv(
        lab_path,
        usecols=["subject_id", "hadm_id", "itemid", "valuenum"],
        dtype={"itemid": int},
    )
    labs_raw = labs_raw[
        labs_raw["itemid"].isin(lab_item_ids) & labs_raw["hadm_id"].notna()
    ].copy()
    labs_raw["hadm_id"] = labs_raw["hadm_id"].astype(int)
    lab_pivot = (
        labs_raw.groupby(["subject_id", "hadm_id", "itemid"])["valuenum"]
        .median()
        .unstack("itemid")
        .reset_index()
    )
    lab_pivot.columns.name = None
    lab_pivot = lab_pivot.rename(columns={
        CREATININE_ITEMID: "creatinine",
        ALT_ITEMID: "alt",
        AST_ITEMID: "ast",
    })
    # Ensure all three columns exist even if an itemid had no rows
    for col in ["creatinine", "alt", "ast"]:
        if col not in lab_pivot.columns:
            lab_pivot[col] = np.nan
else:
    print("  [warn] labevents.csv.gz not found — lab values will be median-imputed")
    lab_pivot = pd.DataFrame(columns=["subject_id", "hadm_id", "creatinine", "alt", "ast"])

# --- Merge all expanded features into data ---
data = data.merge(polypharmacy, on=["subject_id", "hadm_id"], how="left")
data = data.merge(renal_per_adm, on=["subject_id", "hadm_id"], how="left")
data = data.merge(liver_per_adm, on=["subject_id", "hadm_id"], how="left")
data = data.merge(
    admissions_supp[["subject_id", "hadm_id", "admission_type", "admittime"]],
    on=["subject_id", "hadm_id"], how="left",
)
data = data.merge(icu_flag, on=["subject_id", "hadm_id"], how="left")
data = data.merge(lab_pivot, on=["subject_id", "hadm_id"], how="left")
data = data.merge(prior_adr_src, on=["subject_id", "hadm_id", "drug"], how="left")

# Fill expanded feature NaNs
data["polypharmacy_count"] = data["polypharmacy_count"].fillna(1)
data["renal_flag"]         = data["renal_flag"].fillna(False).astype(int)
data["liver_flag"]         = data["liver_flag"].fillna(False).astype(int)
data["admission_type"]     = data["admission_type"].fillna("UNKNOWN")
data["icu_stay"]           = data["icu_stay"].fillna(0).astype(int)
data["prior_adr"]          = data["prior_adr"].fillna(0).astype(int)
for lab_col in ["creatinine", "alt", "ast"]:
    data[lab_col] = data[lab_col].fillna(data[lab_col].median())

print(f"  Polypharmacy, renal/liver flags, admission type, ICU, labs, prior ADR added")

# ============================================================================
# 2c. FILTER TO MIN1000 DRUGS (saved as separate dataset)
# ============================================================================
print(f"\n[2c] Building min{MIN_DRUG_COUNT} drug-filtered dataset...")

drug_counts = data["drug"].value_counts()
qualifying_drugs = drug_counts[drug_counts >= MIN_DRUG_COUNT].index
data_min1000 = data[data["drug"].isin(qualifying_drugs)].copy()

print(f"Drugs with >= {MIN_DRUG_COUNT} rows: {len(qualifying_drugs):,} "
      f"(of {drug_counts.shape[0]:,} total)")
print(f"Min{MIN_DRUG_COUNT} rows: {len(data_min1000):,} (of {len(data):,} total)")

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
# Drug-level median preserves medication-specific dosing scale when possible.
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
# Preserve row count while making categorical NaNs explicit for encoding.
data["dose_unit_rx"] = data["dose_unit_rx"].fillna("Unknown")
data["route"] = data["route"].fillna("Unknown")

# Handle time columns
# Coercing errors guards against malformed timestamps in raw tables.
data["starttime"] = pd.to_datetime(data["starttime"], errors='coerce')
data["stoptime"] = pd.to_datetime(data["stoptime"], errors='coerce')

# Calculate treatment duration
# Duration becomes a direct exposure-intensity feature for the classifier.
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
# Bucket age into coarse bins for non-linear age-risk relationships.
data["age_group"] = pd.cut(
    data["anchor_age"],
    bins=[0, 30, 50, 65, 100],
    labels=["young", "middle", "senior", "elderly"]
)

# Dose features
# Log transform reduces skew from very large dose values.
data["log_dose"] = np.log1p(data["dose_val_rx"])

# Drug frequency (how common is this drug)
# This approximates population-level exposure prevalence.
drug_freq = data["drug"].value_counts()
data["drug_frequency"] = data["drug"].map(drug_freq)

# Patient history features — cumulative up to current admission only.
# Using all-time counts would leak future admission/prescription data into past rows.
def add_cumulative_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add patient_admission_count and patient_prescription_count as cumulative
    counts up to and including the current admission, ordered by admittime."""
    adm_order = (
        df[["subject_id", "hadm_id", "admittime"]]
        .drop_duplicates(subset=["subject_id", "hadm_id"])
        .sort_values(["subject_id", "admittime"])
        .copy()
    )
    adm_order["patient_admission_count"] = adm_order.groupby("subject_id").cumcount() + 1

    adm_rx = (
        df.groupby(["subject_id", "hadm_id"])
        .size()
        .reset_index(name="_adm_rx")
        .merge(adm_order[["subject_id", "hadm_id", "admittime"]], on=["subject_id", "hadm_id"], how="left")
        .sort_values(["subject_id", "admittime"])
    )
    adm_rx["patient_prescription_count"] = adm_rx.groupby("subject_id")["_adm_rx"].cumsum()

    df = df.merge(adm_order[["subject_id", "hadm_id", "patient_admission_count"]],
                  on=["subject_id", "hadm_id"], how="left")
    df = df.merge(adm_rx[["subject_id", "hadm_id", "patient_prescription_count"]],
                  on=["subject_id", "hadm_id"], how="left")
    return df

data = add_cumulative_patient_features(data)

# Risk score — kept for backwards compatibility with saved flat MLP checkpoint.
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
# Persisting these encoders enables consistent inference preprocessing later.
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
# Mix encoded categoricals with scaled numeric clinical context variables.
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
    "risk_score",
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
    "risk_score",
]

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("Numerical features scaled using StandardScaler")

# ============================================================================
# 8. TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n[8/8] Splitting data into train/val/test sets...")

# First split: separate test set (20%)
# Stratification preserves ADR prevalence across all splits.
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save splits
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_val.to_csv(OUTPUT_DIR / "y_val.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

# Save preprocessors
# These artifacts are required to reproduce train-time transforms.
with open(OUTPUT_DIR / "label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names
with open(OUTPUT_DIR / "feature_names.txt", "w") as f:
    f.write("\n".join(feature_cols))

# Save full processed dataset for reference
# Useful for audit/debugging feature values outside model scripts.
data.to_csv(OUTPUT_DIR / "full_processed_data.csv", index=False)

print(f"\nAll data saved to {OUTPUT_DIR}/")

# ============================================================================
# 9b. SAVE MIN1000 PROCESSED DATA
# ============================================================================
print(f"\n[9b] Saving min{MIN_DRUG_COUNT} processed dataset...")

# Re-run the same preprocessing steps on the min1000 subset.
data_m = data_min1000.copy()
data_m["dose_val_rx"] = pd.to_numeric(data_m["dose_val_rx"], errors="coerce")
data_m["dose_val_rx"] = data_m.groupby("drug")["dose_val_rx"].transform(
    lambda x: x.fillna(x.median()) if x.notna().any() else x.fillna(0)
)
overall_median_m = data_m["dose_val_rx"].median()
data_m["dose_val_rx"] = data_m["dose_val_rx"].fillna(overall_median_m if pd.notna(overall_median_m) else 0)
data_m["dose_unit_rx"] = data_m["dose_unit_rx"].fillna("Unknown")
data_m["route"] = data_m["route"].fillna("Unknown")
data_m["starttime"] = pd.to_datetime(data_m["starttime"], errors="coerce")
data_m["stoptime"] = pd.to_datetime(data_m["stoptime"], errors="coerce")
data_m["treatment_duration_hours"] = (
    (data_m["stoptime"] - data_m["starttime"]).dt.total_seconds() / 3600
)
data_m["treatment_duration_hours"] = data_m["treatment_duration_hours"].fillna(
    data_m["treatment_duration_hours"].median()
)
data_m["age_group"] = pd.cut(
    data_m["anchor_age"], bins=[0, 30, 50, 65, 100],
    labels=["young", "middle", "senior", "elderly"]
)
data_m["log_dose"] = np.log1p(data_m["dose_val_rx"].clip(lower=0))
drug_freq_m = data_m["drug"].value_counts()
data_m["drug_frequency"] = data_m["drug"].map(drug_freq_m)
data_m = add_cumulative_patient_features(data_m)
data_m["risk_score"] = (
    data_m["anchor_age"] / 100 +
    np.log1p(data_m["patient_prescription_count"]) / 10
)
label_encoders_m = {}
for col in categorical_cols:
    le_m = LabelEncoder()
    data_m[f"{col}_encoded"] = le_m.fit_transform(data_m[col].astype(str))
    label_encoders_m[col] = le_m

X_m = data_m[feature_cols].copy()
y_m = data_m["ADR"].copy()
X_m.fillna(0, inplace=True)
scaler_m = StandardScaler()
X_m[numerical_cols] = scaler_m.fit_transform(X_m[numerical_cols])

X_temp_m, X_test_m, y_temp_m, y_test_m = train_test_split(
    X_m, y_m, test_size=0.2, random_state=RANDOM_STATE, stratify=y_m
)
X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(
    X_temp_m, y_temp_m, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp_m
)

OUTPUT_DIR_MIN1000.mkdir(parents=True, exist_ok=True)
X_train_m.to_csv(OUTPUT_DIR_MIN1000 / "X_train.csv", index=False)
X_val_m.to_csv(OUTPUT_DIR_MIN1000 / "X_val.csv", index=False)
X_test_m.to_csv(OUTPUT_DIR_MIN1000 / "X_test.csv", index=False)
y_train_m.to_csv(OUTPUT_DIR_MIN1000 / "y_train.csv", index=False)
y_val_m.to_csv(OUTPUT_DIR_MIN1000 / "y_val.csv", index=False)
y_test_m.to_csv(OUTPUT_DIR_MIN1000 / "y_test.csv", index=False)
data_m.to_csv(OUTPUT_DIR_MIN1000 / "full_processed_data.csv", index=False)
with open(OUTPUT_DIR_MIN1000 / "label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders_m, f)
with open(OUTPUT_DIR_MIN1000 / "scaler.pkl", "wb") as f:
    pickle.dump(scaler_m, f)
with open(OUTPUT_DIR_MIN1000 / "feature_names.txt", "w") as f:
    f.write("\n".join(feature_cols))

print(f"Min{MIN_DRUG_COUNT} data saved to {OUTPUT_DIR_MIN1000}/")
print(f"  Train: {X_train_m.shape}, ADR rate: {y_train_m.mean():.3f}")
print(f"  Val:   {X_val_m.shape}, ADR rate: {y_val_m.mean():.3f}")
print(f"  Test:  {X_test_m.shape}, ADR rate: {y_test_m.mean():.3f}")

# ============================================================================
# 10. EXPANDED FEATURE PIPELINE — FULL DATASET
# ============================================================================
print("\n[10] Saving expanded feature dataset (full)...")

def run_expanded_pipeline(df, output_dir, label):
    """Encode, scale, split and save a dataset using the expanded feature set."""
    d = df.copy()

    # Encode original categoricals
    enc = {}
    for col in categorical_cols:
        le = LabelEncoder()
        d[f"{col}_encoded"] = le.fit_transform(d[col].astype(str))
        enc[col] = le

    # Encode new categorical: admission_type
    le_adm = LabelEncoder()
    d["admission_type_encoded"] = le_adm.fit_transform(d["admission_type"].astype(str))
    enc["admission_type"] = le_adm

    expanded_feature_cols = feature_cols + [
        "polypharmacy_count",
        "renal_flag",
        "liver_flag",
        "admission_type_encoded",
        "creatinine",
        "alt",
        "ast",
    ]

    expanded_numerical_cols = numerical_cols + [
        "polypharmacy_count",
        "creatinine",
        "alt",
        "ast",
    ]

    X_exp = d[expanded_feature_cols].copy()
    y_exp = d["ADR"].copy()
    X_exp.fillna(0, inplace=True)

    sc = StandardScaler()
    X_exp[expanded_numerical_cols] = sc.fit_transform(X_exp[expanded_numerical_cols])

    X_tmp, X_tst, y_tmp, y_tst = train_test_split(
        X_exp, y_exp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_exp
    )
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tmp
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    X_tr.to_csv(output_dir / "X_train.csv", index=False)
    X_vl.to_csv(output_dir / "X_val.csv", index=False)
    X_tst.to_csv(output_dir / "X_test.csv", index=False)
    y_tr.to_csv(output_dir / "y_train.csv", index=False)
    y_vl.to_csv(output_dir / "y_val.csv", index=False)
    y_tst.to_csv(output_dir / "y_test.csv", index=False)
    d.to_csv(output_dir / "full_processed_data.csv", index=False)
    with open(output_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(enc, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(sc, f)
    with open(output_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(expanded_feature_cols))

    print(f"{label} expanded data saved to {output_dir}/")
    print(f"  Train: {X_tr.shape}, ADR rate: {y_tr.mean():.3f}")
    print(f"  Val:   {X_vl.shape}, ADR rate: {y_vl.mean():.3f}")
    print(f"  Test:  {X_tst.shape}, ADR rate: {y_tst.mean():.3f}")

run_expanded_pipeline(data, OUTPUT_DIR_EXPANDED, "Full")

# ============================================================================
# 11. EXPANDED FEATURE PIPELINE — MIN1000 DATASET
# ============================================================================
print("\n[11] Saving expanded feature dataset (min1000)...")
run_expanded_pipeline(data[data["drug"].isin(qualifying_drugs)].copy(), OUTPUT_DIR_MIN1000_EXPANDED, f"Min{MIN_DRUG_COUNT}")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)

# ============================================================================
# 12. SUMMARY STATISTICS
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
