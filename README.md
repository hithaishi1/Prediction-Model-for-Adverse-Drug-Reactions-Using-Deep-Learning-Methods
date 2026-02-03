# Prediction Model for Adverse Drug Reactions Using Deep Learning Methods

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
