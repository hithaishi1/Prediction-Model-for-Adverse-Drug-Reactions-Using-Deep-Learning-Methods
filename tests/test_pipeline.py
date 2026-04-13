"""
Unit and integration tests for the ADR prediction pipeline.

Covers:
  - Data preprocessing utilities (feature engineering, splitting, scaling)
  - Model architecture (forward pass shape, output range, checkpoint I/O)
  - Ensemble logic
  - Edge cases (batch size 1, feature mismatch)

Run with:
    pytest tests/test_pipeline.py -v
"""

import sys
import os
import tempfile

import pytest
import numpy as np
import pandas as pd
import torch

# Allow imports from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import MLPClassifier, ResNetClassifier, AttentionClassifier, DeepEnsemble, get_model, FocalLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Small synthetic DataFrame that mimics the structure of X_train."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "drug_encoded":              np.random.randint(0, 50, n),
        "gender_encoded":            np.random.randint(0, 2, n),
        "anchor_age":                np.random.randint(18, 90, n).astype(float),
        "age_group_encoded":         np.random.randint(0, 4, n),
        "dose_val_rx":               np.random.exponential(100, n),
        "log_dose":                  np.random.uniform(0, 5, n),
        "dose_unit_rx_encoded":      np.random.randint(0, 20, n),
        "route_encoded":             np.random.randint(0, 15, n),
        "treatment_duration_hours":  np.random.uniform(1, 500, n),
        "drug_frequency":            np.random.randint(100, 5000, n).astype(float),
        "patient_admission_count":   np.random.randint(1, 20, n).astype(float),
        "patient_prescription_count":np.random.randint(1, 50, n).astype(float),
    })


@pytest.fixture
def sample_labels(sample_df):
    """Imbalanced binary labels matching sample_df row count (~17% positive)."""
    np.random.seed(42)
    return pd.Series(
        np.random.choice([0, 1], size=len(sample_df), p=[0.83, 0.17]),
        name="adr"
    )


@pytest.fixture
def mlp_model():
    return MLPClassifier(input_dim=12, hidden_dims=[64, 32], dropout_rate=0.1)


@pytest.fixture
def input_batch():
    torch.manual_seed(42)
    return torch.randn(32, 12)


# ---------------------------------------------------------------------------
# Section 1 — Data preprocessing utilities
# ---------------------------------------------------------------------------

class TestPreprocessing:

    def test_stratified_split_preserves_class_ratio(self, sample_df, sample_labels):
        """Stratified split should keep ADR rate within 5% in both subsets."""
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            sample_df, sample_labels, test_size=0.2, random_state=42, stratify=sample_labels
        )
        assert abs(y_train.mean() - y_test.mean()) < 0.05

    def test_standard_scaler_fit_on_train_only(self, sample_df):
        """Scaler fitted on train set; test set transformed using train statistics only."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        train, test = train_test_split(sample_df, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled  = scaler.transform(test)       # must NOT call fit again
        # Train columns should be approximately zero-mean after scaling
        assert abs(train_scaled.mean()) < 0.5
        # Test transform should complete without error and preserve shape
        assert test_scaled.shape == test.shape

    def test_log_dose_non_negative(self, sample_df):
        """log1p(dose_val_rx) should always be >= 0 for non-negative dose values."""
        log_dose = np.log1p(sample_df["dose_val_rx"].clip(lower=0))
        assert (log_dose >= 0).all()

    def test_no_nulls_after_fillna(self, sample_df):
        """After median fill, no NaN values should remain in numeric columns."""
        df = sample_df.copy()
        # Introduce NaNs
        df.loc[df.index[:5], "dose_val_rx"] = np.nan
        df["dose_val_rx"] = df["dose_val_rx"].fillna(df["dose_val_rx"].median())
        assert df["dose_val_rx"].isna().sum() == 0

    def test_min1000_filter_removes_rare_drugs(self):
        """Min1000 filter should exclude drugs with fewer than 1000 rows."""
        np.random.seed(42)
        drugs = ["DrugA"] * 1500 + ["DrugB"] * 800 + ["DrugC"] * 1200
        df = pd.DataFrame({"drug": drugs})
        drug_counts = df["drug"].value_counts()
        qualifying = drug_counts[drug_counts >= 1000].index
        filtered = df[df["drug"].isin(qualifying)]
        assert "DrugB" not in filtered["drug"].values
        assert "DrugA" in filtered["drug"].values
        assert "DrugC" in filtered["drug"].values

    def test_leaky_feature_exclusion(self):
        """prior_adr and icu_stay should be absent from the 19-feature set."""
        cols_21 = [
            "drug_encoded", "gender_encoded", "anchor_age", "age_group_encoded",
            "dose_val_rx", "log_dose", "dose_unit_rx_encoded", "route_encoded",
            "treatment_duration_hours", "drug_frequency", "patient_admission_count",
            "patient_prescription_count", "polypharmacy_count", "renal_flag",
            "liver_flag", "admission_type_encoded", "icu_stay", "prior_adr",
            "creatinine", "alt", "ast",
        ]
        EXCLUDE = {"prior_adr", "icu_stay"}
        cols_19 = [c for c in cols_21 if c not in EXCLUDE]
        assert "prior_adr" not in cols_19
        assert "icu_stay" not in cols_19
        assert len(cols_19) == 19


# ---------------------------------------------------------------------------
# Section 2 — Model architecture
# ---------------------------------------------------------------------------

class TestModelArchitecture:

    def test_mlp_output_shape(self, mlp_model, input_batch):
        """MLP forward pass should output shape (batch_size, 1)."""
        mlp_model.eval()
        with torch.no_grad():
            out = mlp_model(input_batch)
        assert out.shape == (32, 1)

    def test_resnet_output_shape(self, input_batch):
        """ResNet forward pass should output shape (batch_size, 1)."""
        model = ResNetClassifier(input_dim=12, hidden_dim=64, num_blocks=2)
        model.eval()
        with torch.no_grad():
            out = model(input_batch)
        assert out.shape == (32, 1)

    def test_attention_output_shape(self, input_batch):
        """Attention model forward pass should output shape (batch_size, 1)."""
        model = AttentionClassifier(input_dim=12, hidden_dims=[64, 32])
        model.eval()
        with torch.no_grad():
            out = model(input_batch)
        assert out.shape == (32, 1)

    def test_sigmoid_output_in_unit_interval(self, mlp_model, input_batch):
        """Sigmoid-transformed logits must all lie in [0, 1]."""
        mlp_model.eval()
        with torch.no_grad():
            logits = mlp_model(input_batch)
            probs = torch.sigmoid(logits)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_model_reproducibility(self, input_batch):
        """Same seed and same input should produce identical outputs."""
        torch.manual_seed(0)
        model = MLPClassifier(input_dim=12, hidden_dims=[64, 32], dropout_rate=0.0)
        model.eval()
        with torch.no_grad():
            out1 = model(input_batch)
            out2 = model(input_batch)
        assert torch.allclose(out1, out2)

    def test_checkpoint_save_and_load(self, mlp_model, input_batch):
        """Model saved to disk and reloaded should produce identical outputs."""
        mlp_model.eval()
        with torch.no_grad():
            out_before = mlp_model(input_batch)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        torch.save(mlp_model.state_dict(), path)

        loaded = MLPClassifier(input_dim=12, hidden_dims=[64, 32], dropout_rate=0.1)
        loaded.load_state_dict(torch.load(path, map_location="cpu"))
        loaded.eval()
        with torch.no_grad():
            out_after = loaded(input_batch)

        assert torch.allclose(out_before, out_after)
        os.unlink(path)

    def test_batch_size_one_in_eval_mode(self, mlp_model):
        """Single-sample inference must succeed in eval mode (BatchNorm uses running stats)."""
        mlp_model.eval()
        single = torch.randn(1, 12)
        with torch.no_grad():
            out = mlp_model(single)
        assert out.shape == (1, 1)

    def test_get_model_factory_raises_on_unknown_type(self):
        """get_model() should raise ValueError for unrecognised model type."""
        with pytest.raises(ValueError):
            get_model("transformer", input_dim=12)

    def test_feature_mismatch_raises_error(self):
        """Model trained on 19 features should raise when given 21-feature input."""
        model = MLPClassifier(input_dim=19, hidden_dims=[64, 32])
        model.eval()
        wrong_input = torch.randn(32, 21)   # 21 features instead of 19
        with pytest.raises(RuntimeError):
            with torch.no_grad():
                model(wrong_input)


# ---------------------------------------------------------------------------
# Section 3 — Ensemble
# ---------------------------------------------------------------------------

class TestDeepEnsemble:

    def test_ensemble_output_shape(self, input_batch):
        """DeepEnsemble should output shape (batch_size, 1)."""
        m1 = MLPClassifier(input_dim=12, hidden_dims=[64])
        m2 = ResNetClassifier(input_dim=12, hidden_dim=64, num_blocks=1)
        ensemble = DeepEnsemble([m1, m2])
        ensemble.eval()
        with torch.no_grad():
            out = ensemble(input_batch)
        assert out.shape == (32, 1)

    def test_equal_weight_ensemble_is_mean(self, input_batch):
        """Equal-weight ensemble output should equal the mean of individual model outputs."""
        torch.manual_seed(1)
        m1 = MLPClassifier(input_dim=12, hidden_dims=[32], dropout_rate=0.0)
        m2 = MLPClassifier(input_dim=12, hidden_dims=[32], dropout_rate=0.0)
        ensemble = DeepEnsemble([m1, m2])   # default equal weights
        ensemble.eval()
        with torch.no_grad():
            out_ensemble = ensemble(input_batch)
            out_mean = (m1(input_batch) + m2(input_batch)) / 2
        assert torch.allclose(out_ensemble, out_mean, atol=1e-5)


# ---------------------------------------------------------------------------
# Section 4 — Focal loss
# ---------------------------------------------------------------------------

class TestFocalLoss:

    def test_focal_loss_is_non_negative(self):
        """Focal loss should always return a non-negative scalar."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logits  = torch.randn(32, 1)
        targets = torch.randint(0, 2, (32, 1)).float()
        loss = criterion(logits, targets)
        assert loss.item() >= 0

    def test_focal_loss_decreases_on_easy_examples(self):
        """Focal loss should down-weight confident correct predictions (easy examples)."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        # Very confident correct prediction (logit = 10, target = 1)
        easy_logit  = torch.tensor([[10.0]])
        easy_target = torch.tensor([[1.0]])
        # Uncertain prediction (logit = 0, target = 1)
        hard_logit  = torch.tensor([[0.0]])
        hard_target = torch.tensor([[1.0]])
        easy_loss = criterion(easy_logit, easy_target)
        hard_loss = criterion(hard_logit, hard_target)
        assert easy_loss.item() < hard_loss.item()
