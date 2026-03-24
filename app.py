"""
ADR Prediction Dashboard — Streamlit Frontend
Run with: streamlit run app.py
Place this file in the project root (same level as src/ and processed_data/).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

DATA_DIR      = PROJECT_ROOT / "processed_data"
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "results"

from models import get_model  # noqa: E402  (imported after path fix)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADR Prediction Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3a3d4e;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c83fd; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .risk-high   { color: #ef4444; font-weight: 700; font-size: 1.4rem; }
    .risk-medium { color: #f59e0b; font-weight: 700; font-size: 1.4rem; }
    .risk-low    { color: #10b981; font-weight: 700; font-size: 1.4rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    div[data-testid="stSidebar"] { background-color: #1e2130; }
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────
DRUG_OPTIONS = [
    "Acetaminophen", "Aspirin", "Furosemide", "Heparin", "Insulin",
    "Metoprolol", "Morphine", "Ondansetron", "Pantoprazole",
    "Potassium Chloride", "0.9% Sodium Chloride", "Vancomycin",
    "Warfarin", "Metformin", "Lisinopril", "Amiodarone",
    "Ceftriaxone", "Dexamethasone", "Lorazepam", "Magnesium Sulfate",
]

ROUTE_OPTIONS = ["PO/NG", "IV", "SC", "IM", "PO", "IV PUSH", "TOPICAL", "Unknown"]
DOSE_UNIT_OPTIONS = ["mg", "mL", "UNIT", "mcg", "Unknown"]


@st.cache_resource
def load_preprocessors():
    enc_path    = DATA_DIR / "label_encoders.pkl"
    scaler_path = DATA_DIR / "scaler.pkl"
    feat_path   = DATA_DIR / "feature_names.txt"
    if not (enc_path.exists() and scaler_path.exists()):
        return None, None, None
    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    feat_names = feat_path.read_text().strip().splitlines() if feat_path.exists() else None
    return encoders, scaler, feat_names


@st.cache_resource
def load_dl_model(model_name: str, input_dim: int):
    cfg = {
        "mlp":       dict(hidden_dims=[256,128,64,32], dropout_rate=0.3),
        "resnet":    dict(hidden_dim=256, num_blocks=3,  dropout_rate=0.3),
        "attention": dict(hidden_dims=[256,128],         dropout_rate=0.3),
    }
    path = MODELS_DIR / f"{model_name}_best.pth"
    if not path.exists():
        return None
    model = get_model(model_name, input_dim=input_dim, **cfg[model_name])
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # Rename "net.*" keys → "network.*" if the checkpoint was saved with
    # an older model definition that used self.net instead of self.network.
    if any(k.startswith("net.") for k in state):
        state = {k.replace("net.", "network.", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data
def load_results():
    paths = {
        "baseline_metrics": RESULTS_DIR / "baseline_models_metrics.csv",
        "deep_metrics":     RESULTS_DIR / "deep_models_metrics.csv",
        "combined":         RESULTS_DIR / "combined_model_metrics.csv",
        "baseline_preds":   RESULTS_DIR / "best_baseline_test_predictions.csv",
        "deep_preds":       RESULTS_DIR / "deep_models_test_predictions.json",
        "threshold":        RESULTS_DIR / "model_threshold_metrics.csv",
    }
    out = {}
    for k, p in paths.items():
        if not p.exists():
            continue
        if p.suffix == ".csv":
            out[k] = pd.read_csv(p)
        else:
            with open(p) as f:
                out[k] = json.load(f)
    return out


@st.cache_data
def load_history(model_name: str):
    p = MODELS_DIR / f"{model_name}_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_test_data():
    xp = DATA_DIR / "X_test.csv"
    yp = DATA_DIR / "y_test.csv"
    if not (xp.exists() and yp.exists()):
        return None, None
    return pd.read_csv(xp), pd.read_csv(yp)


def encode_input(row: dict, encoders, scaler, feat_names: list) -> np.ndarray | None:
    """Convert a raw patient-drug dict into the scaled 13-feature vector."""
    try:
        def safe_encode(enc, val):
            val = str(val)
            if val in enc.classes_:
                return int(enc.transform([val])[0])
            return int(enc.transform([enc.classes_[0]])[0])

        age = float(row["anchor_age"])
        if   age < 30: age_grp = "young"
        elif age < 50: age_grp = "middle"
        elif age < 65: age_grp = "senior"
        else:          age_grp = "elderly"

        dose      = float(row["dose_val_rx"])
        log_dose  = float(np.log1p(dose))
        dur       = float(row["treatment_duration_hours"])
        drug_freq = float(row.get("drug_frequency", 1000))
        adm_cnt   = float(row.get("patient_admission_count", 1))
        prx_cnt   = float(row.get("patient_prescription_count", 5))
        risk      = age / 100 + np.log1p(prx_cnt) / 10

        feat = {
            "drug_encoded":                safe_encode(encoders["drug"],          row["drug"]),
            "gender_encoded":              safe_encode(encoders["gender"],         row["gender"]),
            "anchor_age":                  age,
            "age_group_encoded":           safe_encode(encoders["age_group"],      age_grp),
            "dose_val_rx":                 dose,
            "log_dose":                    log_dose,
            "dose_unit_rx_encoded":        safe_encode(encoders["dose_unit_rx"],   row["dose_unit_rx"]),
            "route_encoded":               safe_encode(encoders["route"],          row["route"]),
            "treatment_duration_hours":    dur,
            "drug_frequency":              drug_freq,
            "patient_admission_count":     adm_cnt,
            "patient_prescription_count":  prx_cnt,
            "risk_score":                  risk,
        }

        ordered = feat_names if feat_names else list(feat.keys())
        vec = np.array([[feat[k] for k in ordered]], dtype=np.float32)

        num_cols = ["anchor_age","dose_val_rx","log_dose","treatment_duration_hours",
                    "drug_frequency","patient_admission_count","patient_prescription_count","risk_score"]
        num_idx  = [ordered.index(c) for c in num_cols if c in ordered]
        vec[0, num_idx] = scaler.transform(vec[:, num_idx])[0]
        return vec
    except Exception as e:
        st.error(f"Feature encoding error: {e}")
        return None


def risk_badge(prob: float) -> str:
    if prob >= 0.6:
        return f'<span class="risk-high">🔴 HIGH RISK ({prob:.1%})</span>'
    if prob >= 0.35:
        return f'<span class="risk-medium">🟡 MEDIUM RISK ({prob:.1%})</span>'
    return f'<span class="risk-low">🟢 LOW RISK ({prob:.1%})</span>'


def gauge_chart(prob: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": "#fff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
            "bar":  {"color": "#7c83fd"},
            "steps": [
                {"range": [0,  35], "color": "#064e3b"},
                {"range": [35, 60], "color": "#78350f"},
                {"range": [60,100], "color": "#7f1d1d"},
            ],
            "threshold": {
                "line": {"color": "#fff", "width": 3},
                "thickness": 0.85,
                "value": prob * 100,
            },
        },
        title={"text": "ADR Probability", "font": {"color": "#9ca3af", "size": 16}},
    ))
    fig.update_layout(
        paper_bgcolor="#1e2130", font_color="#fff",
        margin=dict(t=60, b=20, l=30, r=30), height=280,
    )
    return fig


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 ADR Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔬 Predict ADR", "📊 Model Performance", "🗂️ Past Predictions"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Model**")
    model_choice = st.selectbox("Select model", ["mlp", "resnet", "attention"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("MIMIC-IV · ADR Prediction Project")

# ── load shared assets ────────────────────────────────────────────────────────
encoders, scaler, feat_names = load_preprocessors()
results = load_results()

# infer input_dim from test data or feature list
X_test, y_test = load_test_data()
input_dim = X_test.shape[1] if X_test is not None else (len(feat_names) if feat_names else 13)

dl_model = load_dl_model(model_choice, input_dim)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT ADR
# ═══════════════════════════════════════════════════════════════════════════
if page == "🔬 Predict ADR":
    st.title("🔬 ADR Risk Prediction")
    st.markdown("Enter patient and drug information below to estimate adverse drug reaction risk.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("👤 Patient")
            age    = st.slider("Age", 18, 91, 55)
            gender = st.selectbox("Sex", ["M", "F"])
            adm    = st.number_input("Admission count", 1, 50, 2)
            prx    = st.number_input("Prior prescriptions", 1, 200, 10)

        with c2:
            st.subheader("💊 Drug")
            drug  = st.selectbox("Drug", DRUG_OPTIONS)
            dose  = st.number_input("Dose value", 0.0, 10000.0, 500.0, step=50.0)
            d_unit = st.selectbox("Dose unit", DOSE_UNIT_OPTIONS)
            route = st.selectbox("Route", ROUTE_OPTIONS)

        with c3:
            st.subheader("⏱️ Treatment")
            dur   = st.number_input("Duration (hours)", 0.0, 720.0, 24.0, step=1.0)
            freq  = st.number_input("Drug frequency (population)", 100, 100000, 5000, step=100)
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Predict ADR Risk", use_container_width=True)

    if submitted:
        if dl_model is None or encoders is None:
            st.error("Model or preprocessors not found. Run training first.")
        else:
            row = dict(drug=drug, gender=gender, anchor_age=age,
                       dose_val_rx=dose, dose_unit_rx=d_unit, route=route,
                       treatment_duration_hours=dur, drug_frequency=freq,
                       patient_admission_count=adm, patient_prescription_count=prx)
            vec = encode_input(row, encoders, scaler, feat_names)
            if vec is not None:
                with torch.no_grad():
                    prob = float(torch.sigmoid(dl_model(torch.FloatTensor(vec))).item())

                st.markdown("---")
                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    st.plotly_chart(gauge_chart(prob), use_container_width=True)
                with rc2:
                    st.markdown("### Risk Assessment")
                    st.markdown(risk_badge(prob), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**Drug:** {drug}  |  **Dose:** {dose} {d_unit}  |  **Route:** {route}")
                    st.markdown(f"**Patient:** {gender}, Age {age}  |  **Duration:** {dur}h")
                    st.markdown("---")
                    if prob >= 0.6:
                        st.error("⚠️ High ADR risk detected. Consider alternative medications or dose adjustment.")
                    elif prob >= 0.35:
                        st.warning("⚠️ Moderate ADR risk. Monitor patient closely.")
                    else:
                        st.success("✅ Low ADR risk based on available patient features.")

                    # save to session history
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "Drug": drug, "Age": age, "Sex": gender,
                        "Dose": f"{dose} {d_unit}", "Route": route,
                        "Model": model_choice.upper(), "ADR Risk": f"{prob:.1%}",
                        "Level": "High" if prob>=0.6 else ("Medium" if prob>=0.35 else "Low"),
                    })


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    tab1, tab2, tab3 = st.tabs(["📈 Metric Comparison", "📉 ROC / PR Curves", "🔄 Training History"])

    # ── tab 1: summary metrics ──────────────────────────────────────────────
    with tab1:
        df = results.get("combined") or results.get("baseline_metrics")
        if df is not None:
            # headline KPI cards
            best = df.sort_values("Test AUROC", ascending=False).iloc[0]
            k1, k2, k3, k4 = st.columns(4)
            for col, label, val in [
                (k1, "Best AUROC",  f"{best['Test AUROC']:.4f}"),
                (k2, "Best AUPRC",  f"{best['Test AUPRC']:.4f}"),
                (k3, "Best Model",  best["Model"]),
                (k4, "# Models",    str(len(df))),
            ]:
                col.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

            # bar chart
            fig = px.bar(
                df.sort_values("Test AUROC"),
                x="Test AUROC", y="Model",
                color="Test AUROC",
                color_continuous_scale="Viridis",
                orientation="h",
                title="Test AUROC by Model",
                text=df.sort_values("Test AUROC")["Test AUROC"].map("{:.4f}".format),
            )
            fig.update_layout(
                paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                font_color="#fff", coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            # AUROC vs AUPRC scatter
            fig2 = px.scatter(
                df, x="Test AUROC", y="Test AUPRC",
                text="Model", size_max=18,
                color="Test AUROC", color_continuous_scale="Viridis",
                title="AUROC vs AUPRC (Test Set)",
            )
            fig2.update_traces(textposition="top center", marker_size=12)
            fig2.update_layout(
                paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                font_color="#fff", coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(df.style.format({
                "Test AUROC": "{:.4f}", "Test AUPRC": "{:.4f}",
                "Val AUROC":  "{:.4f}", "Val AUPRC":  "{:.4f}",
            }), use_container_width=True)
        else:
            st.info("Run the training notebooks to generate result files.")

    # ── tab 2: ROC / PR curves ──────────────────────────────────────────────
    with tab2:
        if X_test is not None and y_test is not None:
            y_true = y_test.values.flatten()
            deep_preds = results.get("deep_preds", {})
            baseline_preds_df = results.get("baseline_preds")

            roc_fig = go.Figure()
            pr_fig  = go.Figure()

            colors = px.colors.qualitative.Plotly

            # deep models
            for i, (mname, payload) in enumerate(deep_preds.items()):
                yp = np.array(payload["y_proba"])
                yt = np.array(payload["y_true"])
                fpr, tpr, _ = roc_curve(yt, yp)
                p, r, _     = precision_recall_curve(yt, yp)
                auc  = roc_auc_score(yt, yp)
                aprc = average_precision_score(yt, yp)
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{mname.upper()} (AUC={auc:.3f})", line=dict(color=colors[i], width=2)))
                pr_fig.add_trace( go.Scatter(x=r,   y=p,   name=f"{mname.upper()} (AUPRC={aprc:.3f})", line=dict(color=colors[i], width=2)))

            # best baseline
            if baseline_preds_df is not None and "y_proba" in baseline_preds_df.columns:
                yp = baseline_preds_df["y_proba"].values
                yt = baseline_preds_df["y_true"].values
                fpr, tpr, _ = roc_curve(yt, yp)
                p, r, _     = precision_recall_curve(yt, yp)
                auc  = roc_auc_score(yt, yp)
                aprc = average_precision_score(yt, yp)
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Best Baseline (AUC={auc:.3f})", line=dict(color="gold", width=2, dash="dash")))
                pr_fig.add_trace( go.Scatter(x=r,   y=p,   name=f"Best Baseline (AUPRC={aprc:.3f})", line=dict(color="gold", width=2, dash="dash")))

            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(color="gray", dash="dot")))

            for fig_, title_, xlab_, ylab_ in [
                (roc_fig, "ROC Curves",            "False Positive Rate", "True Positive Rate"),
                (pr_fig,  "Precision-Recall Curves","Recall",             "Precision"),
            ]:
                fig_.update_layout(
                    title=title_, xaxis_title=xlab_, yaxis_title=ylab_,
                    paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                    font_color="#fff", legend=dict(bgcolor="#1e2130"),
                )
                st.plotly_chart(fig_, use_container_width=True)
        else:
            st.info("Test data not found. Run preprocessing.py first.")

    # ── tab 3: training history ─────────────────────────────────────────────
    with tab3:
        sel = st.selectbox("Select model", ["mlp", "resnet", "attention"], key="history_model_select")
        hist = load_history(sel)
        if hist:
            epochs = list(range(1, len(hist["train_loss"]) + 1))
            for metric, title in [("loss","Loss"), ("auroc","AUROC"), ("auprc","AUPRC")]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=epochs, y=hist[f"train_{metric}"], name=f"Train {title}", line=dict(width=2)))
                fig.add_trace(go.Scatter(x=epochs, y=hist[f"val_{metric}"],   name=f"Val {title}",   line=dict(width=2, dash="dash")))
                fig.update_layout(
                    title=f"{sel.upper()} — {title} per Epoch",
                    xaxis_title="Epoch", yaxis_title=title,
                    paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                    font_color="#fff",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No training history found for {sel}. Run train.py first.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — PAST PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Past Predictions":
    st.title("🗂️ Past Predictions")

    history = st.session_state.get("history", [])

    if not history:
        st.info("No predictions made yet in this session. Go to **🔬 Predict ADR** to get started.")
    else:
        df_hist = pd.DataFrame(history)

        # summary KPIs
        h1, h2, h3 = st.columns(3)
        h1.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_hist)}</div><div class="metric-label">Total Predictions</div></div>', unsafe_allow_html=True)
        h2.markdown(f'<div class="metric-card"><div class="metric-value">{(df_hist["Level"]=="High").sum()}</div><div class="metric-label">High Risk</div></div>', unsafe_allow_html=True)
        h3.markdown(f'<div class="metric-card"><div class="metric-value">{(df_hist["Level"]=="Low").sum()}</div><div class="metric-label">Low Risk</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # search / filter
        sc1, sc2 = st.columns(2)
        search_drug = sc1.text_input("🔍 Filter by drug", "")
        filter_risk = sc2.selectbox("Filter by risk level", ["All", "High", "Medium", "Low"])

        filtered = df_hist.copy()
        if search_drug:
            filtered = filtered[filtered["Drug"].str.contains(search_drug, case=False)]
        if filter_risk != "All":
            filtered = filtered[filtered["Level"] == filter_risk]

        st.dataframe(filtered, use_container_width=True, height=350)

        # risk distribution pie
        pie = px.pie(
            df_hist, names="Level",
            color="Level",
            color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"},
            title="Risk Level Distribution",
        )
        pie.update_layout(paper_bgcolor="#1e2130", font_color="#fff")
        st.plotly_chart(pie, use_container_width=True)

        # export
        csv = filtered.to_csv(index=False).encode()
        st.download_button("⬇️ Export to CSV", csv, "adr_predictions.csv", "text/csv")