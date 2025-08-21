import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import tensorflow as tf

st.set_page_config(page_title="Credit Card Fraud Ensemble", layout="wide")
ART_DIR = Path("artifacts")

@st.cache_resource
def load_artifacts():
    with open(ART_DIR / "config.json", "r") as f:
        cfg = json.load(f)
    feature_order = cfg["feature_order"]
    threshold = float(cfg.get("decision_threshold", 0.5))

    # Load scalers
    scaler_amount = joblib.load(ART_DIR / "scaler_amount.joblib")
    scaler_time = joblib.load(ART_DIR / "scaler_time.joblib")

    # Load models
    rf = joblib.load(ART_DIR / "base_rf.joblib")
    xgb_model = joblib.load(ART_DIR / "base_xgb.joblib")
    mlp = tf.keras.models.load_model(ART_DIR / "base_mlp.keras")

    return cfg, feature_order, threshold, scaler_amount, scaler_time, rf, xgb_model, mlp

cfg, feature_order, threshold, scaler_amount, scaler_time, rf, xgb_model, mlp = load_artifacts()

st.title("Credit Card Fraud, Soft Voting Ensemble")
st.caption("Models, rf, xgb_model, mlp. Ensemble, average of their fraud probabilities.")

def ensure_scaled(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_scaled = "Scaled_Amount" in df.columns and "Scaled_Time" in df.columns
    has_raw = "Amount" in df.columns and "Time" in df.columns

    if not has_scaled and not has_raw:
        raise ValueError("Provide either Scaled_Amount and Scaled_Time or raw Amount and Time")

    # Compute scaled if only raw provided
    if not has_scaled and has_raw:
        df["Scaled_Amount"] = scaler_amount.transform(df[["Amount"]])
        df["Scaled_Time"] = scaler_time.transform(df[["Time"]])

    # Drop raw if present
    for col in ["Amount", "Time"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Reorder columns to match training
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}")

    return df[feature_order]

def predict_df(df_raw: pd.DataFrame, threshold: float):
    X = ensure_scaled(df_raw)

    # Base model probabilities for positive class
    proba_rf = rf.predict_proba(X)[:, 1]
    proba_xgb = xgb_model.predict_proba(X)[:, 1]
    proba_mlp = mlp.predict(X, verbose=0).ravel()

    # Hybrid soft voting
    hybrid_proba = (proba_rf + proba_xgb + proba_mlp) / 3.0
    hybrid_pred = (hybrid_proba >= threshold).astype(int)

    out = df_raw.copy()
    out["prob_rf"] = proba_rf
    out["prob_xgb"] = proba_xgb
    out["prob_mlp"] = proba_mlp
    out["prob_hybrid"] = hybrid_proba
    out["pred_hybrid"] = hybrid_pred
    return out

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", 0.01, 0.99, float(threshold), 0.01)
    st.write("Base models:", ", ".join(cfg["base_model_order"]))
    st.write("Expected features:", ", ".join(feature_order))

st.subheader("Batch prediction, upload CSV")
st.caption("CSV should include V1 to V28, and either Amount and Time or Scaled_Amount and Scaled_Time")
file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    try:
        df_in = pd.read_csv(file)
        with st.spinner("Predicting"):
            res = predict_df(df_in, threshold)
        st.success(f"Predicted {len(res)} rows")
        st.dataframe(res.head(50))
        st.download_button("Download predictions", res.to_csv(index=False).encode(), "predictions.csv")
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.subheader("Single transaction form")
with st.form("single"):
    cols = st.columns(3)
    values = {}
    # Ask for V1 to V28
    for i in range(1, 29):
        with cols[(i - 1) % 3]:
            values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01)
    with cols[0]:
        values["Amount"] = st.number_input("Amount", value=0.0, step=0.01)
    with cols[1]:
        values["Time"] = st.number_input("Time", value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict")
    if submitted:
        one = pd.DataFrame([values])
        try:
            res = predict_df(one, threshold)
            st.metric("Hybrid probability of fraud", f"{float(res.loc[0, 'prob_hybrid']):.4f}")
            st.write("Prediction label, 1 means fraud, 0 means genuine:", int(res.loc[0, "pred_hybrid"]))
            with st.expander("Base model probabilities"):
                st.json({
                    "rf": float(res.loc[0, "prob_rf"]),
                    "xgb": float(res.loc[0, "prob_xgb"]),
                    "mlp": float(res.loc[0, "prob_mlp"]),
                })
        except Exception as e:
            st.error(str(e))
