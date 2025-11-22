# app.py
# Streamlit app - polished layout and controls

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(page_title="Claim Risk Scanner", layout="wide", initial_sidebar_state="expanded")

# header / style
st.markdown("""
    <style>
    .stApp { background-color: #0f1724; color: #e6eef8; }
    .header { text-align: center; color: #cfe8ff; font-size: 24px; margin-bottom: 6px; }
    .sub { color: #9fb6d8; margin-bottom: 20px; }
    .card { background-color:#0b1220; padding:12px; border-radius:8px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>Insurance Claim Risk Scanner</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Upload dataset → inspect → detect suspicious claims → download results</div>", unsafe_allow_html=True)

# load artifacts if present
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    artifacts_ready = True
except Exception:
    model = None
    scaler = None
    artifacts_ready = False

uploaded = st.file_uploader("Upload CSV (insurance_claims_synthetic.csv)", type=["csv"])

# sidebar controls
st.sidebar.header("Detection Options")
contam = st.sidebar.slider("IsolationForest contamination (for retrain)", 0.005, 0.1, 0.03, 0.005)
top_pct = st.sidebar.slider("Top % to flag by score (fallback)", 0.001, 0.05, 0.015, 0.001)
rule_boost_on = st.sidebar.checkbox("Apply rule boosts (claim_ratio, past_claims)", True)
show_top = st.sidebar.number_input("Show top N anomalies", 5, 50, 10)

if uploaded:
    df = pd.read_csv(uploaded)
    st.markdown("### Preview")
    st.dataframe(df.head(8))

    # ensure columns exist
    expected = ['age','vehicle_price','annual_premium','claim_amount','accident_severity','past_claims','policy_tenure_months']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()

    # quick copy for processing
    proc = df.copy()
    # feature engineering same as in notebook
    proc['claim_to_vehicle_ratio'] = proc['claim_amount'] / (proc['vehicle_price'] + 1)
    proc['premium_to_vehicle_ratio'] = proc['annual_premium'] / (proc['vehicle_price'] + 1)
    proc['past_claims_flag'] = (proc['past_claims'] >= 2).astype(int)
    proc['severity_norm'] = proc['accident_severity'] / (proc['accident_severity'].max() + 1)

    feat_cols = [
        'age','vehicle_price','annual_premium','claim_amount',
        'claim_to_vehicle_ratio','premium_to_vehicle_ratio',
        'past_claims','past_claims_flag','policy_tenure_months','severity_norm'
    ]
    X = proc[feat_cols].copy()

    # scale using saved scaler if present else local fit
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xs = sc.fit_transform(X)

    # model predict: if saved model present use it else create local IsolationForest with slider param
    if model is not None:
        iso = model
    else:
        iso = __import__('sklearn.ensemble', fromlist=['IsolationForest']).IsolationForest(
            n_estimators=200, contamination=contam, random_state=101
        )
        iso.fit(Xs)

    raw = iso.decision_function(Xs)
    anom_raw = -raw
    # normalize
    a_min, a_max = anom_raw.min(), anom_raw.max()
    anom_norm = (anom_raw - a_min) / (a_max - a_min + 1e-9)

    # rule boost
    rule_boost = np.zeros(len(proc))
    if rule_boost_on:
        rule_boost = (
            ((X['claim_to_vehicle_ratio'] > 0.2).astype(int) * 0.25) +
            ((X['past_claims'] >= 3).astype(int) * 0.20) +
            ((X['claim_amount'] > X['annual_premium'] * 3).astype(int) * 0.15)
        )

    combined = 0.75 * anom_norm + 0.25 * (rule_boost.clip(0,1))
    combined = np.clip(combined, 0, 1)

    # threshold: use top_pct OR absolute
    thr = max(np.quantile(combined, 1 - top_pct), 0.62)
    flags = (combined >= thr).astype(int)
    labels = np.where(flags==1, "Suspicious", "Normal")

    proc['anomaly_score'] = combined.round(4)
    proc['anomaly_flag'] = flags
    proc['anomaly_label'] = labels

    # highlight stats
    normal_count = int((proc['anomaly_flag'] == 0).sum())
    suspicious_count = int((proc['anomaly_flag'] == 1).sum())

    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("Total claims", len(proc))
    c2.metric("Normal", normal_count)
    c3.metric("Suspicious", suspicious_count)

    # top anomalies table
    st.markdown("### Top flagged claims")
    top_df = proc.sort_values("anomaly_score", ascending=False).head(show_top)
    st.dataframe(top_df[[
        'policy_id','age','vehicle_price','annual_premium','claim_amount',
        'past_claims','anomaly_score','anomaly_label'
    ]])

    # charts area
    st.markdown("### Visuals")
    left, right = st.columns([2,1])

    with left:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.scatterplot(data=proc, x='annual_premium', y='claim_amount', hue='anomaly_label', palette=['#2ecc71','#e74c3c'], alpha=0.8, ax=ax)
        ax.set_title("Claim Amount vs Premium")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(8,2.2))
        sns.histplot(proc['anomaly_score'], bins=40, ax=ax2, color='#3498db')
        ax2.set_title("Anomaly Score Distribution")
        st.pyplot(fig2)

    with right:
        st.markdown("#### Risk breakdown")
        st.bar_chart(proc['anomaly_label'].value_counts())

        # small diagnostics
        st.markdown("#### Quick diagnostics")
        st.write(proc[['anomaly_score','claim_to_vehicle_ratio','past_claims']].describe().T)

    # download
    csv = proc.to_csv(index=False).encode()
    st.download_button("Download results CSV", csv, "claims_analyzed.csv", "text/csv")

    # optional: save processed to file server
    st.markdown("---")
    if st.button("Save processed_output.csv (server)"):
        proc.to_csv("processed_output.csv", index=False)
        st.success("Saved processed_output.csv")

else:
    st.info("Upload the dataset to begin. If you ran notebook_preprocess_and_model.py earlier, model.pkl and scaler.pkl will be used automatically.")
