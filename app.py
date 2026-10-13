"""
Step 2 - the web app.   Run:  streamlit run app.py
Needs model.joblib (created by train_model.py).
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="🛡️", layout="wide")

RAW_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_bundle():
    model_path = BASE_DIR / "model.joblib"
    if not model_path.exists():
        model_path = Path("model.joblib")
    return joblib.load(model_path)


try:
    B = load_bundle()
except FileNotFoundError:
    st.error("model.joblib not found. Run `python train_model.py --csv creditcard.csv` first.")
    st.stop()


def score(raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Raw transactions (Time, V1-V28, Amount) -> predictions from both models + hybrid verdict."""
    X = raw[RAW_COLS].copy()
    X[["scaled_time", "scaled_amount"]] = B["scaler"].transform(X[["Time", "Amount"]])
    X = X.drop(columns=["Time", "Amount"])[B["feature_cols"]]
    prob = B["lr"].predict_proba(X)[:, 1]
    iso_flag = (B["iso"].predict(X) == -1)
    out = raw.copy()
    out["fraud_probability"] = prob.round(4)
    out["lr_flag"] = prob >= threshold
    out["anomaly_flag"] = iso_flag
    out["anomaly_score"] = (-B["iso"].decision_function(X)).round(4)
    out["verdict"] = np.where(out["lr_flag"] | out["anomaly_flag"], "⚠️ SUSPICIOUS", "✅ Legitimate")
    return out


# ---------------- sidebar ----------------
st.sidebar.title("🛡️ Fraud Detector")
st.sidebar.caption("Logistic Regression + Isolation Forest (hybrid)")
threshold = st.sidebar.slider("Fraud probability threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Lower = catches more fraud but raises more false alarms.")
st.sidebar.markdown(f"Trained on **{B['n_train']:,}** transactions, tested on **{B['n_test']:,}**.")

tab1, tab2, tab3 = st.tabs(["🔍 Check a transaction", "📂 Batch upload", "📊 Model performance"])

# ---------------- tab 1 ----------------
with tab1:
    st.subheader("Check a single transaction")
    st.caption("Load a real example from the test set, or type your own values.")

    def load_example(kind):
        r = B["examples"][kind].sample(1).iloc[0]
        for c in RAW_COLS:
            st.session_state[f"in_{c}"] = float(r[c])

    for c in RAW_COLS:
        st.session_state.setdefault(f"in_{c}", 0.0)

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Load legitimate example", on_click=load_example, args=("legit",))
    b2.button("Load fraud example", on_click=load_example, args=("fraud",))

    c1, c2 = st.columns(2)
    c1.number_input("Amount ($)", key="in_Amount", min_value=0.0, format="%.2f")
    c2.number_input("Time (seconds since first transaction)", key="in_Time", min_value=0.0)
    with st.expander("Anonymized PCA features V1–V28"):
        cols = st.columns(4)
        for i in range(1, 29):
            cols[(i - 1) % 4].number_input(f"V{i}", key=f"in_V{i}", format="%.4f")

    if st.button("Analyze transaction", type="primary"):
        row = pd.DataFrame([{c: st.session_state[f"in_{c}"] for c in RAW_COLS}])
        res = score(row, threshold).iloc[0]
        if res["verdict"].startswith("⚠️"):
            st.error(f"{res['verdict']} — this transaction should be reviewed.")
        else:
            st.success(f"{res['verdict']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud probability (Logistic Regression)", f"{res['fraud_probability']:.1%}")
        m2.metric("Anomaly score (Isolation Forest)", f"{res['anomaly_score']:.3f}")
        m3.metric("Flagged by", ", ".join([n for n, f in [("Logistic Regression", res["lr_flag"]),
                  ("Isolation Forest", res["anomaly_flag"])] if f]) or "neither model")

# ---------------- tab 2 ----------------
with tab2:
    st.subheader("Score many transactions at once")
    st.caption("Upload a CSV with columns: Time, V1–V28, Amount (a Class column is ignored).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            res = score(data, threshold)
            n_flag = int((res["verdict"] != "✅ Legitimate").sum())
            a, b, c = st.columns(3)
            a.metric("Transactions", f"{len(res):,}")
            b.metric("Flagged suspicious", f"{n_flag:,}")
            c.metric("Flag rate", f"{n_flag / len(res):.2%}")
            show = res.sort_values("fraud_probability", ascending=False)
            st.dataframe(show[["Time", "Amount", "fraud_probability", "anomaly_score", "verdict"]],
                         width="stretch", height=380)
            st.download_button("Download results (CSV)", show.to_csv(index=False).encode(),
                               "fraud_results.csv", "text/csv")

# ---------------- tab 3 ----------------
with tab3:
    st.subheader("How well do the models perform?")
    st.caption("Evaluated on the 20% held-out test set.")
    mt = pd.DataFrame([{k: v for k, v in m.items() if k != "cm"} for m in B["metrics"]]).set_index("Model")
    st.dataframe(mt.style.format("{:.4f}", na_rep="—"), width="stretch")
    st.info("Accuracy looks high even for a bad model because only ~0.17% of transactions are fraud. "
            "Judge the models by Recall, Precision and F1.")
    st.markdown("**Confusion matrices**")
    cols = st.columns(3)
    for col, m in zip(cols, B["metrics"]):
        fig, ax = plt.subplots(figsize=(3.2, 3))
        cm = np.array(m["cm"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Legit", "Fraud"]); ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(m["Model"], fontsize=10)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        fig.tight_layout(); col.pyplot(fig); plt.close(fig)
