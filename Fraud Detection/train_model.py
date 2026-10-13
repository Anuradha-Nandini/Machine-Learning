"""
Step 1 - train the models and save them for the web app.
Run:  python train_model.py --csv creditcard.csv
Creates: model.joblib
"""
import argparse
import json
import platform
import datetime
import sklearn
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

SEED = 42
RAW_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def main(csv):
    df = pd.read_csv(csv).drop_duplicates()
    X, y = df[RAW_COLS], df["Class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    scaler = StandardScaler().fit(X_tr[["Time", "Amount"]])

    def prep(raw):
        out = raw.copy()
        out[["scaled_time", "scaled_amount"]] = scaler.transform(raw[["Time", "Amount"]])
        return out.drop(columns=["Time", "Amount"])

    Xtr, Xte = prep(X_tr), prep(X_te)
    feature_cols = list(Xtr.columns)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(Xtr, y_tr)
    iso = IsolationForest(n_estimators=100, contamination=float(y_tr.mean()),
                          random_state=SEED, n_jobs=-1).fit(Xtr)

    lr_prob = lr.predict_proba(Xte)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    iso_pred = (iso.predict(Xte) == -1).astype(int)
    iso_score = -iso.decision_function(Xte)
    hyb_pred = ((lr_pred == 1) | (iso_pred == 1)).astype(int)

    def row(name, p, s=None):
        return {"Model": name, "Accuracy": accuracy_score(y_te, p),
                "Precision": precision_score(y_te, p, zero_division=0),
                "Recall": recall_score(y_te, p), "F1-score": f1_score(y_te, p),
                "ROC-AUC": roc_auc_score(y_te, s) if s is not None else np.nan,
                "cm": confusion_matrix(y_te, p).tolist()}

    metrics = [row("Logistic Regression", lr_pred, lr_prob),
               row("Isolation Forest", iso_pred, iso_score),
               row("Hybrid (LR or IF)", hyb_pred)]

    # real example rows from the TEST set so the app can demo them
    rng = np.random.default_rng(SEED)
    pick = lambda c, k: X_te[y_te == c].sample(min(k, int((y_te == c).sum())), random_state=SEED)
    examples = {"legit": pick(0, 50), "fraud": pick(1, 50)}

    joblib.dump({"scaler": scaler, "lr": lr, "iso": iso, "feature_cols": feature_cols,
                 "metrics": metrics, "examples": examples,
                 "n_train": len(X_tr), "n_test": len(X_te)}, "model.joblib")
    # Step 10: save the final model + a human-readable record of how it was made
    info = {"saved_on": datetime.date.today().isoformat(), "python": platform.python_version(),
            "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "features": feature_cols, "target": "Class (0 = legitimate, 1 = fraud)",
            "split": "80/20 stratified, random_state=42",
            "models": {"Logistic Regression": "class_weight='balanced', max_iter=1000",
                       "Isolation Forest": "n_estimators=100, contamination=train fraud rate"},
            "test_metrics": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in m.items()}
                             for m in metrics]}
    json.dump(info, open("model_info.json", "w"), indent=2)
    print("Saved model.joblib and model_info.json")
    for m in metrics:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "cm"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="creditcard.csv")
    main(ap.parse_args().csv)
