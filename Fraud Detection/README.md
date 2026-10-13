# Credit Card Fraud Detector

Streamlit web app that flags suspicious credit card transactions using
Logistic Regression + Isolation Forest (hybrid), trained on the Kaggle ULB
Credit Card Fraud dataset (284,807 transactions, 492 frauds).

## Files
| File | Purpose |
|---|---|
| `app.py` | The web app (Streamlit) |
| `model.joblib` | Final saved model (scaler + both models + example rows + metrics) |
| `model_info.json` | Record of how the model was built (versions, features, test metrics) |
| `train_model.py` | Re-trains and re-saves the model from `creditcard.csv` |
| `requirements.txt` | Pinned library versions (must match the saved model) |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Re-train (optional)
Place `creditcard.csv` in this folder, then:
```
python train_model.py --csv creditcard.csv
```

## Deploy (Streamlit Community Cloud, free)
1. Push this folder to a public GitHub repo (without creditcard.csv).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create app -> pick the repo, branch `main`, main file `app.py`.
4. Advanced settings -> Python 3.12 (or 3.13) -> Deploy.
