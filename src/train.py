import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

def train():

    # Load dataset
    df = pd.read_csv("data/processed/processed.csv")

    # Basic features + label
    X = df[["SMA_10","SMA_50","EMA_12","EMA_26","MACD","RSI","vol_10"]]
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)  # next-day direction

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Validation Accuracy:", acc)

    # ───────────────────────────────────────────
    # MLflow logs
    # ───────────────────────────────────────────
    mlflow.set_experiment("finance-model")
    with mlflow.start_run():
        mlflow.log_metric("val_accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    # ───────────────────────────────────────────
    # SAVE MODEL LOCALLY (FIX)
    # ───────────────────────────────────────────

    # Make sure models/ folder exists
    Path("models").mkdir(parents=True, exist_ok=True)

    # Save model inside the folder
    joblib.dump(model, "models/model.pkl")

    print("Model saved → models/model.pkl")

if __name__ == "__main__":
    train()
