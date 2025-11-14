import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow
import joblib
import os

def load_data():
    return pd.DataFrame({
        'open':[100,102,101,103,104],
        'close':[101,103,102,104,105]
    })

def train():
    mlflow.set_tracking_uri("file:./mlflow")
    mlflow.set_experiment("finance-model")

    df = load_data()
    X = df[['open']]
    y = df['close']

    model = LinearRegression()

    with mlflow.start_run():
        model.fit(X, y)
        mlflow.log_param("model", "LinearRegression")

        # Ensure models/ exists even if running from src/
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_dir = os.path.abspath(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)

        # log into mlflow as usual
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train()
