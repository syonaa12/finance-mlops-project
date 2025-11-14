import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import joblib

def load_data():
    return pd.DataFrame({
        'open': [100, 102, 101, 103, 104],
        'close': [101, 103, 102, 104, 105]
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

        # Add input_example
        input_example = {"open": 105}

        # Log model with signature
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=input_example
        )

        joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    train()
