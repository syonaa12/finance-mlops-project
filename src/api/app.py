from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('models/model.pkl')

@app.get('/predict')
def predict(open_price: float):
    pred = model.predict([[open_price]])[0]
    return {'prediction': float(pred)}
