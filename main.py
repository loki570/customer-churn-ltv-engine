from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
# Load the trained model
model = joblib.load("ltv_model.pkl")
# Create FastAPI app
app = FastAPI()

# ---------------- HOME API ----------------

@app.get("/")
def home():
    return {
        "message": "Customer Lifetime Value Prediction API is Running Successfully"
    }


# ---------------- INPUT SCHEMA ----------------
class Customer(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    TotalCharges: float
    Churn: int

# ---------------- SINGLE CUSTOMER PREDICTION ----------------

@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    prediction = model.predict(data)

    return {
        "Predicted_LTV": float(prediction[0])
    }


# ---------------- BATCH PREDICTION ----------------

@app.post("/batch_predict")
def batch_predict(customers: List[Customer]):

    data = pd.DataFrame(
        [customer.model_dump() for customer in customers]
    )

    predictions = model.predict(data)

    return {
        "Predicted_LTV": predictions.tolist()
    }