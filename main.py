from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd 
from fastapi.middleware.cors import CORSMiddleware

from typing import List

app = FastAPI(title="ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ['http://localhost:3000']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("src\model_ml_10k-.pkl")
model_10k_plus = joblib.load("src/model_ml_10k+.pkl")

class InputData(BaseModel):
    CFA: float
    primary_scope: List[str]
    secondary_scope: List[str]
    discipline: List[str]

@app.get("/")
def root():
    return {"message": "ML API is running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([{
        "CFA (BUA)": data.CFA,
        "Primary scopes": ", ".join(data.primary_scope),
        "Secondary Scopes": ", ".join(data.secondary_scope),
        "Discipline": ", ".join(data.discipline)
    }])

    prediction_log = model.predict(input_df)
    prediction = np.expm1(prediction_log[0])  
    return {"prediction": round(float(prediction), 2)}

@app.post("/predict/10kplus")
def predict_10k_plus(data: InputData):
    input_df = pd.DataFrame([{
        "CFA (BUA)": data.CFA,
        "Primary scopes": ", ".join(data.primary_scope),
        "Secondary Scopes": ", ".join(data.secondary_scope),
        "Discipline": ", ".join(data.discipline)
    }])

    prediction_log = model_10k_plus.predict(input_df)
    prediction = np.expm1(prediction_log[0])
    return {
        "prediction": round(float(prediction), 2)
    }
