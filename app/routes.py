import pandas as pd

from fastapi import APIRouter, HTTPException

from app.schema import PredictionInput

from app.utils import (
    load_model,
    predict_output,
    load_label_encoder,
    decode_prediction
)

router = APIRouter()

model = load_model()

label_encoder = load_label_encoder()


@router.get("/")
def home():

    return {
        "message": "Welcome to Credit Risk Prediction API"
    }


@router.get("/health")
def health():

    return {
        "status": "API is running"
    }


@router.post("/predict")
def predict(data: PredictionInput):

    try:

        input_data = pd.DataFrame([{
            "enq_L3m": data.enq_L3m,
            "Age_Oldest_TL": data.Age_Oldest_TL,
            "num_times_delinquent": data.num_times_delinquent,
            "pct_PL_enq_L6m_of_ever": data.pct_PL_enq_L6m_of_ever,
            "num_std_6mts": data.num_std_6mts,
            "num_std": data.num_std,
            "num_deliq_12mts": data.num_deliq_12mts,
            "max_recent_level_of_deliq": data.max_recent_level_of_deliq,
            "Age_Newest_TL": data.Age_Newest_TL,
            "num_times_60p_dpd": data.num_times_60p_dpd,
            "tot_enq": data.tot_enq,
            "PL_enq_L6m": data.PL_enq_L6m,
            "time_since_recent_payment_missing": data.time_since_recent_payment_missing,
            "pct_tl_open_L12M": data.pct_tl_open_L12M,
            "Tot_TL_closed_L12M": data.Tot_TL_closed_L12M,
            "Home_TL": data.Home_TL,
            "pct_tl_open_L6M": data.pct_tl_open_L6M,
            "pct_closed_tl": data.pct_closed_tl
        }])

        prediction = predict_output(model, input_data)

        decoded_prediction = decode_prediction(
            label_encoder,
            prediction
        )

        return {
            "prediction": decoded_prediction[0]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )