import os
import json
import pandas as pd

import mlflow
import dagshub

from src.data_ingestion import load_config
from src.utils.logger import get_logger
from src.utils.exception import CustomException


mlflow.set_tracking_uri(
    "https://dagshub.com/sdntheone/Credit_Lending_Risk_Analysis.mlflow"
)

dagshub.init(
    repo_owner="sdntheone",
    repo_name="Credit_Lending_Risk_Analysis",
    mlflow=True
)

logger = get_logger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_PATH = os.path.join(BASE_DIR, "reports", "model_info.json")
DATA_PATH = os.path.join(BASE_DIR,"data","processed","final_data.csv")

def load_model_info():

    try:

        if not os.path.exists(REPORT_PATH):
            raise FileNotFoundError(f"{REPORT_PATH} not found")

        with open(REPORT_PATH, "r") as file:
            return json.load(file)

    except Exception as e:
        logger.error(f"Error in load_model_info: {e}")
        raise CustomException(e)


def load_model(model_uri):

    try:
        return mlflow.pyfunc.load_model(model_uri)

    except Exception as e:
        logger.error(f"Error in load_model: {e}")
        raise CustomException(e)


def load_sample_data():

    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"{DATA_PATH} not found")

        df = pd.read_csv(DATA_PATH)
        if df.empty:
            raise ValueError("Dataset is empty")

        target_col = load_config()["features"]["target_column"]
        if target_col not in df.columns:
            raise ValueError(f"{target_col} not found in dataset")

        X = df.drop(columns=[target_col])
        return X.iloc[[0]]

    except Exception as e:
        logger.error(f"Error in load_sample_data: {e}")
        raise CustomException(e)


def predict(model, data):

    try:
        return model.predict(data)

    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise CustomException(e)


def main():
    try:
        model_uri = load_model_info()["model_uri"]
        model = load_model(model_uri)
        sample_data = load_sample_data()
        print("\nSample Input:\n")
        print(sample_data)
        prediction = predict(model, sample_data)

        print("\nPrediction:\n")
        print(prediction)

        logger.info("Prediction completed successfully")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise CustomException(e)


if __name__ == "__main__":
    main()