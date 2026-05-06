import pandas as pd
import os
import joblib

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from src.data_ingestion import load_config
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


def load_model():
    try:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(BASE_DIR, "artifacts", "model", "model.pkl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")

        return joblib.load(path)

    except Exception as e:
        logger.error(f"Error in load_model: {e}")
        raise CustomException(e)


def load_data():
    try:
        config = load_config()
        target_col = config["features"]["target_column"]

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(BASE_DIR, "data", "processed", "final_data.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("Loaded dataframe is empty")

        if target_col not in df.columns:
            raise ValueError(f"{target_col} not found in dataset")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        return X, y

    except Exception as e:
        logger.error(f"Error in load_data: {e}")
        raise CustomException(e)


def evaluate(model, X, y):
    try:
        if X.empty:
            raise ValueError("Evaluation data is empty")

        y_pred = model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        f1_weighted = f1_score(y, y_pred, average='weighted')
        f1_macro = f1_score(y, y_pred, average='macro')

        precision, recall, f1_vals, _ = precision_recall_fscore_support(y, y_pred)

        class_names = ["P1", "P2", "P3", "P4"]

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1 Weighted: {f1_weighted}")
        logger.info(f"F1 Macro: {f1_macro}")

        for i in range(len(f1_vals)):
            logger.info(
                f"{class_names[i]} -> Precision: {precision[i]}, Recall: {recall[i]}, F1: {f1_vals[i]}"
            )

    except Exception as e:
        logger.error(f"Error in evaluate: {e}")
        raise CustomException(e)


def main():
    try:
        model = load_model()
        X, y = load_data()
        evaluate(model, X, y)
        logger.info("Evaluation completed")

    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {e}")
        raise CustomException(e)


if __name__ == "__main__":
    main()