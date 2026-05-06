import pandas as pd
import os
import joblib

from src.data_ingestion import load_config, load_params
from src.utils.logger import get_logger
from src.utils.exception import CustomException

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter

logger = get_logger(__name__)


def split_data(df, target_col, test_size, random_state):
    try:
        if target_col not in df.columns:
            raise ValueError(f"{target_col} not found in dataset")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.empty:
            raise ValueError("Feature set is empty before split")

        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

    except Exception as e:
        logger.error(f"Error in split_data: {e}")
        raise CustomException(e)


def apply_smote(X_train, y_train, params):
    try:
        counts = Counter(y_train)

        sampling_strategy = {
            0: int(counts.get(0, 0) * params["p0_multiplier"]),
            1: int(counts.get(1, 0) * params["p1_multiplier"]),
            2: int(counts.get(2, 0) * params["p2_multiplier"]),
            3: int(counts.get(3, 0) * params["p3_multiplier"])
        }

        logger.info(f"Sampling strategy: {sampling_strategy}")

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42
        )

        X_res, y_res = smote.fit_resample(X_train, y_train)

        return (
            pd.DataFrame(X_res, columns=X_train.columns),
            pd.Series(y_res)
        )

    except Exception as e:
        logger.error(f"Error in apply_smote: {e}")
        raise CustomException(e)


def train_model(X_train, y_train, xgb_params):
    try:
        if X_train.empty:
            raise ValueError("Training data is empty")

        model = XGBClassifier(
            learning_rate=xgb_params["learning_rate"],
            max_depth=xgb_params["max_depth"],
            n_estimators=xgb_params["n_estimators"],
            eval_metric=xgb_params["eval_metric"],
            random_state=42
        )

        weights = y_train.map({
            0: 1.2,
            1: 1,
            2: 2.2,
            3: 1.5
        })

        model.fit(X_train, y_train, sample_weight=weights)

        return model

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise CustomException(e)


def save_model(model):
    try:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(BASE_DIR, "artifacts", "model", "model.pkl")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

        logger.info(f"Model saved at: {path}")

    except Exception as e:
        logger.error(f"Error in save_model: {e}")
        raise CustomException(e)


def main():
    try:
        config = load_config()
        params = load_params()

        target_col = config["features"]["target_column"]

        test_size = params["model"]["test_size"]
        random_state = params["model"]["random_state"]

        xgb_params = params["xgboost"]
        smote_params = params["smote"]

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(BASE_DIR, "data", "processed", "final_data.csv")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")

        df = pd.read_csv(data_path)

        if df.empty:
            raise ValueError("Loaded dataframe is empty")

        logger.info(f"Loaded data: {df.shape}")

        X_train, X_test, y_train, y_test = split_data(
            df, target_col, test_size, random_state
        )

        X_train, y_train = apply_smote(X_train, y_train, smote_params)

        model = train_model(X_train, y_train, xgb_params)

        save_model(model)

        logger.info("Training pipeline completed")

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise CustomException(e)


if __name__ == "__main__":
    main()