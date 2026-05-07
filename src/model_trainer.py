import pandas as pd
import os
import joblib
import json
import shutil

from collections import Counter

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from mlflow.models import infer_signature

from src.data_ingestion import load_config, load_params
from src.model_evaluation import evaluate
from src.utils.logger import get_logger
from src.utils.exception import CustomException

import mlflow
import mlflow.xgboost
import dagshub


mlflow.set_tracking_uri(
    "https://dagshub.com/sdntheone/Credit_Lending_Risk_Analysis.mlflow"
)

dagshub.init(
    repo_owner="sdntheone",
    repo_name="Credit_Lending_Risk_Analysis",
    mlflow=True
)

logger = get_logger(__name__)


def split_data(df, target_col, test_size, random_state):

    try:

        if target_col not in df.columns:
            raise ValueError(f"{target_col} not found in dataset")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.empty:
            raise ValueError("Feature set is empty before split")

        return train_test_split(X,y,test_size=test_size,stratify=y,random_state=random_state)

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

        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)

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


def save_model_info(run_id, model_uri):

    try:

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        model_info_path = os.path.join(
            BASE_DIR,
            "reports",
            "model_info.json"
        )

        os.makedirs(os.path.dirname(model_info_path), exist_ok=True)

        with open(model_info_path, "w") as f:

            json.dump({
                "run_id": run_id,
                "model_name": "CreditRiskModel",
                "model_uri": model_uri
            }, f, indent=4)

        logger.info(f"Model info saved at: {model_info_path}")

    except Exception as e:
        logger.error(f"Error in save_model_info: {e}")
        raise CustomException(e)


def main():

    try:

        config = load_config()
        params = load_params()

        target_col = config["features"]["target_column"]

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        data_path = os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "final_data.csv"
        )

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")

        df = pd.read_csv(data_path)

        if df.empty:
            raise ValueError("Loaded dataframe is empty")

        logger.info(f"Loaded data: {df.shape}")

        test_size = params["model"]["test_size"]
        random_state = params["model"]["random_state"]

        smote_params = params["smote"]

        X_train, X_test, y_train, y_test = split_data(df, target_col, test_size, random_state)

        X_train, y_train = apply_smote(X_train, y_train, smote_params)

        learning_rates = [0.05, 0.1]
        max_depths = [3, 5]
        n_estimators_list = [200, 300]

        best_model = None
        best_metrics = None
        best_f1 = 0
        best_run_id = None

        mlflow.set_experiment("Credit Lending Experiment v2")

        with mlflow.start_run(run_name="credit_risk_xgboost") as parent_run:

            mlflow.set_tags({
                "project": "Credit Lending Risk Analysis",
                "model_type": "XGBoost",
                "developer": "Sudhanshu",
                "tracking_server": "Dagshub",
                "stage": "training"
            })

            mlflow.log_params({
                "test_size": test_size,
                "random_state": random_state
            })

            mlflow.log_params(smote_params)

            run_number = 1

            for lr in learning_rates:

                for depth in max_depths:

                    for n_est in n_estimators_list:

                        xgb_params = {
                            "learning_rate": lr,
                            "max_depth": depth,
                            "n_estimators": n_est,
                            "eval_metric": params["xgboost"]["eval_metric"]
                        }

                        with mlflow.start_run(run_name=f"xgboost_run_{run_number}", nested=True) as child_run:

                            logger.info(f"Running Experiment {run_number}")

                            mlflow.log_params(xgb_params)

                            model = train_model(X_train, y_train, xgb_params)

                            metrics = evaluate(model, X_test, y_test)

                            mlflow.log_metrics(metrics)

                            signature = infer_signature(
                                X_train,
                                model.predict(X_train)
                            )

                            temp_model_dir = os.path.join(BASE_DIR, "temp_mlflow_model")

                            if os.path.exists(temp_model_dir):
                                shutil.rmtree(temp_model_dir)

                            mlflow.xgboost.save_model(
                                xgb_model=model,
                                path=temp_model_dir,
                                signature=signature
                            )

                            mlflow.xgboost.log_model(
                                xgb_model=model,
                                artifact_path="model",
                                signature=signature
                            )

                            mlflow.log_artifacts(
                                temp_model_dir,
                                artifact_path="model"
                            )

                            shutil.rmtree(temp_model_dir)

                            current_f1 = metrics["f1_weighted"]

                            logger.info(f"Run {run_number} F1 Weighted: {current_f1}")

                            if current_f1 > best_f1:

                                best_f1 = current_f1
                                best_model = model
                                best_metrics = metrics
                                best_run_id = child_run.info.run_id

                                logger.info(f"New Best Model Found -> F1: {best_f1}")

                            run_number += 1

            metrics_path = os.path.join(BASE_DIR, "reports", "metrics.json")

            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

            with open(metrics_path, "w") as f:
                json.dump(best_metrics, f, indent=4)

            save_model(best_model)

            model_uri = f"runs:/{best_run_id}/model"

            save_model_info(run_id=best_run_id, model_uri=model_uri)

            logger.info(f"Best Run ID: {best_run_id}")

            logger.info(f"Best F1 Weighted: {best_f1}")

            logger.info("Training pipeline completed")

    except Exception as e:

        logger.error(f"Error in main pipeline: {e}")

        raise CustomException(e)

if __name__ == "__main__":
    main()