import pandas as pd
import numpy as np
import os

from xgboost import XGBClassifier
from src.data_ingestion import load_config,load_params
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


def model_feature_selection(X, y, imp_threshold):
    try:
        if X.empty:
            raise ValueError("Input features are empty")

        model = XGBClassifier(eval_metric="mlogloss")
        model.fit(X, y)

        importance = model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        }).sort_values(by="importance", ascending=False)

        selected_features = feature_importance_df[
            feature_importance_df["importance"] > imp_threshold
        ]["feature"].tolist()

        if not selected_features:
            logger.warning("No features selected, using top features")
            selected_features = feature_importance_df.head(20)["feature"].tolist()

        return selected_features

    except Exception as e:
        logger.error(f"Error in model_feature_selection: {e}")
        raise CustomException(e)


def correlation_filter(X, corr_threshold):
    try:
        if X.empty:
            raise ValueError("Feature set is empty before correlation filtering")

        X_numeric = X.select_dtypes(include=[np.number])

        if X_numeric.shape[1] == 0:
            return X

        corr_matrix = X_numeric.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]

        if to_drop:
            logger.info(f"Dropping correlated features: {to_drop}")

        return X.drop(columns=to_drop, errors="ignore")

    except Exception as e:
        logger.error(f"Error in correlation_filter: {e}")
        raise CustomException(e)


def feature_refinement_pipeline():
    try:
        config = load_config()
        params=load_params()
        imp_threshold=params['feature_refinement']['importance_threshold']
        corr_threshold=params['feature_refinement']['correlation_threshold']


        target_col = config["features"]["target_column"]

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(BASE_DIR, "data", "processed", "selected_data.csv")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")

        df = pd.read_csv(data_path)

        if df.empty:
            raise ValueError("Loaded dataframe is empty")

        df = df.drop(columns=["PROSPECTID"], errors="ignore")

        if target_col not in df.columns:
            raise ValueError(f"{target_col} not found in dataset")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.empty:
            raise ValueError("No features available after split")

        X = correlation_filter(X, corr_threshold)

        if X.shape[1] == 0:
            raise ValueError("All features removed after correlation filtering")

        selected_features = model_feature_selection(X, y, imp_threshold)

        if not selected_features:
            raise ValueError("No features selected")

        df_final = X[selected_features].copy()
        df_final[target_col] = y

        if df_final.empty:
            raise ValueError("Final dataset is empty")

        save_path = os.path.join(BASE_DIR, "data", "processed", "final_data.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        df_final.to_csv(save_path, index=False)
        logger.info(f"Final data saved at: {save_path}")

        return df_final, selected_features

    except Exception as e:
        logger.error(f"Error in feature_refinement_pipeline: {e}")
        raise CustomException(e)


if __name__ == "__main__":
    try:
        df_final, features = feature_refinement_pipeline()
        logger.info(f"Pipeline completed. Features: {features}")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise