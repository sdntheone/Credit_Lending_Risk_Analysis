from src.data_ingestion import load_config
from src.data_ingestion import load_params
from src.utils.logger import get_logger
from src.utils.exception import CustomException

import pandas as pd
import numpy as np
import os

from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = get_logger(__name__)


def chi_square_selection(df, categorical_cols, target_col, p_threshold):
    try:
        selected_cols = []

        for col in categorical_cols:
            if col not in df.columns:
                continue

            _, pval, _, _ = chi2_contingency(
                pd.crosstab(df[col], df[target_col])
            )

            logger.info(f"{col} p-value: {pval}")

            if pval <= p_threshold:
                selected_cols.append(col)

        return selected_cols

    except Exception as e:
        logger.error(f"Error in chi_square_selection: {e}")
        raise CustomException(e)


def remove_high_vif(X, vif_threshold):
    try:
        X = X.copy()

        while True:
            vif_values = [
                variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])
            ]

            vif_df = pd.DataFrame({
                "feature": X.columns,
                "vif": vif_values
            }).sort_values(by="vif", ascending=False)

            max_vif = vif_df["vif"].iloc[0]

            if max_vif < vif_threshold:
                break

            drop_col = vif_df["feature"].iloc[0]
            logger.info(f"Dropping {drop_col} (VIF={max_vif})")

            X.drop(columns=[drop_col], inplace=True)

        return X.columns.tolist()

    except Exception as e:
        logger.error(f"Error in remove_high_vif: {e}")
        raise CustomException(e)


def anova_selection(df, numerical_cols, target_col, p_threshold):
    try:
        selected_cols = []

        for col in numerical_cols:
            if col not in df.columns:
                continue

            groups = [
                df[df[target_col] == cls][col].values
                for cls in df[target_col].unique()
            ]

            if len(groups) < 2:
                continue

            _, pval = f_oneway(*groups)

            if pval <= p_threshold:
                selected_cols.append(col)

        return selected_cols

    except Exception as e:
        logger.error(f"Error in anova_selection: {e}")
        raise CustomException(e)


def save_dataframe(df, filename):
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(base_dir, "data", "processed", filename)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

        logger.info(f"Saved file at: {path}")

    except Exception as e:
        logger.error(f"Error in save_dataframe: {e}")
        raise CustomException(e)


def feature_selection_pipeline():
    try:
        logger.info("Starting feature selection pipeline")

        config = load_config()
        target_col = config["features"]["target_column"]
        categorical_cols = config["features"]["categorical_cols"]

        params=load_params()
        p_threshold=params['feature_selection']['p_value_threshold']
        vif_threshold=params['feature_selection']['vif_threshold']
        logger.info(f"p_threshold: {p_threshold}, vif_threshold: {vif_threshold}")


        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(base_dir, "data", "processed", "encoded_data.csv")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded encoded data: {df.shape}")

        selected_cat_cols = chi_square_selection(df, categorical_cols, target_col,p_threshold)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        cols_after_vif = remove_high_vif(df[numeric_cols],vif_threshold)

        selected_num_cols = anova_selection(df, cols_after_vif, target_col,p_threshold)

        final_features = selected_num_cols + selected_cat_cols

        if not final_features:
            raise ValueError("No features selected after feature selection")

        logger.info(f"Selected features count: {len(final_features)}")

        df_selected = df[final_features + [target_col]].copy()

        save_dataframe(df_selected, "selected_data.csv")

        logger.info(f"Final shape: {df_selected.shape}")

        return df_selected, final_features

    except Exception as e:
        logger.error(f"Error in feature_selection_pipeline: {e}")
        raise CustomException(e)


if __name__ == "__main__":
    try:
        df_selected, features = feature_selection_pipeline()
        logger.info(f"Pipeline completed. Features: {features}")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise