from src.data_ingestion import load_raw_data
from src.data_ingestion import load_config, load_params
import pandas as pd
import numpy as np
import os
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)

config = load_config()


def clean_df1(df1):
    try:
        logger.info("Cleaning df1: removing invalid Age_Oldest_TL values")
        df1 = df1[df1['Age_Oldest_TL'] != -99999]
        return df1

    except Exception as e:
        logger.error(f"Error in clean_df1: {e}")
        raise CustomException(e)


def clean_df2(df2):
    try:
        logger.info("Cleaning df2")

        columns_to_removed = []
        columns_treated = []

        for col in df2.columns:
            count = (df2[col] == -99999).sum()

            if count > 10000:
                columns_to_removed.append(col)
            elif count > 0:
                columns_treated.append(col)

        df2 = df2.drop(columns=columns_to_removed)

        logger.info(f"Removed columns: {columns_to_removed}")
        logger.info(f"Columns to treat: {columns_treated}")

        return df2, columns_treated

    except Exception as e:
        logger.error(f"Error in clean_df2: {e}")
        raise CustomException(e)


def treat_columns(df, columns_treated):
    try:
        logger.info("Treating missing values")

        time_cols = config['features']['time_cols']
        ratio_cols = config['features']['ratio_cols']

        count_cols = [col for col in columns_treated if col not in time_cols + ratio_cols]

        df[columns_treated] = df[columns_treated].replace(-99999, np.nan)

        for col in time_cols:
            if col in df.columns:
                df[col + "_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(df[col].max() + 1)

        df[count_cols] = df[count_cols].fillna(0)

        for col in ratio_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error in treat_columns: {e}")
        raise CustomException(e)


def merge_dataframe(df1, df2):
    try:
        logger.info("Merging df1 and df2")
        return pd.merge(df1, df2, how='inner', on='PROSPECTID')

    except Exception as e:
        logger.error(f"Error in merge_dataframe: {e}")
        raise CustomException(e)


def transformed_data():
    try:
        logger.info("Starting data transformation pipeline")

        df1, df2 = load_raw_data()
        logger.info(f"Loaded df1: {df1.shape}, df2: {df2.shape}")

        df1 = clean_df1(df1)
        df2, columns_treated = clean_df2(df2)
        df2 = treat_columns(df2, columns_treated)

        df = merge_dataframe(df1, df2)
        logger.info(f"Merged data shape: {df.shape}")

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        save_path = os.path.join(BASE_DIR, "data", "processed", "merged_data.csv")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        logger.info(f"Data saved at: {save_path}")

        return df

    except Exception as e:
        logger.error(f"Error in transformation pipeline: {e}")
        raise CustomException(e)


if __name__ == "__main__":
    transformed_data()