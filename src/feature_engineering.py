from src.data_transformation import transformed_data
from src.data_ingestion import load_config
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import os
import pandas as pd

logger = get_logger(__name__)



def encode_categorical(df, categorical_cols):
    try:
        logger.info("Encoding categorical features")

        categorical_cols = [col for col in categorical_cols if col in df.columns]

        encoder = OrdinalEncoder()
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

        return df, categorical_cols

    except Exception as e:
        logger.error(f"Error in encode_categorical: {e}")
        raise CustomException(e)


def encode_target(df, target_col):
    try:
        logger.info("Encoding target column")

        lb = LabelEncoder()
        df[target_col] = lb.fit_transform(df[target_col])

        return df

    except Exception as e:
        logger.error(f"Error in encode_target: {e}")
        raise CustomException(e)


def encoded_data():
    try:
        logger.info("Starting encoding pipeline")
        config = load_config()

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        target_col = config["features"]["target_column"]
        categorical_cols = config["features"]["categorical_cols"]

        data_path = os.path.join(BASE_DIR, "data", "processed", "merged_data.csv")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded merged data: {df.shape}")

        df, categorical_cols = encode_categorical(df, categorical_cols)
        df = encode_target(df, target_col)
        cat_indices = [df.columns.get_loc(col) for col in categorical_cols]
        logger.info("Encoding completed")

        save_path = os.path.join(BASE_DIR, "data", "processed", "encoded_data.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path,index=False)
        logger.info(f"Encoded data saved at: {save_path}")

        return df, cat_indices

    except Exception as e:
        logger.error(f"Error in encoded_data pipeline: {e}")
        raise CustomException(e)


if __name__ == "__main__":
    try:
        df, cat_indices = encoded_data()
        logger.info(f"Final data shape: {df.shape}")
        logger.info(f"Categorical indices: {cat_indices}")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise