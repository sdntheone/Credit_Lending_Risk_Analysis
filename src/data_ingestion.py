import pandas as pd
import os
import yaml
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


def load_params():
    try:
        logger.info("loading params.yaml file")
        with open("params.yaml", "r") as f:
            params= yaml.safe_load(f)
        logger.info("params.yaml loaded successfully")
        return params

    except FileNotFoundError as e:
        logger.error("params.yaml file not found")
        raise CustomException(e)
    
    except Exception as e:
        logger.error(f"erros loading parmas.yaml file: {str(e)}")
        raise CustomException(e)


def load_config(config_path="config/config.yaml"):
    try:
        logger.info("loading config.yaml file")
        with open(config_path, "r") as f:
            config= yaml.safe_load(f)
        logger.info("config.yaml file loaded successfully")
        return config
    
    except FileNotFoundError as e:
        logger.error("config.yaml file not found")
        raise CustomException(e)
    
    except Exception as e:
        logger.error(f"an error occured loading config.yaml file: {str(e)}")
        raise CustomException(e)


def load_data(raw_path_1, raw_path_2):
    try:
        logger.info("loading both excel data files")
        df1 = pd.read_excel(raw_path_1)
        logger.info("successfully loaded case_study1.xlsx excel file")
        df2 = pd.read_excel(raw_path_2)
        logger.info("successfully loaded case_study2.xlsx file")
        return df1, df2

    except FileNotFoundError as e:
        logger.error("files not found")
        raise CustomException(e)

    except Exception as e:
        logger.error(f"some error occurred in loading file:{str(e)}")
        raise CustomException(e)


def load_raw_data():
    config = load_config()

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    raw_path_1 = os.path.join(BASE_DIR, config["data"]["raw_path_1"])
    raw_path_2 = os.path.join(BASE_DIR, config["data"]["raw_path_2"])

    logger.info("Starting raw data loading pipeline")
    df1, df2 = load_data(raw_path_1, raw_path_2)
    logger.info("Raw data loaded successfully")

    save_path_1 = os.path.join(BASE_DIR, "data", "interim", "case_study1.csv")
    save_path_2 = os.path.join(BASE_DIR, "data", "interim", "case_study2.csv")

    os.makedirs(os.path.dirname(save_path_1), exist_ok=True)

    df1.to_csv(save_path_1, index=False)
    logger.info("Saved case_study1.csv to data/interim")

    df2.to_csv(save_path_2, index=False)
    logger.info("Saved case_study2.csv to data/interim")

    logger.info(f"Shape df1: {df1.shape}")
    logger.info(f"Shape df2: {df2.shape}")

    return df1, df2


if __name__ == "__main__":
    df1, df2 = load_raw_data()
    print(df1.shape, df2.shape)