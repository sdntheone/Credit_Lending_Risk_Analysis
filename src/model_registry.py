import json
from mlflow import MlflowClient
import os
import dagshub
import mlflow

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
REPORT_PATH = os.path.join(BASE_DIR, "reports","model_info.json")


def register_model(info_path):
    try:
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"{info_path} not found")
        
        with open(info_path,'r') as f:
            model_info=json.load(f)

        run_id = model_info["run_id"]
        model_name = model_info["model_name"]
        model_uri = model_info["model_uri"]

        client = MlflowClient()

        try:
            client.create_registered_model(model_name)
            logger.info(f"registred model creatd: {model_name}")

        except Exception as e:
            logger.info(f"registered model already exist: {model_name}")

        result=client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        logger.info(f"model version created: Version {result.version}")

        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=result.version
        )
        logger.info(f"Alias 'champion' assigned to version {result.version}")

        alias_model=client.get_model_version_by_alias(
            name=model_name,
            alias="champion"
        )
        logger.info(f"Champion model version: {alias_model.version}")

        print(f"Model Name: {alias_model.name}")
        print(f"Model Version: {alias_model.version}")
        print(f"Alias: champion")
        print(f"Run ID: {alias_model.run_id}")
        print(f"Source: {alias_model.source}")

    except Exception as e:

        logger.error(f"Error in register_model: {e}")

        raise CustomException(e)


def main():

    try:

        register_model(REPORT_PATH)

        logger.info("Model registry pipeline completed")

    except Exception as e:

        logger.error(f"Error in registry pipeline: {e}")

        raise CustomException(e)


if __name__ == "__main__":
    main()



        