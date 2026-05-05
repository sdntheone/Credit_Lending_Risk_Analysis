import pandas as pd
import os
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(raw_path_1, raw_path_2):
    df1 = pd.read_excel(raw_path_1)
    df2 = pd.read_excel(raw_path_2)
    return df1, df2


def load_raw_data():
    config = load_config()

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    raw_path_1 = os.path.join(BASE_DIR, config["data"]["raw_path_1"])
    raw_path_2 = os.path.join(BASE_DIR, config["data"]["raw_path_2"])

    df1, df2 = load_data(raw_path_1, raw_path_2)

    return df1, df2


if __name__ == "__main__":
    df1, df2 = load_raw_data()
    print(df1.shape, df2.shape)