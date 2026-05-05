from src.data_ingestion import load_raw_data
from src.data_ingestion import load_config,load_params
import pandas as pd
import numpy as np
import os

config=load_config()

def clean_df1(df1):
    df1 = df1[df1['Age_Oldest_TL'] != -99999]
    return df1


def clean_df2(df2):
    columns_to_removed = []

    for col in df2.columns:
        if (df2[col] == -99999).sum() > 10000:
            columns_to_removed.append(col)

    df2 = df2.drop(columns=columns_to_removed)

    columns_treated = []

    for col in df2.columns:
        count = (df2[col] == -99999).sum()
        percentage = (count / len(df2)) * 100

        if count > 0:
            print(f"Column: {col}, Count: {count}, Percentage: {percentage:.2f}%")
            columns_treated.append(col)

    return df2, columns_treated


def treat_columns(df, columns_treated):
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


def find_common_features(df1, df2):
    common_columns = set(df1.columns).intersection(set(df2.columns))
    print("Common Columns:", common_columns)
    return common_columns


def merge_dataframe(df1, df2):
    df = pd.merge(df1, df2, how='inner', on='PROSPECTID')
    return df


def transformed_data():
    # Step 1: Load data
    df1, df2 = load_raw_data()

    # Step 2: Clean df1
    df1 = clean_df1(df1)

    # Step 3: Clean df2
    df2, columns_treated = clean_df2(df2)

    # Step 4: Treat missing values
    df2 = treat_columns(df2, columns_treated)

    # Step 5: Merge
    df = merge_dataframe(df1, df2)

    # Step 6: Save merged data
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_path = os.path.join(BASE_DIR, "data", "processed", "merged_data.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)

    print("Merged data saved at:", save_path)
    print("Final shape:", df.shape)

    return df

if __name__ == "__main__":
    df = transformed_data()