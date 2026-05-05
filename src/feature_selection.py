from src.feature_engineering import encoded_data
from src.data_ingestion import load_config

import pandas as pd
import numpy as np
import os

from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor


# -------------------------------
# 1. Chi-Square Selection
# -------------------------------
def chi_square_selection(df, categorical_cols, target_col, p_threshold=0.05):
    selected_cols = []

    for col in categorical_cols:
        if col not in df.columns:
            continue

        chi2, pval, _, _ = chi2_contingency(
            pd.crosstab(df[col], df[target_col])
        )

        print(f"{col} → p-value: {pval}")

        if pval <= p_threshold:
            selected_cols.append(col)

    return selected_cols


# -------------------------------
# 2. VIF Filtering
# -------------------------------
def remove_high_vif(X, threshold=10):
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

        print("\nVIF Table:\n", vif_df)

        max_vif = vif_df["vif"].iloc[0]

        if max_vif < threshold:
            break

        drop_col = vif_df["feature"].iloc[0]
        print(f"Dropping {drop_col} (VIF={max_vif})")

        X.drop(columns=[drop_col], inplace=True)

    return X.columns.tolist()


# -------------------------------
# 3. ANOVA Selection
# -------------------------------
def anova_selection(df, numerical_cols, target_col, p_threshold=0.05):
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


# -------------------------------
# 4. Save Utility
# -------------------------------
def save_dataframe(df, filename):
    base_dir = os.getcwd()
    path = os.path.join(base_dir, "data", "processed", filename)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\nSaved file at: {path}")


# -------------------------------
# 5. Main Pipeline
# -------------------------------
def feature_selection_pipeline():
    config = load_config()

    target_col = config["features"]["target_column"]
    categorical_cols = config["features"]["categorical_cols"]

    # Load encoded data
    df, _ = encoded_data()

    # ------------------
    # Categorical Selection
    # ------------------
    selected_cat_cols = chi_square_selection(
        df, categorical_cols, target_col
    )

    # ------------------
    # Numerical Selection
    # ------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # VIF
    cols_after_vif = remove_high_vif(df[numeric_cols])

    # ANOVA
    selected_num_cols = anova_selection(
        df, cols_after_vif, target_col
    )

    # ------------------
    # Final Feature Set
    # ------------------
    final_features = selected_num_cols + selected_cat_cols

    print("\nFinal Selected Features:\n", final_features)

    df_selected = df[final_features + [target_col]].copy()

    # Save output
    save_dataframe(df_selected, "selected_data.csv")

    print("\nFinal Shape:", df_selected.shape)

    return df_selected, final_features


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    df_selected, features = feature_selection_pipeline()