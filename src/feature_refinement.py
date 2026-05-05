import pandas as pd
import numpy as np
import os

from xgboost import XGBClassifier
from src.data_ingestion import load_config


# -------------------------------
# 1. Load Selected Data
# -------------------------------
def load_selected_data():
    base_dir = os.getcwd()
    path = os.path.join(base_dir, "data", "processed", "selected_data.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print("Loaded selected data:", df.shape)

    return df


# -------------------------------
# 2. Model-Based Feature Selection
# -------------------------------
def model_feature_selection(X, y, threshold=0.01):
    model = XGBClassifier(eval_metric="mlogloss")
    model.fit(X, y)

    importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:\n", feature_importance_df)

    selected_features = feature_importance_df[
        feature_importance_df["importance"] > threshold
    ]["feature"].tolist()

    # safety check
    if len(selected_features) == 0:
        print("Warning: No features selected, lowering threshold")
        selected_features = feature_importance_df.head(20)["feature"].tolist()

    return selected_features


# -------------------------------
# 3. Correlation Filtering
# -------------------------------
def correlation_filter(X, threshold=0.85):
    X_numeric = X.select_dtypes(include=[np.number])

    corr_matrix = X_numeric.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print("\nDropping highly correlated features:", to_drop)

    return X.drop(columns=to_drop, errors="ignore")


# -------------------------------
# 4. Main Pipeline
# -------------------------------
def feature_refinement_pipeline():
    config = load_config()
    target_col = config["features"]["target_column"]

    # Step 1: Load data
    df = load_selected_data()

    # Step 2: Remove ID columns (IMPORTANT)
    df = df.drop(columns=["PROSPECTID"], errors="ignore")

    # Step 3: Split
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 4: Correlation filtering
    X = correlation_filter(X, threshold=0.85)

    # Step 5: Model-based selection
    selected_features = model_feature_selection(X, y, threshold=0.01)

    print("\nFinal Selected Features:", selected_features)

    # Step 6: Final dataset
    df_final = X[selected_features].copy()
    df_final[target_col] = y

    print("\nFinal shape after refinement:", df_final.shape)

    # Step 7: Save
    base_dir = os.getcwd()
    save_path = os.path.join(base_dir, "data", "processed", "final_data.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_final.to_csv(save_path, index=False)

    print("Final data saved at:", save_path)

    return df_final, selected_features


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    df_final, features = feature_refinement_pipeline()