import pandas as pd
import os
import joblib

from src.data_ingestion import load_config

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter


# -------------------------------
# 1. Load Data
# -------------------------------
def load_data():
    path = os.path.join(os.getcwd(), "data", "processed", "final_data.csv")

    if not os.path.exists(path):
        raise FileNotFoundError("final_data.csv not found")

    df = pd.read_csv(path)
    print("Loaded data:", df.shape)

    return df


# -------------------------------
# 2. Split
# -------------------------------
def split_data(df, target_col, test_size, random_state):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


# -------------------------------
# 3. Targeted SMOTE
# -------------------------------
def apply_smote(X_train, y_train):

    counts = Counter(y_train)

    sampling_strategy = {
        1: counts[1],
        2: int(counts[2] * 2.0),   # 🔥 increased boost for class 2
        3: int(counts[3] * 2.5),
        0: int(counts[0] * 1.8)
    }

    print("Sampling strategy:", sampling_strategy)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    return (
        pd.DataFrame(X_res, columns=X_train.columns),
        pd.Series(y_res)
    )


# -------------------------------
# 4. Train Final Model
# -------------------------------
def train_model(X_train, y_train):

    model = XGBClassifier(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=200,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=y_train.map({
    0: 1.2,
    1: 1,
    2: 2.2,
    3: 1.5
    }))

    return model


# -------------------------------
# 5. Save Model
# -------------------------------
def save_model(model):
    path = os.path.join(os.getcwd(), "artifacts", "model", "model.pkl")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

    print("Model saved at:", path)


# -------------------------------
# 6. Main
# -------------------------------
def main():
    config = load_config()

    target_col = config["features"]["target_column"]
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(
        df, target_col, test_size, random_state
    )

    # Apply targeted SMOTE
    X_train, y_train = apply_smote(X_train, y_train)

    # Train final model
    model = train_model(X_train, y_train)

    # Save model
    save_model(model)


if __name__ == "__main__":
    main()