import pandas as pd
import os
import joblib

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from src.data_ingestion import load_config


# -------------------------------
# 1. Load Model
# -------------------------------
def load_model():
    path = os.path.join(os.getcwd(), "artifacts", "model", "model.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError("Model not found")

    return joblib.load(path)


# -------------------------------
# 2. Load Data
# -------------------------------
def load_data():
    config = load_config()
    target_col = config["features"]["target_column"]

    path = os.path.join(os.getcwd(), "data", "processed", "final_data.csv")

    df = pd.read_csv(path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


# -------------------------------
# 3. Evaluate
# -------------------------------
def evaluate(model, X, y):
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    f1_weighted = f1_score(y, y_pred, average='weighted')
    f1_macro = f1_score(y, y_pred, average='macro')

    precision, recall, f1_vals, _ = precision_recall_fscore_support(y, y_pred)

    # 🔥 Mapping
    class_names = ["P1", "P2", "P3", "P4"]

    print("\n===== FINAL MODEL EVALUATION =====")
    print("Accuracy:", accuracy)
    print("F1 Weighted:", f1_weighted)
    print("F1 Macro:", f1_macro)

    for i in range(len(f1_vals)):
        print(f"\nClass {class_names[i]}:")
        print("Precision:", precision[i])
        print("Recall:", recall[i])
        print("F1:", f1_vals[i])
# -------------------------------
# 4. Main
# -------------------------------
def main():
    model = load_model()
    X, y = load_data()

    evaluate(model, X, y)


if __name__ == "__main__":
    main()