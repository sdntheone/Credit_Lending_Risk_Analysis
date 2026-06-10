import mlflow
import mlflow.pyfunc
import dagshub
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

mlflow.set_tracking_uri(
    "https://dagshub.com/sdntheone/Credit_Lending_Risk_Analysis.mlflow"
)

# dagshub.init(
#     repo_owner="sdntheone",
#     repo_name="Credit_Lending_Risk_Analysis",
#     mlflow=True
# )

MODEL_URI = "models:/CreditRiskModel@champion"


def load_model(): 
    return mlflow.pyfunc.load_model(MODEL_URI)


def predict_output(model, data): 
    return model.predict(data)


def load_label_encoder():

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    encoder_path = os.path.join(BASE_DIR, "artifacts", "label_encoder.pkl")

    label_encoder = joblib.load(encoder_path)

    return label_encoder


def decode_prediction(label_encoder, prediction):

    decoded_prediction = label_encoder.inverse_transform(prediction)

    return decoded_prediction