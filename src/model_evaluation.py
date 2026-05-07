from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.utils.logger import get_logger
from src.utils.exception import CustomException


logger = get_logger(__name__)


def evaluate(model, X, y):

    try:

        if X.empty:
            raise ValueError("Evaluation data is empty")

        y_pred = model.predict(X)

        accuracy = accuracy_score(y, y_pred)

        f1_weighted = f1_score(
            y,
            y_pred,
            average="weighted"
        )

        f1_macro = f1_score(
            y,
            y_pred,
            average="macro"
        )

        precision, recall, f1_vals, _ = precision_recall_fscore_support(
            y,
            y_pred
        )

        logger.info(f"Accuracy: {accuracy}")

        logger.info(f"F1 Weighted: {f1_weighted}")

        logger.info(f"F1 Macro: {f1_macro}")

        metrics = {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro
        }

        class_names = ["P1", "P2", "P3", "P4"]

        for i in range(len(precision)):

            logger.info(
                f"{class_names[i]} -> "
                f"Precision: {precision[i]}, "
                f"Recall: {recall[i]}, "
                f"F1: {f1_vals[i]}"
            )

            metrics[f"precision_{class_names[i]}"] = precision[i]

            metrics[f"recall_{class_names[i]}"] = recall[i]

            metrics[f"f1_{class_names[i]}"] = f1_vals[i]

        return metrics

    except Exception as e:

        logger.error(f"Error in evaluate: {e}")

        raise CustomException(e)