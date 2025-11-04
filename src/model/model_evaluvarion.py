# ...existing code...
import os
import sys
import json
import logging

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

control_handeler = logging.StreamHandler(sys.stdout)
control_handeler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
control_handeler.setFormatter(formatter)
logger.addHandler(control_handeler)

def evaluate_model(model, x_test, y_test):
    try:
        if x_test is None or y_test is None:
            raise ValueError("x_test or y_test is None")

        if len(y_test) == 0 or getattr(x_test, "shape", (None,))[0] == 0:
            raise ValueError("Test set is empty")

        # Make predictions
        y_pred = model.predict(x_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

        # Print metrics and report
        logger.info("Model Evaluation Metrics: Accuracy: %.4f Precision: %.4f Recall: %.4f F1: %.4f",
                    accuracy, precision, recall, f1)
        logger.info("\n%s", classification_report(y_test, y_pred, zero_division=0))

        return metrics
    except Exception:
        logger.exception("Failed to evaluate model")
        raise

def save_metrics(metrics: dict, out_path: str = "metrics.json"):
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Saved metrics to %s", out_path)
    except Exception:
        logger.exception("Failed to save metrics to %s", out_path)
        raise

def main():
    try:
        test_path = os.path.join("data", "features", "test_bow.csv")
        model_path = os.path.join("data", "gradient_boosting_model.pkl")
        metrics_path = "metrics.json"

        if not os.path.exists(test_path):
            logger.error("Test features file not found: %s", test_path)
            sys.exit(1)
        if not os.path.exists(model_path):
            logger.error("Model file not found: %s", model_path)
            sys.exit(1)

        # Load test data
        test_data = pd.read_csv(test_path)
        if "sentiment" not in test_data.columns:
            logger.error("'sentiment' column missing in %s", test_path)
            sys.exit(1)

        x_test = test_data.drop("sentiment", axis=1).values
        y_test = test_data["sentiment"].values

        # Load the trained model
        try:
            model = joblib.load(model_path)
        except Exception:
            logger.exception("Failed to load model from %s", model_path)
            raise

        metrics = evaluate_model(model, x_test, y_test)
        save_metrics(metrics, metrics_path)

    except SystemExit:
        raise
    except Exception:
        logger.exception("Model evaluation pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
# ...existing code...