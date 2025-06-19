"""
logistic_model_trainer.py – Task 4 Binary Classifier (Logistic Regression) – B5W3
-------------------------------------------------------------------------------
Trains a logistic regression model on the AlphaCare insurance dataset and
evaluates performance for claim frequency prediction.

Core responsibilities:
  • Fits a logistic regression model using sklearn
  • Supports training, prediction, and evaluation in modular fashion
  • Returns metrics: accuracy, precision, recall, F1-score, confusion matrix
  • Logs shape diagnostics and warnings for input issues

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard & Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For data handling
import numpy as np  # For numerical safety checks
from sklearn.linear_model import LogisticRegression  # Model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)  # Evaluation metrics
import traceback  # For detailed error tracebacks


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: LogisticModelTrainer
# ───────────────────────────────────────────────────────────────────────────────
class LogisticModelTrainer:
    """
    Trains and evaluates a Logistic Regression model for binary classification.

    Supports training, prediction, and evaluation in a fully modular and defensively
    programmed class interface.
    """

    def __init__(self, penalty="l2", C=1.0, solver="liblinear", random_state=42):
        """
        Initialize the logistic regression model with customizable hyperparameters.

        Args:
            penalty (str): Regularization type (default: "l2").
            C (float): Inverse of regularization strength (smaller = stronger regularization).
            solver (str): Optimization algorithm (default: "liblinear" for binary tasks).
            random_state (int): Seed for reproducibility.
        """
        try:
            self.model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                random_state=random_state,
            )
        except Exception as e:
            print("❌ Failed to initialize Logistic Regression model:")
            print(traceback.format_exc())

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the logistic regression model on the provided dataset.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Binary classification target.
        """
        try:
            assert not X_train.empty and not y_train.empty, "Training data is empty."
            self.model.fit(X_train, y_train)
            print("✅ Logistic Regression model trained successfully.")
        except Exception as e:
            print("❌ Error during training:")
            print(traceback.format_exc())

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate binary predictions on test data.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            np.ndarray: Predicted binary class labels.
        """
        try:
            assert hasattr(self.model, "coef_"), "Model not trained yet."
            return self.model.predict(X_test)
        except Exception as e:
            print("❌ Prediction failed:")
            print(traceback.format_exc())
            return np.array([])

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        """
        Compute classification metrics comparing predictions with true labels.

        Args:
            y_true (pd.Series): Actual binary labels.
            y_pred (np.ndarray): Predicted binary labels.

        Returns:
            dict: Dictionary of evaluation metrics (accuracy, precision, recall, F1, confusion matrix).
        """
        try:
            assert len(y_true) == len(
                y_pred
            ), "Mismatch between prediction and label lengths."

            # Compute standard classification metrics
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "confusion_matrix": confusion_matrix(
                    y_true, y_pred
                ).tolist(),  # List for export compatibility
            }

            print("📊 Evaluation Metrics:")
            for k, v in metrics.items():
                if k != "confusion_matrix":
                    print(f"  {k.title()}: {v:.4f}")
            print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

            return metrics

        except Exception as e:
            print("❌ Evaluation failed:")
            print(traceback.format_exc())
            return {}
