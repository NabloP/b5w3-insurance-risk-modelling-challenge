"""
random_forest_trainer.py – Task 4 Random Forest Trainer (B5W3)
-------------------------------------------------------------------------------
Trains a Random Forest Classifier on the balanced insurance dataset to predict
ClaimFrequency (binary classification task).

Core responsibilities:
  • Fits a RandomForestClassifier on training features and binary labels
  • Predicts outcomes on test data with optional probability scores
  • Computes classification metrics (accuracy, precision, recall, F1, ROC AUC)
  • Provides SHAP feature attribution for interpretability
  • Includes full defensive programming for training/prediction/evaluation

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame operations
import numpy as np  # For array checks and validation
import warnings  # To suppress warnings gracefully
from sklearn.ensemble import RandomForestClassifier  # ML model
from sklearn.metrics import (  # Evaluation metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: RandomForestModelTrainer
# ───────────────────────────────────────────────────────────────────────────────
class RandomForestModelTrainer:
    """
    Wraps a RandomForestClassifier pipeline for training, prediction, evaluation,
    and SHAP-based feature attribution. Built for modular, defensible modeling.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the trainer with model parameters and reproducibility.

        Args:
            random_state (int): Seed for deterministic tree building.
        """
        self.model = RandomForestClassifier(random_state=random_state)
        self.trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the model on training features and binary labels.

        Args:
            X_train (pd.DataFrame): Feature matrix.
            y_train (pd.Series): Target labels.

        Raises:
            ValueError: If data has NaNs or lacks label diversity.
        """
        try:
            if X_train.isnull().any().any() or y_train.isnull().any():
                raise ValueError("Training data contains NaNs.")

            if len(np.unique(y_train)) < 2:
                raise ValueError("Target must have at least two classes.")

            self.model.fit(X_train, y_train)
            self.trained = True
            print("✅ Random Forest training complete.")

        except Exception as e:
            print(f"❌ Error during training: {e}")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Generate predictions on test data.

        Args:
            X_test (pd.DataFrame): Test feature matrix.

        Returns:
            pd.Series: Predicted labels.
        """
        try:
            assert self.trained, "Model must be trained before prediction."
            y_pred = self.model.predict(X_test)
            return pd.Series(y_pred, index=X_test.index, name="y_pred")

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return pd.Series([], name="y_pred")

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Evaluate model using standard classification metrics.

        Args:
            y_true (pd.Series): Ground-truth labels.
            y_pred (pd.Series): Model predictions.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        try:
            assert len(y_true) == len(y_pred), "Label and prediction lengths differ."

            results = {
                "Accuracy": round(accuracy_score(y_true, y_pred), 4),
                "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "ROC AUC": round(roc_auc_score(y_true, y_pred), 4),
            }

            print("📈 Evaluation Metrics (Random Forest):")
            for k, v in results.items():
                print(f"• {k}: {v}")

            return results

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {}

    def explain_with_shap(self, X_sample: pd.DataFrame, max_display: int = 10):
        """
        Run SHAP TreeExplainer for model interpretability.

        Args:
            X_sample (pd.DataFrame): Subsample of features for SHAP.
            max_display (int): Top features to show.

        Raises:
            AssertionError: If model not trained.
        """
        try:
            import shap  # Lazy import for optional dependency

            assert self.trained, "Model must be trained before SHAP analysis."

            print("🔍 Running SHAP feature attribution...")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            print("📊 SHAP Feature Importance (Random Forest):")
            shap.summary_plot(
                shap_values[1], X_sample, plot_type="bar", max_display=max_display
            )

        except Exception as e:
            print(f"❌ SHAP analysis failed: {e}")
