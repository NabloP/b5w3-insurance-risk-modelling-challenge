"""
xgboost_model_trainer.py – Task 4 XGBoost Trainer (B5W3)
-------------------------------------------------------------------------------
Trains and evaluates XGBoost models for:
  • Claim Frequency Classification (binary target: ClaimFrequency)
  • Claim Severity Regression (continuous target: TotalClaims on subset)

Core Features:
  • Supports both classification and regression tasks
  • Defensive validation of training data and inputs
  • Robust evaluation using appropriate metrics (F1, AUC, RMSE, R²)
  • Returns predictions for integration into premium pricing logic

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard & Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
import warnings


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: XGBoostModelTrainer
# ───────────────────────────────────────────────────────────────────────────────
class XGBoostModelTrainer:
    """
    A unified trainer for both classification and regression models using XGBoost.
    Handles training, evaluation, and prediction logic for insurance modeling tasks.
    """

    def __init__(self, task_type: str = "classification", random_state: int = 42):
        """
        Initializes the trainer and selects the model type.

        Args:
            task_type (str): 'classification' or 'regression'
            random_state (int): Random seed for reproducibility
        """
        self.task_type = task_type.lower()
        self.random_state = random_state

        # Select appropriate model
        if self.task_type == "classification":
            self.model = XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        elif self.task_type == "regression":
            self.model = XGBRegressor(
                random_state=self.random_state, n_jobs=-1, verbosity=0
            )
        else:
            raise ValueError(
                f"❌ Invalid task_type '{task_type}'. Must be 'classification' or 'regression'."
            )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the model on input training data.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Raises:
            ValueError: If training data contains nulls or classification target is single-class
        """
        # Defensive data check
        if X_train.isnull().any().any() or y_train.isnull().any():
            raise ValueError("❌ Training data contains missing values.")

        if self.task_type == "classification" and y_train.nunique() < 2:
            raise ValueError(
                "❌ Classification requires at least two unique target classes."
            )

        # Train model
        self.model.fit(X_train, y_train)
        print("✅ Model training complete.")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the trained model.

        Args:
            X_test (pd.DataFrame): Test features

        Returns:
            np.ndarray: Predicted labels or scores
        """
        if not hasattr(self.model, "feature_importances_"):
            raise RuntimeError("❌ Model must be trained before calling `.predict()`.")
        return self.model.predict(X_test)

    def evaluate(self, X_test: pd.DataFrame, y_true: pd.Series) -> dict:
        """
        Evaluates the model and returns metrics and predictions.

        Args:
            X_test (pd.DataFrame): Features for prediction
            y_true (pd.Series): True target values

        Returns:
            dict: Evaluation results (metrics + predictions)
        """
        if not hasattr(self.model, "feature_importances_"):
            raise RuntimeError("❌ Model must be trained before evaluation.")

        # Classification logic
        if self.task_type == "classification":
            y_pred_label = self.model.predict(X_test)
            y_pred_prob = self.model.predict_proba(X_test)[:, 1]

            report = classification_report(y_true, y_pred_label, output_dict=True)
            auc = roc_auc_score(y_true, y_pred_prob)

            print("📊 Classification Report (XGBoost):")
            print(classification_report(y_true, y_pred_label))
            print(f"🏆 ROC-AUC Score: {auc:.4f}")

            return {
                "f1_score": f1_score(y_true, y_pred_label),
                "roc_auc": auc,
                "report": report,
                "y_true": y_true,
                "y_pred": y_pred_prob,  # Needed for expected premium calculation
            }

        # Regression logic
        elif self.task_type == "regression":
            y_pred = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Backward compatible
            r2 = r2_score(y_true, y_pred)

            print("📈 Regression Results (XGBoost):")
            print(f"🏆 RMSE: {rmse:.2f}")
            print(f"📊 R-squared: {r2:.4f}")

            return {
                "rmse": rmse,
                "r_squared": r2,
                "y_true": y_true,
                "y_pred": y_pred,  # Needed for expected premium calculation
            }

        else:
            warnings.warn("⚠️ Unknown task type during evaluation.")
            return {}
