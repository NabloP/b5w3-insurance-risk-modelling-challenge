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
  • SHAP-ready design for post-hoc model explainability

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard and Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import numpy as np  # For numeric operations
import pandas as pd  # For DataFrame handling
from xgboost import XGBClassifier, XGBRegressor  # XGBoost models
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,  # Classification metrics
    mean_squared_error,
    r2_score,  # Regression metrics
)
import warnings  # For defensive user alerts


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: XGBoostModelTrainer
# ───────────────────────────────────────────────────────────────────────────────
class XGBoostModelTrainer:
    """
    Handles both classification (claim probability) and regression (claim severity)
    model training and evaluation using XGBoost.
    """

    def __init__(self, task_type: str = "classification", random_state: int = 42):
        """
        Initialize the model based on task type and set reproducibility.

        Args:
            task_type (str): "classification" or "regression"
            random_state (int): Random seed for reproducibility
        """
        self.task_type = task_type.lower()
        self.random_state = random_state

        # Initialize the appropriate XGBoost model based on task type
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
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )
        else:
            raise ValueError(
                f"❌ Invalid task_type: {task_type}. Choose 'classification' or 'regression'."
            )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the XGBoost model on provided training data.

        Args:
            X_train (pd.DataFrame): Feature matrix for training
            y_train (pd.Series): Target vector for training

        Raises:
            ValueError: If input contains NaNs or only one class (classification)
        """
        # Validate training data
        if X_train.isnull().any().any() or y_train.isnull().any():
            raise ValueError(
                "❌ Training data contains missing values. Impute or drop NaNs before training."
            )

        if self.task_type == "classification" and y_train.nunique() < 2:
            raise ValueError(
                "❌ Classification task requires at least two classes in y_train."
            )

        # Fit the model
        self.model.fit(X_train, y_train)
        print("✅ Model training complete.")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions on test data.

        Args:
            X_test (pd.DataFrame): Feature matrix for prediction

        Returns:
            np.ndarray: Predicted values (labels or scores)
        """
        if not hasattr(self.model, "feature_importances_"):
            raise RuntimeError(
                "❌ Model not trained yet. Run `.train()` before `.predict()`."
            )

        return self.model.predict(X_test)

    def evaluate(self, X_test: pd.DataFrame, y_true: pd.Series) -> dict:
        """
        Evaluate the model using appropriate metrics.

        Args:
            X_test (pd.DataFrame): Test features
            y_true (pd.Series): Ground truth targets

        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Predict on test data
        y_pred = self.predict(X_test)

        # Classification metrics
        if self.task_type == "classification":
            report = classification_report(y_true, y_pred, output_dict=True)
            auc = roc_auc_score(y_true, self.model.predict_proba(X_test)[:, 1])
            print("📊 Classification Report (XGBoost):")
            print(classification_report(y_true, y_pred))
            print(f"🏆 ROC-AUC Score: {auc:.4f}")

            return {
                "f1_score": f1_score(y_true, y_pred),
                "roc_auc": auc,
                "report": report,
            }

        # Regression metrics
        elif self.task_type == "regression":
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            r2 = r2_score(y_true, y_pred)
            print("📈 Regression Results (XGBoost):")
            print(f"🏆 RMSE: {rmse:.2f}")
            print(f"📊 R-squared: {r2:.4f}")

            return {"rmse": rmse, "r_squared": r2}

        else:
            warnings.warn("⚠️ Unknown task type during evaluation.")
            return {}
