"""
xgboost_regressor_trainer.py – Task 4 XGBoost Regressor Trainer (B5W3)
------------------------------------------------------------------------------
Trains an XGBoost Regressor to model claim severity (TotalClaims) on the subset
of insurance policyholders who have filed at least one claim.

Core responsibilities:
  • Filters training data to include only rows with ClaimFrequency == 1
  • Trains an XGBoost regressor on selected features
  • Evaluates model using RMSE and R²
  • Generates SHAP explainability plot inline and optionally saves it
  • Returns trained model, predictions, evaluation metrics, and feature importances

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os  # File path handling

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import numpy as np  # Array operations
import pandas as pd  # DataFrame manipulation
from xgboost import XGBRegressor  # XGBoost model
from sklearn.metrics import mean_squared_error, r2_score  # Eval metrics
import shap  # SHAP explainability
import matplotlib.pyplot as plt  # For SHAP plots
import warnings  # To suppress expected model warnings

# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class Definition
# ───────────────────────────────────────────────────────────────────────────────
class XGBoostRegressorTrainer:
    """
    Trains and evaluates an XGBoost regression model for claim severity prediction
    on policyholders who have made at least one claim (ClaimFrequency == 1).
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the trainer with a reproducible random state.

        Args:
            random_state (int): Random seed for model consistency
        """
        self.random_state = random_state
        self.model = None  # Placeholder for the trained model

    def filter_claim_positive(self, X: pd.DataFrame, y: pd.Series):
        """
        Filters dataset to include only rows where ClaimFrequency == 1.

        Args:
            X (pd.DataFrame): Feature matrix (must include 'ClaimFrequency')
            y (pd.Series): Target variable (TotalClaims)

        Returns:
            X_filtered (pd.DataFrame), y_filtered (pd.Series): Subset data
        """
        if "ClaimFrequency" not in X.columns:
            raise ValueError("Missing 'ClaimFrequency' column in X.")

        mask = X["ClaimFrequency"] == 1
        X_filtered = X[mask].copy()
        y_filtered = y[mask].copy()
        X_filtered = X_filtered.drop(columns=["ClaimFrequency"])

        return X_filtered, y_filtered

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains an XGBoost regressor on the input training data.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values

        Returns:
            None
        """
        warnings.filterwarnings("ignore", category=UserWarning)

        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the trained model on test data.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): True target values

        Returns:
            dict: Dictionary containing RMSE and R² scores
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation.")

        preds = self.model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        return {
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "y_true": y_test,
            "y_pred": preds,
        }

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Retrieves feature importances from the trained XGBoost model.

        Args:
            feature_names (list): List of feature names

        Returns:
            pd.DataFrame: Sorted DataFrame of feature importances
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before retrieving feature importances.")

        importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        return importance_df

    def compute_shap(
        self,
        X_sample: pd.DataFrame,
        feature_names: list,
        save_path: str = None,
    ):
        """
        Computes and plots SHAP values for the trained XGBoost model.

        Args:
            X_sample (pd.DataFrame): Data sample to explain (should match training schema)
            feature_names (list): List of feature names (used for labeling)
            save_path (str, optional): File path to save the SHAP plot. If None, no file is saved.

        Returns:
            shap_values (np.ndarray): Matrix of SHAP values
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before computing SHAP.")

        # Use SHAP TreeExplainer optimized for XGBoost
        explainer = shap.Explainer(self.model, X_sample)

        # Compute SHAP values
        shap_values = explainer(X_sample)

        # Plot summary inline
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)

        # Save plot to file if requested
        if save_path:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")

        # Show inline after saving (if running in notebook)
        plt.show()

        return shap_values
