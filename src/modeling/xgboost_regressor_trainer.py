"""
xgboost_regressor_trainer.py – Task 4 XGBoost Regressor Trainer (B5W3)
------------------------------------------------------------------------------
Trains an XGBoost Regressor to model claim severity (TotalClaims) on the subset
of insurance policyholders who have filed at least one claim.

Core responsibilities:
  • Filters training data to include only rows with ClaimFrequency == 1
  • Trains an XGBoost regressor on selected features
  • Evaluates model using RMSE and R² (version-safe logic)
  • Generates SHAP explainability plot inline and optionally saves it
  • Returns trained model, predictions, evaluation metrics, and feature importances

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Imports
# ───────────────────────────────────────────────────────────────────────────────
import os  # For file saving
import numpy as np  # Numeric operations
import pandas as pd  # DataFrame operations
import shap  # SHAP explainability
import matplotlib.pyplot as plt  # Plotting
import warnings  # Defensive warnings

from xgboost import XGBRegressor  # XGBoost model
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation metrics


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: XGBoostRegressorTrainer
# ───────────────────────────────────────────────────────────────────────────────
class XGBoostRegressorTrainer:
    """
    Trains and evaluates an XGBoost regression model for claim severity prediction
    on policyholders with ClaimFrequency == 1.
    """

    def __init__(self, random_state: int = 42):
        """
        Initializes the trainer.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None  # Placeholder until model is trained

    def filter_claim_positive(self, X: pd.DataFrame, y: pd.Series):
        """
        Filters X and y to include only rows where ClaimFrequency == 1.

        Args:
            X (pd.DataFrame): Feature matrix (must include 'ClaimFrequency')
            y (pd.Series): Target variable (e.g. TotalClaims or Margin)

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Filtered X and y
        """
        if "ClaimFrequency" not in X.columns:
            raise ValueError("Missing 'ClaimFrequency' column in X.")

        # Mask for claim-positive rows
        mask = X["ClaimFrequency"] == 1
        X_filtered = X[mask].drop(columns=["ClaimFrequency"]).copy()
        y_filtered = y[mask].copy()

        return X_filtered, y_filtered

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains an XGBoost regressor.

        Args:
            X_train (pd.DataFrame): Features
            y_train (pd.Series): Target

        Returns:
            None
        """
        # Ignore benign warnings during training
        warnings.filterwarnings("ignore", category=UserWarning)

        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )

        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the trained model on test data using RMSE and R².

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): True target values

        Returns:
            dict: Dictionary with RMSE, R², true/predicted values
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation.")

        # Predict using trained model
        preds = self.model.predict(X_test)

        # Compute RMSE manually (squared=False is version-dependent)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        return {
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "y_true": y_test,
            "y_pred": preds,
        }

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Returns feature importances as a sorted DataFrame.

        Args:
            feature_names (list): List of column names

        Returns:
            pd.DataFrame: Sorted importance table
        """
        if self.model is None:
            raise RuntimeError(
                "Model must be trained before retrieving feature importances."
            )

        importances = self.model.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        )

        return importance_df.sort_values(by="Importance", ascending=False).reset_index(
            drop=True
        )

    def compute_shap(
        self,
        X_sample: pd.DataFrame,
        feature_names: list,
        save_path: str = None,
    ):
        """
        Computes and plots SHAP values for the trained model.

        Args:
            X_sample (pd.DataFrame): Subset of test or training data
            feature_names (list): Feature labels
            save_path (str, optional): Where to save the SHAP plot

        Returns:
            np.ndarray: SHAP values array
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before computing SHAP.")

        # Compute SHAP values using XGBoost-aware explainer
        explainer = shap.Explainer(self.model, X_sample)
        shap_values = explainer(X_sample)

        # Plot SHAP summary
        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_names, show=False
        )

        # Save plot if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            print(f"✅ SHAP plot saved to {save_path}")

        # Always show the plot
        plt.show()

        return shap_values
