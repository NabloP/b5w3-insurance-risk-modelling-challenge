"""
feature_scaler.py – Task 4 Feature Scaling Utility (B5W3)
------------------------------------------------------------------------------
Scales numeric feature columns using StandardScaler or RobustScaler.
Supports consistent transformation of training and test datasets.

Core responsibilities:
  • Fits scaler on X_train, applies to both X_train and X_test
  • Option to use StandardScaler (default) or RobustScaler (resistant to outliers)
  • Defensive checks for numeric column presence
  • Returns scaled NumPy arrays (X_train_scaled, X_test_scaled)

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame operations
from sklearn.preprocessing import StandardScaler, RobustScaler  # Scaling utilities
import numpy as np  # For numeric array handling
import warnings  # For fallback alerts


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: FeatureScaler
# ───────────────────────────────────────────────────────────────────────────────
class FeatureScaler:
    """
    Scales numeric features in X_train and X_test using chosen method.
    """

    def __init__(self, method: str = "standard"):
        """
        Initialize scaler type.

        Args:
            method (str): Scaling strategy. Options: 'standard' or 'robust'.
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError("❌ Invalid method: choose 'standard' or 'robust'.")

    def scale(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply scaling to numeric features in X_train and X_test.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.

        Returns:
            tuple: (X_train_scaled, X_test_scaled) as NumPy arrays.
        """
        # Identify numeric columns for scaling
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            warnings.warn(
                "⚠️ No numeric columns found to scale. Returning unmodified arrays."
            )
            return X_train.values, X_test.values

        # Fit scaler on training set only
        self.scaler.fit(X_train[numeric_cols])

        # Apply transformation
        X_train_scaled = self.scaler.transform(X_train[numeric_cols])
        X_test_scaled = self.scaler.transform(X_test[numeric_cols])

        return X_train_scaled, X_test_scaled
