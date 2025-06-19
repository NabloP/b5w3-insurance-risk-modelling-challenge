"""
class_balancer.py – Task 4 Class Balancer via SMOTE (B5W3)
-------------------------------------------------------------------------------
Balances the training dataset using Synthetic Minority Over-sampling Technique (SMOTE),
targeting the binary classification label `ClaimFrequency`.

Core responsibilities:
  • Applies SMOTE only to training data (post-split)
  • Handles extreme class imbalance with tunable parameters (e.g., k_neighbors=1)
  • Preserves alignment with y_reg_train if needed in downstream logic
  • Outputs balanced X and y_class for classification tasks
  • Warns user if SMOTE fails due to lack of minority support or sampling errors

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # DataFrame manipulation
import numpy as np  # Numerical array handling
from imblearn.over_sampling import SMOTE  # SMOTE for synthetic class balancing
import warnings  # Defensive logging for edge cases


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: ClassBalancer
# ───────────────────────────────────────────────────────────────────────────────
class ClassBalancer:
    """
    Applies SMOTE to oversample the minority class in binary classification problems.
    Handles highly imbalanced datasets where traditional sampling may fail.
    """

    def __init__(
        self,
        random_state: int = 42,
        k_neighbors: int = 1,
        sampling_strategy: str = "auto",
    ):
        """
        Initialize SMOTE configuration with customizable resampling behavior.

        Args:
            random_state (int): Random seed for reproducibility.
            k_neighbors (int): Number of nearest neighbors for synthetic sample generation.
            sampling_strategy (str): Defines sampling strategy, e.g., "auto", "minority", dict.
        """
        self.smote = SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )

    def balance(
        self, X_train: pd.DataFrame, y_class_train: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Balances the dataset using SMOTE and returns oversampled versions of inputs.

        Args:
            X_train (pd.DataFrame): Feature matrix (training set only).
            y_class_train (pd.Series): Classification labels (training set only).

        Returns:
            tuple:
                X_balanced (pd.DataFrame): Oversampled feature matrix.
                y_balanced (pd.Series): Oversampled binary labels.
        """
        try:
            # Run SMOTE to generate synthetic samples for minority class
            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_class_train)

            # Wrap outputs in pandas structures for downstream compatibility
            X_balanced = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_balanced = pd.Series(y_resampled, name=y_class_train.name)

            # Show updated class distribution after balancing
            print("✅ SMOTE applied → New class distribution:")
            print(y_balanced.value_counts(normalize=True).round(3).to_string())

            # Additional alert if class imbalance still persists significantly
            if y_balanced.value_counts(normalize=True).min() < 0.3:
                warnings.warn("⚠️ Class imbalance remains significant after SMOTE.")

            return X_balanced, y_balanced

        except ValueError as e:
            # Log error if SMOTE fails (e.g., no minority samples present)
            warnings.warn(f"❌ SMOTE failed: {e}")
            return X_train, y_class_train  # Fallback to original data
