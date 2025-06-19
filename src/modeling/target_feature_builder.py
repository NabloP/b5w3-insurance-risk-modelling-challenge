"""
target_feature_builder.py – Task 4 Target & Feature Preparation (B5W3)
------------------------------------------------------------------------------
Prepares the AlphaCare insurance dataset for predictive modeling by:

  • Dropping known leakage columns (IDs and claim-derived)
  • Defining binary and regression targets
  • Isolating numeric features for modeling
  • Performing class imbalance diagnostics

Designed for defensible use across multiple modeling pipelines (e.g., ClaimFrequency, Margin).
Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import warnings  # For handling suppressed errors

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame manipulation


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: TargetFeatureBuilder
# ───────────────────────────────────────────────────────────────────────────────
class TargetFeatureBuilder:
    """
    Constructs modeling frame by defining targets and extracting features.
    Modular, reusable class for Task 4 claim prediction workflows.
    """

    def __init__(
        self,
        classification_target: str = "ClaimFrequency",
        regression_target: str = "Margin",
    ):
        # Define target variable names (user can override)
        self.classification_target = classification_target
        self.regression_target = regression_target

        # Known leakage columns that should never be used as features
        self.leakage_cols = [
            "PolicyID",
            "UnderwrittenCoverID",  # Unique identifiers
            "TotalClaims",
            "TotalPremium",
            "NumClaims",  # Derived from label logic
        ]

    def prepare(self, df: pd.DataFrame) -> tuple:
        """
        Main method to drop leakage columns, extract targets, and isolate features.

        Args:
            df (pd.DataFrame): Cleaned insurance DataFrame.

        Returns:
            tuple: (X, y_class, y_reg) = features, binary target, regression target
        """
        df = df.copy()  # Defensive copy to avoid in-place mutations

        # ────────────────────────────────────────────────────────────────────
        # Step 1: Drop leakage columns (ignore if missing)
        # ────────────────────────────────────────────────────────────────────
        present_leaks = [col for col in self.leakage_cols if col in df.columns]
        df.drop(columns=present_leaks, inplace=True)
        print(f"🧹 Dropped leakage columns: {present_leaks} → New shape: {df.shape}")

        # ────────────────────────────────────────────────────────────────────
        # Step 2: Validate target columns exist
        # ────────────────────────────────────────────────────────────────────
        missing_targets = [
            col
            for col in [self.classification_target, self.regression_target]
            if col not in df.columns
        ]
        if missing_targets:
            raise ValueError(f"❌ Missing required target columns: {missing_targets}")

        # ────────────────────────────────────────────────────────────────────
        # Step 3: Extract target columns
        # ────────────────────────────────────────────────────────────────────
        y_class = df[self.classification_target]  # Binary classification target
        y_reg = df[self.regression_target]  # Continuous regression target

        # ────────────────────────────────────────────────────────────────────
        # Step 4: Remove targets from features
        # ────────────────────────────────────────────────────────────────────
        X = df.drop(columns=[self.classification_target, self.regression_target])
        print(
            f"✅ Features isolated → X shape: {X.shape}, Targets: y_class={y_class.shape}, y_reg={y_reg.shape}"
        )

        # ────────────────────────────────────────────────────────────────────
        # Step 5: Report class balance (ClaimFrequency)
        # ────────────────────────────────────────────────────────────────────
        try:
            class_dist = y_class.value_counts(normalize=True).round(3)
            print("📊 ClaimFrequency Distribution:")
            print(class_dist.to_string())
            if class_dist.min() < 0.3:
                print("⚠️ Warning: Significant class imbalance detected.")
        except Exception as e:
            warnings.warn(f"⚠️ Class distribution diagnostics failed: {e}")

        return X, y_class, y_reg
