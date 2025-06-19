"""
train_test_splitter.py – Task 4 Train/Test Split Utility (B5W3)
------------------------------------------------------------------------------
Splits the AlphaCare insurance dataset into stratified training and test sets
for dual modeling tasks: Claim Frequency (classification) and Margin (regression).

Core responsibilities:
  • Stratified sampling using ClaimFrequency to ensure class balance
  • Synchronized splitting across classification and regression targets
  • Defensive programming with shape validation
  • Flexible configuration of test size and random state
  • Outputs 6 objects: features + targets for train/test

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # DataFrame operations
from sklearn.model_selection import train_test_split  # Sklearn utility for splitting


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: TrainTestSplitter
# ───────────────────────────────────────────────────────────────────────────────
class TrainTestSplitter:
    """
    Handles stratified train/test splits for AlphaCare classification and regression tasks.
    """

    def __init__(self, test_size=0.2, random_state=42):
        """
        Constructor to initialize splitter configuration.

        Args:
            test_size (float): Fraction of data to reserve for testing (e.g., 0.2 = 20%).
            random_state (int): Random seed for reproducibility.
        """
        self.test_size = test_size  # Store test set size
        self.random_state = random_state  # Set seed for consistent splits

    def split(self, X: pd.DataFrame, y_class: pd.Series, y_reg: pd.Series):
        """
        Splits X and y targets into stratified train/test sets.

        Args:
            X (pd.DataFrame): Features for model input.
            y_class (pd.Series): Binary classification target (ClaimFrequency).
            y_reg (pd.Series): Continuous regression target (Margin).

        Returns:
            Tuple of 6 objects:
                - X_train, X_test
                - y_class_train, y_class_test
                - y_reg_train, y_reg_test
        """

        # Defensive check: validate input types
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y_class, pd.Series) or not isinstance(y_reg, pd.Series):
            raise TypeError("y_class and y_reg must be pandas Series.")

        # Defensive check: matching lengths
        if not (len(X) == len(y_class) == len(y_reg)):
            raise ValueError("Input lengths for X, y_class, and y_reg must match.")

        # ✅ Step 1: Perform stratified train/test split on classification target
        (
            X_train,
            X_test,
            y_class_train,
            y_class_test,
        ) = train_test_split(
            X,  # Features
            y_class,  # Stratify based on classification target
            test_size=self.test_size,  # Fraction reserved for testing
            stratify=y_class,  # Preserve target class distribution
            random_state=self.random_state,  # Ensure reproducible results
        )

        # ✅ Step 2: Align regression targets to same train/test index split
        y_reg_train = y_reg.loc[
            y_class_train.index
        ]  # Align regression training targets
        y_reg_test = y_reg.loc[y_class_test.index]  # Align regression test targets

        # ✅ Step 3: Return full tuple of outputs
        return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
