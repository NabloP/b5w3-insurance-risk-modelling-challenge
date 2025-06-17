"""
iqr_outlier_detector.py – IQR-Based Outlier Detection Module (B5W3)
------------------------------------------------------------------------------
Identifies outliers in key numeric columns using the Interquartile Range (IQR) rule.

Key Responsibilities:
  • Validates presence of target numeric columns
  • Computes IQR-based thresholds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
  • Flags and summarizes outlier rows per column
  • Visualizes outliers using boxplots

Used in Task 1 EDA to flag unusual claims, premiums, and custom valuation entries.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import Dict

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame operations
import matplotlib.pyplot as plt  # For plotting boxplots
import seaborn as sns  # For styled statistical plots


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: IQRBasedOutlierDetector
# ───────────────────────────────────────────────────────────────────────────────
class IQRBasedOutlierDetector:
    """
    Flags outliers using the Interquartile Range (IQR) method.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the dataset and validate column types.

        Args:
            df (pd.DataFrame): Insurance dataset with numeric fields.

        Raises:
            TypeError: If input is not a DataFrame.
            ValueError: If no expected numeric columns are found.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        self.df = df.copy()  # Work on a defensive copy

        # Define numeric columns to check
        self.numeric_cols = [
            col
            for col in ["TotalClaims", "TotalPremium", "CustomValueEstimate"]
            if col in self.df.columns
        ]

        if not self.numeric_cols:
            raise ValueError("❌ No valid numeric columns found for IQR analysis.")

        self.outlier_flags: Dict[str, pd.Series] = {}  # Holds outlier boolean masks

    def detect_outliers(self) -> None:
        """
        Apply the IQR method to each numeric column and flag outliers.
        Generates inline boxplots with annotations.
        """
        for col in self.numeric_cols:
            # Compute quartiles and IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier boundaries
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # Flag values outside bounds
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            self.outlier_flags[col] = mask

            # Print outlier summary
            count = mask.sum()
            pct = (count / len(self.df)) * 100
            print(f"⚠️ {col}: {count:,} outliers flagged ({pct:.2f}%)")

            # Plot boxplot for visual inspection
            plt.figure(figsize=(8, 1.5))
            sns.boxplot(data=self.df, x=col, color="salmon")
            plt.title(f"🧼 Boxplot for {col} (IQR Outlier Detection)", fontsize=11)
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()

    def get_outlier_flags(self) -> Dict[str, pd.Series]:
        """
        Return the dictionary of outlier masks.

        Returns:
            Dict[str, pd.Series]: Mapping of column name → boolean mask.
        """
        return self.outlier_flags
