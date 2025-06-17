"""
distribution_analyzer.py – Numeric Distribution Analyzer (B5W3)
------------------------------------------------------------------------------
Performs descriptive statistical audits on numeric features in a DataFrame.
Includes summary stats, skewness/kurtosis diagnostics, and stylized outputs.

Core responsibilities:
  • Computes descriptive stats for all float/int columns
  • Evaluates distribution shape via skewness and kurtosis
  • Flags non-normal variables needing transformation
  • Provides robust fallback logic for empty or malformed data

Used in Task 1 EDA and transformation readiness checks.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import List  # For type annotations

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # Core DataFrame manipulation
import numpy as np  # Required for skew/kurtosis operations
from pandas.io.formats.style import Styler  # For styled DataFrame returns
from IPython.display import display  # For notebook-friendly output rendering


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: DistributionAnalyzer
# ───────────────────────────────────────────────────────────────────────────────
class DistributionAnalyzer:
    """
    Class to audit the shape and distribution of numeric columns in a DataFrame.
    Provides descriptive stats, skewness, kurtosis, and visual flags.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame to analyze.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.

        Raises:
            TypeError: If input is not a DataFrame.
            ValueError: If no numeric columns are found.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"`df` must be a pandas DataFrame, got {type(df)}")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if not numeric_cols:
            raise ValueError("No numeric columns found for distribution analysis.")

        self.df = df
        self.numeric_cols = numeric_cols

    def describe_numerics(self) -> pd.DataFrame:
        """
        Generate standard descriptive statistics for numeric features.

        Returns:
            pd.DataFrame: Transposed describe() table for numeric columns.
        """
        try:
            return self.df[self.numeric_cols].describe().T
        except Exception as e:
            raise RuntimeError(f"Failed to compute descriptive stats: {e}")

    def compute_skew_kurt(self) -> pd.DataFrame:
        """
        Compute skewness and kurtosis for all numeric columns.

        Returns:
            pd.DataFrame: Sorted table of skewness and kurtosis values.
        """
        try:
            result = pd.DataFrame(
                {
                    "Skewness": self.df[self.numeric_cols].skew(),
                    "Kurtosis": self.df[self.numeric_cols].kurt(),
                }
            ).sort_values("Skewness", ascending=False)
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute skew/kurtosis: {e}")

    def styled_skew_kurt(self) -> Styler:
        """
        Return a styled display of skewness/kurtosis diagnostics.

        Returns:
            Styler: Gradient-colored skewness/kurtosis table.
        """
        df_sk = self.compute_skew_kurt()
        try:
            return (
                df_sk.style.background_gradient(cmap="PuBu", subset=["Skewness"])
                .background_gradient(cmap="BuPu", subset=["Kurtosis"])
                .format({"Skewness": "{:.2f}", "Kurtosis": "{:.2f}"})
            )
        except Exception as e:
            raise RuntimeError(f"Failed to style skew/kurtosis display: {e}")

    def print_distribution_warnings(
        self, skew_threshold: float = 2.0, kurt_threshold: float = 10.0
    ) -> None:
        """
        Print warnings for features with extreme skew/kurtosis.

        Args:
            skew_threshold (float): Absolute skewness threshold.
            kurt_threshold (float): Excess kurtosis threshold.
        """
        try:
            df_sk = self.compute_skew_kurt()
            extreme = df_sk[
                (df_sk["Skewness"].abs() > skew_threshold)
                | (df_sk["Kurtosis"] > kurt_threshold)
            ]

            if not extreme.empty:
                print(
                    "⚠️ Features with extreme skew or kurtosis (possible transformation needed):"
                )
                display(
                    extreme.style.applymap(
                        lambda val: "background-color: salmon;",
                        subset=["Skewness", "Kurtosis"],
                    ).format(precision=2)
                )
            else:
                print("✅ No features with extreme skew or kurtosis detected.")

        except Exception as e:
            raise RuntimeError(f"Failed to analyze distribution shape warnings: {e}")
