"""
metric_definitions.py – Task 3 Metric Derivation (B5W3)
------------------------------------------------------------------------------
Computes core KPIs for statistical hypothesis testing and risk modeling
within AlphaCare’s insurance dataset. All metrics are derived defensively
and made reproducible for downstream segmentation and modeling tasks.

Core responsibilities:
  • ClaimFrequency – Binary flag for whether a claim occurred
  • ClaimSeverity – Amount of claim if one occurred, else NaN
  • Margin – Net profitability per policy (TotalPremium - TotalClaims)

Used in Task 3 (Hypothesis Testing) and Task 4 (Predictive Modeling).

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import warnings  # Optional: suppress np warnings

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame manipulation
import numpy as np  # For numerical operations


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: MetricDefiner
# ───────────────────────────────────────────────────────────────────────────────
class MetricDefiner:
    """
    OOP utility for enriching insurance datasets with key performance metrics.
    All logic is defensively coded and intended for statistical audit and
    model training input preparation.
    """

    def __init__(self):
        """Initialize the MetricDefiner instance (no params required)."""
        pass

    def add_claim_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds binary 'ClaimFrequency' column based on TotalClaims > 0.

        Args:
            df (pd.DataFrame): Raw insurance dataset.

        Returns:
            pd.DataFrame: Copy of df with ClaimFrequency column added.

        Raises:
            KeyError: If 'TotalClaims' column is missing.
        """
        if "TotalClaims" not in df.columns:
            raise KeyError(
                "Column 'TotalClaims' is required to compute ClaimFrequency."
            )

        df = df.copy()
        df["ClaimFrequency"] = np.where(df["TotalClaims"] > 0, 1, 0)

        return df

    def add_claim_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'ClaimSeverity' column as the claim amount if TotalClaims > 0, else NaN.

        Args:
            df (pd.DataFrame): Raw or enriched insurance dataset.

        Returns:
            pd.DataFrame: Copy of df with ClaimSeverity column added.

        Raises:
            KeyError: If 'TotalClaims' column is missing.
        """
        if "TotalClaims" not in df.columns:
            raise KeyError("Column 'TotalClaims' is required to compute ClaimSeverity.")

        df = df.copy()
        df["ClaimSeverity"] = np.where(df["TotalClaims"] > 0, df["TotalClaims"], np.nan)

        return df

    def add_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'Margin' column as TotalPremium - TotalClaims.

        Args:
            df (pd.DataFrame): Raw or enriched insurance dataset.

        Returns:
            pd.DataFrame: Copy of df with Margin column added.

        Raises:
            KeyError: If required columns are missing.
        """
        for col in ["TotalPremium", "TotalClaims"]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' is required to compute Margin.")

        df = df.copy()
        df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

        return df

    def enrich_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method to apply all metric enrichments at once.

        Args:
            df (pd.DataFrame): Raw insurance dataset.

        Returns:
            pd.DataFrame: Copy of df with all KPI columns added.
        """
        df = self.add_claim_frequency(df)
        df = self.add_claim_severity(df)
        df = self.add_margin(df)
        return df
