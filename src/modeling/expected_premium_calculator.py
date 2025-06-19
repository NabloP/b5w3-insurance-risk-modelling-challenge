"""
expected_premium_calculator.py – Task 4 Premium Estimator (B5W3)
===============================================================================
Combines predicted claim probability and expected claim severity to calculate
an expected insurance premium per customer using the equation:

    Premium = P(Claim) × E[ClaimAmount] + Expenses + Margin

📌 Core Responsibilities:
    • Merge classifier and regressor predictions
    • Compute expected premium per row
    • Add optional fixed margin or loading
    • Return final DataFrame with premium drivers

Author: Nabil Mohamed
Challenge: B5W3 – Insurance Risk Analytics & Predictive Modeling
Company: AlphaCare Insurance Solutions (ACIS)
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame operations
import numpy as np  # For numeric logic


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class Definition
# ───────────────────────────────────────────────────────────────────────────────
class ExpectedPremiumCalculator:
    """
    Combines classification and regression outputs to compute predicted insurance premiums.
    """

    def __init__(self, margin: float = 0.0):
        """
        Initialize the calculator with optional fixed margin or expense factor.

        Args:
            margin (float): Optional additive margin to include in the premium (e.g., 50 units)
        """
        self.margin = margin

    def compute_expected_premium(
        self,
        df: pd.DataFrame,
        prob_col: str = "predicted_claim_prob",
        severity_col: str = "predicted_claim_severity",
    ) -> pd.DataFrame:
        """
        Computes expected premium using predicted probability and severity.

        Args:
            df (pd.DataFrame): DataFrame with probability and severity predictions
            prob_col (str): Column name containing P(Claim)
            severity_col (str): Column name containing E[ClaimAmount]

        Returns:
            pd.DataFrame: Original DataFrame with new 'expected_premium' column
        """
        # Check column existence
        if prob_col not in df.columns:
            raise ValueError(f"Missing column '{prob_col}' for claim probability.")
        if severity_col not in df.columns:
            raise ValueError(f"Missing column '{severity_col}' for claim severity.")

        # Defensive: ensure no NaNs before multiplying
        df = df.copy()
        df[prob_col] = df[prob_col].fillna(0)
        df[severity_col] = df[severity_col].fillna(0)

        # Compute expected premium per row
        df["expected_premium"] = (df[prob_col] * df[severity_col]) + self.margin

        return df
