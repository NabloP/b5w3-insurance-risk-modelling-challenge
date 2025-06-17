"""
temporal_analyzer.py – Time-Based Claims & Premium Analysis (B5W3)
------------------------------------------------------------------------------
Performs temporal aggregation and visualization of monthly insurance metrics:
  • Total Premiums
  • Total Claims
  • Monthly Loss Ratios

Handles datetime parsing, missing value coercion, and output visualization.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: TemporalClaimsAnalyzer
# ───────────────────────────────────────────────────────────────────────────────
class TemporalClaimsAnalyzer:
    """
    Analyzes trends over time in claims, premiums, and loss ratio.
    Requires a 'TransactionMonth' column in the input DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with a raw DataFrame.

        Args:
            df (pd.DataFrame): Insurance dataset with time and numeric features.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If 'TransactionMonth' column is missing.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(df)}")

        if "TransactionMonth" not in df.columns:
            raise ValueError("Missing required column: 'TransactionMonth'")

        self.df = df.copy()
        self.cleaned_df = None
        self.monthly_df = None

    def prepare_temporal_data(self) -> pd.DataFrame:
        """
        Parses 'TransactionMonth' into datetime, drops bad rows.

        Returns:
            pd.DataFrame: Cleaned DataFrame with valid TransactionMonth values.
        """
        self.df["TransactionMonth"] = pd.to_datetime(
            self.df["TransactionMonth"], errors="coerce"
        )
        self.cleaned_df = self.df.dropna(subset=["TransactionMonth"]).copy()

        if self.cleaned_df.empty:
            raise ValueError("All rows dropped after parsing 'TransactionMonth'.")

        return self.cleaned_df

    def aggregate_monthly_metrics(self) -> pd.DataFrame:
        """
        Aggregates TotalClaims and TotalPremium monthly and computes loss ratio.

        Returns:
            pd.DataFrame: Aggregated DataFrame by TransactionMonth.
        """
        if self.cleaned_df is None:
            self.prepare_temporal_data()

        monthly = (
            self.cleaned_df.groupby("TransactionMonth")[["TotalClaims", "TotalPremium"]]
            .sum()
            .reset_index()
        )

        monthly["LossRatio"] = (
            monthly["TotalClaims"] / monthly["TotalPremium"]
        ).replace([float("inf"), -float("inf")], float("nan"))

        self.monthly_df = monthly
        return self.monthly_df

    def plot_trend_lines(self) -> None:
        """
        Plots line charts of monthly premiums, claims, and loss ratio.
        """
        if self.monthly_df is None:
            self.aggregate_monthly_metrics()

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        sns.lineplot(
            data=self.monthly_df, x="TransactionMonth", y="TotalPremium", ax=axs[0]
        )
        axs[0].set_title("💰 Monthly Total Premiums", fontsize=12)
        axs[0].set_ylabel("ZAR")

        sns.lineplot(
            data=self.monthly_df, x="TransactionMonth", y="TotalClaims", ax=axs[1]
        )
        axs[1].set_title("📉 Monthly Total Claims", fontsize=12)
        axs[1].set_ylabel("ZAR")

        sns.lineplot(
            data=self.monthly_df, x="TransactionMonth", y="LossRatio", ax=axs[2]
        )
        axs[2].set_title("📊 Monthly Loss Ratio (Claims ÷ Premiums)", fontsize=12)
        axs[2].set_ylabel("Ratio")

        plt.xlabel("Transaction Month")
        plt.tight_layout()
        plt.show()

    def report_missing_months(self) -> None:
        """
        Prints the % of rows with missing or invalid TransactionMonth entries.
        """
        missing_count = self.df["TransactionMonth"].isna().sum()
        total = len(self.df)
        print(
            f"🧾 Missing or invalid 'TransactionMonth' entries: {missing_count} rows "
            f"({missing_count / total:.2%})"
        )
