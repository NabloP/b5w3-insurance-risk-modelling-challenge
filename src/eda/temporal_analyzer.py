"""
temporal_analyzer.py – Time-Based Claims & Premium Analysis (B5W3)
------------------------------------------------------------------------------
Performs temporal aggregation and visualization of monthly insurance metrics:
  • Total Premiums
  • Total Claims
  • Monthly Loss Ratios
  • Monthly Claim Count (NEW)

Handles datetime parsing, missing value coercion, and visual diagnostics.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame operations
import seaborn as sns  # For styled plots
import matplotlib.pyplot as plt  # For figure rendering


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: TemporalClaimsAnalyzer
# ───────────────────────────────────────────────────────────────────────────────
class TemporalClaimsAnalyzer:
    """
    Analyzes trends over time in claims, premiums, loss ratio, and claim volume.
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

        self.df = df.copy()  # Store original data safely
        self.cleaned_df = None  # Will store cleaned data with parsed dates
        self.monthly_df = None  # Will store aggregated monthly KPIs

    def prepare_temporal_data(self) -> pd.DataFrame:
        """
        Parses 'TransactionMonth' into datetime format, drops bad rows.

        Returns:
            pd.DataFrame: Cleaned DataFrame with valid 'TransactionMonth' values.
        """
        # Coerce invalid dates to NaT
        self.df["TransactionMonth"] = pd.to_datetime(
            self.df["TransactionMonth"], errors="coerce"
        )

        # Drop rows with missing TransactionMonth
        self.cleaned_df = self.df.dropna(subset=["TransactionMonth"]).copy()

        if self.cleaned_df.empty:
            raise ValueError("All rows dropped after parsing 'TransactionMonth'.")

        return self.cleaned_df

    def aggregate_monthly_metrics(self) -> pd.DataFrame:
        """
        Aggregates TotalClaims and TotalPremium monthly and computes loss ratio.

        Returns:
            pd.DataFrame: Aggregated DataFrame by 'TransactionMonth'.
        """
        if self.cleaned_df is None:
            self.prepare_temporal_data()

        # Group by month and sum financial columns
        monthly = (
            self.cleaned_df.groupby("TransactionMonth")[["TotalClaims", "TotalPremium"]]
            .sum()
            .reset_index()
        )

        # Compute monthly loss ratio
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

        # Create subplots for each KPI
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # 📈 Monthly Premiums
        sns.lineplot(
            data=self.monthly_df, x="TransactionMonth", y="TotalPremium", ax=axs[0]
        )
        axs[0].set_title("💰 Monthly Total Premiums", fontsize=12)
        axs[0].set_ylabel("ZAR")

        # 📉 Monthly Claims
        sns.lineplot(
            data=self.monthly_df, x="TransactionMonth", y="TotalClaims", ax=axs[1]
        )
        axs[1].set_title("📉 Monthly Total Claims", fontsize=12)
        axs[1].set_ylabel("ZAR")

        # 📊 Monthly Loss Ratio
        sns.lineplot(
            data=self.monthly_df, x="TransactionMonth", y="LossRatio", ax=axs[2]
        )
        axs[2].set_title("📊 Monthly Loss Ratio (Claims ÷ Premiums)", fontsize=12)
        axs[2].set_ylabel("Ratio")

        plt.xlabel("Transaction Month")
        plt.tight_layout()
        plt.show()

    def plot_monthly_claim_count(self) -> None:
        """
        📅 NEW: Plots the number of policies with > 0 claims each month.
        """
        if self.cleaned_df is None:
            self.prepare_temporal_data()

        try:
            # Filter to claim-making rows only
            df_claims = self.cleaned_df[self.cleaned_df["TotalClaims"] > 0]

            # Count number of claim rows per month
            monthly_counts = (
                df_claims.groupby("TransactionMonth")
                .size()
                .reset_index(name="ClaimCount")
            )

            # Line plot of monthly claim volume
            plt.figure(figsize=(10, 5))
            sns.lineplot(
                data=monthly_counts,
                x="TransactionMonth",
                y="ClaimCount",
                marker="o",
                linewidth=2,
            )
            plt.title("📅 Monthly Count of Claims Filed")
            plt.xlabel("Transaction Month")
            plt.ylabel("Number of Claims > 0")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Failed to plot monthly claim count: {e}")

    def report_missing_months(self) -> None:
        """
        Prints the % of rows with missing or invalid 'TransactionMonth' values.
        """
        missing_count = self.df["TransactionMonth"].isna().sum()
        total = len(self.df)

        print(
            f"🧾 Missing or invalid 'TransactionMonth': {missing_count} rows "
            f"({missing_count / total:.2%})"
        )
