"""
gender_risk_profiler.py – Gender-Based Risk Metrics (B5W3)
------------------------------------------------------------------------------
Computes and visualizes insurance risk segmentation by gender group.

Core responsibilities:
  • Validates required gender-related fields
  • Aggregates TotalClaims and TotalPremium by gender
  • Computes LossRatio for each gender group
  • Generates bar and box plots for claims and premiums

Used in: Task 1 – EDA Layer 6 (Risk segmentation by Gender)

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import Optional

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # Data manipulation
import seaborn as sns  # For plotting
import matplotlib.pyplot as plt  # For figure rendering


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: GenderRiskProfiler
# ───────────────────────────────────────────────────────────────────────────────
class GenderRiskProfiler:
    """
    Computes and visualizes loss ratio insights by gender.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the profiler with a validated DataFrame.

        Args:
            df (pd.DataFrame): The insurance dataset with gender fields.

        Raises:
            ValueError: If required columns are missing.
        """
        # ✅ Required columns for analysis
        required_cols = {"Gender", "TotalClaims", "TotalPremium"}

        # ❌ Raise error if any required field is missing
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(
                f"❌ Missing required columns for gender analysis: {missing}"
            )

        # 🧼 Drop rows missing gender values
        self.df = df.dropna(subset=["Gender"]).copy()

    def compute_loss_ratios(self) -> pd.DataFrame:
        """
        Aggregates claims and premiums by gender and computes loss ratios.

        Returns:
            pd.DataFrame: Aggregated metrics with loss ratios.
        """
        # 📊 Group and compute metrics
        agg = (
            self.df.groupby("Gender")[["TotalClaims", "TotalPremium"]]
            .sum()
            .assign(
                LossRatio=lambda x: (x["TotalClaims"] / x["TotalPremium"]).replace(
                    [float("inf"), -float("inf")], float("nan")
                )
            )
            .reset_index()
            .sort_values("LossRatio", ascending=False)
        )
        return agg

    def plot_loss_ratio_bar(self, gender_agg: Optional[pd.DataFrame] = None) -> None:
        """
        Plots loss ratios by gender as a bar chart.

        Args:
            gender_agg (pd.DataFrame, optional): Precomputed metrics. Will compute if None.
        """
        # 🧠 Use provided data or compute fresh
        if gender_agg is None:
            gender_agg = self.compute_loss_ratios()

        # 🎨 Barplot
        plt.figure(figsize=(8, 5))
        sns.barplot(data=gender_agg, x="LossRatio", y="Gender", palette="Set2")
        plt.title("⚖️ Loss Ratio by Gender", fontsize=14)
        plt.xlabel("Loss Ratio (Claims ÷ Premium)", fontsize=11)
        plt.ylabel("Gender", fontsize=11)
        plt.tight_layout()
        plt.show()

    def plot_distribution_boxplots(self) -> None:
        """
        Plots boxplots of claims and premiums by gender for distribution insights.
        """
        # 🎨 Subplot canvas
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # 💸 Claims boxplot
        sns.boxplot(
            data=self.df, x="Gender", y="TotalClaims", ax=axs[0], palette="pastel"
        )
        axs[0].set_title("💸 Distribution of Total Claims by Gender", fontsize=12)
        axs[0].set_xlabel("Gender", fontsize=10)
        axs[0].set_ylabel("Total Claims (ZAR)", fontsize=10)

        # 💰 Premiums boxplot
        sns.boxplot(
            data=self.df, x="Gender", y="TotalPremium", ax=axs[1], palette="muted"
        )
        axs[1].set_title("💰 Distribution of Total Premium by Gender", fontsize=12)
        axs[1].set_xlabel("Gender", fontsize=10)
        axs[1].set_ylabel("Total Premium (ZAR)", fontsize=10)

        plt.tight_layout()
        plt.show()
