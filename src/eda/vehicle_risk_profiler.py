"""
vehicle_risk_profiler.py – Vehicle Make & Model Risk Analysis (B5W3)
------------------------------------------------------------------------------
Computes loss ratios by car Make and Model using the AlphaCare dataset.

Key Functions:
  • Validates required vehicle fields
  • Aggregates TotalClaims / TotalPremium by Make and Model
  • Flags top 15 highest- and lowest-risk vehicle makes
  • Visualizes bar plots for risk segmentation (Make & Model)

Used in Task 1 to identify high-risk vehicle segments and guide underwriting.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import Tuple  # For type hints in method returns

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # Data manipulation
import seaborn as sns  # Plotting
import matplotlib.pyplot as plt  # Visualization


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: VehicleRiskProfiler
# ───────────────────────────────────────────────────────────────────────────────
class VehicleRiskProfiler:
    """
    Computes and visualizes vehicle-level insurance risk by Make and Model.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the profiler and validates vehicle columns.

        Args:
            df (pd.DataFrame): Input insurance dataframe.

        Raises:
            ValueError: If required columns are missing.
        """
        # Define required columns
        required_cols = {"make", "Model", "TotalClaims", "TotalPremium"}

        # Check for missing columns
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"❌ Missing vehicle-related columns: {missing}")

        # Drop rows with missing make/model
        self.df = df.dropna(subset=["make", "Model"]).copy()

    def compute_loss_ratios(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Aggregates loss ratios by Make and Model.

        Returns:
            Tuple containing:
            - top 15 risky makes
            - top 15 safe makes
            - top 15 risky models
        """
        # Aggregate by make
        make_agg = (
            self.df.groupby("make")[["TotalClaims", "TotalPremium"]]
            .sum()
            .assign(
                LossRatio=lambda x: (x["TotalClaims"] / x["TotalPremium"]).replace(
                    [float("inf"), -float("inf")], float("nan")
                )
            )
            .reset_index()
        )

        # Get extremes
        top_risky_makes = make_agg.sort_values("LossRatio", ascending=False).head(15)
        top_safe_makes = make_agg.sort_values("LossRatio", ascending=True).head(15)

        # Aggregate by model
        model_agg = (
            self.df.groupby("Model")[["TotalClaims", "TotalPremium"]]
            .sum()
            .assign(
                LossRatio=lambda x: (x["TotalClaims"] / x["TotalPremium"]).replace(
                    [float("inf"), -float("inf")], float("nan")
                )
            )
            .reset_index()
            .sort_values("LossRatio", ascending=False)
            .head(15)
        )

        return top_risky_makes, top_safe_makes, model_agg

    def plot_vehicle_risk(self):
        """
        Generates bar plots for top risky/safe vehicle Makes and risky Models.
        """
        # Get loss ratio tables
        risky_makes, safe_makes, risky_models = self.compute_loss_ratios()

        # Plot: Riskiest Makes
        plt.figure(figsize=(12, 5))
        sns.barplot(data=risky_makes, x="LossRatio", y="make", palette="rocket")
        plt.title("🚩 Top 15 Vehicle Makes by Loss Ratio", fontsize=14)
        plt.xlabel("Loss Ratio (Claims / Premium)", fontsize=11)
        plt.ylabel("Vehicle Make", fontsize=11)
        plt.tight_layout()
        plt.show()

        # Plot: Safest Makes
        plt.figure(figsize=(12, 5))
        sns.barplot(data=safe_makes, x="LossRatio", y="make", palette="crest")
        plt.title("🛡️ 15 Safest Vehicle Makes by Loss Ratio", fontsize=14)
        plt.xlabel("Loss Ratio", fontsize=11)
        plt.ylabel("Vehicle Make", fontsize=11)
        plt.tight_layout()
        plt.show()

        # Plot: Riskiest Models
        plt.figure(figsize=(12, 5))
        sns.barplot(data=risky_models, x="LossRatio", y="Model", palette="flare")
        plt.title("🔧 Top 15 Vehicle Models by Loss Ratio", fontsize=14)
        plt.xlabel("Loss Ratio", fontsize=11)
        plt.ylabel("Vehicle Model", fontsize=11)
        plt.tight_layout()
        plt.show()
