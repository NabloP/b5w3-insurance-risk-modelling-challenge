"""
plan_feature_risk_profiler.py – Plan-Based Risk Segment Analyzer (B5W3)
------------------------------------------------------------------------------
Analyzes loss ratios across insurance plan features such as CoverType,
CoverGroup, and TermFrequency. Identifies structural plan-level drivers of risk.

Core responsibilities:
  • Validates presence of segmentation columns
  • Aggregates TotalClaims and TotalPremium by feature value
  • Computes LossRatio = Claims / Premium with NaN safeguards
  • Generates bar plots per segmentation feature for loss ratio comparison

Used in Task 1 EDA to support pricing strategy refinement based on product tiers.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import List

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For bar plots
import seaborn as sns  # For styled plots


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: PlanFeatureRiskProfiler
# ───────────────────────────────────────────────────────────────────────────────
class PlanFeatureRiskProfiler:
    """
    Evaluates loss ratio distribution across product plan features.
    Segments risk by CoverType, CoverGroup, TermFrequency.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the insurance dataset.

        Args:
            df (pd.DataFrame): Cleaned DataFrame with claim/premium metrics.

        Raises:
            TypeError: If input is not a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a DataFrame, got {type(df)}")

        self.df = df.copy()  # Preserve original data
        self.segment_cols = [
            "CoverType",
            "CoverGroup",
            "TermFrequency",
        ]  # Expected segmentation fields

    def analyze(self) -> None:
        """
        Analyze and plot risk segmentation across available plan features.
        Raises:
            ValueError: If none of the expected segmentation columns exist.
        """
        # ✅ Identify which expected segmenting columns exist in dataset
        valid_segments: List[str] = [
            col for col in self.segment_cols if col in self.df.columns
        ]

        # ❌ Raise if no valid segmenting fields found
        if not valid_segments:
            raise ValueError(
                "❌ None of the expected segmentation columns are present in the dataset."
            )

        # 📊 Iterate through valid segmenting features and visualize risk
        for segment in valid_segments:
            print(f"\n📂 Segmenting by: {segment}\n")

            # 🧼 Drop missing entries in the segment field
            seg_df = self.df.dropna(subset=[segment]).copy()

            # 🧮 Aggregate total claims and premium by segment, compute loss ratio
            seg_agg = (
                seg_df.groupby(segment)[["TotalClaims", "TotalPremium"]]
                .sum()
                .assign(
                    LossRatio=lambda x: (x["TotalClaims"] / x["TotalPremium"]).replace(
                        [float("inf"), -float("inf")], float("nan")
                    )
                )
                .reset_index()
                .sort_values("LossRatio", ascending=False)
            )

            # 🎨 Plot loss ratio by segment value
            plt.figure(figsize=(10, 5))
            sns.barplot(data=seg_agg, x="LossRatio", y=segment, palette="coolwarm")
            plt.title(f"📊 Loss Ratio by {segment}", fontsize=14)
            plt.xlabel("Loss Ratio (Claims ÷ Premium)", fontsize=11)
            plt.ylabel(segment, fontsize=11)
            plt.tight_layout()
            plt.show()
