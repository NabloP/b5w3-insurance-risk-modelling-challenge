"""
hypothesis_tester.py – Task 3 KPI Hypothesis Testing Module (B5W3)
------------------------------------------------------------------------------
Runs statistical tests on pre-segmented A/B insurance groups to evaluate
differences in KPIs like ClaimFrequency, Severity, and Margin.

Core responsibilities:
  • Run t-test or Mann–Whitney U depending on normality assumptions
  • Report p-values, test used, and effect size
  • Modular for use in notebooks or orchestrator pipelines

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import warnings  # For assumption diagnostics

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For dataframe operations
import numpy as np  # For numerical calculations
from scipy.stats import ttest_ind, mannwhitneyu, shapiro  # Statistical tests


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: HypothesisTester
# ───────────────────────────────────────────────────────────────────────────────
class HypothesisTester:
    """
    Modular tester for KPI differences between A/B insurance segments.
    """

    def __init__(self):
        """Initialize the tester with no parameters."""
        pass

    def check_normality(self, data: pd.Series, alpha: float = 0.05) -> bool:
        """
        Runs Shapiro–Wilk test to check if data is normally distributed.

        Args:
            data (pd.Series): Numeric data for a group.
            alpha (float): Significance threshold (default: 0.05).

        Returns:
            bool: True if data is normal, False otherwise.
        """
        # Shapiro–Wilk returns (statistic, p-value)
        stat, p = shapiro(data)
        return p > alpha  # If p > alpha, data is normal

    def compute_effect_size(self, group_a: pd.Series, group_b: pd.Series) -> float:
        """
        Computes Cohen's d effect size between two numeric groups.

        Args:
            group_a (pd.Series): Control group.
            group_b (pd.Series): Test group.

        Returns:
            float: Cohen's d value.
        """
        # Difference in means divided by pooled standard deviation
        mean_diff = group_a.mean() - group_b.mean()
        pooled_std = np.sqrt((group_a.std() ** 2 + group_b.std() ** 2) / 2)
        return round(mean_diff / pooled_std, 3)

    def run_test(self, df: pd.DataFrame, metric: str) -> dict:
        """
        Executes the correct test between Group A and Group B for a given metric.

        Args:
            df (pd.DataFrame): Must include 'ABGroup' and the selected metric.
            metric (str): KPI to compare (e.g., 'ClaimFrequency').

        Returns:
            dict: Results including test type, p-value, effect size, and normality flags.
        """
        # 🛡️ Check prerequisites
        if "ABGroup" not in df.columns:
            raise KeyError("No 'ABGroup' column found. Segment data first.")
        if metric not in df.columns:
            raise KeyError(f"Metric '{metric}' not found in DataFrame.")

        # 🔍 Split groups
        group_a = df[df["ABGroup"] == "A"][metric].dropna()
        group_b = df[df["ABGroup"] == "B"][metric].dropna()

        # 🔎 Check normality for both groups
        normal_a = self.check_normality(group_a)
        normal_b = self.check_normality(group_b)

        # 🔬 Run appropriate test
        if normal_a and normal_b:
            # Use independent t-test
            test_name = "t-test"
            stat, p_value = ttest_ind(group_a, group_b, equal_var=False)
        else:
            # Use Mann–Whitney U for non-normal distributions
            test_name = "Mann–Whitney U"
            stat, p_value = mannwhitneyu(group_a, group_b, alternative="two-sided")

        # 📐 Compute effect size
        effect_size = self.compute_effect_size(group_a, group_b)

        # 📦 Return test summary
        return {
            "metric": metric,
            "test_used": test_name,
            "p_value": round(p_value, 5),
            "effect_size": effect_size,
            "group_a_normal": normal_a,
            "group_b_normal": normal_b,
        }
