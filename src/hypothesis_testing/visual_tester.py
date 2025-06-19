"""
visual_tester.py – Task 3 Hypothesis Visual Diagnostics (B5W3)
-------------------------------------------------------------------------------
Generates violin plots, boxplots, and histogram overlays to compare KPI metric
distributions (e.g., ClaimFrequency, ClaimSeverity, Margin) across A/B groups.

Use case:
  • Validate differences in distribution shape before/after statistical testing
  • Visualize segmentation impact for stakeholders and business analysts

Author: Nabil Mohamed
"""

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ─────────────────────────────────────────────────────────────────────────────
import os  # For optional path manipulation

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd  # Data handling
import matplotlib.pyplot as plt  # Plotting engine
import seaborn as sns  # High-level statistical plots


# ─────────────────────────────────────────────────────────────────────────────
# 🧪 VisualTester Class – A/B Plotting Engine
# ─────────────────────────────────────────────────────────────────────────────
class VisualTester:
    """
    Visual diagnostic engine to plot KPI metrics by A/B test groups.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the VisualTester with an enriched DataFrame.

        Args:
            df (pd.DataFrame): Cleaned and segmented insurance dataset with
            KPI metrics and an 'ABGroup' column.
        """
        self.df = df  # Store dataframe as a class attribute

        # Define expected KPI columns to visualize
        self.kpi_columns = ["ClaimFrequency", "ClaimSeverity", "Margin"]

        # Validate that required column exists before proceeding
        self._validate_abgroup_presence()

    def _validate_abgroup_presence(self):
        """
        Internal method to ensure 'ABGroup' exists in the DataFrame.
        """
        if "ABGroup" not in self.df.columns:
            raise KeyError("Missing 'ABGroup' column. Please segment data first.")

    def _validate_kpi_column(self, column: str):
        """
        Check if the selected column is available in the DataFrame.

        Args:
            column (str): The KPI metric to validate

        Raises:
            KeyError: If column is missing from the DataFrame
        """
        if column not in self.df.columns:
            raise KeyError(f"Missing KPI column: '{column}'")

    def plot_violin(self, column: str):
        """
        Generate a violin plot comparing the KPI metric across A/B groups.

        Args:
            column (str): Column to visualize (e.g., 'Margin')
        """
        self._validate_kpi_column(column)  # Ensure column exists

        # Setup plot layout
        plt.figure(figsize=(8, 5))
        sns.violinplot(
            data=self.df, x="ABGroup", y=column, inner="quartile", palette="Set2"
        )
        plt.title(f"Violin Plot – {column} by A/B Group")
        plt.xlabel("A/B Group")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()  # Render inline

    def plot_boxplot(self, column: str):
        """
        Generate a boxplot for the given KPI metric across A/B groups.

        Args:
            column (str): Column to visualize (e.g., 'ClaimSeverity')
        """
        self._validate_kpi_column(column)  # Ensure column exists

        # Setup plot
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x="ABGroup", y=column, palette="pastel")
        plt.title(f"Boxplot – {column} by A/B Group")
        plt.xlabel("A/B Group")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()

    def plot_histogram_overlay(self, column: str, bins: int = 30):
        """
        Overlay histograms for each A/B group on the same axis for comparison.

        Args:
            column (str): KPI metric to visualize
            bins (int): Number of histogram bins
        """
        self._validate_kpi_column(column)  # Ensure column exists

        # Setup figure
        plt.figure(figsize=(8, 5))

        # Iterate through A and B groups and plot histograms
        for group in ["A", "B"]:
            subset = self.df[self.df["ABGroup"] == group]  # Filter by group
            plt.hist(subset[column], bins=bins, alpha=0.5, label=f"Group {group}")

        # Finalize plot
        plt.title(f"Histogram Overlay – {column} by A/B Group")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()
