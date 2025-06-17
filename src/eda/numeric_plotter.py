"""
numeric_plotter.py – Numeric Feature Plotting Utilities (B5W3)
------------------------------------------------------------------------------
Visualizes numeric column distributions using dual-sided diagnostics:
• Left = Histogram + KDE for density estimation
• Right = Boxplot for outlier inspection

Core responsibilities:
  • Filters visualizable numeric columns from df
  • Handles missing or empty columns defensively
  • Renders plots with consistent styling
  • Supports multiple numeric targets in sequence

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: NumericPlotter
# ───────────────────────────────────────────────────────────────────────────────
class NumericPlotter:
    """
    Visualizes distributional diagnostics for numeric variables using
    histogram-KDE + boxplot pairs.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the plotter with a validated DataFrame.

        Args:
            df (pd.DataFrame): The numeric DataFrame to visualize.

        Raises:
            TypeError: If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"`df` must be a pandas DataFrame, got {type(df)}")

        self.df = df.copy()
        self._apply_style()

    def _apply_style(self):
        """
        Sets consistent seaborn/matplotlib styling.
        """
        sns.set(style="whitegrid")
        plt.rcParams["axes.labelsize"] = 10
        plt.rcParams["figure.dpi"] = 120

    def plot_distribution_pair(self, column: str) -> None:
        """
        Generates a side-by-side histogram/KDE and boxplot for a given column.

        Args:
            column (str): Column name to visualize.

        Raises:
            ValueError: If the column does not exist or is not numeric.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' must be numeric to visualize.")

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram + KDE
        sns.histplot(
            self.df[column],
            kde=True,
            ax=axs[0],
            color="cornflowerblue",
            edgecolor="white",
        )
        axs[0].set_title(f"{column} – Histogram w/ KDE", fontsize=11)
        axs[0].set_xlabel(column)
        axs[0].set_ylabel("Frequency")

        # Boxplot
        sns.boxplot(x=self.df[column], ax=axs[1], color="lightcoral")
        axs[1].set_title(f"{column} – Boxplot View", fontsize=11)
        axs[1].set_xlabel(column)

        plt.tight_layout()
        plt.show()

    def plot_all(self, columns: list) -> None:
        """
        Visualizes distribution diagnostics for multiple columns.

        Args:
            columns (list): List of column names to visualize.
        """
        # ✅ Filter only valid numeric columns that exist
        valid_cols = [
            col
            for col in columns
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])
        ]

        if not valid_cols:
            print("⚠️ No target numeric columns found for visualization.")
            return

        for col in valid_cols:
            self.plot_distribution_pair(col)
