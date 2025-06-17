"""
schema_auditor.py – DataFrame Schema Diagnostic Tool (B5W3)
------------------------------------------------------------------------------
Provides detailed structural diagnostics on insurance-related DataFrames.
Summarizes missingness, uniqueness, dtype consistency, and schema stability.

Core responsibilities:
  • Computes per-column stats: dtype, % missing, constant-value flags
  • Flags high-null fields with criticality bands
  • Checks duplicate identifier values
  • Supports styled summaries for visual EDA
  • Raises detailed exceptions if used incorrectly

Used in Task 1 EDA and validation of any intermediate or enriched outputs.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import List  # For method annotations

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame manipulation


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: SchemaAuditor
# ───────────────────────────────────────────────────────────────────────────────
class SchemaAuditor:
    """
    Class for auditing a DataFrame’s structural integrity, null distribution,
    uniqueness, and column-level stability.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the schema auditor with a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to inspect.

        Raises:
            TypeError: If df is not a pandas DataFrame.
            ValueError: If df is empty or lacks columns.
        """
        # 🛡️ Type check
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"`df` must be a pandas DataFrame, got {type(df)}")

        # 🛡️ Basic structural check
        if df.empty or df.shape[1] == 0:
            raise ValueError("Input DataFrame is empty or has no columns.")

        self.df = df
        self.schema_df = None  # Will hold the computed schema summary

    def summarize_schema(self) -> pd.DataFrame:
        """
        Computes column-wise stats: dtype, n_unique, % missing, constant-value flags.

        Returns:
            pd.DataFrame: Schema summary with diagnostics.
        """
        try:
            # 📊 Build core schema metrics
            schema = pd.DataFrame(
                {
                    "dtype": self.df.dtypes,
                    "n_unique": self.df.nunique(),
                    "n_missing": self.df.isna().sum(),
                }
            )

            # ➕ Compute missing % per column
            schema["%_missing"] = (schema["n_missing"] / len(self.df) * 100).round(2)

            # 🧱 Flag constants and null severity
            schema["is_constant"] = schema["n_unique"] <= 1
            schema["high_null_flag"] = pd.cut(
                schema["%_missing"],
                bins=[-1, 0, 20, 50, 100],
                labels=["✅ OK", "🟡 Moderate", "🟠 High", "🔴 Critical"],
            )

            # 🔃 Store sorted version
            self.schema_df = schema.sort_values("%_missing", ascending=False)
            return self.schema_df

        except Exception as e:
            raise RuntimeError(f"Failed to generate schema summary: {e}")

    def styled_summary(self):
        """
        Returns a styled version of the schema for notebook display.

        Returns:
            pd.io.formats.style.Styler: Styled DataFrame with color-coded warnings.
        """
        try:
            if self.schema_df is None:
                self.summarize_schema()

            return (
                self.schema_df.style.background_gradient(
                    subset="%_missing", cmap="OrRd"
                )
                .applymap(
                    lambda val: (
                        "background-color: gold; font-weight: bold;" if val else ""
                    ),
                    subset=["is_constant"],
                )
                .format({"%_missing": "{:.2f}"})
            )

        except Exception as e:
            raise RuntimeError(f"Failed to style schema output: {e}")

    def print_diagnostics(self) -> None:
        """
        Prints summary diagnostics of column stability and missingness.

        Raises:
            RuntimeError: If schema summary is missing or malformed.
        """
        if self.schema_df is None:
            self.summarize_schema()

        try:
            n_const = self.schema_df[
                "is_constant"
            ].sum()  # Count constant-value columns
            n_null_20 = (
                self.schema_df["%_missing"] > 20
            ).sum()  # Count columns with >20% missing
            n_null_50 = (
                self.schema_df["%_missing"] > 50
            ).sum()  # Count columns with >50% missing

            print("\n🧾 Summary Diagnostics:")
            print(f"→ Constant-value columns:  {n_const}")
            print(f"→ Columns >20% missing:    {n_null_20}")
            print(f"→ Columns >50% missing:    {n_null_50}")

        except Exception as e:
            raise RuntimeError(f"Error during diagnostics printing: {e}")

    def check_duplicate_ids(self, id_columns: List[str]) -> None:
        """
        Checks for duplicate values in key identifier columns.

        Args:
            id_columns (List[str]): List of identifier column names to check.

        Raises:
            ValueError: If any column in id_columns does not exist.
        """
        for col in id_columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

            duplicates = self.df[col].duplicated().sum()
            print(
                f"→ {col}: {duplicates:,} duplicates" + (" ⚠️" if duplicates > 0 else "")
            )
