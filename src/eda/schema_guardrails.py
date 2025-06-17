"""
schema_guardrails.py – Schema Fixes & Modeling Guardrails (B5W3)
------------------------------------------------------------------------------
Applies schema corrections and modeling exclusion logic, including:
  • Coercing misparsed PostalCode and TransactionMonth columns
  • Dropping constant-value and high-cardinality fields from prior audits
  • Displaying modeling exclusions and final schema dtype summary

Used in Layer 10 of Task 1 (B5W3) for risk modeling hygiene and guardrail setup.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import List
import pandas as pd


# ───────────────────────────────────────────────────────────────────────────────
# 🧰 Class: SchemaGuardrails
# ───────────────────────────────────────────────────────────────────────────────
class SchemaGuardrails:
    """
    Applies schema coercion and modeling guardrails to the insurance dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        constant_cols: List[str] = None,
        high_cardinality: List[str] = None,
    ):
        """
        Initializes the guardrails processor with raw DataFrame and exclusion lists.

        Args:
            df (pd.DataFrame): The working DataFrame to sanitize.
            constant_cols (List[str], optional): Constant-value column names.
            high_cardinality (List[str], optional): High-cardinality column names.

        Raises:
            TypeError: If input DataFrame is not a valid pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("❌ Input must be a valid pandas DataFrame.")

        self.df = df.copy()
        self.constant_cols = constant_cols or []
        self.high_cardinality = high_cardinality or []
        self.excluded_cols: List[str] = []

    def apply_guardrails(self) -> pd.DataFrame:
        """
        Applies type coercions, column exclusions, and prints diagnostics.

        Returns:
            pd.DataFrame: Cleaned DataFrame (with no excluded columns dropped).
        """
        # ✅ Coerce PostalCode to string if needed
        if "PostalCode" in self.df.columns:
            if not pd.api.types.is_string_dtype(self.df["PostalCode"]):
                self.df["PostalCode"] = self.df["PostalCode"].astype(str)
                print("🛠️ Coercion applied: 'PostalCode' converted to string ✅")
            else:
                print("🔍 'PostalCode' already correctly typed as string.")

        # ✅ Coerce TransactionMonth to datetime if needed
        if "TransactionMonth" in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df["TransactionMonth"]):
                self.df["TransactionMonth"] = pd.to_datetime(
                    self.df["TransactionMonth"], errors="coerce"
                )
                print("🛠️ Coercion applied: 'TransactionMonth' parsed as datetime ✅")
            else:
                print("🔍 'TransactionMonth' already correctly typed as datetime.")
        else:
            print(
                "⚠️ Column 'TransactionMonth' not found. No datetime coercion attempted."
            )

        # 🚫 Flag modeling exclusions
        if self.constant_cols:
            self.excluded_cols.extend(self.constant_cols)
        if self.high_cardinality:
            self.excluded_cols.extend(self.high_cardinality)

        # 🧹 Drop duplicates while preserving order
        self.excluded_cols = list(dict.fromkeys(self.excluded_cols))

        # 📣 Print exclusions
        print("\n🚫 Columns recommended for exclusion from modeling:")
        if self.excluded_cols:
            for col in self.excluded_cols:
                print(f"   • {col}")
        else:
            print("   ✅ No columns flagged for exclusion.")

        # 🧾 Print final dtype distribution
        print("\n🧾 Final schema summary by data type:")
        print(self.df.dtypes.value_counts())

        # Return unmodified structure (no columns are dropped)
        return self.df

    def get_excluded_columns(self) -> List[str]:
        """
        Returns the list of excluded columns flagged by guardrails.

        Returns:
            List[str]: List of column names to exclude from modeling.
        """
        return self.excluded_cols
