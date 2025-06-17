"""
defensive_schema_auditor.py – Defensive Schema & Risk Diagnostic Auditor (B5W3)
------------------------------------------------------------------------------
Performs structural diagnostics on the insurance dataset, including:
  • Constant-value column detection
  • High-cardinality categorical column flags
  • Uniqueness checks for ID columns
  • Null audit for business-critical fields
  • Schema conformance (type mismatch detection)

Used in Task 1 of the B5W3 challenge to ensure structural integrity
before modeling or inference.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import List, Dict, Any  # Type annotations

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # Data manipulation
import numpy as np  # Numerical analysis


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: DefensiveSchemaAuditor
# ───────────────────────────────────────────────────────────────────────────────
class DefensiveSchemaAuditor:
    """
    Provides defensive schema validation and diagnostic summaries for insurance risk datasets.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the auditor with the provided DataFrame.

        Args:
            df (pd.DataFrame): Cleaned input DataFrame.

        Raises:
            TypeError: If input is not a valid DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("❌ Input must be a pandas DataFrame.")

        self.df = df.copy()  # Make a copy to avoid modifying original
        self.audit_report: Dict[str, Any] = {}  # Dictionary to collect diagnostics

        # Expected schema types by key fields
        self.expected_types = {
            "PolicyID": object,
            "UnderwrittenCoverID": object,
            "TotalPremium": "number",
            "TotalClaims": "number",
            "TransactionMonth": "datetime64[ns]",
            "PostalCode": "string",
        }

        # Key columns to check for nulls
        self.critical_cols = [
            "TotalClaims",
            "TotalPremium",
            "Gender",
            "TransactionMonth",
        ]

    def run_audit(self) -> None:
        """
        Executes all schema diagnostics and prints results.
        """
        print("🔍 Running full schema audit...\n")
        self._constant_column_check()
        self._cardinality_check()
        self._id_uniqueness_check()
        self._null_check()
        self._type_mismatch_check()

    def _constant_column_check(self):
        """
        Identifies columns with only one unique value (uninformative).
        """
        constant_cols = [
            col for col in self.df.columns if self.df[col].nunique(dropna=False) == 1
        ]
        if constant_cols:
            print(f"🧼 Constant-value columns (uninformative): {constant_cols}")
        self.audit_report["constant_columns"] = constant_cols

    def _cardinality_check(self):
        """
        Flags object/category columns with too many unique values (over 50).
        """
        high_card_cols = [
            col
            for col in self.df.select_dtypes(include=["object", "category"])
            if self.df[col].nunique() > 50
        ]
        if high_card_cols:
            print(f"🕳️ High-cardinality columns (likely unscalable): {high_card_cols}")
        self.audit_report["high_cardinality"] = high_card_cols

    def _id_uniqueness_check(self):
        """
        Checks uniqueness of identifier fields.
        """
        id_checks = {}
        for id_col in ["PolicyID", "UnderwrittenCoverID"]:
            if id_col in self.df.columns:
                unique_vals = self.df[id_col].nunique()
                ratio = unique_vals / len(self.df)
                if ratio < 1.0:
                    print(
                        f"⚠️ {id_col} is not unique: {unique_vals:,} unique values in {len(self.df):,} rows"
                    )
                id_checks[id_col] = {"unique_count": unique_vals, "ratio": ratio}
        self.audit_report["id_uniqueness"] = id_checks

    def _null_check(self):
        """
        Audits nulls for critical business columns.
        """
        null_summary = {}
        print("\n💀 Null Audit for Critical Columns:")
        for col in self.critical_cols:
            if col in self.df.columns:
                nulls = self.df[col].isna().sum()
                pct = (nulls / len(self.df)) * 100
                print(f"   • {col}: {nulls:,} nulls ({pct:.2f}%)")
                null_summary[col] = {"nulls": nulls, "pct": round(pct, 2)}
        self.audit_report["null_summary"] = null_summary

    def _type_mismatch_check(self):
        """
        Checks whether actual column types match expected schema.
        """
        mismatches = {}
        print("\n🚨 Type Mismatch Warnings:")
        for col, expected in self.expected_types.items():
            if col not in self.df.columns:
                continue  # Skip missing columns

            actual = self.df[col].dtype

            # Handle each expected type category
            if expected == "number" and not pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"   • {col}: ❌ Not numeric (actual type: {actual})")
                mismatches[col] = f"Expected numeric, got {actual}"

            elif (
                expected == "datetime64[ns]"
                and not pd.api.types.is_datetime64_any_dtype(self.df[col])
            ):
                print(f"   • {col}: ❌ Not datetime (actual type: {actual})")
                mismatches[col] = f"Expected datetime, got {actual}"

            elif expected == "string" and not pd.api.types.is_string_dtype(
                self.df[col]
            ):
                print(f"   • {col}: ❌ Not string (actual type: {actual})")
                mismatches[col] = f"Expected string, got {actual}"

            elif (
                expected not in ["number", "datetime64[ns]", "string"]
                and actual != expected
            ):
                print(
                    f"   • {col}: ❌ Type mismatch (expected: {expected}, got: {actual})"
                )
                mismatches[col] = f"Expected {expected}, got {actual}"

        self.audit_report["type_mismatches"] = mismatches

    def get_report(self) -> Dict[str, Any]:
        """
        Returns the audit report dictionary for programmatic inspection.

        Returns:
            Dict[str, Any]: Audit results from all checks.
        """
        return self.audit_report
