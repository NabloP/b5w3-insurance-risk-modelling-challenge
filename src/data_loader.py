"""
data_loader.py – Insurance Risk Dataset Loader (B5W3)
------------------------------------------------------------------------------
Safely loads the raw South African car insurance dataset for AlphaCare.
Performs robust file validation, column inspection, and TSV parsing.

Core responsibilities:
  • Validates file path and format
  • Loads tab-separated values from the official data dump
  • Checks for basic structural integrity (non-empty, tabular)
  • Provides helpful diagnostics on shape and column count

Used in Task 1 EDA and all downstream statistical and modeling pipelines.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os  # For file path checks

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For data loading and frame validation


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: InsuranceDataLoader
# ───────────────────────────────────────────────────────────────────────────────
class InsuranceDataLoader:
    """
    OOP wrapper for safely loading the AlphaCare insurance claims dataset.
    Ensures TSV parsing and structural validity before use.
    """

    def __init__(self, filepath: str):
        """
        Initialize the loader with the expected insurance data path.

        Args:
            filepath (str): Full path to the .txt file (tab-separated).

        Raises:
            TypeError: If the filepath is not a string.
            FileNotFoundError: If the file does not exist at the given path.
        """
        # 🛡️ Validate input type
        if not isinstance(filepath, str):
            raise TypeError(f"filepath must be str, got {type(filepath)}")

        # 🛡️ Ensure the file exists at the given path
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Cannot find file at: {filepath}")

        # ✅ Store for loading
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        """
        Loads the TSV insurance dataset and validates its structure.

        Returns:
            pd.DataFrame: Parsed DataFrame with structural validation passed.

        Raises:
            ValueError: If the file is empty or the format is incorrect.
        """
        try:
            # 📥 Load using tab separator
            df = pd.read_csv(self.filepath, sep="|")

            # ❌ Raise error if DataFrame is completely empty
            if df.empty:
                raise ValueError("Loaded DataFrame is empty.")

            # ✅ Success message with basic shape
            print(
                f"✅ Insurance dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns"
            )
            return df

        except pd.errors.ParserError as e:
            # ❌ Raise error if parsing fails
            raise ValueError(f"Could not parse TSV: {e}")

        except Exception as e:
            # ❌ Catch-all for unexpected issues
            raise RuntimeError(f"Unexpected error during data load: {e}")
