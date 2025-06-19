"""
data_cleaner.py – Task 3–4 Data Cleaning Pipeline (B5W3)
------------------------------------------------------------------------------
Cleans and prepares the AlphaCare insurance dataset for hypothesis testing
and predictive modeling. Designed based on EDA insights from Task 1.

Core responsibilities:
  • Drop constant and high-null columns (>60%)
  • Impute moderate-missing numeric features and label missing categoricals as 'Missing'
  • Encode categorical variables via Label Encoding (with mapping saved to CSV)
  • Remove duplicate rows (exact duplicates only)
  • Optional outlier clipping (excludes zeros and uses IQR logic)

Output is a fully numeric, model-ready DataFrame for Tasks 3 and 4.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os  # For saving files
import warnings  # To suppress non-critical runtime warnings

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # Data manipulation
import numpy as np  # Numerical computations
from sklearn.preprocessing import LabelEncoder  # For categorical encoding


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: DataCleaner
# ───────────────────────────────────────────────────────────────────────────────
class DataCleaner:
    """
    Modular cleaning pipeline for AlphaCare insurance dataset. Applies
    defensive preprocessing logic for Task 3 testing and Task 4 modeling.
    """

    def __init__(self):
        # Dictionary to store encoders for all label-encoded columns
        self.label_encoders = {}

    def drop_high_null_and_constant_columns(
        self, df: pd.DataFrame, null_thresh=0.6
    ) -> pd.DataFrame:
        """
        Removes columns with only one unique value and those with >60% nulls.

        Args:
            df (pd.DataFrame): Raw input DataFrame.
            null_thresh (float): Maximum acceptable null ratio before dropping.

        Returns:
            pd.DataFrame: Cleaned DataFrame with redundant columns removed.
        """
        df = df.copy()  # Avoid modifying original DataFrame

        # Identify and drop constant columns
        constant_cols = [
            col for col in df.columns if df[col].nunique(dropna=False) <= 1
        ]
        df.drop(columns=constant_cols, inplace=True)

        # Identify and drop columns with too many nulls
        high_null_cols = [
            col for col in df.columns if df[col].isnull().mean() > null_thresh
        ]
        df.drop(columns=high_null_cols, inplace=True)

        return df  # Return reduced dataset

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values in numeric columns with median and
        categoricals with the string 'Missing'.

        Args:
            df (pd.DataFrame): Input DataFrame with nulls.

        Returns:
            pd.DataFrame: Imputed DataFrame.
        """
        df = df.copy()

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())  # Impute numerics
                else:
                    df[col] = df[col].fillna("Missing")  # Impute categoricals

        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label-encodes all object/categorical columns and saves mapping to disk.

        Args:
            df (pd.DataFrame): DataFrame with string labels.

        Returns:
            pd.DataFrame: DataFrame with encoded categories.
        """
        df = df.copy()
        mapping_records = []

        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

            for original_value, encoded_value in zip(
                le.classes_, le.transform(le.classes_)
            ):
                mapping_records.append(
                    {
                        "column": col,
                        "original_value": original_value,
                        "encoded_value": encoded_value,
                    }
                )

        # Save all mappings to CSV
        mapping_df = pd.DataFrame(mapping_records)
        os.makedirs("data/mappings", exist_ok=True)
        mapping_df.to_csv("data/mappings/label_encodings.csv", index=False)

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes exact row-level duplicates.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Deduplicated DataFrame.
        """
        return df.drop_duplicates()

    def clip_outliers(self, df: pd.DataFrame, numeric_cols=None) -> pd.DataFrame:
        """
        Clips outliers using IQR, excluding zero values from the IQR estimation.
        Flags if more than 10% of values are clipped per column.

        Args:
            df (pd.DataFrame): DataFrame with numerics.
            numeric_cols (list): Columns to clip. Defaults to ["TotalClaims", "TotalPremium", "CustomValueEstimate"].

        Returns:
            pd.DataFrame: Outlier-trimmed DataFrame.
        """
        df = df.copy()

        # Default set of numeric fields to trim if none provided
        if numeric_cols is None:
            numeric_cols = ["TotalClaims", "TotalPremium", "CustomValueEstimate"]

        for col in numeric_cols:
            if col not in df.columns:
                print(f"⚠️ Column '{col}' not found. Skipping...")
                continue

            # Exclude zero values for IQR estimation
            non_zero_vals = df[col][df[col] > 0]
            if non_zero_vals.empty or non_zero_vals.count() < 10:
                print(f"⚠️ Too few non-zero entries in '{col}' to trim reliably.")
                continue

            Q1 = non_zero_vals.quantile(0.25)
            Q3 = non_zero_vals.quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # Store original value count
            original_count = df[col].count()

            # Apply clipping
            df[col] = np.where(df[col] < lower, lower, df[col])
            df[col] = np.where(df[col] > upper, upper, df[col])

            # Calculate number of values clipped
            clipped_count = ((df[col] == lower) | (df[col] == upper)).sum()
            clip_ratio = clipped_count / original_count

            if clip_ratio > 0.10:
                print(
                    f"🔎 Warning: {clip_ratio:.1%} of '{col}' values clipped (>{clipped_count} rows)."
                )

        return df

    def run_full_cleaning_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full cleaning pipeline: drop low-signal columns, dedupe, impute, encode, clip.

        Args:
            df (pd.DataFrame): Raw input.

        Returns:
            pd.DataFrame: Cleaned, numeric DataFrame ready for modeling.
        """
        df = self.drop_high_null_and_constant_columns(df)  # Step 1: Drop junk cols
        df = self.remove_duplicates(df)  # Step 2: Remove exact dupes
        df = self.impute_missing_values(df)  # Step 3: Impute missing
        df = self.encode_categoricals(df)  # Step 4: Encode labels
        df = self.clip_outliers(df)  # Step 5: Trim outliers (IQR, zero-aware)
        return df
