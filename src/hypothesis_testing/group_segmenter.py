"""
group_segmenter.py – Task 3 A/B Group Creator (B5W3)
------------------------------------------------------------------------------
Segments the enriched AlphaCare insurance dataset into testable A/B groups
for statistical hypothesis testing. Each method isolates a binary comparison
(e.g., Male vs Female, Gauteng vs WC) with covariate filtering if needed.

Core responsibilities:
  • Slice groups based on categorical fields (e.g., Gender, Province, ZipCode)
  • Decode human-readable labels into encoded numeric values (if decode=True)
  • Optionally validate covariate balance (e.g., same PlanType across groups)
  • Return DataFrame filtered for test and control groups

Used in Task 3 hypothesis tests (Claim Frequency, Severity, Margin analysis)

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import warnings  # For optional silent warnings
import os  # To validate mapping file path

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # Data manipulation


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: GroupSegmenter
# ───────────────────────────────────────────────────────────────────────────────
class GroupSegmenter:
    """
    OOP utility for segmenting the enriched dataset into two statistically
    comparable groups for hypothesis testing (A/B or Control/Test).
    Supports decoding label-encoded columns using a saved mapping file.
    """

    def __init__(self, label_mapping_path: str = "data/mappings/label_encodings.csv"):
        """
        Initializes the segmenter and loads optional label decoder map.

        Args:
            label_mapping_path (str): Path to saved label encoding map (CSV).
        """
        self.label_mapping_path = label_mapping_path  # Save file path
        self.decoder = self._load_decoder_map()  # Load decoding DataFrame

    def _load_decoder_map(self) -> pd.DataFrame:
        """
        Load the label encoding map to reverse numeric codes to original labels.

        Returns:
            pd.DataFrame: Decoder map with ['column', 'original_value', 'encoded_value'].
        """
        # Check if mapping file exists
        if not os.path.exists(self.label_mapping_path):
            warnings.warn(f"⚠️ Label mapping file not found: {self.label_mapping_path}")
            return pd.DataFrame()  # Return empty if missing

        try:
            # Load mapping file
            decoder = pd.read_csv(self.label_mapping_path)

            # Validate required columns exist
            required_cols = {"column", "original_value", "encoded_value"}
            if not required_cols.issubset(set(decoder.columns)):
                raise ValueError("Missing required columns in label mapping file.")

            return decoder  # Return loaded decoder map

        except Exception as e:
            warnings.warn(f"⚠️ Failed to load label mappings: {e}")
            return pd.DataFrame()  # Return empty decoder

    def _decode_column_value(self, col: str, original_value: str) -> int:
        """
        Given a column name and original value (e.g., 'Western Cape'), return
        the encoded numeric version based on the mapping.

        Args:
            col (str): Column name (e.g., 'Province')
            original_value (str): Original string label to decode

        Returns:
            int: Encoded integer corresponding to the original value.

        Raises:
            ValueError: If mapping is not found.
        """
        if self.decoder.empty:
            raise ValueError("Decoder map not loaded. Cannot decode original labels.")

        match = self.decoder[
            (self.decoder["column"] == col)
            & (self.decoder["original_value"] == original_value)
        ]

        if match.empty:
            raise ValueError(
                f"Could not decode value '{original_value}' in column '{col}'."
            )

        return match["encoded_value"].iloc[0]

    def segment_by_column(
        self,
        df: pd.DataFrame,
        column: str,
        group_a_value,
        group_b_value,
        covariate_filters: dict = None,
        decode: bool = True,
    ) -> pd.DataFrame:
        """
        Filter the dataset into A/B groups based on a specified column.
        Optionally decodes group values into encoded form before filtering.

        Args:
            df (pd.DataFrame): Enriched dataset with derived KPIs.
            column (str): Column to segment by (e.g., 'Gender', 'Province').
            group_a_value (str|int): Value to assign to Group A (control).
            group_b_value (str|int): Value to assign to Group B (test).
            covariate_filters (dict, optional): Filters to control for confounders.
            decode (bool): Whether to decode labels to encoded form before filtering.

        Returns:
            pd.DataFrame: Filtered DataFrame with 'ABGroup' column added.

        Raises:
            KeyError: If segmentation column is missing.
            ValueError: If decoding fails or values not found.
        """
        # 🛡️ Ensure segmentation column exists
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in dataset.")

        # 🔁 Decode string inputs if decode=True
        encoded_a = (
            self._decode_column_value(column, group_a_value)
            if decode
            else group_a_value
        )
        encoded_b = (
            self._decode_column_value(column, group_b_value)
            if decode
            else group_b_value
        )

        # 🧹 Apply covariate filters if provided
        if covariate_filters:
            for cov_col, cov_val in covariate_filters.items():
                if cov_col not in df.columns:
                    warnings.warn(
                        f"Covariate column '{cov_col}' not found; skipping filter."
                    )
                    continue
                df = df[df[cov_col] == cov_val]

        # 🪓 Filter to just the two selected groups
        filtered_df = df[df[column].isin([encoded_a, encoded_b])].copy()

        # 🏷️ Label A/B group
        filtered_df["ABGroup"] = filtered_df[column].apply(
            lambda x: "A" if x == encoded_a else "B"
        )

        return filtered_df

    def summarize_group_counts(self, df: pd.DataFrame) -> None:
        """
        Print the group counts for A/B segments in a labeled DataFrame.

        Args:
            df (pd.DataFrame): Must include 'ABGroup' column.
        """
        if "ABGroup" not in df.columns:
            raise KeyError("No 'ABGroup' column found. Run segmentation first.")

        counts = df["ABGroup"].value_counts()
        print("🧮 Group Counts:")
        print(counts.to_string())
