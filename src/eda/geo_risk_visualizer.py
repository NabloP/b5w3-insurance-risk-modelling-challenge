"""
geo_risk_visualizer.py – Layer 4 Geographic Loss Ratio Plotting (B5W3)
------------------------------------------------------------------------------
Computes and visualizes insurance loss ratios at the postal and province level.
Supports bar charts and a bubble map overlaid on South African geography.

Core responsibilities:
  • Compute and sort loss ratios by Province and PostalCode
  • Join with latitude/longitude using cleaned postal dataset
  • Plot province-level and postal-level bar charts
  • Plot bubble map where color and size reflect LossRatio

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import Point


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: GeoRiskVisualizer
# ───────────────────────────────────────────────────────────────────────────────
class GeoRiskVisualizer:
    """
    Computes and visualizes geographic insurance loss ratios by province/postal code.
    """

    def __init__(self, df: pd.DataFrame, coord_path: str):
        """
        Initialize with raw DataFrame and postal geolocation file path.

        Args:
            df (pd.DataFrame): Insurance dataset with Province, PostalCode, Claims, Premiums.
            coord_path (str): Path to CSV containing PostalCode, latitude, longitude.

        Raises:
            ValueError: If required columns are missing.
        """
        required = {"Province", "PostalCode", "TotalClaims", "TotalPremium"}
        if not required.issubset(df.columns):
            missing = required.difference(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df.dropna(subset=["Province", "PostalCode"]).copy()
        self.df["PostalCode"] = self.df["PostalCode"].astype(str).str.zfill(4)
        self.coord_path = coord_path

        self.province_agg = None
        self.gdf = None  # GeoDataFrame for postal-level mapping

    def compute_loss_ratios(self):
        """
        Aggregates by province and postal code, computes loss ratios.
        """
        # Province-level
        self.province_agg = (
            self.df.groupby("Province")[["TotalClaims", "TotalPremium"]]
            .sum()
            .assign(
                LossRatio=lambda x: (x["TotalClaims"] / x["TotalPremium"]).replace(
                    [np.inf, -np.inf], np.nan
                )
            )
            .reset_index()
            .sort_values("LossRatio", ascending=False)
        )

        # Postal-level
        postal_agg = (
            self.df.groupby("PostalCode")[["TotalClaims", "TotalPremium"]]
            .sum()
            .assign(
                LossRatio=lambda x: (x["TotalClaims"] / x["TotalPremium"]).replace(
                    [np.inf, -np.inf], np.nan
                )
            )
            .reset_index()
        )

        self.postal_agg = postal_agg

    def merge_coordinates(self):
        """
        Merges postal-level loss data with latitude/longitude.
        Filters to valid geographic bounds for South Africa.
        """
        location_df = pd.read_csv(self.coord_path, dtype=str)
        location_df = location_df.rename(columns={"street_code": "PostalCode"})[
            ["PostalCode", "latitude", "longitude"]
        ].dropna()

        # Convert to float
        location_df["latitude"] = pd.to_numeric(
            location_df["latitude"], errors="coerce"
        )
        location_df["longitude"] = pd.to_numeric(
            location_df["longitude"], errors="coerce"
        )

        # Merge coordinates
        self.postal_agg["PostalCode"] = self.postal_agg["PostalCode"].astype(str)
        merged = self.postal_agg.merge(
            location_df, on="PostalCode", how="left"
        ).dropna()

        # Filter to South African bounding box
        merged = merged[
            merged["latitude"].between(-35, -20) & merged["longitude"].between(15, 35)
        ].copy()

        # Convert to GeoDataFrame
        merged["geometry"] = merged.apply(
            lambda row: Point(row["longitude"], row["latitude"]), axis=1
        )
        self.gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")

    def plot_province_bar(self):
        """
        Bar chart: Loss Ratio by Province.
        """
        if self.province_agg is None:
            self.compute_loss_ratios()

        plt.figure(figsize=(12, 5))
        sns.barplot(
            data=self.province_agg, x="LossRatio", y="Province", palette="coolwarm"
        )
        plt.title("📍 Loss Ratio by Province", fontsize=14)
        plt.xlabel("Loss Ratio (Claims / Premium)", fontsize=11)
        plt.ylabel("Province", fontsize=11)
        plt.tight_layout()
        plt.show()

    def plot_top_postal_codes(self, top_n: int = 20):
        """
        Bar chart: Top N postal codes by Loss Ratio.

        Args:
            top_n (int): Number of postal codes to display.
        """
        if self.gdf is None:
            self.merge_coordinates()

        top_agg = self.gdf.sort_values("LossRatio", ascending=False).head(top_n)
        plt.figure(figsize=(12, 5))
        sns.barplot(data=top_agg, x="LossRatio", y="PostalCode", palette="viridis")
        plt.title(f"📮 Top {top_n} Postal Codes by Loss Ratio", fontsize=14)
        plt.xlabel("Loss Ratio")
        plt.ylabel("Postal Code")
        plt.tight_layout()
        plt.show()

    def plot_loss_ratio_map(self, output_path: str = None):
        """
        Bubble map of Loss Ratio by postal code.

        Args:
            output_path (str): Optional path to save PNG export.
        """
        if self.gdf is None:
            self.merge_coordinates()

        self.gdf["bubble_size"] = self.gdf["LossRatio"] * 400
        self.gdf["bubble_color"] = self.gdf["LossRatio"]

        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            self.gdf.geometry.x,
            self.gdf.geometry.y,
            s=self.gdf["bubble_size"],
            c=self.gdf["bubble_color"],
            cmap="coolwarm",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
        )

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.01)
        cbar.set_label("Loss Ratio (Claims / Premium)", fontsize=10)

        ax.set_title(
            "📮 Geographic Loss Ratio by Postal Code – South Africa", fontsize=16
        )
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_aspect("equal")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
