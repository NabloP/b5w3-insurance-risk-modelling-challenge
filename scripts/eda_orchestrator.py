"""
🎯 Task 1 Orchestrator – Full EDA Pipeline for Insurance Risk Analytics
Author: Nabil Mohamed | AlphaCare | June 2025
Challenge: B5W3 – Insurance Risk Analytics & Predictive Modeling

This script auto-runs all 10 layers of EDA:
- Schema audits, stats, plots, risk profiling, schema fixes
- Modular, reproducible, with inline output only
"""

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Environment & Path Setup – Run from Project Root
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# Move up if executed from notebooks/
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
    print("📂 Changed directory to project root")

# Add root to sys.path for src/ imports
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    print("🔧 Added project root to sys.path")

# ─────────────────────────────────────────────────────────────────────────────
# 📥 Load Raw Data
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from src.data_loader import InsuranceDataLoader

data_path = "data/raw/MachineLearningRating_v3.txt"
print(f"\n📥 Loading raw dataset from: {data_path}")
loader = InsuranceDataLoader(filepath=data_path)

try:
    df = loader.load()
    print(f"✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load data: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 🧾 Layer 1 – Schema Audit
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.schema_auditor import SchemaAuditor

print("🔎 Running Layer 1: Schema Audit...")
auditor = SchemaAuditor(df)
auditor.check_duplicate_ids(["PolicyID", "UnderwrittenCoverID"])
auditor.summarize_schema()
auditor.print_diagnostics()

# ─────────────────────────────────────────────────────────────────────────────
# 📊 Layer 2A – Descriptive Stats
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.distribution_analyzer import DistributionAnalyzer

print("\n📊 Running Layer 2A: Distribution Descriptives...")
dist_analyzer = DistributionAnalyzer(df)
_ = dist_analyzer.describe_numerics()
_ = dist_analyzer.styled_skew_kurt()
dist_analyzer.print_distribution_warnings()

# ─────────────────────────────────────────────────────────────────────────────
# 📉 Layer 2B – Visual Distributions
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.numeric_plotter import NumericPlotter

print("\n📈 Running Layer 2B: Histogram and Boxplot Visuals...")
plot_cols = ["TotalPremium", "TotalClaims", "CustomValueEstimate"]
plotter = NumericPlotter(df)
plotter.plot_all(plot_cols)

# ─────────────────────────────────────────────────────────────────────────────
# 📅 Layer 3 – Temporal Trends
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.temporal_analyzer import TemporalClaimsAnalyzer

print("\n📅 Running Layer 3: Monthly Temporal Trends...")
temporal = TemporalClaimsAnalyzer(df)
temporal.prepare_temporal_data()
temporal.aggregate_monthly_metrics()
temporal.plot_trend_lines()
temporal.report_missing_months()

# ─────────────────────────────────────────────────────────────────────────────
# 🌍 Layer 4 – Geographic Risk Map
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.geo_risk_visualizer import GeoRiskVisualizer

print("\n🌍 Running Layer 4: Geographic Loss Ratio...")
geo_path = "data/raw/opendb-2025-06-17.csv"

try:
    geo_viz = GeoRiskVisualizer(df, coord_path=geo_path)
    geo_viz.compute_loss_ratios()
    geo_viz.merge_coordinates()
    geo_viz.plot_province_bar()
    geo_viz.plot_top_postal_codes()
    geo_viz.plot_loss_ratio_map()
except Exception as e:
    print(f"⚠️ Skipping Layer 4 due to error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 🚘 Layer 5 – Vehicle Make & Model Risk
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.vehicle_risk_profiler import VehicleRiskProfiler

print("\n🚗 Running Layer 5: Vehicle Risk Profiling...")
try:
    profiler = VehicleRiskProfiler(df)
    profiler.plot_vehicle_risk()
except Exception as e:
    print(f"⚠️ Skipping vehicle risk plot: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ⚖️ Layer 6 – Gender Risk Segmentation
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.gender_risk_profiler import GenderRiskProfiler

print("\n⚖️ Running Layer 6: Gender-Based Risk...")
gender_profiler = GenderRiskProfiler(df)
agg = gender_profiler.compute_loss_ratios()
gender_profiler.plot_loss_ratio_bar(agg)
gender_profiler.plot_distribution_boxplots()

# ─────────────────────────────────────────────────────────────────────────────
# 🧩 Layer 7 – Policy Feature Segments
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.plan_feature_risk_profiler import PlanFeatureRiskProfiler

print("\n🧩 Running Layer 7: Policy Segment Profiling...")
segment_profiler = PlanFeatureRiskProfiler(df)
segment_profiler.analyze()

# ─────────────────────────────────────────────────────────────────────────────
# 🧼 Layer 8 – Outlier Detection (IQR)
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.iqr_outlier_detector import IQRBasedOutlierDetector

print("\n🧼 Running Layer 8: Outlier Detection...")
iqr_detector = IQRBasedOutlierDetector(df)
iqr_detector.detect_outliers()
iqr_outlier_flags = iqr_detector.get_outlier_flags()

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Layer 9 – Schema Defensive Audit
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.defensive_schema_auditor import DefensiveSchemaAuditor

print("\n🛡️ Running Layer 9: Defensive Schema Audit...")
schema_auditor = DefensiveSchemaAuditor(df)
schema_auditor.run_audit()
schema_report = schema_auditor.get_report()

constant_cols = schema_report.get("constant_cols", [])
high_cardinality = schema_report.get("high_cardinality_cols", [])

# ─────────────────────────────────────────────────────────────────────────────
# 🧰 Layer 10 – Guardrails & Fixes
# ─────────────────────────────────────────────────────────────────────────────

from src.eda.schema_guardrails import SchemaGuardrails

print("\n🧰 Running Layer 10: Schema Fixes and Exclusion Rules...")
guardrails = SchemaGuardrails(df, constant_cols, high_cardinality)
df = guardrails.apply_guardrails()
excluded_cols = guardrails.get_excluded_columns()

# ─────────────────────────────────────────────────────────────────────────────
# ✅ Completion
# ─────────────────────────────────────────────────────────────────────────────

print("\n✅ EDA pipeline completed successfully.")
print(f"🧺 Excluded Columns: {excluded_cols if excluded_cols else 'None'}")
