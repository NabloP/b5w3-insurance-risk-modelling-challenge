# ---------------------------------------------------------------------------
# 🧪 Task 3 – Hypothesis Testing Orchestrator: Province Risk Comparison
# ---------------------------------------------------------------------------
# 📘 Version: 2025-06-19
# 🔍 Compares Western Cape vs Gauteng on KPIs: Claim Frequency, Severity, Margin
# ---------------------------------------------------------------------------
# Author: Nabil Mohamed
# Project: B5W3 – Insurance Risk Analytics & Predictive Modeling
# Module: Task 3 – Province-Level Statistical Validation
# ---------------------------------------------------------------------------

# -----------------------
# 📦 Standard Library Imports
# -----------------------
import os  # File path manipulation
import sys  # Module path extension
import warnings  # Warning suppression
import time  # Execution timing

# -----------------------
# 📦 Core Libraries
# -----------------------
import pandas as pd  # Data I/O and manipulation
import seaborn as sns  # Visual aesthetics
import matplotlib.pyplot as plt  # Plot rendering

# -----------------------
# 🔬 Statistical Libraries
# -----------------------
from scipy import stats  # For t-tests, Mann–Whitney, etc.


# -----------------------
# 🛠 Project Root Setup
# -----------------------
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir("..")  # Navigate to root if running from scripts/
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Ensure src/ modules importable
    print(f"✅ Project root added to sys.path: {project_root}")


# -----------------------
# 🧠 Custom Module Imports (from src/)
# -----------------------
from src.data_loader import InsuranceDataLoader
from src.hypothesis_testing.data_cleaner import DataCleaner
from src.hypothesis_testing.metric_definitions import MetricDefiner
from src.hypothesis_testing.group_segmenter import GroupSegmenter
from src.hypothesis_testing.hypothesis_tester import HypothesisTester
from src.hypothesis_testing.visual_tester import VisualTester


# -----------------------
# ⏱️ Start Execution Timer
# -----------------------
start = time.time()

# -----------------------
# 📥 Step 1: Load Insurance Dataset
# -----------------------
data_path = "data/raw/MachineLearningRating_v3.txt"
loader = InsuranceDataLoader(filepath=data_path)
try:
    df = loader.load()
    print(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load dataset: {e}")

# -----------------------
# 🧹 Step 2: Clean and Prepare
# -----------------------
cleaner = DataCleaner()
df = cleaner.drop_high_null_and_constant_columns(df)
df = cleaner.remove_duplicates(df)
df = cleaner.impute_missing_values(df)
df = cleaner.encode_categoricals(df)
df = cleaner.clip_outliers(
    df, numeric_cols=["TotalClaims", "TotalPremium", "CustomValueEstimate"]
)

# -----------------------
# 📊 Step 3: Define Risk KPIs
# -----------------------
metric_definer = MetricDefiner()
df = metric_definer.enrich_all(df)

# -----------------------
# 🧪 Step 4: Segment A/B by Province
# -----------------------
segmenter = GroupSegmenter()
try:
    province_df = segmenter.segment_by_column(
        df=df,
        column="Province",
        group_a_value="Western Cape",
        group_b_value="Gauteng",
        decode=True,
    )
    segmenter.summarize_group_counts(province_df)
except Exception as e:
    raise RuntimeError(f"❌ Province segmentation failed: {e}")

# ✅ Assert group presence
assert province_df["ABGroup"].nunique() == 2, "❌ ABGroup encoding incomplete or failed"

# -----------------------
# 🧪 Step 5: Run Statistical Tests
# -----------------------
tester = HypothesisTester()
kpi_metrics = ["ClaimFrequency", "ClaimSeverity", "Margin"]
results = []

for metric in kpi_metrics:
    try:
        result = tester.run_test(province_df, metric)
        print(
            f"\n📊 {metric} — {result['test_used']}\n"
            f"p-value: {result['p_value']:.4f}, Effect Size: {result['effect_size']:.3f}\n"
            f"Normality — Group A: {result['group_a_normal']} | Group B: {result['group_b_normal']}"
        )
        if result["p_value"] < 0.05:
            print("✅ Statistically significant difference detected.")
        else:
            print("⚠️ No significant difference.")
        results.append(result)
    except Exception as e:
        print(f"❌ Error running test on {metric}: {e}")

# -----------------------
# 💾 Step 6: Save Results
# -----------------------
results_df = pd.DataFrame(results)
try:
    os.makedirs("data/outputs/", exist_ok=True)
    print("💾 Saving results to CSV...")
    results_df.to_csv("data/outputs/hypothesis_results.csv", index=False)
    print("✅ Results saved: data/outputs/hypothesis_results.csv")
except Exception as e:
    print(f"❌ Failed to save CSV: {e}")

# -----------------------
# 📈 Step 7: Visualize KPI Distributions
# -----------------------
visualizer = VisualTester(df=province_df)
for metric in kpi_metrics:
    print(f"\n🔍 Visualizing: {metric} by Province Group (A/B)...")
    try:
        visualizer.plot_violin(metric)
    except Exception as e:
        print(f"❌ Violin plot failed for {metric}: {e}")
    try:
        visualizer.plot_boxplot(metric)
    except Exception as e:
        print(f"❌ Boxplot failed for {metric}: {e}")
    try:
        visualizer.plot_histogram_overlay(metric)
    except Exception as e:
        print(f"❌ Histogram failed for {metric}: {e}")

# -----------------------
# ✅ Done
# -----------------------
print(f"\n✅ Script completed in {round(time.time() - start, 2)} seconds.")
