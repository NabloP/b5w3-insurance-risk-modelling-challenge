"""
predictive_model_orchestrator.py – Task 4 Predictive Modeling Pipeline (B5W3)
===============================================================================
Orchestrates the full modeling pipeline for AlphaCare Insurance:
- Claim frequency classification (Logistic, RF, XGBoost)
- Claim severity regression (XGBoost)
- Expected premium calculation
- SHAP interpretability
- Leaderboard-ready metrics output

Author: Nabil Mohamed
Challenge: B5W3 – Insurance Risk Analytics & Predictive Modeling
"""

# ─────────────────────────────────────────────────────────────────────────────
# 📂 Project Root Setup
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

# Ensure we're in project root (for relative paths and src/ imports)
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir("..")
    print("📁 Changed working directory to project root")

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    print(f"✅ sys.path updated: {os.getcwd()}")

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Core Imports
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 📥 Load Data
# ─────────────────────────────────────────────────────────────────────────────
data_path = "data/processed/enriched_insurance_data.csv"
df = pd.read_csv(data_path)
print(f"✅ Loaded dataset: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 🎯 Target / Feature Prep
# ─────────────────────────────────────────────────────────────────────────────
from src.modeling.target_feature_builder import TargetFeatureBuilder

builder = TargetFeatureBuilder()
X, y_class, y_reg = builder.prepare(df)

# ─────────────────────────────────────────────────────────────────────────────
# 🔀 Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────
from src.modeling.train_test_splitter import TrainTestSplitter

splitter = TrainTestSplitter(test_size=0.2, random_state=42)
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = splitter.split(
    X, y_class, y_reg
)

# ─────────────────────────────────────────────────────────────────────────────
# ⚖️ Class Balancing via SMOTE
# ─────────────────────────────────────────────────────────────────────────────
from src.modeling.class_balancer import ClassBalancer

balancer = ClassBalancer(random_state=42)
X_train_balanced, y_class_train_balanced = balancer.balance(X_train, y_class_train)

# ─────────────────────────────────────────────────────────────────────────────
# 🧼 Median Imputation (post-SMOTE)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_train_balanced = pd.DataFrame(
    imputer.fit_transform(X_train_balanced), columns=X_train_balanced.columns
)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# ─────────────────────────────────────────────────────────────────────────────
# 🔍 Feature Scaling
# ─────────────────────────────────────────────────────────────────────────────
from src.modeling.feature_scaler import FeatureScaler

scaler = FeatureScaler(method="standard")
X_train_scaled, X_test_scaled = scaler.scale(X_train_balanced, X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Logistic Regression Classifier
# ─────────────────────────────────────────────────────────────────────────────
from src.modeling.logistic_model_trainer import LogisticModelTrainer

logreg = LogisticModelTrainer()
logreg.train(X_train_scaled, y_class_train_balanced)
y_pred_log = logreg.predict(X_test_scaled)
logreg_metrics = logreg.evaluate(y_class_test, y_pred_log)
logreg_metrics["model"] = "LogisticRegression"

# ─────────────────────────────────────────────────────────────────────────────
# 🌲 Random Forest Classifier
# ─────────────────────────────────────────────────────────────────────────────
from src.modeling.random_forest_trainer import RandomForestModelTrainer

rf = RandomForestModelTrainer(random_state=42)
rf.train(X_train_balanced, y_class_train_balanced)
y_pred_rf = rf.predict(X_test)
rf_metrics = rf.evaluate(y_class_test, y_pred_rf)
rf_metrics["model"] = "RandomForestClassifier"

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 XGBoost Classifier
# ─────────────────────────────────────────────────────────────────────────────
from src.modeling.xgboost_model_trainer import XGBoostModelTrainer

xgb_classifier = XGBoostModelTrainer(task_type="classification", random_state=42)
xgb_classifier.train(X_train_balanced, y_class_train_balanced)
xgb_class_results = xgb_classifier.evaluate(X_test, y_class_test)
xgb_class_results["model"] = "XGBoostClassifier"

# ─────────────────────────────────────────────────────────────────────────────
# 💡 XGBoost Regressor for Claim Severity
# ─────────────────────────────────────────────────────────────────────────────
from src.modeling.xgboost_regressor_trainer import XGBoostRegressorTrainer

xgb_regressor = XGBoostRegressorTrainer(random_state=42)
X_train_sev, y_train_sev = xgb_regressor.filter_claim_positive(
    X_train.copy(), y_reg_train.copy()
)
X_test_sev, y_test_sev = xgb_regressor.filter_claim_positive(
    X_test.copy(), y_reg_test.copy()
)
xgb_regressor.train(X_train_sev, y_train_sev)
xgb_reg_results = xgb_regressor.evaluate(X_test_sev, y_test_sev)
xgb_reg_results["model"] = "XGBoostRegressor"

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 SHAP Explainability (XGB Regressor)
# ─────────────────────────────────────────────────────────────────────────────
shap_vals = xgb_regressor.compute_shap(
    X_sample=X_test_sev,
    feature_names=X_test_sev.columns.tolist(),
    save_path="data/outputs/shap_summary.png",
)

# ─────────────────────────────────────────────────────────────────────────────
# 💸 Premium Estimation
# ─────────────────────────────────────────────────────────────────────────────
from src.modeling.expected_premium_calculator import ExpectedPremiumCalculator

premium_calc = ExpectedPremiumCalculator(margin=50.0)
premium_df = pd.DataFrame(
    {
        "predicted_claim_prob": xgb_class_results["y_pred"],
        "predicted_claim_severity": xgb_reg_results["y_pred"],
    }
)
premium_df = premium_calc.compute_expected_premium(premium_df)

# ─────────────────────────────────────────────────────────────────────────────
# 💾 Save Results for Reporting
# ─────────────────────────────────────────────────────────────────────────────
results_dir = "data/outputs/"
os.makedirs(results_dir, exist_ok=True)

# Save metrics
metrics_df = pd.DataFrame(
    [logreg_metrics, rf_metrics, xgb_class_results, xgb_reg_results]
)
metrics_df.to_csv(
    os.path.join(results_dir, "model_performance_metrics.csv"), index=False
)

# Save premium predictions
premium_df.to_csv(os.path.join(results_dir, "predicted_premiums.csv"), index=False)

print("✅ All Task 4 models trained, evaluated, and exported.")
