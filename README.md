# B5W3: Insurance Risk Analytics & Predictive Modeling Week 3 - 10 Academy

## 🗂 Challenge Context
This repository documents the submission for 10 Academy’s **B5W3: Insurance Risk Analytics & Predictive Modeling** challenge.
The goal is to support AlphaCare Insurance Solutions (ACIS) in optimizing underwriting and pricing by analyzing customer, vehicle, and claims data to:

- Identify low-risk customer segments

- Predict future risk exposure

- Enable data-driven premium optimization

This project simulates the role of a risk analyst at AlphaCare Insurance Solutions (ACIS), supporting actuarial and underwriting teams with data-driven insights for optimizing premium pricing and minimizing claims exposure.

The project includes:

- 🧹 Clean and structured ingestion of raw customer, vehicle, and claims datasets

- 📊 Multi-layered Exploratory Data Analysis (EDA) across customer, product, geographic, and vehicle dimensions

- 🧠 Modular profiling of loss ratio, outliers, and segment-specific profitability

- 🗃️ Defensive schema auditing and data quality validation routines

- 📦 Reproducible data versioning using DVC with Git and local cache integration

- 🧪 Scaffolded modeling pipeline for classification-based claims risk prediction (planned)

- ✅ Structured orchestration of insights through testable, class-based Python modules and `eda_orchestrator.py` runner script


## 🔧 Project Setup

To reproduce this environment:

1. Clone the repository:

```bash
git clone https://github.com/NabloP/b5w3-insurance-risk-modelling-challenge.git
cd b5w3-insurance-risk-modelling-challenge
```

2. Create and activate a virtual environment:
   
**On Windows:**
    
```bash
python -m venv insurance-challenge
.\insurance-challenge\Scripts\Activate.ps1
```

**On macOS/Linux:**

```bash
python3 -m venv insurance-challenge
source insurance-challenge/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ CI/CD (GitHub Actions)

This project uses GitHub Actions for Continuous Integration. On every `push` or `pull_request` event, the following workflow is triggered:

- Checkout repo

- Set up Python 3.10

- Install dependencies from `requirements.txt`

CI workflow is defined at:

    `.github/workflows/unittests.yml`

## 📁 Project Structure

<!-- TREE START -->
📁 Project Structure

solar-challenge-week1/
├── LICENSE
├── README.md
├── pytest.ini
├── requirements.txt
├── .github/
│   └── workflows/
│       ├── unittests.yml
├── data/
│   ├── cleaned/
│   ├── mappings/
│   │   ├── label_encodings.csv
│   ├── outputs/
│   │   ├── hypothesis_results.csv
│   │   ├── loss_ratio_bubble_map.png
│   │   ├── plots/
│   │   └── segments/
│   │       ├── province_comparison.csv
│   ├── processed/
│   │   ├── cleaned_insurance_data.csv
│   │   ├── enriched_insurance_data.csv
│   └── raw/
│       ├── MachineLearningRating_v3.txt
│       ├── MachineLearningRating_v3.txt.dvc
│       ├── opendb-2025-06-17.csv
│       ├── opendb-2025-06-17.csv.dvc
├── docs/
├── models/
├── notebooks/
│   ├── task-1-eda-statistical-planning.ipynb
│   ├── task-3-hypothesis-testing.ipynb
├── scripts/
│   ├── eda_orchestrator.py
│   ├── generate_tree.py
│   ├── hypothesis_testing_orchestrator.py
│   ├── version_datasets.py
├── src/
│   ├── data_loader.py
│   ├── eda/
│   │   ├── defensive_schema_auditor.py
│   │   ├── distribution_analyzer.py
│   │   ├── gender_risk_profiler.py
│   │   ├── geo_risk_visualizer.py
│   │   ├── iqr_outlier_detector.py
│   │   ├── numeric_plotter.py
│   │   ├── plan_feature_risk_profiler.py
│   │   ├── schema_auditor.py
│   │   ├── schema_guardrails.py
│   │   ├── temporal_analyzer.py
│   │   ├── vehicle_risk_profiler.py
│   └── hypothesis_testing/
│       ├── data_cleaner.py
│       ├── group_segmenter.py
│       ├── hypothesis_tester.py
│       ├── metric_definitions.py
│       ├── visual_tester.py
├── tests/
└── ui/
<!-- TREE END -->


## ✅ Status

- ☑️ Task 1 complete: Full EDA pipeline implemented across 10 modular risk layers (loss ratio, outliers, geo, schema, etc.)

- ☑️ Task 2 complete: DVC tracking initialized with Git integration, local remote configured, and raw datasets versioned

- ☑️ Task 3 complete: Fully modular A/B testing pipeline implemented with group segmentation, adaptive t-test or Mann–Whitney testing, effect size logging, and KPI visualizations. Province-level comparison between Western Cape and Gauteng now reproducible via `scripts/hypothesis_testing_orchestrator.py`.

- 🏗️ Task 4 scaffolded: Feature engineering and pricing optimization logic designed (implementation upcoming)

☑️ Project architecture: Fully modular `src/`, `scripts/`, and `notebooks/` structure with reproducible orchestration via `eda_orchestrator.py` and `v`ersion_datasets.py`


## 📦 What's in This Repo

This repository is structured to maximize modularity, reusability, and clarity:

- 📁 Layered Python module structure for risk profiling (src/eda/), geographic mapping (src/geo/), and schema auditing (src/)

- 🧪 CI-ready architecture using GitHub Actions for reproducible tests via pytest

- 📦 DVC integration for versioned tracking of raw and processed datasets (with local remote and cache routing)

- 🧹 Clean orchestration scripts (eda_orchestrator.py, version_datasets.py) for Task 1–2 reproducibility

- 🧠 Risk analysis modules written with defensive programming, strong validation, and class-based design

- 📊 Insightful plots (loss ratio heatmaps, bar charts, outlier maps) auto-rendered via orchestrator pipeline

- 🧾 Consistent Git hygiene with .gitignore, no committed .csv or .venv, and modular commit history

- 🧪 Notebook-first development approach supported by CLI runners and reusable core modules

- 🧪 Statistical hypothesis testing orchestrator for A/B segment comparison across provinces, genders, or zip codes, powered by `GroupSegmenter`, `HypothesisTester`, and `VisualTester`


- 🧠 **My Contributions:** All project scaffolding, README setup, automation scripts, and CI configuration were done from scratch by me


## 🔐 DVC Configuration & Versioning (Task 2)
This project uses **Data Version Control (DVC)** to ensure auditable and reproducible handling of insurance datasets across all preprocessing stages.

### ✅ Versioned Artifacts
The following DVC artifacts are tracked and committed to the repository:

File	| Purpose
--------|---------
data/raw/MachineLearningRating_v3.txt.dvc	| Tracks raw dataset (customer + claims)
data/raw/opendb-2025-06-17.csv.dvc	| Tracks auxiliary postal code metadata
.dvc/config	| Stores remote and cache settings
.gitignore	| Automatically updated to ignore large .csv files

Note: This project currently uses .dvc-style tracking (per file), not dvc.yaml pipelines. The dvc.yaml file will be added in Task 3–4 for full ML pipeline definition.

### 📦 DVC Remote Configuration
DVC is configured to use a local remote directory (outside the Git repo) for safe, decoupled storage:

```swift

Remote path: C:/Users/admin/Documents/GIT Repositories/dvc_remote/.dvc/cache
```

This is specified in .dvc/config as:

```ini
['cache']
    dir = C:/Users/admin/Documents/GIT Repositories/dvc_remote/.dvc/cache
```

And confirmed via:

```bash
dvc config cache.dir "C:/Users/admin/Documents/GIT Repositories/dvc_remote/.dvc/cache"
```

### 🔁 How to Push to DVC Remote

To sync all .dvc-tracked data to the configured local remote:

```bash
dvc add data/raw/MachineLearningRating_v3.txt
dvc add data/raw/opendb-2025-06-17.csv
git add data/raw/*.dvc .gitignore
git commit -m "Track raw datasets with DVC"
dvc push
```

### 🔧 Automation Support

For reproducibility, all versioning steps can be automated using:

```bash
python scripts/version_datasets.py
```

This script:

- Adds tracked datasets to DVC

- Commits .dvc files to Git

- Pushes artifacts to remote

- Logs all actions to dvc_logs/





## 🧪 Usage

This project supports both **script-based automation** and **notebook-driven workflows** across all four challenge tasks. Below are the primary orchestration scripts and how to use them.

---

### 🔬 Task 1 – Full EDA Pipeline

Run the complete multi-layered risk EDA pipeline using:

    python scripts/eda_orchestrator.py

This script executes all 10 analytical layers in sequence:

Layer | Focus Area
------|------------
1     | Schema structure audit (duplicates, types, nulls)
2A    | Descriptive statistics (mean, std, skew, kurtosis)
2B    | Histogram and boxplot visualization
3     | Monthly temporal trend analysis
4     | Geographic loss ratio heatmap and bar charts
5     | Vehicle model and make risk profiling
6     | Gender-based loss segmentation
7     | Policy segment comparison (CoverType, TermFrequency, etc.)
8     | IQR-based outlier flagging
9     | Defensive schema diagnostics (cardinality, constants)
10    | Guardrail-based schema cleanup and exclusions

Outputs are **printed inline only** for review — no files are saved by default.

---

### 💾 Task 2 – Dataset Versioning with DVC

To version all raw datasets and push to the configured DVC remote, run:

    python scripts/version_datasets.py

This script:

- Verifies DVC initialization
- Adds each file to DVC tracking (.dvc pointers)
- Commits pointer files to Git
- Pushes all tracked data to the local remote
- Logs actions to: dvc_logs/dvc_versioning_log_<DATE>.txt

Tracked artifacts include:

- data/raw/MachineLearningRating_v3.txt
- data/raw/opendb-2025-06-17.csv

Remote path is set to:
C:/Users/admin/Documents/GIT Repositories/dvc_remote/.dvc/cache

---

### 🧪 Task 3 – Province-Level Hypothesis Testing

To statistically compare risk KPIs across **Western Cape vs Gauteng**, run:

    python scripts/hypothesis_testing_orchestrator.py

This script:

- Loads, cleans, and prepares insurance data
- Derives 3 KPIs:
    - ClaimFrequency
    - ClaimSeverity
    - Margin
- Segments data into A/B groups using province labels
- Applies adaptive statistical tests:
    - Mann–Whitney U (nonparametric)
    - Independent t-test (parametric)
- Logs:
    - p-values
    - test method
    - effect size
    - group normality results
- Saves result table to:
    - data/outputs/hypothesis_results.csv
- Renders 3 visualizations per KPI:
    - Violin plot
    - Boxplot
    - Histogram overlay

Script can be extended to compare **Gender**, **ZipCode**, or any other categorical column by updating the GroupSegmenter.



## 🧠 Design Philosophy
This project was developed with a focus on:

- ✅ Modular Python design using classes, helper modules, and runners (clean script folders and testable code)
- ✅ High commenting density to meet AI and human readability expectations
- ✅ Clarity (in folder structure, README, and docstrings)
- ✅ Reproducibility through consistent Git hygiene and generate_tree.py
- ✅ Rubric-alignment (clear deliverables, EDA, and insights)

## 🚀 Author
Nabil Mohamed
AIM Bootcamp Participant
GitHub: [NabloP](https://github.com/NabloP)