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
│   ├── outputs/
│   │   ├── loss_ratio_bubble_map.png
│   │   └── plots/
│   └── raw/
│       ├── MachineLearningRating_v3.txt.dvc
│       ├── opendb-2025-06-17.csv.dvc
├── docs/
├── models/
├── notebooks/
│   ├── task-1-eda-statistical-planning.ipynb
├── scripts/
│   ├── eda_orchestrator.py
│   ├── generate_tree.py
│   ├── version_datasets.py
├── src/
│   ├── data_loader.py
│   └── eda/
│       ├── defensive_schema_auditor.py
│       ├── distribution_analyzer.py
│       ├── gender_risk_profiler.py
│       ├── geo_risk_visualizer.py
│       ├── iqr_outlier_detector.py
│       ├── numeric_plotter.py
│       ├── plan_feature_risk_profiler.py
│       ├── schema_auditor.py
│       ├── schema_guardrails.py
│       ├── temporal_analyzer.py
│       ├── vehicle_risk_profiler.py
├── tests/
└── ui/
<!-- TREE END -->


## ✅ Status

- ☑️ Task 1 complete: Full EDA pipeline implemented across 10 modular risk layers (loss ratio, outliers, geo, schema, etc.)

- ☑️ Task 2 complete: DVC tracking initialized with Git integration, local remote configured, and raw datasets versioned

- 🏗️ Task 3 scaffolded: Modeling modules prepared for claims classification and segment-level risk prediction

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

### 🎛️ Option 1: Using the Streamlit App

The Streamlit UI (`ui/app_streamlit.py`) provides an interactive interface to perform both review scraping and cleaning with no code required.

**To launch the app locally:**
```bash
streamlit run ui/app_streamlit.py
```

**🧩 Streamlit Features:**

- Scrape Google Play reviews for CBE, BOA, or Dashen

- Export reviews per-bank or as a combined dataset

- Preview scraped files in-app

- Clean any raw file from `data/raw/`

- View sidebar diagnostics for:

    - Missing fields dropped

    - Blank reviews removed

    - Duplicate `reviewIds` filtered

- Download cleaned outputs directly

All exports are timestamped and saved to `data/raw/` or `data/cleaned/` depending on context.


### 🐍 Option 2: Using Python Scripts
For automated, reproducible runs from the command line or notebooks, use the modular runners in the `scripts/` folder.

**🔹 Scraping Reviews**
To scrape reviews from one or more banks and export to CSV:

```python
 scripts/scraping_runner.py --bank CBE --num_reviews 100
 ```

Options:

- `--bank`: one of `CBE`, `BOA`, `Dashen`, or `all`

- `--num_reviews`: maximum number of reviews per app


**🔹 Cleaning Reviews**
To clean a raw file and export the cleaned result:

```python 
scripts/cleaning_runner.py --input_file data/raw/reviews_BOA_20250607_124729.csv
```

This removes:

- Rows with missing fields

- Blank or whitespace-only reviews

- Duplicate entries by `reviewId`

Cleaned files are saved under `data/cleaned/`.

### 🔁 How It Works Internally

Both the Streamlit app and script-based runners share the same core logic, implemented in the following modules:

- `src/scraper/review_scraper.py` – Fetches reviews from the Play Store

- `src/cleaning/review_cleaner.py` – Cleans and validates reviews

- `src/utils/preprocessing.py` – Shared text preprocessing functions

- `scripts/run_streamlit.py` – Optional wrapper for launching UI from CLI

- `scripts/generate_tree.py` – Auto-generates folder tree for `README.md`

Each module is written using object-oriented principles and is fully reusable across CLI, notebook, and UI contexts.


### 🧠 NLP Pipeline + Storage + Visuals

This section summarizes the full enrichment, database, and insights flow from **Task 2**, **Task 3**, and **Task 4**.

---

#### 🔹 Task 2: Sentiment + Theme Enrichment

Run the full sentiment and theme extraction pipeline with:

```bash
python scripts/sentiment_pipeline.py
```

What it does:

- 🧹 Loads cleaned reviews from `data/cleaned/`

- 🔡 Normalizes text using **SymSpell** and **spaCy**

- 💬 Applies ensemble sentiment scoring:

    - `VADER`
    - `TextBlob`
    - `DistilBERT` (locally loaded from `models/`)

- 📊 Assigns each review:

    - `ensemble_sentiment` label (bullish, neutral, bearish)
    - `sentiment_uncertainty` (std deviation of ensemble)
    - `sentiment_mismatch_flag` if BERT disagrees with VADER/TextBlob

🔍 Extracts:

- Top **keywords** (via TF-IDF)

- Top **keyphrases** (noun chunking)

- UX **themes** (from curated keyword–theme dictionaries)

💾 Saves outputs to:

- `data/outputs/reviews_enriched_all.csv`

- `data/outputs/reviews_with_sentiment_themes.csv`

✅ Designed to **auto-run** with no input flags or prompts.

---

### 🛢️ Task 3: Oracle XE Database Integration (Relational Storage)

The project includes full support for enterprise-grade relational storage using Oracle XE. Reviews are inserted into a normalized schema with integrity constraints.

**Key Features:**

- ✅ `banks` and `reviews` tables in 3NF

- ✅ Modular insert logic via `oracle_insert.py`

- ✅ Connection security via `os.getenv()` (no hardcoded secrets)

- ✅ Full rollback + diagnostic prints on insert failure

- ✅ Schema DDL included for reproducibility

### 🧱 Schema Design

1. **`banks` Table**

| Column      | Type        | Description            |
|-------------|-------------|------------------------|
| `bank_id`   | INTEGER PK  | Unique bank identifier |
| `bank_name` | VARCHAR(50) | Name of the bank       |

2. reviews Table

| Column        | Type             | Description                          |
|---------------|------------------|--------------------------------------|
| `review_id`   | VARCHAR2(100) PK | Unique review ID                     |
| `bank_id`     | INTEGER FK       | Foreign key referencing banks        |
| `review_text` | CLOB             | Full text of the review              |
| `rating`      | INTEGER          | App rating (1 to 5)                  |
| `review_date` | DATE             | Parsed date of review                |
| `source`      | VARCHAR2(50)     | Always 'Google Play' in this project |

### 🔌 Usage

To insert reviews into Oracle XE after enrichment:

```bash
python scripts/oracle_insert.py
```

What it does:

- 📦 Loads enriched review file from:

    - `data/outputs/reviews_with_sentiment_themes.csv`

- 🏗️ Defines and initializes schema:

    - `reviews` table with fields for rating, sentiment, text, and metadata
    - `themes` table (if normalized)

- 🔗 Connects to Oracle XE (uses `src/db/oracle_connector.py`)

- 🧩 Inserts cleaned records with:

    - Full error handling
    - Per-row diagnostic feedback if insertions fail

- 🛑 Can be rerun idempotently (e.g. if schema already exists)

☑️ Safe to run from CLI after enrichment is complete.

📁 Related Files:

- `scripts/oracle_insert.py`: Insert runner

- `src/db/oracle_connector.py`: Secure DB connection

- `task-3-oracle-storage.ipynb`: Notebook walkthrough of schema + insert test


**📄 [Oracle XE Storage Documentation](docs/oracle_storage_overview.md)**

---

### 📈 Task 4: Visualization & KPI Dashboard
Run all Task 4 UX and sentiment visualizations with:

```bash
python scripts/visualize_insights.py
```

What it does:

- 🧠 Loads enriched reviews from `data/outputs/reviews_with_sentiment_themes.csv`

- 📊 Auto-generates plots for:

    - 📈 Average rating per theme per bank (heatmap)
    - 🔥 Complaint clusters (negative sentiment by theme and bank)
    - ☁️ Word clouds (bank-specific vocabulary)
    - 🛠️ Feature request volume (negative themes only)
    - 💎 Bubble chart for theme occurrence vs avg. rating

- 📍 Top-level theme in bubble chart is editable in-code

- 📂 Does not auto-save — plots are rendered inline

- ✅ Designed for notebook-first and CLI workflows

No flags or CLI inputs required — this script auto-runs all diagnostics sequentially.


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