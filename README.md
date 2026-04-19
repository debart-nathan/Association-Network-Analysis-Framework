# Association Network Analysis Framework

**Purpose**  
A reusable, dataset‑agnostic Python backend and Dash UI for building association networks from tabular data. The project separates concerns into three layers: Data, Associations, Network, and exposes a thin Dash frontend.

**Goals**
- Normalize arbitrary tabular datasets (CSV, Parquet, SQL, API).  
- Auto‑select and compute appropriate association metrics for numeric, categorical, and mixed data.  
- Build weighted graphs and compute network analytics.  
- Provide a reusable backend that can be used from notebooks, CLI, or a Dash app.

---

## Quickstart

```bash
# create and activate virtual environment
python -m venv .venv
# macOS Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# run tests
pytest -q

# start local Dash app
python -m association_framework.app
# open http://localhost:8050
```


## Installation Guide

**Prerequisites**
- Python 3.10 or newer
- Git optional
- You already created a `.env` file in the project root

---

## 1 Create and activate virtual environment

```bash
python -m venv .venv
# macOS Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```
---

## 2 Upgrade pip and install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3 Configure environment variables

You already have a .env. Ensure it contains at least:

| Name       | Purpose                         | Example            |
|------------|----------------------------------|--------------------|
| APP_ENV    | runtime environment              | development        |
| SECRET_KEY | Dash secret for sessions         | change_me_secure   |
| DATA_DIR   | default folder for datasets      | ./data             |
| CACHE_DIR  | cache for association matrices   | ./cache            |
| LOG_LEVEL  | logging verbosity                | INFO               |


The app reads .env using python-dotenv.

## 4 Run tests and start app

```bash
pytest -q
python -m association_framework.app
# open http://localhost:8050
```

