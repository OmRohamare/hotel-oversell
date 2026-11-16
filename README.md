# Hotel Oversell Buffer

A reproducible pipeline that sets an oversell buffer for each hotel on each arrival night. A cost rule from the client scores every method.

> **Scope of this document.** This README comes from the source code in the ZIP file and from `hotels.csv`. The ZIP contains no notebook, no API, no license file and no slide deck. Where the code does not give an answer, this document says: "Purpose could not be determined from source code."

---

## Overview

### Business problem

Some guests cancel their booking. A cancelled booking leaves a room empty. A hotel can sell extra rooms to cover these cancellations. This practice is called *overselling*.

Overselling has two risks:

- **Empty room.** More guests cancel than the hotel expected. The room stays empty. The cost is the average stay value of that night (`adr` x `nights`).
- **Walked guest.** Fewer guests cancel than the hotel expected. The hotel sends a guest to another hotel. The cost is EUR 300 for each walked guest.

The client also sets a limit. A policy is acceptable only if it walks 10 guests or fewer for each 1,000 arrivals.

### Source of the brief

The Build 2 announcement deck sets the task. The client is a two-hotel group in Portugal: a city hotel in Lisbon and a resort in the Algarve. The deck gives the client request in this form: for each arrival night at each hotel, say how many rooms beyond capacity the hotel can keep selling. The hotel must not walk more than one guest in a hundred.

The deck says the client and its brief are fictional. It says the bookings are the open Hotel Booking Demand datasets (Antonio, de Almeida and Nunes, 2019), under CC BY 4.0.

The deck names two commands for this repository:

- `python src/backtest.py` (proof-period table of all methods)
- `python src/nightly.py --from 2017-08-18 --nights 14` (table for the next 14 nights)

### What this project does

This project predicts the cancellation probability of each booking. It adds the probabilities for each hotel and arrival night. It then subtracts a *buffer* to get the number of rooms to oversell:

```
rooms to oversell = max(0, floor(expected cancellations - buffer))
```

The code limits this number to the number of bookings for that night. A negative buffer sells more rooms than the expected cancellations.

The project compares four methods. It chooses a buffer for each method. It then scores every method with the client rule in `contract.py`.

---

## Features

- Four methods for cancellation probability: flat rule, segment table, logistic regression, random forest.
- One cost rule for all methods (`contract.py`). The code does not change this rule.
- A time-based split: train, choose, proof. The split is never random.
- A check that no column from after the booking enters the model (`features.assert_no_leakage`).
- Two data variants: `dedup` (primary) and `raw` (sensitivity). The pipeline runs both.
- Two buffer families: `constant` and `spread`.
- A bootstrap estimate of the saving against the flat rule.
- A log of every look at the proof period (`docs/PROVE_RUNS.txt`).
- Data version control with DVC and an S3 remote.
- Optional MLflow tracking on DagsHub.
- Secret redaction in log messages (`src/logger.py`).
- CI with GitHub Actions: lint, test, and a pipeline run up to `freeze_buffer`.

---

## Architecture

### Main components

| Component | Where | Role |
|---|---|---|
| Client rule | `contract.py` | Defines the split dates, the cost of empty rooms and walks, and the walk limit |
| Pipeline code | `src/` | Validates data, splits data, trains methods, chooses buffers, scores the proof period |
| Pipeline definition | `dvc.yaml`, `dvc.lock` | Defines five stages and records their inputs and outputs |
| Settings | `params.yaml` | Holds all model, grid and split settings |
| Data store | `.dvc/config`, `data/raw/hotels.csv.dvc` | Points to the raw CSV in S3 |
| Tracking | `src/tracking.py` | Sends proof results to MLflow on DagsHub |
| Tests | `tests/` | Check each stage with synthetic data |
| CI | `.github/workflows/ci.yaml` | Runs lint, tests and the pipeline |

### Data flow

```
data/raw/hotels.csv  (DVC, S3)
        |
        v
[1 validate]  --> data/interim/clean.parquet
                  reports/brief_checks.csv, decisions_log.csv,
                  duplicates_by_hotel.csv, base_rates.csv, dup_facts.csv
        |
        v
[2 split]     --> data/processed/{dedup,raw}/{train,choose,proof}.parquet
                  reports/split_summary.csv
        |
        v
[3 backtest]  --> models/{dedup,raw}/*.joblib
                  reports/choose_diagnostics_*.csv, drift_table_*.csv
        |
        v
[4 freeze_buffer] --> reports/frozen_buffers_{dedup,raw}.json
                      reports/choose_table_*.csv, variant_comparison.csv,
                      reports/buffer_sweep_*.png
        |
        v
[5 prove]  (run by hand, one time)
              --> reports/headline_table.csv, poc_method_table_*.csv,
                  poc_per_hotel_*.csv, poc_bootstrap.csv,
                  reports/DECISIONS_dedup_entry.md
```

### Services

The project has no web service. It has two external services:

- **AWS S3** stores the raw data through DVC (`s3://hotel-oversell/dvc`, region `ap-southeast-2`).
- **DagsHub MLflow** stores the optional tracking run.

### Modules

| Module | Group |
|---|---|
| `src/params.py`, `src/logger.py`, `src/data.py` | Shared base |
| `src/features.py` | Feature rules |
| `src/validate.py`, `src/split.py` | Data stages |
| `src/backtest.py`, `src/rf_calibrate.py` | Methods |
| `src/costs.py`, `src/tune.py`, `src/freeze_buffer.py`, `src/prove.py` | Cost and buffer logic |
| `src/nightly.py` | Use of frozen buffers |
| `src/tracking.py` | MLflow helpers |

---

## Repository Structure

### Tree

```
hotel-oversell-main/
├── .dvc/
│   ├── .gitignore
│   └── config
├── .dvcignore
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yaml
├── .gitignore
├── README.md
├── artifacts/                  (output of src/nightly.py)
│   └── .gitkeep
├── backups/
│   └── before-tuning/          (copy of earlier outputs)
│       ├── data/{interim,processed/{dedup,raw}}/*.parquet
│       ├── models/{dedup,raw}/*.joblib
│       ├── dvc.yaml, dvc.lock
│       ├── DECISIONS_dedup_entry.md, PROVE_RUNS.txt
│       └── experiment_comparison.{csv,txt}, headline_table.csv,
│           poc_bootstrap.csv, poc_method_table_*.csv, poc_per_hotel_*.csv
├── contract.py
├── data/
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   └── raw/
│       ├── .gitkeep
│       └── hotels.csv.dvc
├── docs/
│   ├── DATA_DICTIONARY.md      (empty file)
│   ├── MODEL_CARD.md           (empty file)
│   ├── PROVE_RUNS.txt
│   └── RUNBOOK.md
├── dvc.lock
├── dvc.yaml
├── logs/.gitkeep
├── models/
│   ├── .gitignore
│   └── .gitkeep
├── params.yaml
├── reports/                    (32 files, see "Output Artifacts")
├── requirements.lock.txt
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── backtest.py
│   ├── costs.py
│   ├── data.py
│   ├── features.py
│   ├── freeze_buffer.py
│   ├── logger.py
│   ├── nightly.py
│   ├── params.py
│   ├── prove.py
│   ├── rf_calibrate.py
│   ├── split.py
│   ├── tracking.py
│   ├── tune.py
│   └── validate.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_costs.py
    ├── test_features.py
    ├── test_pipeline.py
    ├── test_split.py
    └── test_validate.py
```

### Root files

**`contract.py`**
- Purpose: Holds the client scoring rule. The file says: read it, do not change it.
- Key functions: `split_periods(df)`, `nightly_table(bookings, prob)`, `night_cost(nights, oversell)`.
- Constants: `WALK_COST = 300.0`, `WALK_CEILING = 10.0`, `TRAIN_END = "2016-07-01"`, `PROOF_START = "2017-01-01"`, `REQUIRED`.
- Inputs: a bookings table with the columns `hotel`, `arrival_date`, `is_canceled`, `adr`, `nights`; one probability for each booking.
- Outputs: a table with one row for each hotel and arrival night (`book`, `cancelled`, `expected`, `spread`, `stay_value`); a cost summary dictionary.
- Dependencies: numpy, pandas.
- Note: The file mentions `docs/EVALUATION_CONTRACT.md` and `tests/test_contract.py`. Neither file is in the ZIP.

**`params.yaml`**
- Purpose: Holds all settings (seed, cleaning, variants, model settings, buffer grid, bootstrap, MLflow).
- Inputs: none. Outputs: none (read by code and by DVC).
- Dependencies: PyYAML (through `src/params.py`).

**`dvc.yaml`**
- Purpose: Defines the five stages `validate`, `split`, `backtest`, `freeze_buffer`, `prove`.
- Note: A comment says the `prove` stage is frozen. The file has no `frozen: true` key. A bare `dvc repro` therefore can run `prove`. See Troubleshooting.

**`dvc.lock`**
- Purpose: Records the hash of each stage input and output.
- Do not edit this file by hand.

**`.dvc/config`**
- Purpose: Sets the DVC remote `storage` to `s3://hotel-oversell/dvc` in `ap-southeast-2`.

**`.dvcignore`**
- Purpose: Lists DVC ignore patterns. The file holds only the default comment.

**`.env.example`**
- Purpose: Template for the local `.env` file.
- Variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `DAGSHUB_USER_TOKEN`, `MLFLOW_TRACKING_URI`.
- Note: `src/tracking.py` also reads `DAGSHUB_USERNAME`. The template does not list it.

**`.gitignore`**
- Purpose: Excludes the virtual environment, caches, secrets, logs, data files, model files and editor files from Git.

**`requirements.txt`**
- Purpose: Lists 13 packages without versions: pandas, numpy, scikit-learn, pyarrow, PyYAML, joblib, dvc, dvc-s3, mlflow, dagshub, python-dotenv, pytest, ruff, matplotlib.

**`requirements.lock.txt`**
- Purpose: Lists pinned versions of all installed packages.

**`.github/workflows/ci.yaml`**
- Purpose: Runs CI on each pull request and on each push to `main`.
- Jobs: `lint-and-test` (`ruff check .`, `pytest -q`); `pipeline` (`dvc pull`, `dvc repro freeze_buffer`, check that `dvc.lock` is current, upload `reports/`).
- Dependencies: Python 3.10, AWS secrets in GitHub.

### `/src`

**`src/params.py`**
- Purpose: Loads `params.yaml` and defines the project paths.
- Key functions: `load_params`, `require_params`, `variant_slugs`, `frozen_path`, `proof_table_path`.
- Inputs: `params.yaml`. Outputs: a settings dictionary; path constants.
- Dependencies: PyYAML, `src.logger`.

**`src/logger.py`**
- Purpose: Gives all modules one logging format. Writes to the console and to `logs/hotel_oversell.log`. Hides secrets.
- Key items: `get_logger`, `RedactSecretsFilter`, `log_dataframe_summary`, `log_stage` (decorator), `run_cli`.
- Inputs: the `LOG_LEVEL` environment variable (default `INFO`). Outputs: log lines.
- Dependencies: Python standard library.

**`src/data.py`**
- Purpose: Reads and writes data.
- Key functions: `read_raw`, `arrival_date`, `read_parquet_checked`, `load_period`, `write_parquet`, `write_json`, `read_json`, `load_contract`.
- Inputs: `data/raw/hotels.csv` (through DVC or local file), parquet files, JSON files.
- Outputs: DataFrames; files.
- Dependencies: pandas, pyarrow, dvc (loaded only for `source: dvc`).
- Rule: `read_raw` stops if the shape differs from 119,390 rows and 32 columns.

**`src/features.py`**
- Purpose: Defines which columns the models may use.
- Key items: `AUDIT` (four groups), `FEATURES`, `CAT`, `NUM`, `assert_no_leakage`, class `RarePooler`.
- Groups in `AUDIT`: known at booking; changes after booking (excluded); the outcome itself; bookkeeping.
- `RarePooler`: keeps the top categories for `country`, `agent` and `company` (counts from training data only). It sets all other values to `other`.
- Inputs: a training DataFrame. Outputs: a DataFrame with feature columns only.
- Dependencies: pandas.

**`src/validate.py`** (stage 1)
- Purpose: Checks the raw data, flags repeated rows, repairs impossible values.
- Key functions: `check_columns`, `flag_duplicates`, `duplicate_report`, `brief_checks`, `clean`, `main`. Class `DataValidationError`.
- Inputs: `data/raw/hotels.csv`, `params.yaml` (`cleaning.adr_cap`, `data.*`).
- Outputs: `data/interim/clean.parquet`; `reports/brief_checks.csv`, `decisions_log.csv`, `duplicates_by_hotel.csv`, `base_rates.csv`, `dup_facts.csv`.
- Dependencies: pandas, `src.data`, `src.features`.
- The code drops no row. It adds `is_dup`, `row_key` and `nights`.

**`src/split.py`** (stage 2)
- Purpose: Splits the bookings by arrival date for each variant.
- Key functions: `check_split`, `check_no_group_straddles`, `variant_frames`, `main`.
- Inputs: `data/interim/clean.parquet`, `contract.split_periods`.
- Outputs: `data/processed/{dedup,raw}/{train,choose,proof}.parquet`; `reports/split_summary.csv`.
- Dependencies: pandas, `contract`.

**`src/backtest.py`** (stage 3)
- Purpose: Trains the four methods on the train period. Writes diagnostics for the choose period.
- Key classes: `FlatRate`, `SegmentTable`, `LogisticMethod`, `ForestMethod`.
- Key functions: `build_methods`, `fit_all`, `save_methods`, `load_methods`, `nightly_tables`, `choose_diagnostics`, `drift_table`, `show_proof_tables`, `cli`.
- Inputs: `train.parquet` and `choose.parquet` for each variant; `params.yaml`.
- Outputs: `models/{dedup,raw}/{flat_rule,segment_table,logistic,random_forest}.joblib`; `reports/choose_diagnostics_*.csv`; `reports/drift_table_*.csv`.
- Dependencies: scikit-learn, joblib, pandas, numpy.
- Options: `--stage-only` (train and save only); `--prove` (also runs the proof stage).

**`src/costs.py`**
- Purpose: Turns expected cancellations into rooms to oversell. Compares methods in euros.
- Key functions: `oversell`, `night_cost_vector`, `proof_table`, `per_hotel_table`, `bootstrap_savings`, `check_nightly`.
- Inputs: nightly tables, frozen buffers. Outputs: cost tables.
- Dependencies: numpy, pandas, `contract`.

**`src/tune.py`**
- Purpose: Searches the buffer grid on the choose period.
- Key functions: `build_grid`, `sweep`, `pick`, `on_grid_edge`, `choose_headline`.
- Rule in `pick`: take the cheapest buffer that meets the walk limit. If none meets it, take the buffer with the fewest walks and flag it.
- Dependencies: numpy, pandas, `src.costs`.

**`src/freeze_buffer.py`** (stage 4)
- Purpose: Chooses one buffer for each method and variant. Saves the buffers. Never opens the proof period.
- Key functions: `freeze_variant`, `compare_variants`, `plot_sweeps`, `main`.
- Inputs: saved models, `choose.parquet`, `params.yaml` (`buffer_grid`).
- Outputs: `reports/frozen_buffers_{dedup,raw}.json`, `choose_table_*.csv`, `variant_comparison.csv`, `buffer_sweep_*.png`.
- Dependencies: matplotlib, pandas.

**`src/prove.py`** (stage 5)
- Purpose: Scores the proof period with the frozen buffers. Writes the final tables.
- Key functions: `prove_variant`, `headline_table`, `ranking`, `decision_entry`, `record_run`, `log_to_mlflow`, `main`.
- Inputs: frozen buffer files, saved models, `proof.parquet`.
- Outputs: `reports/headline_table.csv`, `poc_method_table_*.csv`, `poc_per_hotel_*.csv`, `poc_bootstrap.csv`, `DECISIONS_dedup_entry.md`; one new line in `docs/PROVE_RUNS.txt`.
- Dependencies: pandas, `src.costs`, optional `src.tracking`, `git` (for the commit hash).

**`src/rf_calibrate.py`**
- Purpose: Tests a recalibrated random forest with expanding-window folds. Reads train and choose periods only. This script is not a DVC stage.
- Key items: `Calibrator` (`intercept`, `platt`, `isotonic`), `CalibratedForest`, `forward_folds`, `score_grid`, `run_candidate`, `main`.
- Inputs: `params.yaml` (`rf_calibration`, `forward_chaining`), train and choose parquet files of the primary variant.
- Outputs: `reports/rf_forward_chain.csv`, `reports/rf_forward_chain_folds.csv`.
- Dependencies: scikit-learn, numpy, pandas.
- The script stops if the data reaches the proof period.

**`src/nightly.py`**
- Purpose: Applies the frozen buffers to a window of nights.
- Key functions: `parse_args`, `main`.
- Inputs: `--from DATE`, `--nights N`, optional `--variant`, optional `--method`; `data/interim/clean.parquet`; frozen buffers; saved models.
- Outputs: `artifacts/nightly_<date>_<N>n.csv` (columns `book`, `expected`, `spread`, `buffer`, `rooms_to_oversell`).
- Dependencies: pandas, `src.backtest`, `src.costs`, `contract`.
- Limit: `contract.nightly_table` needs the real outcome column. The script header says that live use with undecided bookings needs a change.

**`src/tracking.py`**
- Purpose: Helpers for MLflow on DagsHub.
- Key functions: `setup_tracking`, `tracked_run`, `start_run`, `end_run`, `log_params`, `log_metrics`, `log_artifact`, `log_model_reference`.
- Inputs: `MLFLOW_TRACKING_URI`, `DAGSHUB_USERNAME`, `DAGSHUB_USER_TOKEN` (or `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`).
- Outputs: MLflow runs.
- Dependencies: mlflow, python-dotenv.
- Only `src/prove.py` calls this module.

### `/tests`

The folder holds 69 test functions.

| File | Tests | What it checks |
|---|---|---|
| `conftest.py` | fixtures | `build_raw` makes synthetic bookings; fixtures `make_raw` and `contract` |
| `test_costs.py` | 16 | Buffer formulas, input checks, contract constants, proof table, bootstrap input check |
| `test_features.py` | 14 | Leakage checks, feature groups, rare-category pooling |
| `test_pipeline.py` | 15 | Runs all stages on synthetic data; checks models, frozen buffers, proof runs, `nightly` |
| `test_split.py` | 9 | Period order, overlap, empty period, duplicate groups |
| `test_validate.py` | 15 | Column audit, duplicate flags, cleaning rules |

The tests use synthetic data. They need no AWS access.

### `/docs`

| File | Purpose |
|---|---|
| `RUNBOOK.md` | Team setup and daily workflow (Git, DVC, AWS, MLflow) |
| `PROVE_RUNS.txt` | One line for each proof run: time, commit, hash of frozen buffers |
| `DATA_DICTIONARY.md` | Empty file (0 bytes) |
| `MODEL_CARD.md` | Empty file (0 bytes) |

### `/data`, `/models`, `/artifacts`, `/logs`

These folders hold generated files. Git tracks only the `.gitkeep` files. DVC tracks `data/raw/hotels.csv` through `hotels.csv.dvc` (MD5 `5bf588c5a949443e021fb7c847d31b27`, 16,855,599 bytes).

### `/backups/before-tuning`

- Purpose: Holds a copy of earlier data, models, DVC files and reports.
- No source file reads this folder. The folder name suggests a copy from before a tuning step. The code does not confirm this.

### `/reports`

See "Output Artifacts".

---

## File Summary Table

| File | Purpose | Key Components | Dependencies |
|---|---|---|---|
| `contract.py` | Client cost rule and split dates | `split_periods`, `nightly_table`, `night_cost` | numpy, pandas |
| `params.yaml` | All settings | `seed`, `models`, `buffer_grid`, `variants` | PyYAML |
| `dvc.yaml` | Pipeline definition | 5 stages | DVC |
| `dvc.lock` | Pipeline hashes | stage records | DVC |
| `.dvc/config` | DVC remote | `s3://hotel-oversell/dvc` | dvc-s3 |
| `.env.example` | Environment template | 5 variables | none |
| `requirements.txt` | Package list | 13 packages | pip |
| `requirements.lock.txt` | Pinned packages | full pin list | pip |
| `.github/workflows/ci.yaml` | CI | `lint-and-test`, `pipeline` | GitHub Actions, AWS secrets |
| `src/params.py` | Settings and paths | `load_params`, `require_params`, `variant_slugs` | PyYAML |
| `src/logger.py` | Logging | `get_logger`, `RedactSecretsFilter`, `log_stage` | standard library |
| `src/data.py` | Data input and output | `read_raw`, `load_period`, `load_contract` | pandas, pyarrow, dvc |
| `src/features.py` | Feature rules | `AUDIT`, `FEATURES`, `RarePooler` | pandas |
| `src/validate.py` | Stage 1 | `flag_duplicates`, `clean`, `brief_checks` | pandas |
| `src/split.py` | Stage 2 | `check_split`, `variant_frames` | pandas, `contract` |
| `src/backtest.py` | Stage 3 | `FlatRate`, `SegmentTable`, `LogisticMethod`, `ForestMethod` | scikit-learn, joblib |
| `src/costs.py` | Cost logic | `oversell`, `proof_table`, `bootstrap_savings` | numpy, pandas |
| `src/tune.py` | Buffer search | `build_grid`, `sweep`, `pick` | numpy, pandas |
| `src/freeze_buffer.py` | Stage 4 | `freeze_variant`, `compare_variants` | matplotlib |
| `src/prove.py` | Stage 5 | `prove_variant`, `record_run`, `decision_entry` | pandas, mlflow (optional) |
| `src/rf_calibrate.py` | Forest calibration test | `Calibrator`, `CalibratedForest`, `forward_folds` | scikit-learn |
| `src/nightly.py` | Nightly use | `parse_args`, `main` | pandas |
| `src/tracking.py` | MLflow helpers | `tracked_run`, `log_metrics` | mlflow, python-dotenv |
| `tests/*.py` | Tests | 69 test functions | pytest |
| `docs/RUNBOOK.md` | Team guide | setup, workflow, fixes | none |
| `docs/PROVE_RUNS.txt` | Proof-run log | 7 lines | none |

---

## Prerequisites

- Python 3.10 or later (CI uses 3.10).
- Git.
- An AWS account with access to the S3 bucket `hotel-oversell` (to pull data with DVC).
- AWS CLI.
- A DagsHub account and token (only for MLflow tracking).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OmRohamare/hotel-oversell.git
   cd hotel-oversell
   ```
   (The URL comes from `docs/RUNBOOK.md`.)
2. Make a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
   ```
3. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
5. Set up AWS credentials:
   ```bash
   aws configure
   ```
6. Get the data:
   ```bash
   dvc pull
   ```

---

## Configuration

### Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | DVC | S3 access |
| `MLFLOW_TRACKING_URI` | `src/tracking.py` | MLflow server address |
| `DAGSHUB_USERNAME` | `src/tracking.py` | DagsHub user name (not in `.env.example`) |
| `DAGSHUB_USER_TOKEN` | `src/tracking.py` | DagsHub token |
| `LOG_LEVEL` | `src/logger.py` | Log level, default `INFO` |

Never commit `.env`.

### `params.yaml`

| Key | Value | Meaning |
|---|---|---|
| `seed` | 42 | Random seed |
| `data.source` | `dvc` | Read the CSV through DVC. Use `local` to read `data/raw/hotels.csv` directly |
| `data.expected_rows`, `expected_cols` | 119390, 32 | Shape check |
| `cleaning.adr_cap` | 2000 | Upper limit for `adr` |
| `variants.primary`, `variants.list` | `dedup`; `dedup`, `raw` | Data variants |
| `features.min_frequency` | 20 | Minimum count for one-hot categories (logistic) |
| `features.pool_top` | country 20, agent 30, company 10 | Categories kept before pooling to `other` |
| `models.segment_table.shrink_m` | 10 | Shrink strength toward the hotel rate |
| `models.logistic` | `C` 0.5, `max_iter` 2000 | Logistic settings |
| `models.random_forest` | 500 trees, `min_samples_leaf` 100, `max_features` 0.3, `max_depth` 8, `max_samples` 0.5 | Forest settings |
| `buffer_grid` | constant -10 to 25; spread 0 to 6 in steps of 0.25 | Buffer search grid |
| `bootstrap.n_boot` | 2000 | Bootstrap resamples |
| `rf_calibration` | `platt`, 3 months | Used by `src/rf_calibrate.py` |
| `forward_chaining` | 9 months minimum train, 3-month folds, buffer -6 to 10, target 6.5 walks per 1,000 | Used by `src/rf_calibrate.py` |
| `mlflow` | enabled, experiment `hotel-oversell` | Tracking |

**Known gaps in settings:**
- `build_methods` in `src/backtest.py` passes `n_estimators`, `min_samples_leaf`, `max_features`, `n_jobs` and `random_state` to the forest. It does not pass `max_depth` or `max_samples`. These two settings have no effect in the code.
- The header of `src/backtest.py` says "300 trees, at least 20 bookings per leaf". `params.yaml` says 500 trees and 100 per leaf. The code uses `params.yaml`.
- `rf_calibration.method` is `platt`. The saved report `reports/rf_forward_chain.csv` names an `intercept` calibration. The run that made that report used a different setting.

---

## Usage

Run all commands from the repository root.

| Task | Command |
|---|---|
| Run stages 1 to 4 | `dvc repro freeze_buffer` |
| Run one stage | `python -m src.validate` (same for `split`, `freeze_buffer`) |
| Train and save models only | `python -m src.backtest --stage-only` |
| Train, freeze buffers, print proof table | `python src/backtest.py` |
| Score the proof period (one time only) | `python src/backtest.py --prove` |
| Test forest calibration | `python -m src.rf_calibrate` |
| Make a schedule for 14 nights | `python src/nightly.py --from 2017-08-18 --nights 14` |
| Run tests | `pytest -q` |
| Lint | `ruff check .` |

---

## Example Workflow

1. Run the tests:
   ```bash
   pytest -q
   ```
2. Pull the data:
   ```bash
   dvc pull
   ```
3. Reproduce stages 1 to 4:
   ```bash
   dvc repro freeze_buffer
   ```
4. Read the buffers and the variant comparison:
   ```bash
   cat reports/frozen_buffers_dedup.json
   cat reports/variant_comparison.csv
   ```
5. Agree on the final buffers with the team.
6. Score the proof period one time. Do not run this in CI.
   ```bash
   python -m src.prove
   ```
7. Make a schedule:
   ```bash
   python src/nightly.py --from 2017-08-18 --nights 14
   ```
   The output is `artifacts/nightly_2017-08-18_14n.csv`.

---

## Input Data

**File:** `data/raw/hotels.csv`. 119,390 rows and 32 columns. One row is one booking. The file you gave matches the MD5 hash in `hotels.csv.dvc`.

**Hotels:** `City Hotel` (79,330 rows) and `Resort Hotel` (40,060 rows). Arrivals run from July 2015 to August 2017.

**Target:** `is_canceled` (1 = cancelled).

**Column groups** (from `src/features.py`):

| Group | Columns |
|---|---|
| Known at booking (24 features) | `hotel`, `lead_time`, `arrival_date_month`, `arrival_date_week_number`, `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `meal`, `country`, `market_segment`, `distribution_channel`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`, `reserved_room_type`, `deposit_type`, `agent`, `company`, `days_in_waiting_list`, `customer_type`, `adr` |
| Change after booking (excluded) | `booking_changes`, `required_car_parking_spaces`, `total_of_special_requests`, `assigned_room_type` |
| The outcome (never a feature) | `reservation_status`, `reservation_status_date`, `is_canceled` |
| Bookkeeping | `arrival_date_year`, `arrival_date`, `is_dup`, `row_key` |

The file `docs/DATA_DICTIONARY.md` is empty. The table above is the only column description in the repository.

**Cleaning rules** (`reports/decisions_log.csv`):

| Rule | Rows changed |
|---|---|
| `adr` below 0 set to 0 | 1 |
| `adr` above 2,000 set to 2,000 | 1 |
| `children` missing set to 0 | 4 |
| `agent` missing set to `none` | 16,340 |
| `company` missing set to `none` | 112,593 |
| `country` missing set to `none` | 488 |
| Rows with 0 nights kept | 715 |
| Rows dropped | 0 |

**Repeated rows:** 31,994 rows (26.8%) repeat an earlier row in every column. The code flags them with `is_dup`. These rows cancel at 63.1%. First copies cancel at 27.5%. 92.9% of `Non Refund` rows are repeats. The ZIP does not show whether repeated rows are real separate bookings. The decision record leaves this question open for the client.

---

## Output Artifacts

| Artifact | Created by | Meaning |
|---|---|---|
| `data/interim/clean.parquet` | `validate` | All bookings with `is_dup`, `row_key`, `nights` |
| `data/processed/{dedup,raw}/{train,choose,proof}.parquet` | `split` | Time-based periods |
| `models/{dedup,raw}/{flat_rule,segment_table,logistic,random_forest}.joblib` | `backtest` | Trained methods |
| `reports/frozen_buffers_{dedup,raw}.json` | `freeze_buffer` | One buffer for each method; the headline model |
| `reports/choose_table_*.csv`, `variant_comparison.csv` | `freeze_buffer` | Choose-period results |
| `reports/buffer_sweep_*.png` | `freeze_buffer` | Cost and walks against buffer |
| `reports/choose_diagnostics_*.csv`, `drift_table_*.csv` | `backtest` | AUC, night-level error, drift |
| `reports/split_summary.csv` | `split` | Rows and cancellation rate for each period |
| `reports/brief_checks.csv`, `decisions_log.csv`, `duplicates_by_hotel.csv`, `base_rates.csv`, `dup_facts.csv` | `validate` | Data checks |
| `reports/headline_table.csv`, `poc_method_table_*.csv`, `poc_per_hotel_*.csv`, `poc_bootstrap.csv` | `prove` | Proof-period results |
| `reports/DECISIONS_dedup_entry.md` | `prove` | Generated text about the duplicate decision |
| `reports/rf_forward_chain*.csv` | `rf_calibrate` | Forest calibration test |
| `reports/variant_comparison.csv` | `freeze_buffer` | Buffers on both variants |
| `reports/experiment_comparison.{csv,txt}` | Purpose could not be determined from source code. | The `.csv` file is not UTF-8 text (it looks like UTF-16). The `.txt` file looks like a `dvc exp show` table. |
| `artifacts/nightly_<date>_<N>n.csv` | `nightly` | Rooms to oversell for each hotel and night |
| `docs/PROVE_RUNS.txt` | `prove` | Log of proof runs |
| `logs/hotel_oversell.log` | all stages | Rotating log file (5 MB, 3 backups) |

---

## Training / Inference Process

### Periods

`contract.split_periods` splits by arrival date.

| Period | Arrival dates | Use |
|---|---|---|
| train | before 2016-07-01 | Fit the methods |
| choose | 2016-07-01 to 2016-12-31 | Choose each buffer |
| proof | from 2017-01-01 | Score one time |

### Methods

| # | Method | Description |
|---|---|---|
| 1 | Flat rule | Each hotel's training cancellation rate for every booking |
| 2 | Segment table | Rate for each hotel, market segment and channel, shrunk toward the hotel rate |
| 3 | Logistic | One-hot categories, scaled numbers, `LogisticRegression` |
| 4 | Random forest | Ordinal-coded categories, `RandomForestClassifier` |

All methods output one cancellation probability for each booking.

### Steps

1. Build a nightly table: `book`, `cancelled`, `expected` (sum of probabilities), `spread`, `stay_value`.
2. For each method, try every buffer in the grid on the choose period.
3. Pick the cheapest buffer that keeps walks at or below 10 per 1,000 arrivals.
4. Save the buffers in `reports/frozen_buffers_*.json`.
5. Choose the headline model: the cheapest method that meets the walk limit on the choose period.
6. Score the proof period with the frozen buffers. Compare all methods with the flat rule.
7. Resample nights 2,000 times to estimate how stable the saving is.

### Committed results

These numbers come from the files in `reports/`. They are not new runs.

**Proof period, deduplicated (primary), 486 hotel-nights:**

| Method | Buffer | Cost (EUR) | Walks per 1,000 | Ceiling met | Saving vs flat |
|---|---|---|---|---|---|
| 1 Flat rule | spread 0.25 | 2,017,456 | 5.00 | Yes | 0.0% |
| 2 Segment table | constant 1 | 1,781,573 | 5.51 | Yes | 11.7% |
| 3 Logistic | constant 0 | 1,316,070 | 6.81 | Yes | 34.8% |
| 4 Random forest | constant -2 | 1,368,529 | 13.25 | No | 32.2% |
| Reference: sell nothing extra | - | 4,830,633 | 0.00 | Yes | -139.4% |

**Proof period, raw (sensitivity):** The logistic buffer (constant -1) walks 12.31 per 1,000 and breaks the limit. The random forest walks 15.56 per 1,000 and also breaks the limit. Only the flat rule and the segment table meet the limit on raw data. The segment table costs 7.4% more than the flat rule.

**The recommendation flips between variants.** The primary variant keeps logistic at buffer 0 within the limit. The raw variant does not. `reports/DECISIONS_dedup_entry.md` states this and marks the decision as open.

**Per hotel (primary).** The random forest breaks the limit at the City Hotel (17.32 walks per 1,000). The logistic model meets the limit at each hotel (City 9.56, Resort 2.41).

**Bootstrap (primary).** All three methods beat the flat rule in 100% of resamples. The logistic 5th to 95th percentile saving is EUR 612,862 to EUR 795,330. The code treats nights as independent.

**Drift (primary).** Cancellation rate rises from 22.2% (train) to 29.0% (choose) to 31.9% (proof). Mean lead time rises from 58.0 to 92.3 days between train and choose.

### Forest calibration test (`src/rf_calibrate.py`)

The saved `reports/rf_forward_chain.csv` shows buffer 4 for logistic (cost EUR 1,890,437), buffer 0 for the plain forest (EUR 1,541,789) and buffer 3 for the intercept-calibrated forest (EUR 1,806,768). All three stay at or below 6.5 walks per 1,000 in the worst fold.

### Inference

`src/nightly.py` loads the saved model and the frozen buffer. It builds a nightly table for the requested window. It writes `rooms_to_oversell` for each hotel and night. The script reads bookings from `data/interim/clean.parquet`. It is not a live service.

---

## API Endpoints

The repository has no API. Purpose could not be determined from source code.

---

## Notebook Descriptions

The ZIP contains no `.ipynb` file. Some source files refer to "notebook cells" in their headers (for example `src/validate.py`: "cells 5, 7, 8"). The notebook itself is not in the ZIP. This README does not describe it.

---

## Scripts Description

| Script | Command | Description |
|---|---|---|
| `src/validate.py` | `python -m src.validate` | Stage 1. Checks data, flags repeats, cleans values |
| `src/split.py` | `python -m src.split` | Stage 2. Writes train, choose and proof files |
| `src/backtest.py` | `python -m src.backtest --stage-only` | Stage 3. Trains and saves models |
| `src/backtest.py` | `python src/backtest.py [--prove]` | Trains, freezes buffers, prints the proof table |
| `src/freeze_buffer.py` | `python -m src.freeze_buffer` | Stage 4. Chooses and saves buffers |
| `src/prove.py` | `python -m src.prove` | Stage 5. Scores the proof period |
| `src/rf_calibrate.py` | `python -m src.rf_calibrate` | Tests a calibrated forest |
| `src/nightly.py` | `python src/nightly.py --from DATE --nights N` | Writes a schedule of rooms to oversell |
| `contract.py` | `python contract.py` | Prints the module description |

---

## Dependencies

| Package | Use |
|---|---|
| pandas, numpy, pyarrow | Data handling and parquet files |
| scikit-learn | Methods and metrics |
| joblib | Model files |
| PyYAML | Settings |
| matplotlib | Buffer charts |
| dvc, dvc-s3 | Data and pipeline versioning |
| mlflow, dagshub, python-dotenv | Optional tracking |
| pytest, ruff | Tests and lint |

`requirements.txt` has no version pins. `requirements.lock.txt` pins the full set. CI installs from `requirements.txt` only.

---

## Troubleshooting

| Problem | Action |
|---|---|
| `dvc pull` fails with an access error | Run `aws configure`. Check the bucket name and region in `.dvc/config` |
| `Unexpected raw shape` | The CSV differs from 119,390 rows and 32 columns. Pull the correct file or change `data.expected_*` |
| `contract.py (the client's rule) is needed` | Run from the repository root. Do not move `contract.py` |
| `Model file missing` | Run `python -m src.backtest --stage-only` first |
| `proof ... not found. Freeze the buffers` | Run `dvc repro freeze_buffer` first |
| Log warning: proof period scored before | A proof run already exists in `docs/PROVE_RUNS.txt`. Do not change buffers after a proof result |
| MLflow run does not appear | Set `MLFLOW_TRACKING_URI`, `DAGSHUB_USERNAME` and `DAGSHUB_USER_TOKEN` in `.env`. The proof stage still saves local results if MLflow fails |
| Merge conflict in `dvc.lock` | Merge code first. Run `dvc repro`. Do not edit the file by hand |
| `dvc repro` runs `prove` | The `prove` stage has no `frozen: true` key in `dvc.yaml`. Run `dvc freeze prove`, or use `dvc repro freeze_buffer` |

---

## Future Improvements

These items come from gaps in the code and files. They are not planned work.

1. Write `docs/DATA_DICTIONARY.md` and `docs/MODEL_CARD.md` (both are empty).
2. Add `docs/EVALUATION_CONTRACT.md` and `tests/test_contract.py`. `contract.py` refers to both.
3. Pass `max_depth` and `max_samples` to the forest in `build_methods`, or remove them from `params.yaml`.
4. Add `DAGSHUB_USERNAME` to `.env.example`.
5. Fix the header of `src/backtest.py` (it lists old forest settings) and the header of `src/prove.py` (it names `docs/DECISIONS.md`, but the code writes `reports/DECISIONS_dedup_entry.md`).
6. Add `frozen: true` for `prove` in `dvc.yaml`, or change the comment.
7. Pin package versions in `requirements.txt`.
8. Adapt `nightly.py` for live use. At present it needs the real outcome column.
9. Add a license file.
10. Get the client answer on whether repeated rows are separate bookings.

---

## License

**Code:** The ZIP contains no license file. License: not stated. Add a `LICENSE` file before public release.

**Data:** The Build 2 deck states that the bookings are the Hotel Booking Demand datasets by Antonio, de Almeida and Nunes (2019), licensed CC BY 4.0. Keep this credit if you publish the data or results. Check the license terms at the original source.
