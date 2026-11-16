# **Hotel Oversell Decision System (hotel-oversell-main)** 

## **Overview** 

The Hotel Oversell Decision System calculates the number of rooms ( ) that a hotel can sell above physical capacity for each arrival night. 

### **Business Problem** 

Hotels lose revenue when guests cancel bookings and rooms stay empty. If a hotel sells too many extra rooms, guests arrive at a full hotel. The hotel must send ("walk") these excess guests to another hotel. Walking a guest costs per guest. An empty room costs the average booked stay value ( ) for that arrival night. 

This software solves the revenue optimization problem for two properties in Portugal: 

1. **Resort Hotel ( )** : Located in the Algarve region ( bookings). 

2. **City Hotel ( )** : Located in Lisbon ( bookings). 

For each arrival night and hotel, the system predicts expected cancellations and subtracts a safety buffer: 



The system enforces a strict service guardrail: the hotel must not walk more than guests per arrivals ( ). Any policy that exceeds this limit fails validation, regardless of euro savings. 

## **Features** 

- **Data Contract Validation** : Enforces strict column timing rules (contract.py, src/validate.py) to stop target leakage from post-booking variables (reservation_status, reservation_status_date, assigned_room_type). 

- **Temporal Data Splitting** : Separates bookings by arrival date into three chronological periods (src/split.py): 

   - **Train** : Prior to 2016-07-01 (before July 2016). 

   - **Choose** : 2016-07-01 to 2016-12-31 (July to December 2016). 

   - **Prove** : 2017-01-01 to 2017-08-31 (January to August 2017). 

- **Dual-Track Deduplication** : Evaluates pipelines on both raw data (raw) and deduplicated data (dedup) to measure the impact of duplicate PMS records. 

- **Multi-Method Benchmarking** : Evaluates baseline rules, lookup tables, Logistic Regression, LightGBM, and XGBoost models (src/backtest.py, src/tune.py). 

- **Asymmetric Euro Cost Scoring** : Scores predictions in euros using empty-room opportunity cost and the walk penalty (src/costs.py). 

● **Guardrail Buffer Freezing** : Sweeps safety buffers on the Choose split, selects the minimum-cost buffer that satisfies , and freezes the value in JSON format (src/freeze_buffer.py). 

- **One-Command Nightly CLI** : Generates a 14-night oversell schedule from booking-time features (src/nightly.py). 

- **Reproducible MLOps Pipeline** : Tracks data, parameters, models, and reports with DVC (dvc.yaml, dvc.lock, params.yaml) and GitHub Actions CI (.github/workflows/ci.yaml). 

## **Architecture** 

### **Main Components** 

1. **Contract Layer (contract.py)** : Defines the schema, feature availability timing, temporal split boundaries, and financial constants. 

2. **Data & Validation Layer (src/data.py, src/validate.py, src/split.py)** : Loads raw CSV records, verifies data contracts, logs data cleaning decisions, and partitions datasets by arrival date. 

3. **Feature Engineering Layer (src/features.py)** : Transforms booking-time attributes into numerical and categorical features without future data leakage. 

4. **Modeling & Cost Engine (src/costs.py, src/tune.py, src/backtest.py)** : Trains cancellation probability estimators, aggregates booking probabilities into nightly expected cancellations, and computes euro costs and walk rates. 

5. **Buffer Selection & Proof Layer (src/freeze_buffer.py, src/prove.py)** : Sweeps candidate buffers on the Choose split, freezes the optimal buffer, and executes a single evaluation on the Prove split. 

6. **Production CLI Service (src/nightly.py)** : Loads frozen buffers and trained models to 

output nightly oversell recommendations ( ) for upcoming arrival windows. 

### **Data Flow** 

[data/raw/hotels.csv] │ 

▼ 

[src/validate.py & src/data.py] ──► (Validates schema, logs duplicates to reports/dup_facts.csv) │ 

▼ 

[src/split.py] ───────────────────► (Splits by arrival date: Train < Jul 2016 | Choose H2 2016 | Prove 2017) │ ▼ [src/features.py] ────────────────► (Extracts booking-time features; blocks post-booking leakage) │ ├──► [src/tune.py & src/backtest.py] ──► (Trains models on Train; evaluates methods on Choose) │ ▼ [src/freeze_buffer.py] ───────────► (Sweeps buffers on Choose; writes reports/frozen_buffers_*.json) │ ├──► [src/prove.py] ───────────────────► (Runs single evaluation on Prove split; writes docs/PROVE_RUNS.txt) │ ▼ [src/nightly.py] ─────────────────► (Generates 14-night oversell table for hotel revenue managers) 

### **Services and Modules** 

This repository operates as a batch CLI application and DVC pipeline. It does not expose an HTTP server or web dashboard. All operations run through modular Python CLI entry points in src/. 

## **Repository Structure** 

### **Repository Tree** 

hotel-oversell-main/ ├── .dvc/ │ ├── .gitignore │ └── config ├── .github/ │ └── workflows/ │ └── ci.yaml ├── artifacts/ │ └── .gitkeep ├── data/ │ ├── interim/ 

│ │ └── .gitkeep │ ├── processed/ │ │ └── .gitkeep │ └── raw/ │ ├── .gitkeep │ └── hotels.csv.dvc ├── docs/ │ ├── DATA_DICTIONARY.md │ ├── MODEL_CARD.md │ ├── PROVE_RUNS.txt │ └── RUNBOOK.md ├── logs/ │ └── .gitkeep ├── models/ │ ├── .gitignore │ └── .gitkeep ├── reports/ │ ├── .gitkeep │ ├── base_rates.csv │ ├── brief_checks.csv │ ├── buffer_sweep_dedup.png │ ├── buffer_sweep_raw.png │ ├── choose_diagnostics_dedup.csv │ ├── choose_diagnostics_raw.csv │ ├── choose_table_dedup.csv │ ├── choose_table_raw.csv │ ├── decisions_log.csv │ ├── drift_table_dedup.csv │ ├── drift_table_raw.csv │ ├── dup_facts.csv │ ├── duplicates_by_hotel.csv │ ├── frozen_buffers_dedup.json │ ├── frozen_buffers_raw.json │ ├── split_summary.csv │ └── variant_comparison.csv ├── src/ │ ├── __init__.py │ ├── backtest.py │ ├── costs.py │ ├── data.py │ ├── features.py │ ├── freeze_buffer.py │ ├── logger.py 

│ ├── nightly.py │ ├── params.py │ ├── prove.py │ ├── split.py │ ├── tracking.py │ ├── tune.py │ └── validate.py ├── tests/ │ ├── __init__.py │ ├── conftest.py │ ├── test_costs.py │ ├── test_features.py │ ├── test_pipeline.py │ ├── test_split.py │ └── test_validate.py ├── .dvcignore ├── .env.example ├── .gitignore ├── README.md ├── contract.py ├── dvc.lock ├── dvc.yaml ├── params.yaml ├── requirements.lock.txt └── requirements.txt 

### **Detailed File-by-File Explanation** 

#### **Root Directory** 

- **contract.py** 

   - **Purpose** : Defines the immutable data schema, column timing classifications (booking-time vs. post-booking leakage), split date boundaries, and business cost constants ( walk cost, walks per arrivals limit). 

   - **Key Functions or Classes** : Schema constants, BOOKING_TIME_COLS, LEAKAGE_COLS, TARGET_COL, cost/guardrail constants. 

   - **Inputs** : None. 

   - **Outputs** : Imported constants across src/ and tests/. 

   - **Dependencies** : Python standard library (typing, dataclasses). 

- **params.yaml** 

   - **Purpose** : Stores central configuration parameters for data paths, temporal split dates, 

model hyperparameters, buffer sweep ranges, and cost thresholds. 

   - **Key Functions or Classes** : YAML configuration blocks (data, split, costs, models, buffer_sweep). 

   - **Inputs** : Read by src/params.py and DVC stages. 

   - **Outputs** : Runtime configuration dictionary. 

   - **Dependencies** : None. 

- **dvc.yaml & dvc.lock** 

   - **Purpose** : Defines the Data Version Control (DVC) pipeline stages, dependencies, commands, and locked MD5 hashes of data and report artifacts. 

   - **Key Functions or Classes** : Pipeline stage definitions (validate, split, backtest, freeze_buffer, prove). 

   - **Inputs** : data/raw/hotels.csv, params.yaml, src/*.py. 

   - **Outputs** : Tracked artifacts in data/, models/, and reports/. 

   - **Dependencies** : dvc. 

- **requirements.txt & requirements.lock.txt** 

   - **Purpose** : Lists direct Python package dependencies (requirements.txt) and exact pinned environment versions (requirements.lock.txt). 

   - **Key Functions or Classes** : Package specifications. 

   - **Inputs** : Used by pip install. 

   - **Outputs** : Configured Python virtual environment. 

   - **Dependencies** : pip. 

#### **/src Directory** 

- **src/__init__.py** 

   - **Purpose** : Marks src as a Python package. 

- **src/params.py** 

   - **Purpose** : Loads and validates configuration settings from params.yaml. 

   - **Key Functions or Classes** : load_params(). 

   - **Inputs** : params.yaml. 

   - **Outputs** : Python dictionary / configuration object. 

   - **Dependencies** : pyyaml, pathlib. 

- **src/logger.py** 

   - **Purpose** : Configures standardized console and file logging across all pipeline modules. 

   - **Key Functions or Classes** : get_logger(). 

   - **Inputs** : Module name, log level, logs/ directory path. 

   - **Outputs** : Formatted log streams and log files in logs/. 

   - **Dependencies** : logging, sys, pathlib. 

- **src/tracking.py** 

   - **Purpose** : Logs experiment metrics, parameters, decisions, and run metadata to local CSV/JSON artifacts. 

   - **Key Functions or Classes** : Experiment and decision logging helper functions. 

   - **Inputs** : Model metrics, parameter dictionaries, data cleaning counts. 

   - **Outputs** : reports/decisions_log.csv and run tracking metadata. 

   - **Dependencies** : json, csv, pandas, src.logger. 

- **src/data.py** 

   - **Purpose** : Reads raw hotel booking records (hotels.csv), constructs arrival dates from year/month/day columns, handles NULL string literals, and executes optional deduplication (raw vs dedup variants). 

   - **Key Functions or Classes** : Data loading, date parsing, deduplication routines. 

   - **Inputs** : data/raw/hotels.csv, params.yaml. 

   - **Outputs** : Cleaned pandas DataFrame, reports/dup_facts.csv, reports/duplicates_by_hotel.csv, reports/brief_checks.csv. 

   - **Dependencies** : pandas, numpy, contract, src.logger. 

- **src/validate.py** 

   - **Purpose** : Enforces the data contract on input DataFrames. Verifies required columns, checks value ranges, blocks leakage columns, and validates brief statistics against actual dataset counts. 

   - **Key Functions or Classes** : Schema validation functions, leakage assertions, brief verification checks. 

   - **Inputs** : Raw or processed DataFrame, contract.py. 

   - **Outputs** : Validated DataFrame, validation reports (reports/brief_checks.csv, reports/decisions_log.csv). 

   - **Dependencies** : pandas, numpy, contract, src.logger. 

- **src/split.py** 

   - **Purpose** : Splits the dataset chronologically by arrival date into Train ( 





      - ), Choose ( to ), and Prove ( 

      - ). 

   - **Key Functions or Classes** : Temporal split functions and split summary generators. 

   - **Inputs** : Validated DataFrame, split date thresholds from params.yaml / contract.py. 

   - **Outputs** : Partitioned DataFrames, reports/split_summary.csv, reports/base_rates.csv, reports/drift_table_raw.csv, reports/drift_table_dedup.csv. 

   - **Dependencies** : pandas, contract, src.params, src.logger. 

- **src/features.py** 

   - **Purpose** : Builds feature matrices using only columns known at the time of booking. Strips post-booking columns (assigned_room_type, reservation_status, reservation_status_date, booking_changes, days_in_waiting_list if flagged) to prevent target leakage. 

   - **Key Functions or Classes** : Feature preprocessing pipeline, categorical encoders, booking-time feature selectors. 

   - **Inputs** : Split DataFrames (Train, Choose, Prove). 

   - **Outputs** : Feature matrix and target vector (is_canceled). 

   - **Dependencies** : pandas, numpy, scikit-learn, contract. 

- **src/costs.py** 

   - **Purpose** : Calculates nightly oversell quantities ( ), empty rooms, walked guests, walk rate per arrivals, and total euro costs. 

   - **Key Functions or Classes** : Nightly aggregation functions, euro cost calculator, guardrail check ( walks per arrivals). 

   - **Inputs** : Nightly actual cancellations, predicted cancellations, buffer value, nightly average stay value ( ), walk penalty ( ). 

   - **Outputs** : Nightly and aggregate cost metrics, walk rates, guardrail pass/fail booleans. 

   - **Dependencies** : numpy, pandas, contract. 

- **src/tune.py** 

   - **Purpose** : Trains and tunes cancellation prediction models on the Train split using time-aware cross-validation and hyperparameter search. 

   - **Key Functions or Classes** : Model training and hyperparameter tuning routines. 

   - **Inputs** : Train feature matrix and target vector, params.yaml. 

   - **Outputs** : Serialized model artifacts in models/. 

   - **Dependencies** : scikit-learn, lightgbm, xgboost, joblib, src.features, src.params. 

- **src/backtest.py** 

   - **Purpose** : Evaluates all candidate forecasting methods (flat baseline rule, empirical lookup table, Logistic Regression, LightGBM, XGBoost) on the Choose split across both raw and dedup tracks. 

   - **Key Functions or Classes** : Backtest runner, method comparison aggregator. 

   - **Inputs** : Train and Choose splits, params.yaml. 

   - **Outputs** : reports/choose_table_raw.csv, reports/choose_table_dedup.csv, reports/choose_diagnostics_raw.csv, reports/choose_diagnostics_dedup.csv, reports/variant_comparison.csv, trained models in models/. 

   - **Dependencies** : pandas, numpy, scikit-learn, lightgbm, src.costs, src.features, src.split. 

- **src/freeze_buffer.py** 

   - **Purpose** : Sweeps candidate safety buffer values on the Choose split, plots cost vs. walk-rate curves, selects the optimal buffer per hotel and method that obeys the walk limit ( per ), and freezes the selected buffers to JSON. 

   - **Key Functions or Classes** : Buffer sweep engine, plot generator, JSON serializer. 

   - **Inputs** : Choose predictions from src/backtest.py, params.yaml. 

   - **Outputs** : reports/frozen_buffers_raw.json, reports/frozen_buffers_dedup.json, reports/buffer_sweep_raw.png, reports/buffer_sweep_dedup.png. 

   - **Dependencies** : pandas, numpy, matplotlib, json, src.costs. 

- **src/prove.py** 

   - **Purpose** : Executes the one-time out-of-sample evaluation on the 2017 Prove split using the frozen models and frozen buffers from the Choose period. 

   - **Key Functions or Classes** : Proof evaluation runner, proof audit logger. 

   - **Inputs** : Prove split data, models/, reports/frozen_buffers_*.json. 

   - **Outputs** : Proof performance tables and docs/PROVE_RUNS.txt. 

   - **Dependencies** : pandas, numpy, json, joblib, src.costs, src.features. 

- **src/nightly.py** 

   - **Purpose** : Production CLI tool for hotel revenue managers. Outputs a 14-night table of recommended oversell room counts ( ) per hotel starting from a specified arrival date. 

   - **Key Functions or Classes** : CLI argument parser (--from, --nights), nightly inference and aggregation runner. 

   - **Inputs** : Booking records known prior to the forecast window, trained models (models/), frozen buffers (reports/frozen_buffers_*.json). 

   - **Outputs** : Formatted 14-night oversell recommendation table (console output and/or CSV artifact). 

   - **Dependencies** : argparse, pandas, joblib, json, src.features, src.costs. 

#### **/tests Directory** 

- **tests/conftest.py** : Provides shared pytest fixtures, synthetic booking DataFrames, and temporary test directories. 

- **tests/test_validate.py** : Tests schema validation, NULL handling, and post-booking leakage rejection. 

- **tests/test_split.py** : Verifies chronological split boundaries and asserts zero temporal overlap between Train, Choose, and Prove. 

- **tests/test_features.py** : Verifies that feature engineering excludes forbidden post-booking columns (assigned_room_type, reservation_status, reservation_status_date). 

- **tests/test_costs.py** : Verifies euro cost calculations, empty room valuations, walk penalties, and the ceiling. 



- **tests/test_pipeline.py** : Runs end-to-end integration tests across validation, splitting, backtesting, buffer freezing, and nightly inference. 

#### **/docs Directory** 

- **docs/DATA_DICTIONARY.md** : Documents all 32 dataset columns, data types, PMS source tables (BO, BL, TR, CP, ML, DC, NT, MS), and booking-time availability status. 

- **docs/MODEL_CARD.md** : Documents model architecture, intended use, training/choose/prove datasets, evaluation metrics, fairness/leakage constraints, and operational caveats. 

- **docs/PROVE_RUNS.txt** : Immutable audit log of executions on the 2017 Prove dataset. 

- **docs/RUNBOOK.md** : Step-by-step operational guide for running the pipeline, executing nightly forecasts, and troubleshooting errors. 

### **Important Files Summary Table** 

|**File Name**|**Purpose**|**Key Components**|**Dependencies**|
|---|---|---|---|



|contract.py|Defines data<br>contract, leakage<br>rules, and cost<br>constants|BOOKING_TIME_C<br>OLS,<br>LEAKAGE_COLS,<br>walk cost/ceiling<br>constants|Python stdlib|
|---|---|---|---|
|params.yaml|Central<br>configuration file|Split dates, model<br>hyperparameters,<br>cost & buffer sweep<br>settings|pyyaml, dvc|
|dvc.yaml|Defines<br>reproducible<br>MLOps pipeline<br>stages|Stages: validate,<br>split, backtest,<br>freeze_buffer,<br>prove|dvc|
|src/data.py|Loads raw CSV data<br>and handles<br>deduplication|Raw loader, arrival<br>date builder,<br>deduplication<br>tracker|pandas, numpy,<br>contract|
|src/validate.py|Validates schema<br>and checks brief<br>claims|Contract validator,<br>leakage blocker,<br>brief facts verifier|pandas, contract,<br>src.logger|
|src/split.py|Performs<br>chronological<br>Train/Choose/Prove<br>splitting|Date-based splitter,<br>drift and base-rate<br>calculators|pandas, contract,<br>src.params|
|src/features.py|Extracts<br>booking-time<br>features without<br>leakage|Feature<br>transformer,<br>leakage column<br>filter, encoders|pandas, scikit-learn,<br>contract|
|src/costs.py|Computes euro loss<br>and walk-rate<br>guardrails|Nightly<br>scorer,<br>walk rate calculator,<br>guardrail filter|numpy, pandas,<br>contract|
|src/tune.py|Tunes ML models<br>on the Train split|Time-series<br>cross-validation<br>and|scikit-learn,<br>lightgbm, xgboost|



|||hyperparameter<br>tuner||
|---|---|---|---|
|src/backtest.py|Benchmarks<br>candidate methods<br>on the Choose split|Baseline rules,<br>lookup tables, ML<br>classifiers,<br>evaluator|pandas, scikit-learn,<br>lightgbm, src.costs|
|src/freeze_buffer.py|Sweeps and freezes<br>safety buffers on<br>Choose split|Buffer grid sweep,<br>guardrail<br>enforcement, plot &<br>JSON writer|pandas, matplotlib,<br>json, src.costs|
|src/prove.py|Evaluates frozen<br>models and buffers<br>on Prove split|Single-shot proof<br>evaluator,<br>PROVE_RUNS.txt<br>logger|pandas, joblib, json,<br>src.costs|
|src/nightly.py|CLI entry point for<br>14-night oversell<br>schedule|--from and --nights<br>CLI parser, nightly<br>generator|argparse, pandas,<br>joblib, src.costs|
|src/tracking.py|Logs cleaning<br>decisions and<br>experiment runs|Decision log writer,<br>metric tracker|pandas, csv, json|
|src/logger.py|Standardized<br>logging utility|get_logger()|logging, pathlib|
|src/params.py|YAML parameter<br>loader|load_params()|pyyaml, pathlib|
|tests/test_costs.py|Unit tests for euro<br>cost and walk<br>ceiling formulas|Cost formula<br>assertions,<br>guardrail boundary<br>tests|pytest, src.costs|
|tests/test_features.<br>py|Unit tests for<br>leakage prevention|Column audit and<br>leakage exclusion<br>tests|pytest, src.features,<br>contract|
|tests/test_pipeline.<br>py|End-to-end<br>integration test|Full pipeline<br>execution test on|pytest, src.*|



suite 

synthetic fixtures 

## **Prerequisites** 

Install the following software before you set up the project: 

- **Operating System** : Linux, macOS, or Windows (WSL2 recommended). 

- **Python** : Version 3.10 or 3.11. 

- **Git** : Version 2.30 or higher. 

- **DVC (Data Version Control)** : Version 3.x (optional, for data and pipeline reproduction). 

## **Installation** 

Follow these steps to install the project and its dependencies: 

1. Clone the repository to your local machine: git clone <repository-url> cd hotel-oversell-main 

2. Create and activate a Python virtual environment: python3 -m venv .venv source .venv/bin/activate 

3. Upgrade pip and install the locked dependencies: pip install --upgrade pip pip install -r requirements.lock.txt 

_(Note: If your platform does not match the locked OS wheels, run pip install -r requirements.txt.)_ 

4. Place the raw dataset (hotels.csv) in the data/raw/ directory, or pull it with DVC: cp /path/to/hotels.csv data/raw/hotels.csv # Or, if a DVC remote is configured: dvc pull 

## **Configuration** 

1. Copy the example environment file: cp .env.example .env 

2. Review params.yaml to verify or adjust pipeline parameters: 

   - **Data paths** : Input (data/raw/hotels.csv) and output report directories (reports/, models/). 

   - **Temporal split dates** : 

      - train_end: 2016-06-30 

      - choose_start: 2016-07-01 

      - choose_end: 2016-12-31 

      - prove_start: 2017-01-01 

   - **Business rules** : 

      - walk_cost_eur: 300.0 

      - max_walks_per_1000: 10.0 

   - **Buffer sweep grid** : Candidate safety buffer values tested on the Choose split. 

## **Usage** 

Run all commands from the root directory (hotel-oversell-main/). 

### **1. Validate Data and Check Brief Claims** 

python -m src.validate 

### **2. Split the Dataset and Generate Drift Diagnostics** 

python -m src.split 

### **3. Run the Backtest on the Choose Period (POC)** 

python src/backtest.py 

### **4. Sweep and Freeze the Safety Buffer** 

python -m src.freeze_buffer 

### **5. Run the One-Time Proof Evaluation** 

python -m src.prove 

### **6. Generate the 14-Night Oversell Schedule (MVP)** 

python src/nightly.py --from 2017-08-18 --nights 14 

### **7. Run the Automated Test Suite** 

pytest -v 

## **Example Workflow** 

To execute the complete end-to-end MLOps workflow from raw data validation to nightly production inference: 

# Step 1: Run unit and contract tests pytest 

# Step 2: Reproduce the full DVC pipeline (validate -> split -> backtest -> freeze_buffer -> prove) 

dvc repro 

# Step 3: Inspect the method comparison and frozen buffers cat reports/variant_comparison.csv cat reports/frozen_buffers_raw.json cat reports/frozen_buffers_dedup.json 

# Step 4: Run the production CLI for a 14-night arrival window python src/nightly.py --from 2017-08-18 --nights 14 

## **Input Data** 

### **Dataset Overview** 

The project uses the open **Hotel Booking Demand Dataset** published by Nuno Antonio, Ana de Almeida, and Luis Nunes ( _Data in Brief_ , 22, 2019, pp. 41–49, CC BY 4.0). 

- **File Location** : data/raw/hotels.csv (tracked via data/raw/hotels.csv.dvc). 

- **Total Records** : booking rows across two hotels in Portugal: 

   - **Resort Hotel ( , Algarve)** : rows. 

   - **City Hotel ( , Lisbon)** : rows. 

- **Arrival Date Range** : 2015-07-01 to 2017-08-31. 

- **Columns** : columns ( original PMS variables plus the hotel identifier column). 

### **Column Timing and Leakage Audit** 

The PMS extracted variables from eight relational tables (BO Bookings, BL Bookings Change Log, TR Transactions, CP Customer Profiles, ML Meals, DC Distribution Channels, NT Nationalities, MS Market Segments). To prevent future information leakage, contract.py and src/features.py classify columns by when the hotel learns their value: 

1. **Target Column** : 

   - is_canceled ( if canceled, if arrived). 

2. **Forbidden Post-Booking / Leakage Columns (Excluded from Features)** : 

   - reservation_status: Contains final outcome (Canceled, Check-Out, No-Show). Directly leaks is_canceled. 

   - reservation_status_date: Timestamp when the final status was recorded. 

   - assigned_room_type: Assigned at check-in or during operations; differs from reserved_room_type much more frequently for non-canceled bookings than canceled bookings. 

3. **Caution / Operational Drift Columns (Audited in src/validate.py & src/features.py)** : 

   - booking_changes, days_in_waiting_list, country: The research paper notes that guests frequently modify booking attributes or provide nationality at check-in. In addition, exact duplicate rows exist in the raw extract (primarily from group/OTA batch 

   - entries), which justifies the dual raw vs. dedup evaluation tracks. 

4. **Booking-Time Columns (Allowed for src/nightly.py)** : 

   - hotel, lead_time, arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal, market_segment, distribution_channel, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, reserved_room_type, deposit_type, agent, company, customer_type, adr, required_car_parking_spaces, total_of_special_requests. 

## **Output Artifacts** 

The pipeline writes structured outputs to reports/, models/, and docs/: 

- **Data Audit & Split Reports** : 

   - reports/brief_checks.csv: Verification of numbers stated in the client brief against raw data. 

   - reports/decisions_log.csv: Audit log of every data cleaning and filtering rule, with row counts before and after. 

   - reports/dup_facts.csv & reports/duplicates_by_hotel.csv: Quantification of duplicate booking rows across Resort Hotel and City Hotel. 

   - reports/split_summary.csv: Booking counts, arrival night counts, and cancellation rates per split (Train, Choose, Prove). 

   - reports/base_rates.csv: Historical cancellation base rates by hotel, channel, and deposit type. 

   - reports/drift_table_raw.csv & reports/drift_table_dedup.csv: Feature and target distribution shift across temporal splits. 

- **Model & Buffer Selection Reports** : 

   - reports/choose_table_raw.csv & reports/choose_table_dedup.csv: Performance table of all candidate methods on the Choose period (euro cost, empty rooms, walks per arrivals). 

   - reports/choose_diagnostics_raw.csv & reports/choose_diagnostics_dedup.csv: Detailed nightly diagnostics on the Choose split. 

   - reports/buffer_sweep_raw.png & reports/buffer_sweep_dedup.png: Visual trade-off curves of euro cost vs. walked guests across buffer values. 

   - reports/frozen_buffers_raw.json & reports/frozen_buffers_dedup.json: Locked buffer parameters selected on Choose for each hotel and method. 

   - reports/variant_comparison.csv: Direct comparison between raw and dedup pipelines. 

- **Proof & Model Artifacts** : 

   - models/: Trained model binaries (raw and dedup subdirectories). 

   - docs/PROVE_RUNS.txt: Single-run proof metrics on the 2017 Prove split. 

## **Training / Inference Process** 

### **1. Problem Formulation** 

Instead of optimizing classification accuracy or ROC-AUC in isolation, the pipeline optimizes the nightly operational decision: 



where: 

- is the number of rooms to oversell at hotel on arrival night . 

- is the expected number of cancellations on night  (sum of 

- predicted cancellation probabilities across all bookings due to arrive on night ). 

- . 

- is the frozen safety buffer for hotel 

### **2. Nightly Scoring Rule (Euro Cost & Guardrail)** 

On arrival night  with actual cancellations and actual arrivals : 

● **Unfilled Cancellations (Empty Rooms)** : 



Each empty room costs the night's average booked stay value: 



- **Walked Guests** : 



Each walked guest costs . 

● **Guardrail** : Across the evaluation period, total walks per actual arrivals must obey: 



### **3. Three-Stage Temporal Discipline** 

1. **Train (< 2016-07-01)** : Fit baseline rates, empirical lookup tables, Logistic Regression, LightGBM, and XGBoost classifiers. 

2. **Choose (2016-07-01 to 2016-12-31)** : Predict for every night in the second half of 2016. Sweep buffer values (src/freeze_buffer.py). Filter out any pair that exceeds walks per arrivals. Select the method and buffer that minimize total euro cost, and save to reports/frozen_buffers_*.json. 

3. **Prove (2017-01-01 to 2017-08-31)** : Apply the frozen model and frozen buffer once to the 2017 arrival nights (src/prove.py) to report honest out-of-sample euro savings and walk rates. 

## **API Endpoints** 

Not applicable. This repository provides a command-line interface (src/nightly.py) and batch pipeline scripts. In accordance with the project brief ("What to Drop: Building a dashboard or a web app, when the MVP is one command"), no REST API or web server is included. 

## **Notebook Descriptions** 

No .ipynb Jupyter Notebook files exist in the repository (hotel-oversell-main). All exploratory data checks, brief verifications, duplicate analyses, and model comparisons are implemented as deterministic, testable Python scripts in src/ (src/validate.py, src/split.py, src/backtest.py) that output auditable CSV tables to reports/. 

## **Scripts Description** 

- **python -m src.validate** : Executes contract checks on data/raw/hotels.csv, verifies claims from the client brief (reports/brief_checks.csv), and logs cleaning decisions (reports/decisions_log.csv). 

- **python -m src.split** : Splits data into Train, Choose, and Prove sets by arrival date and outputs reports/split_summary.csv, reports/base_rates.csv, and drift tables. 

- **python src/backtest.py** : Trains and evaluates all candidate methods on the Choose period and generates reports/choose_table_*.csv and reports/variant_comparison.csv. 

- **python -m src.freeze_buffer** : Runs the safety buffer sweep on the Choose split, generates reports/buffer_sweep_*.png, and writes reports/frozen_buffers_*.json. 

- **python -m src.prove** : Runs the locked evaluation on the 2017 Prove split and records results in docs/PROVE_RUNS.txt. 

- **python src/nightly.py --from YYYY-MM-DD --nights N** : Runs the production MVP 

command to output nightly oversell counts ( ) for each hotel for the next nights. 

- **python -m src.tune** : Runs hyperparameter tuning on the Train split. 

## **Dependencies** 

Key packages defined in requirements.txt and requirements.lock.txt: 

|**Package**|**Role in Project**|
|---|---|
|pandas|DataFrame manipulation, temporal splitting,<br>and nightly aggregation|
|numpy|Vectorized array operations and cost/buffer<br>math|
|scikit-learn|Preprocessing pipelines, Logistic<br>Regression, and evaluation metrics|
|lightgbm|Gradient boosted decision trees for<br>cancellation probability estimation|
|xgboost|Extreme gradient boosting classifier for<br>benchmark comparison|



|matplotlib|Generation of buffer sweep trade-off plots<br>(buffer_sweep_*.png)|
|---|---|
|pyyaml|Parsing params.yaml configuration files|
|joblib|Saving and loading trained model artifacts<br>in models/|
|dvc|Data and pipeline version control (dvc.yaml,<br>dvc.lock)|
|pytest|Unit and integration testing framework<br>(tests/)|
|ruff / flake8|Code linting in CI workflow<br>(.github/workflows/ci.yaml)|



## **Troubleshooting** 

- **Error: FileNotFoundError: data/raw/hotels.csv** 

   - **Cause** : The raw dataset file is tracked by DVC (hotels.csv.dvc) and is not stored directly in Git. 

   - **Solution** : Copy hotels.csv into data/raw/hotels.csv or run dvc pull. 

- **Error: ContractViolationError or Leakage Assertion Failure in src/validate.py / src/features.py** 

   - **Cause** : Input data contains forbidden post-booking columns (reservation_status, reservation_status_date, assigned_room_type) in the feature list, or required columns are missing. 

   - **Solution** : Inspect contract.py to ensure custom features only reference BOOKING_TIME_COLS. 

- **Error: No valid buffer satisfies walk ceiling in src/freeze_buffer.py** 

   - **Cause** : The candidate buffer range in params.yaml is too narrow to bring walked guests below per arrivals on the Choose split. 

   - **Solution** : Increase the upper bound of the buffer sweep grid in params.yaml and re-run python -m src.freeze_buffer. 

- **Error: FileNotFoundError: reports/frozen_buffers_*.json when running src/nightly.py or src/prove.py** 

   - **Cause** : Buffers have not been frozen yet. 

   - **Solution** : Run python src/backtest.py followed by python -m src.freeze_buffer before running prove.py or nightly.py. 

## **Future Improvements** 

1. **Full-Night Capacity & Dynamic Walk Cost Caveat** : As noted in the project problem statement and docs/MODEL_CARD.md, the dataset records booking-level arrivals and cancellations but does not provide exact daily physical room inventory or multi-night stay-over occupancy limits. Integrating live PMS inventory counts will allow the buffer to 

   - condition on whether the hotel is near physical occupancy. 

2. **Lead-Time Horizon Conditioning** : In src/nightly.py, bookings for an arrival date 

   - continue to arrive and cancel between and . Filtering and re-calibrating cancellation probabilities by exact days-before-arrival ( ) using Bookings Change Log (BL) timestamps will improve multi-day horizon accuracy. 

3. **Uncertainty Quantile Buffers** : Replace static additive hotel buffers ( ) with dynamic Poisson-binomial quantile thresholds computed directly from the nightly variance 



4. **Automated Drift Retraining Trigger** : Connect reports/drift_table_*.csv metrics to an automated retraining alert when cancellation base rates shift across seasons (e.g., Algarve summer peak vs. winter low season). 

## **License** 

- **Dataset License** : The Hotel Booking Demand dataset (Antonio, de Almeida, and Nunes, 2019) is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)** . 

- **Source Code License** : Check the repository root for license terms. If no LICENSE file is present, standard copyright laws apply; contact the repository authors for commercial or open-source distribution permissions. 

# **Comprehensive Project Assessment & MLOps Analysis Report** 

## **1. Alignment Across Project Artifacts** 

This assessment cross-references all four supplied artifacts: 

1. **GitHub Repository (hotel-oversell-main.zip)** 

2. **Problem Statement Presentation (C2_W06_D02_Build2_Announcement.pptx)** 

3. **Research Paper (reasearch paperr.pdf — Antonio et al.,** **_Data in Brief_ 22, 2019, 41–49)** 

4. **Raw Dataset (hotels.csv — rows, columns)** 

### **Six Traps Audit (From Slide 12 of Problem Statement PPTX)** 

|**Trap in Brief**|**How the Repository**<br>**Addresses It**|**Evidence in Code &**<br>**Reports**|
|---|---|---|
|**Trap 1**: Reporting accuracy<br>when the client pays in<br>euros.|Evaluates all methods in<br>euros (<br>vs.<br>walk<br>penalty) under the<br>ceiling.|src/costs.py,<br>reports/choose_table_raw.c<br>sv,<br>reports/choose_table_dedu<br>p.csv|
|**Trap 2**: Splitting at random<br>when the client proves on<br>2017.|Enforces strict<br>chronological splitting by<br>arrival date (Train<br>, Choose<br>, Prove<br>).|src/split.py,<br>tests/test_split.py,<br>reports/split_summary.csv|
|**Trap 3**: Using columns<br>written after the booking<br>was made.|Explicitly blocks<br>reservation_status,<br>reservation_status_date,<br>and assigned_room_type in<br>the data contract and<br>feature pipeline.|contract.py, src/features.py,<br>tests/test_features.py,<br>docs/DATA_DICTIONARY.m<br>d|
|**Trap 4**: Picking the buffer<br>while looking at the proof<br>period.|Selects and freezes the<br>buffer strictly on the<br>Choose split (H2 2016),<br>writes it to<br>frozen_buffers_*.json, and<br>evaluates Prove (2017) only<br>once.|src/freeze_buffer.py,<br>src/prove.py,<br>reports/frozen_buffers_*.jso<br>n, docs/PROVE_RUNS.txt|
|**Trap 5**: Counting savings<br>against selling nothing<br>extra.|Benchmarks ML models<br>against the client's flat<br>baseline rule and empirical<br>lookup tables rather than<br>zero-oversell alone.|src/backtest.py,<br>reports/choose_table_*.csv|



|**Trap 6**: Trusting a number|Audits every number in the|src/validate.py, src/data.py,|
|---|---|---|
|because the client wrote it.|brief against the raw|reports/brief_checks.csv,|
||dataset and discovers|reports/dup_facts.csv,|
||exact duplicate|reports/variant_comparison|
||rows, running parallel raw|.csv|
||and dedup pipelines.||



## **2. Notebook Deep Analysis** 

- **Finding** : The repository contains **zero** .ipynb Jupyter Notebook files. 

- **Architectural Assessment** : This is a deliberate software engineering decision aligned with Slide 7 and Slide 13 of the Problem Statement (C2_W06_D02_Build2_Announcement.pptx), which specify reproducible CLI scripts (python src/backtest.py and python src/nightly.py) and automated verification over ad-hoc notebooks. All exploratory analyses (base rates, brief checks, duplicate audits, and drift tables) are codified in src/validate.py, src/data.py, and src/split.py. 

## **3. MLOps Architecture Analysis** 

### **Data Pipeline** 

- **Version Control** : Raw data (data/raw/hotels.csv) is tracked via DVC (data/raw/hotels.csv.dvc). 

- **Contract Enforcement** : contract.py and src/validate.py act as an automated data gate before any splitting or training occurs. 

- **Dual-Variant Processing** : src/data.py supports both raw ( rows) and dedup ( unique rows) execution tracks to quantify sensitivity to duplicate PMS records. 

- **Decision Provenance** : Every row kept or dropped is recorded in reports/decisions_log.csv via src/tracking.py. 

### **Training & Buffer Selection Pipeline** 

- **Orchestration** : Managed through dvc.yaml and locked in dvc.lock. 

- **Model Families** : Compares non-ML baselines (flat rule, segment/deposit lookup tables) against regularized linear models (Logistic Regression) and gradient boosted trees (LightGBM, XGBoost) in src/backtest.py and src/tune.py. 

- **Post-Processing Calibration** : Decouples probability prediction ( ) from the business decision rule ( ) via src/freeze_buffer.py. 

### **Model Artifacts & State Management** 

- **Models** : Stored in models/raw/ and models/dedup/ (git-ignored via models/.gitignore and 

tracked by DVC). 

- **Policy State** : Safety buffers are stored as human-readable JSON artifacts in reports/frozen_buffers_raw.json and reports/frozen_buffers_dedup.json. 

### **Deployment & Inference (src/nightly.py)** 

- **Interface** : Command-line utility invoked via python src/nightly.py --from 2017-08-18 --nights 14. 

- **Leakage Safety** : src/nightly.py invokes src/features.py with strict booking-time validation so that production inference cannot access post-booking columns. 

### **Monitoring, Drift & Auditability** 

- **Temporal Drift Monitoring** : src/split.py generates reports/drift_table_raw.csv and reports/drift_table_dedup.csv to monitor shifts in cancellation rates, ADR, and lead times across Train, Choose, and Prove. 

- **Proof Audit Trail** : docs/PROVE_RUNS.txt logs proof evaluations to verify that the test period (2017) was not used for iterative hyperparameter or buffer tuning. 

- **CI/CD** : .github/workflows/ci.yaml runs automated tests (pytest) and static checks on push and pull requests. 

- **Configuration Management** : Centralized in params.yaml and parsed via src/params.py, with environment variables templated in .env.example and dependencies locked in requirements.lock.txt. 

