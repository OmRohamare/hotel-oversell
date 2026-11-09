# hotel-oversell

Predicting hotel oversell risk with a reproducible ML pipeline.

**Stack:** DVC (data versioning, S3 remote), MLflow (experiment tracking), DagsHub, GitHub Actions.

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your credentials
dvc pull               # fetch versioned data once the remote is configured
```

## Project structure
```
data/{raw,interim,processed}/  DVC-managed data (not in Git)
models/ artifacts/ reports/    pipeline outputs
logs/                          run logs (ignored)
docs/                          data dictionary, model card, runbook
src/                           pipeline code
tests/                         pytest tests
params.yaml / dvc.yaml         parameters and pipeline definition
```

## Status
Initial project skeleton.
