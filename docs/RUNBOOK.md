# RUNBOOK

How to set up and work on `hotel-oversell` as a team. Commands assume **Git Bash on Windows**; for Mac/Linux, replace `.venv/Scripts/activate` with `.venv/bin/activate` and `python` with `python3`.

## 1. One-time setup

### Accounts and access
- [ ] GitHub: accept the collaborator invite for `OmRohamare/hotel-oversell`
- [ ] DagsHub: accept the project invite and create your own access token
- [ ] AWS: get an IAM user with access to the project's S3 bucket (ask the repo owner)
- [ ] Install: `git`, Python 3.10+, AWS CLI

### AWS credentials
Use your **own** access key. Never share keys or commit them.

```bash
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region name:   <REGION>
```

### Clone and install
```bash
cd /c/dev          # any folder that is NOT inside another git repo
git clone https://github.com/OmRohamare/hotel-oversell.git
cd hotel-oversell
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env     # fill in your own values; .env is never committed
```

### Get the data
```bash
dvc pull
dvc status        # should report data and pipeline are up to date
```

## 2. Daily workflow

```bash
git checkout main && git pull
dvc pull                          # sync data and outputs to the committed version
git checkout -b <yourname>/<short-task>
# ... make changes ...
ruff check .
pytest
git add <files>
git commit -m "feat: <what changed>"
git push -u origin <yourname>/<short-task>
# open a pull request on GitHub; do not commit directly to main
```

## 3. Running the pipeline

```bash
dvc repro            # re-runs only the stages whose code, data or params changed
dvc metrics show     # compare metrics once metrics files are defined
```

- Change hyperparameters and settings in `params.yaml`, not in the code.
- `data/interim/`, `data/processed/`, `models/` and `artifacts/` are **not** in Git. They are rebuilt by `dvc repro` or pulled with `dvc pull`.
- Run names in MLflow should identify the person, e.g. `om-rf-v1`.

## 4. Changing the data

`data/raw/` is read-only by default. If the raw data really changes:

```bash
dvc add data/raw/hotels.csv
dvc push
git add data/raw/hotels.csv.dvc
git commit -m "data: update raw hotels.csv (<reason>)"
git push
```

Everyone else then runs `git pull` followed by `dvc pull`.

Before sharing any dataset, confirm it contains no guest personal data (names, emails, phone numbers, card numbers).

## 5. Sharing results

After `dvc repro`, if teammates need your outputs:

```bash
dvc push
git add dvc.lock
git commit -m "pipeline: update outputs"
git push
```

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `dvc pull` says access denied or no credentials | Re-run `aws configure`; check the bucket name and region in `.dvc/config` |
| `dvc pull` says file not in cache or remote | The author has not run `dvc push` yet; ask them |
| Merge conflict in `dvc.lock` | Merge code and `params.yaml` first, then run `dvc repro` to regenerate `dvc.lock`; do not hand-edit it |
| `gh` or other command not found in VS Code | Fully restart VS Code so it picks up the new PATH |
| Wrong Python packages / import errors | Check the venv is active (`(.venv)` in prompt), then `pip install -r requirements.txt` |
| Terminal shows a branch name in the home folder | Your home directory is a git repo; work in a separate folder such as `/c/dev` |
| MLflow runs not appearing in DagsHub | Check `MLFLOW_TRACKING_URI` and your DagsHub token in `.env` |

## 7. Rules of thumb
- Never commit `.env`, keys, tokens or raw data files.
- One branch per task; pull requests into `main`.
- Do not edit `dvc.lock` by hand.
- Keep `requirements.txt` and `requirements.lock.txt` in sync when adding packages (`pip freeze > requirements.lock.txt`).
