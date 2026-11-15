"""Data access: the raw CSV (DVC + S3), parquet files, JSON and the client's contract."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pandas as pd

from src.logger import get_logger, log_dataframe_summary
from src.params import PROCESSED, ROOT, load_params

logger = get_logger(__name__)

RAW_REL = "data/raw/hotels.csv"
TARGET = "is_canceled"


def read_raw(rev: str | None = None, source: str | None = None) -> pd.DataFrame:
    """Return every booking exactly as exported.

    source: "dvc" fetches through DVC (S3 remote or local cache); "local" reads data/raw/hotels.csv.
            Default comes from params.yaml (data.source).
    rev:    optional git commit, tag or branch to read an older data version (dvc only).
    Raises ValueError if the shape differs from params.yaml (data.expected_rows/cols).
    """
    params = load_params()
    source = source or params.get("data", {}).get("source", "dvc")
    expected = (params["data"]["expected_rows"], params["data"]["expected_cols"])

    try:
        if source == "dvc":
            import dvc.api  # imported here so unit tests need not install DVC

            logger.info("Fetching %s via DVC (rev=%s)", RAW_REL, rev or "current checkout")
            with dvc.api.open(RAW_REL, repo=str(ROOT), rev=rev, mode="r") as f:
                df = pd.read_csv(f)
        elif source == "local":
            path = ROOT / RAW_REL
            logger.info("Reading local file %s", RAW_REL)
            df = pd.read_csv(path)
        else:
            raise ValueError(f"unknown data source {source!r}; use 'dvc' or 'local'")
    except FileNotFoundError:
        logger.error("Raw file missing. Run `dvc pull` or set data.source to 'dvc'.")
        raise
    except ValueError:
        raise
    except Exception as exc:
        logger.error("Could not fetch raw data (%s). Check `aws configure`, the DVC remote "
                     "and your network, then retry.", type(exc).__name__)
        raise RuntimeError(f"could not fetch {RAW_REL}: {exc}") from exc

    if df.shape != expected:
        logger.error("Unexpected raw shape %s, expected %s", df.shape, expected)
        raise ValueError(f"Unexpected raw shape {df.shape}, expected {expected}")
    log_dataframe_summary(logger, df, "raw bookings")
    return df


def arrival_date(df: pd.DataFrame) -> pd.Series:
    """Build the arrival date from year, month name and day of month."""
    needed = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"cannot build arrival_date, missing columns: {missing}")
    text = (df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"].astype(str)
            + "-" + df["arrival_date_day_of_month"].astype(str))
    dates = pd.to_datetime(text, format="%Y-%B-%d", errors="coerce")
    bad = int(dates.isna().sum())
    if bad:
        logger.error("%d bookings have an invalid arrival date", bad)
        raise ValueError(f"{bad} bookings have an invalid arrival date")
    return dates


def read_parquet_checked(path: Path, required: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet file, with clear errors for a missing file or missing columns."""
    path = Path(path)
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        logger.error("Input missing: %s. Run the earlier pipeline stage (dvc repro) first.", path)
        raise
    except Exception as exc:
        logger.error("Could not read %s: %s", path, exc)
        raise
    missing = [c for c in (required or []) if c not in df.columns]
    if missing:
        logger.error("%s is missing columns %s", path.name, missing)
        raise ValueError(f"{path.name} is missing columns {missing}")
    if df.empty:
        raise ValueError(f"{path.name} contains no rows")
    log_dataframe_summary(logger, df, path.name,
                          date_col="arrival_date" if "arrival_date" in df.columns else None)
    return df


def load_period(slug: str, period: str) -> pd.DataFrame:
    """Read one period ('train', 'choose' or 'proof') of one data variant ('dedup' or 'raw')."""
    if period not in ("train", "choose", "proof"):
        raise ValueError(f"unknown period {period!r}")
    return read_parquet_checked(PROCESSED / slug / f"{period}.parquet", ["arrival_date", TARGET])


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet file (creating the folder) and log the output path and row count."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception as exc:
        logger.error("Could not write %s: %s", path, exc)
        raise
    logger.info("Wrote %s (%d rows)", path, len(df))


def write_json(obj: dict, path: Path) -> None:
    """Write JSON atomically so a crash never leaves a half-written file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        logger.error("Could not write %s: %s", path, exc)
        raise
    logger.info("Wrote %s", path)


def read_json(path: Path) -> dict:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("File missing: %s. Run the stage that creates it first.", path)
        raise
    except json.JSONDecodeError as exc:
        logger.error("%s is not valid JSON: %s", path, exc)
        raise


_contract = None


def load_contract():
    """Import the client's contract.py (read-only). Looks in the repo root, then src/."""
    global _contract
    if _contract is None:
        try:
            import contract as module
        except ModuleNotFoundError as exc:
            if exc.name != "contract":
                raise
            try:
                module = importlib.import_module("src.contract")
            except ModuleNotFoundError as exc2:
                if exc2.name not in ("src.contract", "contract"):
                    raise
                logger.debug("contract.py not found in the repo root or src/")
                raise ModuleNotFoundError(
                    "contract.py (the client's rule) is needed in the repo root or src/") from exc2
        _contract = module
    return _contract
