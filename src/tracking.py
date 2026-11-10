"""MLflow tracking helpers for hotel-oversell (DagsHub-hosted MLflow).

Credentials come from environment variables (loaded from a local .env file).
Nothing secret is ever written to code, params.yaml, logs or reports.

Required environment variables (see .env.example):
    MLFLOW_TRACKING_URI       e.g. https://dagshub.com/<owner>/hotel-oversell.mlflow
    DAGSHUB_USERNAME          your DagsHub username
    DAGSHUB_USER_TOKEN        your DagsHub access token
(MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD are used if already set.)
"""
from src.logger import get_logger
logger = get_logger(__name__)
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import mlflow
from dotenv import load_dotenv

DEFAULT_EXPERIMENT = "hotel-oversell"


def setup_tracking(experiment_name: str = DEFAULT_EXPERIMENT) -> None:
    """Configure the MLflow tracking URI, credentials and experiment."""
    load_dotenv()

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set. Fill in your .env file.")

    # Map DagsHub variable names onto the ones MLflow reads, without overriding.
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_USER_TOKEN")
    if username and not os.getenv("MLFLOW_TRACKING_USERNAME"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if token and not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    if not os.getenv("MLFLOW_TRACKING_USERNAME") or not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        raise RuntimeError(
            "DagsHub credentials missing. Set DAGSHUB_USERNAME and DAGSHUB_USER_TOKEN in .env."
        )

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def start_run(run_name: str | None = None, tags: Mapping[str, str] | None = None):
    """Start an MLflow run (call end_run() when finished)."""
    return mlflow.start_run(run_name=run_name, tags=dict(tags) if tags else None)


def end_run(status: str = "FINISHED") -> None:
    """End the active MLflow run."""
    mlflow.end_run(status=status)


@contextmanager
def tracked_run(
    run_name: str | None = None,
    tags: Mapping[str, str] | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> Iterator[mlflow.ActiveRun]:
    """Context manager: set up tracking, start a run, mark FAILED on exceptions."""
    setup_tracking(experiment_name)
    run = start_run(run_name=run_name, tags=tags)
    try:
        yield run
    except Exception:
        end_run(status="FAILED")
        raise
    else:
        end_run()


def log_params(params: Mapping[str, Any]) -> None:
    """Log parameters. Values are stringified by MLflow; keep secrets out of here."""
    mlflow.log_params({str(k): v for k, v in params.items()})


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log numeric metrics."""
    mlflow.log_metrics({str(k): float(v) for k, v in metrics.items()}, step=step)


def log_artifact(path: str | os.PathLike, artifact_path: str | None = None) -> None:
    """Log a local file (report, plot, model file) as a run artifact."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")
    mlflow.log_artifact(str(p), artifact_path=artifact_path)


def log_model_reference(name: str, location: str) -> None:
    """Record where a model lives (e.g. a DVC path or S3 URI) as a run tag,
    without uploading the model itself."""
    mlflow.set_tag(f"model_ref.{name}", location)
