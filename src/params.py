"""Load project parameters and define the standard project paths."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"

PROVE_RUNS_PATH = DOCS / "PROVE_RUNS.txt"

VARIANT_LABELS = {"dedup": "deduplicated (primary)", "raw": "raw (sensitivity)"}


def load_params(path: Path | str = PARAMS_PATH) -> dict:
    """Read params.yaml. Raises FileNotFoundError or ValueError with a clear message."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Parameter file not found: %s", path)
        raise
    except yaml.YAMLError as exc:
        logger.error("Parameter file %s is not valid YAML: %s", path.name, exc)
        raise ValueError(f"{path.name} is not valid YAML") from exc
    if not isinstance(params, dict):
        raise TypeError(f"{path.name} is not a mapping")

    if not params:
        raise ValueError(f"{path.name} is empty")
    logger.info("Loaded parameters from %s", path.name)
    return params


def require_params(params: dict, *dotted_keys: str) -> None:
    """Raise ValueError naming every missing key, e.g. require_params(p, 'seed', 'models.logistic')."""
    missing = []
    for dotted in dotted_keys:
        node = params
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.append(dotted)
                break
            node = node[part]
    if missing:
        logger.error("params.yaml is missing required keys: %s", missing)
        raise ValueError(f"params.yaml is missing required keys: {missing}")


def variant_slugs(params: dict) -> list[str]:
    """The data variants to run, e.g. ['dedup', 'raw']; the primary one must be among them."""
    require_params(params, "variants.primary", "variants.list")
    slugs = list(params["variants"]["list"])
    unknown = [s for s in slugs if s not in VARIANT_LABELS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; use {list(VARIANT_LABELS)}")
    if params["variants"]["primary"] not in slugs:
        raise ValueError("variants.primary must be one of variants.list")
    return slugs


def frozen_path(slug: str) -> Path:
    return REPORTS / f"frozen_buffers_{slug}.json"


def proof_table_path(slug: str) -> Path:
    return REPORTS / f"poc_method_table_{slug}.csv"
