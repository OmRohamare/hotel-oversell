"""Apply the frozen PRIMARY buffers to the next N nights.

    python src/nightly.py --from 2017-08-18 --nights 14 [--variant dedup] [--method "3 Logistic"]

Uses only known-at-booking columns for the prediction. Writes artifacts/nightly_<date>.csv.
Note: the nightly table comes from the client's contract.nightly_table, which here also
needs the realised outcome columns; for live use with undecided bookings it must be adapted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Lets `python src/nightly.py ...` work as well as `python -m src.nightly ...`:
# a script run puts src/ on the path, so add the repo root for `import src.*` and `import contract`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest import load_methods
from src.costs import oversell
from src.data import load_contract, read_json, read_parquet_checked
from src.logger import get_logger, log_stage, run_cli
from src.params import (
    ARTIFACTS,
    INTERIM,
    MODELS,
    frozen_path,
    load_params,
    variant_slugs,
)

logger = get_logger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Oversell recommendation per hotel and night"
    )
    ap.add_argument(
        "--from", dest="start", required=True, help="first arrival date, YYYY-MM-DD"
    )
    ap.add_argument(
        "--nights", type=int, default=14, help="number of nights (default 14)"
    )
    ap.add_argument(
        "--variant", default=None, help="data variant; default: the primary variant"
    )
    ap.add_argument(
        "--method", default=None, help="method name; default: the frozen headline model"
    )
    return ap.parse_args(argv)


@log_stage("nightly")
def main(argv=None) -> None:
    args = parse_args(argv)
    params = load_params()
    variant = args.variant or params["variants"]["primary"]
    if variant not in variant_slugs(params):
        raise ValueError(
            f"unknown variant {variant!r}; choose one of {variant_slugs(params)}"
        )
    try:
        start = pd.Timestamp(args.start)
    except ValueError as exc:
        raise ValueError(
            f"--from must be a date like 2017-08-18, got {args.start!r}"
        ) from exc
    if args.nights < 1:
        raise ValueError("--nights must be at least 1")
    end = start + pd.Timedelta(days=args.nights)

    frame = read_parquet_checked(INTERIM / "clean.parquet", ["arrival_date", "is_dup"])
    if variant == "dedup":
        frame = frame.loc[~frame.is_dup]
    window = frame[(frame.arrival_date >= start) & (frame.arrival_date < end)]
    if window.empty:
        logger.error("No bookings arrive between %s and %s", start.date(), end.date())
        raise ValueError(f"no bookings between {start.date()} and {end.date()}")
    logger.info(
        "Window %s to %s (%s): %d bookings",
        start.date(),
        end.date(),
        variant,
        len(window),
    )

    frozen_file = read_json(frozen_path(variant))
    frozen = frozen_file["buffers"]
    method_name = args.method or frozen_file["headline_model"]
    if method_name not in frozen:
        raise ValueError(
            f"unknown method {method_name!r}; choose one of {sorted(frozen)}"
        )
    fb = frozen[method_name]
    logger.info(
        "Method %s with frozen buffer %s %g", method_name, fb["family"], fb["buffer"]
    )

    method = load_methods(MODELS / variant)[method_name]
    K = load_contract()
    full = K.nightly_table(window, method.predict_proba(window))
    nt = full[["book", "expected", "spread"]].copy()
    nt["buffer"] = f"{fb['family']} {fb['buffer']:g}"
    nt["rooms_to_oversell"] = oversell(full, fb["family"], fb["buffer"])
    nights_found = len(nt.index.get_level_values(-1).unique())
    if nights_found < args.nights:
        logger.warning(
            "Only %d of %d requested nights have bookings", nights_found, args.nights
        )

    try:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        out = ARTIFACTS / f"nightly_{start:%Y-%m-%d}_{args.nights}n.csv"
        nt.to_csv(out)
    except OSError as exc:
        logger.error("Could not write %s: %s", ARTIFACTS, exc)
        raise
    logger.info("Wrote %s (%d hotel-nights)", out, len(nt))


if __name__ == "__main__":
    run_cli(lambda: main(sys.argv[1:]))
