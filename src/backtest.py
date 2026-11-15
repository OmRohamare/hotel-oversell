"""The four methods compared in the project, fitted on the TRAIN period of each data variant.

1 Flat rule       each hotel's training cancellation rate (the client's naive benchmark)
2 Segment table   rate by hotel x market segment x channel, shrunk toward the hotel rate
3 Logistic        one-hot categoricals, standardised numerics
4 Random forest   300 trees, at least 20 bookings per leaf, ordinal-coded categoricals

Every method returns one cancellation probability per booking, so the same cost rule scores all
of them on the same nights. Notebook cells: 18 (methods), 20 and 22 (diagnostics, drift).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Lets `python src/backtest.py` work as well as `python -m src.backtest`:
# a script run puts src/ on the path, so add the repo root for `import src.*` and `import contract`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data import TARGET, load_contract, load_period
from src.features import CAT, NUM, RarePooler
from src.logger import get_logger, log_stage, run_cli
from src.params import (
    MODELS,
    REPORTS,
    VARIANT_LABELS,
    load_params,
    proof_table_path,
    require_params,
    variant_slugs,
)

logger = get_logger(__name__)

FLAT = "1 Flat rule"
SEGMENT = "2 Segment table"
LOGISTIC = "3 Logistic"
FOREST = "4 Random forest"
SLUGS = {FLAT: "flat_rule", SEGMENT: "segment_table", LOGISTIC: "logistic", FOREST: "random_forest"}
SEG = ["hotel", "market_segment", "distribution_channel"]


class FlatRate:
    """P(cancel) = the hotel's training cancellation rate, for every booking."""

    name = FLAT

    def fit(self, train: pd.DataFrame) -> FlatRate:
        self.rate_ = train.groupby("hotel")[TARGET].mean()
        self.overall_ = float(train[TARGET].mean())
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["hotel"].map(self.rate_).fillna(self.overall_).to_numpy(dtype=float)


class SegmentTable:
    """Cancellation rate per hotel x segment x channel, shrunk toward the hotel rate."""

    name = SEGMENT

    def __init__(self, shrink_m: float):
        if shrink_m < 0:
            raise ValueError("shrink_m must be >= 0")
        self.shrink_m = shrink_m

    def fit(self, train: pd.DataFrame) -> SegmentTable:
        self.hotel_rate_ = train.groupby("hotel")[TARGET].mean()
        self.overall_ = float(train[TARGET].mean())
        g = train.groupby(SEG)[TARGET].agg(["sum", "count"]).reset_index()
        g["prior"] = g.hotel.map(self.hotel_rate_)
        g["rate"] = (g["sum"] + self.shrink_m * g.prior) / (g["count"] + self.shrink_m)
        self.table_ = g.set_index(SEG).rate
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        key = pd.MultiIndex.from_frame(frame[SEG])
        p = pd.Series(self.table_.reindex(key).to_numpy(), index=frame.index)
        fallback = frame.hotel.map(self.hotel_rate_).fillna(self.overall_)
        return p.fillna(fallback).to_numpy(dtype=float)


class _PooledSklearn:
    """Category pooling + encoding + an sklearn classifier, as one picklable object."""

    name = ""

    def __init__(self, pool_top: dict[str, int]):
        self.pool_top = pool_top

    def _pipeline(self) -> Pipeline:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, train: pd.DataFrame):
        self.pooler_ = RarePooler(self.pool_top).fit(train)
        self.pipe_ = self._pipeline()
        self.pipe_.fit(self.pooler_.transform(train), train[TARGET].to_numpy())
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict_proba(self.pooler_.transform(frame))[:, 1]


class LogisticMethod(_PooledSklearn):
    name = LOGISTIC

    def __init__(self, pool_top: dict[str, int], C: float, max_iter: int, min_frequency: int):
        super().__init__(pool_top)
        self.C, self.max_iter, self.min_frequency = C, max_iter, min_frequency

    def _pipeline(self) -> Pipeline:
        pre = ColumnTransformer([
            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=self.min_frequency), CAT),
            ("n", StandardScaler(), NUM),
        ])
        return Pipeline([("pre", pre), ("est", LogisticRegression(max_iter=self.max_iter, C=self.C))])


class ForestMethod(_PooledSklearn):
    name = FOREST

    def __init__(self, pool_top: dict[str, int], estimator: RandomForestClassifier):
        super().__init__(pool_top)
        self.estimator = estimator

    def _pipeline(self) -> Pipeline:
        pre = ColumnTransformer(
            [("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
            remainder="passthrough")
        return Pipeline([("pre", pre), ("est", clone(self.estimator))])


def build_methods(params: dict) -> dict:
    """Create the four unfitted methods from params.yaml."""
    seed = params["seed"]
    top = params["features"]["pool_top"]
    seg, lg, rf = (params["models"][k] for k in ("segment_table", "logistic", "random_forest"))
    return {
        FLAT: FlatRate(),
        SEGMENT: SegmentTable(seg["shrink_m"]),
        LOGISTIC: LogisticMethod(top, lg["C"], lg["max_iter"], params["features"]["min_frequency"]),
        FOREST: ForestMethod(top, RandomForestClassifier(
            n_estimators=rf["n_estimators"], min_samples_leaf=rf["min_samples_leaf"],
            max_features=rf["max_features"], n_jobs=rf["n_jobs"], random_state=seed)),
    }


def fit_all(train: pd.DataFrame, params: dict) -> dict:
    """Fit every method on the training period and return them by name."""
    require_params(params, "seed", "features.pool_top", "features.min_frequency",
                   "models.segment_table", "models.logistic", "models.random_forest")
    if train.empty or train[TARGET].nunique() < 2:
        raise ValueError("training data must contain both cancelled and kept bookings")
    methods = build_methods(params)
    for name, method in methods.items():
        logger.info("Fitting %s on %d training bookings", name, len(train))
        try:
            method.fit(train)
        except Exception:
            logger.error("Fitting %s failed", name)
            raise
    return methods


def save_methods(methods: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, method in methods.items():
            path = out_dir / f"{SLUGS[name]}.joblib"
            joblib.dump(method, path)
            logger.info("Saved %s -> %s", name, path)
    except OSError as exc:
        logger.error("Could not save models to %s: %s", out_dir, exc)
        raise


def load_methods(in_dir: Path) -> dict:
    in_dir = Path(in_dir)
    methods = {}
    for name, slug in SLUGS.items():
        path = in_dir / f"{slug}.joblib"
        try:
            methods[name] = joblib.load(path)
        except FileNotFoundError:
            logger.error("Model file missing: %s. Run the backtest stage first.", path)
            raise
        except Exception as exc:
            logger.error("Could not load %s: %s", path, exc)
            raise
    logger.info("Loaded %d models from %s", len(methods), in_dir)
    return methods


def nightly_tables(methods: dict, frame: pd.DataFrame) -> dict:
    """Score a period: one nightly table per method, through the client's contract."""
    K = load_contract()
    tables = {}
    for name, method in methods.items():
        try:
            p = method.predict_proba(frame)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError(f"{name} produced invalid probabilities")
            tables[name] = K.nightly_table(frame, p)
        except Exception:
            logger.error("Scoring %s failed on %d bookings", name, len(frame))
            raise
    return tables


def choose_diagnostics(methods: dict, choose: pd.DataFrame) -> pd.DataFrame:
    """AUC is information only; the night-level bias and error are what matter."""
    tables = nightly_tables(methods, choose)
    rows = []
    for name, nt in tables.items():
        err = nt.expected - nt.cancelled
        p = methods[name].predict_proba(choose)
        if choose[TARGET].nunique() == 2:
            auc = roc_auc_score(choose[TARGET], p)
        else:
            auc = float("nan")
            logger.warning("choose period has one class only; AUC not defined")
        rows.append({"method": name, "AUC (info only)": auc, "mean err (rooms/night)": err.mean(),
                     "MAE (rooms/night)": err.abs().mean(), "avg spread": nt.spread.mean()})
        if err.mean() < -1:
            logger.warning("%s expects fewer cancellations than happen (mean err %.2f rooms/night); "
                           "its buffer may sit at 0", name, err.mean())
    return pd.DataFrame(rows).set_index("method")


def drift_table(train: pd.DataFrame, choose: pd.DataFrame) -> pd.DataFrame:
    """What moved after the training year: cancellation rate, Non Refund share, lead time."""
    def stats(f):
        return [f[TARGET].mean(), (f.deposit_type == "Non Refund").mean(), f.lead_time.mean()]
    return pd.DataFrame({"train": stats(train), "choose": stats(choose)},
                        index=["cancel rate", "share Non Refund", "mean lead time (days)"])


@log_stage("backtest")
def main() -> None:
    params = load_params()
    for slug in variant_slugs(params):
        train, choose = load_period(slug, "train"), load_period(slug, "choose")
        methods = fit_all(train, params)
        save_methods(methods, MODELS / slug)

        diag = choose_diagnostics(methods, choose)
        drift = drift_table(train, choose)
        try:
            REPORTS.mkdir(parents=True, exist_ok=True)
            diag.to_csv(REPORTS / f"choose_diagnostics_{slug}.csv")
            drift.to_csv(REPORTS / f"drift_table_{slug}.csv")
        except OSError as exc:
            logger.error("Could not write backtest reports: %s", exc)
            raise
        logger.info("[%s] choose-period diagnostics:\n%s", slug, diag.round(2).to_string())


def show_proof_tables(params: dict) -> bool:
    """Print the method table on the PROOF period for each variant, if it has been scored.

    This only reads what `src.prove` already wrote; it never scores the proof period itself,
    so printing it can be repeated safely. Returns True if at least one table was printed.
    """
    shown = False
    for slug in variant_slugs(params):
        path = proof_table_path(slug)
        if not path.exists():
            continue
        tbl = pd.read_csv(path, index_col=0)
        print(f"\n=== Proof period, {VARIANT_LABELS[slug]}: every method, frozen buffers ===")
        print(tbl.round(2).to_string())
        shown = True
    if not shown:
        print("\nThe proof period has not been scored yet. When the team agrees the buffers are "
              "final, run it ONCE with:  python src/backtest.py --prove")
    return shown


def cli(argv=None) -> None:
    """The command the course asks for: python src/backtest.py

    Default: fit every method on train, choose each buffer on the choose period and freeze it,
    then print the proof-period table if it has been scored.
    --prove: also score the proof period (once; every run is logged in docs/PROVE_RUNS.txt).
    --stage-only: fit and save only (what the DVC `backtest` stage runs).
    """
    ap = argparse.ArgumentParser(description="Fit all methods, freeze buffers, show the proof table")
    ap.add_argument("--stage-only", action="store_true", help="only fit and save the models")
    ap.add_argument("--prove", action="store_true",
                    help="also score the proof period; do this once, after the buffers are final")
    args = ap.parse_args(argv)
    if args.stage_only and args.prove:
        ap.error("--stage-only and --prove cannot be combined")

    main()
    if args.stage_only:
        return
    # imported here because these modules import this one
    from src.freeze_buffer import main as freeze_main
    freeze_main()
    if args.prove:
        from src.prove import main as prove_main
        prove_main()
    show_proof_tables(load_params())


if __name__ == "__main__":
    # Import the stage from the package so saved models reference src.backtest.*, not __main__
    # (a model pickled from __main__ cannot be loaded by freeze_buffer, prove or nightly).
    from src.backtest import cli as _cli

    run_cli(_cli)
