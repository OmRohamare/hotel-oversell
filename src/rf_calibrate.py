"""Forward-chaining check of a recalibrated random forest. Reads train + choose ONLY.

The proof period is never opened here. Run it as a module so pickled models resolve:

    python -m src.rf_calibrate

What it does, on the primary variant (params.yaml: variants.primary):
  - splits train + choose into expanding-window folds (fit on the past, validate on the next
    `fold_months` months), so period-to-period drift shows up inside model selection;
  - for each fold fits three things: the Logistic (control), the plain random forest, and the
    random forest recalibrated on the most recent `calib_months` of its training window;
  - for every constant buffer in the grid, scores the fold through the client's contract and
    keeps the cheapest buffer whose WORST fold (and, if asked, worst hotel) stays under
    `target_walks_per_1000`;
  - writes reports/rf_forward_chain.csv and logs the per-fold mean error (rooms/night).

Constant buffers only; negative values are allowed because k = floor(expected - buffer), so a
negative buffer sells above the expected number of cancellations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.backtest import FOREST, LOGISTIC, build_methods
from src.data import TARGET, load_contract, load_period
from src.logger import get_logger, log_stage, run_cli
from src.params import REPORTS, load_params, require_params

logger = get_logger(__name__)

METHODS = ("intercept", "platt", "isotonic")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def month_number(df: pd.DataFrame) -> pd.Series:
    """Months as a running integer (year * 12 + month) of the arrival date."""
    d = pd.to_datetime(df["arrival_date"])
    return d.dt.year * 12 + d.dt.month


class Calibrator:
    """Map raw forest probabilities to calibrated ones, fitted on held-out recent bookings."""

    def __init__(self, method: str):
        if method not in METHODS:
            raise ValueError(f"unknown calibration method {method!r}; use {METHODS}")
        self.method = method

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        y = np.asarray(y, dtype=float)
        if len(np.unique(y)) < 2:
            raise ValueError("calibration window has one class only")
        if self.method == "isotonic":
            self.m_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.m_.fit(p, y)
        elif self.method == "platt":
            self.m_ = LogisticRegression(C=1e6, max_iter=1000)
            self.m_.fit(_logit(p).reshape(-1, 1), y)
        else:  # intercept: one number that moves the average prediction onto the observed rate
            z, target = _logit(p), float(y.mean())
            lo, hi = -6.0, 6.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if _sigmoid(z + mid).mean() < target:
                    lo = mid
                else:
                    hi = mid
            self.shift_ = (lo + hi) / 2
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            out = self.m_.predict(p)
        elif self.method == "platt":
            out = self.m_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        else:
            out = _sigmoid(_logit(p) + self.shift_)
        return np.clip(out, 0.0, 1.0)


class CalibratedForest:
    """Drop-in for backtest.ForestMethod: same name, fit(train), predict_proba(frame).

    The forest is fitted on the older part of the training window and the calibrator on the
    most recent `calib_months`, which the forest has not seen.
    """

    name = FOREST

    def __init__(self, forest, method: str, calib_months: int):
        if calib_months < 1:
            raise ValueError("calib_months must be >= 1")
        self.forest, self.method, self.calib_months = forest, method, calib_months

    def fit(self, train: pd.DataFrame) -> CalibratedForest:
        m = month_number(train)
        cut = m.max() - self.calib_months + 1
        older, recent = train[m < cut], train[m >= cut]
        if older.empty or recent.empty:
            raise ValueError("training window too short to hold out a calibration slice")
        self.forest.fit(older)
        self.cal_ = Calibrator(self.method).fit(
            self.forest.predict_proba(recent), recent[TARGET].to_numpy()
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.cal_.predict(self.forest.predict_proba(frame))


def forward_folds(df: pd.DataFrame, min_train_months: int, fold_months: int):
    """Expanding-window folds: yield (train_df, validation_df) in time order."""
    m = month_number(df)
    start = m.min() + min_train_months
    while start <= m.max():
        end = start + fold_months
        val = df[(m >= start) & (m < end)]
        if not val.empty:
            yield df[m < start], val
        start = end


def _k_vector(nt: pd.DataFrame, buffer: float) -> np.ndarray:
    """Same rule as src.costs.oversell (constant family) but a negative buffer is allowed."""
    k = np.floor(nt["expected"] - buffer).clip(lower=0)
    return np.minimum(k, nt["book"]).astype(int).to_numpy()


def score_grid(nt: pd.DataFrame, grid: list[int]) -> pd.DataFrame:
    """Cost and walks for every constant buffer, pooled and per hotel, via the client's contract."""
    K = load_contract()
    rows = []
    for b in grid:
        pooled = K.night_cost(nt, _k_vector(nt, b))
        per_hotel = {}
        for hotel, sub in nt.groupby(level="hotel"):
            per_hotel[hotel] = K.night_cost(sub, _k_vector(sub, b))["walks_per_1000"]
        rows.append(
            {
                "buffer": b,
                "cost_eur": pooled["cost_eur"],
                "walks_per_1000": pooled["walks_per_1000"],
                "worst_hotel_walks": max(per_hotel.values()),
            }
        )
    return pd.DataFrame(rows)


def run_candidate(name: str, make, data: pd.DataFrame, fc: dict, grid: list[int]):
    """Forward-chain one candidate. Returns (fold diagnostics, per-buffer summary, chosen buffer)."""
    K = load_contract()
    fold_rows, scored = [], []
    for i, (tr, va) in enumerate(
        forward_folds(data, fc["min_train_months"], fc["fold_months"]), start=1
    ):
        model = make().fit(tr)
        p = model.predict_proba(va)
        nt = K.nightly_table(va, p)
        err = nt["expected"] - nt["cancelled"]
        fold_rows.append(
            {
                "candidate": name,
                "fold": i,
                "train_rate": float(tr[TARGET].mean()),
                "val_rate": float(va[TARGET].mean()),
                "mean_err_rooms_night": float(err.mean()),
                "mae_rooms_night": float(err.abs().mean()),
            }
        )
        s = score_grid(nt, grid)
        s["fold"] = i
        scored.append(s)
    folds = pd.DataFrame(fold_rows)
    allfolds = pd.concat(scored)
    walk_col = "worst_hotel_walks" if fc["per_hotel"] else "walks_per_1000"
    summary = allfolds.groupby("buffer").agg(
        total_cost_eur=("cost_eur", "sum"),
        worst_fold_walks=(walk_col, "max"),
    )
    ok = summary[summary.worst_fold_walks <= fc["target_walks_per_1000"]]
    chosen = int(ok.total_cost_eur.idxmin()) if len(ok) else None
    return folds, summary, chosen


@log_stage("rf_forward_chain")
def main() -> None:
    params = load_params()
    require_params(
        params,
        "variants.primary",
        "models.random_forest",
        "rf_calibration.method",
        "rf_calibration.calib_months",
        "forward_chaining.min_train_months",
        "forward_chaining.fold_months",
        "forward_chaining.buffer_min",
        "forward_chaining.buffer_max",
        "forward_chaining.target_walks_per_1000",
        "forward_chaining.per_hotel",
    )
    fc, cal = params["forward_chaining"], params["rf_calibration"]
    K = load_contract()
    slug = params["variants"]["primary"]
    data = pd.concat(
        [load_period(slug, "train"), load_period(slug, "choose")], ignore_index=True
    )
    if pd.to_datetime(data["arrival_date"]).max() >= pd.Timestamp(K.PROOF_START):
        raise RuntimeError("train + choose reaches into the proof period; stopping")
    grid = list(range(int(fc["buffer_min"]), int(fc["buffer_max"]) + 1))

    candidates = {
        "Logistic (control)": lambda: build_methods(params)[LOGISTIC],
        "Forest, plain": lambda: build_methods(params)[FOREST],
        f"Forest, calibrated ({cal['method']})": lambda: CalibratedForest(
            build_methods(params)[FOREST], cal["method"], int(cal["calib_months"])
        ),
    }
    all_folds, out_rows = [], []
    for name, make in candidates.items():
        folds, summary, chosen = run_candidate(name, make, data, fc, grid)
        all_folds.append(folds)
        logger.info("%s per fold:\n%s", name, folds.round(3).to_string(index=False))
        if chosen is None:
            logger.warning(
                "%s: no constant buffer in [%d, %d] keeps every fold at or under %.1f "
                "walks per 1,000; rejected under the rule fixed in advance",
                name,
                grid[0],
                grid[-1],
                fc["target_walks_per_1000"],
            )
        else:
            row = summary.loc[chosen]
            logger.info(
                "%s: buffer %d, total cost EUR %.0f, worst fold %.2f walks per 1,000",
                name,
                chosen,
                row.total_cost_eur,
                row.worst_fold_walks,
            )
            if chosen in (grid[0], grid[-1]):
                logger.warning("%s: chosen buffer sits at the grid edge", name)
        out_rows.append(
            {
                "candidate": name,
                "chosen_buffer": chosen,
                "total_cost_eur": None if chosen is None else summary.loc[chosen].total_cost_eur,
                "worst_fold_walks": None if chosen is None else summary.loc[chosen].worst_fold_walks,
            }
        )
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.concat(all_folds).to_csv(REPORTS / "rf_forward_chain_folds.csv", index=False)
    result = pd.DataFrame(out_rows)
    result.to_csv(REPORTS / "rf_forward_chain.csv", index=False)
    logger.info("forward-chaining summary:\n%s", result.round(2).to_string(index=False))


if __name__ == "__main__":
    run_cli(main)
