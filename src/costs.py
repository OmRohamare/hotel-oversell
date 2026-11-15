"""The oversell rule and the euro comparison between methods.

Notebook cells: 22 (oversell), 26 (proof table), 29 (per-hotel view), 31 (bootstrap).
The client's cost function (contract.night_cost) is used as is; nothing here changes it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import load_contract
from src.logger import get_logger

logger = get_logger(__name__)

VALID_FAMILIES = ("constant", "spread")
REQUIRED_NIGHTLY = ["expected", "spread", "cancelled", "book", "stay_value"]
REF_ZERO = "(ref) k = 0, sell nothing extra"


def check_nightly(nt: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_NIGHTLY if c not in nt.columns]
    if missing:
        raise ValueError(f"nightly table is missing columns {missing}")
    if nt.empty:
        raise ValueError("nightly table has no nights")


def oversell(nt: pd.DataFrame, family: str, value: float) -> pd.Series:
    """k = max(0, floor(expected - buffer)), never more than the night's bookings."""
    if family not in VALID_FAMILIES:
        raise ValueError(f"unknown buffer family {family!r}; use one of {VALID_FAMILIES}")
    if value < 0:
        raise ValueError(f"buffer must be >= 0, got {value}")
    check_nightly(nt)
    buf = value if family == "constant" else value * nt["spread"]
    k = np.floor(nt["expected"] - buf).clip(lower=0)
    return np.minimum(k, nt["book"]).astype(int)


def night_cost_vector(nt: pd.DataFrame, family: str, value: float) -> np.ndarray:
    """Cost in euros of every night for one buffer setting."""
    K = load_contract()
    k = pd.Series(np.asarray(oversell(nt, family, value)), index=nt.index)
    empty = (nt["cancelled"] - k).clip(lower=0)
    walks = (k - nt["cancelled"]).clip(lower=0)
    return (empty * nt["stay_value"] + walks * K.WALK_COST).to_numpy()


def proof_table(nightly: dict, frozen: dict, baseline: str) -> pd.DataFrame:
    """Score every method once with its frozen buffer; savings are vs the flat rule."""
    K = load_contract()
    if set(frozen) != set(nightly):
        raise ValueError(f"frozen buffers {sorted(frozen)} do not match methods {sorted(nightly)}")
    if baseline not in nightly:
        raise ValueError(f"baseline method {baseline!r} not among {sorted(nightly)}")
    rows = []
    for name, nt in nightly.items():
        fb = frozen[name]
        r = K.night_cost(nt, oversell(nt, fb["family"], fb["buffer"]))
        rows.append({"method": name, "buffer": f'{fb["family"]} {fb["buffer"]:g}', **r})
    first = nightly[baseline]
    zero = K.night_cost(first, np.zeros(len(first), dtype=int))
    rows.append({"method": REF_ZERO, "buffer": "-", **zero})

    tbl = pd.DataFrame(rows).set_index("method")
    base_cost = tbl.loc[baseline, "cost_eur"]
    if not base_cost:
        raise ValueError("baseline cost is zero; savings are undefined")
    tbl["saving_vs_flat_eur"] = base_cost - tbl.cost_eur
    tbl["saving_vs_flat_%"] = 100 * tbl.saving_vs_flat_eur / base_cost
    for name, row in tbl.iterrows():
        if not name.startswith("(ref)") and not row.ceiling_met:
            logger.warning("%s breaks the pooled walk ceiling on proof (%.2f per 1,000); rejected",
                           name, row.walks_per_1000)
    return tbl


def per_hotel_table(nightly: dict, frozen: dict, baseline: str) -> pd.DataFrame:
    """The ceiling per hotel: a method can pass pooled and still break it at one hotel."""
    K = load_contract()
    rows = []
    for name, nt in nightly.items():
        fb = frozen[name]
        for hotel, sub in nt.groupby(level="hotel"):
            r = K.night_cost(sub, oversell(sub, fb["family"], fb["buffer"]))
            base_sub = nightly[baseline].xs(hotel, level="hotel", drop_level=False)
            fbb = frozen[baseline]
            rb = K.night_cost(base_sub, oversell(base_sub, fbb["family"], fbb["buffer"]))
            ok = bool(r["walks_per_1000"] <= K.WALK_CEILING)
            rows.append({"method": name, "hotel": hotel, "nights": r["nights"],
                         "cost_eur": r["cost_eur"], "baseline_cost_eur": rb["cost_eur"],
                         "saving_eur": rb["cost_eur"] - r["cost_eur"],
                         "walks_per_1000": r["walks_per_1000"], "ceiling_met_at_hotel": ok})
            if not ok:
                logger.warning("%s breaks the walk ceiling at %s on its own (%.2f per 1,000)",
                               name, hotel, r["walks_per_1000"])
    return pd.DataFrame(rows).set_index(["method", "hotel"])


def bootstrap_savings(nightly: dict, frozen: dict, baseline: str,
                      n_boot: int, seed: int) -> pd.DataFrame:
    """Resample nights to see how stable the saving is. Nights are treated as independent."""
    if n_boot < 100:
        raise ValueError("n_boot should be at least 100")
    rng = np.random.default_rng(seed)
    base = night_cost_vector(nightly[baseline], frozen[baseline]["family"],
                             frozen[baseline]["buffer"])
    rows = []
    for name, nt in nightly.items():
        if name == baseline:
            continue
        diff = base - night_cost_vector(nt, frozen[name]["family"], frozen[name]["buffer"])
        boots = np.array([rng.choice(diff, size=len(diff), replace=True).sum()
                          for _ in range(n_boot)])
        rows.append({"method": name, "saving_eur": diff.sum(), "p5": np.percentile(boots, 5),
                     "p95": np.percentile(boots, 95), "share_of_bootstraps_>0": (boots > 0).mean()})
    return pd.DataFrame(rows).set_index("method")
