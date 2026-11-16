"""Buffer search on the CHOOSE period only (notebook cells 22 and 23).

Not AUC tuning: that is out of scope. This only searches the buffer grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.costs import oversell
from src.data import load_contract
from src.logger import get_logger
from src.params import require_params

logger = get_logger(__name__)


def build_grid(params: dict) -> list[tuple[str, float]]:
    """Constant buffers 0..constant_max, and spread buffers z * spread for z on a grid."""
    require_params(
        params,
        "buffer_grid.constant_max",
        "buffer_grid.spread_max",
        "buffer_grid.spread_step",
    )
    g = params["buffer_grid"]
    if g["spread_step"] <= 0 or g["constant_max"] < 0 or g["spread_max"] < 0:
        raise ValueError("buffer_grid values must be positive")
    cmin = int(g.get("constant_min", 0))
    grid = [("constant", float(v)) for v in range(cmin, int(g["constant_max"]) + 1)]
    grid += [
        ("spread", round(float(z), 2))
        for z in np.arange(0, g["spread_max"] + 1e-9, g["spread_step"])
    ]
    return grid


def sweep(nt: pd.DataFrame, grid: list[tuple[str, float]]) -> pd.DataFrame:
    """Cost, empty rooms and walks for every buffer in the grid."""
    K = load_contract()
    out = []
    for family, value in grid:
        r = K.night_cost(nt, oversell(nt, family, value))
        out.append({"family": family, "buffer": value, **r})
    return pd.DataFrame(out)


def pick(sw: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Cheapest setting that meets the ceiling; else the fewest walks, flagged."""
    ok = sw[sw.ceiling_met]
    if len(ok):
        return ok.loc[ok.cost_eur.idxmin()], True
    return sw.loc[sw.walks_per_1000.idxmin()], False


def on_grid_edge(family: str, value: float, grid: list[tuple[str, float]]) -> bool:
    """True if the chosen buffer is 0 or the largest value of its family."""
    vals = [v for f, v in grid if f == family]
    return value in (min(vals), max(vals))



def choose_headline(choose_tbl: pd.DataFrame) -> str:
    """Cheapest method that meets the ceiling on the CHOOSE period (never looks at proof);
    if none meets it, the method with the fewest walks."""
    ok = choose_tbl[choose_tbl.ceiling_met_on_choose]
    if len(ok):
        return str(ok.cost_eur.idxmin())
    logger.warning(
        "no method meets the ceiling on the choose period; using the fewest walks"
    )
    return str(choose_tbl.walks_per_1000.idxmin())
