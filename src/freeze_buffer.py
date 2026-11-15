"""Stage 4: choose each method's buffer on the CHOOSE period, then freeze it to disk, per variant.

Notebook cells: 20 (choose and freeze), 24 (chart), 27 (do the buffers move between variants).
The proof period is never touched here. The headline model is the cheapest method that meets
the ceiling on the choose period.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import FLAT, load_methods, nightly_tables
from src.data import load_contract, load_period, write_json
from src.logger import get_logger, log_stage, run_cli
from src.params import (
    MODELS,
    REPORTS,
    VARIANT_LABELS,
    frozen_path,
    load_params,
    variant_slugs,
)
from src.tune import build_grid, choose_headline, on_grid_edge, pick, sweep

logger = get_logger(__name__)


def plot_sweeps(sweeps: dict, ceiling: float, title: str, out_path) -> None:
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for name, sw in sweeps.items():
            s = sw[sw.family == "constant"]
            axes[0].plot(s.buffer, s.cost_eur / 1e3, marker="o", ms=3, label=name)
            axes[1].plot(s.buffer, s.walks_per_1000, marker="o", ms=3, label=name)
        axes[0].set(
            title=f"Choose period ({title}): cost vs constant buffer",
            xlabel="buffer (rooms)",
            ylabel="cost (EUR thousand)",
        )
        axes[1].axhline(ceiling, color="crimson", ls="--", label="ceiling")
        axes[1].set(
            title="Walks per 1,000 vs constant buffer",
            xlabel="buffer (rooms)",
            ylabel="walked guests per 1,000 arrivals",
        )
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Wrote %s", out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
        "Could not draw the buffer chart (%s); continuing",
        exc,
        )


def freeze_variant(slug: str, params: dict) -> dict:
    """Fit-free: load the saved models, score the choose period, pick and freeze buffers."""
    K = load_contract()
    choose = load_period(slug, "choose")
    methods = load_methods(MODELS / slug)
    grid = build_grid(params)

    nightly = nightly_tables(methods, choose)
    sweeps = {name: sweep(nt, grid) for name, nt in nightly.items()}

    frozen, rows = {}, []
    for name, sw in sweeps.items():
        best, met = pick(sw)
        frozen[name] = {
            "family": str(best.family),
            "buffer": float(best.buffer),
            "ceiling_met_on_choose": bool(met),
        }
        rows.append(
            {
                "method": name,
                **frozen[name],
                "cost_eur": best.cost_eur,
                "empty_rooms": best.empty_rooms,
                "walked": best.walked_guests,
                "walks_per_1000": best.walks_per_1000,
            }
        )
        logger.info(
            "[%s] %s: %s %g, cost EUR %.0f, walks %.2f per 1,000",
            slug,
            name,
            best.family,
            best.buffer,
            best.cost_eur,
            best.walks_per_1000,
        )
        if on_grid_edge(best.family, float(best.buffer), grid):
            logger.warning(
                "[%s] %s: chosen buffer %s %g sits at an edge of the search grid; "
                "the true optimum may lie outside it",
                slug,
                name,
                best.family,
                best.buffer,
            )
        if not met:
            logger.warning(
                "[%s] %s: no buffer in the grid meets the ceiling on the choose period",
                slug,
                name,
            )
    choose_tbl = pd.DataFrame(rows).set_index("method")
    headline = choose_headline(choose_tbl)

    write_json(
        {
            "variant": slug,
            "chosen_on": f"{K.TRAIN_END} to {K.PROOF_START}",
            "headline_model": headline,
            "buffers": frozen,
        },
        frozen_path(slug),
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    choose_tbl.to_csv(REPORTS / f"choose_table_{slug}.csv")
    plot_sweeps(
        sweeps,
        K.WALK_CEILING,
        VARIANT_LABELS[slug],
        REPORTS / f"buffer_sweep_{slug}.png",
    )
    logger.info(
        "[%s] buffers frozen -> %s | headline model: %s",
        slug,
        frozen_path(slug).name,
        headline,
    )
    return {"frozen": frozen, "headline": headline, "choose_tbl": choose_tbl}


def _buf(entry: dict) -> str:
    return f"{entry['family']} {entry['buffer']:g}"


def compare_variants(results: dict, primary: str) -> tuple[pd.DataFrame, bool]:
    """Do the buffers, and the headline recommendation, move between the two variants?"""
    other = next((s for s in results if s != primary), None)
    if other is None:
        return pd.DataFrame(), False
    first, second = results[primary], results[other]
    table = pd.DataFrame(
        {
            f"buffer ({primary})": {
                n: _buf(first["frozen"][n]) for n in first["frozen"]
            },
            f"buffer ({other})": {
                n: _buf(second["frozen"][n]) for n in second["frozen"]
            },
            f"choose cost ({primary}, EUR)": first["choose_tbl"].cost_eur,
            f"choose cost ({other}, EUR)": second["choose_tbl"].cost_eur,
        }
    )
    table["buffer moved?"] = table.iloc[:, 0] != table.iloc[:, 1]
    flips = not (
        first["headline"] == second["headline"]
        and _buf(first["frozen"][first["headline"]])
        == _buf(second["frozen"][second["headline"]])
    )
    return table, flips


@log_stage("freeze_buffer")
def main() -> None:
    params = load_params()
    slugs = variant_slugs(params)
    primary = params["variants"]["primary"]
    results = {slug: freeze_variant(slug, params) for slug in slugs}

    table, flips = compare_variants(results, primary)
    if not table.empty:
        table.to_csv(REPORTS / "variant_comparison.csv")
        logger.info("variant comparison:\n%s", table.round(0).to_string())
        if flips:
            logger.warning(
                "THE BUFFER RECOMMENDATION FLIPS between the variants: %s",
                {s: r["headline"] for s, r in results.items()},
            )
        else:
            logger.info(
                "The buffer recommendation does NOT flip: same headline model and buffer "
                "rule on both variants (%s)",
                results[primary]["headline"],
            )
    if FLAT not in results[primary]["frozen"]:
        raise ValueError("the flat rule must be among the frozen methods")


if __name__ == "__main__":
    run_cli(main)
