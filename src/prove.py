"""Stage 5: score the PROOF period once per variant, with the buffers frozen in the previous stage.

Notebook cells: 29 (proof tables), 31 and 32 (headline and per-hotel views), 35 (bootstrap),
38 (decision record), 39 (MLflow). Nothing here may change a buffer, a feature or a method.
Every run is appended to docs/PROVE_RUNS.txt so each look at the proof period is on record.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess

import pandas as pd

from src.backtest import FLAT, load_methods, nightly_tables
from src.costs import bootstrap_savings, per_hotel_table, proof_table
from src.data import TARGET, load_contract, load_period, read_json
from src.logger import get_logger, log_stage, run_cli
from src.params import (
    MODELS,
    PROVE_RUNS_PATH,
    REPORTS,
    ROOT,
    VARIANT_LABELS,
    frozen_path,
    load_params,
    proof_table_path,
    require_params,
    variant_slugs,
)

logger = get_logger(__name__)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def record_run(hashes: dict[str, str]) -> int:
    """Append one line to PROVE_RUNS.txt and return how many proof runs existed before."""
    try:
        PROVE_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        previous = 0
        if PROVE_RUNS_PATH.exists():
            previous = sum(
                1
                for line in PROVE_RUNS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            )
        shown = ", ".join(f"{k}={v}" for k, v in hashes.items())
        with PROVE_RUNS_PATH.open("a", encoding="utf-8") as f:
            f.write(
                f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} | commit {_git_commit()} "
                f"| frozen sha256 {shown}\n"
            )
        return previous
    except OSError as exc:
        logger.warning(
            "Could not update %s (%s); the proof run is not recorded",
            PROVE_RUNS_PATH,
            exc,
        )
        return 0


def prove_variant(slug: str, params: dict) -> dict:
    """Score the proof period for one variant using exactly what was frozen on disk."""
    frozen_file = read_json(frozen_path(slug))
    frozen = frozen_file["buffers"]
    methods = load_methods(MODELS / slug)
    proof = load_period(slug, "proof")
    nightly = nightly_tables(methods, proof)

    tbl = proof_table(nightly, frozen, FLAT)
    hotels = per_hotel_table(nightly, frozen, FLAT)
    tbl.to_csv(proof_table_path(slug))
    hotels.to_csv(REPORTS / f"poc_per_hotel_{slug}.csv")
    logger.info(
        "[%s] proof table written -> %s\n%s",
        slug,
        proof_table_path(slug).name,
        tbl.round(2).to_string(),
    )

    headline = frozen_file["headline_model"]
    if headline not in tbl.index:
        raise ValueError(f"headline model {headline!r} is not in the proof table")
    if not hotels.loc[headline].ceiling_met_at_hotel.all():
        logger.warning(
            "[%s] %s meets the ceiling pooled but not at every hotel; confirm whether "
            "the client means 'at each hotel'",
            slug,
            headline,
        )
    return {
        "slug": slug,
        "frozen": frozen,
        "headline": headline,
        "tbl": tbl,
        "hotels": hotels,
        "nightly": nightly,
        "proof": proof,
    }


def headline_table(results: dict) -> pd.DataFrame:
    """One line per variant: base rate, headline model, frozen buffer, proof outcome (cell 31)."""
    rows = []
    for slug, res in results.items():
        h = res["headline"]
        r = res["tbl"].loc[h]
        train = load_period(slug, "train")
        rows.append(
            {
                "variant": VARIANT_LABELS[slug],
                "base rate (train)": train[TARGET].mean(),
                "base rate (proof)": res["proof"][TARGET].mean(),
                "headline model": h,
                "frozen buffer": r.buffer,
                "proof walks/1000": r.walks_per_1000,
                "ceiling met": bool(r.ceiling_met),
                "saving vs flat %": r["saving_vs_flat_%"],
            }
        )
    return pd.DataFrame(rows).set_index("variant")


def ranking(res: dict) -> list[str]:
    t = res["tbl"].drop(index=[i for i in res["tbl"].index if i.startswith("(ref)")])
    return list(t[t.ceiling_met].sort_values("cost_eur").index)


def decision_entry(results: dict, head: pd.DataFrame, primary: str) -> str | None:
    """Text for docs/DECISIONS.md, generated from this run (notebook cell 38)."""
    try:
        by_hotel = pd.read_csv(REPORTS / "duplicates_by_hotel.csv", index_col=0)
        facts = pd.read_csv(REPORTS / "dup_facts.csv", index_col=0)["value"]
    except FileNotFoundError:
        logger.warning(
            "duplicate reports not found; the decision record is not written "
            "(run the validate stage first)"
        )
        return None
    others = [s for s in results if s != primary]
    if not others:
        return None
    D, R = results[primary], results[others[0]]

    def fb(res: dict) -> str:
        entry = res["frozen"][res["headline"]]
        return f"{entry['family']} {entry['buffer']:g}"

    flips = not (D["headline"] == R["headline"] and fb(D) == fb(R))
    n_dup, n_all = (
        int(by_hotel.loc["All", "flagged_dup"]),
        int(by_hotel.loc["All", "bookings"]),
    )
    hp, hr = head.iloc[0], head.iloc[1]
    moved = [
        n
        for n in D["frozen"]
        if f"{D['frozen'][n]['family']} {D['frozen'][n]['buffer']:g}"
        != f"{R['frozen'][n]['family']} {R['frozen'][n]['buffer']:g}"
    ]
    return f"""## Decision: repeated bookings are flagged at load, deduplicated for modelling, and reported both ways

**Date:** {pd.Timestamp.today():%Y-%m-%d}   **Status:** open until the client says whether repeats are real bookings

**What we found.** {n_dup:,} of {n_all:,} rows ({n_dup / n_all:.1%}) are exact repeats of an earlier row in every column the export holds. `is_dup` marks them; nothing is deleted at loading.

**What we chose.** The deduplicated data is the primary training and scoring set. `row_key` (hash of every column except `is_canceled`) groups identical rows. The split stays by arrival date, so identical rows always fall in the same period (checked in the split stage).

**What it is worth (same code and seed on each variant).**

| | Deduplicated (primary) | Raw |
|---|---|---|
| Base rate, train period | {hp["base rate (train)"]:.1%} | {hr["base rate (train)"]:.1%} |
| Headline model and buffer | {D["headline"]} with {fb(D)} | {R["headline"]} with {fb(R)} |
| Proof: walks per 1,000 / ceiling met | {hp["proof walks/1000"]:.2f} / {hp["ceiling met"]} | {hr["proof walks/1000"]:.2f} / {hr["ceiling met"]} |
| Proof: saving vs the flat rule | {hp["saving vs flat %"]:.1f}% | {hr["saving vs flat %"]:.1f}% |

**What the repeats look like.** Repeated rows cancel at {facts.loc["cancel rate, repeats only"]:.1%} against {facts.loc["cancel rate, first copies"]:.1%} for first copies, and {facts.loc["share of Non Refund rows that are repeats"]:.1%} of Non Refund rows are repeats.

**Does the buffer recommendation flip?** {"YES. The headline model or its buffer differs between the variants." if flips else "No. The same headline model and buffer rule come out on both variants."} Buffers that differ between variants: {", ".join(moved) if moved else "none"}.

**Still open.** Ask the client whether repeated rows are separate bookings. If yes, the raw variant is closer to their books.
"""


def log_to_mlflow(
    params: dict, tbl: pd.DataFrame, primary: str, hashes: dict[str, str]
) -> None:
    """Optional DagsHub logging for the primary variant; a failure must not lose local results."""
    try:
        from src import tracking as T

        K = load_contract()
        with T.tracked_run(
            run_name="poc-method-table",
            tags={"stage": "POC", "variant": primary},
            experiment_name=params["mlflow"].get("experiment", "hotel-oversell"),
        ):
            T.log_params(
                {
                    "walk_cost": K.WALK_COST,
                    "ceiling": K.WALK_CEILING,
                    "seed": params["seed"],
                    "git_commit": _git_commit(),
                    "frozen_sha256": hashes.get(primary, "unknown"),
                }
            )
            for name, row in tbl.iterrows():
                if name.startswith("(ref)"):
                    continue
                slug = name.split(" ", 1)[0]
                T.log_metrics(
                    {
                        f"m{slug}_cost_eur": row.cost_eur,
                        f"m{slug}_walks_per_1000": row.walks_per_1000,
                    }
                )
            T.log_artifact(proof_table_path(primary))
            T.log_artifact(frozen_path(primary))
        logger.info("Primary proof run logged to MLflow")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
        "MLflow logging failed (%s: %s); results are saved locally",
        type(exc).__name__,
        exc,
        )


@log_stage("prove")
def main() -> None:
    params = load_params()
    require_params(params, "seed", "bootstrap.n_boot")
    slugs = variant_slugs(params)
    primary = params["variants"]["primary"]

    for slug in slugs:
        if not frozen_path(slug).exists():
            logger.error(
                "%s not found. Freeze the buffers on the choose period first.",
                frozen_path(slug),
            )
            raise FileNotFoundError(str(frozen_path(slug)))
    hashes = {slug: _sha256(frozen_path(slug)) for slug in slugs}

    previous = record_run(hashes)
    if previous:
        logger.warning(
            "The proof period has already been scored %d time(s). Changing anything "
            "after seeing a result makes the proof no longer a proof.",
            previous,
        )

    results = {slug: prove_variant(slug, params) for slug in slugs}
    head = headline_table(results)
    head.to_csv(REPORTS / "headline_table.csv")
    logger.info("headline table:\n%s", head.round(2).to_string())
    for slug, res in results.items():
        logger.info(
            "proof ranking, %s (cheapest first): %s", VARIANT_LABELS[slug], ranking(res)
        )
    if len(results) > 1:
        logger.info(
            "same ranking on both variants: %s",
            len({tuple(ranking(r)) for r in results.values()}) == 1,
        )

    res = results[primary]
    boot = bootstrap_savings(
        res["nightly"],
        res["frozen"],
        FLAT,
        params["bootstrap"]["n_boot"],
        params["seed"],
    )
    boot.to_csv(REPORTS / "poc_bootstrap.csv")

    entry = decision_entry(results, head, primary)
    if entry:
        (REPORTS / "DECISIONS_dedup_entry.md").write_text(entry, encoding="utf-8")
        logger.info("Decision record written -> reports/DECISIONS_dedup_entry.md")

    if params.get("mlflow", {}).get("enabled"):
        log_to_mlflow(params, res["tbl"], primary, hashes)


if __name__ == "__main__":
    run_cli(main)
