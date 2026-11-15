"""Stage 2: split by arrival date into train / choose / proof, for each data variant.

Variants: dedup = first copy of every repeated row (primary); raw = every row (sensitivity).
Notebook cells: 16 (split, straddle check, the two variants). The dates come from the client's
contract.split_periods; the split is never random.
"""

from __future__ import annotations

import pandas as pd

from src.data import TARGET, load_contract, read_parquet_checked, write_parquet
from src.logger import get_logger, log_dataframe_summary, log_stage, run_cli
from src.params import (
    INTERIM,
    PROCESSED,
    REPORTS,
    VARIANT_LABELS,
    load_params,
    variant_slugs,
)

logger = get_logger(__name__)

RATE_GAP_WARN = 0.05  # warn when cancellation rates differ this much between periods


def check_split(
    df: pd.DataFrame, train: pd.DataFrame, choose: pd.DataFrame, proof: pd.DataFrame
) -> None:
    """Raise if the periods overlap, leave rows out, or are empty."""
    for name, part in [("train", train), ("choose", choose), ("proof", proof)]:
        if part.empty:
            logger.error("The %s period is empty", name)
            raise ValueError(f"the {name} period is empty")
    if len(train) + len(choose) + len(proof) != len(df):
        raise ValueError(
            "split lost or duplicated bookings: "
            f"{len(train)} + {len(choose)} + {len(proof)} != {len(df)}"
        )
    if not (
        train.arrival_date.max() < choose.arrival_date.min()
        and choose.arrival_date.max() < proof.arrival_date.min()
    ):
        raise ValueError("periods overlap in time; the split must be by arrival date")

    rates = {
        n: float(p[TARGET].mean())
        for n, p in [("train", train), ("choose", choose), ("proof", proof)]
    }
    logger.info("cancellation rates: %s", {k: round(v, 3) for k, v in rates.items()})
    if max(rates.values()) - min(rates.values()) > RATE_GAP_WARN:
        logger.warning(
            "cancellation rates differ by more than %.0f points between periods; "
            "expect drift and do not tune the buffer to one level",
            100 * RATE_GAP_WARN,
        )


def check_no_group_straddles(
    train: pd.DataFrame, choose: pd.DataFrame, proof: pd.DataFrame
) -> None:
    """Identical rows must never sit in two periods (arrival_date is part of the key)."""
    keys = {
        "train": set(train.row_key),
        "choose": set(choose.row_key),
        "proof": set(proof.row_key),
    }
    straddle = {
        f"{a}/{b}": len(keys[a] & keys[b])
        for a, b in [("train", "choose"), ("train", "proof"), ("choose", "proof")]
    }
    logger.info("identical-row groups that straddle two periods: %s", straddle)
    if any(straddle.values()):
        raise ValueError(f"a group of identical rows spans two periods: {straddle}")


def variant_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """dedup = first copies only (primary); raw = every row. Nothing was deleted at load."""
    return {"dedup": df.loc[~df.is_dup].reset_index(drop=True), "raw": df}


@log_stage("split")
def main() -> None:
    params = load_params()
    slugs = variant_slugs(params)
    df = read_parquet_checked(
        INTERIM / "clean.parquet",
        ["arrival_date", TARGET, "hotel", "is_dup", "row_key"],
    )
    K = load_contract()

    try:
        t_all, c_all, p_all = K.split_periods(df)
    except Exception:
        logger.error("contract.split_periods failed on %d rows", len(df))
        raise
    check_split(df, t_all, c_all, p_all)
    check_no_group_straddles(t_all, c_all, p_all)

    summary = []
    for slug, frame in variant_frames(df).items():
        if slug not in slugs:
            continue
        label = VARIANT_LABELS[slug]
        train, choose, proof = K.split_periods(frame)
        check_split(frame, train, choose, proof)
        for name, part in [("train", train), ("choose", choose), ("proof", proof)]:
            log_dataframe_summary(
                logger, part, f"{slug}/{name}", date_col="arrival_date"
            )
            write_parquet(part, PROCESSED / slug / f"{name}.parquet")
            summary.append(
                {
                    "variant": label,
                    "period": name,
                    "bookings": len(part),
                    "from": part.arrival_date.min().date(),
                    "to": part.arrival_date.max().date(),
                    "cancel_rate": part[TARGET].mean(),
                }
            )
        summary.append(
            {
                "variant": label,
                "period": "ALL",
                "bookings": len(frame),
                "from": frame.arrival_date.min().date(),
                "to": frame.arrival_date.max().date(),
                "cancel_rate": frame[TARGET].mean(),
            }
        )
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(REPORTS / "split_summary.csv", index=False)


if __name__ == "__main__":
    run_cli(main)
