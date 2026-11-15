"""Stage 1: validate the raw bookings, flag repeats, apply the cleaning decisions.

Writes data/interim/clean.parquet with EVERY booking plus the is_dup and row_key columns.
Notebook cells: 5, 7, 8 (check the brief, duplicates), 11 (cleaning and decisions log),
13 (column audit assertion). Nothing is deleted at loading.
"""

from __future__ import annotations

import pandas as pd

from src.data import TARGET, arrival_date, load_contract, read_raw, write_parquet
from src.features import AUDIT
from src.logger import get_logger, log_stage, run_cli
from src.params import INTERIM, REPORTS, load_params, require_params

logger = get_logger(__name__)

WARN_SHARE = 0.01  # warn when one cleaning choice touches more than 1% of rows


class DataValidationError(ValueError):
    """The raw data does not look like the dataset this project expects."""


def check_columns(frame: pd.DataFrame) -> None:
    """Every audited column must exist, and every column must have an audit bin."""
    binned = {c for cols in AUDIT.values() for c in cols}
    required = binned | {
        "arrival_date_month",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        TARGET,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        logger.error("Data is missing columns: %s", missing)
        raise DataValidationError(f"data is missing columns: {missing}")
    unbinned = [c for c in frame.columns if c not in binned]
    if unbinned:
        logger.error("Columns without an audit bin: %s", unbinned)
        raise DataValidationError(f"columns without an audit bin: {unbinned}")
    logger.info("Column audit passed: every column has a bin")


def flag_duplicates(raw: pd.DataFrame) -> pd.DataFrame:
    """Add is_dup (True for every repeat after the first copy) and row_key (hash of all columns
    except the outcome). Nothing is dropped."""
    out = raw.copy()
    key_cols = [c for c in raw.columns if c != TARGET]
    out["is_dup"] = raw.duplicated(keep="first")
    out["row_key"] = pd.util.hash_pandas_object(raw[key_cols], index=False)
    mixed = int((out.groupby("row_key")[TARGET].nunique() > 1).sum())
    if mixed:
        logger.warning(
            "%d groups of identical rows disagree on the outcome; check the key columns",
            mixed,
        )
    logger.info(
        "Flagged %d repeated rows of %d (%.1f%%); none removed",
        int(out.is_dup.sum()),
        len(out),
        100 * out.is_dup.mean(),
    )
    return out


def duplicate_report(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables that show how much repeats and whether repeating looks different (cells 7 and 8)."""
    dups = df.groupby("hotel").agg(
        bookings=("is_dup", "size"), flagged_dup=("is_dup", "sum")
    )
    dups.loc["All"] = dups.sum()
    dups["share_dup"] = dups.flagged_dup / dups.bookings

    base = pd.Series(
        {
            "raw (every row)": df[TARGET].mean(),
            "deduplicated (first copies only)": df.loc[~df.is_dup, TARGET].mean(),
            "the flagged repeats alone": df.loc[df.is_dup, TARGET].mean(),
        },
        name="cancellation base rate",
    ).to_frame()

    nr = df.deposit_type == "Non Refund"
    facts = pd.Series(
        {
            "cancel rate, repeats only": df.loc[df.is_dup, TARGET].mean(),
            "cancel rate, first copies": df.loc[~df.is_dup, TARGET].mean(),
            "share of Non Refund rows that are repeats": df.loc[nr, "is_dup"].mean(),
            "share Non Refund, raw": nr.mean(),
            "share Non Refund, deduplicated": (
                df.loc[~df.is_dup, "deposit_type"] == "Non Refund"
            ).mean(),
            "cancel rate among Non Refund, raw": df.loc[nr, TARGET].mean(),
            "cancel rate among Non Refund, deduplicated": df.loc[
                nr & ~df.is_dup, TARGET
            ].mean(),
        },
        name="value",
    ).to_frame()
    for key, row in facts.iterrows():
        logger.info("duplicate fact | %s | %.3f", key, row["value"])
    return {"duplicates_by_hotel": dups, "base_rates": base, "dup_facts": facts}


def brief_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Compare what the brief states with what the data (and contract) say."""
    rows = [
        ("Bookings", "119,390", f"{len(df):,}"),
        (
            "Arrival range",
            "July 2015 to August 2017",
            f"{df.arrival_date.min():%B %Y} to {df.arrival_date.max():%B %Y}",
        ),
        (
            "Hotels",
            "Lisbon city hotel + Algarve resort",
            ", ".join(sorted(df.hotel.unique())),
        ),
    ]
    try:
        K = load_contract()
        rows.append(
            (
                "Ceiling",
                "'one guest in a hundred' = 10 per 1,000",
                f"contract WALK_CEILING = {K.WALK_CEILING:g} per 1,000",
            )
        )
        rows.append(("Walk cost (EUR)", "300", f"{K.WALK_COST:g}"))
    except ModuleNotFoundError:
        logger.warning("contract.py not found; ceiling and walk-cost checks skipped")
    checks = pd.DataFrame(rows, columns=["item", "brief says", "data / contract says"])
    for _, r in checks.iterrows():
        logger.info(
            "brief check | %s | brief: %s | data: %s",
            r["item"],
            r["brief says"],
            r["data / contract says"],
        )
    return checks


def clean(df_in: pd.DataFrame, adr_cap: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repair only impossible values; keep every booking. Returns (clean frame, decisions log)."""
    decisions: list[dict] = []

    def decide(name: str, rows: int, why: str, warn: bool = True) -> None:
        decisions.append({"decision": name, "rows_touched": int(rows), "why": why})
        logger.info("decision: %s (%d rows)", name, rows)
        if warn and rows > WARN_SHARE * len(df_in):
            logger.warning(
                "decision '%s' touches %.1f%% of rows; check the data",
                name,
                100 * rows / len(df_in),
            )

    df = df_in.copy()
    n0 = len(df)

    df["nights"] = df.stays_in_weekend_nights + df.stays_in_week_nights
    decide(
        "nights = weekend + week nights",
        len(df),
        "contract.py needs a 'nights' column",
        warn=False,
    )

    neg = df.adr < 0
    df.loc[neg, "adr"] = 0.0
    decide("adr < 0 set to 0", neg.sum(), "a negative nightly rate is not a price")

    big = df.adr > adr_cap
    df.loc[big, "adr"] = float(adr_cap)
    decide(
        f"adr > {adr_cap:,.0f} capped",
        big.sum(),
        "one extreme rate would dominate a night's average stay value",
    )

    nokid = df.children.isna()
    df["children"] = df.children.fillna(0)
    decide(
        "children missing -> 0",
        nokid.sum(),
        "blank child count; adults are listed separately",
    )

    for col in ["agent", "company", "country"]:
        missing = df[col].isna()
        df[col] = df[col].astype(object)
        df.loc[missing, col] = "none"
        df[col] = df[col].astype(str)  # ids arrive as floats; text keeps parquet happy
        decide(
            f"{col} missing -> 'none'",
            missing.sum(),
            "missing is a category, not an error",
            warn=False,
        )

    zero = df.nights == 0
    decide(
        "nights == 0 kept (flagged only)",
        zero.sum(),
        "they carry stay value 0, so an empty room from one costs nothing in the client's rule",
        warn=False,
    )
    if "is_dup" in df.columns:
        decide(
            "is_dup flagged, nothing dropped at load",
            df.is_dup.sum(),
            "whether repeats are real bookings is a modelling decision (the two variants)",
            warn=False,
        )

    if len(df) != n0:
        raise DataValidationError(f"cleaning dropped rows: {n0} -> {len(df)}")
    decide(
        "rows dropped at load",
        0,
        "none; the client scores every booking due that night",
        warn=False,
    )
    return df.reset_index(drop=True), pd.DataFrame(decisions)


@log_stage("validate")
def main() -> None:
    params = load_params()
    require_params(
        params, "cleaning.adr_cap", "data.expected_rows", "data.expected_cols"
    )

    raw = read_raw()
    df = flag_duplicates(raw)  # row_key must be computed before arrival_date is added
    df["arrival_date"] = arrival_date(df)
    check_columns(df)

    checks = brief_checks(df)
    report = duplicate_report(df)
    df, decisions = clean(df, params["cleaning"]["adr_cap"])
    logger.info("Rows before: %d, rows after: %d", len(raw), len(df))

    write_parquet(df, INTERIM / "clean.parquet")
    try:
        REPORTS.mkdir(parents=True, exist_ok=True)
        checks.to_csv(REPORTS / "brief_checks.csv", index=False)
        decisions.to_csv(REPORTS / "decisions_log.csv", index=False)
        for name, table in report.items():
            table.to_csv(REPORTS / f"{name}.csv")
    except OSError as exc:
        logger.error("Could not write validation reports: %s", exc)
        raise
    logger.info("Validation reports written to %s", REPORTS)


if __name__ == "__main__":
    run_cli(main)
