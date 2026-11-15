"""Tests for src/split.py: the split is by arrival date, never random."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import arrival_date
from src.split import check_no_group_straddles, check_split, variant_frames
from src.validate import flag_duplicates


def _period(
    start: str, end: str, n: int = 50, hotel: str = "City Hotel"
) -> pd.DataFrame:
    dates = pd.date_range(start, end, periods=n)
    return pd.DataFrame(
        {
            "arrival_date": dates,
            "is_canceled": [i % 2 for i in range(n)],
            "hotel": hotel,
        }
    )


@pytest.fixture
def periods():
    train = _period("2015-07-01", "2016-06-30")
    choose = _period("2016-07-01", "2016-12-31")
    proof = _period("2017-01-01", "2017-08-31")
    return train, choose, proof, pd.concat([train, choose, proof], ignore_index=True)


def test_clean_split_passes(periods):
    train, choose, proof, df = periods
    check_split(df, train, choose, proof)


def test_overlapping_periods_are_rejected(periods):
    train, _choose, proof, _df = periods
    overlapping = _period("2016-06-01", "2016-12-31")
    with pytest.raises(ValueError, match="overlap"):
        check_split(pd.concat([train, overlapping, proof]), train, overlapping, proof)


def test_empty_period_is_rejected(periods):
    train, choose, proof, df = periods
    with pytest.raises(ValueError, match="empty"):
        check_split(df, train, choose.iloc[0:0], proof)


def test_lost_rows_are_rejected(periods):
    train, choose, proof, df = periods
    with pytest.raises(ValueError, match="lost or duplicated"):
        check_split(df, train.iloc[:-1], choose, proof)


def test_contract_split_is_ordered_by_date(make_raw, contract):
    raw = make_raw(4000, seed=3)
    raw["arrival_date"] = arrival_date(raw)
    raw["nights"] = raw.stays_in_weekend_nights + raw.stays_in_week_nights
    shuffled = raw.sample(frac=1.0, random_state=1)  # input order must not matter
    train, choose, proof = contract.split_periods(shuffled)
    assert train.arrival_date.max() < choose.arrival_date.min()
    assert choose.arrival_date.max() < proof.arrival_date.min()
    assert len(train) + len(choose) + len(proof) == len(raw)


def test_contract_dates_match_the_project_rule(contract):
    assert str(contract.TRAIN_END) == "2016-07-01"
    assert str(contract.PROOF_START) == "2017-01-01"


def _flagged(make_raw):
    df = flag_duplicates(make_raw(4000, seed=5))
    df["arrival_date"] = arrival_date(df)
    df["nights"] = df.stays_in_weekend_nights + df.stays_in_week_nights
    return df


def test_no_group_of_identical_rows_spans_two_periods(make_raw, contract):
    df = _flagged(make_raw)
    train, choose, proof = contract.split_periods(df)
    check_no_group_straddles(train, choose, proof)  # must not raise


def test_a_straddling_group_is_rejected(make_raw, contract):
    df = _flagged(make_raw)
    train, choose, proof = contract.split_periods(df)
    leaked = train.iloc[[0]].copy()
    with pytest.raises(ValueError, match="spans two periods"):
        check_no_group_straddles(train, pd.concat([choose, leaked]), proof)


def test_variants_are_dedup_primary_and_raw(make_raw):
    df = _flagged(make_raw)
    frames = variant_frames(df)
    assert set(frames) == {"dedup", "raw"}
    assert len(frames["raw"]) == len(df)  # nothing deleted for the raw variant
    assert not frames["dedup"].is_dup.any()  # first copies only
    assert len(frames["dedup"]) == len(df) - int(df.is_dup.sum())
