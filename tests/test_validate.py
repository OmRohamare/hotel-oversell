"""Tests for src/validate.py: duplicate flags, column audit and cleaning decisions."""
from __future__ import annotations

import numpy as np
import pytest

from src.data import arrival_date
from src.validate import (
    DataValidationError,
    check_columns,
    clean,
    duplicate_report,
    flag_duplicates,
)


@pytest.fixture
def prepared(make_raw):
    """Raw bookings with is_dup, row_key and arrival_date added, as the validate stage does."""
    raw = make_raw()
    df = flag_duplicates(raw)
    df["arrival_date"] = arrival_date(df)
    return raw, df


def test_column_audit_passes_on_expected_layout(prepared):
    check_columns(prepared[1])


def test_missing_column_is_rejected(prepared):
    with pytest.raises(DataValidationError, match="lead_time"):
        check_columns(prepared[1].drop(columns=["lead_time"]))


def test_unaudited_column_is_rejected(prepared):
    df = prepared[1].copy()
    df["mystery_column"] = 1
    with pytest.raises(DataValidationError, match="mystery_column"):
        check_columns(df)


def test_flagging_repeats_never_drops_a_row(prepared):
    raw, df = prepared
    assert len(df) == len(raw)
    assert df.is_dup.sum() == raw.duplicated(keep="first").sum() > 0


def test_first_copy_is_not_flagged(prepared):
    _, df = prepared
    firsts = df.groupby("row_key").head(1)
    assert not firsts.is_dup.any()


def test_identical_rows_share_a_key_and_an_outcome(prepared):
    _, df = prepared
    assert (df.groupby("row_key").is_canceled.nunique() == 1).all()
    assert df.row_key.nunique() == len(df) - df.is_dup.sum()


def test_key_ignores_the_outcome_column(make_raw):
    raw = make_raw()
    flipped = raw.copy()
    flipped["is_canceled"] = 1 - flipped["is_canceled"]
    assert (flag_duplicates(raw).row_key == flag_duplicates(flipped).row_key).all()


def test_duplicate_report_has_the_expected_tables(prepared):
    report = duplicate_report(prepared[1])
    assert set(report) == {"duplicates_by_hotel", "base_rates", "dup_facts"}
    by_hotel = report["duplicates_by_hotel"]
    assert by_hotel.loc["All", "bookings"] == len(prepared[1])
    assert "cancel rate, repeats only" in report["dup_facts"].index


def test_cleaning_never_drops_a_booking(prepared):
    _, df = prepared
    out, decisions = clean(df, adr_cap=1000)
    assert len(out) == len(df)
    assert out.is_dup.sum() == df.is_dup.sum()
    last = decisions.iloc[-1]
    assert last["decision"] == "rows dropped at load" and last["rows_touched"] == 0


def test_impossible_adr_values_are_repaired(prepared):
    df = prepared[1].copy()
    df.loc[df.index[0], "adr"] = -5.0
    df.loc[df.index[1], "adr"] = 5400.0
    out, _ = clean(df, adr_cap=1000)
    assert out.adr.min() >= 0
    assert out.adr.max() <= 1000


def test_adr_cap_comes_from_the_parameter(prepared):
    df = prepared[1].copy()
    df.loc[df.index[0], "adr"] = 500.0
    out, _ = clean(df, adr_cap=300)
    assert out.adr.max() == 300


def test_decisions_log_counts_the_rows_touched(prepared):
    df = prepared[1].copy()
    df.loc[df.index[:3], "adr"] = -1.0
    _, decisions = clean(df, adr_cap=1000)
    row = decisions[decisions.decision == "adr < 0 set to 0"].iloc[0]
    assert row.rows_touched == 3


def test_missing_values_become_categories_or_zero(prepared):
    df = prepared[1].copy()
    df.loc[df.index[0], "children"] = np.nan
    out, _ = clean(df, adr_cap=1000)
    assert out.children.isna().sum() == 0
    for col in ["agent", "company", "country"]:
        assert out[col].isna().sum() == 0
        assert out[col].map(type).eq(str).all()  # text keeps parquet writing safe
        assert (out[col] == "none").any()


def test_nights_column_is_added(prepared):
    out, _ = clean(prepared[1], adr_cap=1000)
    assert (out.nights == out.stays_in_weekend_nights + out.stays_in_week_nights).all()


def test_input_frame_is_not_modified(prepared):
    df = prepared[1]
    before = df.copy()
    clean(df, adr_cap=1000)
    assert df.equals(before)
