"""Tests for src/features.py: only columns known at booking may feed a method."""
from __future__ import annotations

import pandas as pd
import pytest

from src import features
from src.features import (
    AUDIT,
    BOOKKEEPING,
    CAT,
    EXCLUDED,
    FEATURES,
    NON_FEATURES,
    NUM,
    RarePooler,
    assert_no_leakage,
)


def test_no_excluded_column_is_a_feature():
    assert not set(FEATURES) & set(EXCLUDED)


def test_bookkeeping_columns_are_never_features():
    assert set(BOOKKEEPING) == {"arrival_date_year", "arrival_date", "is_dup", "row_key"}
    assert not set(FEATURES) & set(BOOKKEEPING)
    assert not set(FEATURES) & set(NON_FEATURES)


def test_outcome_columns_are_never_features():
    for col in ["is_canceled", "reservation_status", "reservation_status_date"]:
        assert col not in FEATURES


def test_post_booking_columns_are_excluded():
    for col in ["booking_changes", "required_car_parking_spaces",
                "total_of_special_requests", "assigned_room_type"]:
        assert col in EXCLUDED and col not in FEATURES


def test_features_are_exactly_the_known_at_booking_bin():
    assert FEATURES == AUDIT["known at booking"]


def test_categorical_and_numeric_columns_partition_the_features():
    assert set(CAT) | set(NUM) == set(FEATURES)
    assert not set(CAT) & set(NUM)


def test_leakage_check_catches_a_leaking_column(monkeypatch):
    monkeypatch.setattr(features, "FEATURES", [*FEATURES, "reservation_status"])
    with pytest.raises(AssertionError, match="reservation_status"):
        assert_no_leakage()


def test_leakage_check_catches_a_bookkeeping_column(monkeypatch):
    monkeypatch.setattr(features, "FEATURES", [*FEATURES, "is_dup"])
    with pytest.raises(AssertionError, match="is_dup"):
        assert_no_leakage()


def test_every_feature_exists_in_the_raw_layout(make_raw):
    assert set(FEATURES) <= set(make_raw().columns)


def test_pooling_uses_training_counts_only():
    train = pd.DataFrame({c: ["x"] * 12 for c in FEATURES})
    train["country"] = ["PRT"] * 10 + ["GBR"] * 2
    test = train.copy()
    test["country"] = ["ESP"] * 12  # common in test, unseen in train
    pooler = RarePooler({"country": 1}).fit(train)
    assert pooler.keep_["country"] == {"PRT"}
    assert (pooler.transform(test)["country"] == "other").all()


def test_transform_returns_only_the_feature_columns(make_raw):
    raw = make_raw()
    out = RarePooler({"country": 5, "agent": 5, "company": 2}).fit(raw).transform(raw)
    assert list(out.columns) == FEATURES


def test_transform_rejects_a_missing_feature(make_raw):
    raw = make_raw()
    pooler = RarePooler({"country": 5}).fit(raw)
    with pytest.raises(ValueError, match="lead_time"):
        pooler.transform(raw.drop(columns=["lead_time"]))


def test_transform_before_fit_is_an_error(make_raw):
    with pytest.raises(RuntimeError):
        RarePooler({"country": 5}).transform(make_raw())


def test_fit_on_empty_frame_is_rejected(make_raw):
    with pytest.raises(ValueError):
        RarePooler({"country": 5}).fit(make_raw().iloc[0:0])
