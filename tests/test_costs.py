"""Tests for src/costs.py: the oversell rule and the euro comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.costs import (
    REF_ZERO,
    bootstrap_savings,
    night_cost_vector,
    oversell,
    proof_table,
)


def _nightly() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["City Hotel"], pd.date_range("2017-03-01", periods=4)],
        names=["hotel", "date"],
    )
    return pd.DataFrame(
        {
            "expected": [10.0, 3.2, 0.4, 20.0],
            "spread": [2.0, 1.0, 0.5, 3.0],
            "cancelled": [9, 3, 0, 15],
            "book": [12, 3, 5, 8],
            "stay_value": [200.0, 150.0, 100.0, 180.0],
        },
        index=index,
    )


def test_constant_buffer_formula():
    k = oversell(_nightly(), "constant", 2)
    assert list(k) == [
        8,
        1,
        0,
        8,
    ]  # floor(expected - 2), floored at 0, capped at bookings


def test_spread_buffer_formula():
    k = oversell(_nightly(), "spread", 1)
    assert list(k) == [8, 2, 0, 8]  # floor(expected - 1 x spread)


def test_zero_buffer_is_floor_of_expected():
    nt = _nightly()
    k = oversell(nt, "constant", 0)
    assert list(k) == [
        int(np.floor(e)) if e < b else b
        for e, b in zip(nt.expected, nt.book, strict=True)
    ]


def test_never_negative_and_never_above_bookings():
    nt = _nightly()
    for family, value in [
        ("constant", 0),
        ("constant", 5),
        ("spread", 0),
        ("spread", 3),
    ]:
        k = oversell(nt, family, value)
        assert (k >= 0).all() and (k <= nt.book).all()


def test_a_bigger_buffer_never_oversells_more():
    nt = _nightly()
    for family in ("constant", "spread"):
        small, large = oversell(nt, family, 1), oversell(nt, family, 3)
        assert (large <= small).all()


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="family"):
        oversell(_nightly(), "magic", 1)


def test_negative_buffer_is_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        oversell(_nightly(), "constant", -1)


def test_missing_nightly_columns_are_rejected():
    with pytest.raises(ValueError, match="spread"):
        oversell(_nightly().drop(columns=["spread"]), "constant", 1)


def test_empty_nightly_table_is_rejected():
    with pytest.raises(ValueError, match="no nights"):
        oversell(_nightly().iloc[0:0], "constant", 1)


# ---- tests that use the client's contract (skipped if contract.py is not in the repo) ----------


def test_client_rule_constants(contract):
    assert contract.WALK_COST == 300
    assert contract.WALK_CEILING == 10


def test_cost_vector_adds_up_to_the_contract_cost(contract):
    nt = _nightly()
    k = oversell(nt, "constant", 1)
    expected = contract.night_cost(nt, k)["cost_eur"]
    assert night_cost_vector(nt, "constant", 1).sum() == pytest.approx(expected)


def test_proof_table_needs_buffers_for_every_method(contract):
    nightly = {"1 Flat rule": _nightly(), "2 Logistic": _nightly()}
    frozen = {"1 Flat rule": {"family": "constant", "buffer": 1.0}}
    with pytest.raises(ValueError, match="do not match"):
        proof_table(nightly, frozen, "1 Flat rule")


def test_proof_table_needs_the_baseline(contract):
    nightly = {"2 Logistic": _nightly()}
    frozen = {"2 Logistic": {"family": "constant", "buffer": 1.0}}
    with pytest.raises(ValueError, match="baseline"):
        proof_table(nightly, frozen, "1 Flat rule")


def test_proof_table_savings_are_measured_against_the_flat_rule(contract):
    nightly = {"1 Flat rule": _nightly(), "2 Logistic": _nightly()}
    frozen = {
        "1 Flat rule": {"family": "constant", "buffer": 1.0},
        "2 Logistic": {"family": "constant", "buffer": 1.0},
    }
    tbl = proof_table(nightly, frozen, "1 Flat rule")
    assert tbl.loc["1 Flat rule", "saving_vs_flat_eur"] == 0
    assert tbl.loc["2 Logistic", "saving_vs_flat_eur"] == pytest.approx(0)
    assert REF_ZERO in tbl.index


def test_bootstrap_rejects_too_few_resamples(contract):
    nightly = {"1 Flat rule": _nightly(), "2 Logistic": _nightly()}
    frozen = {n: {"family": "constant", "buffer": 1.0} for n in nightly}
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_savings(nightly, frozen, "1 Flat rule", n_boot=10, seed=0)
