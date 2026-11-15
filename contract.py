"""The client's scoring rule, as code. Read it; do not change it.

The hotel group judges any oversell policy the same way, whichever method produced it, so every team
is scored on the same terms. This file holds that rule and nothing else: it does not choose your
bookings, your columns, your method or your buffer. Those are your team's decisions.

The rule in words (docs/EVALUATION_CONTRACT.md says the same thing):
  - The unit is one hotel on one arrival night. A night's book is every booking due to arrive that night.
  - Your policy sells k extra rooms for that night. If more bookings cancel than k, the difference
    stays empty; if fewer cancel than k, the difference is guests walked to another hotel.
  - An empty room costs the night's average booked stay value (ADR x nights of that night's bookings).
  - A walked guest costs WALK_COST euros.
  - A policy is acceptable only if it walks at most WALK_CEILING guests per 1,000 arrivals.
  - Train on arrivals before TRAIN_END, choose on arrivals from TRAIN_END to PROOF_START, and prove
    once on arrivals from PROOF_START. Use the proof period once, at the end.
"""

import numpy as np
import pandas as pd

WALK_COST = 300.0  # euros per walked guest, stated by the client
WALK_CEILING = 10.0  # walked guests per 1,000 arrivals, the client's brand standard
TRAIN_END = "2016-07-01"  # train on arrivals before this date
PROOF_START = (
    "2017-01-01"  # choose on arrivals before this date, prove once on arrivals from it
)

REQUIRED = ["hotel", "arrival_date", "is_canceled", "adr", "nights"]


def split_periods(df):
    """Return (train, choose, proof) by arrival date. df needs an 'arrival_date' datetime column."""
    train = df[df["arrival_date"] < TRAIN_END]
    choose = df[(df["arrival_date"] >= TRAIN_END) & (df["arrival_date"] < PROOF_START)]
    proof = df[df["arrival_date"] >= PROOF_START]
    return train, choose, proof


def nightly_table(bookings, prob):
    """One row per hotel and arrival night.

    bookings: the bookings you are scoring, with the REQUIRED columns.
    prob: your probability that each booking cancels, in the same row order as bookings.
    Returns columns: book (bookings due), cancelled (what actually happened), expected (the sum of
    your probabilities), spread (the standard deviation of that sum if bookings cancel independently)
    and stay_value (the night's average ADR x nights).
    """
    missing = [c for c in REQUIRED if c not in bookings.columns]
    if missing:
        raise ValueError(f"bookings is missing columns: {missing}")
    p = np.asarray(prob, dtype=float)
    if len(p) != len(bookings):
        raise ValueError("prob must have one value per booking, in the same order")
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("every probability must sit between 0 and 1")
    f = bookings.assign(
        _p=p, _var=p * (1 - p), _value=bookings["adr"] * bookings["nights"]
    )
    out = f.groupby(["hotel", "arrival_date"]).agg(
        book=("is_canceled", "size"),
        cancelled=("is_canceled", "sum"),
        expected=("_p", "sum"),
        spread=("_var", lambda v: float(np.sqrt(v.sum()))),
        stay_value=("_value", "mean"),
    )
    return out


def night_cost(nights, oversell):
    """Score an oversell count per night against what actually happened.

    nights: a table from nightly_table. oversell: rooms sold beyond capacity, one value per night
    (whole numbers, zero or more). Returns the total cost in euros, the empty rooms, the walked
    guests and the walks per 1,000 arrivals, and whether the ceiling was met.
    """
    k = pd.Series(np.asarray(oversell, dtype=float), index=nights.index)
    if (k < 0).any():
        raise ValueError("oversell counts cannot be negative")
    if (k != k.round()).any():
        raise ValueError("oversell counts are whole rooms; round them before scoring")
    empty = (nights["cancelled"] - k).clip(lower=0)
    walks = (k - nights["cancelled"]).clip(lower=0)
    arrivals = (nights["book"] - nights["cancelled"]).sum()
    walk_rate = 1000.0 * walks.sum() / arrivals if arrivals else float("nan")
    cost = float((empty * nights["stay_value"]).sum() + (walks * WALK_COST).sum())
    return {
        "cost_eur": round(cost, 2),
        "empty_rooms": int(empty.sum()),
        "walked_guests": int(walks.sum()),
        "walks_per_1000": round(float(walk_rate), 2),
        "ceiling_met": bool(walk_rate <= WALK_CEILING),
        "nights": len(nights),
    }


if __name__ == "__main__":
    print(__doc__)

# Test cases (run: pytest -q, which reads tests/test_contract.py)
# 1. One night of 4 bookings, 2 cancel, stay value EUR 200, oversell 1 -> 1 empty room, 0 walks, EUR 200.
# 2. The same night, oversell 3 -> 0 empty rooms, 1 walked guest, EUR 300.
# 3. Add a night of 2 bookings with none cancelling and oversell 1 -> 2 walks over 4 arrivals,
#    500 per 1,000, ceiling not met.
# 4. A probability of 1.2, one probability too few, a negative oversell or 1.5 rooms -> ValueError.
