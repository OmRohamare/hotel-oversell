"""Shared test helpers: a synthetic raw dataset with the real 32 columns, and the client's contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

RAW_COLUMNS = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
]


def build_raw(n: int = 3000, seed: int = 0, repeat_share: float = 0.08) -> pd.DataFrame:
    """Synthetic bookings spanning July 2015 to August 2017, with the real column layout.

    The last repeat_share of the rows are exact copies of earlier rows, so is_dup has work to do.
    """
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2015-07-01") + pd.to_timedelta(
        rng.integers(0, 790, n), unit="D"
    )
    df = pd.DataFrame(
        {
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
            "is_canceled": 0,
            "lead_time": rng.integers(0, 400, n),
            "arrival_date_year": dates.year,
            "arrival_date_month": [MONTHS[m - 1] for m in dates.month],
            "arrival_date_week_number": dates.isocalendar().week.to_numpy(),
            "arrival_date_day_of_month": dates.day,
            "stays_in_weekend_nights": rng.integers(0, 3, n),
            "stays_in_week_nights": rng.integers(0, 6, n),
            "adults": 2,
            "children": np.where(rng.random(n) < 0.01, np.nan, 0.0),
            "babies": 0,
            "meal": rng.choice(["BB", "HB"], n),
            "country": rng.choice(
                ["PRT", "GBR", "FRA", None] + [f"C{i}" for i in range(40)], n
            ),
            "market_segment": rng.choice(["Online TA", "Direct", "Groups"], n),
            "distribution_channel": rng.choice(["TA/TO", "Direct"], n),
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": rng.choice(["A", "D"], n),
            "assigned_room_type": "A",
            "booking_changes": 0,
            "deposit_type": rng.choice(["No Deposit", "Non Refund"], n),
            "agent": rng.choice(
                [9.0, 240.0, np.nan] + [float(i) for i in range(50)], n
            ),
            "company": rng.choice([np.nan, 45.0, 67.0], n),
            "days_in_waiting_list": 0,
            "customer_type": rng.choice(["Transient", "Contract"], n),
            "adr": rng.normal(100, 30, n).clip(1),
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
            "reservation_status": "Check-Out",
            "reservation_status_date": "2016-01-01",
        }
    )
    score = (
        df.lead_time / 400
        + (df.deposit_type == "Non Refund") * 0.5
        + rng.normal(0, 0.3, n)
    )
    df["is_canceled"] = (score > 0.6).astype(int)
    df = df[RAW_COLUMNS]
    n_rep = int(n * repeat_share)
    if n_rep:
        copies = df.iloc[:n_rep].copy()
        df = pd.concat([df.iloc[: n - n_rep], copies], ignore_index=True)
    return df


@pytest.fixture(scope="session")
def make_raw():
    """Returns the builder function: make_raw(n=3000, seed=0)."""
    return build_raw


@pytest.fixture(scope="session")
def contract():
    """The client's contract module; tests that need it are skipped if it is not in the repo."""
    from src.data import load_contract

    try:
        return load_contract()
    except ModuleNotFoundError:
        pytest.skip("contract.py not found (repo root or src/)")
