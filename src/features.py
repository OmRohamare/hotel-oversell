"""Feature definitions: only columns known on the day of booking may be used."""
from __future__ import annotations

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

AUDIT = {
    "known at booking": [
        "hotel", "lead_time", "arrival_date_month", "arrival_date_week_number",
        "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies", "meal", "country", "market_segment",
        "distribution_channel", "is_repeated_guest", "previous_cancellations",
        "previous_bookings_not_canceled", "reserved_room_type", "deposit_type",
        "agent", "company", "days_in_waiting_list", "customer_type", "adr",
    ],
    "changes after booking (excluded to be safe)": [
        "booking_changes", "required_car_parking_spaces",
        "total_of_special_requests", "assigned_room_type",
    ],
    "the outcome itself (never a feature)": [
        "reservation_status", "reservation_status_date", "is_canceled",
    ],
    "bookkeeping, not a feature": ["arrival_date_year", "arrival_date", "is_dup", "row_key"],
}

FEATURES = list(AUDIT["known at booking"])
CAT = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type",
    "agent", "company", "customer_type",
]
NUM = [c for c in FEATURES if c not in CAT]
BOOKKEEPING = list(AUDIT["bookkeeping, not a feature"])
EXCLUDED = [c for key, cols in AUDIT.items()
            if key not in ("known at booking", "bookkeeping, not a feature") for c in cols]
NON_FEATURES = EXCLUDED + BOOKKEEPING


def assert_no_leakage() -> None:
    """Fail loudly if an excluded column has slipped into FEATURES."""
    leaked = sorted(set(FEATURES) & set(NON_FEATURES))
    if leaked:
        logger.error("Leaking columns in FEATURES: %s", leaked)
        raise AssertionError(f"leaking columns in FEATURES: {leaked}")


class RarePooler:
    """Select FEATURES and pool rare categories as 'other', using TRAIN counts only."""

    def __init__(self, top: dict[str, int]):
        self.top = dict(top)
        self.keep_: dict[str, set[str]] = {}

    def fit(self, train: pd.DataFrame) -> RarePooler:
        if train.empty:
            raise ValueError("cannot fit category pooling on an empty training frame")
        missing = [c for c in self.top if c not in train.columns]
        if missing:
            raise ValueError(f"training frame is missing columns {missing}")
        for col, n in self.top.items():
            values = train[col].astype(str)
            self.keep_[col] = set(values.value_counts().nlargest(n).index)
        logger.info("Fitted category pooling on %d training rows", len(train))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURES if c not in frame.columns]
        if missing:
            logger.error("Input is missing feature columns: %s", missing)
            raise ValueError(f"input is missing feature columns: {missing}")
        if not self.keep_:
            raise RuntimeError("RarePooler.transform called before fit")
        X = frame[FEATURES].copy()
        for col, keep in self.keep_.items():
            values = X[col].astype(str)
            X[col] = values.where(values.isin(keep), "other")
        return X


assert_no_leakage()
