"""Transformation helpers for local-time resampling and calendar expansion."""

from __future__ import annotations

from typing import Literal

import pandas as pd


Frequency = Literal["daily", "hourly", "D", "H", "h"]
REQUIRED_TRANSACTION_COLUMNS = ("local_timestamp", "local_date", "tx_status")
_FREQUENCY_ALIASES = {
    "daily": "D",
    "D": "D",
    "hourly": "h",
    "H": "h",
    "h": "h",
}


def normalize_frequency(frequency: Frequency) -> str:
    """Return the canonical pandas frequency for supported report grains."""
    try:
        return _FREQUENCY_ALIASES[frequency]
    except KeyError as error:
        raise ValueError(f"Unsupported frequency: {frequency}") from error


def validate_transaction_frame(dataframe: pd.DataFrame) -> None:
    """Ensure the cleaned transaction frame contains aggregation columns."""
    missing_columns = [
        column for column in REQUIRED_TRANSACTION_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required transaction columns: {missing}")

    if not isinstance(dataframe["local_timestamp"].dtype, pd.DatetimeTZDtype):
        raise ValueError("local_timestamp must be timezone-aware")


def add_period_start(
    dataframe: pd.DataFrame,
    *,
    frequency: Frequency = "daily",
    timestamp_column: str = "local_timestamp",
    output_column: str = "period_start",
) -> pd.DataFrame:
    """Attach a normalized local period boundary for daily or hourly grouping."""
    validate_transaction_frame(dataframe)
    normalized_frequency = normalize_frequency(frequency)

    transformed = dataframe.copy()
    if normalized_frequency == "D":
        transformed[output_column] = transformed[timestamp_column].dt.normalize()
    else:
        transformed[output_column] = transformed[timestamp_column].dt.floor(normalized_frequency)
    return transformed


def build_period_frame(
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    *,
    frequency: Frequency = "daily",
    timezone: str = "America/Toronto",
    column_name: str = "period_start",
) -> pd.DataFrame:
    """Create a complete local calendar so missing days remain visible in reports."""
    normalized_frequency = normalize_frequency(frequency)

    if start_at.tzinfo is None:
        start_at = start_at.tz_localize(timezone)
    else:
        start_at = start_at.tz_convert(timezone)

    if end_at.tzinfo is None:
        end_at = end_at.tz_localize(timezone)
    else:
        end_at = end_at.tz_convert(timezone)

    if normalized_frequency == "D":
        start_at = start_at.normalize()
        end_at = end_at.normalize()
    else:
        start_at = start_at.floor(normalized_frequency)
        end_at = end_at.floor(normalized_frequency)

    if end_at < start_at:
        raise ValueError("end_at must be greater than or equal to start_at")

    return pd.DataFrame(
        {
            column_name: pd.date_range(
                start=start_at,
                end=end_at,
                freq=normalized_frequency,
                tz=timezone,
            )
        }
    )
