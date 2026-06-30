"""CSV-backed transaction data source with contract validation and cleaning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .base import TransactionDataSource


REQUIRED_COLUMNS = (
    "tx_id",
    "timestamp",
    "merchant_id",
    "crypto_asset",
    "fiat_volume_cad",
    "tx_status",
    "latency_ms",
)
VALID_ASSETS = frozenset({"USDC", "USDT", "ETH", "BTC"})
VALID_STATUSES = frozenset({"Completed", "Pending", "Failed"})
BASE_OUTPUT_COLUMNS = [
    "tx_id",
    "timestamp",
    "local_timestamp",
    "local_date",
    "merchant_id",
    "crypto_asset",
    "fiat_volume_cad",
    "tx_status",
    "latency_ms",
]
OPTIONAL_TIMESTAMP_COLUMNS = (
    "order_created_at",
    "channel_selected_at",
    "kyc_passed_at",
    "payment_completed_at",
    "settled_at",
)
OPTIONAL_NUMERIC_COLUMNS = (
    "flat_fee_cad",
    "spread_income_cad",
    "risk_score",
    "velocity_1h",
    "provider_amount_cad",
    "ledger_amount_cad",
    "recon_delta_cad",
    "settlement_latency_min",
)
OPTIONAL_BOOLEAN_COLUMNS = ("aml_flag", "is_high_risk")


class DataContractError(ValueError):
    """Raised when the input file does not satisfy the required schema."""


class DataQualityError(ValueError):
    """Raised when the input file contains too many invalid rows."""


@dataclass(frozen=True)
class LoadAudit:
    """Structured audit details produced by the data source layer."""

    source_path: str
    input_rows: int
    duplicate_rows: int
    deduplicated_rows: int
    invalid_rows: int
    invalid_ratio: float
    filtered_rows: int
    output_rows: int
    analysis_start_date: str
    analysis_end_date: str
    local_timezone: str
    invalid_reasons: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _normalize_bound(
    value: str | pd.Timestamp | None,
    *,
    timezone: str,
    is_end: bool,
) -> pd.Timestamp | None:
    """Convert optional bounds to local timestamps."""
    if value is None:
        return None

    parsed = pd.Timestamp(value)
    is_date_like = (
        isinstance(value, str)
        and len(value) <= 10
        and parsed.hour == 0
        and parsed.minute == 0
        and parsed.second == 0
    )

    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(timezone)
    else:
        parsed = parsed.tz_convert(timezone)

    if is_date_like:
        parsed = parsed.normalize()
        if is_end:
            parsed += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    return parsed


def _normalize_string(series: pd.Series) -> pd.Series:
    """Trim whitespace while preserving missing values as empty strings."""
    return series.fillna("").astype(str).str.strip()


def _count_invalid_reasons(dataframe: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """Build the invalid-row mask and per-reason counts."""
    tx_id = _normalize_string(dataframe["tx_id"])
    merchant_id = _normalize_string(dataframe["merchant_id"])
    crypto_asset = _normalize_string(dataframe["crypto_asset"])
    tx_status = _normalize_string(dataframe["tx_status"])
    amount = pd.to_numeric(dataframe["fiat_volume_cad"], errors="coerce")
    latency = pd.to_numeric(dataframe["latency_ms"], errors="coerce")
    parsed_timestamp = pd.to_datetime(dataframe["timestamp"], utc=True, errors="coerce")

    reason_masks = {
        "missing_tx_id": tx_id.eq(""),
        "missing_merchant_id": merchant_id.eq(""),
        "invalid_timestamp": parsed_timestamp.isna(),
        "invalid_crypto_asset": ~crypto_asset.isin(VALID_ASSETS),
        "invalid_tx_status": ~tx_status.isin(VALID_STATUSES),
        "invalid_fiat_volume_cad": amount.isna() | (amount < 0),
        "invalid_latency_ms": latency.isna() | (latency < 0),
    }

    invalid_mask = pd.Series(False, index=dataframe.index)
    invalid_reasons: dict[str, int] = {}

    for reason, mask in reason_masks.items():
        invalid_mask |= mask
        invalid_reasons[reason] = int(mask.sum())

    dataframe["tx_id"] = tx_id
    dataframe["merchant_id"] = merchant_id
    dataframe["crypto_asset"] = crypto_asset
    dataframe["tx_status"] = tx_status
    dataframe["fiat_volume_cad"] = amount
    dataframe["latency_ms"] = latency
    dataframe["timestamp"] = parsed_timestamp
    return invalid_mask, invalid_reasons


def clean_transaction_frame(
    dataframe: pd.DataFrame,
    *,
    source_path: str = "<in-memory>",
    local_timezone: str = "America/Toronto",
    window_days: int = 30,
    invalid_ratio_threshold: float = 0.05,
    start_at: str | pd.Timestamp | None = None,
    end_at: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, LoadAudit]:
    """Validate, clean and trim a raw transaction DataFrame."""
    if window_days <= 0:
        raise ValueError("window_days must be greater than zero")
    if not 0 <= invalid_ratio_threshold <= 1:
        raise ValueError("invalid_ratio_threshold must be between 0 and 1")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise DataContractError(f"Missing required columns: {missing}")

    working = dataframe.copy()
    input_rows = len(working)

    deduplicated = working.drop_duplicates(subset="tx_id", keep="last").copy()
    duplicate_rows = input_rows - len(deduplicated)

    invalid_mask, invalid_reasons = _count_invalid_reasons(deduplicated)
    invalid_rows = int(invalid_mask.sum())
    denominator = len(deduplicated) or 1
    invalid_ratio = invalid_rows / denominator
    if invalid_ratio > invalid_ratio_threshold:
        raise DataQualityError(
            "Invalid row ratio exceeded the configured threshold: "
            f"{invalid_rows}/{len(deduplicated)} ({invalid_ratio:.2%})"
        )

    cleaned = deduplicated.loc[~invalid_mask].copy()
    if cleaned.empty:
        raise DataQualityError("No valid transaction rows remain after cleaning")

    cleaned["timestamp"] = cleaned["timestamp"].dt.tz_convert("UTC")
    cleaned["local_timestamp"] = cleaned["timestamp"].dt.tz_convert(local_timezone)
    cleaned["local_date"] = cleaned["local_timestamp"].dt.date
    cleaned["latency_ms"] = cleaned["latency_ms"].astype(int)
    cleaned["fiat_volume_cad"] = cleaned["fiat_volume_cad"].astype(float)
    for column in OPTIONAL_TIMESTAMP_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], utc=True, errors="coerce")
    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in OPTIONAL_BOOLEAN_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .replace({"True": True, "False": False, "true": True, "false": False})
                .astype("boolean")
            )
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

    local_end = _normalize_bound(end_at, timezone=local_timezone, is_end=True)
    local_start = _normalize_bound(start_at, timezone=local_timezone, is_end=False)

    if local_end is None:
        latest_local_date = cleaned["local_timestamp"].max().normalize()
        local_end = latest_local_date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if local_start is None:
        local_start = local_end.normalize() - pd.Timedelta(days=window_days - 1)

    window_mask = cleaned["local_timestamp"].between(local_start, local_end, inclusive="both")
    passthrough_columns = [
        column for column in cleaned.columns if column not in BASE_OUTPUT_COLUMNS
    ]
    filtered = cleaned.loc[
        window_mask, BASE_OUTPUT_COLUMNS + passthrough_columns
    ].reset_index(drop=True)
    if filtered.empty:
        raise DataQualityError("No valid transaction rows remain inside the analysis window")

    audit = LoadAudit(
        source_path=source_path,
        input_rows=input_rows,
        duplicate_rows=duplicate_rows,
        deduplicated_rows=len(deduplicated),
        invalid_rows=invalid_rows,
        invalid_ratio=invalid_ratio,
        filtered_rows=len(cleaned) - len(filtered),
        output_rows=len(filtered),
        analysis_start_date=filtered["local_date"].min().isoformat(),
        analysis_end_date=filtered["local_date"].max().isoformat(),
        local_timezone=local_timezone,
        invalid_reasons=invalid_reasons,
    )

    filtered.attrs["load_audit"] = audit.as_dict()
    filtered.attrs["source_path"] = source_path
    return filtered, audit


@dataclass
class CsvTransactionDataSource(TransactionDataSource):
    """Read transactions from a UTF-8 CSV file."""

    csv_path: Path
    local_timezone: str = "America/Toronto"
    window_days: int = 30
    invalid_ratio_threshold: float = 0.05

    def load(
        self,
        start_at: str | pd.Timestamp | None = None,
        end_at: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load and clean transactions from the configured CSV."""
        dataframe = pd.read_csv(self.csv_path, encoding="utf-8")
        cleaned, audit = clean_transaction_frame(
            dataframe,
            source_path=str(self.csv_path),
            local_timezone=self.local_timezone,
            window_days=self.window_days,
            invalid_ratio_threshold=self.invalid_ratio_threshold,
            start_at=start_at,
            end_at=end_at,
        )
        cleaned.attrs["load_audit"] = audit.as_dict()
        return cleaned
