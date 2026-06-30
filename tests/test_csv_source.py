"""Tests for the CSV transaction data source."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_mock_data import generate_mock_data, write_mock_data  # noqa: E402
from src.data_sources.csv_source import (  # noqa: E402
    CsvTransactionDataSource,
    DataContractError,
    DataQualityError,
    clean_transaction_frame,
)


def _write_csv(tmp_path: Path, dataframe: pd.DataFrame, name: str = "transactions.csv") -> Path:
    path = tmp_path / name
    dataframe.to_csv(path, index=False)
    return path


def test_csv_source_loads_default_latest_thirty_local_days(tmp_path: Path) -> None:
    dataframe = generate_mock_data(days=35, end_date="2026-06-29")
    csv_path = tmp_path / "transactions.csv"
    write_mock_data(dataframe, csv_path)

    source = CsvTransactionDataSource(csv_path=csv_path)
    loaded = source.load()

    local_dates = pd.Series(loaded["local_date"])
    assert len(local_dates.unique()) == 30
    assert local_dates.min().isoformat() == "2026-05-31"
    assert local_dates.max().isoformat() == "2026-06-29"
    assert loaded.attrs["load_audit"]["analysis_end_date"] == "2026-06-29"


def test_timezone_conversion_handles_dst_boundaries() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "tx_id": "TX-001",
                "timestamp": "2026-03-08T06:30:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "USDC",
                "fiat_volume_cad": 100.0,
                "tx_status": "Completed",
                "latency_ms": 500,
            },
            {
                "tx_id": "TX-002",
                "timestamp": "2026-03-08T07:30:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "USDT",
                "fiat_volume_cad": 120.0,
                "tx_status": "Completed",
                "latency_ms": 700,
            },
        ]
    )

    cleaned, _ = clean_transaction_frame(
        dataframe,
        local_timezone="America/Toronto",
        invalid_ratio_threshold=1.0,
    )

    rendered = cleaned["local_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %z").tolist()
    assert rendered == [
        "2026-03-08 01:30:00 -0500",
        "2026-03-08 03:30:00 -0400",
    ]
    assert cleaned["local_date"].astype(str).tolist() == ["2026-03-08", "2026-03-08"]


def test_duplicate_transaction_ids_keep_last_row(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "tx_id": "TX-001",
                "timestamp": "2026-06-29T10:00:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "USDC",
                "fiat_volume_cad": 100.0,
                "tx_status": "Completed",
                "latency_ms": 400,
            },
            {
                "tx_id": "TX-001",
                "timestamp": "2026-06-29T11:00:00Z",
                "merchant_id": "M1002",
                "crypto_asset": "BTC",
                "fiat_volume_cad": 250.0,
                "tx_status": "Failed",
                "latency_ms": 900,
            },
        ]
    )
    csv_path = _write_csv(tmp_path, dataframe)

    source = CsvTransactionDataSource(csv_path=csv_path, invalid_ratio_threshold=1.0)
    loaded = source.load()

    assert len(loaded) == 1
    assert loaded.iloc[0]["merchant_id"] == "M1002"
    assert loaded.iloc[0]["fiat_volume_cad"] == pytest.approx(250.0)
    assert loaded.attrs["load_audit"]["duplicate_rows"] == 1


def test_invalid_rows_are_removed_when_ratio_is_within_threshold() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "tx_id": "TX-001",
                "timestamp": "2026-06-29T10:00:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "USDC",
                "fiat_volume_cad": 100.0,
                "tx_status": "Completed",
                "latency_ms": 400,
            },
            {
                "tx_id": "TX-002",
                "timestamp": "not-a-timestamp",
                "merchant_id": "M1002",
                "crypto_asset": "USDT",
                "fiat_volume_cad": 200.0,
                "tx_status": "Completed",
                "latency_ms": 500,
            },
            {
                "tx_id": "TX-003",
                "timestamp": "2026-06-29T12:00:00Z",
                "merchant_id": "M1003",
                "crypto_asset": "DOGE",
                "fiat_volume_cad": 50.0,
                "tx_status": "Completed",
                "latency_ms": -1,
            },
            {
                "tx_id": "TX-004",
                "timestamp": "2026-06-29T13:00:00Z",
                "merchant_id": "M1004",
                "crypto_asset": "BTC",
                "fiat_volume_cad": 80.0,
                "tx_status": "Pending",
                "latency_ms": 300,
            },
        ]
    )

    cleaned, audit = clean_transaction_frame(
        dataframe,
        invalid_ratio_threshold=0.6,
    )

    assert cleaned["tx_id"].tolist() == ["TX-001", "TX-004"]
    assert audit.invalid_rows == 2
    assert audit.invalid_reasons["invalid_timestamp"] == 1
    assert audit.invalid_reasons["invalid_crypto_asset"] == 1
    assert audit.invalid_reasons["invalid_latency_ms"] == 1


def test_invalid_ratio_threshold_raises_error() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "tx_id": "TX-001",
                "timestamp": "2026-06-29T10:00:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "USDC",
                "fiat_volume_cad": 100.0,
                "tx_status": "Completed",
                "latency_ms": 400,
            },
            {
                "tx_id": "TX-002",
                "timestamp": "bad",
                "merchant_id": "M1002",
                "crypto_asset": "USDT",
                "fiat_volume_cad": 200.0,
                "tx_status": "Completed",
                "latency_ms": 500,
            },
            {
                "tx_id": "TX-003",
                "timestamp": "2026-06-29T12:00:00Z",
                "merchant_id": "M1003",
                "crypto_asset": "ETH",
                "fiat_volume_cad": -5.0,
                "tx_status": "Completed",
                "latency_ms": 100,
            },
        ]
    )

    with pytest.raises(DataQualityError, match="Invalid row ratio exceeded"):
        clean_transaction_frame(dataframe, invalid_ratio_threshold=0.5)


def test_missing_required_column_raises_contract_error(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "tx_id": "TX-001",
                "timestamp": "2026-06-29T10:00:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "USDC",
                "fiat_volume_cad": 100.0,
                "tx_status": "Completed",
            }
        ]
    )
    csv_path = _write_csv(tmp_path, dataframe)

    source = CsvTransactionDataSource(csv_path=csv_path)
    with pytest.raises(DataContractError, match="Missing required columns"):
        source.load()
