"""Tests for the deterministic mock transaction dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pandas.testing as pdt
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_mock_data import generate_mock_data, write_mock_data  # noqa: E402


EXPECTED_COLUMNS = [
    "tx_id",
    "timestamp",
    "merchant_id",
    "crypto_asset",
    "fiat_volume_cad",
    "tx_status",
    "latency_ms",
]
END_DATE = "2026-06-29"


@pytest.fixture(scope="module")
def mock_data() -> pd.DataFrame:
    return generate_mock_data(end_date=END_DATE)


def test_schema_uniqueness_and_value_constraints(mock_data: pd.DataFrame) -> None:
    assert list(mock_data.columns) == EXPECTED_COLUMNS
    assert len(mock_data) == 3_000
    assert mock_data["tx_id"].is_unique
    assert mock_data[EXPECTED_COLUMNS].notna().all().all()
    assert (mock_data["fiat_volume_cad"] >= 0).all()
    assert (mock_data["latency_ms"] >= 0).all()
    assert set(mock_data["crypto_asset"]) == {"USDC", "USDT", "BTC", "ETH"}
    assert set(mock_data["tx_status"]) == {"Completed", "Pending", "Failed"}


def test_dataset_has_thirty_complete_local_days(mock_data: pd.DataFrame) -> None:
    local_dates = mock_data["timestamp"].dt.tz_convert("America/Toronto").dt.date
    daily_counts = mock_data.groupby(local_dates).size()

    assert len(daily_counts) == 30
    assert (daily_counts == 100).all()
    assert daily_counts.index[-1].isoformat() == END_DATE


def test_expected_asset_and_status_distribution(mock_data: pd.DataFrame) -> None:
    asset_share = mock_data["crypto_asset"].value_counts(normalize=True)
    status_share = mock_data["tx_status"].value_counts(normalize=True)

    assert asset_share["USDC"] == pytest.approx(0.55)
    assert asset_share["USDT"] == pytest.approx(0.25)
    assert asset_share["BTC"] == pytest.approx(0.12)
    assert asset_share["ETH"] == pytest.approx(0.08)
    assert 0.02 < status_share["Failed"] < 0.03
    assert status_share["Pending"] == pytest.approx(0.03)


def test_exactly_one_high_failure_anomaly_day(mock_data: pd.DataFrame) -> None:
    data = mock_data.copy()
    data["local_date"] = (
        data["timestamp"].dt.tz_convert("America/Toronto").dt.date
    )
    terminal = data[data["tx_status"].isin(["Completed", "Failed"])]
    daily_failure_rate = terminal.groupby("local_date")["tx_status"].apply(
        lambda values: (values == "Failed").mean()
    )

    anomaly_days = daily_failure_rate[daily_failure_rate >= 0.10]
    normal_days = daily_failure_rate[daily_failure_rate < 0.10]

    assert len(anomaly_days) == 1
    assert anomaly_days.iloc[0] == pytest.approx(15 / 97)
    assert all(value == pytest.approx(2 / 97) for value in normal_days)
    assert anomaly_days.index[0].isoformat() == mock_data.attrs["anomaly_date"]


def test_generation_is_reproducible() -> None:
    first = generate_mock_data(seed=42, end_date=END_DATE)
    second = generate_mock_data(seed=42, end_date=END_DATE)

    pdt.assert_frame_equal(first, second)


def test_csv_output_uses_utc_iso_timestamps(
    mock_data: pd.DataFrame, tmp_path: Path
) -> None:
    output = tmp_path / "mock.csv"
    write_mock_data(mock_data, output)
    loaded = pd.read_csv(output)

    assert output.exists()
    assert list(loaded.columns) == EXPECTED_COLUMNS
    assert len(loaded) == 3_000
    assert loaded["timestamp"].str.match(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
    ).all()
