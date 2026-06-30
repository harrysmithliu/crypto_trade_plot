"""Tests for transform, metric aggregation and anomaly detection."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_mock_data import generate_mock_data  # noqa: E402
from src.anomaly import FailureRateAnomalyConfig, detect_failure_rate_anomalies  # noqa: E402
from src.data_sources.csv_source import clean_transaction_frame  # noqa: E402
from src.metrics import (  # noqa: E402
    compute_asset_preference,
    compute_daily_metrics,
    summarize_report_kpis,
)
from src.transform import add_period_start, build_period_frame  # noqa: E402


def _build_cleaned_transactions() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "tx_id": "TX-001",
                "timestamp": "2026-06-27T14:00:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "USDC",
                "fiat_volume_cad": 100.0,
                "tx_status": "Completed",
                "latency_ms": 100,
            },
            {
                "tx_id": "TX-002",
                "timestamp": "2026-06-27T16:00:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "BTC",
                "fiat_volume_cad": 50.0,
                "tx_status": "Completed",
                "latency_ms": 300,
            },
            {
                "tx_id": "TX-003",
                "timestamp": "2026-06-27T17:00:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "ETH",
                "fiat_volume_cad": 10.0,
                "tx_status": "Failed",
                "latency_ms": 800,
            },
            {
                "tx_id": "TX-004",
                "timestamp": "2026-06-27T18:00:00Z",
                "merchant_id": "M1001",
                "crypto_asset": "USDT",
                "fiat_volume_cad": 75.0,
                "tx_status": "Pending",
                "latency_ms": 400,
            },
            {
                "tx_id": "TX-005",
                "timestamp": "2026-06-28T12:00:00Z",
                "merchant_id": "M1002",
                "crypto_asset": "USDC",
                "fiat_volume_cad": 20.0,
                "tx_status": "Pending",
                "latency_ms": 500,
            },
            {
                "tx_id": "TX-006",
                "timestamp": "2026-06-29T13:00:00Z",
                "merchant_id": "M1003",
                "crypto_asset": "ETH",
                "fiat_volume_cad": 60.0,
                "tx_status": "Failed",
                "latency_ms": 700,
            },
            {
                "tx_id": "TX-007",
                "timestamp": "2026-06-29T15:00:00Z",
                "merchant_id": "M1003",
                "crypto_asset": "BTC",
                "fiat_volume_cad": 80.0,
                "tx_status": "Failed",
                "latency_ms": 900,
            },
        ]
    )
    cleaned, _ = clean_transaction_frame(raw, invalid_ratio_threshold=1.0)
    return cleaned


def test_transform_helpers_build_daily_and_hourly_periods() -> None:
    cleaned = _build_cleaned_transactions()

    hourly = add_period_start(cleaned, frequency="hourly")
    daily = add_period_start(cleaned, frequency="daily")
    calendar = build_period_frame(
        pd.Timestamp("2026-06-27", tz="America/Toronto"),
        pd.Timestamp("2026-06-29", tz="America/Toronto"),
        frequency="daily",
    )

    assert "period_start" in hourly.columns
    assert hourly["period_start"].dt.minute.eq(0).all()
    assert daily["period_start"].dt.hour.eq(0).all()
    assert len(calendar) == 3


def test_daily_metrics_follow_business_definitions() -> None:
    cleaned = _build_cleaned_transactions()
    daily_metrics = compute_daily_metrics(cleaned)

    first_day = daily_metrics.iloc[0]
    second_day = daily_metrics.iloc[1]
    third_day = daily_metrics.iloc[2]

    assert first_day["local_date"].isoformat() == "2026-06-27"
    assert first_day["submitted_tx_count"] == 4
    assert first_day["terminal_tx_count"] == 3
    assert first_day["completed_tx_count"] == 2
    assert first_day["failed_tx_count"] == 1
    assert first_day["pending_tx_count"] == 1
    assert first_day["total_volume_cad"] == pytest.approx(150.0)
    assert first_day["success_rate"] == pytest.approx(66.6666667)
    assert first_day["failure_rate"] == pytest.approx(33.3333333)
    assert first_day["avg_latency_ms"] == pytest.approx(200.0)
    assert first_day["p95_latency_ms"] == pytest.approx(290.0)

    assert second_day["local_date"].isoformat() == "2026-06-28"
    assert second_day["terminal_tx_count"] == 0
    assert pd.isna(second_day["success_rate"])
    assert pd.isna(second_day["failure_rate"])
    assert second_day["total_volume_cad"] == pytest.approx(0.0)

    assert third_day["local_date"].isoformat() == "2026-06-29"
    assert third_day["completed_tx_count"] == 0
    assert third_day["failed_tx_count"] == 2
    assert third_day["success_rate"] == pytest.approx(0.0)
    assert third_day["failure_rate"] == pytest.approx(100.0)
    assert pd.isna(third_day["avg_latency_ms"])


def test_asset_preference_uses_completed_orders_only() -> None:
    cleaned = _build_cleaned_transactions()
    summary = compute_asset_preference(cleaned)

    assert summary["crypto_asset"].tolist() == ["USDC", "BTC"]
    assert summary["completed_tx_count"].tolist() == [1, 1]
    assert summary["completed_tx_share"].tolist() == pytest.approx([50.0, 50.0])
    assert summary["completed_volume_share"].tolist() == pytest.approx(
        [66.6666667, 33.3333333]
    )


def test_anomaly_detector_flags_mock_failure_spike() -> None:
    mock_data = generate_mock_data(end_date="2026-06-29")
    cleaned, _ = clean_transaction_frame(mock_data, invalid_ratio_threshold=1.0)
    daily_metrics = compute_daily_metrics(cleaned)
    anomalies = detect_failure_rate_anomalies(daily_metrics)

    flagged = anomalies[anomalies["is_anomaly"]]
    assert len(flagged) == 1
    assert flagged.iloc[0]["local_date"].isoformat() == mock_data.attrs["anomaly_date"]
    assert flagged.iloc[0]["baseline_day_count"] >= 3
    assert "baseline" in flagged.iloc[0]["anomaly_reason"]


def test_anomaly_detector_uses_absolute_rule_when_baseline_is_short() -> None:
    daily_metrics = pd.DataFrame(
        {
            "local_date": pd.to_datetime(
                ["2026-06-01", "2026-06-02", "2026-06-03"]
            ).date,
            "terminal_tx_count": [60, 60, 60],
            "failure_rate": [12.0, 2.0, 3.0],
        }
    )

    anomalies = detect_failure_rate_anomalies(
        daily_metrics,
        FailureRateAnomalyConfig(min_terminal_tx=50, absolute_failure_rate_threshold=10.0),
    )

    assert anomalies.iloc[0]["is_anomaly"] == True
    assert anomalies.iloc[0]["baseline_day_count"] == 0
    assert "insufficient baseline history" in anomalies.iloc[0]["anomaly_reason"]


def test_anomaly_detector_rejects_low_sample_days() -> None:
    daily_metrics = pd.DataFrame(
        {
            "local_date": pd.to_datetime(["2026-06-01"]).date,
            "terminal_tx_count": [10],
            "failure_rate": [50.0],
        }
    )

    anomalies = detect_failure_rate_anomalies(
        daily_metrics,
        FailureRateAnomalyConfig(min_terminal_tx=50),
    )

    assert anomalies.iloc[0]["is_anomaly"] == False
    assert anomalies.iloc[0]["anomaly_reason"] == ""


def test_report_kpis_include_anomaly_count() -> None:
    cleaned = _build_cleaned_transactions()
    daily_metrics = compute_daily_metrics(cleaned)
    anomalies = detect_failure_rate_anomalies(
        daily_metrics,
        FailureRateAnomalyConfig(min_terminal_tx=1, absolute_failure_rate_threshold=30.0),
    )

    kpis = summarize_report_kpis(daily_metrics, anomalies)

    assert kpis["window_start_date"] == "2026-06-27"
    assert kpis["window_end_date"] == "2026-06-29"
    assert kpis["total_submitted_tx_count"] == 7
    assert kpis["total_completed_tx_count"] == 2
    assert kpis["total_failed_tx_count"] == 3
    assert kpis["total_volume_cad"] == pytest.approx(150.0)
    assert kpis["anomaly_day_count"] >= 1
