"""Tests for Plotly chart builders and HTML report rendering."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_mock_data import generate_mock_data  # noqa: E402
from src.anomaly import detect_failure_rate_anomalies  # noqa: E402
from src.charts import (  # noqa: E402
    build_asset_preference_figure,
    build_latency_scatter_figure,
    build_volume_success_figure,
)
from src.data_sources.csv_source import clean_transaction_frame  # noqa: E402
from src.metrics import compute_asset_preference, compute_daily_metrics  # noqa: E402
from src.report import render_html_report, write_html_report  # noqa: E402


def _build_report_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = generate_mock_data(end_date="2026-06-29")
    cleaned, _ = clean_transaction_frame(raw, invalid_ratio_threshold=1.0)
    daily_metrics = compute_daily_metrics(cleaned)
    anomaly_metrics = detect_failure_rate_anomalies(daily_metrics)
    asset_summary = compute_asset_preference(cleaned)
    return cleaned, daily_metrics, anomaly_metrics, asset_summary


def test_volume_success_figure_contains_required_traces() -> None:
    _, _, anomaly_metrics, _ = _build_report_inputs()
    figure = build_volume_success_figure(anomaly_metrics)

    assert figure.layout.title.text == "Daily Settled Volume and Success Rate Trend"
    assert len(figure.data) == 3
    assert figure.data[0].type == "bar"
    assert figure.data[1].type == "scatter"
    assert figure.data[2].name == "Anomaly Days"


def test_asset_preference_figure_supports_empty_state() -> None:
    empty_summary = pd.DataFrame(
        columns=[
            "crypto_asset",
            "completed_tx_count",
            "completed_tx_share",
            "completed_volume_cad",
            "completed_volume_share",
        ]
    )
    figure = build_asset_preference_figure(empty_summary)

    assert figure.layout.title.text == "Completed Order Asset Preference"
    assert len(figure.data) == 0
    assert "No completed transactions" in figure.layout.annotations[0].text


def test_latency_scatter_figure_uses_log_x_axis_and_p95_reference() -> None:
    cleaned, _, _, _ = _build_report_inputs()
    figure = build_latency_scatter_figure(cleaned, max_points=200)

    assert figure.layout.title.text == "Large Transactions and Processing Latency"
    assert figure.layout.xaxis.type == "log"
    assert len(figure.data) >= 4
    assert figure.layout.shapes[0].type == "line"


def test_render_html_report_includes_required_sections_and_titles() -> None:
    cleaned, daily_metrics, anomaly_metrics, asset_summary = _build_report_inputs()
    html = render_html_report(
        cleaned_transactions=cleaned,
        daily_metrics=daily_metrics,
        anomaly_metrics=anomaly_metrics,
        asset_summary=asset_summary,
        generated_at=pd.Timestamp("2026-06-29 08:00:00", tz="America/Toronto"),
    )

    assert "Crypto Trade Daily Report" in html
    assert "Data Quality Summary" in html
    assert "Key Metrics" in html
    assert "Daily Settled Volume and Success Rate" in html
    assert "Completed Order Asset Preference" in html
    assert "Large Transactions and Processing Latency" in html
    assert "Anomaly Day Details" in html
    assert "Plotly.newPlot" in html


def test_write_html_report_outputs_dated_and_latest_files(tmp_path: Path) -> None:
    cleaned, daily_metrics, anomaly_metrics, asset_summary = _build_report_inputs()
    html = render_html_report(
        cleaned_transactions=cleaned,
        daily_metrics=daily_metrics,
        anomaly_metrics=anomaly_metrics,
        asset_summary=asset_summary,
    )

    report_paths = write_html_report(
        html,
        output_dir=tmp_path,
        report_date="2026-06-29",
    )

    assert report_paths.dated_report_path.exists()
    assert report_paths.latest_report_path.exists()
    assert "Crypto_Trade_Daily_Report_2026-06-29.html" in str(report_paths.dated_report_path)
    assert "Crypto_Trade_Daily_Report_latest.html" in str(report_paths.latest_report_path)
    assert report_paths.dated_report_path.read_text(encoding="utf-8") == html
    assert report_paths.latest_report_path.read_text(encoding="utf-8") == html
