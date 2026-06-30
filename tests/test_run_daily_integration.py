"""Integration tests for the non-interactive daily pipeline entry point."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_command(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_daily_generates_dated_and_latest_reports(tmp_path: Path) -> None:
    input_csv = tmp_path / "raw" / "transactions.csv"
    output_dir = tmp_path / "reports"
    log_dir = tmp_path / "logs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            data_source:
              type: csv
              csv_path: {input_csv}
              local_timezone: America/Toronto
              window_days: 30
              invalid_ratio_threshold: 0.05

            mock_data:
              enabled: true
              output_path: {input_csv}
              days: 30
              transactions_per_day: 100
              seed: 42
              end_date: 2026-06-29

            runtime:
              output_dir: {output_dir}
              log_dir: {log_dir}
              log_level: INFO

            anomaly:
              min_terminal_tx: 50
              absolute_failure_rate_threshold: 10.0
              relative_failure_multiplier: 3.0
              baseline_lookback_days: 7
              min_baseline_days: 3

            report:
              title: Crypto Trade Daily Report
              prefix: Crypto_Trade_Daily_Report
              scatter_max_points:
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = _run_command(
        [sys.executable, "run_daily.py", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "success"
    assert summary["window_end_date"] == "2026-06-29"
    assert Path(summary["dated_report_path"]).exists()
    assert Path(summary["latest_report_path"]).exists()
    assert (output_dir / "Crypto_Trade_Daily_Report_2026-06-29.html").exists()
    assert (output_dir / "Crypto_Trade_Daily_Report_latest.html").exists()

    html = (output_dir / "Crypto_Trade_Daily_Report_latest.html").read_text(encoding="utf-8")
    assert "Daily Settled Volume and Success Rate" in html
    assert "Completed Order Asset Preference" in html
    assert "Large Transactions and Processing Latency" in html
    assert "Anomaly Day Details" in html
    assert (log_dir / "run_daily.log").exists()


def test_run_daily_fails_cleanly_for_invalid_input(tmp_path: Path) -> None:
    invalid_csv = tmp_path / "invalid.csv"
    output_dir = tmp_path / "reports"
    log_dir = tmp_path / "logs"
    config_path = tmp_path / "config.yaml"

    invalid_csv.write_text(
        "tx_id,timestamp,merchant_id,crypto_asset,fiat_volume_cad,tx_status\n"
        "TX-001,2026-06-29T10:00:00Z,M1001,USDC,100.0,Completed\n",
        encoding="utf-8",
    )
    config_path.write_text(
        textwrap.dedent(
            f"""
            data_source:
              type: csv
              csv_path: {invalid_csv}
              local_timezone: America/Toronto
              window_days: 30
              invalid_ratio_threshold: 0.05

            mock_data:
              enabled: false

            runtime:
              output_dir: {output_dir}
              log_dir: {log_dir}
              log_level: INFO

            anomaly:
              min_terminal_tx: 50
              absolute_failure_rate_threshold: 10.0
              relative_failure_multiplier: 3.0
              baseline_lookback_days: 7
              min_baseline_days: 3

            report:
              title: Crypto Trade Daily Report
              prefix: Crypto_Trade_Daily_Report
              scatter_max_points:
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = _run_command(
        [sys.executable, "run_daily.py", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
    )

    assert result.returncode != 0
    assert "Pipeline failed" in result.stderr
    assert not (output_dir / "Crypto_Trade_Daily_Report_latest.html").exists()
