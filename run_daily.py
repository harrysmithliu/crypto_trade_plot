"""Command-line entry point for generating the crypto trade HTML report."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import yaml

from scripts.generate_mock_data import generate_mock_data, write_mock_data
from src.anomaly import FailureRateAnomalyConfig, detect_failure_rate_anomalies
from src.data_sources.csv_source import CsvTransactionDataSource
from src.logging_config import configure_logging
from src.metrics import compute_asset_preference, compute_daily_metrics
from src.report import render_html_report, write_html_report


@dataclass(frozen=True)
class RunResult:
    """Structured result returned by the non-interactive daily pipeline."""

    status: str
    source_path: str
    dated_report_path: str | None
    latest_report_path: str | None
    window_start_date: str | None
    window_end_date: str | None
    anomaly_day_count: int
    generated_mock_data: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the crypto trade daily HTML report."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--input-csv", help="Override the CSV input path.")
    parser.add_argument("--output-dir", help="Override the report output directory.")
    parser.add_argument("--window-days", type=int, help="Override the analysis window size.")
    parser.add_argument("--timezone", help="Override the local timezone.")
    parser.add_argument(
        "--invalid-ratio-threshold",
        type=float,
        help="Override the maximum invalid row ratio before failing.",
    )
    parser.add_argument(
        "--absolute-failure-rate-threshold",
        type=float,
        help="Override the anomaly absolute failure-rate threshold.",
    )
    parser.add_argument(
        "--relative-failure-multiplier",
        type=float,
        help="Override the anomaly relative failure multiplier.",
    )
    parser.add_argument(
        "--min-terminal-tx",
        type=int,
        help="Override the anomaly minimum terminal transaction count.",
    )
    parser.add_argument("--log-level", help="Override the runtime log level.")
    parser.add_argument(
        "--generate-mock-data",
        action="store_true",
        help="Generate mock data before building the report.",
    )
    parser.add_argument("--mock-output-path", help="Override the mock data output path.")
    parser.add_argument("--mock-seed", type=int, help="Override the mock data seed.")
    parser.add_argument("--mock-days", type=int, help="Override the mock data day count.")
    parser.add_argument(
        "--mock-transactions-per-day",
        type=int,
        help="Override the mock transactions per day.",
    )
    parser.add_argument(
        "--mock-end-date",
        help="Optional local mock end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--scatter-max-points",
        type=int,
        help="Maximum plotted completed transactions for the latency scatter.",
    )
    return parser


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config into a nested dictionary."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI overrides into the runtime config."""
    merged = {
        "data_source": dict(config.get("data_source", {})),
        "mock_data": dict(config.get("mock_data", {})),
        "runtime": dict(config.get("runtime", {})),
        "anomaly": dict(config.get("anomaly", {})),
        "report": dict(config.get("report", {})),
    }

    if args.input_csv:
        merged["data_source"]["csv_path"] = args.input_csv
    if args.output_dir:
        merged["runtime"]["output_dir"] = args.output_dir
    if args.window_days is not None:
        merged["data_source"]["window_days"] = args.window_days
    if args.timezone:
        merged["data_source"]["local_timezone"] = args.timezone
    if args.invalid_ratio_threshold is not None:
        merged["data_source"]["invalid_ratio_threshold"] = args.invalid_ratio_threshold
    if args.absolute_failure_rate_threshold is not None:
        merged["anomaly"]["absolute_failure_rate_threshold"] = (
            args.absolute_failure_rate_threshold
        )
    if args.relative_failure_multiplier is not None:
        merged["anomaly"]["relative_failure_multiplier"] = args.relative_failure_multiplier
    if args.min_terminal_tx is not None:
        merged["anomaly"]["min_terminal_tx"] = args.min_terminal_tx
    if args.log_level:
        merged["runtime"]["log_level"] = args.log_level
    if args.generate_mock_data:
        merged["mock_data"]["enabled"] = True
    if args.mock_output_path:
        merged["mock_data"]["output_path"] = args.mock_output_path
    if args.mock_seed is not None:
        merged["mock_data"]["seed"] = args.mock_seed
    if args.mock_days is not None:
        merged["mock_data"]["days"] = args.mock_days
    if args.mock_transactions_per_day is not None:
        merged["mock_data"]["transactions_per_day"] = args.mock_transactions_per_day
    if args.mock_end_date:
        merged["mock_data"]["end_date"] = args.mock_end_date
    if args.scatter_max_points is not None:
        merged["report"]["scatter_max_points"] = args.scatter_max_points
    return merged


def generate_mock_dataset(config: dict[str, Any]) -> Path:
    """Generate mock CSV data if enabled in config or CLI."""
    mock_config = config["mock_data"]
    output_path = Path(
        mock_config.get("output_path")
        or config["data_source"].get("csv_path")
        or "data/raw/crypto_transactions_30d.csv"
    )
    dataframe = generate_mock_data(
        days=int(mock_config.get("days", 30)),
        transactions_per_day=int(mock_config.get("transactions_per_day", 100)),
        seed=int(mock_config.get("seed", 42)),
        end_date=mock_config.get("end_date"),
        timezone=config["data_source"].get("local_timezone", "America/Toronto"),
    )
    write_mock_data(dataframe, output_path)
    return output_path


def run_pipeline(config: dict[str, Any]) -> RunResult:
    """Execute the full daily report pipeline and return a structured result."""
    runtime_config = config["runtime"]
    logger = configure_logging(
        log_level=runtime_config.get("log_level", "INFO"),
        log_dir=runtime_config.get("log_dir", "logs"),
    )
    logger.info("Starting daily report pipeline")

    timings: dict[str, float] = {}
    generated_mock_data = bool(config["mock_data"].get("enabled", False))

    source_path = Path(config["data_source"].get("csv_path", "data/raw/crypto_transactions_30d.csv"))
    if generated_mock_data:
        started = perf_counter()
        source_path = generate_mock_dataset(config)
        timings["mock_generation_seconds"] = perf_counter() - started
        logger.info("Generated mock data at %s", source_path)

    if config["data_source"].get("type", "csv") != "csv":
        raise ValueError("Only the CSV data source is implemented in this demo version")

    started = perf_counter()
    data_source = CsvTransactionDataSource(
        csv_path=source_path,
        local_timezone=config["data_source"].get("local_timezone", "America/Toronto"),
        window_days=int(config["data_source"].get("window_days", 30)),
        invalid_ratio_threshold=float(
            config["data_source"].get("invalid_ratio_threshold", 0.05)
        ),
    )
    cleaned_transactions = data_source.load()
    timings["data_source_seconds"] = perf_counter() - started

    load_audit = cleaned_transactions.attrs.get("load_audit", {})
    logger.info(
        "Loaded transactions from %s | input_rows=%s valid_rows=%s invalid_rows=%s duplicate_rows=%s window=%s..%s",
        source_path,
        load_audit.get("input_rows"),
        load_audit.get("output_rows"),
        load_audit.get("invalid_rows"),
        load_audit.get("duplicate_rows"),
        load_audit.get("analysis_start_date"),
        load_audit.get("analysis_end_date"),
    )

    started = perf_counter()
    daily_metrics = compute_daily_metrics(cleaned_transactions)
    asset_summary = compute_asset_preference(cleaned_transactions)
    timings["metrics_seconds"] = perf_counter() - started

    started = perf_counter()
    anomaly_metrics = detect_failure_rate_anomalies(
        daily_metrics,
        FailureRateAnomalyConfig(
            min_terminal_tx=int(config["anomaly"].get("min_terminal_tx", 50)),
            absolute_failure_rate_threshold=float(
                config["anomaly"].get("absolute_failure_rate_threshold", 10.0)
            ),
            relative_failure_multiplier=float(
                config["anomaly"].get("relative_failure_multiplier", 3.0)
            ),
            baseline_lookback_days=int(config["anomaly"].get("baseline_lookback_days", 7)),
            min_baseline_days=int(config["anomaly"].get("min_baseline_days", 3)),
        ),
    )
    timings["anomaly_seconds"] = perf_counter() - started

    started = perf_counter()
    html = render_html_report(
        cleaned_transactions=cleaned_transactions,
        daily_metrics=daily_metrics,
        anomaly_metrics=anomaly_metrics,
        asset_summary=asset_summary,
        report_title=config["report"].get("title", "Crypto Trade Daily Report"),
        scatter_max_points=config["report"].get("scatter_max_points"),
    )
    report_date = daily_metrics["local_date"].max().isoformat()
    report_paths = write_html_report(
        html,
        output_dir=runtime_config.get("output_dir", "reports"),
        report_date=report_date,
        report_prefix=config["report"].get("prefix", "Crypto_Trade_Daily_Report"),
    )
    timings["report_seconds"] = perf_counter() - started

    anomaly_count = int(anomaly_metrics["is_anomaly"].sum())
    logger.info(
        "Generated report files | dated=%s latest=%s anomaly_days=%s",
        report_paths.dated_report_path,
        report_paths.latest_report_path,
        anomaly_count,
    )
    logger.info("Stage timings | %s", json.dumps({k: round(v, 4) for k, v in timings.items()}))
    logger.info("Finished daily report pipeline successfully")

    return RunResult(
        status="success",
        source_path=str(source_path),
        dated_report_path=str(report_paths.dated_report_path),
        latest_report_path=str(report_paths.latest_report_path),
        window_start_date=daily_metrics["local_date"].min().isoformat(),
        window_end_date=report_date,
        anomaly_day_count=anomaly_count,
        generated_mock_data=generated_mock_data,
    )


def main() -> int:
    args = build_parser().parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    logger = configure_logging(
        log_level=config["runtime"].get("log_level", "INFO"),
        log_dir=config["runtime"].get("log_dir", "logs"),
    )

    try:
        result = run_pipeline(config)
    except Exception as error:
        logger.exception("Pipeline failed: %s", error)
        return 1

    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
