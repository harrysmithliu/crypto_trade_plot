"""Failure-rate anomaly detection for the crypto trade report."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FailureRateAnomalyConfig:
    """Parameters for the default daily failure-rate anomaly rule."""

    min_terminal_tx: int = 50
    absolute_failure_rate_threshold: float = 10.0
    relative_failure_multiplier: float = 3.0
    baseline_lookback_days: int = 7
    min_baseline_days: int = 3


def _passes_relative_threshold(
    failure_rate: float,
    baseline_failure_rate: float,
    multiplier: float,
) -> bool:
    """Compare a day against its historical baseline with zero-safe logic."""
    if baseline_failure_rate == 0:
        return failure_rate > 0
    return failure_rate >= baseline_failure_rate * multiplier


def detect_failure_rate_anomalies(
    daily_metrics: pd.DataFrame,
    config: FailureRateAnomalyConfig | None = None,
) -> pd.DataFrame:
    """Flag daily failure-rate anomalies using absolute and relative thresholds."""
    if config is None:
        config = FailureRateAnomalyConfig()

    required_columns = ("local_date", "terminal_tx_count", "failure_rate")
    missing_columns = [column for column in required_columns if column not in daily_metrics.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required anomaly columns: {missing}")

    ordered = daily_metrics.copy().sort_values("local_date").reset_index(drop=True)
    ordered["baseline_failure_rate"] = pd.NA
    ordered["baseline_day_count"] = 0
    ordered["has_sufficient_baseline"] = False
    ordered["is_anomaly"] = False
    ordered["anomaly_reason"] = ""

    prior_valid_failure_rates: list[float] = []

    for row_index, row in ordered.iterrows():
        recent_baseline = prior_valid_failure_rates[-config.baseline_lookback_days :]
        baseline_day_count = len(recent_baseline)
        baseline_failure_rate = (
            None if baseline_day_count == 0 else sum(recent_baseline) / baseline_day_count
        )
        has_sufficient_baseline = baseline_day_count >= config.min_baseline_days

        ordered.at[row_index, "baseline_day_count"] = baseline_day_count
        ordered.at[row_index, "has_sufficient_baseline"] = has_sufficient_baseline
        if baseline_failure_rate is not None:
            ordered.at[row_index, "baseline_failure_rate"] = baseline_failure_rate

        failure_rate = row["failure_rate"]
        terminal_tx_count = int(row["terminal_tx_count"])
        passes_min_volume = terminal_tx_count >= config.min_terminal_tx
        passes_absolute = pd.notna(failure_rate) and (
            float(failure_rate) >= config.absolute_failure_rate_threshold
        )

        is_anomaly = False
        anomaly_reason = ""

        if passes_min_volume and passes_absolute:
            if has_sufficient_baseline:
                relative_pass = _passes_relative_threshold(
                    float(failure_rate),
                    float(baseline_failure_rate),
                    config.relative_failure_multiplier,
                )
                if relative_pass:
                    is_anomaly = True
                    anomaly_reason = (
                        f"Failure rate {float(failure_rate):.2f}% exceeded "
                        f"{config.absolute_failure_rate_threshold:.2f}% and was at least "
                        f"{config.relative_failure_multiplier:.2f}x the {baseline_day_count}-day "
                        f"baseline of {float(baseline_failure_rate):.2f}%."
                    )
            else:
                is_anomaly = True
                anomaly_reason = (
                    f"Failure rate {float(failure_rate):.2f}% exceeded "
                    f"{config.absolute_failure_rate_threshold:.2f}% with insufficient baseline "
                    f"history ({baseline_day_count} valid days)."
                )

        ordered.at[row_index, "is_anomaly"] = is_anomaly
        ordered.at[row_index, "anomaly_reason"] = anomaly_reason

        if pd.notna(failure_rate):
            prior_valid_failure_rates.append(float(failure_rate))

    return ordered
