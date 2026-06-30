"""Business metric aggregations for the crypto trade report."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .transform import add_period_start, build_period_frame, validate_transaction_frame


def _daily_date_index(periods: pd.Series) -> pd.Series:
    """Convert local midnight timestamps into plain local dates."""
    return periods.dt.date


def compute_daily_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions into daily operational metrics."""
    validate_transaction_frame(dataframe)

    required_columns = ("fiat_volume_cad", "latency_ms", "crypto_asset")
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required metric columns: {missing}")

    with_periods = add_period_start(dataframe, frequency="daily")
    period_calendar = build_period_frame(
        with_periods["period_start"].min(),
        with_periods["period_start"].max(),
        frequency="daily",
        timezone=str(with_periods["local_timestamp"].dt.tz),
    ).set_index("period_start")

    status_counts = (
        with_periods.groupby(["period_start", "tx_status"]).size().unstack(fill_value=0)
    )
    for status in ("Completed", "Failed", "Pending"):
        if status not in status_counts.columns:
            status_counts[status] = 0
    status_counts = status_counts[["Completed", "Failed", "Pending"]]

    completed_only = with_periods[with_periods["tx_status"] == "Completed"]
    volume_by_day = completed_only.groupby("period_start")["fiat_volume_cad"].sum()
    latency_by_day = completed_only.groupby("period_start")["latency_ms"].agg(
        avg_latency_ms="mean",
        p95_latency_ms=lambda values: values.quantile(0.95),
    )

    metrics = period_calendar.join(status_counts, how="left").join(volume_by_day, how="left")
    metrics = metrics.join(latency_by_day, how="left")
    metrics = metrics.rename(columns={"fiat_volume_cad": "total_volume_cad"})
    metrics[["Completed", "Failed", "Pending"]] = (
        metrics[["Completed", "Failed", "Pending"]].fillna(0).astype(int)
    )
    metrics["total_volume_cad"] = metrics["total_volume_cad"].fillna(0.0)

    metrics["submitted_tx_count"] = (
        metrics["Completed"] + metrics["Failed"] + metrics["Pending"]
    )
    metrics["terminal_tx_count"] = metrics["Completed"] + metrics["Failed"]
    metrics["completed_tx_count"] = metrics["Completed"]
    metrics["failed_tx_count"] = metrics["Failed"]
    metrics["pending_tx_count"] = metrics["Pending"]

    terminal_counts = metrics["terminal_tx_count"].replace(0, pd.NA)
    metrics["success_rate"] = (metrics["completed_tx_count"] / terminal_counts) * 100
    metrics["failure_rate"] = (metrics["failed_tx_count"] / terminal_counts) * 100

    result = (
        metrics.reset_index()
        .assign(local_date=lambda frame: _daily_date_index(frame["period_start"]))
        .loc[
            :,
            [
                "local_date",
                "submitted_tx_count",
                "terminal_tx_count",
                "completed_tx_count",
                "failed_tx_count",
                "pending_tx_count",
                "total_volume_cad",
                "success_rate",
                "failure_rate",
                "avg_latency_ms",
                "p95_latency_ms",
            ],
        ]
    )

    result.attrs["source_path"] = dataframe.attrs.get("source_path")
    result.attrs["load_audit"] = dataframe.attrs.get("load_audit")
    return result


def compute_asset_preference(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize completed-order asset mix by count and volume."""
    validate_transaction_frame(dataframe)
    completed_only = dataframe[dataframe["tx_status"] == "Completed"].copy()

    if completed_only.empty:
        columns = [
            "crypto_asset",
            "completed_tx_count",
            "completed_tx_share",
            "completed_volume_cad",
            "completed_volume_share",
        ]
        return pd.DataFrame(columns=columns)

    summary = (
        completed_only.groupby("crypto_asset")
        .agg(
            completed_tx_count=("tx_id", "size"),
            completed_volume_cad=("fiat_volume_cad", "sum"),
        )
        .sort_values(["completed_tx_count", "completed_volume_cad"], ascending=False)
        .reset_index()
    )

    total_completed_count = summary["completed_tx_count"].sum()
    total_completed_volume = summary["completed_volume_cad"].sum()
    summary["completed_tx_share"] = (
        summary["completed_tx_count"] / total_completed_count * 100
    )
    summary["completed_volume_share"] = (
        summary["completed_volume_cad"] / total_completed_volume * 100
    )

    summary.attrs["source_path"] = dataframe.attrs.get("source_path")
    summary.attrs["load_audit"] = dataframe.attrs.get("load_audit")
    return summary


def summarize_report_kpis(
    daily_metrics: pd.DataFrame,
    anomalies: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build top-line KPI values for the HTML report header."""
    if daily_metrics.empty:
        raise ValueError("daily_metrics must not be empty")

    latest_row = daily_metrics.iloc[-1]
    anomaly_count = 0 if anomalies is None else int(anomalies["is_anomaly"].sum())

    return {
        "window_start_date": daily_metrics["local_date"].min().isoformat(),
        "window_end_date": daily_metrics["local_date"].max().isoformat(),
        "total_submitted_tx_count": int(daily_metrics["submitted_tx_count"].sum()),
        "total_completed_tx_count": int(daily_metrics["completed_tx_count"].sum()),
        "total_failed_tx_count": int(daily_metrics["failed_tx_count"].sum()),
        "total_volume_cad": float(daily_metrics["total_volume_cad"].sum()),
        "latest_success_rate": (
            None if pd.isna(latest_row["success_rate"]) else float(latest_row["success_rate"])
        ),
        "latest_failure_rate": (
            None if pd.isna(latest_row["failure_rate"]) else float(latest_row["failure_rate"])
        ),
        "anomaly_day_count": anomaly_count,
    }
