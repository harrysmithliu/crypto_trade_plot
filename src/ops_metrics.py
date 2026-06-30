"""Metric builders for the sectioned operations dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _completed_only(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe[dataframe["tx_status"] == "Completed"].copy()


def build_daily_ttv_trend(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize completed daily transaction volume by asset."""
    completed = _completed_only(dataframe)
    if completed.empty:
        return pd.DataFrame(
            columns=["local_date", "crypto_asset", "total_volume_cad", "tx_count", "daily_total_volume_cad"]
        )

    by_asset = (
        completed.groupby(["local_date", "crypto_asset"])
        .agg(
            total_volume_cad=("fiat_volume_cad", "sum"),
            tx_count=("tx_id", "size"),
        )
        .reset_index()
    )
    totals = (
        completed.groupby("local_date")["fiat_volume_cad"]
        .sum()
        .rename("daily_total_volume_cad")
        .reset_index()
    )
    return by_asset.merge(totals, on="local_date", how="left")


def build_revenue_spread_breakdown(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily flat-fee and spread revenue."""
    completed = _completed_only(dataframe)
    if completed.empty:
        return pd.DataFrame(
            columns=["local_date", "flat_fee_cad", "spread_income_cad", "total_revenue_cad"]
        )

    summary = (
        completed.groupby("local_date")
        .agg(
            flat_fee_cad=("flat_fee_cad", "sum"),
            spread_income_cad=("spread_income_cad", "sum"),
        )
        .reset_index()
    )
    summary["total_revenue_cad"] = summary["flat_fee_cad"] + summary["spread_income_cad"]
    return summary


def build_crypto_inflow_outflow(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize completed flow mix by asset and direction."""
    completed = _completed_only(dataframe)
    if completed.empty:
        return pd.DataFrame(
            columns=[
                "crypto_asset",
                "flow_direction",
                "total_volume_cad",
                "tx_count",
                "volume_share",
            ]
        )

    summary = (
        completed.groupby(["crypto_asset", "flow_direction"])
        .agg(
            total_volume_cad=("fiat_volume_cad", "sum"),
            tx_count=("tx_id", "size"),
        )
        .reset_index()
    )
    total_volume = summary["total_volume_cad"].sum() or 1.0
    summary["volume_share"] = summary["total_volume_cad"] / total_volume * 100
    return summary.sort_values("total_volume_cad", ascending=False)


def build_payment_funnel_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build the end-to-end order funnel counts."""
    stages = [
        ("Order Created", dataframe["order_created_at"].notna().sum()),
        ("Channel Selected", dataframe["channel_selected_at"].notna().sum()),
        ("KYC Passed", dataframe["kyc_passed_at"].notna().sum()),
        ("Payment Completed", dataframe["payment_completed_at"].notna().sum()),
    ]
    funnel = pd.DataFrame(stages, columns=["stage", "stage_count"])
    start_count = max(int(funnel.iloc[0]["stage_count"]), 1)
    funnel["conversion_rate"] = funnel["stage_count"] / start_count * 100
    return funnel


def build_decline_reason_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize failed transactions by decline reason and payment channel."""
    failed = dataframe[dataframe["tx_status"] == "Failed"].copy()
    if failed.empty:
        return pd.DataFrame(columns=["decline_reason", "payment_channel", "failed_tx_count"])

    summary = (
        failed.groupby(["decline_reason", "payment_channel"])
        .agg(failed_tx_count=("tx_id", "size"))
        .reset_index()
        .sort_values("failed_tx_count", ascending=False)
    )
    return summary


def build_risk_alert_heatmap_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepare high-risk transactions for heatmap and outlier scatter views."""
    enriched = dataframe.copy()
    enriched["risk_bucket"] = pd.cut(
        enriched["risk_score"],
        bins=[0, 40, 60, 75, 90, 100],
        labels=["0-40", "41-60", "61-75", "76-90", "91-100"],
        include_lowest=True,
    )
    return enriched[
        [
            "tx_id",
            "merchant_name",
            "fiat_volume_cad",
            "risk_score",
            "velocity_1h",
            "aml_flag",
            "is_high_risk",
            "ip_country",
            "risk_bucket",
        ]
    ].copy()


def build_top_merchants_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize merchant concentration using a Pareto-ready table."""
    completed = _completed_only(dataframe)
    if completed.empty:
        return pd.DataFrame(
            columns=[
                "merchant_id",
                "merchant_name",
                "total_volume_cad",
                "tx_count",
                "volume_share",
                "cumulative_volume_share",
            ]
        )

    summary = (
        completed.groupby(["merchant_id", "merchant_name"])
        .agg(
            total_volume_cad=("fiat_volume_cad", "sum"),
            tx_count=("tx_id", "size"),
        )
        .reset_index()
        .sort_values("total_volume_cad", ascending=False)
    )
    total_volume = summary["total_volume_cad"].sum() or 1.0
    summary["volume_share"] = summary["total_volume_cad"] / total_volume * 100
    summary["cumulative_volume_share"] = summary["volume_share"].cumsum()
    return summary


def build_user_cohort_retention(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build a weekly cohort retention grid."""
    activity = dataframe[["user_id", "local_date"]].dropna().copy()
    if activity.empty:
        return pd.DataFrame()

    activity["activity_week"] = pd.to_datetime(activity["local_date"]).dt.to_period("W-SUN")
    first_week = activity.groupby("user_id")["activity_week"].min().rename("cohort_week")
    cohort_events = activity.merge(first_week, on="user_id", how="left")
    cohort_events["week_number"] = (
        cohort_events["activity_week"] - cohort_events["cohort_week"]
    ).apply(lambda delta: delta.n)
    grouped = (
        cohort_events.groupby(["cohort_week", "week_number"])["user_id"]
        .nunique()
        .rename("active_users")
        .reset_index()
    )
    cohort_sizes = (
        grouped[grouped["week_number"] == 0][["cohort_week", "active_users"]]
        .rename(columns={"active_users": "cohort_size"})
    )
    retention = grouped.merge(cohort_sizes, on="cohort_week", how="left")
    retention["retention_rate"] = retention["active_users"] / retention["cohort_size"] * 100
    matrix = retention.pivot(
        index="cohort_week",
        columns="week_number",
        values="retention_rate",
    ).sort_index()
    matrix.index = [period.start_time.date().isoformat() for period in matrix.index]
    matrix.columns = [f"Week {int(column)}" for column in matrix.columns]
    return matrix


def build_settlement_latency_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return completed transaction settlement latency observations."""
    completed = _completed_only(dataframe)
    if "settlement_latency_min" not in completed.columns:
        return pd.DataFrame(columns=["tx_id", "crypto_asset", "settlement_latency_min"])
    return completed[
        ["tx_id", "crypto_asset", "payment_channel", "settlement_latency_min"]
    ].dropna(subset=["settlement_latency_min"])


def build_reconciliation_monitor(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reconciliation discrepancies by day."""
    summary = (
        dataframe.groupby("local_date")
        .agg(
            mismatch_tx_count=("recon_status", lambda values: (values == "Mismatch").sum()),
            absolute_recon_delta_cad=("recon_delta_cad", lambda values: np.abs(values).sum()),
        )
        .reset_index()
    )
    summary["is_recon_alert"] = (
        (summary["mismatch_tx_count"] > 0)
        | (summary["absolute_recon_delta_cad"] >= 25)
    )
    return summary
