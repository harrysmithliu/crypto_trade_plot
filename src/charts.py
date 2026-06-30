"""Plotly chart builders for the crypto trade report."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


def _validate_daily_metrics(dataframe: pd.DataFrame) -> None:
    required_columns = {
        "local_date",
        "total_volume_cad",
        "success_rate",
        "completed_tx_count",
        "failed_tx_count",
        "pending_tx_count",
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required daily metric columns: {missing}")


def build_volume_success_figure(daily_metrics: pd.DataFrame) -> go.Figure:
    """Create the dual-axis daily volume and success-rate trend chart."""
    _validate_daily_metrics(daily_metrics)
    chart_data = daily_metrics.copy()
    chart_data["date_label"] = pd.Series(chart_data["local_date"]).astype(str)

    anomaly_reason = (
        chart_data["anomaly_reason"]
        if "anomaly_reason" in chart_data.columns
        else pd.Series([""] * len(chart_data))
    )
    is_anomaly = (
        chart_data["is_anomaly"]
        if "is_anomaly" in chart_data.columns
        else pd.Series([False] * len(chart_data))
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=chart_data["local_date"],
            y=chart_data["total_volume_cad"],
            name="Daily Settled Volume (CAD)",
            marker_color="#0f766e",
            customdata=list(
                zip(
                    chart_data["completed_tx_count"],
                    chart_data["failed_tx_count"],
                    chart_data["pending_tx_count"],
                    is_anomaly,
                    anomaly_reason,
                )
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Settled Volume: CAD %{y:,.2f}<br>"
                "Completed: %{customdata[0]:,}<br>"
                "Failed: %{customdata[1]:,}<br>"
                "Pending: %{customdata[2]:,}<br>"
                "Anomaly: %{customdata[3]}<br>"
                "Reason: %{customdata[4]}<extra></extra>"
            ),
            yaxis="y1",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data["local_date"],
            y=chart_data["success_rate"],
            name="Success Rate (%)",
            mode="lines+markers",
            line={"color": "#1d4ed8", "width": 3},
            marker={"size": 8, "color": "#1d4ed8"},
            customdata=list(
                zip(
                    chart_data["total_volume_cad"],
                    chart_data["completed_tx_count"],
                    chart_data["failed_tx_count"],
                    chart_data["pending_tx_count"],
                    is_anomaly,
                    anomaly_reason,
                )
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Success Rate: %{y:.2f}%<br>"
                "Settled Volume: CAD %{customdata[0]:,.2f}<br>"
                "Completed: %{customdata[1]:,}<br>"
                "Failed: %{customdata[2]:,}<br>"
                "Pending: %{customdata[3]:,}<br>"
                "Anomaly: %{customdata[4]}<br>"
                "Reason: %{customdata[5]}<extra></extra>"
            ),
            yaxis="y2",
        )
    )

    anomaly_points = chart_data[is_anomaly.fillna(False)]
    if not anomaly_points.empty:
        figure.add_trace(
            go.Scatter(
                x=anomaly_points["local_date"],
                y=anomaly_points["success_rate"],
                name="Anomaly Days",
                mode="markers",
                marker={
                    "size": 12,
                    "color": "#dc2626",
                    "symbol": "diamond",
                    "line": {"color": "#7f1d1d", "width": 1},
                },
                customdata=anomaly_points["anomaly_reason"],
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Success Rate: %{y:.2f}%<br>"
                    "Alert: %{customdata}<extra></extra>"
                ),
                yaxis="y2",
            )
        )

    figure.update_layout(
        title="Daily Settled Volume and Success Rate Trend",
        barmode="group",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 40, "t": 70, "b": 40},
        yaxis={
            "title": "Settled Volume (CAD)",
            "rangemode": "tozero",
            "tickprefix": "CAD ",
        },
        yaxis2={
            "title": "Success Rate (%)",
            "overlaying": "y",
            "side": "right",
            "range": [0, 105],
            "ticksuffix": "%",
        },
        xaxis={"title": "Local Date"},
        template="plotly_white",
    )
    return figure


def build_asset_preference_figure(asset_summary: pd.DataFrame) -> go.Figure:
    """Create the completed-order asset preference donut chart."""
    figure = go.Figure()
    if asset_summary.empty:
        figure.add_annotation(
            text="No completed transactions are available in the current window.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font={"size": 16, "color": "#334155"},
        )
        figure.update_layout(
            title="Completed Order Asset Preference",
            template="plotly_white",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return figure

    figure.add_trace(
        go.Pie(
            labels=asset_summary["crypto_asset"],
            values=asset_summary["completed_tx_count"],
            hole=0.55,
            textinfo="label+percent",
            customdata=list(
                zip(
                    asset_summary["completed_tx_count"],
                    asset_summary["completed_tx_share"],
                    asset_summary["completed_volume_cad"],
                )
            ),
            hovertemplate=(
                "Asset: %{label}<br>"
                "Completed Transactions: %{customdata[0]:,}<br>"
                "Completed Share: %{customdata[1]:.2f}%<br>"
                "Settled Volume: CAD %{customdata[2]:,.2f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Completed Order Asset Preference",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        template="plotly_white",
    )
    return figure


def _sample_completed_transactions(
    dataframe: pd.DataFrame,
    *,
    max_points: int | None,
    preserve_top_n: int,
) -> pd.DataFrame:
    """Optionally sample dense scatter-plot inputs while keeping extreme values."""
    if max_points is None or len(dataframe) <= max_points:
        return dataframe.copy()

    preserve_count = min(preserve_top_n, len(dataframe))
    high_amount = dataframe.nlargest(preserve_count, "fiat_volume_cad")
    high_latency = dataframe.nlargest(preserve_count, "latency_ms")
    preserved = (
        pd.concat([high_amount, high_latency])
        .drop_duplicates(subset="tx_id", keep="first")
        .copy()
    )

    remaining_budget = max(max_points - len(preserved), 0)
    remaining = dataframe.loc[~dataframe["tx_id"].isin(preserved["tx_id"])].copy()
    if remaining_budget and not remaining.empty:
        sampled = remaining.sample(
            n=min(remaining_budget, len(remaining)),
            random_state=42,
        )
        preserved = pd.concat([preserved, sampled], ignore_index=True)

    return preserved.sort_values("fiat_volume_cad").reset_index(drop=True)


def build_latency_scatter_figure(
    transactions: pd.DataFrame,
    *,
    max_points: int | None = None,
    preserve_top_n: int = 50,
) -> go.Figure:
    """Create the completed-order amount-versus-latency scatter chart."""
    required_columns = {
        "tx_id",
        "local_timestamp",
        "merchant_id",
        "crypto_asset",
        "fiat_volume_cad",
        "tx_status",
        "latency_ms",
    }
    missing_columns = sorted(required_columns.difference(transactions.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required scatter columns: {missing}")

    completed_only = transactions[transactions["tx_status"] == "Completed"].copy()
    completed_only = _sample_completed_transactions(
        completed_only,
        max_points=max_points,
        preserve_top_n=preserve_top_n,
    )

    figure = go.Figure()
    if completed_only.empty:
        figure.add_annotation(
            text="No completed transactions are available for the latency scatter plot.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font={"size": 16, "color": "#334155"},
        )
        figure.update_layout(
            title="Large Transactions and Processing Latency",
            template="plotly_white",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return figure

    for asset in sorted(completed_only["crypto_asset"].unique()):
        asset_slice = completed_only[completed_only["crypto_asset"] == asset]
        figure.add_trace(
            go.Scatter(
                x=asset_slice["fiat_volume_cad"],
                y=asset_slice["latency_ms"],
                name=asset,
                mode="markers",
                marker={"size": 9, "opacity": 0.75},
                customdata=list(
                    zip(
                        asset_slice["tx_id"],
                        asset_slice["merchant_id"],
                        asset_slice["local_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    )
                ),
                hovertemplate=(
                    "Transaction ID: %{customdata[0]}<br>"
                    "Merchant: %{customdata[1]}<br>"
                    "Asset: "
                    + asset
                    + "<br>"
                    "Local Time: %{customdata[2]}<br>"
                    "Amount: CAD %{x:,.2f}<br>"
                    "Latency: %{y:,.0f} ms<extra></extra>"
                ),
            )
        )

    p95_latency = completed_only["latency_ms"].quantile(0.95)
    figure.add_hline(
        y=p95_latency,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"P95 Latency: {p95_latency:.0f} ms",
        annotation_position="top left",
    )
    figure.update_layout(
        title="Large Transactions and Processing Latency",
        xaxis={
            "title": "Transaction Amount (CAD, log scale)",
            "type": "log",
        },
        yaxis={"title": "Processing Latency (ms)"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 40, "t": 70, "b": 40},
        template="plotly_white",
    )
    return figure
