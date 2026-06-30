"""Plotly chart builders for the sectioned operations dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _empty_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font={"size": 16, "color": "#475569"},
    )
    figure.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 24, "r": 24, "t": 24, "b": 24},
    )
    return figure


def create_daily_ttv_trend_figure(summary: pd.DataFrame) -> go.Figure:
    title = "Daily TTV / TPV Trend"
    if summary.empty:
        return _empty_figure(title, "No completed volume is available in the current window.")

    figure = go.Figure()
    totals = summary.drop_duplicates("local_date")[["local_date", "daily_total_volume_cad"]]
    figure.add_trace(
        go.Scatter(
            x=totals["local_date"],
            y=totals["daily_total_volume_cad"],
            mode="lines+markers",
            name="Total Volume",
            line={"width": 4, "color": "#0f766e"},
        )
    )
    for asset in summary["crypto_asset"].dropna().unique():
        asset_slice = summary[summary["crypto_asset"] == asset]
        figure.add_trace(
            go.Scatter(
                x=asset_slice["local_date"],
                y=asset_slice["total_volume_cad"],
                mode="lines+markers",
                name=str(asset),
            )
        )
    figure.update_layout(
        xaxis={"title": "Local Date"},
        yaxis={"title": "Settled Volume (CAD)", "tickprefix": "CAD "},
        hovermode="x unified",
        margin={"l": 56, "r": 20, "t": 24, "b": 56},
        template="plotly_white",
    )
    return figure


def create_revenue_spread_figure(summary: pd.DataFrame) -> go.Figure:
    title = "Revenue & Spread Analysis"
    if summary.empty:
        return _empty_figure(title, "No completed revenue is available in the current window.")

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=summary["local_date"],
            y=summary["flat_fee_cad"],
            name="Flat Fees",
            marker_color="#1d4ed8",
        )
    )
    figure.add_trace(
        go.Bar(
            x=summary["local_date"],
            y=summary["spread_income_cad"],
            name="Spread Income",
            marker_color="#f59e0b",
        )
    )
    figure.update_layout(
        barmode="stack",
        xaxis={"title": "Local Date"},
        yaxis={"title": "Revenue (CAD)", "tickprefix": "CAD "},
        margin={"l": 56, "r": 20, "t": 24, "b": 56},
        template="plotly_white",
    )
    return figure


def create_crypto_inflow_outflow_figure(summary: pd.DataFrame) -> go.Figure:
    title = "Crypto Inflow / Outflow Distribution"
    if summary.empty:
        return _empty_figure(title, "No completed crypto flows are available in the current window.")

    labels = [
        f"{row.crypto_asset} · {row.flow_direction}" for row in summary.itertuples()
    ]
    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=summary["total_volume_cad"],
                hole=0.55,
                customdata=list(
                    zip(
                        summary["tx_count"],
                        summary["volume_share"],
                    )
                ),
                hovertemplate=(
                    "Segment: %{label}<br>"
                    "Volume: CAD %{value:,.2f}<br>"
                    "Transactions: %{customdata[0]:,}<br>"
                    "Volume Share: %{customdata[1]:.2f}%<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 24, "b": 24},
    )
    return figure


def create_payment_funnel_figure(summary: pd.DataFrame) -> go.Figure:
    title = "Payment Conversion & Success Funnel"
    if summary.empty:
        return _empty_figure(title, "No funnel stages are available in the current window.")

    figure = go.Figure(
        go.Funnel(
            y=summary["stage"],
            x=summary["stage_count"],
            text=[f"{value:.1f}%" for value in summary["conversion_rate"]],
            textposition="inside",
            marker={"color": ["#0f766e", "#14b8a6", "#38bdf8", "#1d4ed8"]},
            hovertemplate="%{y}: %{x:,} orders<br>Conversion: %{text}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 32, "r": 24, "t": 24, "b": 24},
    )
    return figure


def create_decline_reason_figure(summary: pd.DataFrame) -> go.Figure:
    title = "Transaction Decline Reason Distribution"
    if summary.empty:
        return _empty_figure(title, "No failed transactions are available in the current window.")

    figure = go.Figure()
    ordered_reasons = (
        summary.groupby("decline_reason")["failed_tx_count"].sum().sort_values().index
    )
    for channel in summary["payment_channel"].dropna().unique():
        channel_slice = (
            summary[summary["payment_channel"] == channel]
            .set_index("decline_reason")
            .reindex(ordered_reasons, fill_value=0)
            .reset_index()
        )
        figure.add_trace(
            go.Bar(
                y=channel_slice["decline_reason"],
                x=channel_slice["failed_tx_count"],
                name=str(channel),
                orientation="h",
            )
        )
    figure.update_layout(
        barmode="stack",
        xaxis={"title": "Failed Transaction Count"},
        yaxis={"title": "Decline Reason"},
        margin={"l": 120, "r": 20, "t": 24, "b": 56},
        template="plotly_white",
    )
    return figure


def create_risk_alert_heatmap_figure(summary: pd.DataFrame) -> go.Figure:
    title = "High-Risk Transaction & AML Alert Heatmap"
    if summary.empty:
        return _empty_figure(title, "No risk observations are available in the current window.")

    figure = go.Figure()
    figure.add_trace(
        go.Histogram2dContour(
            x=summary["fiat_volume_cad"],
            y=summary["risk_score"],
            colorscale="YlOrRd",
            showscale=True,
            contours={"coloring": "heatmap"},
            name="Risk Density",
        )
    )
    high_risk = summary[summary["is_high_risk"].fillna(False)]
    if not high_risk.empty:
        figure.add_trace(
            go.Scatter(
                x=high_risk["fiat_volume_cad"],
                y=high_risk["risk_score"],
                mode="markers",
                name="High Risk",
                marker={"color": "#7f1d1d", "size": 9, "opacity": 0.75},
                customdata=list(
                    zip(
                        high_risk["tx_id"],
                        high_risk["merchant_name"],
                        high_risk["ip_country"],
                        high_risk["velocity_1h"],
                    )
                ),
                hovertemplate=(
                    "Transaction: %{customdata[0]}<br>"
                    "Merchant: %{customdata[1]}<br>"
                    "IP Country: %{customdata[2]}<br>"
                    "Velocity 1H: %{customdata[3]}<br>"
                    "Amount: CAD %{x:,.2f}<br>"
                    "Risk Score: %{y:.2f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        xaxis={"title": "Transaction Amount (CAD)", "type": "log"},
        yaxis={"title": "Risk Score"},
        margin={"l": 56, "r": 20, "t": 24, "b": 56},
        template="plotly_white",
    )
    return figure


def create_top_merchants_figure(summary: pd.DataFrame) -> go.Figure:
    title = "Top Merchants by Volume & Concentration"
    if summary.empty:
        return _empty_figure(title, "No merchant volume is available in the current window.")

    top_summary = summary.head(12)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=top_summary["merchant_name"],
            y=top_summary["total_volume_cad"],
            name="Settled Volume",
            marker_color="#0f766e",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=top_summary["merchant_name"],
            y=top_summary["cumulative_volume_share"],
            name="Cumulative Share",
            mode="lines+markers",
            line={"color": "#dc2626", "width": 3},
        ),
        secondary_y=True,
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 32, "r": 20, "t": 24, "b": 64},
    )
    figure.update_yaxes(title_text="Settled Volume (CAD)", secondary_y=False, tickprefix="CAD ")
    figure.update_yaxes(title_text="Cumulative Share (%)", secondary_y=True, ticksuffix="%")
    return figure


def create_user_cohort_retention_figure(matrix: pd.DataFrame) -> go.Figure:
    title = "User Cohort Retention Analysis"
    if matrix.empty:
        return _empty_figure(title, "Not enough user history is available for retention analysis.")

    figure = go.Figure(
        data=[
            go.Heatmap(
                z=matrix.values,
                x=list(matrix.columns),
                y=list(matrix.index),
                colorscale="Blues",
                colorbar={"title": "Retention %"},
                hovertemplate="Cohort: %{y}<br>%{x}: %{z:.2f}%<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        xaxis_title="Weeks Since First Activity",
        yaxis_title="Cohort Week",
        margin={"l": 72, "r": 24, "t": 24, "b": 56},
        template="plotly_white",
    )
    return figure


def create_settlement_latency_figure(summary: pd.DataFrame) -> go.Figure:
    title = "Settlement Latency Distribution"
    if summary.empty:
        return _empty_figure(title, "No completed settlements are available in the current window.")

    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )
    figure.add_trace(
        go.Histogram(
            x=summary["settlement_latency_min"],
            nbinsx=28,
            name="Latency Distribution",
            marker_color="#0f766e",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Box(
            x=summary["settlement_latency_min"],
            name="Latency Spread",
            marker_color="#1d4ed8",
            boxpoints="outliers",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        template="plotly_white",
        showlegend=False,
        margin={"l": 40, "r": 20, "t": 24, "b": 56},
    )
    figure.update_xaxes(title_text="Settlement Latency (minutes)", row=2, col=1)
    figure.update_yaxes(title_text="Transactions", row=1, col=1)
    return figure


def create_reconciliation_monitor_figure(summary: pd.DataFrame) -> go.Figure:
    title = "Reconciliation Discrepancy Monitor"
    if summary.empty:
        return _empty_figure(title, "No reconciliation observations are available in the current window.")

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=summary["local_date"],
            y=summary["absolute_recon_delta_cad"],
            name="Absolute Recon Delta",
            mode="lines+markers",
            line={"color": "#1d4ed8", "width": 3},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=summary["local_date"],
            y=summary["mismatch_tx_count"],
            name="Mismatch Count",
            marker_color="#f59e0b",
            opacity=0.55,
        ),
        secondary_y=True,
    )
    alerts = summary[summary["is_recon_alert"]]
    if not alerts.empty:
        figure.add_trace(
            go.Scatter(
                x=alerts["local_date"],
                y=alerts["absolute_recon_delta_cad"],
                name="Recon Alerts",
                mode="markers",
                marker={"color": "#dc2626", "size": 12, "symbol": "diamond"},
            ),
            secondary_y=False,
        )
    figure.update_layout(
        hovermode="x unified",
        margin={"l": 56, "r": 20, "t": 24, "b": 56},
        template="plotly_white",
    )
    figure.update_yaxes(title_text="Absolute Recon Delta (CAD)", secondary_y=False, tickprefix="CAD ")
    figure.update_yaxes(title_text="Mismatch Transaction Count", secondary_y=True)
    return figure
