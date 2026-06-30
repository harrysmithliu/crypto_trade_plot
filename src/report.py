"""HTML report assembly for the crypto trade dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.io as pio

from .charts import (
    build_asset_preference_figure,
    build_latency_scatter_figure,
    build_volume_success_figure,
)
from .metrics import summarize_report_kpis


DEFAULT_REPORT_TITLE = "Crypto Trade Daily Report"


@dataclass(frozen=True)
class ReportPaths:
    """Output paths for the dated and rolling latest HTML files."""

    dated_report_path: Path
    latest_report_path: Path


def _format_integer(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def _format_currency(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"CAD {float(value):,.2f}"


def _format_percent(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def _format_latency(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.0f} ms"


def _render_quality_summary(load_audit: dict[str, Any] | None) -> str:
    if not load_audit:
        return "<p class='empty-state'>No load audit is available for this report run.</p>"

    input_rows = int(load_audit.get("input_rows", 0))
    invalid_rows = int(load_audit.get("invalid_rows", 0))
    duplicate_rows = int(load_audit.get("duplicate_rows", 0))
    valid_rows = int(load_audit.get("output_rows", 0))

    return f"""
    <div class="quality-grid">
      <div class="quality-item"><span class="label">Input Rows</span><span class="value">{input_rows:,}</span></div>
      <div class="quality-item"><span class="label">Valid Rows</span><span class="value">{valid_rows:,}</span></div>
      <div class="quality-item"><span class="label">Invalid Rows</span><span class="value">{invalid_rows:,}</span></div>
      <div class="quality-item"><span class="label">Duplicate Rows</span><span class="value">{duplicate_rows:,}</span></div>
    </div>
    """


def _render_kpi_cards(
    cleaned_transactions: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    anomaly_metrics: pd.DataFrame,
) -> str:
    kpis = summarize_report_kpis(daily_metrics, anomaly_metrics)
    latest_day = daily_metrics.iloc[-1]
    completed_only = cleaned_transactions[cleaned_transactions["tx_status"] == "Completed"]
    overall_p95_latency = (
        None if completed_only.empty else completed_only["latency_ms"].quantile(0.95)
    )

    cards = [
        ("30-Day Settled Volume", _format_currency(kpis["total_volume_cad"])),
        ("Latest Day Settled Volume", _format_currency(latest_day["total_volume_cad"])),
        ("Latest Day Success Rate", _format_percent(latest_day["success_rate"])),
        ("30-Day Pending Transactions", _format_integer(daily_metrics["pending_tx_count"].sum())),
        ("Anomaly Days", _format_integer(int(anomaly_metrics["is_anomaly"].sum()))),
        ("30-Day Completed P95 Latency", _format_latency(overall_p95_latency)),
    ]

    return "".join(
        f"""
        <div class="kpi-card">
          <span class="label">{label}</span>
          <span class="value">{value}</span>
        </div>
        """
        for label, value in cards
    )


def _render_anomaly_table(anomaly_metrics: pd.DataFrame) -> str:
    anomaly_rows = anomaly_metrics[anomaly_metrics["is_anomaly"]].copy()
    if anomaly_rows.empty:
        return "<p class='empty-state'>No anomaly days were detected in the current 30-day window.</p>"

    rows_html = []
    for _, row in anomaly_rows.iterrows():
        rows_html.append(
            f"""
            <tr>
              <td>{row["local_date"].isoformat()}</td>
              <td>{_format_integer(int(row["terminal_tx_count"]))}</td>
              <td>{_format_percent(row["failure_rate"])}</td>
              <td>{_format_percent(row["baseline_failure_rate"])}</td>
              <td>{row["anomaly_reason"]}</td>
            </tr>
            """
        )

    return f"""
    <table class="detail-table">
      <thead>
        <tr>
          <th>Date</th>
          <th>Terminal Transactions</th>
          <th>Failure Rate</th>
          <th>Baseline Failure Rate</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """


def _render_notes() -> str:
    return """
    <ul class="notes-list">
      <li>All metrics are aggregated by the local America/Toronto calendar date.</li>
      <li>Settled volume includes only completed transactions and is shown in CAD.</li>
      <li>Success rate is calculated as Completed / (Completed + Failed) × 100%.</li>
      <li>Pending transactions are counted by transaction date and excluded from success-rate denominators.</li>
      <li>Failure-rate anomaly days require a minimum terminal transaction count and an elevated failure rate versus historical baseline.</li>
    </ul>
    """


def render_html_report(
    *,
    cleaned_transactions: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    anomaly_metrics: pd.DataFrame,
    asset_summary: pd.DataFrame,
    generated_at: pd.Timestamp | None = None,
    report_title: str = DEFAULT_REPORT_TITLE,
    scatter_max_points: int | None = None,
) -> str:
    """Build a standalone offline HTML report with embedded Plotly figures."""
    if generated_at is None:
        generated_at = pd.Timestamp.now(tz="America/Toronto")
    elif generated_at.tzinfo is None:
        generated_at = generated_at.tz_localize("America/Toronto")
    else:
        generated_at = generated_at.tz_convert("America/Toronto")

    load_audit = cleaned_transactions.attrs.get("load_audit")
    window_start = daily_metrics["local_date"].min().isoformat()
    window_end = daily_metrics["local_date"].max().isoformat()

    volume_figure = build_volume_success_figure(anomaly_metrics)
    asset_figure = build_asset_preference_figure(asset_summary)
    scatter_figure = build_latency_scatter_figure(
        cleaned_transactions,
        max_points=scatter_max_points,
    )

    volume_chart_html = pio.to_html(
        volume_figure,
        full_html=False,
        include_plotlyjs="inline",
        config={"displaylogo": False, "responsive": True},
    )
    asset_chart_html = pio.to_html(
        asset_figure,
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
    )
    scatter_chart_html = pio.to_html(
        scatter_figure,
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{report_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f7fb;
      --panel: #ffffff;
      --ink: #102a43;
      --muted: #486581;
      --accent: #0f766e;
      --accent-soft: #dff6f3;
      --border: #d9e2ec;
      --alert: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 32%),
        linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
      color: var(--ink);
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    .hero {{
      background: linear-gradient(135deg, #0f766e, #164e63);
      color: white;
      border-radius: 24px;
      padding: 28px 32px;
      box-shadow: 0 20px 45px rgba(15, 23, 42, 0.12);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 2rem;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 6px 0;
      color: rgba(255, 255, 255, 0.92);
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 24px;
      margin-top: 22px;
      box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
      overflow-x: auto;
    }}
    .section h2 {{
      margin: 0 0 16px;
      font-size: 1.25rem;
    }}
    .quality-grid,
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
    }}
    .quality-item,
    .kpi-card {{
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      background: linear-gradient(180deg, #ffffff, #f8fbff);
    }}
    .label {{
      display: block;
      font-size: 0.88rem;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .value {{
      display: block;
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--ink);
    }}
    .chart-block + .chart-block {{
      margin-top: 20px;
    }}
    .detail-table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}
    .detail-table th,
    .detail-table td {{
      padding: 12px 10px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    .detail-table th {{
      font-size: 0.88rem;
      color: var(--muted);
      background: #f8fafc;
    }}
    .notes-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    .notes-list li + li {{
      margin-top: 8px;
    }}
    .empty-state {{
      color: var(--muted);
      margin: 0;
    }}
    @media (max-width: 720px) {{
      .page {{
        padding: 20px 14px 40px;
      }}
      .hero {{
        padding: 24px 20px;
      }}
      .section {{
        padding: 18px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>{report_title}</h1>
      <p>Generated At: {generated_at.strftime("%Y-%m-%d %H:%M:%S %Z")}</p>
      <p>Data Window: {window_start} to {window_end}</p>
    </section>

    <section class="section">
      <h2>Data Quality Summary</h2>
      {_render_quality_summary(load_audit)}
    </section>

    <section class="section">
      <h2>Key Metrics</h2>
      <div class="kpi-grid">
        {_render_kpi_cards(cleaned_transactions, daily_metrics, anomaly_metrics)}
      </div>
    </section>

    <section class="section chart-block">
      <h2>Daily Settled Volume and Success Rate</h2>
      {volume_chart_html}
    </section>

    <section class="section chart-block">
      <h2>Completed Order Asset Preference</h2>
      {asset_chart_html}
    </section>

    <section class="section chart-block">
      <h2>Large Transactions and Processing Latency</h2>
      {scatter_chart_html}
    </section>

    <section class="section">
      <h2>Anomaly Day Details</h2>
      {_render_anomaly_table(anomaly_metrics)}
    </section>

    <section class="section">
      <h2>Metric Notes</h2>
      {_render_notes()}
    </section>
  </main>
</body>
</html>
"""


def write_html_report(
    html: str,
    *,
    output_dir: str | Path,
    report_date: str,
    report_prefix: str = DEFAULT_REPORT_TITLE.replace(" ", "_"),
) -> ReportPaths:
    """Write both the dated report and rolling latest report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dated_report_path = output_path / f"{report_prefix}_{report_date}.html"
    latest_report_path = output_path / f"{report_prefix}_latest.html"
    dated_report_path.write_text(html, encoding="utf-8")
    latest_report_path.write_text(html, encoding="utf-8")

    return ReportPaths(
        dated_report_path=dated_report_path,
        latest_report_path=latest_report_path,
    )
