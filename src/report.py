"""HTML report assembly for the crypto trade operations dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.io as pio

from .metrics import summarize_report_kpis
from .ops_charts import (
    create_crypto_inflow_outflow_figure,
    create_daily_ttv_trend_figure,
    create_decline_reason_figure,
    create_payment_funnel_figure,
    create_reconciliation_monitor_figure,
    create_revenue_spread_figure,
    create_risk_alert_heatmap_figure,
    create_settlement_latency_figure,
    create_top_merchants_figure,
    create_user_cohort_retention_figure,
)
from .ops_metrics import (
    build_crypto_inflow_outflow,
    build_daily_ttv_trend,
    build_decline_reason_summary,
    build_payment_funnel_metrics,
    build_reconciliation_monitor,
    build_revenue_spread_breakdown,
    build_risk_alert_heatmap_data,
    build_settlement_latency_distribution,
    build_top_merchants_summary,
    build_user_cohort_retention,
)


DEFAULT_REPORT_TITLE = "Crypto Trade Operations Dashboard"


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
    return f"{float(value):,.0f} min"


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
    completed = cleaned_transactions[cleaned_transactions["tx_status"] == "Completed"].copy()
    total_revenue = float(
        completed["flat_fee_cad"].fillna(0).sum() + completed["spread_income_cad"].fillna(0).sum()
    )
    high_risk_count = int(cleaned_transactions["is_high_risk"].fillna(False).sum())
    recon_mismatch_count = int((cleaned_transactions["recon_status"] == "Mismatch").sum())
    top_merchants = build_top_merchants_summary(cleaned_transactions)
    top_merchant_share = (
        None if top_merchants.empty else float(top_merchants.iloc[0]["volume_share"])
    )
    p95_settlement_latency = (
        None
        if completed["settlement_latency_min"].dropna().empty
        else float(completed["settlement_latency_min"].dropna().quantile(0.95))
    )

    cards = [
        ("30-Day Settled Volume", _format_currency(kpis["total_volume_cad"])),
        ("30-Day Revenue", _format_currency(total_revenue)),
        ("Failed Transactions", _format_integer(kpis["total_failed_tx_count"])),
        ("High-Risk Transactions", _format_integer(high_risk_count)),
        ("Top Merchant Concentration", _format_percent(top_merchant_share)),
        ("P95 Settlement Latency", _format_latency(p95_settlement_latency)),
        ("Recon Mismatch Transactions", _format_integer(recon_mismatch_count)),
        ("Failure-Rate Anomaly Days", _format_integer(kpis["anomaly_day_count"])),
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


def _render_notes() -> str:
    return """
    <ul class="notes-list">
      <li>Current dashboard replaces the previous 3-chart daily report layout and is grouped by Financial, Payment Ops / Fraud, Merchant / User, and System / Reconciliation.</li>
      <li>All figures use simulated transactions generated from the project mock-data pipeline and aggregated by local America/Toronto time.</li>
      <li>Revenue uses mock flat-fee and spread-income fields; reconciliation uses simulated provider-vs-ledger deltas for anomaly highlighting.</li>
      <li>Batch-two-level advanced analytics are represented with production-style metrics, but their business rules remain demo assumptions until a real source system is wired in.</li>
    </ul>
    """


def _figure_to_html(figures: list[Any], *, include_plotlyjs: bool = False) -> list[str]:
    html_blocks: list[str] = []
    for index, figure in enumerate(figures):
        html_blocks.append(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs="inline" if include_plotlyjs and index == 0 else False,
                default_width="100%",
                default_height="100%",
                config={"displaylogo": False, "responsive": True},
            )
        )
    return html_blocks


def _render_chart_card(title: str, description: str, figure_html: str, *, tall: bool = False) -> str:
    """Render a chart card with an expand action for modal viewing."""
    body_class = "chart-body tall" if tall else "chart-body"
    return f"""
    <div class="chart-card">
      <div class="chart-card-header">
        <h3>{title}</h3>
        <button class="expand-chart-button" type="button" aria-label="Expand {title}">Expand</button>
      </div>
      <p>{description}</p>
      <div class="{body_class}">{figure_html}</div>
    </div>
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
    """Build a standalone sectioned operations dashboard."""
    del asset_summary, scatter_max_points

    if generated_at is None:
        generated_at = pd.Timestamp.now(tz="America/Toronto")
    elif generated_at.tzinfo is None:
        generated_at = generated_at.tz_localize("America/Toronto")
    else:
        generated_at = generated_at.tz_convert("America/Toronto")

    load_audit = cleaned_transactions.attrs.get("load_audit")
    window_start = daily_metrics["local_date"].min().isoformat()
    window_end = daily_metrics["local_date"].max().isoformat()

    financial_figures = _figure_to_html(
        [
            create_daily_ttv_trend_figure(build_daily_ttv_trend(cleaned_transactions)),
            create_revenue_spread_figure(build_revenue_spread_breakdown(cleaned_transactions)),
            create_crypto_inflow_outflow_figure(build_crypto_inflow_outflow(cleaned_transactions)),
        ],
        include_plotlyjs=True,
    )
    payment_figures = _figure_to_html(
        [
            create_payment_funnel_figure(build_payment_funnel_metrics(cleaned_transactions)),
            create_decline_reason_figure(build_decline_reason_summary(cleaned_transactions)),
            create_risk_alert_heatmap_figure(build_risk_alert_heatmap_data(cleaned_transactions)),
        ],
    )
    merchant_user_figures = _figure_to_html(
        [
            create_top_merchants_figure(build_top_merchants_summary(cleaned_transactions)),
            create_user_cohort_retention_figure(build_user_cohort_retention(cleaned_transactions)),
        ],
    )
    system_figures = _figure_to_html(
        [
            create_settlement_latency_figure(build_settlement_latency_distribution(cleaned_transactions)),
            create_reconciliation_monitor_figure(build_reconciliation_monitor(cleaned_transactions)),
        ],
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
      --panel: #ffffff;
      --ink: #12263a;
      --muted: #52667a;
      --border: #d8e1ea;
      --accent: #0f766e;
      --bg-a: #f7fbff;
      --bg-b: #eef4fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(29, 78, 216, 0.10), transparent 28%),
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 30%),
        linear-gradient(180deg, var(--bg-a), var(--bg-b));
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 30px 24px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, #0f766e, #1d4ed8);
      color: white;
      padding: 30px 34px;
      border-radius: 26px;
      box-shadow: 0 24px 48px rgba(15, 23, 42, 0.14);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 2.1rem;
      line-height: 1.05;
    }}
    .hero p {{
      margin: 6px 0;
      color: rgba(255, 255, 255, 0.92);
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 24px;
      margin-top: 22px;
      box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
      overflow: visible;
    }}
    .section-title {{
      display: flex;
      align-items: baseline;
      gap: 12px;
      margin: 0 0 18px;
    }}
    .section-title h2 {{
      margin: 0;
      font-size: 1.55rem;
    }}
    .section-title span {{
      color: var(--muted);
      font-size: 0.98rem;
    }}
    .quality-grid,
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
    }}
    .quality-item,
    .kpi-card {{
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: linear-gradient(180deg, #ffffff, #f9fbfd);
    }}
    .label {{
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .value {{
      display: block;
      font-weight: 700;
      font-size: 1.2rem;
    }}
    .chart-grid.three {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
    }}
    .chart-grid.two {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .chart-card {{
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      background: #ffffff;
      min-height: auto;
      overflow: visible;
      display: flex;
      flex-direction: column;
      align-self: start;
    }}
    .chart-card-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .chart-card h3 {{
      margin: 0;
      font-size: 1.02rem;
    }}
    .chart-card p {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .expand-chart-button {{
      border: 1px solid var(--border);
      background: #f8fbff;
      color: var(--ink);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.16s ease, background 0.16s ease, border-color 0.16s ease;
      flex-shrink: 0;
    }}
    .expand-chart-button:hover {{
      background: #edf6ff;
      border-color: #b9d2ea;
      transform: translateY(-1px);
    }}
    .chart-body {{
      height: 320px;
      width: 100%;
    }}
    .chart-body.tall {{
      height: 360px;
    }}
    .chart-body .plotly-graph-div,
    .chart-body .js-plotly-plot,
    .chart-body .plot-container,
    .chart-body > div {{
      width: 100% !important;
      height: 100% !important;
      max-width: 100%;
      overflow: hidden !important;
    }}
    .chart-card .plotly-graph-div,
    .chart-card .js-plotly-plot,
    .chart-card .plot-container {{
      width: 100% !important;
      max-width: 100%;
      overflow: visible !important;
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
      margin: 0;
      color: var(--muted);
    }}
    .chart-modal {{
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.55);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      z-index: 2000;
    }}
    .chart-modal.is-open {{
      display: flex;
    }}
    .chart-modal-dialog {{
      width: min(1180px, 96vw);
      max-height: 92vh;
      background: #ffffff;
      border-radius: 24px;
      box-shadow: 0 32px 64px rgba(15, 23, 42, 0.22);
      padding: 24px 24px 18px;
      display: flex;
      flex-direction: column;
    }}
    .chart-modal-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 14px;
    }}
    .chart-modal-header h3 {{
      margin: 0 0 6px;
      font-size: 1.35rem;
    }}
    .chart-modal-header p {{
      margin: 0;
      color: var(--muted);
    }}
    .chart-modal-close {{
      border: none;
      background: #12263a;
      color: white;
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      flex-shrink: 0;
    }}
    .chart-modal-graph {{
      min-height: 0;
      height: min(72vh, 820px);
    }}
    @media (max-width: 1180px) {{
      .chart-grid.three,
      .chart-grid.two {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .chart-body {{
        height: 300px;
      }}
      .chart-body.tall {{
        height: 340px;
      }}
      .chart-modal {{
        padding: 14px;
      }}
      .chart-modal-dialog {{
        width: 100%;
        max-height: 94vh;
        padding: 18px 16px 14px;
      }}
      .chart-modal-header {{
        flex-direction: column;
      }}
      .chart-modal-graph {{
        height: 68vh;
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
      <p>This dashboard replaces the previous 3-chart daily report with a sectioned 10-chart operations board.</p>
    </section>

    <section class="section">
      <div class="section-title">
        <h2>Data Quality Summary</h2>
        <span>Input validation and cleaning audit</span>
      </div>
      {_render_quality_summary(load_audit)}
    </section>

    <section class="section">
      <div class="section-title">
        <h2>Executive KPI Cards</h2>
        <span>Cross-functional daily operating pulse</span>
      </div>
      <div class="kpi-grid">
        {_render_kpi_cards(cleaned_transactions, daily_metrics, anomaly_metrics)}
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h2>1. Financial</h2>
        <span>Revenue, flow mix, and daily transaction volume</span>
      </div>
      <div class="chart-grid three">
        {_render_chart_card("Daily TTV / TPV Trend", "Daily settled transaction volume by major asset and total platform flow.", financial_figures[0])}
        {_render_chart_card("Revenue &amp; Spread Analysis", "Mock platform income split into flat fees and spread contribution.", financial_figures[1])}
        {_render_chart_card("Crypto Inflow / Outflow Distribution", "Completed crypto movement mix across assets and flow directions.", financial_figures[2])}
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h2>2. Payment Ops / Fraud</h2>
        <span>Conversion efficiency, failure root cause, and high-risk activity</span>
      </div>
      <div class="chart-grid three">
        {_render_chart_card("Payment Conversion &amp; Success Funnel", "Order-stage conversion from creation through payment completion.", payment_figures[0], tall=True)}
        {_render_chart_card("Transaction Decline Reason Distribution", "Failed transaction root causes split by payment channel.", payment_figures[1], tall=True)}
        {_render_chart_card("High-Risk Transaction &amp; AML Alert Heatmap", "Risk density and highlighted high-risk outliers for operational review.", payment_figures[2], tall=True)}
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h2>3. Merchant / User</h2>
        <span>Merchant concentration and user retention</span>
      </div>
      <div class="chart-grid two">
        {_render_chart_card("Top Merchants by Volume &amp; Concentration", "Pareto-style concentration view of the merchants driving platform volume.", merchant_user_figures[0])}
        {_render_chart_card("User Cohort Retention Analysis", "Weekly retention heatmap based on first observed platform activity.", merchant_user_figures[1], tall=True)}
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h2>4. System / Reconciliation</h2>
        <span>Settlement timeliness and reconciliation health</span>
      </div>
      <div class="chart-grid two">
        {_render_chart_card("Settlement Latency Distribution", "Histogram and spread view of completed transaction settlement times.", system_figures[0], tall=True)}
        {_render_chart_card("Reconciliation Discrepancy Monitor", "Daily mismatch counts and absolute provider-versus-ledger delta alerts.", system_figures[1])}
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h2>Metric Notes</h2>
        <span>Assumptions for the demo dashboard</span>
      </div>
      {_render_notes()}
    </section>
  </main>
  <div class="chart-modal" id="chart-modal" aria-hidden="true">
    <div class="chart-modal-dialog" role="dialog" aria-modal="true" aria-labelledby="chart-modal-title">
      <div class="chart-modal-header">
        <div>
          <h3 id="chart-modal-title"></h3>
          <p id="chart-modal-description"></p>
        </div>
        <button type="button" class="chart-modal-close" id="chart-modal-close">Close</button>
      </div>
      <div class="chart-modal-graph" id="chart-modal-graph"></div>
    </div>
  </div>
  <script>
    (() => {{
      const modal = document.getElementById("chart-modal");
      const modalTitle = document.getElementById("chart-modal-title");
      const modalDescription = document.getElementById("chart-modal-description");
      const modalGraph = document.getElementById("chart-modal-graph");
      const modalClose = document.getElementById("chart-modal-close");

      function cloneSerializable(value) {{
        if (window.structuredClone) {{
          return window.structuredClone(value);
        }}
        return JSON.parse(JSON.stringify(value));
      }}

      function closeModal() {{
        modal.classList.remove("is-open");
        modal.setAttribute("aria-hidden", "true");
        modalGraph.innerHTML = "";
      }}

      function openModal(card) {{
        const plot = card.querySelector(".plotly-graph-div");
        if (!plot || !window.Plotly || !plot.data || !plot.layout) {{
          return;
        }}

        modalTitle.textContent = card.querySelector("h3")?.textContent || "Expanded Chart";
        modalDescription.textContent = card.querySelector("p")?.textContent || "";
        modal.classList.add("is-open");
        modal.setAttribute("aria-hidden", "false");

        const data = cloneSerializable(plot.data);
        const layout = cloneSerializable(plot.layout);
        layout.autosize = true;
        delete layout.width;
        delete layout.height;
        layout.margin = Object.assign({{ l: 72, r: 32, t: 28, b: 64 }}, layout.margin || {{}});

        window.Plotly.newPlot(
          modalGraph,
          data,
          layout,
          {{
            displaylogo: false,
            responsive: true,
          }}
        ).then(() => {{
          window.Plotly.Plots.resize(modalGraph);
        }});
      }}

      document.querySelectorAll(".expand-chart-button").forEach((button) => {{
        button.addEventListener("click", () => {{
          const card = button.closest(".chart-card");
          if (card) {{
            openModal(card);
          }}
        }});
      }});

      modalClose.addEventListener("click", closeModal);
      modal.addEventListener("click", (event) => {{
        if (event.target === modal) {{
          closeModal();
        }}
      }});
      document.addEventListener("keydown", (event) => {{
        if (event.key === "Escape" && modal.classList.contains("is-open")) {{
          closeModal();
        }}
      }});
    }})();
  </script>
</body>
</html>
"""


def write_html_report(
    html: str,
    *,
    output_dir: str | Path,
    report_date: str,
    report_prefix: str = "Crypto_Trade_Daily_Report",
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
