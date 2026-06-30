"""Generate deterministic mock crypto transaction data for the demo report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = Path("data/raw/crypto_transactions_30d.csv")
DEFAULT_TIMEZONE = "America/Toronto"
ASSETS = ("USDC", "USDT", "BTC", "ETH")
ASSET_PROBABILITIES = (0.55, 0.25, 0.12, 0.08)
STATUSES = ("Completed", "Pending", "Failed")
FIAT_CURRENCIES = ("CAD", "USD")
FIAT_CURRENCY_PROBABILITIES = (0.72, 0.28)
PAYMENT_CHANNELS = (
    "Interac e-Transfer",
    "Wire Transfer",
    "Crypto Network",
    "Card",
)
PAYMENT_CHANNEL_PROBABILITIES = (0.42, 0.22, 0.20, 0.16)
FLOW_DIRECTIONS = ("Inflow", "Outflow")
FLOW_DIRECTION_PROBABILITIES = (0.58, 0.42)
IP_COUNTRIES = ("CA", "US", "GB", "HK", "SG", "NG")
IP_COUNTRY_PROBABILITIES = (0.48, 0.21, 0.10, 0.08, 0.08, 0.05)
FAILED_REASON_WEIGHTS = {
    "Interac e-Transfer": {
        "Insufficient Funds": 0.34,
        "API Timeout": 0.21,
        "Fraud Block": 0.19,
        "KYC Review": 0.12,
        "Bank Rejection": 0.14,
    },
    "Wire Transfer": {
        "Insufficient Funds": 0.18,
        "API Timeout": 0.15,
        "Fraud Block": 0.11,
        "KYC Review": 0.24,
        "Bank Rejection": 0.32,
    },
    "Crypto Network": {
        "Insufficient Funds": 0.09,
        "API Timeout": 0.31,
        "Fraud Block": 0.17,
        "KYC Review": 0.11,
        "Network Congestion": 0.32,
    },
    "Card": {
        "Insufficient Funds": 0.28,
        "API Timeout": 0.16,
        "Fraud Block": 0.34,
        "KYC Review": 0.08,
        "Bank Rejection": 0.14,
    },
}
PROCESSOR_CODES = {
    "Insufficient Funds": "DECLINE_051",
    "API Timeout": "PROC_TIMEOUT",
    "Fraud Block": "AML_BLOCK",
    "KYC Review": "KYC_PENDING",
    "Bank Rejection": "BANK_DECLINE",
    "Network Congestion": "CHAIN_DELAY",
}


def _parse_end_date(end_date: str | pd.Timestamp | None, timezone: str) -> pd.Timestamp:
    """Return a normalized timezone-aware local end date."""
    if end_date is None:
        return pd.Timestamp.now(tz=timezone).normalize()

    parsed = pd.Timestamp(end_date)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(timezone)
    else:
        parsed = parsed.tz_convert(timezone)
    return parsed.normalize()


def _allocate_categories(
    rng: np.random.Generator,
    categories: tuple[str, ...],
    probabilities: tuple[float, ...],
    size: int,
) -> np.ndarray:
    """Allocate deterministic category counts, then shuffle their row order."""
    expected = np.asarray(probabilities, dtype=float) * size
    counts = np.floor(expected).astype(int)
    remainder = size - int(counts.sum())

    if remainder:
        fractional_order = np.argsort(-(expected - counts))
        counts[fractional_order[:remainder]] += 1

    values = np.repeat(np.asarray(categories, dtype=object), counts)
    rng.shuffle(values)
    return values


def generate_mock_data(
    *,
    days: int = 30,
    transactions_per_day: int = 100,
    seed: int = 42,
    end_date: str | pd.Timestamp | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Build a reproducible transaction DataFrame containing one anomaly day.

    Normal days contain approximately 95% Completed, 3% Pending and 2% Failed
    transactions. The seventh day from the end contains approximately 82%, 3%
    and 15%, respectively, so the anomaly detector has a known signal.
    """
    if days <= 0:
        raise ValueError("days must be greater than zero")
    if transactions_per_day <= 0:
        raise ValueError("transactions_per_day must be greater than zero")

    rng = np.random.default_rng(seed)
    local_end_date = _parse_end_date(end_date, timezone)
    local_dates = pd.date_range(
        end=local_end_date,
        periods=days,
        freq="D",
        tz=timezone,
    )
    anomaly_index = max(0, days - 7)
    records: list[dict[str, object]] = []

    for day_index, local_day in enumerate(local_dates):
        is_anomaly_day = day_index == anomaly_index
        status_probabilities = (
            (0.82, 0.03, 0.15) if is_anomaly_day else (0.95, 0.03, 0.02)
        )

        statuses = _allocate_categories(
            rng,
            STATUSES,
            status_probabilities,
            transactions_per_day,
        )
        assets = _allocate_categories(
            rng,
            ASSETS,
            ASSET_PROBABILITIES,
            transactions_per_day,
        )
        fiat_currencies = _allocate_categories(
            rng,
            FIAT_CURRENCIES,
            FIAT_CURRENCY_PROBABILITIES,
            transactions_per_day,
        )
        payment_channels = _allocate_categories(
            rng,
            PAYMENT_CHANNELS,
            PAYMENT_CHANNEL_PROBABILITIES,
            transactions_per_day,
        )
        flow_directions = _allocate_categories(
            rng,
            FLOW_DIRECTIONS,
            FLOW_DIRECTION_PROBABILITIES,
            transactions_per_day,
        )
        ip_countries = _allocate_categories(
            rng,
            IP_COUNTRIES,
            IP_COUNTRY_PROBABILITIES,
            transactions_per_day,
        )
        seconds = rng.integers(0, 86_400, size=transactions_per_day)
        local_timestamps = local_day + pd.to_timedelta(seconds, unit="s")

        # A log-normal distribution creates many ordinary transactions and a
        # small number of high-value transactions, as expected in payment data.
        amounts = np.clip(
            rng.lognormal(mean=5.5, sigma=1.1, size=transactions_per_day),
            5,
            100_000,
        )

        # Latency is noisy, with a deliberately weak positive relationship to
        # transaction value and a small population of slow outliers.
        latency = (
            rng.lognormal(mean=6.2, sigma=0.5, size=transactions_per_day)
            + np.sqrt(amounts) * 4
        ).astype(int)
        slow_mask = rng.random(transactions_per_day) < 0.01
        latency[slow_mask] *= 5

        for row_index in range(transactions_per_day):
            amount = float(amounts[row_index])
            status = str(statuses[row_index])
            asset = str(assets[row_index])
            payment_channel = str(payment_channels[row_index])
            fiat_currency = str(fiat_currencies[row_index])
            flow_direction = str(flow_directions[row_index])
            ip_country = str(ip_countries[row_index])
            merchant_number = int(rng.integers(1001, 1021))
            merchant_id = f"M{merchant_number}"
            merchant_name = f"Merchant {merchant_number}"
            user_id = f"U{int(rng.integers(1, 751)):05d}"

            order_created_local = local_timestamps[row_index] - pd.Timedelta(
                seconds=int(rng.integers(45, 4_200))
            )
            channel_selected_local = (
                order_created_local + pd.Timedelta(seconds=int(rng.integers(20, 600)))
                if rng.random() >= 0.03
                else pd.NaT
            )
            kyc_passed_local = (
                channel_selected_local + pd.Timedelta(seconds=int(rng.integers(30, 1_500)))
                if pd.notna(channel_selected_local) and rng.random() >= 0.08
                else pd.NaT
            )
            payment_completed_local = (
                local_timestamps[row_index] if status == "Completed" else pd.NaT
            )

            settlement_latency_min = pd.NA
            settled_local = pd.NaT
            if status == "Completed":
                latency_base = 18 if asset in {"USDC", "USDT"} else 42
                settlement_latency_min = int(
                    np.clip(
                        rng.lognormal(mean=np.log(latency_base), sigma=0.55),
                        2,
                        1_440,
                    )
                )
                settled_local = payment_completed_local + pd.Timedelta(
                    minutes=int(settlement_latency_min)
                )

            flat_fee_cad = 0.0
            spread_income_cad = 0.0
            if status == "Completed":
                flat_fee_cad = round(0.85 + amount * 0.0014, 2)
                spread_bps = 0.0021 if asset in {"BTC", "ETH"} else 0.0012
                spread_income_cad = round(amount * spread_bps, 2)

            decline_reason = ""
            processor_code = "APPROVED" if status == "Completed" else "PENDING"
            if status == "Failed":
                reason_weights = FAILED_REASON_WEIGHTS[payment_channel]
                decline_reason = rng.choice(
                    tuple(reason_weights.keys()),
                    p=tuple(reason_weights.values()),
                )
                processor_code = PROCESSOR_CODES[decline_reason]

            velocity_1h = int(rng.poisson(1.6))
            risk_score = (
                rng.beta(2.2, 5.0) * 100
                + np.log10(max(amount, 10)) * 6
                + velocity_1h * 3.5
                + (24 if decline_reason == "Fraud Block" else 0)
                + (14 if ip_country == "NG" else 0)
                + (9 if is_anomaly_day else 0)
            )
            risk_score = round(float(np.clip(risk_score, 1, 99)), 2)
            aml_flag = bool(
                risk_score >= 89
                or (amount >= 18_000 and ip_country in {"NG", "HK"})
                or velocity_1h >= 6
            )
            is_high_risk = bool(risk_score >= 85 or aml_flag)

            ledger_amount_cad = round(amount, 2)
            recon_delta_cad = 0.0
            recon_probability = 0.05 if is_anomaly_day else 0.018
            if rng.random() < recon_probability:
                recon_delta_cad = round(float(rng.normal(loc=0.0, scale=18.0)), 2)
                if recon_delta_cad == 0:
                    recon_delta_cad = 2.75
            provider_amount_cad = round(ledger_amount_cad + recon_delta_cad, 2)
            recon_status = "Mismatch" if abs(recon_delta_cad) >= 1 else "Matched"

            records.append(
                {
                    "tx_id": f"TX-{local_day.strftime('%Y%m%d')}-{row_index + 1:04d}",
                    "order_id": f"ORD-{local_day.strftime('%Y%m%d')}-{row_index + 1:04d}",
                    "user_id": user_id,
                    "timestamp": local_timestamps[row_index].tz_convert("UTC"),
                    "merchant_id": merchant_id,
                    "merchant_name": merchant_name,
                    "crypto_asset": asset,
                    "fiat_currency": fiat_currency,
                    "fiat_volume_cad": round(amount, 2),
                    "tx_status": status,
                    "payment_channel": payment_channel,
                    "decline_reason": decline_reason,
                    "processor_code": processor_code,
                    "flat_fee_cad": flat_fee_cad,
                    "spread_income_cad": spread_income_cad,
                    "flow_direction": flow_direction,
                    "risk_score": risk_score,
                    "aml_flag": aml_flag,
                    "ip_country": ip_country,
                    "velocity_1h": velocity_1h,
                    "is_high_risk": is_high_risk,
                    "order_created_at": order_created_local.tz_convert("UTC"),
                    "channel_selected_at": (
                        channel_selected_local.tz_convert("UTC")
                        if pd.notna(channel_selected_local)
                        else pd.NaT
                    ),
                    "kyc_passed_at": (
                        kyc_passed_local.tz_convert("UTC")
                        if pd.notna(kyc_passed_local)
                        else pd.NaT
                    ),
                    "payment_completed_at": (
                        payment_completed_local.tz_convert("UTC")
                        if pd.notna(payment_completed_local)
                        else pd.NaT
                    ),
                    "settled_at": (
                        settled_local.tz_convert("UTC") if pd.notna(settled_local) else pd.NaT
                    ),
                    "provider_amount_cad": provider_amount_cad,
                    "ledger_amount_cad": ledger_amount_cad,
                    "recon_delta_cad": recon_delta_cad,
                    "recon_status": recon_status,
                    "latency_ms": int(latency[row_index]),
                    "settlement_latency_min": settlement_latency_min,
                }
            )

    dataframe = (
        pd.DataFrame.from_records(records)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    dataframe.attrs["anomaly_date"] = local_dates[anomaly_index].date().isoformat()
    dataframe.attrs["timezone"] = timezone
    return dataframe


def write_mock_data(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write mock data as a UTF-8 CSV using ISO 8601 UTC timestamps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic mock crypto transaction data."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--transactions-per-day", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--end-date",
        help="Final local date in YYYY-MM-DD format; defaults to today.",
    )
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataframe = generate_mock_data(
        days=args.days,
        transactions_per_day=args.transactions_per_day,
        seed=args.seed,
        end_date=args.end_date,
        timezone=args.timezone,
    )
    write_mock_data(dataframe, args.output)

    print(f"Generated {len(dataframe):,} rows: {args.output}")
    print(f"Date window: {dataframe['timestamp'].min()} to {dataframe['timestamp'].max()}")
    print(f"Injected anomaly date ({args.timezone}): {dataframe.attrs['anomaly_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
