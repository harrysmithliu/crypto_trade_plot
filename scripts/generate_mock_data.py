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
            records.append(
                {
                    "tx_id": (
                        f"TX-{local_day.strftime('%Y%m%d')}-"
                        f"{row_index + 1:04d}"
                    ),
                    "timestamp": local_timestamps[row_index].tz_convert("UTC"),
                    "merchant_id": f"M{rng.integers(1001, 1021)}",
                    "crypto_asset": assets[row_index],
                    "fiat_volume_cad": round(float(amounts[row_index]), 2),
                    "tx_status": statuses[row_index],
                    "latency_ms": int(latency[row_index]),
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
