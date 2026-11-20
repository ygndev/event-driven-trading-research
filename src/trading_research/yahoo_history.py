"""Acquire and audit daily ETF history from Yahoo Finance via yfinance."""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import astuple, fields
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFException, YFRateLimitError

from .daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
    validate_daily_bars,
)

SYMBOLS = ("SPY", "QQQ", "IWM")
OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
ACTION_COLUMNS = {
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
    "Capital Gains": "capital_gains",
}
HISTORY_OPTIONS = {
    "interval": "1d",
    "prepost": False,
    "auto_adjust": False,
    "back_adjust": False,
    "actions": True,
    "repair": False,
    "keepna": True,
    "rounding": False,
    "timeout": 10,
}
TIMESTAMP_COLUMN = "provider_timestamp"
LIMITATIONS = (
    "Yahoo labels Close as split-adjusted. The provider OHLC is carried "
    "through on its reported basis, without independent verification of "
    "the price or volume adjustment factors.",
    "Adj Close is separate and may reflect splits, dividends and capital "
    "gains. yfinance can copy Close when the upstream Adj Close is absent.",
    "yfinance can sort and deduplicate rows, merge events and replace "
    "missing Volume with zero before returning its table. Quality counts "
    "apply to the returned table.",
    "Calendar completeness and historical data availability are unverified.",
)


def _download_history(symbol: str, start: date, end_exclusive: date) -> pd.DataFrame:
    try:
        return yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=end_exclusive.isoformat(),
            **HISTORY_OPTIONS,
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"yfinance could not parse Yahoo's response: {exc}") from exc


def _optional_values(frame: pd.DataFrame, column: str) -> list[int | float]:
    if column not in frame:
        return []
    values = frame[column].dropna().to_numpy(dtype=object).tolist()
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not isfinite(value))
        ):
            raise ValueError(f"invalid numeric value in {column!r}: {value!r}")
        if (column == "Adj Close" and value <= 0) or (
            column == "Stock Splits" and value < 0
        ):
            raise ValueError(f"invalid value in {column!r}: {value!r}")
    return values


def _normalize_history(
    frame: pd.DataFrame, metadata: DatasetMetadata, start: date, end: date
) -> tuple[list[DailyBar], dict[str, object]]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("expected a yfinance DataFrame")
    if len(frame.index) == 0:
        raise ValueError("Yahoo returned no daily observations")
    if not frame.columns.is_unique or any(
        not isinstance(column, str) or not column for column in frame.columns
    ):
        raise ValueError("expected unique, nonempty Yahoo column names")
    if TIMESTAMP_COLUMN in frame:
        raise ValueError(f"unexpected Yahoo column {TIMESTAMP_COLUMN!r}")

    absent = [column for column in OHLCV_COLUMNS if column not in frame]
    if absent:
        raise ValueError(f"missing required Yahoo columns: {', '.join(absent)}")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans:
        raise ValueError("expected timezone-aware Yahoo dates without missing values")

    # Daily labels use the exchange calendar date, including across UTC midnight.
    sessions = index.date.tolist()
    first, last = min(sessions), max(sessions)
    if first < start or last > end:
        raise ValueError(f"observed dates {first} to {last} exceed {start} to {end}")

    ohlcv = frame.loc[:, list(OHLCV_COLUMNS)]
    bars = [
        DailyBar(metadata.symbol, session, *values)
        for session, values in zip(sessions, ohlcv.to_numpy(dtype=object), strict=True)
    ]
    actions = {}
    for column, event in ACTION_COLUMNS.items():
        if column in frame:
            values = _optional_values(frame, column)
            actions[event] = {
                "event_rows": sum(value != 0 for value in values),
                "missing_values": len(bars) - len(values),
            }
        else:
            actions[event] = None
    quality: dict[str, object] = {
        "observations": len(bars),
        "first_session": first.isoformat(),
        "last_session": last.isoformat(),
        "duplicate_dates": len(sessions) - len(set(sessions)),
        "missing_ohlcv_values": int(ohlcv.isna().to_numpy().sum()),
        "invalid_ohlcv_rows": 0,
        "zero_volume_rows": sum(
            type(bar.volume) is int and bar.volume == 0 for bar in bars
        ),
        "adjusted_close_values": len(_optional_values(frame, "Adj Close")),
        "corporate_actions": actions,
    }

    try:
        validate_daily_bars(bars, metadata)
    except ValueError as exc:
        # A singleton checks the OHLCV independently of surrounding dates.
        invalid = 0
        for bar in bars:
            try:
                validate_daily_bars((bar,), metadata)
            except ValueError:
                invalid += 1
        quality["invalid_ohlcv_rows"] = invalid
        raise ValueError(f"{exc}. Quality: {json.dumps(quality)}") from exc

    return bars, quality


def _ingest_symbol(
    symbol: str, start: date, end: date, data_dir: Path
) -> tuple[Path, dict[str, object]]:
    end_exclusive = end + timedelta(days=1)
    frame = _download_history(symbol, start, end_exclusive)
    metadata = DatasetMetadata(
        source="Yahoo Finance",
        symbol=symbol,
        adjustment=PriceAdjustment.SPLIT_ADJUSTED,
        retrieved_at=datetime.now(UTC),
    )
    bars, quality = _normalize_history(frame, metadata, start, end)
    parent = data_dir / "yahoo" / symbol
    parent.mkdir(parents=True, exist_ok=True)
    snapshot = parent / metadata.retrieved_at.strftime("%Y%m%dT%H%M%S.%fZ")

    with TemporaryDirectory(prefix=".snapshot-", dir=parent) as temporary:
        staging = Path(temporary)
        source = staging / "provider.csv"
        frame.to_csv(
            source, index_label=TIMESTAMP_COLUMN, lineterminator="\n", encoding="utf-8"
        )
        normalized = staging / "daily_bars.csv"
        with normalized.open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(field.name for field in fields(DailyBar))
            writer.writerows(astuple(bar) for bar in bars)
        manifest = {
            "provider": metadata.source,
            "symbol": metadata.symbol,
            "request": {
                "start_inclusive": start.isoformat(),
                "end_inclusive": end.isoformat(),
                "api_end_exclusive": end_exclusive.isoformat(),
                **HISTORY_OPTIONS,
            },
            "retrieved_at_utc": metadata.retrieved_at.isoformat().replace("+00:00", "Z"),
            "yfinance_version": yf.__version__,
            "price_adjustment": {
                "convention": metadata.adjustment.value,
                "basis": "provider-reported",
                "independently_verified": False,
            },
            "source_file": source.name,
            "source_sha256": sha256(source.read_bytes()).hexdigest(),
            "quality": quality,
            "limitations": LIMITATIONS,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        # Publish the three files together and keep earlier retrievals.
        staging.rename(snapshot)

    return snapshot, quality


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="+", choices=SYMBOLS)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    args = parser.parse_args(argv)
    if args.start > args.end or args.end == date.max:
        parser.error("use an inclusive end date from start through 9999-12-30")
    if len(args.symbols) != len(set(args.symbols)):
        parser.error("specify each symbol once")

    data_dir = Path.cwd() / "data"
    yf.set_tz_cache_location(str(data_dir / ".yfinance-cache"))
    hide_exceptions = yf.config.debug.hide_exceptions
    yf.config.debug.hide_exceptions = False
    failed = False
    try:
        for symbol in args.symbols:
            try:
                snapshot, quality = _ingest_symbol(
                    symbol, args.start, args.end, data_dir
                )
            except YFRateLimitError:
                print(
                    f"{symbol}: Yahoo Finance rate limit reached. "
                    "Remaining requests were skipped.",
                    file=sys.stderr,
                )
                return 1
            except (YFException, OSError, ValueError) as exc:
                print(
                    f"{symbol} ({args.start} through {args.end}): {exc}",
                    file=sys.stderr,
                )
                failed = True
            else:
                print(f"{symbol}: {snapshot}\n{json.dumps(quality)}")
    finally:
        yf.config.debug.hide_exceptions = hide_exceptions

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
