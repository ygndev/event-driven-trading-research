"""Data contract for historical daily US equity and ETF bars."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from math import isfinite


class PriceAdjustment(Enum):
    """Price convention claimed by the data source."""

    RAW = "raw"
    SPLIT_ADJUSTED = "split_adjusted"
    TOTAL_RETURN_ADJUSTED = "total_return_adjusted"


@dataclass(frozen=True, slots=True)
class DailyBar:
    """A daily OHLCV record labeled by exchange session date."""

    symbol: str
    session_date: date
    open: int | float
    high: int | float
    low: int | float
    close: int | float
    volume: int


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Source and retrieval information for a single-instrument dataset."""

    source: str
    symbol: str
    adjustment: PriceAdjustment
    retrieved_at: datetime
    source_version: str | None = None

    def __post_init__(self) -> None:
        _validate_text(self.source, "source")
        _validate_text(self.symbol, "symbol")
        if self.source_version is not None:
            _validate_text(self.source_version, "source_version")
        if not isinstance(self.adjustment, PriceAdjustment):
            raise ValueError("adjustment must be a PriceAdjustment")
        retrieved_at = self.retrieved_at
        if (
            not isinstance(retrieved_at, datetime)
            or retrieved_at.utcoffset() is None
        ):
            raise ValueError("retrieved_at must be a timezone-aware datetime")


def _validate_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field} must be a nonempty string without surrounding whitespace"
        )


def _validate_ohlcv(bar: DailyBar, row: str) -> None:
    for field in ("open", "high", "low", "close"):
        price = getattr(bar, field)
        if (
            isinstance(price, bool)
            or not isinstance(price, (int, float))
            or (isinstance(price, float) and not isfinite(price))
            or price <= 0
        ):
            raise ValueError(
                f"{row}: {field} must be a finite, positive int or float"
            )

    volume = bar.volume
    if isinstance(volume, bool) or not isinstance(volume, int) or volume < 0:
        raise ValueError(f"{row}: volume must be a nonnegative integer")

    if bar.low > bar.high:
        raise ValueError(f"{row}: low exceeds high")
    for field in ("open", "close"):
        price = getattr(bar, field)
        if bar.high < price:
            raise ValueError(f"{row}: high is below {field}")
        if bar.low > price:
            raise ValueError(f"{row}: low is above {field}")


def validate_daily_bars(
    bars: Sequence[DailyBar], metadata: DatasetMetadata
) -> None:
    """Check one instrument's daily bars, raising ValueError on failure."""
    if not isinstance(metadata, DatasetMetadata):
        raise ValueError("metadata must be DatasetMetadata")
    if not isinstance(bars, Sequence):
        raise ValueError("bars must be a sequence of DailyBar records")
    if not bars:
        raise ValueError("daily bar dataset is empty")

    symbol = metadata.symbol
    seen_dates: set[date] = set()
    previous_date: date | None = None
    for index, bar in enumerate(bars, start=1):
        row = f"bar {index}"
        if not isinstance(bar, DailyBar):
            raise ValueError(f"{row}: expected a DailyBar")
        if not isinstance(bar.symbol, str) or bar.symbol != symbol:
            raise ValueError(
                f"{row}: expected symbol {symbol!r}, got {bar.symbol!r}"
            )

        session_date = bar.session_date
        if (
            not isinstance(session_date, date)
            or isinstance(session_date, datetime)
        ):
            raise ValueError(f"{row}: session_date must be a date")
        row = f"{row} ({session_date.isoformat()})"
        if session_date in seen_dates:
            raise ValueError(f"{row}: duplicate session date for {symbol!r}")
        if previous_date is not None and session_date < previous_date:
            raise ValueError(
                f"{row}: session date is out of order after {previous_date}"
            )

        _validate_ohlcv(bar, row)
        seen_dates.add(session_date)
        previous_date = session_date
