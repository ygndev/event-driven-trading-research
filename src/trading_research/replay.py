"""Replay validated daily bars at their conceptual session close."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from heapq import merge

from .daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
    validate_daily_bars,
)


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A completed-session observation and its declared price convention."""

    bar: DailyBar
    adjustment: PriceAdjustment


def replay_daily_bars(
    datasets: Iterable[tuple[DatasetMetadata, Sequence[DailyBar]]],
    on_event: Callable[[MarketEvent], None],
) -> None:
    """Validate all datasets, then deliver events by session date and symbol.

    Each callback runs at the conceptual end of the bar's session. It receives
    only that event, and replay advances after the callback returns.
    """
    streams: list[tuple[DailyBar, ...]] = []
    adjustments: dict[str, PriceAdjustment] = {}
    for metadata, bars in datasets:
        if not isinstance(bars, Sequence):
            raise ValueError("bars must be a sequence of DailyBar records")
        # Keep validation valid even if a callback edits the input sequences.
        stream = tuple(bars)
        validate_daily_bars(stream, metadata)
        symbol = metadata.symbol
        if symbol in adjustments:
            raise ValueError(f"multiple datasets for symbol {symbol!r}")
        streams.append(stream)
        adjustments[symbol] = metadata.adjustment

    if not streams:
        raise ValueError("replay requires at least one dataset")

    for bar in merge(*streams, key=lambda bar: (bar.session_date, bar.symbol)):
        on_event(MarketEvent(bar, adjustments[bar.symbol]))
