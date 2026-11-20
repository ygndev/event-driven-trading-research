from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime

import pytest

from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.replay import MarketEvent, replay_daily_bars

FIRST = date(2025, 1, 3)
NEXT = date(2025, 1, 6)
LATER = date(2025, 1, 8)


def dataset(symbol, *sessions, adjustment=PriceAdjustment.RAW):
    metadata = DatasetMetadata(
        source="Synthetic",
        symbol=symbol,
        adjustment=adjustment,
        retrieved_at=datetime(2025, 1, 10, tzinfo=UTC),
    )
    bar = DailyBar(symbol, FIRST, 100, 104, 99, 101, 1200)
    bars = [replace(bar, session_date=session) for session in sessions]
    return metadata, bars


def collect(*datasets):
    events: list[MarketEvent] = []
    replay_daily_bars(datasets, events.append)
    return events


def test_replays_one_instrument_across_missing_sessions():
    metadata, bars = dataset("SPY", FIRST, LATER)
    original = bars.copy()

    events = collect((metadata, bars))

    assert [event.bar for event in events] == original
    assert bars == original
    assert [event.adjustment for event in events] == [PriceAdjustment.RAW] * 2
    with pytest.raises(FrozenInstanceError):
        setattr(events[0], "bar", bars[1])


def test_merges_instruments_by_session_then_symbol():
    spy = dataset("SPY", FIRST, LATER)
    qqq = dataset(
        "QQQ", FIRST, NEXT, adjustment=PriceAdjustment.SPLIT_ADJUSTED
    )
    iwm = dataset("IWM", FIRST, LATER)

    events = collect(spy, qqq, iwm)

    keys = [(event.bar.session_date, event.bar.symbol) for event in events]
    assert keys == [
        (FIRST, "IWM"),
        (FIRST, "QQQ"),
        (FIRST, "SPY"),
        (NEXT, "QQQ"),
        (LATER, "IWM"),
        (LATER, "SPY"),
    ]
    assert events[1].adjustment is PriceAdjustment.SPLIT_ADJUSTED
    assert events[2].adjustment is PriceAdjustment.RAW
    assert events == collect(spy, qqq, iwm) == collect(iwm, spy, qqq)


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        (lambda bars: [bars[0], bars[0]], "duplicate session date"),
        (lambda bars: bars[::-1], "out of order"),
        (
            lambda bars: [bars[0], replace(bars[1], close=105)],
            "high is below close",
        ),
        (
            lambda bars: [bars[0], replace(bars[1], symbol="IWM")],
            "expected symbol",
        ),
        (lambda bars: [], "dataset is empty"),
    ],
)
def test_rejects_invalid_later_stream_before_any_delivery(malformed, message):
    first = dataset("SPY", FIRST)
    metadata, bars = dataset("QQQ", NEXT, LATER)
    received = []

    invalid = (metadata, malformed(bars))
    with pytest.raises(ValueError, match=message):
        replay_daily_bars([first, invalid], received.append)

    assert received == []


def test_rejects_multiple_datasets_for_one_symbol_even_without_overlap():
    received = []
    with pytest.raises(ValueError, match="multiple datasets for symbol 'SPY'"):
        replay_daily_bars(
            [dataset("SPY", FIRST), dataset("SPY", LATER)], received.append
        )
    assert received == []


def test_rejects_empty_replay():
    with pytest.raises(ValueError, match="at least one dataset"):
        replay_daily_bars([], pytest.fail)


def test_stopping_inside_first_session_does_not_deliver_a_future_bar():
    class StopAtClose(Exception):
        pass

    received = []

    def stop_after_second_event(event):
        received.append(event.bar)
        if len(received) == 2:
            raise StopAtClose

    with pytest.raises(StopAtClose):
        replay_daily_bars(
            [dataset("SPY", FIRST, LATER), dataset("QQQ", FIRST, NEXT)],
            stop_after_second_event,
        )

    assert [(bar.session_date, bar.symbol) for bar in received] == [
        (FIRST, "QQQ"),
        (FIRST, "SPY"),
    ]


def test_changing_a_future_price_does_not_change_the_earlier_event():
    metadata, bars = dataset("SPY", FIRST, LATER)
    changed = [bars[0], replace(bars[1], close=102)]
    original_events = collect((metadata, bars))
    changed_events = collect((metadata, changed))

    assert original_events[0] == changed_events[0]
    assert original_events[1].bar.close == 101
    assert changed_events[1].bar.close == 102


def test_consumer_cannot_change_the_validated_sequence_by_mutating_input():
    metadata, bars = dataset("SPY", FIRST, LATER)
    original = bars.copy()
    received = []

    def consume(event):
        received.append(event.bar)
        if event.bar.session_date == FIRST:
            bars[1] = replace(bars[1], volume=-1)

    replay_daily_bars([(metadata, bars)], consume)

    assert received == original
    assert bars[1].volume == -1


def test_session_end_callback_follows_all_market_events_for_the_session():
    trace = []
    replay_daily_bars(
        [dataset("SPY", FIRST, LATER), dataset("QQQ", FIRST, NEXT)],
        lambda event: trace.append((event.bar.session_date, event.bar.symbol)),
        on_session_end=lambda session: trace.append((session, "end")),
    )
    assert trace == [
        (FIRST, "QQQ"),
        (FIRST, "SPY"),
        (FIRST, "end"),
        (NEXT, "QQQ"),
        (NEXT, "end"),
        (LATER, "SPY"),
        (LATER, "end"),
    ]
