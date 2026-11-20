from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime
from operator import setitem

import pytest

from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.signals import SignalEvent, generate_signals

FIRST = date(2025, 1, 3)
NEXT = date(2025, 1, 6)
LAST = date(2025, 1, 8)


def dataset(symbol, *sessions):
    metadata = DatasetMetadata(
        source="Synthetic",
        symbol=symbol,
        adjustment=PriceAdjustment.RAW,
        retrieved_at=datetime(2025, 1, 10, tzinfo=UTC),
    )
    return metadata, [
        DailyBar(symbol, session, 100, 104, 99, 101 + index, 1000)
        for index, session in enumerate(sessions)
    ]


def test_single_instrument_evaluates_once_per_observed_session():
    evaluated = []

    def strategy(context):
        evaluated.append(context.session_date)
        event, = context.events
        assert event.bar.session_date == context.session_date
        target = 1 if context.session_date == FIRST else 0
        return (SignalEvent(event.bar.symbol, context.session_date, target),)

    signals = generate_signals([dataset("SPY", FIRST, LAST)], strategy)

    assert evaluated == [FIRST, LAST]
    assert signals == (
        SignalEvent("SPY", FIRST, 1),
        SignalEvent("SPY", LAST, 0),
    )


def test_multi_asset_strategy_requires_both_observations_on_one_session():
    spy = dataset("SPY", FIRST, NEXT, LAST)
    qqq = dataset("QQQ", FIRST, LAST)

    def run(datasets):
        evaluations = []

        def strategy(context):
            symbols = tuple(event.bar.symbol for event in context.events)
            evaluations.append((context.session_date, symbols))
            if {"SPY", "QQQ"} != set(symbols):
                return ()
            # Emit in reverse symbol order to exercise the signal tie-break.
            return (
                SignalEvent("SPY", context.session_date, 0.5),
                SignalEvent("QQQ", context.session_date, 0.25),
            )

        return generate_signals(datasets, strategy), evaluations

    signals, evaluations = run([spy, qqq])
    assert evaluations == [
        (FIRST, ("QQQ", "SPY")),
        (NEXT, ("SPY",)),
        (LAST, ("QQQ", "SPY")),
    ]
    assert signals == (
        SignalEvent("QQQ", FIRST, 0.25),
        SignalEvent("SPY", FIRST, 0.5),
        SignalEvent("QQQ", LAST, 0.25),
        SignalEvent("SPY", LAST, 0.5),
    )
    assert run([spy, qqq]) == run([qqq, spy]) == (signals, evaluations)


def test_history_contains_only_observed_bars_and_remains_bounded():
    spy = dataset("SPY", FIRST, NEXT, LAST)
    qqq = dataset("QQQ", FIRST, LAST)
    iwm = dataset("IWM", LAST)
    contexts = []

    def strategy(context):
        contexts.append(context)
        assert all(
            event.bar.session_date <= context.session_date
            for symbol in ("SPY", "QQQ", "IWM")
            for event in context.history(symbol)
        )
        return iter(())

    assert generate_signals([spy, qqq, iwm], strategy) == ()
    first, middle, last = contexts
    assert [event.bar for event in first.history("SPY")] == spy[1][:1]
    assert first.history("IWM") == middle.history("IWM") == ()
    assert [event.bar for event in middle.history("QQQ")] == qqq[1][:1]
    assert [event.bar for event in last.history("QQQ")] == qqq[1]
    assert [event.bar for event in last.history("IWM")] == iwm[1]


def test_can_emit_for_a_previously_observed_symbol_without_a_current_bar():
    def strategy(context):
        if context.session_date == NEXT:
            assert [event.bar.symbol for event in context.events] == ["SPY"]
            return (SignalEvent("QQQ", NEXT, 0),)
        return ()

    signals = generate_signals(
        [dataset("SPY", FIRST, NEXT), dataset("QQQ", FIRST)], strategy
    )
    assert signals == (SignalEvent("QQQ", NEXT, 0),)


@pytest.mark.parametrize(
    "target",
    [True, False, float("nan"), float("inf"), float("-inf"), -0.01, 1.01, "0.5"],
)
def test_rejects_invalid_target_exposure(target):
    def strategy(context):
        return [SignalEvent("SPY", context.session_date, target)]

    with pytest.raises(ValueError, match="target_exposure must be a finite"):
        generate_signals([dataset("SPY", FIRST)], strategy)


def test_rejects_duplicate_signals_for_an_instrument_in_same_session():
    def strategy(context):
        return (
            SignalEvent("SPY", context.session_date, 0),
            SignalEvent("SPY", context.session_date, 1),
        )

    with pytest.raises(ValueError, match="duplicate signal for 'SPY'"):
        generate_signals([dataset("SPY", FIRST)], strategy)


@pytest.mark.parametrize("symbol", ["QQQ", "IWM"])
def test_cannot_signal_for_unobserved_instruments(symbol):
    def strategy(context):
        assert context.history(symbol) == ()
        return (SignalEvent(symbol, context.session_date, 1),)

    with pytest.raises(ValueError, match=rf"unknown instrument '{symbol}'"):
        generate_signals(
            [dataset("SPY", FIRST), dataset("QQQ", LAST)], strategy
        )


@pytest.mark.parametrize("decision_session", [FIRST, LAST])
def test_rejects_past_and_future_decision_dates(decision_session):
    with pytest.raises(
        ValueError, match=f"must match completed session {NEXT}"
    ):
        generate_signals(
            [dataset("SPY", NEXT)],
            lambda context: (SignalEvent("SPY", decision_session, 1),),
        )


@pytest.mark.parametrize(
    ("emitted", "message"),
    [
        (None, "strategy must return iterable signals"),
        ([None], "expected a SignalEvent"),
    ],
)
def test_rejects_malformed_strategy_output(emitted, message):
    with pytest.raises(ValueError, match=message):
        generate_signals([dataset("SPY", FIRST)], lambda context: emitted)


def test_context_and_signal_data_are_immutable():
    def strategy(context):
        observed = context.events[0]
        history = context.history("SPY")
        with pytest.raises(FrozenInstanceError):
            setattr(context, "session_date", LAST)
        with pytest.raises(TypeError):
            setitem(context.events, 0, None)
        with pytest.raises(TypeError):
            setitem(history, 0, None)
        with pytest.raises(FrozenInstanceError):
            setattr(observed, "bar", None)
        with pytest.raises(FrozenInstanceError):
            setattr(history[0].bar, "close", 0)
        return (SignalEvent("SPY", FIRST, 1),)

    signal, = generate_signals([dataset("SPY", FIRST)], strategy)
    with pytest.raises(FrozenInstanceError):
        setattr(signal, "target_exposure", 0)
    assert signal.target_exposure == 1


def test_invalid_later_market_data_is_rejected_before_strategy_evaluation():
    metadata, bars = dataset("SPY", FIRST, LAST)
    bars[1] = replace(bars[1], volume=-1)

    with pytest.raises(
        ValueError, match="volume must be a nonnegative integer"
    ):
        generate_signals([(metadata, bars)], pytest.fail)
