from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research.backtest import run_backtest
from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.momentum import time_series_momentum
from trading_research.orders import OrderSide
from trading_research.performance import evaluate_performance
from trading_research.replay import MarketEvent
from trading_research.signals import (
    SessionContext,
    SignalEvent,
    generate_signals,
)


def session(observation):
    return date(2025, 1, 1) + timedelta(days=observation - 1)


def dataset(symbol, observations, *, adjustment=PriceAdjustment.RAW):
    metadata = DatasetMetadata(
        "Synthetic", symbol, adjustment, datetime(2025, 6, 1, tzinfo=UTC)
    )
    bars = tuple(
        DailyBar(
            symbol, day, opening, max(opening, close), min(opening, close),
            close, 1_000,
        )
        for day, opening, close in observations
    )
    return metadata, bars


def spy_closes(closes):
    return dataset(
        "SPY",
        (
            (session(index), price, price)
            for index, price in enumerate(closes, start=1)
        ),
    )


def completed_context(bars, adjustment=PriceAdjustment.RAW):
    context = None
    for bar in bars:
        context = SessionContext(
            bar.session_date, (MarketEvent(bar, adjustment),), context
        )
    return context


def test_warmup_has_no_signals_through_observation_63():
    spy = spy_closes([90] * 62 + [100])

    assert generate_signals([spy], time_series_momentum) == ()


@pytest.mark.parametrize(
    ("current", "target"), [(110, 1), (90, 0), (100, 0)]
)
def test_first_signal_at_observation_64(current, target):
    spy = spy_closes([100] * 63 + [current])

    assert generate_signals([spy], time_series_momentum) == (
        SignalEvent("SPY", date(2025, 3, 5), target),
    )


def test_uses_current_close_and_close_exactly_63_intervals_before():
    closes = [100, 120] + [115] * 61 + [110]
    spy = dataset(
        "SPY",
        (
            (session(index), 90 if index == 64 else close, close)
            for index, close in enumerate(closes, start=1)
        ),
    )

    # 110 is above observation 1 but below observation 2. The current open is 90.
    assert generate_signals([spy], time_series_momentum) == (
        SignalEvent("SPY", date(2025, 3, 5), 1),
    )


def test_rebalance_cadence_and_rolling_lookback():
    closes = [100] * 111
    closes[21] = 120
    closes[42] = 110
    closes[63] = closes[84] = closes[105] = 110

    signals = generate_signals([spy_closes(closes)], time_series_momentum)

    assert signals == (
        SignalEvent("SPY", date(2025, 3, 5), 1),
        SignalEvent("SPY", date(2025, 3, 26), 0),
        SignalEvent("SPY", date(2025, 4, 16), 0),
    )


def test_missing_spy_bars_do_not_advance_or_repeat_decisions():
    spy_days = [day for day in range(1, 88) if day not in {64, 75}]
    closes = [100] * 85
    closes[63] = closes[84] = 110
    spy = dataset(
        "SPY",
        (
            (session(day), close, close)
            for day, close in zip(spy_days, closes, strict=True)
        ),
    )
    qqq = dataset("QQQ", [(session(day), 50, 50) for day in (64, 75, 88)])

    signals = generate_signals([qqq, spy], time_series_momentum)

    assert signals == generate_signals([spy], time_series_momentum) == (
        SignalEvent("SPY", session(65), 1),
        SignalEvent("SPY", session(87), 1),
    )


def test_future_price_changes_do_not_change_current_signals_or_history():
    contexts = []

    def strategy(context):
        if context.session_date == session(64):
            contexts.append(context)
        return time_series_momentum(context)

    metadata, bars = spy_closes([90] * 63 + [110, 112])
    original = tuple(replace(bar) for bar in bars)
    future = replace(bars[-1], open=900, high=900, low=800, close=800)
    first = generate_signals([(metadata, bars)], strategy)
    second = generate_signals([(metadata, (*bars[:-1], future))], strategy)

    assert first == second == (SignalEvent("SPY", session(64), 1),)
    for context in contexts:
        history = context.history("SPY")
        assert len(history) == 64
        assert history[-1].bar.session_date == session(64)
        assert time_series_momentum(context) == first
        assert context.history("SPY") == history
    assert bars == original


def test_small_positive_change_is_not_lost_to_float_conversion():
    spy = spy_closes([10**18] * 63 + [10**18 + 1])

    assert generate_signals([spy], time_series_momentum) == (
        SignalEvent("SPY", session(64), 1),
    )


@pytest.mark.parametrize("index", [0, 63])
@pytest.mark.parametrize("close", [0, -1, True, float("inf"), Decimal("NaN")])
def test_rejects_invalid_endpoint_close_prices(index, close):
    _, bars = spy_closes([100] * 64)
    observations = list(bars)
    observations[index] = replace(bars[index], close=close)
    context = completed_context(observations)

    with pytest.raises(ValueError, match="SPY close"):
        time_series_momentum(context)


@pytest.mark.parametrize(
    "adjustment",
    [PriceAdjustment.SPLIT_ADJUSTED, PriceAdjustment.TOTAL_RETURN_ADJUSTED],
)
def test_scheduled_signal_requires_raw_closes(adjustment):
    metadata, bars = spy_closes([100] * 64)

    with pytest.raises(ValueError, match="SPY momentum requires RAW closes"):
        generate_signals(
            [(replace(metadata, adjustment=adjustment), bars)],
            time_series_momentum,
        )


def test_full_synthetic_momentum_buy_then_sell():
    closes = [90] * 63 + [110] + [112] * 20 + [100, 106]
    closes[21] = 120
    opens = {65: 108, 86: 105}
    spy = dataset(
        "SPY",
        (
            (session(index), opens.get(index, close), close)
            for index, close in enumerate(closes, start=1)
        ),
    )
    result = run_backtest(
        [spy], time_series_momentum, initial_cash=1000,
        adjustment_bps=25, commission=Decimal("1.25"),
    )
    report = evaluate_performance(result)
    buy_decision = date(2025, 3, 5)
    buy_session = date(2025, 3, 6)
    sell_decision = date(2025, 3, 26)
    sell_session = date(2025, 3, 27)

    assert result.signals == (
        SignalEvent("SPY", buy_decision, 1),
        SignalEvent("SPY", sell_decision, 0),
    )
    assert [
        (order.decision_session, order.side, order.quantity)
        for order in result.orders
    ] == [
        (buy_decision, OrderSide.BUY, 9),
        (sell_decision, OrderSide.SELL, 9),
    ]
    assert [(fill.execution_session, fill.price) for fill in result.fills] == [
        (buy_session, Decimal("108.27")),
        (sell_session, Decimal("104.7375")),
    ]
    assert [
        (entry.cash_after, entry.commission) for entry in result.transactions
    ] == [
        (Decimal("24.32"), Decimal("1.25")),
        (Decimal("965.7075"), Decimal("1.25")),
    ]
    pre_buy, bought = result.sessions[63:65]
    pre_sell, sold = result.sessions[84:86]
    assert pre_buy.session_date == buy_decision
    assert pre_buy.portfolio.cash == Decimal("1000")
    assert pre_buy.portfolio.positions == {}
    assert bought.session_date == buy_session
    assert bought.portfolio.cash == Decimal("24.32")
    assert bought.portfolio.positions == {"SPY": 9}
    assert bought.equity == Decimal("1032.32")
    assert pre_sell.session_date == sell_decision
    assert pre_sell.portfolio.positions == {"SPY": 9}
    assert pre_sell.equity == Decimal("924.32")
    assert sold.session_date == sell_session
    assert sold.portfolio.positions == {}
    assert sold.portfolio.cash == sold.equity == Decimal("965.7075")
    assert sold.portfolio.cumulative_commissions == Decimal("2.50")
    assert result.rejected_orders == result.outstanding_orders == ()
    assert report.total_return == Decimal("-0.0342925")
    assert report.executed_orders == 2
    assert report.executed_notional == Decimal("1917.0675")
    assert report.total_commissions == Decimal("2.50")
