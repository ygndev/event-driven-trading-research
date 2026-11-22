from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from trading_research.backtest import run_backtest
from trading_research.corporate_actions import CashDividend, StockSplit
from trading_research.daily_bars import DailyBar, DatasetMetadata, PriceAdjustment
from trading_research.mean_reversion import short_term_mean_reversion
from trading_research.momentum import time_series_momentum
from trading_research.orders import OrderSide
from trading_research.signals import SignalEvent


def day(observation):
    weeks, weekday = divmod(observation - 1, 5)
    return date(2025, 2, 3) + timedelta(days=7 * weeks + weekday)


def dataset(closes, *, symbol="SPY", observations=None, openings=None):
    metadata = DatasetMetadata(
        "Synthetic", symbol, PriceAdjustment.RAW, datetime(2026, 1, 1, tzinfo=UTC)
    )
    if observations is None:
        observations = range(1, len(closes) + 1)
    openings = openings or {}
    bars = []
    for index, close in zip(observations, closes, strict=True):
        opening = openings.get(index, close)
        bars.append(DailyBar(
            symbol, day(index), opening, max(opening, close),
            min(opening, close), close, 1000,
        ))
    return metadata, tuple(bars)


def run(datasets, strategy, splits=(), *, capital=1000):
    return run_backtest(
        datasets, strategy, initial_cash=capital, adjustment_bps=0, commission=0,
        corporate_actions=splits,
    )


def test_as_of_closes_are_exact_across_successive_splits_and_gaps():
    spy = dataset([200, 100, 34, 68], observations=(1, 3, 4, 5))
    qqq = dataset([300, 301, 302, 303, 304], symbol="QQQ")
    splits = (
        StockSplit("SPY", day(5), Fraction(1, 2)),
        StockSplit("SPY", day(3), 2),
        StockSplit("SPY", day(4), 3),
    )
    contexts = []

    def strategy(context):
        contexts.append(context)
        return ()

    result = run((spy, qqq), strategy, splits)
    raw = contexts[-1].history("SPY")
    first, doubled, tripled, reversed_split = raw

    assert tuple(event.bar for event in raw) == spy[1]
    assert contexts[1].history("SPY") == (first,)
    assert contexts[0].strategy_close(first) == Decimal(200)
    assert contexts[1].strategy_close(first) == Decimal(200)
    assert contexts[2].strategy_close(first) == Decimal(100)
    assert contexts[2].strategy_close(doubled) == Decimal(100)
    with localcontext() as context:
        context.prec = 2
        assert contexts[3].strategy_close(first) == Fraction(100, 3)
        assert contexts[3].strategy_close(doubled) == Fraction(100, 3)
    assert contexts[3].strategy_close(tripled) == Decimal(34)
    assert contexts[-1].strategy_close(tripled) == Decimal(68)
    assert contexts[-1].strategy_close(reversed_split) == Decimal(68)
    assert contexts[-1].strategy_close(first) == Fraction(200, 3)
    assert contexts[-1].strategy_close(first, as_of=day(3)) == Decimal(100)
    assert contexts[-1].strategy_close(contexts[0].history("QQQ")[0]) == Decimal(300)
    assert [row.split.effective_session for row in result.split_adjustments] == [
        day(3), day(4), day(5)
    ]

    with pytest.raises(FrozenInstanceError):
        setattr(contexts[0], "_effective_splits", splits)
    with pytest.raises(ValueError, match="decision date"):
        contexts[0].strategy_close(first, as_of=day(3))
    with pytest.raises(ValueError, match="decision date"):
        contexts[0].strategy_close(doubled)
    with pytest.raises(ValueError, match="requires RAW"):
        contexts[0].strategy_close(
            replace(first, adjustment=PriceAdjustment.SPLIT_ADJUSTED)
        )


def test_mean_reversion_ignores_a_split_without_an_economic_decline():
    spy = dataset([200] * 5 + [100, 100])

    raw = run((spy,), short_term_mean_reversion)
    adjusted = run((spy,), short_term_mean_reversion, (StockSplit("SPY", day(6), 2),))

    assert raw.signals == (SignalEvent("SPY", day(6), 1),)
    assert raw.fills[0].execution_session == day(7)
    assert adjusted.signals == adjusted.fills == ()
    assert adjusted.sessions[-1].equity == 1000


@pytest.mark.parametrize(("close", "enters"), [(32.33, True), (32.34, False)])
def test_mean_reversion_compares_the_exact_nonterminating_split_basis(close, enters):
    spy = dataset([100] * 5 + [close], openings={6: 33.5})

    result = run((spy,), short_term_mean_reversion, (StockSplit("SPY", day(6), 3),))

    expected = (SignalEvent("SPY", day(6), 1),) if enters else ()
    assert result.signals == expected
    assert result.fills == ()


def test_dividend_ex_date_price_move_is_still_visible_to_mean_reversion():
    spy = dataset([100] * 5 + [97, 96])
    dividend = CashDividend("distribution", "SPY", day(6), day(7), 3)

    result = run((spy,), short_term_mean_reversion, (dividend,))

    assert result.signals == (SignalEvent("SPY", day(6), 1),)
    assert result.fills[0].execution_session == day(7)
    assert result.dividend_payments == ()


def test_a_split_during_a_mean_reversion_holding_preserves_the_exit_clock():
    spy = dataset(
        [200] * 5 + [194, 194, 192, 96, 95, 95, 97],
        openings={7: 190, 12: 96},
    )

    result = run(
        (spy,), short_term_mean_reversion, (StockSplit("SPY", day(9), 2),),
        capital=2000,
    )

    assert result.signals == (
        SignalEvent("SPY", day(6), 1), SignalEvent("SPY", day(11), 0)
    )
    assert [
        (fill.execution_session, fill.side, fill.quantity, fill.price)
        for fill in result.fills
    ] == [
        (day(7), OrderSide.BUY, 10, 190), (day(12), OrderSide.SELL, 20, 96)
    ]
    assert result.sessions[8].portfolio.positions == {"SPY": 20}
    assert result.sessions[-1].portfolio.positions == {}
    assert result.sessions[-1].portfolio.cash == result.sessions[-1].equity == 2020


def test_rejected_entry_rearm_still_compares_previous_closes_on_the_same_basis():
    spy = dataset([100] * 5 + [97, 94, 94, 47, 47, 47], openings={7: 110})

    result = run((spy,), short_term_mean_reversion, (StockSplit("SPY", day(9), 2),))

    assert result.signals == (SignalEvent("SPY", day(6), 1),)
    assert result.fills == result.outstanding_orders == ()
    rejected, = result.rejected_orders
    assert "insufficient cash" in rejected.rejection_reason


def test_momentum_preserves_cadence_and_uses_raw_fills_and_marks():
    closes = [90] + [100] * 62 + [110] * 20 + [55] * 22 + [45, 43]
    spy = dataset(closes, openings={65: 108, 107: 44})
    split = StockSplit("SPY", day(84), 2)
    without_actions = run((spy,), time_series_momentum)
    result = run((spy,), time_series_momentum, (split,))

    assert without_actions.signals[1] == SignalEvent("SPY", day(85), 0)
    assert result.signals == (
        SignalEvent("SPY", day(64), 1),
        SignalEvent("SPY", day(85), 1),
        SignalEvent("SPY", day(106), 0),
    )
    assert [
        (fill.execution_session, fill.side, fill.quantity, fill.price)
        for fill in result.fills
    ] == [
        (day(65), OrderSide.BUY, 9, 108), (day(107), OrderSide.SELL, 18, 44)
    ]
    adjustment, = result.split_adjustments
    assert (adjustment.shares_before, adjustment.shares_after) == (9, 18)
    before, effective = result.sessions[82:84]
    assert before.equity == effective.equity == 1018
    assert effective.portfolio.positions == {"SPY": 18}
    assert effective.portfolio.cash == 28
    assert result.sessions[-1].portfolio.positions == {}
    assert result.sessions[-1].portfolio.cash == result.sessions[-1].equity == 820
    assert result == run((spy,), time_series_momentum, (split,))


def test_future_splits_and_prices_cannot_change_an_earlier_decision_or_context():
    contexts = []

    def strategy(context):
        if context.session_date == day(64):
            contexts.append(context)
        return time_series_momentum(context)

    prefix = [90] * 63 + [110, 110]
    first = dataset(prefix + [55], openings={65: 108})
    second = dataset(prefix + [28], openings={65: 108})

    a = run((first,), strategy, (StockSplit("SPY", day(66), 2),))
    b = run((second,), strategy, (StockSplit("SPY", day(66), 4),))

    assert a.signals == b.signals == (SignalEvent("SPY", day(64), 1),)
    assert a.fills == b.fills
    assert a.sessions[:65] == b.sessions[:65]
    assert a.sessions[-1].portfolio.positions != b.sessions[-1].portfolio.positions
    for context in contexts:
        assert context._effective_splits == ()
        assert context.strategy_close(context.history("SPY")[0]) == Decimal(90)
        assert time_series_momentum(context) == a.signals


@pytest.mark.parametrize(
    "adjustment", [PriceAdjustment.SPLIT_ADJUSTED, PriceAdjustment.TOTAL_RETURN_ADJUSTED]
)
def test_split_aware_backtest_still_requires_raw_data(adjustment):
    metadata, bars = dataset([200, 100])

    with pytest.raises(ValueError, match="backtest requires RAW data"):
        run(
            ((replace(metadata, adjustment=adjustment), bars),),
            time_series_momentum, (StockSplit("SPY", day(2), 2),),
        )


def test_split_schedule_rejects_conflicting_records_before_strategy_replay():
    spy = dataset([200, 100])
    split = StockSplit("SPY", day(2), 2)
    for duplicate in (split, replace(split, ratio=3)):
        with pytest.raises(ValueError, match="duplicate or conflicting split"):
            run(
                (spy,), lambda context: pytest.fail("strategy ran with invalid split data"),
                (split, duplicate),
            )
