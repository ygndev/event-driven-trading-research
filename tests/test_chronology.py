from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research import benchmark
from trading_research.benchmark import BuyAndHoldEntryError
from trading_research.chronology import (
    RESEARCH_PERIODS,
    ResearchPeriod,
    evaluate_spy_research_periods,
)
from trading_research.daily_bars import DailyBar, DatasetMetadata, PriceAdjustment
from trading_research.mean_reversion import MeanReversionExitRejected
from trading_research.orders import OrderSide, OrderState
from trading_research.signals import SignalEvent


METADATA = DatasetMetadata(
    "Synthetic", "SPY", PriceAdjustment.RAW, datetime(2026, 1, 1, tzinfo=UTC)
)


def weekdays(start, end, *, closed):
    days = (start + timedelta(days=i) for i in range((end - start).days + 1))
    return [day for day in days if day.weekday() < 5 and day not in closed]


@pytest.fixture
def bars():
    # Sparse fabricated prices on selected session dates.
    warmup = weekdays(
        date(2009, 9, 28), date(2009, 12, 25),
        closed={date(2009, 11, 26), date(2009, 12, 25)},
    )
    development = weekdays(date(2010, 1, 4), date(2010, 1, 11), closed=set())
    development.append(date(2018, 12, 31))
    validation = weekdays(
        date(2019, 1, 2), date(2019, 1, 24), closed={date(2019, 1, 21)}
    )
    validation.append(date(2022, 12, 30))
    final = [date(2023, 1, 3), date(2023, 1, 4), date(2025, 12, 31)]
    closes = (
        [90] + [100] * 62
        + [97, 99, 101, 103, 105, 104, 106]
        + [97, 99, 101, 103, 105, 104, 106] + [104] * 7 + [102, 106, 108]
        + [97, 99, 106]
    )
    opens = {65: 96, 70: 101, 72: 96, 77: 101, 86: 102, 89: 96}
    observations = []
    for number, (session, close) in enumerate(
        zip(warmup + development + validation + final, closes, strict=True),
        start=1,
    ):
        opening = opens.get(number, close)
        observations.append(
            DailyBar(
                "SPY", session, opening, max(opening, close),
                min(opening, close), close, 1000,
            )
        )
    return observations


def evaluate(bars, *, periods=RESEARCH_PERIODS, bps=25, fee=Decimal("1.25"), strict_warmup=True):
    return evaluate_spy_research_periods(
        METADATA, bars, initial_cash=1000, adjustment_bps=bps, commission=fee,
        periods=periods, strict_warmup=strict_warmup,
    )


def test_fixed_period_config_and_scored_dates(bars):
    results = evaluate(bars)

    assert [
        (row.period.name, row.period.start_date, row.period.end_date)
        for row in results
    ] == [
        ("development", date(2010, 1, 1), date(2018, 12, 31)),
        ("validation", date(2019, 1, 1), date(2022, 12, 31)),
        ("final_test", date(2023, 1, 1), date(2025, 12, 31)),
    ]
    assert [row.warmup_observation_count for row in results] == [63, 70, 87]
    assert [
        (row.comparison.start_session, row.comparison.end_session)
        for row in results
    ] == [
        (date(2010, 1, 4), date(2018, 12, 31)),
        (date(2019, 1, 2), date(2022, 12, 30)),
        (date(2023, 1, 3), date(2025, 12, 31)),
    ]
    for row in results:
        period, comparison = row.period, row.comparison
        dates = tuple(
            bar.session_date for bar in bars
            if period.start_date <= bar.session_date <= period.end_date
        )
        assert (
            comparison.initial_cash, comparison.adjustment_bps,
            comparison.commission,
        ) == (1000, 25, Decimal("1.25"))
        for run in (
            comparison.momentum, comparison.mean_reversion,
            comparison.buy_and_hold, comparison.cash,
        ):
            assert tuple(day for day, _ in run.equity_path) == dates
            assert tuple(
                item.session_date for item in run.metrics.session_returns
            ) == dates[1:]
            assert run.equity_path[0] == (dates[0], Decimal(1000))
            baseline = run.backtest.sessions[0].portfolio
            assert baseline.initial_cash == baseline.cash == 1000
            assert baseline.positions == {}
            assert baseline.cumulative_commissions == 0
            assert all(
                dates[0] <= entry.order.decision_session <= dates[-1]
                for entry in run.backtest.order_history
            )
            assert all(
                dates[0] < fill.execution_session <= dates[-1]
                for fill in run.backtest.fills
            )
        assert comparison.cash.equity_path == tuple(
            (day, Decimal(1000)) for day in dates
        )


def test_validation_preserves_momentum_schedule_and_warmup_history(bars, monkeypatch):
    contexts = []
    strategy = benchmark.short_term_mean_reversion

    def capture(context):
        contexts.append(context)
        return strategy(context)

    monkeypatch.setattr(benchmark, "short_term_mean_reversion", capture)
    validation, = evaluate(bars, periods=(RESEARCH_PERIODS[1],))
    comparison = validation.comparison

    first = contexts[0]
    assert first.session_date == date(2019, 1, 2)
    assert tuple(event.bar for event in first.history("SPY")) == tuple(bars[:71])
    assert first.execution.portfolio.positions == {}
    assert first.execution.portfolio.cash == 1000
    assert first.execution.pending_orders == first.execution.outcomes == ()
    assert first.last_order_outcome("SPY") is None
    assert contexts[-1].session_date == date(2022, 12, 30)

    momentum = comparison.momentum.backtest
    assert momentum.signals == (SignalEvent("SPY", date(2019, 1, 23), 1),)
    assert all(session.equity == 1000 for session in momentum.sessions[:15])
    buy, = momentum.fills
    assert buy.execution_session == date(2019, 1, 24)
    assert (buy.quantity, buy.price) == (9, Decimal("102.255"))
    assert momentum.sessions[-1].portfolio.cash == Decimal("78.455")
    assert momentum.sessions[-1].portfolio.positions == {"SPY": 9}
    assert momentum.sessions[-1].equity == Decimal("1050.455")
    assert comparison.mean_reversion.backtest.signals[0] == SignalEvent(
        "SPY", first.session_date, 1
    )


def test_synthetic_development_decisions_fills_and_final_valuation(bars):
    development, = evaluate(bars, periods=(RESEARCH_PERIODS[0],))
    comparison = development.comparison
    momentum, mean_reversion, holding, cash = (
        comparison.momentum, comparison.mean_reversion,
        comparison.buy_and_hold, comparison.cash,
    )
    anchor = date(2010, 1, 4)
    last = date(2018, 12, 31)
    entry = SignalEvent("SPY", anchor, 1)
    assert momentum.backtest.signals == holding.backtest.signals == (entry,)
    assert mean_reversion.backtest.signals == (
        entry, SignalEvent("SPY", date(2010, 1, 11), 0),
    )

    buy, sell = mean_reversion.backtest.transactions
    assert momentum.backtest.transactions == holding.backtest.transactions == (buy,)
    assert cash.backtest.transactions == ()
    assert (buy.fill.execution_session, buy.fill.side, buy.fill.quantity,
            buy.fill.price, buy.commission, buy.cash_after) == (
        date(2010, 1, 5), OrderSide.BUY, 10, Decimal("96.24"),
        Decimal("1.25"), Decimal("36.35"),
    )
    assert (sell.fill.execution_session, sell.fill.side, sell.fill.quantity,
            sell.fill.price, sell.commission, sell.cash_after) == (
        last, OrderSide.SELL, 10, Decimal("100.7475"),
        Decimal("1.25"), Decimal("1042.575"),
    )
    runs = (momentum, mean_reversion, holding, cash)
    assert [
        run.backtest.sessions[-1].portfolio.positions for run in runs
    ] == [{"SPY": 10}, {}, {"SPY": 10}, {}]
    assert [run.equity_path[-1] for run in runs] == [
        (last, Decimal("1096.35")), (last, Decimal("1042.575")),
        (last, Decimal("1096.35")), (last, Decimal(1000)),
    ]
    assert comparison.outstanding_exit is None


def test_period_exit_cannot_fill_in_the_next_period(bars):
    early = ResearchPeriod("early", date(2010, 1, 1), date(2010, 1, 11))
    later = ResearchPeriod("later", date(2010, 1, 12), date(2019, 1, 4))
    first, second = evaluate(bars, periods=(early, later))

    run = first.comparison.mean_reversion
    order, = run.backtest.outstanding_orders
    assert order is first.comparison.outstanding_exit
    assert order.order.decision_session == first.comparison.end_session
    assert order.state is OrderState.ACCEPTED
    assert order.order.side is OrderSide.SELL
    assert order.fill is None
    assert run.metrics.executed_orders == 1
    assert run.metrics.annualized_return is None
    assert "at least 365 days" in run.metrics.annualized_return_reason
    assert run.backtest.sessions[-1].portfolio.positions == {"SPY": 10}
    assert run.equity_path[-1][1] == Decimal("1076.35")

    following = second.comparison.mean_reversion.backtest
    assert second.comparison.start_session == date(2018, 12, 31)
    assert following.sessions[0].portfolio.cash == 1000
    assert following.sessions[0].portfolio.positions == {}
    buy, = following.fills
    assert buy.side is OrderSide.BUY
    assert buy.execution_session == date(2019, 1, 3)
    assert following.outstanding_orders == ()


def test_open_position_without_an_exit_decision_remains_marked_at_final_close(bars):
    _, validation, final = evaluate(bars)

    assert validation.comparison.momentum.backtest.sessions[-1].portfolio.positions == {
        "SPY": 9
    }
    assert final.comparison.momentum.equity_path == final.comparison.cash.equity_path
    assert final.comparison.momentum.backtest.order_history == ()
    run = final.comparison.mean_reversion
    buy, = run.backtest.fills
    assert buy.side is OrderSide.BUY
    assert run.backtest.sessions[-1].portfolio.positions == {"SPY": 10}
    assert run.equity_path[-1] == (date(2025, 12, 31), Decimal("1096.35"))
    assert run.backtest.outstanding_orders == ()
    assert final.comparison.outstanding_exit is None


def test_exit_rejection_can_only_occur_within_the_period(bars):
    bars[64] = replace(bars[64], open=90, low=90)
    bars[69] = replace(bars[69], open=1, low=1)
    truncated = ResearchPeriod("truncated", date(2010, 1, 1), date(2010, 1, 11))
    pending, = evaluate(bars, periods=(truncated,), bps=0, fee=50)
    assert pending.comparison.outstanding_exit.state is OrderState.ACCEPTED

    with pytest.raises(MeanReversionExitRejected) as failure:
        evaluate(bars, periods=(RESEARCH_PERIODS[0],), bps=0, fee=50)

    rejected = failure.value.rejection
    assert rejected.order.decision_session == date(2010, 1, 11)
    assert rejected.state is OrderState.REJECTED
    assert rejected.fill is None
    assert "commission exceeds proceeds" in rejected.rejection_reason


def test_final_test_price_changes_leave_earlier_results_identical(bars):
    original = evaluate(bars)
    changed = bars[:87] + [
        replace(bar, open=price, high=price, low=price, close=price)
        for bar, price in zip(bars[87:], (200, 201, 202), strict=True)
    ]

    changed[88] = replace(changed[88], open=195, low=195)
    altered = evaluate(changed)

    assert altered[:2] == original[:2]
    assert altered[-1].comparison.mean_reversion.backtest.signals == ()
    assert original[-1].comparison.mean_reversion.backtest.fills
    assert evaluate(bars[:87], periods=RESEARCH_PERIODS[:2]) == original[:2]


def test_period_ending_at_anchor_cannot_use_later_open_for_benchmark(bars):
    anchor = date(2010, 1, 4)
    period = ResearchPeriod("anchor_only", anchor, anchor)
    assert any(bar.session_date > anchor for bar in bars)

    with pytest.raises(BuyAndHoldEntryError, match="2010-01-04:SPY.*outstanding") as failure:
        evaluate(bars, periods=(period,))

    error = failure.value
    assert (error.anchor, error.status) == (anchor, "outstanding")
    assert error.entry.state is OrderState.ACCEPTED
    assert error.entry.order.decision_session == anchor
    assert error.entry.fill is error.entry.rejection_reason is None


def test_first_eligible_close_can_follow_in_period_warmup(bars):
    period = ResearchPeriod("partial_history", bars[61].session_date, date(2010, 1, 5))
    result, = evaluate(bars, periods=(period,), strict_warmup=False)

    assert result.warmup_observation_count == 63
    assert result.comparison.start_session == date(2010, 1, 4)
    comparison = result.comparison
    assert comparison.cash.equity_path == (
        (date(2010, 1, 4), Decimal(1000)), (date(2010, 1, 5), Decimal(1000))
    )
    for run in (comparison.momentum, comparison.mean_reversion, comparison.buy_and_hold, comparison.cash):
        interval, = run.metrics.session_returns
        assert interval.session_date == date(2010, 1, 5)
        assert run.metrics.annualized_volatility is run.metrics.sharpe_ratio is None


def test_insufficient_warmup_cannot_borrow_sessions_from_after_the_window(bars):
    period = ResearchPeriod("warmup_only", bars[0].session_date, bars[62].session_date)

    with pytest.raises(
        ValueError, match="warmup_only: evaluation requires 63 earlier SPY closes"
    ):
        evaluate(bars, periods=(period,))


def test_weekend_only_window_has_no_observations(bars):
    period = ResearchPeriod("weekend", date(2010, 1, 2), date(2010, 1, 3))

    with pytest.raises(ValueError, match="weekend: no SPY observations"):
        evaluate(bars, periods=(period,))


def test_rejects_overlapping_inclusive_and_out_of_order_periods(bars):
    first = RESEARCH_PERIODS[0]
    overlap = ResearchPeriod("overlap", first.end_date, date(2019, 1, 2))

    for periods in ((first, overlap), (RESEARCH_PERIODS[1], first)):
        with pytest.raises(ValueError, match="chronological and non-overlapping"):
            evaluate(bars, periods=periods)


def test_research_period_rejects_invalid_boundaries():
    with pytest.raises(ValueError, match="research start must not follow"):
        ResearchPeriod("bad", date(2010, 1, 2), date(2010, 1, 1))
    with pytest.raises(ValueError, match="boundaries must be dates"):
        ResearchPeriod("timestamp", datetime(2010, 1, 1), date(2010, 1, 2))


@pytest.mark.parametrize(("problem", "message"), [
    ("duplicate", "duplicate session date"),
    ("order", "session date is out of order"),
    ("price", "close must be a finite, positive"),
    ("adjusted", "requires RAW data"),
    ("other_symbol", "requires a SPY dataset"),
])
def test_rejects_invalid_or_incompatible_input_before_replay(bars, problem, message):
    metadata = METADATA
    if problem == "duplicate":
        bars[-1] = bars[-2]
    elif problem == "order":
        bars[-2], bars[-1] = bars[-1], bars[-2]
    elif problem == "price":
        bars[-1] = replace(bars[-1], close=0)
    elif problem == "adjusted":
        metadata = replace(metadata, adjustment=PriceAdjustment.TOTAL_RETURN_ADJUSTED)
    else:
        metadata = replace(metadata, symbol="QQQ")

    with pytest.raises(ValueError, match=message):
        evaluate_spy_research_periods(
            metadata, bars, initial_cash=1000, adjustment_bps=25,
            commission=Decimal("1.25"),
        )


def test_period_results_are_deterministic_immutable_and_preserve_input(bars):
    original = tuple(bars)
    results = evaluate(bars)

    assert results == evaluate(bars)
    assert tuple(bars) == original
    with pytest.raises(FrozenInstanceError):
        setattr(results[0], "warmup_observation_count", 0)
