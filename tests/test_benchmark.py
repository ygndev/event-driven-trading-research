from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research import benchmark
from trading_research.benchmark import compare_spy_benchmarks
from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.orders import OrderSide, OrderState
from trading_research.signals import SignalEvent

ANCHOR = date(2025, 3, 5)
FIRST_FILL = date(2025, 3, 6)
REVERSAL = date(2025, 3, 26)
SELL_FILL = date(2025, 3, 27)


def spy_bars(closes, *, openings=None, dates=None):
    openings = openings or {}
    if dates is None:
        dates = (
            date(2025, 1, 1) + timedelta(days=index)
            for index in range(len(closes))
        )
    bars = []
    for index, (session, close) in enumerate(
        zip(dates, closes, strict=True), start=1
    ):
        opening = openings.get(index, close)
        bars.append(
            DailyBar(
                "SPY", session, opening, max(opening, close),
                min(opening, close), close, 1_000,
            )
        )
    return bars


def compare(
    bars, *, adjustment=PriceAdjustment.RAW, bps=25, fee=Decimal("1.25")
):
    metadata = DatasetMetadata(
        "Synthetic", "SPY", adjustment, datetime(2028, 1, 1, tzinfo=UTC)
    )
    return compare_spy_benchmarks(
        metadata, bars, initial_cash=1000, adjustment_bps=bps, commission=fee
    )


def test_three_way_reversal_uses_equal_capital_costs_and_aligned_window():
    closes = [90] * 63 + [110] + [112] * 20 + [100, 106]
    closes[21] = 120
    bars = spy_bars(closes, openings={65: 108, 86: 105})

    result = compare(bars)

    assert result.start_session == ANCHOR
    assert result.end_session == SELL_FILL
    assert result.initial_cash == Decimal("1000")
    assert result.adjustment_bps == Decimal("25")
    assert result.commission == Decimal("1.25")
    runs = (result.momentum, result.buy_and_hold, result.cash)
    assert tuple(run.label for run in runs) == (
        "SPY momentum", "SPY buy-and-hold", "Cash (zero interest)"
    )
    expected_dates = tuple(bar.session_date for bar in bars[63:])
    for run in runs:
        assert tuple(day for day, _ in run.equity_path) == expected_dates
        assert run.equity_path[0] == (ANCHOR, Decimal("1000"))
        first = run.backtest.sessions[0].portfolio
        assert first.initial_cash == first.cash == Decimal("1000")
        assert first.positions == {}
        assert first.cumulative_commissions == 0
        assert run.metrics.session_returns[0].simple_return == 0
        period_dates = tuple(
            row.session_date for row in run.metrics.session_returns
        )
        assert period_dates == expected_dates
        assert run.metrics.annualized_return is None
        assert "at least 365 days" in run.metrics.annualized_return_reason
        assert all(
            order.decision_session >= ANCHOR for order in run.backtest.orders
        )
        assert all(
            fill.execution_session > ANCHOR for fill in run.backtest.fills
        )

    momentum, holding, cash = runs
    assert momentum.backtest.signals == (
        SignalEvent("SPY", ANCHOR, 1), SignalEvent("SPY", REVERSAL, 0),
    )
    assert holding.backtest.signals == (SignalEvent("SPY", ANCHOR, 1),)
    assert momentum.backtest.fills[0] == holding.backtest.fills[0]
    buy, = holding.backtest.fills
    assert (buy.execution_session, buy.side, buy.quantity, buy.price) == (
        FIRST_FILL, OrderSide.BUY, 9, Decimal("108.27")
    )
    sale = momentum.backtest.fills[-1]
    assert (sale.execution_session, sale.side, sale.quantity, sale.price) == (
        SELL_FILL, OrderSide.SELL, 9, Decimal("104.7375")
    )
    for scored in (momentum, holding):
        assert scored.backtest.sessions[1].portfolio.cash == Decimal("24.32")
        assert scored.equity_path[1] == (FIRST_FILL, Decimal("1032.32"))
        assert scored.metrics.session_returns[1].simple_return == Decimal(
            "0.03232"
        )
    assert all(
        item.portfolio.positions == {"SPY": 9}
        for item in holding.backtest.sessions[1:]
    )
    assert holding.backtest.sessions[-1].portfolio.cash == Decimal("24.32")
    assert momentum.backtest.sessions[-1].portfolio.positions == {}
    assert momentum.backtest.sessions[-1].portfolio.cash == Decimal("965.7075")

    assert [run.equity_path[-1][1] for run in runs] == [
        Decimal("965.7075"), Decimal("978.32"), Decimal("1000")
    ]
    assert [run.metrics.total_return for run in runs] == [
        Decimal("-0.0342925"), Decimal("-0.02168"), Decimal(0)
    ]
    assert [run.metrics.total_commissions for run in runs] == [
        Decimal("2.50"), Decimal("1.25"), Decimal(0)
    ]
    assert [run.metrics.executed_orders for run in runs] == [2, 1, 0]
    assert [run.metrics.executed_notional for run in runs] == [
        Decimal("1917.0675"), Decimal("974.43"), Decimal(0)
    ]
    # The 23 closing equities include the anchor and exclude warm-up cash.
    assert momentum.metrics.turnover == pytest.approx(
        Decimal("1.8733748993979653"), rel=Decimal("1e-12")
    )
    assert holding.metrics.turnover == pytest.approx(
        Decimal("0.9517114073439936"), rel=Decimal("1e-12")
    )
    assert [run.return_difference_pp_vs_buy_and_hold for run in runs] == [
        Decimal("-1.26125"), Decimal(0), Decimal("2.168")
    ]
    assert [run.equity_difference_usd_vs_buy_and_hold for run in runs] == [
        Decimal("-12.6125"), Decimal(0), Decimal("21.68")
    ]
    assert cash.backtest.signals == ()
    assert cash.backtest.orders == cash.backtest.fills == ()
    assert all(equity == Decimal("1000") for _, equity in cash.equity_path)
    assert cash.metrics.average_long_exposure == 0
    assert cash.metrics.annualized_volatility == 0
    assert cash.metrics.sharpe_ratio is None


def test_rising_market_executes_a_positive_gap_at_the_next_observed_open():
    dates = [
        date(2025, 1, 1) + timedelta(days=index)
        for index in range(64)
    ] + [date(2025, 3, 10), date(2025, 3, 11)]
    bars = spy_bars(
        [*range(35, 98), 110, 120, 130],
        dates=dates, openings={65: 110.5},
    )

    result = compare(bars)

    observed_dates = tuple(day for day, _ in result.momentum.equity_path)
    assert observed_dates == tuple(dates[63:])
    for run in (result.momentum, result.buy_and_hold):
        fill, = run.backtest.fills
        assert fill.execution_session == date(2025, 3, 10)
        assert fill.price == Decimal("110.77625")
        assert fill.quantity == 9
        assert run.backtest.sessions[-1].portfolio.cash == Decimal("1.76375")
        assert run.equity_path[-1][1] == Decimal("1171.76375")
        assert run.metrics.total_return == Decimal("0.17176375")


def test_falling_market_keeps_same_anchor_when_momentum_targets_zero():
    bars = spy_bars([*range(200, 137, -1), 100, 90, 75])

    result = compare(bars)

    assert result.start_session == ANCHOR
    assert result.momentum.backtest.signals == (SignalEvent("SPY", ANCHOR, 0),)
    assert result.momentum.backtest.fills == ()
    assert result.momentum.equity_path == result.cash.equity_path
    holding = result.buy_and_hold.backtest
    assert holding.signals == (SignalEvent("SPY", ANCHOR, 1),)
    buy, = result.buy_and_hold.backtest.fills
    assert (buy.execution_session, buy.price, buy.quantity) == (
        FIRST_FILL, Decimal("90.225"), 10
    )
    final = result.buy_and_hold.backtest.sessions[-1]
    assert final.portfolio.positions == {"SPY": 10}
    assert final.portfolio.cash == Decimal("96.50")
    assert final.equity == Decimal("846.50")
    assert result.momentum.return_difference_pp_vs_buy_and_hold == Decimal(
        "15.35"
    )
    assert result.momentum.equity_difference_usd_vs_buy_and_hold == Decimal(
        "153.50"
    )


def test_future_open_changes_fill_but_never_anchor_signal_or_quantity():
    low = compare(spy_bars([90] * 63 + [110, 112], openings={65: 105}))
    high = compare(spy_bars([90] * 63 + [110, 112], openings={65: 108}))

    pairs = (
        (low.momentum, high.momentum),
        (low.buy_and_hold, high.buy_and_hold),
    )
    for before, after in pairs:
        assert before.backtest.signals == after.backtest.signals
        assert before.backtest.orders == after.backtest.orders
        assert before.backtest.orders[0].quantity == 9
        assert before.backtest.orders[0].decision_session == ANCHOR
        assert before.backtest.fills[0].price == Decimal("105.2625")
        assert after.backtest.fills[0].price == Decimal("108.27")


def test_unaffordable_first_open_is_rejected_without_retry_or_commission():
    bars = spy_bars([90] * 63 + [110, 120, 100])

    result = compare(bars)

    for run in (result.momentum, result.buy_and_hold):
        assert run.backtest.fills == run.backtest.outstanding_orders == ()
        rejected, = run.backtest.rejected_orders
        assert "insufficient cash" in rejected.rejection_reason
        assert run.metrics.total_commissions == 0
        assert run.equity_path == result.cash.equity_path


def test_anchor_only_has_pending_orders_and_undefined_statistics():
    result = compare(spy_bars([90] * 63 + [110]))

    assert result.start_session == result.end_session == ANCHOR
    for run in (result.momentum, result.buy_and_hold):
        pending, = run.backtest.outstanding_orders
        assert pending.state is OrderState.ACCEPTED
        assert pending.order.quantity == 9
        assert run.backtest.fills == ()
        assert run.equity_path == ((ANCHOR, Decimal("1000")),)
        assert run.metrics.total_return == 0
        assert run.metrics.annualized_return is None
        assert run.metrics.annualized_volatility is None
        assert run.metrics.sharpe_ratio is None


def test_cagr_uses_anchor_to_last_close_instead_of_the_warmup_period():
    dates = [
        date(2025, 1, 1) + timedelta(days=index)
        for index in range(64)
    ] + [ANCHOR + timedelta(days=730)]
    bars = spy_bars([90] * 63 + [120, 126.25], dates=dates, openings={65: 100})

    result = compare(bars, bps=0, fee=0)

    for run in (result.momentum, result.buy_and_hold):
        assert run.equity_path[-1][1] == Decimal("1210")
        assert run.metrics.annualized_return == Decimal("0.10")
        assert run.metrics.annualized_return_reason is None
    assert result.cash.metrics.annualized_return == 0


@pytest.mark.parametrize("count", [0, 63])
def test_insufficient_warmup_is_rejected(count):
    with pytest.raises(ValueError, match="requires 64 SPY closes"):
        compare(spy_bars([100] * count))


def test_adjusted_data_remains_outside_raw_execution_contract():
    with pytest.raises(ValueError, match="backtest requires RAW data"):
        bars = spy_bars([100] * 64)
        compare(bars, adjustment=PriceAdjustment.SPLIT_ADJUSTED)


@pytest.mark.parametrize("problem", ["missing", "incompatible"])
def test_misaligned_valuation_dates_are_rejected_without_forward_filling(
    monkeypatch, problem
):
    run_backtest = benchmark.run_backtest
    runs = []

    def misaligned(*args, **kwargs):
        result = run_backtest(*args, **kwargs)
        runs.append(result)
        if len(runs) != 2:
            return result
        if problem == "missing":
            return replace(result, sessions=result.sessions[:-1])
        final = replace(result.sessions[-1], session_date=date(2025, 3, 7))
        return replace(result, sessions=(*result.sessions[:-1], final))

    monkeypatch.setattr(benchmark, "run_backtest", misaligned)
    with pytest.raises(ValueError, match="valuation dates must match"):
        compare(spy_bars([90] * 63 + [110, 108]))


def test_repeatable_comparison_leaves_source_backtests_intact(monkeypatch):
    run_backtest = benchmark.run_backtest
    sources = []

    def capture(*args, **kwargs):
        result = run_backtest(*args, **kwargs)
        sources.append(result)
        return result

    monkeypatch.setattr(benchmark, "run_backtest", capture)
    bars = spy_bars([90] * 63 + [110, 108])
    original = [replace(bar) for bar in bars]

    result = compare(bars)

    assert result == compare(bars)
    assert bars == original
    assert len(sources) == 6
    first_dates = {source.sessions[0].session_date for source in sources}
    assert first_dates == {date(2025, 1, 1)}
    assert all(len(source.sessions) == 65 for source in sources)
    assert len(result.momentum.backtest.sessions) == 2
    with pytest.raises(FrozenInstanceError):
        setattr(result, "start_session", date(2025, 1, 1))
    with pytest.raises(FrozenInstanceError):
        setattr(result.cash, "label", "other")


def test_rejects_a_changed_initial_equity_at_the_common_anchor(monkeypatch):
    run_backtest = benchmark.run_backtest

    def changed_anchor(*args, **kwargs):
        result = run_backtest(*args, **kwargs)
        sessions = list(result.sessions)
        sessions[63] = replace(sessions[63], equity=Decimal("1001"))
        return replace(result, sessions=tuple(sessions))

    monkeypatch.setattr(benchmark, "run_backtest", changed_anchor)
    with pytest.raises(ValueError, match="same uninvested capital"):
        compare(spy_bars([90] * 63 + [110, 108]))
