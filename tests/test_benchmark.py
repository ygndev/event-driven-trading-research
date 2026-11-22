from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research import benchmark
from trading_research.benchmark import (
    BuyAndHoldEntryError,
    compare_spy_benchmarks,
    compare_spy_common_window,
    compare_spy_mean_reversion_benchmarks,
)
from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.mean_reversion import MeanReversionExitRejected
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


@pytest.fixture
def completed_backtests(monkeypatch):
    completed = []
    run_backtest = benchmark.run_backtest

    def record(*args, **kwargs):
        backtest = run_backtest(*args, **kwargs)
        completed.append(backtest)
        return backtest

    monkeypatch.setattr(benchmark, "run_backtest", record)
    return completed


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
    assert holding.entry is holding.backtest.order_history[-1]
    assert holding.entry.state is OrderState.FILLED
    assert holding.entry.fill is buy
    assert momentum.entry is cash.entry is None
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


def test_unaffordable_first_open_is_rejected_without_retry_or_commission(completed_backtests):
    bars = spy_bars([90] * 63 + [110, 120, 100])

    with pytest.raises(BuyAndHoldEntryError, match="insufficient cash") as failure:
        compare(bars)

    error = failure.value
    assert (error.symbol, error.anchor, error.status) == ("SPY", ANCHOR, "rejected")
    momentum, holding = completed_backtests
    assert error.entry is holding.rejected_orders[0]
    for run in (momentum, holding):
        assert run.fills == run.outstanding_orders == ()
        rejected, = run.rejected_orders
        assert rejected.order.quantity == 9
        assert "insufficient cash" in rejected.rejection_reason
        assert all(session.equity == 1000 for session in run.sessions)
        assert run.sessions[-1].portfolio.cumulative_commissions == 0


def test_audited_rejected_entry_cannot_return_a_buy_and_hold_comparison(monkeypatch):
    days = tuple(date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9, 10, 11))
    bars = spy_bars([100] * 6 + [101, 120], openings={7: 101}, dates=days)

    def unexpected_performance(*args, **kwargs):
        pytest.fail("an invalid benchmark must fail before scoring")

    monkeypatch.setattr(benchmark, "evaluate_performance", unexpected_performance)
    failures = []
    for _ in range(2):
        with pytest.raises(BuyAndHoldEntryError, match=r"2024-01-09:SPY.*insufficient cash") as caught:
            compare_mean_reversion(bars, bps=0, fee=0)
        failures.append(caught.value)
    error, repeated = failures
    assert (error.symbol, error.anchor, error.status) == ("SPY", days[5], "rejected")
    assert error.entry.order.order_id == "2024-01-09:SPY"
    assert error.entry.order.quantity == 10
    assert error.entry.order.decision_session == days[5]
    assert error.entry.state is OrderState.REJECTED
    assert error.entry.fill is None
    assert str(repeated) == str(error)
    assert repeated.entry == error.entry


def test_momentum_anchor_only_reports_accepted_benchmark_entry(completed_backtests):
    with pytest.raises(BuyAndHoldEntryError, match="outstanding at the final close") as failure:
        compare(spy_bars([90] * 63 + [110]))

    error = failure.value
    assert (error.anchor, error.status) == (ANCHOR, "outstanding")
    momentum, holding = completed_backtests
    assert error.entry is holding.outstanding_orders[0]
    for run in (momentum, holding):
        pending, = run.outstanding_orders
        assert (pending.state, pending.order.quantity) == (OrderState.ACCEPTED, 9)
        assert run.fills == run.rejected_orders == ()
        assert (run.sessions[-1].session_date, run.sessions[-1].equity) == (ANCHOR, 1000)


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


def compare_mean_reversion(bars, *, bps=25, fee=Decimal("1.25")):
    metadata = DatasetMetadata(
        "Synthetic", "SPY", PriceAdjustment.RAW,
        datetime(2028, 1, 1, tzinfo=UTC),
    )
    return compare_spy_mean_reversion_benchmarks(
        metadata, bars, initial_cash=1000, adjustment_bps=bps, commission=fee
    )


@pytest.mark.parametrize("count", [0, 5])
def test_mean_reversion_benchmarks_require_six_observations(count):
    with pytest.raises(ValueError, match="requires 6 SPY closes"):
        compare_mean_reversion(spy_bars([100] * count))


def test_mean_reversion_three_way_comparison_at_exact_threshold():
    dates = tuple(
        date(2025, 1, 2) + timedelta(days=offset)
        for offset in (0, 1, 4, 5, 6, 7, 11, 12, 14, 15, 19, 22)
    )
    bars = spy_bars(
        [100] * 5 + [97, 99, 101, 103, 105, 104, 106],
        dates=dates, openings={7: 96, 12: 101},
    )

    result = compare_mean_reversion(bars)

    anchor, entry = dates[5:7]
    exit_decision, exit_fill = dates[10:12]
    assert result.start_session == anchor
    assert result.end_session == exit_fill
    assert (result.initial_cash, result.adjustment_bps, result.commission) == (
        1000, 25, Decimal("1.25")
    )
    mean_reversion, buy_and_hold, cash = (
        result.mean_reversion, result.buy_and_hold, result.cash
    )
    runs = (mean_reversion, buy_and_hold, cash)
    assert tuple(run.label for run in runs) == (
        "SPY mean reversion", "SPY buy-and-hold", "Cash (zero interest)"
    )
    for run in runs:
        anchor_portfolio = run.backtest.sessions[0].portfolio
        assert anchor_portfolio.initial_cash == anchor_portfolio.cash == 1000
        assert anchor_portfolio.positions == {}
        assert anchor_portfolio.cumulative_commissions == 0
        assert run.backtest.rejected_orders == ()
        assert run.backtest.outstanding_orders == ()
        assert all(
            order.decision_session >= anchor for order in run.backtest.orders
        )
        scored_dates = tuple(
            row.session_date for row in run.metrics.session_returns
        )
        assert scored_dates == dates[5:]
        assert run.metrics.session_returns[0].simple_return == 0
        assert run.metrics.annualized_return is None
        assert "at least 365 days" in run.metrics.annualized_return_reason

    assert mean_reversion.backtest.signals == (
        SignalEvent("SPY", anchor, 1), SignalEvent("SPY", exit_decision, 0)
    )
    assert buy_and_hold.backtest.signals == (SignalEvent("SPY", anchor, 1),)
    assert cash.backtest.signals == cash.backtest.order_history == ()
    purchase, sale = mean_reversion.backtest.transactions
    assert buy_and_hold.backtest.transactions == (purchase,)
    assert cash.backtest.transactions == ()
    bought, sold = purchase.fill, sale.fill
    assert (bought.execution_session, bought.side, bought.quantity, bought.price) == (
        entry, OrderSide.BUY, 10, Decimal("96.24")
    )
    assert (sold.execution_session, sold.side, sold.quantity, sold.price) == (
        exit_fill, OrderSide.SELL, 10, Decimal("100.7475")
    )
    assert purchase.commission == sale.commission == Decimal("1.25")
    assert purchase.cash_after == Decimal("36.35")
    assert sale.cash_after == Decimal("1042.575")
    for session in mean_reversion.backtest.sessions[1:-1]:
        assert session.portfolio.positions == {"SPY": 10}
    assert result.outstanding_exit is None

    common_equity = (
        "1000", "1026.35", "1046.35", "1066.35", "1086.35", "1076.35"
    )
    expected_equities = (
        (*common_equity, "1042.575"), (*common_equity, "1096.35"), ("1000",) * 7
    )
    for run, equities in zip(runs, expected_equities, strict=True):
        expected_path = tuple(zip(dates[5:], map(Decimal, equities), strict=True))
        assert run.equity_path == expected_path

    final = [run.backtest.sessions[-1].portfolio for run in runs]
    assert [portfolio.positions for portfolio in final] == [{}, {"SPY": 10}, {}]
    assert [portfolio.cash for portfolio in final] == [
        Decimal("1042.575"), Decimal("36.35"), Decimal("1000")
    ]
    assert [portfolio.cumulative_commissions for portfolio in final] == [
        Decimal("2.50"), Decimal("1.25"), Decimal(0)
    ]
    assert [run.metrics.total_return for run in runs] == [
        Decimal("0.042575"), Decimal("0.09635"), Decimal(0)
    ]
    assert [run.metrics.total_commissions for run in runs] == [
        Decimal("2.50"), Decimal("1.25"), Decimal(0)
    ]
    assert [run.metrics.executed_orders for run in runs] == [2, 1, 0]
    assert [run.metrics.executed_notional for run in runs] == [
        Decimal("1969.875"), Decimal("962.40"), Decimal(0)
    ]
    assert mean_reversion.metrics.max_drawdown == pytest.approx(
        Decimal("43.775") / Decimal("1086.35")
    )
    assert mean_reversion.metrics.turnover == pytest.approx(
        Decimal("1969.875") / (Decimal("7344.325") / 7)
    )
    assert 0 < mean_reversion.metrics.average_long_exposure < 1
    assert mean_reversion.metrics.annualized_volatility > 0
    assert mean_reversion.metrics.sharpe_ratio is not None
    assert cash.metrics.average_long_exposure == 0
    assert cash.metrics.annualized_volatility == 0
    assert cash.metrics.sharpe_ratio is None
    assert mean_reversion.return_difference_pp_vs_buy_and_hold == Decimal(
        "-5.3775"
    )
    assert mean_reversion.equity_difference_usd_vs_buy_and_hold == Decimal(
        "-53.775"
    )
    assert result.return_difference_pp_vs_cash == Decimal("4.2575")


def test_mean_reversion_loses_money_on_a_continuing_decline():
    bars = spy_bars(
        [100] * 5 + [97, 96, 92, 90, 88, 86, 81], openings={12: 82}
    )

    result = compare_mean_reversion(bars)

    mean_reversion = result.mean_reversion
    bought, sold = mean_reversion.backtest.fills
    assert bought.execution_session == bars[6].session_date
    assert sold.execution_session == bars[11].session_date
    assert sold.price == Decimal("81.795")
    assert mean_reversion.equity_path[-1][1] == Decimal("853.05")
    assert result.buy_and_hold.equity_path[-1][1] == Decimal("846.35")
    assert mean_reversion.metrics.total_return == Decimal("-0.14695")
    assert mean_reversion.return_difference_pp_vs_buy_and_hold == Decimal("0.67")
    assert mean_reversion.equity_difference_usd_vs_buy_and_hold == Decimal("6.70")
    assert result.return_difference_pp_vs_cash == Decimal("-14.695")


def test_mean_reversion_without_a_trigger_keeps_the_same_anchor():
    bars = spy_bars([100] * 5 + [97.0001, 99, 100])

    result = compare_mean_reversion(bars)

    anchor = bars[5].session_date
    assert result.start_session == anchor
    assert result.mean_reversion.backtest.signals == ()
    assert result.mean_reversion.backtest.order_history == ()
    assert result.mean_reversion.equity_path == result.cash.equity_path
    assert result.return_difference_pp_vs_cash == 0
    holding = result.buy_and_hold.backtest
    assert holding.signals == (SignalEvent("SPY", anchor, 1),)
    fill, = holding.fills
    assert fill.execution_session == bars[6].session_date


def test_mean_reversion_anchor_only_reports_accepted_benchmark_entry(completed_backtests):
    bars = spy_bars([100] * 5 + [97])

    with pytest.raises(BuyAndHoldEntryError, match="outstanding at the final close") as failure:
        compare_mean_reversion(bars)

    error = failure.value
    assert (error.anchor, error.status) == (bars[5].session_date, "outstanding")
    reversion, holding = completed_backtests
    assert error.entry is holding.outstanding_orders[0]
    for run in (reversion, holding):
        pending, = run.outstanding_orders
        assert pending.state is OrderState.ACCEPTED
        assert pending.order.side is OrderSide.BUY and pending.order.quantity == 10
        assert run.fills == run.rejected_orders == ()
        assert run.sessions[-1].portfolio.cumulative_commissions == 0
        assert (run.sessions[-1].session_date, run.sessions[-1].equity) == (error.anchor, 1000)


def test_mean_reversion_ending_with_a_pending_sell_keeps_an_open_position():
    bars = spy_bars([100] * 5 + [97, 96, 92, 90, 88, 86])

    result = compare_mean_reversion(bars)

    run = result.mean_reversion
    pending, = run.backtest.outstanding_orders
    assert result.outstanding_exit is pending
    assert pending.state is OrderState.ACCEPTED
    assert pending.order.side is OrderSide.SELL
    assert pending.order.decision_session == result.end_session
    assert pending.order.quantity == 10
    assert pending.fill is None
    assert run.backtest.rejected_orders == ()
    assert run.metrics.executed_orders == 1
    assert run.metrics.total_commissions == Decimal("1.25")
    assert run.backtest.sessions[-1].portfolio.positions == {"SPY": 10}
    assert run.equity_path[-1] == (result.end_session, Decimal("896.35"))


def test_rejected_mean_reversion_entry_stays_flat_without_a_fresh_crossing():
    bars = spy_bars([100] * 6 + [97, 95, 94, 93, 92, 91], openings={7: 96, 8: 105})

    result = compare_mean_reversion(bars)

    run = result.mean_reversion
    rejected, = run.backtest.rejected_orders
    assert rejected.order.side is OrderSide.BUY
    assert "insufficient cash" in rejected.rejection_reason
    assert run.backtest.signals == (SignalEvent("SPY", bars[6].session_date, 1),)
    assert run.backtest.transactions == run.backtest.outstanding_orders == ()
    assert run.equity_path == result.cash.equity_path
    assert result.buy_and_hold.entry.fill.execution_session == bars[6].session_date


def test_rejected_mean_reversion_exit_propagates_without_a_comparison():
    bars = spy_bars([100] * 5 + [97, 90, 80, 50, 30, 5, 1, 2, 3])

    with pytest.raises(MeanReversionExitRejected) as failure:
        compare_mean_reversion(bars, bps=0, fee=50)

    rejection = failure.value.rejection
    assert rejection.state is OrderState.REJECTED
    assert rejection.order.side is OrderSide.SELL
    assert rejection.order.decision_session == bars[10].session_date
    assert rejection.fill is None
    assert "commission exceeds proceeds" in rejection.rejection_reason


def test_mean_reversion_comparison_is_repeatable_and_immutable():
    bars = spy_bars([100] * 5 + [97, 96, 92, 90, 88, 86])
    original = tuple(bars)

    result = compare_mean_reversion(bars)

    assert result == compare_mean_reversion(bars)
    assert tuple(bars) == original
    with pytest.raises(FrozenInstanceError):
        setattr(result, "end_session", result.start_session)


def common_window_bars(
    *, entry_open=96, mean_reversion_exit_open=101, momentum_exit_open=110
):
    warmup = [90] * 63
    warmup[5] = 80
    warmup[21] = 120
    warmup[57:62] = [100] * 5
    warmup[62] = 96
    closes = warmup + [97, 99, 101, 103, 105, 104, 106] + [104] * 14 + [102, 106]
    return spy_bars(
        closes, openings={
            7: 79, 64: 95, 65: entry_open,
            70: mean_reversion_exit_open, 86: momentum_exit_open,
        }
    )


def compare_common_window(bars, *, bps=25, fee=Decimal("1.25")):
    metadata = DatasetMetadata(
        "Synthetic", "SPY", PriceAdjustment.RAW,
        datetime(2028, 1, 1, tzinfo=UTC),
    )
    return compare_spy_common_window(
        metadata, bars, initial_cash=1000, adjustment_bps=bps, commission=fee
    )


def test_common_window_four_portfolio_accounting_and_pairwise_differences():
    bars = common_window_bars()
    result = compare_common_window(bars)

    assert result.start_session == ANCHOR == bars[63].session_date
    assert result.end_session == SELL_FILL == bars[-1].session_date
    assert (result.initial_cash, result.adjustment_bps, result.commission) == (
        1000, 25, Decimal("1.25")
    )
    momentum, mean_reversion, holding, cash = (
        result.momentum, result.mean_reversion, result.buy_and_hold, result.cash
    )
    runs = (momentum, mean_reversion, holding, cash)
    dates = tuple(bar.session_date for bar in bars[63:])
    for run in runs:
        assert tuple(day for day, _ in run.equity_path) == dates
        assert run.equity_path[0] == (ANCHOR, Decimal(1000))
        baseline = run.backtest.sessions[0].portfolio
        assert baseline.initial_cash == baseline.cash == 1000
        assert baseline.positions == {}
        assert baseline.cumulative_commissions == 0
        assert tuple(row.session_date for row in run.metrics.session_returns) == dates
        assert run.metrics.session_returns[0].simple_return == 0
        assert run.metrics.annualized_return is None
        assert "at least 365 days" in run.metrics.annualized_return_reason
        assert all(
            entry.order.decision_session >= ANCHOR
            for entry in run.backtest.order_history
        )
        assert run.backtest.outstanding_orders == ()
        assert run.backtest.rejected_orders == ()

    assert momentum.backtest.signals == (
        SignalEvent("SPY", ANCHOR, 1), SignalEvent("SPY", REVERSAL, 0)
    )
    assert mean_reversion.backtest.signals == (
        SignalEvent("SPY", ANCHOR, 1),
        SignalEvent("SPY", date(2025, 3, 10), 0),
    )
    assert holding.backtest.signals == (SignalEvent("SPY", ANCHOR, 1),)
    assert cash.backtest.signals == cash.backtest.order_history == ()
    assert cash.equity_path == tuple((day, Decimal(1000)) for day in dates)
    assert cash.metrics.sharpe_ratio is None
    assert result.outstanding_exit is None

    purchase, momentum_sale = momentum.backtest.transactions
    assert mean_reversion.backtest.transactions[0] == purchase
    assert holding.backtest.transactions == (purchase,)
    assert cash.backtest.transactions == ()
    mean_reversion_sale = mean_reversion.backtest.transactions[1]
    for trade, day, side, price, balance in (
        (purchase, FIRST_FILL, OrderSide.BUY, "96.24", "36.35"),
        (momentum_sale, SELL_FILL, OrderSide.SELL, "109.725", "1132.35"),
        (mean_reversion_sale, date(2025, 3, 11), OrderSide.SELL,
         "100.7475", "1042.575"),
    ):
        fill = trade.fill
        assert (fill.execution_session, fill.side, fill.quantity, fill.price) == (
            day, side, 10, Decimal(price)
        )
        assert trade.commission == Decimal("1.25")
        assert trade.cash_after == Decimal(balance)
    assert all(
        valuation.portfolio.positions == {"SPY": 10}
        for valuation in mean_reversion.backtest.sessions[1:6]
    )
    assert mean_reversion.backtest.sessions[6].portfolio.positions == {}

    final = [run.backtest.sessions[-1] for run in runs]
    assert [session.portfolio.positions for session in final] == [
        {}, {}, {"SPY": 10}, {}
    ]
    assert [session.portfolio.cash for session in final] == list(map(
        Decimal, ("1132.35", "1042.575", "36.35", "1000")
    ))
    assert [session.equity for session in final] == list(map(
        Decimal, ("1132.35", "1042.575", "1096.35", "1000")
    ))
    assert [run.metrics.total_return for run in runs] == list(map(
        Decimal, ("0.13235", "0.042575", "0.09635", "0")
    ))
    assert [run.metrics.total_commissions for run in runs] == list(map(
        Decimal, ("2.50", "2.50", "1.25", "0")
    ))
    assert [run.metrics.executed_orders for run in runs] == [2, 2, 1, 0]
    assert {
        (gap.left, gap.right): (gap.return_difference_pp, gap.equity_difference_usd)
        for gap in result.pairwise_differences
    } == {
        (momentum.label, mean_reversion.label): (Decimal("8.9775"), Decimal("89.775")),
        (momentum.label, holding.label): (Decimal("3.6"), Decimal("36")),
        (momentum.label, cash.label): (Decimal("13.235"), Decimal("132.35")),
        (mean_reversion.label, holding.label): (Decimal("-5.3775"), Decimal("-53.775")),
        (mean_reversion.label, cash.label): (Decimal("4.2575"), Decimal("42.575")),
        (holding.label, cash.label): (Decimal("9.635"), Decimal("96.35")),
    }


def test_common_window_keeps_mean_reversion_inactive_during_warmup(monkeypatch):
    bars = common_window_bars()
    standalone = compare_mean_reversion(bars[:63]).mean_reversion.backtest

    assert standalone.signals[0] == SignalEvent("SPY", bars[5].session_date, 1)
    assert standalone.fills
    pending, = standalone.outstanding_orders
    assert pending.order.decision_session == bars[62].session_date
    assert pending.order.side is OrderSide.BUY

    strategy = benchmark.short_term_mean_reversion
    contexts = []

    def capture(context):
        contexts.append(context)
        return strategy(context)

    monkeypatch.setattr(benchmark, "short_term_mean_reversion", capture)
    active = compare_common_window(bars).mean_reversion.backtest

    first = contexts[0]
    assert first.session_date == ANCHOR
    assert tuple(event.bar for event in first.history("SPY")) == tuple(bars[:64])
    assert first.execution.portfolio.positions == {}
    assert first.execution.pending_orders == first.execution.outcomes == ()
    assert first.last_order_outcome("SPY") is None
    assert active.signals[0] == SignalEvent("SPY", ANCHOR, 1)
    assert active.orders[0].decision_session == ANCHOR
    assert active.fills[0].execution_session == FIRST_FILL
    assert active.sessions[0].portfolio.positions == {}
    assert active.rejected_orders == ()


@pytest.mark.parametrize(("reversion_open", "expected_winner", "expected_equity"), [
    (109, "SPY mean reversion", Decimal("1122.375")),
    (101, "SPY buy-and-hold", Decimal("1096.35")),
])
def test_synthetic_paths_can_favor_either_reversion_or_holding(
    reversion_open, expected_winner, expected_equity
):
    result = compare_common_window(common_window_bars(
        mean_reversion_exit_open=reversion_open, momentum_exit_open=100
    ))

    winner = max(
        (result.momentum, result.mean_reversion, result.buy_and_hold, result.cash),
        key=lambda run: run.equity_path[-1][1],
    )
    assert winner.label == expected_winner
    assert winner.equity_path[-1][1] == expected_equity


def test_common_window_flat_prices_keep_both_active_portfolios_in_cash():
    bars = spy_bars([100] * 107)
    result = compare_common_window(bars, bps=0, fee=0)

    assert result.momentum.backtest.signals == tuple(
        SignalEvent("SPY", bars[index - 1].session_date, 0)
        for index in (64, 85, 106)
    )
    assert result.mean_reversion.backtest.signals == ()
    for run in (result.momentum, result.mean_reversion, result.cash):
        assert run.backtest.orders == run.backtest.fills == ()
    assert all(gap.return_difference_pp == gap.equity_difference_usd == 0
               for gap in result.pairwise_differences)
    buy, = result.buy_and_hold.backtest.fills
    assert buy.quantity == 10
    assert buy.execution_session == FIRST_FILL
    assert result.buy_and_hold.backtest.sessions[-1].portfolio.positions == {"SPY": 10}
    assert result.buy_and_hold.metrics.sharpe_ratio is None


def test_future_gap_and_price_changes_cannot_change_anchor_orders(completed_backtests):
    bars = common_window_bars()
    with pytest.raises(BuyAndHoldEntryError, match="outstanding") as failure:
        compare_common_window(bars[:64])
    anchor_only = tuple(completed_backtests)
    assert failure.value.entry is anchor_only[-1].outstanding_orders[0]

    opening = replace(
        bars[64], session_date=FIRST_FILL + timedelta(days=3),
        open=94, low=94, high=110, close=110,
    )
    gapped = compare_common_window([*bars[:64], opening])
    original = compare_common_window(bars[:65])

    for single, gap, base in zip(
        anchor_only,
        (gapped.momentum, gapped.mean_reversion, gapped.buy_and_hold),
        (original.momentum, original.mean_reversion, original.buy_and_hold),
        strict=True,
    ):
        accepted, = single.outstanding_orders
        assert accepted.state is OrderState.ACCEPTED
        assert single.fills == ()
        assert single.signals == gap.backtest.signals == base.backtest.signals
        assert (accepted.order,) == gap.backtest.orders == base.backtest.orders
        assert accepted.order.quantity == 10
        fill, = gap.backtest.fills
        assert fill.execution_session == opening.session_date
        assert fill.price == Decimal("94.235")
        assert base.backtest.fills[0].price == Decimal("96.24")
        assert tuple(day for day, _ in gap.equity_path) == (ANCHOR, opening.session_date)


def test_common_window_keeps_a_final_accepted_exit_open():
    bars = common_window_bars()[:69]
    result = compare_common_window(bars)

    run = result.mean_reversion
    order, = run.backtest.outstanding_orders
    assert result.outstanding_exit is order
    assert order.state is OrderState.ACCEPTED
    assert order.order.side is OrderSide.SELL
    assert order.order.decision_session == result.end_session
    assert order.fill is None
    assert run.backtest.sessions[-1].portfolio.positions == {"SPY": 10}
    assert run.equity_path[-1][1] == Decimal("1076.35")
    assert run.metrics.executed_orders == 1
    assert run.metrics.total_commissions == Decimal("1.25")


def test_common_window_propagates_a_rejected_exit():
    bars = common_window_bars(entry_open=90, mean_reversion_exit_open=1)

    with pytest.raises(MeanReversionExitRejected) as failure:
        compare_common_window(bars, bps=0, fee=50)

    rejection = failure.value.rejection
    assert rejection.order.side is OrderSide.SELL
    assert rejection.order.decision_session == bars[68].session_date
    assert rejection.state is OrderState.REJECTED
    assert rejection.fill is None
    assert "commission exceeds proceeds" in rejection.rejection_reason


def test_common_window_preserves_the_rejected_entry_policy():
    bars = common_window_bars()[:66]
    bars[63] = replace(bars[63], close=98, high=98)
    bars[64] = replace(bars[64], close=95, low=95)
    bars[65] = replace(bars[65], open=105, close=94, high=105, low=94)

    result = compare_common_window(bars)
    run = result.mean_reversion

    rejected, = run.backtest.rejected_orders
    assert rejected.order.side is OrderSide.BUY
    assert "insufficient cash" in rejected.rejection_reason
    assert run.backtest.signals == (SignalEvent("SPY", FIRST_FILL, 1),)
    assert run.backtest.fills == run.backtest.outstanding_orders == ()
    assert run.equity_path == result.cash.equity_path
    assert run.metrics.total_commissions == 0
    assert result.buy_and_hold.entry.fill.execution_session == FIRST_FILL


def test_common_window_is_reproducible_and_preserves_individual_benchmarks():
    bars = common_window_bars()
    original = tuple(bars)
    momentum = compare(bars)
    mean_reversion = compare_mean_reversion(bars)

    result = compare_common_window(bars)

    assert result == compare_common_window(bars)
    assert tuple(bars) == original
    assert result.momentum == momentum.momentum
    assert result.buy_and_hold == momentum.buy_and_hold
    assert result.cash == momentum.cash
    assert mean_reversion == compare_mean_reversion(bars)
    assert mean_reversion.start_session == bars[5].session_date
    first_fill = mean_reversion.mean_reversion.backtest.fills[0]
    assert first_fill.execution_session == bars[6].session_date
    with pytest.raises(FrozenInstanceError):
        setattr(result, "start_session", result.end_session)


def test_common_window_fails_when_its_benchmark_entry_is_rejected():
    bars = common_window_bars(entry_open=105)[:66]

    with pytest.raises(BuyAndHoldEntryError, match="insufficient cash") as failure:
        compare_common_window(bars)

    assert failure.value.anchor == ANCHOR
    assert failure.value.status == "rejected"
    assert failure.value.entry.state is OrderState.REJECTED
    assert failure.value.entry.fill is None


def test_benchmark_with_zero_whole_shares_fails_without_an_order():
    bars = spy_bars([1200] * 6 + [900])

    with pytest.raises(BuyAndHoldEntryError, match="whole-share sizing produced zero shares") as failure:
        compare_mean_reversion(bars, bps=0, fee=0)

    error = failure.value
    assert (error.anchor, error.status) == (bars[5].session_date, "zero_shares")
    assert error.entry is None


def test_common_window_requires_64_observations():
    with pytest.raises(ValueError, match="requires 64 SPY closes"):
        compare_common_window(common_window_bars()[:63])
