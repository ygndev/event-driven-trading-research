from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from trading_research import backtest
from trading_research.backtest import run_backtest
from trading_research.coordinator import ExecutionCoordinator
from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.mean_reversion import (
    MeanReversionExitRejected,
    short_term_mean_reversion,
)
from trading_research.orders import OrderSide, OrderState
from trading_research.signals import SignalEvent, generate_signals


def day(observation):
    return date(2025, 2, 3) + timedelta(days=observation - 1)


def dataset(closes, *, openings=None, dates=None, symbol="SPY"):
    openings = openings or {}
    if dates is None:
        dates = (day(index) for index in range(1, len(closes) + 1))
    metadata = DatasetMetadata(
        "Synthetic", symbol, PriceAdjustment.RAW,
        datetime(2025, 4, 1, tzinfo=UTC),
    )
    bars = []
    for index, (session, close) in enumerate(
        zip(dates, closes, strict=True), start=1
    ):
        opening = openings.get(index, close)
        bars.append(
            DailyBar(
                symbol, session, opening, max(opening, close),
                min(opening, close), close, 1000,
            )
        )
    return metadata, tuple(bars)


def run(
    *datasets, strategy=short_term_mean_reversion, bps=25, fee=Decimal("1.25")
):
    return run_backtest(
        datasets, strategy, initial_cash=1000,
        adjustment_bps=bps, commission=fee,
    )


def test_no_entry_during_warmup():
    result = run(dataset([100, 100, 100, 100, 70]))

    assert result.signals == result.orders == result.fills == ()


@pytest.mark.parametrize(
    ("sixth_close", "enters"),
    [(97.0001, False), (97, True), (96.9999, True)],
)
def test_first_entry_at_the_exact_five_interval_threshold(sixth_close, enters):
    # Observation 1 is the reference. Using observation 2 would miss the entry.
    result = run(dataset([100, 90, 90, 90, 90, sixth_close]))

    if enters:
        assert result.signals == (SignalEvent("SPY", day(6), 1),)
        pending, = result.outstanding_orders
        assert pending.state is OrderState.ACCEPTED
        assert pending.order.quantity == 10
    else:
        assert result.signals == result.orders == ()
    assert result.fills == ()


def test_exact_threshold_does_not_lose_a_one_unit_price_difference():
    initial = 10**18
    result = run(dataset([initial] * 5 + [97 * 10**16 + 1]))

    assert result.signals == ()


def test_round_trip_counts_observed_closes_from_the_committed_buy():
    dates = [day(index) for index in range(1, 7)] + [
        date(2025, 2, 11), date(2025, 2, 12), date(2025, 2, 14),
        date(2025, 2, 15), date(2025, 2, 18), date(2025, 2, 21),
    ]
    spy = dataset(
        [100] * 5 + [97, 96, 95, 94, 95, 98, 101],
        dates=dates, openings={7: 96, 12: 101},
    )
    no_spy = date(2025, 2, 9)
    qqq = dataset([50], dates=(no_spy,), symbol="QQQ")
    contexts = {}

    def strategy(context):
        contexts[context.session_date] = context
        return short_term_mean_reversion(context)

    result = run(spy, qqq, strategy=strategy)
    decision, entry = dates[5:7]
    exit_decision, sale = dates[10:12]
    assert result.signals == (
        SignalEvent("SPY", decision, 1),
        SignalEvent("SPY", exit_decision, 0),
    )
    assert [(order.side, order.quantity) for order in result.orders] == [
        (OrderSide.BUY, 10), (OrderSide.SELL, 10)
    ]
    bought, sold = result.fills
    assert (bought.execution_session, bought.price) == (
        entry, Decimal("96.24")
    )
    assert (sold.execution_session, sold.price) == (
        sale, Decimal("100.7475")
    )

    pending = contexts[no_spy]
    assert pending.execution.pending_orders == (result.orders[0],)
    assert pending.execution.portfolio.positions == {}
    assert pending.last_order_outcome("SPY") is None
    assert short_term_mean_reversion(pending) == ()
    assert contexts[entry].execution.outcomes[0].fill is bought
    assert contexts[exit_decision].last_order_outcome("SPY").fill is bought
    assert contexts[sale].last_order_outcome("SPY").fill is sold

    by_date = {snapshot.session_date: snapshot for snapshot in result.sessions}
    assert by_date[entry].portfolio.cash == Decimal("36.35")
    assert by_date[entry].equity == Decimal("996.35")
    assert by_date[exit_decision].portfolio.positions == {"SPY": 10}
    assert by_date[exit_decision].equity == Decimal("1016.35")
    assert by_date[sale].portfolio.positions == {}
    assert by_date[sale].portfolio.cash == by_date[sale].equity == Decimal(
        "1042.575"
    )
    assert by_date[sale].portfolio.cumulative_commissions == Decimal("2.50")
    assert [entry.commission for entry in result.transactions] == [
        Decimal("1.25"), Decimal("1.25")
    ]
    assert result.outstanding_orders == result.rejected_orders == ()
    assert run(spy, qqq) == result


def test_pending_buys_and_sells_do_not_repeat_signals(monkeypatch):
    plan = backtest.plan_orders

    def delay_orders(*args):
        return tuple(
            replace(
                order,
                earliest_execution_session=(
                    day(9) if order.side is OrderSide.BUY else day(15)
                ),
            )
            for order in plan(*args)
        )

    monkeypatch.setattr(backtest, "plan_orders", delay_orders)
    spy = dataset([100] * 5 + [97, 96, 96, 96, 96, 95, 95, 94, 93, 110])
    contexts = {}

    def strategy(context):
        contexts[context.session_date] = context
        return short_term_mean_reversion(context)

    result = run(spy, strategy=strategy)

    assert result.signals == (
        SignalEvent("SPY", day(6), 1), SignalEvent("SPY", day(13), 0)
    )
    fill_sessions = [fill.execution_session for fill in result.fills]
    assert fill_sessions == [day(9), day(15)]
    for index in (7, 8):
        feedback = contexts[day(index)].execution
        assert feedback.pending_orders == (result.orders[0],)
        assert feedback.portfolio.positions == {}
    feedback = contexts[day(14)].execution
    assert feedback.pending_orders == (result.orders[1],)
    assert feedback.portfolio.positions == {"SPY": 10}
    assert result.sessions[-1].portfolio.positions == {}


def test_rejected_entry_does_not_start_a_clock_or_retry_the_same_decline():
    spy = dataset(
        [100] * 5 + [97, 96, 95, 94, 93, 92, 91, 90, 89],
        openings={7: 105},
    )

    result = run(spy)

    assert result.signals == (SignalEvent("SPY", day(6), 1),)
    rejected, = result.rejected_orders
    assert rejected.order.side is OrderSide.BUY
    assert "insufficient cash" in rejected.rejection_reason
    assert rejected.fill is None
    assert result.fills == result.outstanding_orders == ()
    assert all(
        snapshot.equity == Decimal("1000")
        and snapshot.portfolio.positions == {}
        and snapshot.portfolio.cumulative_commissions == 0
        for snapshot in result.sessions
    )


def test_rejected_entry_rearms_after_recovery_and_a_fresh_crossing():
    spy = dataset(
        [100] * 5 + [97, 96, 95, 94, 93, 92, 94, 90, 88],
        openings={7: 105},
    )

    result = run(spy)

    assert result.signals == (
        SignalEvent("SPY", day(6), 1), SignalEvent("SPY", day(13), 1)
    )
    assert len(result.rejected_orders) == 1
    fill, = result.fills
    assert fill.execution_session == day(14)
    assert fill.quantity == 11
    assert fill.price == Decimal("88.22")
    assert result.sessions[-1].portfolio.positions == {"SPY": 11}


def test_rejected_exit_aborts_with_rejection_and_accounting_intact(monkeypatch):
    spy = dataset([100] * 5 + [97, 90, 80, 50, 30, 5, 1, 2, 3])
    execute = ExecutionCoordinator.execute
    attempts = []
    contexts = []

    def record_execution(coordinator, order_id, proposal):
        outcome = execute(coordinator, order_id, proposal)
        attempts.append((coordinator, outcome))
        return outcome

    def strategy(context):
        contexts.append(context)
        return short_term_mean_reversion(context)

    monkeypatch.setattr(ExecutionCoordinator, "execute", record_execution)
    with pytest.raises(MeanReversionExitRejected) as failure:
        run(spy, strategy=strategy, bps=0, fee=50)

    (_, filled), (coordinator, rejected) = attempts
    assert filled.state is OrderState.FILLED
    assert rejected.state is OrderState.REJECTED
    assert rejected.order.side is OrderSide.SELL
    assert rejected.order.quantity == 10
    assert rejected.order.decision_session == day(11)
    assert rejected.order.order_id == "2025-02-13:SPY"
    assert rejected.fill is None
    assert "commission exceeds proceeds" in rejected.rejection_reason

    error = failure.value
    assert error.rejection is rejected
    for detail in ("SPY", "2025-02-13", rejected.order.order_id,
                   rejected.rejection_reason, "left the position open"):
        assert detail in str(error)

    assert [context.session_date for context in contexts] == [
        day(index) for index in range(1, 13)
    ]
    assert contexts[-2].last_order_outcome("SPY") is filled
    assert contexts[-1].execution.outcomes == (rejected,)
    assert coordinator.order_snapshot(rejected.order.order_id) is rejected
    assert coordinator.order_history()[-1] is rejected

    trade, = coordinator.portfolio_history()
    assert trade.fill is filled.fill
    assert (trade.fill.side, trade.fill.quantity, trade.fill.price) == (
        OrderSide.BUY, 10, Decimal("90")
    )
    assert trade.fill.execution_session == day(7)
    assert trade.commission == trade.cash_after == Decimal("50")
    portfolio = coordinator.portfolio_snapshot()
    assert portfolio == contexts[-2].execution.portfolio
    assert portfolio.positions == {"SPY": 10}
    assert portfolio.cash == Decimal("50")
    assert portfolio.cumulative_commissions == Decimal("50")


def test_exit_pending_at_dataset_end_remains_outstanding():
    spy = dataset([100] * 5 + [97, 90, 80, 50, 30, 5])

    result = run(spy, bps=0, fee=50)

    assert result.signals == (
        SignalEvent("SPY", day(6), 1), SignalEvent("SPY", day(11), 0)
    )
    pending, = result.outstanding_orders
    assert pending.state is OrderState.ACCEPTED
    assert pending.order.side is OrderSide.SELL
    assert pending.order.decision_session == day(11)
    assert pending.fill is None
    assert result.rejected_orders == ()
    trade, = result.transactions
    assert trade.fill.side is OrderSide.BUY
    assert trade.commission == Decimal("50")
    final = result.sessions[-1].portfolio
    assert final.positions == {"SPY": 10}
    assert final.cash == Decimal("50")
    assert final.cumulative_commissions == Decimal("50")


def test_rejected_exit_failure_is_deterministic():
    spy = dataset([100] * 5 + [97, 90, 80, 50, 30, 5, 1])

    with pytest.raises(MeanReversionExitRejected) as first:
        run(spy, bps=0, fee=50)
    with pytest.raises(MeanReversionExitRejected) as second:
        run(spy, bps=0, fee=50)

    assert first.value.rejection == second.value.rejection
    assert str(first.value) == str(second.value)


def test_strategy_feedback_is_immutable_and_cannot_see_the_next_open():
    snapshots = []

    def strategy(context):
        if context.session_date == day(6):
            snapshots.append(context)
        return short_term_mean_reversion(context)

    closes = [100] * 5 + [97, 96]
    affordable = run(dataset(closes), strategy=strategy)
    rejected = run(dataset(closes, openings={7: 105}), strategy=strategy)

    assert affordable.signals == rejected.signals == (
        SignalEvent("SPY", day(6), 1),
    )
    assert len(affordable.fills) == len(rejected.rejected_orders) == 1
    for context in snapshots:
        assert context.last_order_outcome("SPY") is None
        assert context.execution.outcomes == ()
        assert context.execution.pending_orders == ()
        assert context.execution.portfolio.positions == {}
        assert short_term_mean_reversion(context) == affordable.signals
        with pytest.raises(FrozenInstanceError):
            setattr(context.execution, "outcomes", rejected.rejected_orders)
        with pytest.raises(TypeError):
            context.execution.portfolio.positions["SPY"] = 99


def test_market_only_strategy_interface_has_no_execution_feedback():
    spy = dataset([100] * 5 + [97])
    contexts = []

    def strategy(context):
        contexts.append(context)
        return ()

    assert generate_signals([spy], strategy) == ()
    assert contexts[-1].execution is None
    with pytest.raises(
        ValueError, match="requires backtest execution feedback"
    ):
        short_term_mean_reversion(contexts[-1])


def test_a_completed_exit_allows_a_later_entry_from_flat():
    spy = dataset(
        [100] * 5 + [97, 96, 95, 94, 95, 98, 101, 92, 90]
    )

    result = run(spy)

    assert result.signals == (
        SignalEvent("SPY", day(6), 1),
        SignalEvent("SPY", day(11), 0),
        SignalEvent("SPY", day(13), 1),
    )
    assert result.sessions[11].portfolio.positions == {}
    assert result.fills[-1].execution_session == day(14)
    assert result.fills[-1].side is OrderSide.BUY
    assert result.fills[-1].quantity == 11
    assert result.sessions[-1].portfolio.positions == {"SPY": 11}
