from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, date, datetime
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
from trading_research.execution import NextOpenExecution
from trading_research.orders import (
    MarketOrder,
    OrderLedger,
    OrderSide,
    OrderState,
)
from trading_research.portfolio import PortfolioLedger
from trading_research.replay import MarketOpen, replay_daily_bars
from trading_research.signals import SignalEvent

FIRST = date(2025, 1, 3)
SECOND = date(2025, 1, 6)
THIRD = date(2025, 1, 7)


def dataset(symbol, *observations, adjustment=PriceAdjustment.RAW):
    metadata = DatasetMetadata(
        "Synthetic", symbol, adjustment, datetime(2025, 1, 8, tzinfo=UTC)
    )
    bars = tuple(
        DailyBar(
            symbol, session, opening, max(opening, close),
            min(opening, close), close, 1_000,
        )
        for session, opening, close in observations
    )
    return metadata, bars


def run(datasets, strategy, *, cash=1000, bps=0, fee=0):
    return run_backtest(
        datasets, strategy, initial_cash=cash,
        adjustment_bps=bps, commission=fee,
    )


def test_no_trades_preserve_initial_cash_and_record_immutable_sessions():
    spy = dataset("SPY", (FIRST, 100, 101), (SECOND, 102, 105))

    result = run([spy], lambda context: ())

    valuations = [(value.session_date, value.equity) for value in result.sessions]
    assert valuations == [
        (FIRST, Decimal("1000")), (SECOND, Decimal("1000"))
    ]
    for value in result.sessions:
        assert value.portfolio.cash == Decimal("1000")
        assert value.portfolio.positions == {}
        assert value.portfolio.cumulative_commissions == 0
    assert result.orders == result.fills == result.signals == ()
    assert result.order_history == result.transactions == ()
    assert result.outstanding_orders == result.rejected_orders == ()
    with pytest.raises(FrozenInstanceError):
        setattr(result.sessions[0], "equity", Decimal(0))
    with pytest.raises(FrozenInstanceError):
        setattr(result, "sessions", ())
    with pytest.raises(TypeError):
        result.sessions[0].portfolio.positions["SPY"] = 10


def test_round_trip_preserves_both_opening_gaps_and_charges_each_fee_once():
    spy = dataset("SPY", (FIRST, 95, 100), (SECOND, 110, 105), (THIRD, 90, 92))

    def strategy(context):
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.4),)
        if context.session_date == SECOND:
            return (SignalEvent("SPY", SECOND, 0),)
        return ()

    result = run([spy], strategy, bps=25, fee=Decimal("1.25"))

    assert [
        (order.side, order.quantity, order.decision_session)
        for order in result.orders
    ] == [
        (OrderSide.BUY, 4, FIRST), (OrderSide.SELL, 4, SECOND)
    ]
    assert all(
        order.earliest_execution_session is None for order in result.orders
    )
    assert [(fill.execution_session, fill.price) for fill in result.fills] == [
        (SECOND, Decimal("110.275")), (THIRD, Decimal("89.775"))
    ]
    assert [entry.state for entry in result.order_history] == [
        OrderState.CREATED, OrderState.ACCEPTED, OrderState.FILLED,
        OrderState.CREATED, OrderState.ACCEPTED, OrderState.FILLED,
    ]
    booked = tuple(entry.fill for entry in result.order_history if entry.fill)
    assert booked == result.fills
    assert [entry.commission for entry in result.transactions] == [
        Decimal("1.25"), Decimal("1.25")
    ]
    before, bought, sold = result.sessions
    assert before.portfolio.positions == {}
    assert before.portfolio.cash == before.equity == Decimal("1000")
    assert bought.portfolio.positions == {"SPY": 4}
    assert bought.portfolio.cash == Decimal("557.65")
    assert bought.portfolio.cumulative_commissions == Decimal("1.25")
    assert bought.equity == Decimal("977.65")
    assert sold.portfolio.positions == {}
    assert sold.portfolio.cash == sold.equity == Decimal("915.50")
    assert sold.portfolio.cumulative_commissions == Decimal("2.50")
    assert [entry.cash_after for entry in result.transactions] == [
        Decimal("557.65"), Decimal("915.50")
    ]
    assert result.rejected_orders == result.outstanding_orders == ()


def test_competing_buys_use_symbol_order_and_never_retry_a_rejection():
    spy = dataset("SPY", (FIRST, 50, 50), (SECOND, 60, 55), (THIRD, 50, 60))
    qqq = dataset("QQQ", (FIRST, 50, 50), (SECOND, 60, 55), (THIRD, 50, 60))

    def simulate(datasets):
        contexts = []

        def strategy(context):
            contexts.append(context)
            symbols = [event.bar.symbol for event in context.events]
            assert symbols == ["QQQ", "SPY"]
            if context.session_date == FIRST:
                return (
                    SignalEvent("SPY", FIRST, 0.5),
                    SignalEvent("QQQ", FIRST, 0.5),
                )
            return ()

        result = run(datasets, strategy, cash=100, fee=Decimal("0.50"))
        dates = [context.session_date for context in contexts]
        assert dates == [FIRST, SECOND, THIRD]
        for context in contexts:
            assert all(
                event.bar.session_date <= context.session_date
                for event in context.history("SPY")
            )
        first_history = contexts[0].history("SPY")
        assert [event.bar.session_date for event in first_history] == [FIRST]
        return result

    result = simulate([spy, qqq])
    assert result == simulate([qqq, spy]) == simulate([spy, qqq])
    assert [signal.symbol for signal in result.signals] == ["QQQ", "SPY"]
    assert [order.symbol for order in result.orders] == ["QQQ", "SPY"]
    fill, = result.fills
    assert fill.symbol == "QQQ"
    assert fill.quantity == 1
    assert fill.execution_session == SECOND
    rejected, = result.rejected_orders
    assert rejected.order.symbol == "SPY"
    assert rejected.state is OrderState.REJECTED
    assert rejected.fill is None
    assert "insufficient cash" in rejected.rejection_reason
    assert result.sessions[1].portfolio.cash == Decimal("39.50")
    assert result.sessions[-1].portfolio.positions == {"QQQ": 1}
    assert result.sessions[-1].equity == Decimal("99.50")
    assert result.outstanding_orders == ()


def test_sell_executes_before_a_buy_that_depends_on_its_proceeds():
    spy = dataset("SPY", (FIRST, 50, 50), (SECOND, 50, 50), (THIRD, 50, 50))
    qqq = dataset("QQQ", (FIRST, 25, 25), (SECOND, 25, 25), (THIRD, 25, 26))

    def strategy(context):
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 1),)
        if context.session_date == SECOND:
            return (
                SignalEvent("QQQ", SECOND, 1), SignalEvent("SPY", SECOND, 0)
            )
        return ()

    result = run([spy, qqq], strategy)

    assert [
        (fill.symbol, fill.side, fill.quantity) for fill in result.fills
    ] == [
        ("SPY", OrderSide.BUY, 20),
        ("SPY", OrderSide.SELL, 20),
        ("QQQ", OrderSide.BUY, 40),
    ]
    assert result.sessions[1].portfolio.cash == Decimal(0)
    assert result.sessions[-1].portfolio.cash == Decimal(0)
    assert result.sessions[-1].portfolio.positions == {"QQQ": 40}
    assert result.sessions[-1].equity == Decimal("1040")
    assert result.rejected_orders == ()


def test_partial_signals_retain_weights_and_empty_signals_skip_planning():
    spy = dataset("SPY", (FIRST, 50, 50), (SECOND, 50, 50), (THIRD, 50, 50))
    qqq = dataset("QQQ", (FIRST, 25, 25), (SECOND, 25, 25), (THIRD, 25, 50))

    def strategy(context):
        if context.session_date == FIRST:
            return (
                SignalEvent("SPY", FIRST, 0.5), SignalEvent("QQQ", FIRST, 0.25)
            )
        if context.session_date == SECOND:
            return (SignalEvent("QQQ", SECOND, 0.5),)
        return ()

    result = run([spy, qqq], strategy)

    planned = [
        (order.symbol, order.quantity, order.decision_session)
        for order in result.orders
    ]
    assert planned == [
        ("QQQ", 10, FIRST), ("SPY", 10, FIRST), ("QQQ", 10, SECOND)
    ]
    emitted = [
        (signal.symbol, signal.decision_session) for signal in result.signals
    ]
    assert emitted == [
        ("QQQ", FIRST), ("SPY", FIRST), ("QQQ", SECOND)
    ]
    assert result.sessions[-1].portfolio.positions == {"SPY": 10, "QQQ": 20}
    assert result.sessions[-1].equity == Decimal("1500")
    assert result.outstanding_orders == ()


@pytest.mark.parametrize("returns", [False, True])
def test_missing_symbol_open_keeps_the_order_pending(returns):
    observations = [(FIRST, 25, 25)]
    if returns:
        observations.append((THIRD, 26, 27))
    spy = dataset("SPY", *observations)
    qqq = dataset("QQQ", (SECOND, 40, 41))

    def strategy(context):
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.5),)
        return ()

    result = run([spy, qqq], strategy, cash=100)

    assert result.sessions[1].session_date == SECOND
    assert result.sessions[1].portfolio.cash == Decimal("100")
    assert result.sessions[1].portfolio.positions == {}
    if returns:
        fill, = result.fills
        assert fill.execution_session == THIRD
        assert fill.price == Decimal("26")
        assert result.sessions[-1].portfolio.cash == Decimal("48")
        assert result.sessions[-1].equity == Decimal("102")
        assert result.outstanding_orders == ()
    else:
        pending, = result.outstanding_orders
        assert pending.state is OrderState.ACCEPTED
        assert pending.order.symbol == "SPY"
        assert pending.order.quantity == 2
        assert pending.fill is None
        assert result.fills == ()


def test_newer_flat_target_cancels_the_audited_pending_buy():
    first, second, third = (date(2024, 1, day) for day in (2, 3, 4))
    spy = dataset("SPY", (first, 100, 100), (third, 100, 100))
    qqq = dataset("QQQ", (second, 100, 100))

    def strategy(context):
        if context.session_date == first:
            return (SignalEvent("SPY", first, 1),)
        if context.session_date == second:
            return (SignalEvent("SPY", second, 0),)
        return ()

    result = run([spy, qqq], strategy)
    assert result.signals == (SignalEvent("SPY", first, 1), SignalEvent("SPY", second, 0))
    order_id = "2024-01-02:SPY"
    actual = (
        [(entry.order.order_id, entry.state.name) for entry in result.order_history],
        [(fill.order_id, fill.execution_session, fill.quantity) for fill in result.fills],
        dict(result.sessions[-1].portfolio.positions),
        result.sessions[-1].portfolio.cash,
    )
    assert actual == (
        [(order_id, "CREATED"), (order_id, "ACCEPTED"), (order_id, "CANCELLED")],
        [],
        {},
        Decimal("1000"),
    ), actual
    cancellation = result.order_history[-1]
    assert cancellation.cancellation_session == second
    assert cancellation.fill is cancellation.rejection_reason is None
    assert result.outstanding_orders == result.rejected_orders == result.transactions == ()
    assert all(value.portfolio == result.sessions[0].portfolio for value in result.sessions)
    assert run([spy, qqq], strategy) == result


def test_smaller_target_cancels_and_waits_for_a_current_close_to_resize():
    fourth, fifth = date(2025, 1, 8), date(2025, 1, 9)
    spy = dataset("SPY", (FIRST, 100, 100), (fourth, 120, 80), (fifth, 90, 90))
    qqq = dataset("QQQ", (SECOND, 100, 100), (THIRD, 100, 100))
    feedback = {}

    def strategy(context):
        session = context.session_date
        feedback[session] = context.execution
        if session == FIRST:
            return (SignalEvent("SPY", session, 1),)
        if session == SECOND:
            return (SignalEvent("SPY", session, 0.4),)
        if session in (THIRD, fourth):
            return (SignalEvent("QQQ", session, 0),)
        return ()

    result = run([spy, qqq], strategy, fee=1)

    initial, replacement = result.orders
    assert (initial.order_id, initial.quantity) == ("2025-01-03:SPY", 10)
    assert (replacement.order_id, replacement.quantity) == ("2025-01-08:SPY", 5)
    _, _, cancelled, *_ = result.order_history
    assert cancelled.order is initial
    assert cancelled.state is OrderState.CANCELLED
    assert cancelled.cancellation_session == SECOND
    assert feedback[SECOND].pending_orders == (initial,)
    assert feedback[THIRD].pending_orders == feedback[THIRD].outcomes == ()
    assert feedback[fourth].portfolio.cash == 1000
    assert feedback[fourth].portfolio.positions == {}
    assert feedback[fourth].outcomes == ()
    fill, = result.fills
    assert (fill.order_id, fill.execution_session, fill.quantity) == (
        replacement.order_id, fifth, 5,
    )
    assert result.sessions[-1].portfolio.positions == {"SPY": 5}
    assert result.sessions[-1].portfolio.cash == 549
    assert result.sessions[-1].portfolio.cumulative_commissions == 1


def test_reaffirmed_target_keeps_the_original_accepted_order():
    spy = dataset("SPY", (FIRST, 100, 100), (THIRD, 99, 102))
    qqq = dataset("QQQ", (SECOND, 100, 100))

    def strategy(context):
        if context.session_date in (FIRST, SECOND):
            return (SignalEvent("SPY", context.session_date, 0.5),)
        return ()

    result = run([spy, qqq], strategy)

    order, = result.orders
    assert (order.order_id, order.quantity) == ("2025-01-03:SPY", 5)
    assert [entry.state for entry in result.order_history] == [
        OrderState.CREATED, OrderState.ACCEPTED, OrderState.FILLED,
    ]
    fill, = result.fills
    assert (fill.order_id, fill.execution_session) == (order.order_id, THIRD)
    assert result.sessions[-1].portfolio.positions == {"SPY": 5}


def test_flat_target_at_close_preserves_buy_filled_at_that_open():
    spy = dataset("SPY", (FIRST, 100, 100), (SECOND, 95, 100))

    def strategy(context):
        target = 0.5 if context.session_date == FIRST else 0
        return (SignalEvent("SPY", context.session_date, target),)

    result = run([spy], strategy, fee=1)

    buy, sell = result.orders
    assert (buy.side, sell.side) == (OrderSide.BUY, OrderSide.SELL)
    assert [entry.state for entry in result.order_history] == [
        OrderState.CREATED, OrderState.ACCEPTED, OrderState.FILLED,
        OrderState.CREATED, OrderState.ACCEPTED,
    ]
    fill, = result.fills
    assert fill.order_id == buy.order_id and fill.execution_session == SECOND
    pending, = result.outstanding_orders
    assert pending.order == sell
    assert sell.decision_session == SECOND and sell.quantity == 5
    assert result.sessions[-1].portfolio.positions == {"SPY": 5}
    assert result.sessions[-1].portfolio.cash == 524
    assert result.sessions[-1].portfolio.cumulative_commissions == 1


def test_held_symbol_without_current_close_fails_without_a_stale_mark():
    spy = dataset("SPY", (FIRST, 25, 25), (SECOND, 25, 26))
    qqq = dataset("QQQ", (THIRD, 30, 30))
    evaluated = []

    def strategy(context):
        evaluated.append(context.session_date)
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.5),)
        return ()

    with pytest.raises(ValueError, match="missing valuation price for 'SPY'"):
        run([spy, qqq], strategy, cash=100)

    assert evaluated == [FIRST, SECOND, THIRD]


def test_opening_execution_precedes_strategy_callback(monkeypatch):
    spy = dataset("SPY", (FIRST, 9, 10), (SECOND, 11, 12))
    trace = []
    propose = NextOpenExecution.propose_at_open

    def track_open(execution, order, opening):
        assert [field.name for field in fields(opening)] == [
            "symbol", "session_date", "price", "adjustment"
        ]
        trace.append(("open", opening.session_date, opening.price))
        return propose(execution, order, opening)

    def strategy(context):
        event, = context.events
        trace.append(("close", context.session_date, event.bar.close))
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.5),)
        return ()

    monkeypatch.setattr(NextOpenExecution, "propose_at_open", track_open)
    run([spy], strategy, cash=100)

    assert trace == [
        ("close", FIRST, 10), ("open", SECOND, 11), ("close", SECOND, 12)
    ]


def test_replay_delivers_sorted_opens_before_sorted_completed_bars():
    spy = dataset("SPY", (FIRST, 10, 12))
    qqq = dataset("QQQ", (FIRST, 20, 22))
    trace = []

    def opening(session, observations):
        trace.append((session, observations))

    replay_daily_bars(
        [spy, qqq], lambda event: trace.append(event.bar.symbol),
        on_session_open=opening, on_session_end=trace.append,
    )

    assert trace == [
        (FIRST, (
            MarketOpen("QQQ", FIRST, 20, PriceAdjustment.RAW),
            MarketOpen("SPY", FIRST, 10, PriceAdjustment.RAW),
        )),
        "QQQ", "SPY", FIRST,
    ]


def test_strategy_cannot_target_an_instrument_observed_only_in_future():
    spy = dataset("SPY", (FIRST, 10, 10))
    qqq = dataset("QQQ", (SECOND, 20, 20))
    calls = []

    def strategy(context):
        calls.append(context.session_date)
        assert context.history("QQQ") == ()
        return (SignalEvent("QQQ", FIRST, 1),)

    with pytest.raises(ValueError, match="unknown instrument 'QQQ'"):
        run([spy, qqq], strategy)

    assert calls == [FIRST]


def test_rejects_nonraw_backtest_inputs_even_without_trades():
    spy = dataset(
        "SPY", (FIRST, 10, 10), adjustment=PriceAdjustment.SPLIT_ADJUSTED
    )
    with pytest.raises(ValueError, match="backtest requires RAW"):
        run([spy], pytest.fail)


def test_oversized_sell_is_rejected_for_insufficient_holdings(monkeypatch):
    spy = dataset("SPY", (FIRST, 50, 50), (SECOND, 50, 50), (THIRD, 50, 50))
    plan = backtest.plan_orders

    def oversize_sell(*args, **kwargs):
        orders = plan(*args, **kwargs)
        return tuple(
            replace(order, quantity=order.quantity + 1)
            if order.side is OrderSide.SELL else order
            for order in orders
        )

    def strategy(context):
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.5),)
        if context.session_date == SECOND:
            return (SignalEvent("SPY", SECOND, 0),)
        return ()

    monkeypatch.setattr(backtest, "plan_orders", oversize_sell)
    result = run([spy], strategy, cash=100)

    rejected, = result.rejected_orders
    assert rejected.state is OrderState.REJECTED
    assert rejected.order.quantity == 2
    assert "insufficient holdings" in rejected.rejection_reason
    assert result.sessions[-1].portfolio == result.sessions[1].portfolio
    assert result.sessions[-1].portfolio.positions == {"SPY": 1}
    assert result.sessions[-1].equity == Decimal("100")
    assert len(result.transactions) == 1
    assert result.outstanding_orders == ()


def test_booking_error_preserves_both_ledger_histories(monkeypatch):
    spy = dataset("SPY", (FIRST, 20, 20), (SECOND, 20, 20))
    apply = PortfolioLedger.apply_fill
    commit = ExecutionCoordinator.execute
    seen = []

    def fail_after_booking(ledger, fill, commission):
        apply(ledger, fill, commission)
        assert ledger.snapshot().cash == Decimal("39.5")
        raise RuntimeError("injected after booking on the portfolio copy")

    def audit_commit(coordinator, order_id, proposal):
        def state():
            return (
                coordinator.order_snapshot(order_id),
                coordinator.order_history(),
                coordinator.portfolio_snapshot(),
                coordinator.portfolio_history(),
            )
        before = state()
        try:
            return commit(coordinator, order_id, proposal)
        finally:
            seen.append((before, state()))

    def strategy(context):
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.6),)
        return ()

    monkeypatch.setattr(PortfolioLedger, "apply_fill", fail_after_booking)
    monkeypatch.setattr(ExecutionCoordinator, "execute", audit_commit)
    with pytest.raises(RuntimeError, match="injected after booking"):
        run([spy], strategy, cash=100, fee=Decimal("0.50"))

    (before, after), = seen
    assert after == before
    assert after[0].state is OrderState.ACCEPTED
    assert after[2].cash == Decimal("100")
    assert after[2].positions == {}
    assert after[2].cumulative_commissions == 0
    assert after[3] == ()


def test_submitting_duplicate_orders_preserves_existing_coordinator_state():
    coordinator = ExecutionCoordinator(OrderLedger(), PortfolioLedger(100))
    first = MarketOrder("first", "SPY", OrderSide.BUY, 1, FIRST)
    second = replace(first, order_id="second")
    coordinator.submit_orders((first,))
    history = coordinator.order_history()

    with pytest.raises(ValueError, match="duplicate order identifier 'first'"):
        coordinator.submit_orders((second, first))

    assert coordinator.order_history() == history
    assert coordinator.order_snapshot("first").state is OrderState.ACCEPTED
    assert coordinator.portfolio_snapshot().cash == Decimal("100")
    assert coordinator.portfolio_history() == ()


def test_opening_only_pricing_matches_the_standalone_open_model():
    metadata, bars = dataset("SPY", (SECOND, 100, 104), (THIRD, 110, 108))
    order = MarketOrder(
        "delayed", "SPY", OrderSide.BUY, 2, FIRST,
        earliest_execution_session=THIRD,
    )
    orders = OrderLedger()
    orders.register(order)
    orders.accept(order.order_id)
    accepted = orders.snapshot(order.order_id)
    opening = NextOpenExecution(adjustment_bps=25, commission=1)
    standalone = NextOpenExecution(adjustment_bps=25, commission=1)

    before = MarketOpen("SPY", SECOND, 100, PriceAdjustment.RAW)
    eligible = MarketOpen("SPY", THIRD, 110, PriceAdjustment.RAW)
    assert opening.propose_at_open(accepted, before) is None

    proposal = opening.propose_at_open(accepted, eligible)
    assert proposal == standalone.propose(accepted, bars, metadata)
    assert proposal.fill.execution_session == THIRD
    assert proposal.fill.price == Decimal("110.275")
