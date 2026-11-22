from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
from operator import setitem

import pytest

from trading_research.orders import (
    Fill,
    MarketOrder,
    OrderLedger,
    OrderSide,
    OrderState,
)

DECISION = date(2025, 1, 3)
NEXT = date(2025, 1, 6)
LATER = date(2025, 1, 8)


@pytest.fixture
def order():
    return MarketOrder("order-1", "SPY", OrderSide.BUY, 12, DECISION)


def fill_for(order, session=NEXT):
    return Fill(
        order_id=order.order_id,
        symbol=order.symbol,
        execution_session=session,
        side=order.side,
        quantity=order.quantity,
        price=101.25,
    )


def accepted_ledger(order):
    ledger = OrderLedger()
    ledger.register(order)
    ledger.accept(order.order_id)
    return ledger


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_records_full_buy_and_sell_fills(order, side):
    order = replace(order, side=side)
    ledger = accepted_ledger(order)
    fill = fill_for(order)
    ledger.record_fill(fill)

    created, accepted, filled = ledger.history()
    assert (created.state, accepted.state, filled.state) == (
        OrderState.CREATED,
        OrderState.ACCEPTED,
        OrderState.FILLED,
    )
    assert created.fill is accepted.fill is None
    assert filled.fill is fill
    assert filled.fill.execution_session == NEXT
    assert filled.fill.price == 101.25
    assert filled.fill.quantity == 12
    assert filled.fill.side is side
    assert filled.order is order
    assert filled.rejection_reason is None
    assert ledger.snapshot(order.order_id) is filled


@pytest.mark.parametrize("accept", [False, True])
def test_rejection_retains_the_explicit_reason(order, accept):
    ledger = OrderLedger()
    ledger.register(order)
    expected = [OrderState.CREATED]
    if accept:
        ledger.accept(order.order_id)
        expected.append(OrderState.ACCEPTED)

    ledger.reject(order.order_id, "Eligibility review failed")

    assert [entry.state for entry in ledger.history()] == [
        *expected, OrderState.REJECTED
    ]
    rejected = ledger.snapshot(order.order_id)
    assert rejected.rejection_reason == "Eligibility review failed"
    assert rejected.fill is None


@pytest.mark.parametrize("state", [OrderState.FILLED, OrderState.REJECTED])
@pytest.mark.parametrize("target", ["ACCEPTED", "REJECTED", "FILLED", "CANCELLED"])
def test_terminal_states_reject_further_transitions(order, state, target):
    ledger = accepted_ledger(order)
    fill = fill_for(order)
    if state is OrderState.FILLED:
        ledger.record_fill(fill)
    else:
        ledger.reject(order.order_id, "Eligibility review failed")

    actions = {
        "ACCEPTED": lambda: ledger.accept(order.order_id),
        "REJECTED": lambda: ledger.reject(order.order_id, "Second decision"),
        "FILLED": lambda: ledger.record_fill(fill),
        "CANCELLED": lambda: ledger.cancel(order.order_id, session=LATER),
    }
    history = ledger.history()
    with pytest.raises(ValueError, match=f"from {state.name} to {target}"):
        actions[target]()

    assert ledger.history() == history
    assert ledger.snapshot(order.order_id) is history[-1]


def test_cancellation_only_applies_once_to_an_accepted_order_at_a_later_close(order):
    ledger = accepted_ledger(order)
    original = ledger.history()

    with pytest.raises(ValueError, match="later than the decision session"):
        ledger.cancel(order.order_id, session=DECISION)
    assert ledger.history() == original

    ledger.cancel(order.order_id, session=NEXT)
    cancelled = ledger.snapshot(order.order_id)
    assert cancelled.order is order
    assert cancelled.state is OrderState.CANCELLED
    assert cancelled.cancellation_session == NEXT
    assert cancelled.fill is cancelled.rejection_reason is None

    with pytest.raises(ValueError, match="from CANCELLED to CANCELLED"):
        ledger.cancel(order.order_id, session=LATER)
    with pytest.raises(ValueError, match="from CANCELLED to FILLED"):
        ledger.record_fill(fill_for(order, LATER))
    assert ledger.history() == (*original, cancelled)
    assert original[-1].state is OrderState.ACCEPTED


def test_cannot_fill_before_acceptance_or_accept_twice(order):
    ledger = OrderLedger()
    ledger.register(order)
    with pytest.raises(ValueError, match="from CREATED to FILLED"):
        ledger.record_fill(fill_for(order))
    assert [entry.state for entry in ledger.history()] == [OrderState.CREATED]

    ledger.accept(order.order_id)
    with pytest.raises(ValueError, match="from ACCEPTED to ACCEPTED"):
        ledger.accept(order.order_id)
    assert [entry.state for entry in ledger.history()] == [
        OrderState.CREATED, OrderState.ACCEPTED
    ]


def test_duplicate_id_preserves_existing_order_and_history(order):
    ledger = accepted_ledger(order)
    original = ledger.history()
    with pytest.raises(
        ValueError, match="duplicate order identifier 'order-1'"
    ):
        ledger.register(replace(order, symbol="QQQ"))

    assert ledger.history() == original
    assert ledger.snapshot("order-1").order is order


def test_unknown_order_id_is_rejected_for_each_operation(order):
    ledger = OrderLedger()
    for action in (
        lambda: ledger.snapshot(order.order_id),
        lambda: ledger.accept(order.order_id),
        lambda: ledger.reject(order.order_id, "Missing order"),
        lambda: ledger.record_fill(fill_for(order)),
    ):
        with pytest.raises(
            ValueError, match="unknown order identifier 'order-1'"
        ):
            action()

    assert ledger.history() == ()


@pytest.mark.parametrize("quantity", [0, -1, 1.0, 1.5, True, False, "1"])
def test_rejects_invalid_quantities_on_orders_and_fills(order, quantity):
    message = "quantity must be a positive integer"
    with pytest.raises(ValueError, match=message):
        replace(order, quantity=quantity)
    with pytest.raises(ValueError, match=message):
        replace(fill_for(order), quantity=quantity)


@pytest.mark.parametrize(
    "price",
    [0, -1, float("nan"), float("inf"), float("-inf"), True, False, "10"],
)
def test_rejects_invalid_prices(order, price):
    with pytest.raises(
        ValueError, match="price must be a finite, positive number"
    ):
        replace(fill_for(order), price=price)


@pytest.mark.parametrize("symbol", ["", " ", " SPY", "SP Y", "SPY\0", 42])
def test_rejects_invalid_symbols_on_orders_and_fills(order, symbol):
    with pytest.raises(ValueError, match="symbol must be"):
        replace(order, symbol=symbol)
    with pytest.raises(ValueError, match="symbol must be"):
        replace(fill_for(order), symbol=symbol)


def test_preserves_punctuated_symbol_and_integral_price(order):
    order = replace(order, symbol="BRK.B")
    ledger = accepted_ledger(order)
    fill = replace(fill_for(order), price=100)

    ledger.record_fill(fill)

    assert ledger.snapshot(order.order_id).fill == fill
    assert fill.symbol == "BRK.B"
    assert type(fill.price) is int


@pytest.mark.parametrize("order_id", ["", " order-1", "order-1 ", None])
def test_rejects_invalid_order_ids(order, order_id):
    with pytest.raises(ValueError, match="order_id must be nonempty"):
        replace(order, order_id=order_id)
    with pytest.raises(ValueError, match="order_id must be nonempty"):
        replace(fill_for(order), order_id=order_id)


def test_side_requires_an_explicit_enum_value(order):
    with pytest.raises(ValueError, match="side must be OrderSide"):
        replace(order, side="buy")
    with pytest.raises(ValueError, match="side must be OrderSide"):
        replace(fill_for(order), side="buy")


@pytest.mark.parametrize("execution_session", [date(2025, 1, 2), DECISION])
def test_rejects_same_or_earlier_session_fills(order, execution_session):
    assert order.earliest_execution_session is None
    ledger = accepted_ledger(order)

    with pytest.raises(ValueError, match="fill session must be later"):
        ledger.record_fill(fill_for(order, execution_session))

    assert ledger.snapshot(order.order_id).state is OrderState.ACCEPTED
    assert len(ledger.history()) == 2


@pytest.mark.parametrize(
    "changes, message",
    [
        (
            {"decision_session": datetime(2025, 1, 3)},
            "decision_session must be a date",
        ),
        (
            {"earliest_execution_session": "2025-01-06"},
            "earliest_execution_session must be a date",
        ),
        (
            {"earliest_execution_session": DECISION},
            "must be later than decision_session",
        ),
        (
            {"earliest_execution_session": date(2025, 1, 2)},
            "must be later than decision_session",
        ),
    ],
)
def test_rejects_invalid_order_sessions(order, changes, message):
    with pytest.raises(ValueError, match=message):
        replace(order, **changes)


def test_fill_session_requires_a_date(order):
    with pytest.raises(ValueError, match="execution_session must be a date"):
        replace(fill_for(order), execution_session=datetime(2025, 1, 6))


def test_honors_explicit_execution_eligibility_bound(order):
    delayed = replace(order, earliest_execution_session=LATER)
    ledger = accepted_ledger(delayed)

    with pytest.raises(
        ValueError, match="before the earliest eligible session"
    ):
        ledger.record_fill(fill_for(delayed, NEXT))
    assert ledger.snapshot(delayed.order_id).state is OrderState.ACCEPTED

    ledger.record_fill(fill_for(delayed, LATER))
    assert ledger.snapshot(delayed.order_id).fill.execution_session == LATER


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"symbol": "QQQ"}, "symbol"),
        ({"side": OrderSide.SELL}, "side"),
        ({"quantity": 11}, "quantity"),
        ({"quantity": 13}, "quantity"),
    ],
)
def test_fill_must_match_the_accepted_order(order, changes, field):
    ledger = accepted_ledger(order)
    history = ledger.history()

    with pytest.raises(ValueError, match=f"fill {field} does not match"):
        ledger.record_fill(replace(fill_for(order), **changes))

    assert ledger.history() == history
    assert ledger.snapshot(order.order_id).state is OrderState.ACCEPTED


@pytest.mark.parametrize("reason", ["", " \t\n", None, 5])
def test_rejection_requires_a_nonblank_reason(order, reason):
    ledger = accepted_ledger(order)
    history = ledger.history()

    with pytest.raises(
        ValueError, match="rejection reason must be a nonempty string"
    ):
        ledger.reject(order.order_id, reason)

    assert ledger.history() == history
    assert ledger.snapshot(order.order_id).state is OrderState.ACCEPTED


def test_history_is_deterministic_and_preserves_interleaved_call_order(order):
    second = replace(order, order_id="order-2", symbol="QQQ")

    def run():
        ledger = OrderLedger()
        ledger.register(second)
        ledger.register(order)
        ledger.accept(order.order_id)
        ledger.reject(second.order_id, "No eligible session")
        ledger.record_fill(fill_for(order))
        return ledger.history()

    history = run()
    assert history == run()
    assert [(entry.order.order_id, entry.state) for entry in history] == [
        ("order-2", OrderState.CREATED),
        ("order-1", OrderState.CREATED),
        ("order-1", OrderState.ACCEPTED),
        ("order-2", OrderState.REJECTED),
        ("order-1", OrderState.FILLED),
    ]


def test_retained_snapshots_and_original_records_stay_immutable(order):
    ledger = accepted_ledger(order)
    history = ledger.history()
    accepted = ledger.snapshot(order.order_id)
    fill = fill_for(order)

    ledger.record_fill(fill)

    assert len(history) == 2
    assert history[-1] is accepted
    assert accepted.state is OrderState.ACCEPTED
    assert accepted.fill is None
    assert accepted.order is order
    assert order == MarketOrder("order-1", "SPY", OrderSide.BUY, 12, DECISION)
    assert ledger.history()[:2] == history
    with pytest.raises(TypeError):
        setitem(history, 0, None)
    with pytest.raises(FrozenInstanceError):
        setattr(accepted, "state", OrderState.REJECTED)
    with pytest.raises(FrozenInstanceError):
        setattr(order, "quantity", 1)
    with pytest.raises(FrozenInstanceError):
        setattr(fill, "price", 1)
