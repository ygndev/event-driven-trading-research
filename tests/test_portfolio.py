from dataclasses import FrozenInstanceError, replace
from datetime import date
from decimal import Decimal, Inexact, localcontext
from operator import setitem

import pytest

from trading_research.orders import (
    Fill,
    MarketOrder,
    OrderLedger,
    OrderSide,
    OrderState,
)
from trading_research.portfolio import PortfolioLedger, mark_to_market

DECISION = date(2025, 1, 3)
FIRST = date(2025, 1, 6)
NEXT = date(2025, 1, 7)


def trade(order_id, side, quantity, price, *, symbol="SPY", session=FIRST):
    return Fill(order_id, symbol, session, side, quantity, price)


def test_initial_state_and_empty_valuation():
    ledger = PortfolioLedger(Decimal("1000.001"))
    state = ledger.snapshot()

    assert state.initial_cash == state.cash == Decimal("1000.001")
    assert state.positions == {}
    assert state.cumulative_commissions == Decimal(0)
    assert ledger.history() == ()
    assert mark_to_market(state) == Decimal("1000.001")


@pytest.mark.parametrize(
    "cash",
    [0, -1, True, False, float("nan"), float("inf"), Decimal("sNaN"), "1"],
)
def test_initial_cash_must_be_positive_and_finite(cash):
    with pytest.raises(ValueError, match="initial cash"):
        PortfolioLedger(cash)


def test_buy_then_sell_conserves_marked_equity_apart_from_fees():
    ledger = PortfolioLedger(1000)
    buy = trade("buy", OrderSide.BUY, 4, 25.15)
    sell = trade("sell", OrderSide.SELL, 4, 27.25, session=NEXT)
    before_buy = mark_to_market(ledger.snapshot())

    ledger.apply_fill(buy, Decimal("0.40"))
    bought = ledger.snapshot()
    buy_equity = mark_to_market(bought, {"SPY": 25.15})

    assert bought.positions == {"SPY": 4}
    assert bought.cash == Decimal("899.00")
    assert buy_equity == Decimal("999.60")
    assert before_buy - buy_equity == Decimal("0.40")

    before_sell = mark_to_market(bought, {"SPY": 27.25})
    ledger.apply_fill(sell, 0.60)
    sold = ledger.snapshot()

    assert before_sell == Decimal("1008.00")
    assert sold.positions == {}
    assert sold.cash == mark_to_market(sold) == Decimal("1007.40")
    assert before_sell - sold.cash == Decimal("0.60")
    assert sold.cumulative_commissions == Decimal("1.00")
    assert sold.initial_cash == Decimal("1000")

    first, second = ledger.history()
    assert (first.fill, first.commission, first.cash_after) == (
        buy, Decimal("0.40"), Decimal("899.00")
    )
    assert (second.fill, second.commission, second.cash_after) == (
        sell, Decimal("0.60"), Decimal("1007.40")
    )
    assert second.fill.order_id == "sell"
    assert second.fill.execution_session == NEXT
    assert (second.fill.symbol, second.fill.side, second.fill.quantity) == (
        "SPY", OrderSide.SELL, 4
    )
    assert second.fill.price == 27.25


def test_multiple_symbols_and_a_sale_that_leaves_one_share():
    ledger = PortfolioLedger(1000)
    ledger.apply_fill(trade("first", OrderSide.BUY, 2, 25.50), 0.20)
    ledger.apply_fill(
        trade("second", OrderSide.BUY, 3, 15.25, symbol="QQQ"), 0.35
    )
    sale = trade("third", OrderSide.SELL, 1, 26, session=NEXT)
    ledger.apply_fill(sale, 0.10)

    state = ledger.snapshot()
    assert state.positions == {"SPY": 1, "QQQ": 3}
    assert state.cash == Decimal("928.60")
    assert state.cumulative_commissions == Decimal("0.65")
    marks = {"SPY": 26.25, "QQQ": Decimal("16.10")}
    assert mark_to_market(state, marks) == Decimal("1003.15")
    assert [entry.fill.order_id for entry in ledger.history()] == [
        "first", "second", "third"
    ]


def test_exact_cash_boundary_with_decimal_float_inputs():
    ledger = PortfolioLedger(0.30)

    ledger.apply_fill(trade("buy", OrderSide.BUY, 2, 0.10), 0.10)

    state = ledger.snapshot()
    assert state.positions == {"SPY": 2}
    assert state.cash == Decimal("0.00")
    assert mark_to_market(state, {"SPY": 0.10}) == Decimal("0.20")


def test_zero_fee_fills_transfer_value_without_changing_equity():
    ledger = PortfolioLedger(100)
    ledger.apply_fill(trade("buy", OrderSide.BUY, 2, 17.25))
    assert ledger.snapshot().cash == Decimal("65.50")
    assert mark_to_market(ledger.snapshot(), {"SPY": 17.25}) == Decimal("100")

    ledger.apply_fill(trade("sell", OrderSide.SELL, 1, 17.25))
    assert ledger.snapshot().cash == Decimal("82.75")
    assert mark_to_market(ledger.snapshot(), {"SPY": 17.25}) == Decimal("100")
    assert ledger.snapshot().cumulative_commissions == 0


def test_sell_proceeds_can_exactly_cover_its_commission():
    ledger = PortfolioLedger(10)
    ledger.apply_fill(trade("buy", OrderSide.BUY, 1, 5))

    ledger.apply_fill(trade("sell", OrderSide.SELL, 1, 5), 5)

    assert ledger.snapshot().positions == {}
    assert ledger.snapshot().cash == Decimal("5")
    assert ledger.snapshot().cumulative_commissions == Decimal("5")


@pytest.mark.parametrize(
    ("side", "quantity", "price", "commission", "message"),
    [
        (OrderSide.BUY, 1, 11, 0, "insufficient cash"),
        (OrderSide.BUY, 1, 10, 0.01, "insufficient cash"),
        (OrderSide.SELL, 2, 10, 0, "insufficient holdings"),
        (OrderSide.SELL, 1, 10, 10.01, "commission exceeds proceeds"),
    ],
)
def test_rejected_fill_leaves_state_unchanged_and_id_can_be_retried(
    side, quantity, price, commission, message
):
    ledger = PortfolioLedger(20)
    ledger.apply_fill(trade("seed", OrderSide.BUY, 1, 9.50), 0.50)
    state = ledger.snapshot()
    history = ledger.history()
    rejected = trade("retry", side, quantity, price, session=NEXT)

    with pytest.raises(ValueError, match=message):
        ledger.apply_fill(rejected, commission)

    assert ledger.snapshot() == state
    assert ledger.history() == history

    ledger.apply_fill(replace(rejected, quantity=1, price=10))
    assert len(ledger.history()) == 2
    assert ledger.history()[-1].fill.order_id == "retry"
    assert ledger.snapshot().cumulative_commissions == Decimal("0.50")
    if side is OrderSide.BUY:
        assert ledger.snapshot().cash == Decimal("0")
        assert ledger.snapshot().positions == {"SPY": 2}
    else:
        assert ledger.snapshot().cash == Decimal("20")
        assert ledger.snapshot().positions == {}


@pytest.mark.parametrize(
    "commission",
    [
        -1, True, False, float("inf"), float("-inf"), float("nan"),
        Decimal("NaN"), "0",
    ],
)
def test_invalid_commission_leaves_state_unchanged(commission):
    ledger = PortfolioLedger(100)
    buy = trade("buy", OrderSide.BUY, 1, 10)
    before = ledger.snapshot()

    with pytest.raises(ValueError, match="commission"):
        ledger.apply_fill(buy, commission)

    assert ledger.snapshot() == before
    assert ledger.history() == ()


def test_fill_and_order_identifier_cannot_be_booked_twice():
    ledger = PortfolioLedger(100)
    buy = trade("buy", OrderSide.BUY, 1, 10)
    ledger.apply_fill(buy, 0.05)
    state = ledger.snapshot()
    history = ledger.history()

    for duplicate in (buy, replace(buy, symbol="QQQ", execution_session=NEXT)):
        with pytest.raises(
            ValueError, match="fill for order 'buy' was already applied"
        ):
            ledger.apply_fill(duplicate, 0.15)
        assert ledger.snapshot() == state
        assert ledger.history() == history


def test_retained_snapshots_and_transaction_history_are_read_only():
    ledger = PortfolioLedger(100)
    ledger.apply_fill(trade("buy", OrderSide.BUY, 1, 10), 0.10)
    bought = ledger.snapshot()
    history = ledger.history()

    ledger.apply_fill(trade("sell", OrderSide.SELL, 1, 10, session=NEXT))

    assert bought.positions == {"SPY": 1}
    assert bought.cash == Decimal("89.90")
    assert len(history) == 1
    assert ledger.snapshot().positions == {}
    assert len(ledger.history()) == 2
    with pytest.raises(TypeError):
        setitem(bought.positions, "SPY", 10)
    with pytest.raises(TypeError):
        setitem(history, 0, None)
    with pytest.raises(FrozenInstanceError):
        setattr(bought, "cash", Decimal(0))
    with pytest.raises(FrozenInstanceError):
        setattr(history[0], "commission", Decimal(0))


@pytest.mark.parametrize("prices", [None, {}, {"QQQ": 10}])
def test_valuation_requires_a_price_for_every_open_position(prices):
    ledger = PortfolioLedger(100)
    ledger.apply_fill(trade("buy", OrderSide.BUY, 1, 10))

    with pytest.raises(ValueError, match="missing valuation price for 'SPY'"):
        mark_to_market(ledger.snapshot(), prices)


@pytest.mark.parametrize(
    "price",
    [0, -0.1, True, float("nan"), float("inf"), Decimal("-Infinity"), "10"],
)
def test_invalid_marks_do_not_change_the_ledger(price):
    ledger = PortfolioLedger(100)
    ledger.apply_fill(trade("buy", OrderSide.BUY, 1, 10), 0.10)
    before = ledger.snapshot()
    history = ledger.history()

    with pytest.raises(ValueError, match="valuation price for 'SPY'"):
        mark_to_market(before, {"SPY": price})

    assert ledger.snapshot() == before
    assert ledger.history() == history
    assert mark_to_market(before, {"SPY": 20}) == Decimal("109.90")


def test_accounting_is_independent_of_ambient_decimal_precision():
    with localcontext() as context:
        context.prec = 2
        ledger = PortfolioLedger(100)
        ledger.apply_fill(trade("buy", OrderSide.BUY, 2, 12.45), 0.01)
        state = ledger.snapshot()

        assert state.cash == Decimal("75.09")
        assert mark_to_market(state, {"SPY": 12.45}) == Decimal("99.99")


def test_inexact_arithmetic_does_not_book_the_fill():
    ledger = PortfolioLedger(Decimal("1e35"))
    state = ledger.snapshot()
    buy = trade("buy", OrderSide.BUY, 1, 1)

    with pytest.raises(Inexact):
        ledger.apply_fill(buy)

    assert ledger.snapshot() == state
    assert ledger.history() == ()
    ledger.apply_fill(replace(buy, price=10**34))
    assert ledger.snapshot().cash == Decimal("9e34")


def test_accounting_and_order_ledger_are_independent():
    order = MarketOrder("buy", "SPY", OrderSide.BUY, 2, DECISION)
    orders = OrderLedger()
    orders.register(order)
    orders.accept(order.order_id)
    orders.record_fill(trade(order.order_id, OrderSide.BUY, 2, 10))
    recorded = orders.snapshot(order.order_id)
    ledger = PortfolioLedger(20)

    with pytest.raises(ValueError, match="insufficient cash"):
        ledger.apply_fill(recorded.fill, 0.50)

    assert orders.snapshot(order.order_id) is recorded
    assert recorded.state is OrderState.FILLED
    assert ledger.history() == ()
    assert ledger.snapshot().cash == Decimal("20")
    ledger.apply_fill(recorded.fill)
    assert ledger.snapshot().cash == Decimal("0")
    assert ledger.snapshot().positions == {"SPY": 2}
    assert ledger.history()[0].fill is recorded.fill
    assert orders.snapshot(order.order_id) is recorded
