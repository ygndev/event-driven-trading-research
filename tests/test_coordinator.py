from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from operator import setitem

import pytest

from trading_research.coordinator import ExecutionCoordinator
from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.execution import NextOpenExecution, ProposedExecution
from trading_research.orders import (
    Fill,
    MarketOrder,
    OrderLedger,
    OrderSide,
    OrderState,
)
from trading_research.portfolio import PortfolioLedger

FRIDAY = date(2025, 1, 3)
MONDAY = date(2025, 1, 6)
TUESDAY = date(2025, 1, 7)

BUY = MarketOrder("buy", "SPY", OrderSide.BUY, 4, FRIDAY)
SELL = MarketOrder("sell", "SPY", OrderSide.SELL, 4, MONDAY)


def accepted_orders(*orders):
    ledger = OrderLedger()
    for order in orders:
        ledger.register(order)
        ledger.accept(order.order_id)
    return ledger


def proposal_for(order, *, session=MONDAY, price="25", commission="0.50"):
    price = Decimal(price)
    fill = Fill(
        order.order_id, order.symbol, session, order.side, order.quantity,
        price,
    )
    return ProposedExecution(fill, Decimal(commission), price, Decimal(0))


def state_of(coordinator, *order_ids):
    return (
        tuple(coordinator.order_snapshot(order_id) for order_id in order_ids),
        coordinator.order_history(),
        coordinator.portfolio_snapshot(),
        coordinator.portfolio_history(),
    )


def test_coordinated_round_trip_books_prices_and_commissions_once():
    orders = accepted_orders(BUY, SELL)
    portfolio = PortfolioLedger(1000)
    coordinator = ExecutionCoordinator(orders, portfolio)
    model = NextOpenExecution(adjustment_bps=20, commission=Decimal("0.50"))
    metadata = DatasetMetadata(
        "Synthetic", "SPY", PriceAdjustment.RAW,
        datetime(2025, 1, 8, tzinfo=UTC),
    )
    bars = (
        DailyBar("SPY", FRIDAY, 24, 24, 24, 24, 100),
        DailyBar("SPY", MONDAY, 25, 26, 24, 26, 100),
        DailyBar("SPY", TUESDAY, 28, 29, 27, 29, 100),
    )

    purchase = model.propose(coordinator.order_snapshot("buy"), bars, metadata)
    buy_outcome = coordinator.execute("buy", purchase)
    bought = coordinator.portfolio_snapshot()

    assert buy_outcome.state is OrderState.FILLED
    assert buy_outcome.fill is purchase.fill
    assert purchase.fill.execution_session == MONDAY
    assert purchase.fill.price == Decimal("25.05")
    assert bought.cash == Decimal("899.30")
    assert bought.positions == {"SPY": 4}
    assert bought.cumulative_commissions == Decimal("0.50")

    sale = model.propose(coordinator.order_snapshot("sell"), bars, metadata)
    sell_outcome = coordinator.execute("sell", sale)
    sold = coordinator.portfolio_snapshot()

    assert sell_outcome.state is OrderState.FILLED
    assert sell_outcome.fill is sale.fill
    assert sale.fill.execution_session == TUESDAY
    assert sale.fill.price == Decimal("27.944")
    assert sold.cash == Decimal("1010.576")
    assert sold.positions == {}
    assert sold.cumulative_commissions == Decimal("1.00")
    assert [
        (entry.fill, entry.commission, entry.cash_after)
        for entry in coordinator.portfolio_history()
    ] == [
        (purchase.fill, Decimal("0.50"), Decimal("899.30")),
        (sale.fill, Decimal("0.50"), Decimal("1010.576")),
    ]
    assert coordinator.order_history()[-2:] == (buy_outcome, sell_outcome)
    assert orders.snapshot("buy").state is OrderState.ACCEPTED
    assert orders.snapshot("sell").state is OrderState.ACCEPTED
    assert portfolio.snapshot().cash == Decimal("1000")
    assert portfolio.history() == ()


def test_insufficient_buy_cash_including_commission_rejects_only_the_order():
    coordinator = ExecutionCoordinator(
        accepted_orders(BUY), PortfolioLedger(100)
    )
    proposal = proposal_for(BUY)
    before = state_of(coordinator, "buy")

    outcome = coordinator.execute("buy", proposal)

    assert outcome.state is OrderState.REJECTED
    assert outcome.fill is None
    assert outcome.rejection_reason == "order 'buy': insufficient cash"
    assert coordinator.order_history() == (*before[1], outcome)
    assert coordinator.portfolio_snapshot() == before[2]
    assert coordinator.portfolio_history() == before[3]

    rejected = state_of(coordinator, "buy")
    with pytest.raises(ValueError, match="from REJECTED to FILLED"):
        coordinator.execute("buy", proposal)
    assert state_of(coordinator, "buy") == rejected


@pytest.mark.parametrize(
    ("quantity", "price", "commission", "reason"),
    [
        (5, "28", "0.50", "insufficient holdings"),
        (4, "1", "4.01", "commission exceeds proceeds"),
    ],
)
def test_economic_sell_rejection_keeps_holdings_and_history(
    quantity, price, commission, reason
):
    sell = replace(SELL, quantity=quantity)
    coordinator = ExecutionCoordinator(
        accepted_orders(BUY, sell), PortfolioLedger(1000)
    )
    coordinator.execute("buy", proposal_for(BUY))
    before = state_of(coordinator, "buy", "sell")
    sale = proposal_for(
        sell, session=TUESDAY, price=price, commission=commission
    )

    outcome = coordinator.execute("sell", sale)

    assert outcome.state is OrderState.REJECTED
    assert outcome.rejection_reason == f"order 'sell': {reason}"
    assert outcome.fill is None
    assert coordinator.order_snapshot("buy") is before[0][0]
    assert coordinator.order_history() == (*before[1], outcome)
    assert coordinator.portfolio_snapshot() == before[2]
    assert coordinator.portfolio_history() == before[3]


def test_duplicate_commit_preserves_both_histories():
    coordinator = ExecutionCoordinator(
        accepted_orders(BUY), PortfolioLedger(1000)
    )
    proposal = proposal_for(BUY)
    coordinator.execute("buy", proposal)
    committed = state_of(coordinator, "buy")

    with pytest.raises(ValueError, match="from FILLED to FILLED"):
        coordinator.execute("buy", proposal)

    assert state_of(coordinator, "buy") == committed


@pytest.mark.parametrize("state", [OrderState.CREATED, OrderState.REJECTED])
def test_unaccepted_terminal_or_created_order_cannot_be_committed(state):
    orders = OrderLedger()
    orders.register(BUY)
    if state is OrderState.REJECTED:
        orders.reject("buy", "Cancelled by validation")
    coordinator = ExecutionCoordinator(orders, PortfolioLedger(1000))
    before = state_of(coordinator, "buy")

    with pytest.raises(ValueError, match=f"from {state.name} to FILLED"):
        coordinator.execute("buy", proposal_for(BUY))

    assert state_of(coordinator, "buy") == before


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"order_id": "other"}, "fill order ID does not match"),
        ({"symbol": "QQQ"}, "fill symbol does not match"),
        ({"side": OrderSide.SELL}, "fill side does not match"),
        ({"quantity": 2}, "fill quantity does not match"),
        ({"execution_session": FRIDAY}, "later than the decision session"),
        (
            {"execution_session": MONDAY},
            "before the earliest eligible session",
        ),
    ],
)
def test_fill_identity_and_eligibility_errors_are_atomic(changes, message):
    buy = replace(BUY, earliest_execution_session=TUESDAY)
    other = replace(buy, order_id="other")
    coordinator = ExecutionCoordinator(
        accepted_orders(buy, other), PortfolioLedger(1000)
    )
    proposal = proposal_for(buy, session=TUESDAY)
    invalid = replace(proposal, fill=replace(proposal.fill, **changes))
    before = state_of(coordinator, "buy", "other")

    with pytest.raises(ValueError, match=message):
        coordinator.execute("buy", invalid)

    assert state_of(coordinator, "buy", "other") == before


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"opening_price": Decimal("NaN")}, "opening_price"),
        ({"adjustment_bps": Decimal(-1)}, "adjustment_bps"),
        ({"adjustment_bps": Decimal(20)}, "fill price does not match"),
        ({"commission": Decimal("-0.01")}, "commission must be finite"),
    ],
)
def test_invalid_proposals_leave_the_order_accepted(changes, message):
    coordinator = ExecutionCoordinator(
        accepted_orders(BUY), PortfolioLedger(1)
    )
    before = state_of(coordinator, "buy")

    with pytest.raises(ValueError, match=message):
        coordinator.execute("buy", replace(proposal_for(BUY), **changes))

    assert state_of(coordinator, "buy") == before


def test_unknown_order_is_not_a_business_rejection():
    coordinator = ExecutionCoordinator(OrderLedger(), PortfolioLedger(1000))
    before = state_of(coordinator)

    with pytest.raises(ValueError, match="unknown order identifier 'buy'"):
        coordinator.execute("buy", proposal_for(BUY))

    assert state_of(coordinator) == before


def test_competing_buy_orders_follow_explicit_commit_order():
    first = MarketOrder("z", "SPY", OrderSide.BUY, 1, FRIDAY)
    second = MarketOrder("a", "QQQ", OrderSide.BUY, 1, FRIDAY)

    def run(sequence):
        coordinator = ExecutionCoordinator(
            accepted_orders(first, second), PortfolioLedger(100)
        )
        for order in sequence:
            candidate = proposal_for(order, price="60")
            coordinator.execute(order.order_id, candidate)
        return state_of(coordinator, "z", "a")

    forward = run((first, second))
    reverse = run((second, first))
    assert forward == run((first, second))
    assert [entry.state for entry in forward[0]] == [
        OrderState.FILLED, OrderState.REJECTED
    ]
    assert [entry.state for entry in reverse[0]] == [
        OrderState.REJECTED, OrderState.FILLED
    ]
    assert forward[2].cash == reverse[2].cash == Decimal("39.50")
    assert forward[2].positions == {"SPY": 1}
    assert reverse[2].positions == {"QQQ": 1}
    assert [entry.order.order_id for entry in forward[1][-2:]] == ["z", "a"]
    assert [entry.order.order_id for entry in reverse[1][-2:]] == ["a", "z"]
    assert [entry.fill.order_id for entry in forward[3]] == ["z"]
    assert [entry.fill.order_id for entry in reverse[3]] == ["a"]


@pytest.mark.parametrize("ledger_side", ["order", "portfolio"])
def test_unexpected_failure_after_staged_write_leaves_current_pair_intact(
    monkeypatch, ledger_side
):
    extra = MarketOrder("extra", "QQQ", OrderSide.BUY, 1, FRIDAY)
    coordinator = ExecutionCoordinator(
        accepted_orders(BUY, extra), PortfolioLedger(1000)
    )
    coordinator.execute("buy", proposal_for(BUY))
    candidate = proposal_for(extra, price="10")
    before = state_of(coordinator, "buy", "extra")

    with monkeypatch.context() as patch:
        if ledger_side == "order":
            record_fill = OrderLedger.record_fill

            def fail_after_order_write(ledger, fill):
                record_fill(ledger, fill)
                assert ledger.snapshot("extra").state is OrderState.FILLED
                assert state_of(coordinator, "buy", "extra") == before
                raise RuntimeError("injected after staged order write")

            patch.setattr(OrderLedger, "record_fill", fail_after_order_write)
            error = RuntimeError
        else:
            apply_fill = PortfolioLedger.apply_fill

            def fail_after_portfolio_write(ledger, fill, commission):
                apply_fill(ledger, fill, commission)
                assert ledger.snapshot().cash == Decimal("889.00")
                assert ledger.history()[-1].fill is fill
                assert state_of(coordinator, "buy", "extra") == before
                raise ValueError("injected after staged portfolio write")

            patch.setattr(
                PortfolioLedger, "apply_fill", fail_after_portfolio_write
            )
            error = ValueError

        message = f"injected after staged {ledger_side} write"
        with pytest.raises(error, match=message):
            coordinator.execute("extra", candidate)

    assert state_of(coordinator, "buy", "extra") == before
    result = coordinator.execute("extra", candidate)
    assert result.state is OrderState.FILLED
    after = coordinator.portfolio_snapshot()
    assert after.cash == Decimal("889.00")
    assert after.positions == {"SPY": 4, "QQQ": 1}
    assert after.cumulative_commissions == Decimal("1")
    assert coordinator.portfolio_history()[-1].fill is result.fill
    assert len(coordinator.portfolio_history()) == 2


def test_failed_staged_rejection_preserves_acceptance(monkeypatch):
    coordinator = ExecutionCoordinator(
        accepted_orders(BUY), PortfolioLedger(100)
    )
    candidate = proposal_for(BUY)
    before = state_of(coordinator, "buy")
    reject = OrderLedger.reject

    def fail_after_reject(ledger, order_id, reason):
        reject(ledger, order_id, reason)
        assert ledger.snapshot(order_id).state is OrderState.REJECTED
        assert state_of(coordinator, "buy") == before
        raise RuntimeError("injected after staged rejection")

    with monkeypatch.context() as patch:
        patch.setattr(OrderLedger, "reject", fail_after_reject)
        with pytest.raises(RuntimeError, match="after staged rejection"):
            coordinator.execute("buy", candidate)

    assert state_of(coordinator, "buy") == before
    assert coordinator.execute("buy", candidate).state is OrderState.REJECTED


def test_retained_order_and_portfolio_audit_snapshots_are_immutable():
    coordinator = ExecutionCoordinator(
        accepted_orders(BUY), PortfolioLedger(1000)
    )
    accepted = coordinator.order_history()
    opening_balance = coordinator.portfolio_snapshot()

    filled = coordinator.execute("buy", proposal_for(BUY))
    transaction, = coordinator.portfolio_history()

    assert accepted[-1].state is OrderState.ACCEPTED
    assert opening_balance.cash == Decimal("1000")
    assert opening_balance.positions == {}
    with pytest.raises(TypeError):
        setitem(coordinator.order_history(), 0, None)
    with pytest.raises(TypeError):
        setitem(coordinator.portfolio_snapshot().positions, "SPY", 0)
    with pytest.raises(FrozenInstanceError):
        setattr(filled, "fill", None)
    with pytest.raises(FrozenInstanceError):
        setattr(transaction, "commission", Decimal(0))


@pytest.mark.parametrize("booked_side", ["orders", "portfolio", "both"])
def test_initial_ledger_histories_must_agree(booked_side):
    orders = accepted_orders(BUY)
    portfolio = PortfolioLedger(1000)
    proposal = proposal_for(BUY)
    if booked_side in {"orders", "both"}:
        orders.record_fill(proposal.fill)
    if booked_side in {"portfolio", "both"}:
        portfolio.apply_fill(proposal.fill, proposal.commission)

    if booked_side == "both":
        coordinator = ExecutionCoordinator(orders, portfolio)
        assert coordinator.order_history() == orders.history()
        assert coordinator.portfolio_history() == portfolio.history()
        assert coordinator.portfolio_snapshot().cash == Decimal("899.50")
    else:
        with pytest.raises(ValueError, match="histories disagree on fills"):
            ExecutionCoordinator(orders, portfolio)


def test_initial_fill_order_must_match_between_ledgers():
    extra = MarketOrder("extra", "QQQ", OrderSide.BUY, 1, FRIDAY)
    orders = accepted_orders(BUY, extra)
    portfolio = PortfolioLedger(1000)
    first = proposal_for(BUY)
    second = proposal_for(extra)

    orders.record_fill(first.fill)
    orders.record_fill(second.fill)
    portfolio.apply_fill(second.fill, second.commission)
    portfolio.apply_fill(first.fill, first.commission)

    with pytest.raises(ValueError, match="histories disagree on fills"):
        ExecutionCoordinator(orders, portfolio)
