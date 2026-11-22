from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from fractions import Fraction

import pytest

from trading_research import backtest
from trading_research.backtest import PendingOrderSplitError, run_backtest
from trading_research.coordinator import ExecutionCoordinator
from trading_research.corporate_actions import CashDividend, StockSplit
from trading_research.daily_bars import DailyBar, DatasetMetadata, PriceAdjustment
from trading_research.orders import OrderSide, OrderState
from trading_research.performance import evaluate_performance
from trading_research.portfolio import PortfolioRejection
from trading_research.signals import SignalEvent


FIRST = date(2025, 1, 3)
BUY = date(2025, 1, 6)
SPLIT = date(2025, 1, 7)
EX = date(2025, 1, 10)
SATURDAY = date(2025, 1, 11)
LAST = date(2025, 1, 13)


def dataset(symbol, *observations):
    metadata = DatasetMetadata(
        "Synthetic", symbol, PriceAdjustment.RAW,
        datetime(2025, 1, 15, tzinfo=UTC),
    )
    bars = tuple(
        DailyBar(symbol, day, opened, max(opened, close), min(opened, close), close, 1000)
        for day, opened, close in observations
    )
    return metadata, bars


def hold_spy(context):
    if context.session_date == FIRST:
        return (SignalEvent("SPY", FIRST, 1),)
    return ()


def run(datasets, strategy=hold_spy, actions=(), *, cash=2000):
    return run_backtest(
        datasets, strategy, initial_cash=cash, adjustment_bps=0,
        commission=0, corporate_actions=actions,
    )


def test_no_actions_retains_the_existing_execution_path():
    spy = dataset("SPY", (FIRST, 200, 200), (BUY, 190, 195))
    default = run_backtest(
        (spy,), hold_spy, initial_cash=2000, adjustment_bps=0, commission=0
    )

    assert default == run((spy,))
    before, after = default.sessions
    assert before.equity == 2000
    assert after.equity == 2050
    assert after.portfolio.cash == 100
    assert after.portfolio.positions == {"SPY": 10}
    assert default.split_adjustments == default.dividend_payments == ()


def test_purchase_split_entitlement_weekend_payment_and_final_equity():
    spy = dataset(
        "SPY", (FIRST, 200, 200), (BUY, 200, 200), (SPLIT, 100, 100),
        (EX, 99, 99), (LAST, 101, 102),
    )
    observed = []
    actions = [CashDividend("cash", "SPY", EX, SATURDAY, 1), StockSplit("SPY", SPLIT, 2)]
    original_actions = actions.copy()
    original_bars = spy[1]

    def strategy(context):
        observed.append(context)
        return hold_spy(context)

    result = run((spy,), strategy, actions)

    assert result == run((spy,), hold_spy, reversed(actions))
    assert actions == original_actions and spy[1] == original_bars
    assert [item.session_date for item in observed] == [FIRST, BUY, SPLIT, EX, LAST]
    assert tuple(event.bar for event in observed[1].history("SPY")) == spy[1][:2]
    assert observed[2].events[0].bar is spy[1][2]
    assert observed[2].execution.portfolio.positions == {"SPY": 20}
    assert all(not item.execution.portfolio.unpaid_receivables for item in observed[:3])

    fill, = result.fills
    assert (fill.execution_session, fill.quantity, fill.price) == (BUY, 10, 200)
    assert tuple(entry.fill for entry in result.order_history if entry.fill) == (fill,)
    split, = result.split_adjustments
    assert (split.shares_before, split.shares_after) == (10, 20)
    assert [row.equity for row in result.sessions] == [2000, 2000, 2000, 2000, 2060]
    assert [row.portfolio.cash for row in result.sessions] == [2000, 0, 0, 0, 20]

    claim, = result.sessions[-2].portfolio.unpaid_receivables
    assert (claim.prior_close_session, claim.entitled_shares, claim.total_amount) == (
        SPLIT, 20, Decimal(20)
    )
    payment, = result.dividend_payments
    assert payment.receivable is claim
    assert payment.session_date == SATURDAY
    assert payment.cash_after == 20
    assert result.sessions[-1].portfolio.unpaid_receivables == ()
    assert result.sessions[-1].portfolio.positions == {"SPY": 20}
    with pytest.raises(FrozenInstanceError):
        setattr(payment, "cash_after", Decimal(0))


def test_reverse_split_keeps_whole_shares_and_raw_value():
    spy = dataset("SPY", (FIRST, 100, 100), (BUY, 100, 100), (SPLIT, 200, 200))

    result = run((spy,), actions=(StockSplit("SPY", SPLIT, Fraction(1, 2)),))

    assert result.sessions[-1].portfolio.positions == {"SPY": 10}
    assert result.sessions[-1].portfolio.cash == 0
    assert result.sessions[-2].equity == result.sessions[-1].equity == 2000


@pytest.fixture
def coordinators(monkeypatch):
    tracked = []

    def capture(*args):
        coordinator = ExecutionCoordinator(*args)
        tracked.append(coordinator)
        return coordinator

    monkeypatch.setattr(backtest, "ExecutionCoordinator", capture)
    return tracked


def test_pending_split_order_aborts_before_any_same_day_action(coordinators):
    spy = dataset("SPY", *((d, 100, 100) for d in (FIRST, BUY, SPLIT)))
    qqq = dataset("QQQ", *((d, 100, 100) for d in (FIRST, BUY, SPLIT)))

    def strategy(context):
        day = context.session_date
        if day == FIRST:
            return (SignalEvent("QQQ", day, 0.5),)
        if day == BUY:
            return (SignalEvent("SPY", day, 0.5),)
        pytest.fail("strategy ran after the blocked split")

    actions = (
        StockSplit("QQQ", SPLIT, 2), StockSplit("SPY", SPLIT, 2),
        CashDividend("cash", "QQQ", SPLIT, SPLIT, 1),
    )
    with pytest.raises(PendingOrderSplitError, match="SPY crosses the split") as error:
        run((spy, qqq), strategy, actions, cash=1000)

    coordinator, = coordinators
    assert error.value.order.order_id == f"{BUY}:SPY"
    assert coordinator.order_snapshot(error.value.order.order_id).state is OrderState.ACCEPTED
    assert coordinator.portfolio_snapshot().positions == {"QQQ": 5}
    assert coordinator.portfolio_snapshot().cash == 500
    assert len(coordinator.portfolio_history()) == 1
    assert coordinator.split_history() == coordinator.dividend_payments() == ()


def test_fractional_split_leaves_the_whole_action_batch_unpublished(coordinators):
    spy = dataset("SPY", *((d, 100, 100) for d in (FIRST, BUY, SPLIT)))
    qqq = dataset("QQQ", *((d, 100, 100) for d in (FIRST, BUY, SPLIT)))

    def strategy(context):
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.3), SignalEvent("QQQ", FIRST, 0.4))
        return ()

    with pytest.raises(PortfolioRejection, match="fractional shares"):
        run(
            (spy, qqq), strategy, (
                StockSplit("QQQ", SPLIT, 2), StockSplit("SPY", SPLIT, Fraction(1, 2))
            ),
            cash=1000,
        )

    coordinator, = coordinators
    assert coordinator.portfolio_snapshot().positions == {"QQQ": 4, "SPY": 3}
    assert coordinator.portfolio_snapshot().cash == 300
    assert coordinator.split_history() == ()
    assert len(coordinator.portfolio_history()) == 2


def test_split_is_already_booked_when_unaffected_order_executes(monkeypatch):
    spy = dataset("SPY", (FIRST, 100, 100), (BUY, 100, 100), (SPLIT, 50, 51))
    qqq = dataset("QQQ", *((d, 100, 100) for d in (FIRST, BUY, SPLIT)))
    execute = ExecutionCoordinator.execute
    witnessed = []

    def inspect(coordinator, order_id, proposal):
        if proposal.fill.execution_session == SPLIT:
            witnessed.append(coordinator.portfolio_snapshot())
        return execute(coordinator, order_id, proposal)

    def strategy(context):
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.5),)
        if context.session_date == BUY:
            return (SignalEvent("QQQ", BUY, 0.5),)
        return ()

    monkeypatch.setattr(ExecutionCoordinator, "execute", inspect)
    result = run((spy, qqq), strategy, (StockSplit("SPY", SPLIT, 2),))

    opening, = witnessed
    assert opening.positions == {"SPY": 20}
    assert opening.cash == 1000
    assert result.sessions[-1].portfolio.positions == {"SPY": 20, "QQQ": 10}


def test_entitlement_uses_each_symbols_preceding_close_and_ex_buy_earns_nothing():
    spy = dataset("SPY", (FIRST, 100, 100), (BUY, 100, 100), (EX, 99, 99), (LAST, 99, 99))
    qqq = dataset("QQQ", (FIRST, 100, 100), (EX, 100, 100), (LAST, 100, 100))

    def strategy(context):
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.5), SignalEvent("QQQ", FIRST, 0.5))
        return ()

    result = run(
        (spy, qqq), strategy, (
            CashDividend("qqq", "QQQ", EX, LAST, 2),
            CashDividend("spy", "SPY", EX, LAST, 1),
        ),
        cash=1000,
    )

    assert [(fill.symbol, fill.execution_session) for fill in result.fills] == [
        ("SPY", BUY), ("QQQ", EX)
    ]
    claim, = result.sessions[-2].portfolio.unpaid_receivables
    assert claim.dividend.action_id == "spy" and claim.prior_close_session == BUY
    assert claim.entitled_shares == 5
    payment, = result.dividend_payments
    assert payment.session_date == LAST and payment.cash_after == 5
    assert result.sessions[-1].equity == 1000


@pytest.mark.parametrize("pay_date", [EX, SATURDAY])
def test_ex_date_sales_keep_entitlement_and_buying_power_changes_on_payment(pay_date):
    spy = dataset("SPY", (FIRST, 100, 100), (BUY, 100, 100), (EX, 99, 99), (LAST, 99, 99))
    qqq = dataset("QQQ", (FIRST, 100, 100), (BUY, 100, 100), (EX, 101, 101), (LAST, 101, 101))

    def strategy(context):
        if context.session_date == BUY:
            return (SignalEvent("SPY", BUY, 0), SignalEvent("QQQ", BUY, 1))
        return hold_spy(context)

    result = run(
        (spy, qqq), strategy, (CashDividend("cash", "SPY", EX, pay_date, 2),),
        cash=1000,
    )

    ex = result.sessions[-2]
    payment, = result.dividend_payments
    assert payment.session_date == pay_date
    assert payment.receivable.entitled_shares == 10
    assert payment.receivable.total_amount == 20
    assert result.fills[1].side is OrderSide.SELL
    assert result.fills[1].execution_session == EX
    assert result.sessions[-1].equity == ex.equity == 1010
    if pay_date == EX:
        assert result.fills[-1].symbol == "QQQ"
        assert result.fills[-1].execution_session == EX
        assert payment.cash_after == 20
        assert ex.portfolio.positions == {"QQQ": 10} and ex.portfolio.cash == 0
    else:
        rejected, = result.rejected_orders
        assert "insufficient cash" in rejected.rejection_reason
        assert ex.portfolio.positions == {} and ex.portfolio.cash == 990
        assert ex.portfolio.unpaid_receivables == (payment.receivable,)
        assert payment.cash_after == 1010
        assert evaluate_performance(result).session_returns[-2].long_exposure == 0


def test_weekend_payment_funds_the_next_observed_open():
    spy = dataset(
        "SPY", (FIRST, 100, 100), (BUY, 100, 100), (EX, 99, 99), (LAST, 99, 99)
    )
    qqq = dataset(
        "QQQ", (FIRST, 99, 99), (BUY, 99, 99), (EX, 99, 99), (LAST, 100, 100)
    )

    def strategy(context):
        if context.session_date == EX:
            return (SignalEvent("SPY", EX, 0), SignalEvent("QQQ", EX, 1))
        return hold_spy(context)

    result = run(
        (spy, qqq), strategy, (CashDividend("cash", "SPY", EX, SATURDAY, 2),),
        cash=1000,
    )
    assert result.sessions[-2].portfolio.cash == 0
    payment, = result.dividend_payments
    assert (payment.session_date, payment.cash_after) == (SATURDAY, Decimal(20))
    assert [(fill.symbol, fill.execution_session) for fill in result.fills] == [
        ("SPY", BUY), ("SPY", LAST), ("QQQ", LAST)
    ]
    assert result.sessions[-1].portfolio.cash == 10
    assert result.sessions[-1].portfolio.positions == {"QQQ": 10}
    assert result.sessions[-1].equity == 1010


def test_same_session_split_and_dividend_use_the_pre_split_entitlement():
    spy = dataset("SPY", (FIRST, 200, 200), (BUY, 200, 200), (SPLIT, 99, 99))
    actions = (
        StockSplit("SPY", SPLIT, 2), CashDividend("cash", "SPY", SPLIT, SPLIT, 2)
    )

    result = run((spy,), actions=actions)

    payment, = result.dividend_payments
    assert payment.receivable.entitled_shares == 10
    assert payment.receivable.total_amount == 20
    assert payment.session_date == SPLIT
    assert result.sessions[-1].portfolio.positions == {"SPY": 20}
    assert result.sessions[-1].portfolio.cash == 20
    assert result.sessions[-1].equity == 2000


def test_payment_after_final_bar_has_audit_but_does_not_revise_the_close():
    spy = dataset("SPY", (FIRST, 100, 100), (BUY, 100, 100), (EX, 99, 99))
    dividend = CashDividend("late", "SPY", EX, SATURDAY, 1)

    result = run((spy,), actions=(dividend,), cash=1000)

    assert [row.session_date for row in result.sessions] == [FIRST, BUY, EX]
    assert result.sessions[-1].portfolio.cash == 0
    claim, = result.sessions[-1].portfolio.unpaid_receivables
    payment, = result.dividend_payments
    assert payment.session_date == SATURDAY
    assert payment.receivable is claim
    assert payment.cash_after == 10
    assert result.sessions[-1].equity == 1000
    assert evaluate_performance(result).session_returns[-1].long_exposure == Decimal("0.99")


@pytest.mark.parametrize(("actions", "message"), [
    ((StockSplit("SPY", SPLIT, 2), StockSplit("SPY", SPLIT, 3)), "conflicting split"),
    ((CashDividend("a", "SPY", EX, LAST, 1),
      CashDividend("a", "SPY", SPLIT, LAST, 1)), "conflicting dividend"),
    ((CashDividend("a", "SPY", EX, LAST, 1),
      CashDividend("b", "SPY", EX, LAST, 1)), "conflicting dividend"),
    ((StockSplit("QQQ", SPLIT, 2),), "unknown symbol"),
    ((StockSplit("SPY", SATURDAY, 2),), "requires an observed session"),
    ((CashDividend("a", "SPY", FIRST, LAST, 1),), "requires a preceding observed close"),
    ((CashDividend("a", "SPY", SATURDAY, LAST, 1),), "requires an observed session"),
    ((object(),), "must be StockSplit or CashDividend"),
])
def test_invalid_actions_fail_before_strategy_replay(actions, message):
    spy = dataset("SPY", *((d, 100, 100) for d in (FIRST, BUY, SPLIT, EX, LAST)))

    with pytest.raises(ValueError, match=message):
        run((spy,), lambda context: pytest.fail("unexpected strategy replay"), actions)


@pytest.mark.parametrize("adjustment", [PriceAdjustment.SPLIT_ADJUSTED, PriceAdjustment.TOTAL_RETURN_ADJUSTED])
def test_corporate_actions_cannot_enable_execution_on_adjusted_bars(adjustment):
    metadata, bars = dataset("SPY", (FIRST, 100, 100), (BUY, 100, 100))
    spy = replace(metadata, adjustment=adjustment), bars

    with pytest.raises(ValueError, match="backtest requires RAW data"):
        run((spy,), actions=(CashDividend("a", "SPY", BUY, BUY, 1),))
