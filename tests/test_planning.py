from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal, localcontext

import pytest

from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.orders import Fill, MarketOrder, OrderLedger, OrderSide
from trading_research.planning import plan_orders
from trading_research.portfolio import PortfolioLedger
from trading_research.replay import MarketEvent
from trading_research.signals import (
    SessionContext,
    SignalEvent,
    generate_signals,
)

EARLIER = date(2025, 1, 3)
SESSION = date(2025, 1, 6)
LATER = date(2025, 1, 7)


def market_event(
    symbol, close, *, opening=None, session=SESSION,
    adjustment=PriceAdjustment.RAW,
):
    opening = close if opening is None else opening
    bar = DailyBar(
        symbol, session, opening, max(opening, close), min(opening, close),
        close, 1_000,
    )
    return MarketEvent(bar, adjustment)


def hold(ledger, symbol, quantity, price):
    fill = Fill(
        f"seed:{symbol}", symbol, EARLIER, OrderSide.BUY, quantity, price
    )
    ledger.apply_fill(fill)


def target(symbol, weight):
    return SignalEvent(symbol, SESSION, weight)


def test_buys_from_cash_using_close_instead_of_same_session_open():
    portfolio = PortfolioLedger(1000).snapshot()
    reference = market_event("SPY", 24.25, opening=20)
    context = SessionContext(SESSION, (reference,))

    orders = plan_orders([target("SPY", 0.6)], portfolio, context)

    assert orders == (
        MarketOrder("2025-01-06:SPY", "SPY", OrderSide.BUY, 24, SESSION),
    )
    assert orders[0].earliest_execution_session is None


def test_marks_existing_holdings_and_reduces_to_target():
    ledger = PortfolioLedger(1000)
    hold(ledger, "SPY", 20, 25)
    context = SessionContext(SESSION, (market_event("SPY", 50),))

    orders = plan_orders([target("SPY", 0.4)], ledger.snapshot(), context)

    # Cash is 500 and the position is worth 1000 at the reference close.
    assert orders == (
        MarketOrder("2025-01-06:SPY", "SPY", OrderSide.SELL, 8, SESSION),
    )


def test_explicit_zero_liquidates_while_empty_input_requests_no_changes():
    ledger = PortfolioLedger(200)
    hold(ledger, "SPY", 5, 20)
    portfolio = ledger.snapshot()

    assert plan_orders([], portfolio, SessionContext(SESSION, ())) == ()

    context = SessionContext(SESSION, (market_event("SPY", 22),))
    assert plan_orders([target("SPY", 0)], portfolio, context) == (
        MarketOrder("2025-01-06:SPY", "SPY", OrderSide.SELL, 5, SESSION),
    )


def test_complete_allocation_requires_explicitly_carried_forward_targets():
    ledger = PortfolioLedger(200)
    hold(ledger, "SPY", 5, 20)
    context = SessionContext(
        SESSION, (market_event("QQQ", 25), market_event("SPY", 20))
    )
    portfolio = ledger.snapshot()
    update = target("QQQ", 0.25)
    prior_target = SignalEvent("SPY", EARLIER, 0.5)

    with pytest.raises(
        ValueError, match="complete allocation .* holdings: SPY"
    ):
        plan_orders([update], portfolio, context)

    retained = replace(prior_target, decision_session=SESSION)
    assert plan_orders([update, retained], portfolio, context) == (
        MarketOrder("2025-01-06:QQQ", "QQQ", OrderSide.BUY, 2, SESSION),
    )
    assert ledger.snapshot().positions == {"SPY": 5}


def test_rebalances_two_symbols_without_reserving_or_counting_sale_cash():
    ledger = PortfolioLedger(1000)
    hold(ledger, "SPY", 10, 100)
    portfolio = ledger.snapshot()
    history = ledger.history()
    order_ledger = OrderLedger()
    context = SessionContext(
        SESSION, (market_event("QQQ", 30), market_event("SPY", 120))
    )

    orders = plan_orders(
        [target("QQQ", 0.75), target("SPY", 0.25)], portfolio, context
    )

    assert orders == (
        MarketOrder("2025-01-06:SPY", "SPY", OrderSide.SELL, 8, SESSION),
        MarketOrder("2025-01-06:QQQ", "QQQ", OrderSide.BUY, 30, SESSION),
    )
    assert ledger.snapshot() == portfolio
    assert portfolio.cash == Decimal(0)
    assert portfolio.positions == {"SPY": 10}
    assert ledger.history() == history
    assert order_ledger.history() == ()


def test_orders_are_stably_sorted_within_each_side():
    ledger = PortfolioLedger(1000)
    hold(ledger, "SPY", 10, 50)
    hold(ledger, "QQQ", 20, 25)
    events = (
        market_event("SPY", 50),
        market_event("IWM", 20),
        market_event("QQQ", 25),
        market_event("DIA", 100),
    )
    targets = [
        target("SPY", 0.1), target("IWM", 0.4),
        target("QQQ", 0.1), target("DIA", 0.4),
    ]

    portfolio = ledger.snapshot()
    context = SessionContext(SESSION, events)
    orders = plan_orders(targets, portfolio, context)

    sizes = [(order.symbol, order.side, order.quantity) for order in orders]
    assert sizes == [
        ("QQQ", OrderSide.SELL, 16),
        ("SPY", OrderSide.SELL, 8),
        ("DIA", OrderSide.BUY, 4),
        ("IWM", OrderSide.BUY, 20),
    ]
    assert [order.order_id for order in orders] == [
        "2025-01-06:QQQ", "2025-01-06:SPY", "2025-01-06:DIA", "2025-01-06:IWM",
    ]
    other_context = SessionContext(SESSION, events[::-1])
    assert plan_orders(targets[::-1], portfolio, other_context) == orders


def test_satisfied_tiny_and_unheld_zero_targets_are_noops():
    ledger = PortfolioLedger(200)
    hold(ledger, "SPY", 5, 20)
    context = SessionContext(
        SESSION, (market_event("SPY", 20), market_event("QQQ", 100))
    )

    assert plan_orders(
        [target("SPY", 0.5), target("QQQ", 0.05), target("IWM", 0)],
        ledger.snapshot(),
        context,
    ) == ()


def test_decimal_floor_and_id_are_independent_of_ambient_precision():
    portfolio = PortfolioLedger(100).snapshot()
    context = SessionContext(SESSION, (market_event("SPY", 30),))

    with localcontext() as ambient:
        ambient.prec = 2
        orders = plan_orders([target("SPY", 1)], portfolio, context)

    assert orders[0].quantity == 3
    assert orders == plan_orders([target("SPY", 1)], portfolio, context)

    later = SessionContext(LATER, (market_event("SPY", 30, session=LATER),))
    other = plan_orders([SignalEvent("SPY", LATER, 1)], portfolio, later)
    assert other[0].order_id == "2025-01-07:SPY"
    assert other[0].order_id != orders[0].order_id


def test_rejects_aggregate_weight_above_one_without_rescaling():
    portfolio = PortfolioLedger(100).snapshot()
    context = SessionContext(
        SESSION, (market_event("SPY", 10), market_event("QQQ", 10))
    )
    targets = [target("SPY", 0.6), target("QQQ", 0.4000000000000001)]

    with pytest.raises(
        ValueError, match="aggregate target weights must not exceed 1"
    ):
        plan_orders(targets, portfolio, context)


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ([target("SPY", 0.5), target("SPY", 0.5)], "duplicate target"),
        ([target("SPY", 0.25), target("SPY", 0.5)], "duplicate target"),
        ([SignalEvent("SPY", EARLIER, 0.5)], "must match decision session"),
        ([SignalEvent("SPY", LATER, 0.5)], "must match decision session"),
        ([object()], "targets must contain SignalEvent"),
    ],
)
def test_rejects_invalid_or_conflicting_signals(targets, message):
    portfolio = PortfolioLedger(100).snapshot()
    context = SessionContext(SESSION, (market_event("SPY", 10),))

    with pytest.raises(ValueError, match=message):
        plan_orders(targets, portfolio, context)


def test_requires_a_current_close_for_every_holding_even_when_liquidating():
    ledger = PortfolioLedger(100)
    hold(ledger, "SPY", 5, 10)
    old = SessionContext(
        EARLIER, (market_event("SPY", 10, session=EARLIER),)
    )
    current = SessionContext(SESSION, (market_event("QQQ", 50),), old)

    with pytest.raises(
        ValueError, match="missing current-session reference close for SPY"
    ):
        plan_orders(
            [target("SPY", 0), target("QQQ", 1)], ledger.snapshot(), current
        )


def test_requires_a_current_close_for_a_positive_unheld_target():
    portfolio = PortfolioLedger(100).snapshot()
    context = SessionContext(SESSION, (market_event("QQQ", 10),))

    with pytest.raises(
        ValueError, match="missing current-session reference close for SPY"
    ):
        plan_orders([target("SPY", 0.5)], portfolio, context)


@pytest.mark.parametrize("session", [EARLIER, LATER])
def test_rejects_stale_or_future_observations_in_decision_context(session):
    portfolio = PortfolioLedger(100).snapshot()
    wrong_date = market_event("SPY", 10, session=session)
    context = SessionContext(SESSION, (wrong_date,))

    with pytest.raises(
        ValueError, match="observation must be from the decision session"
    ):
        plan_orders([target("SPY", 1)], portfolio, context)


def test_varying_future_open_does_not_change_the_earlier_plan():
    metadata = DatasetMetadata(
        "Synthetic", "SPY", PriceAdjustment.RAW,
        datetime(2025, 1, 8, tzinfo=UTC),
    )
    decision = market_event("SPY", 30, opening=20).bar
    portfolio = PortfolioLedger(100).snapshot()

    def run(future_open):
        planned = []

        def strategy(context):
            if context.session_date == SESSION:
                targets = [target("SPY", 0.9)]
                planned.extend(plan_orders(targets, portfolio, context))
                return targets
            return ()

        later = market_event("SPY", 40, opening=future_open, session=LATER).bar
        generate_signals([(metadata, (decision, later))], strategy)
        return tuple(planned)

    assert run(10) == run(400) == (
        MarketOrder("2025-01-06:SPY", "SPY", OrderSide.BUY, 3, SESSION),
    )


@pytest.mark.parametrize(
    "adjustment",
    [PriceAdjustment.SPLIT_ADJUSTED, PriceAdjustment.TOTAL_RETURN_ADJUSTED],
)
def test_rejects_adjusted_reference_prices(adjustment):
    state = PortfolioLedger(100).snapshot()
    context = SessionContext(
        SESSION, (market_event("SPY", 10, adjustment=adjustment),)
    )

    with pytest.raises(ValueError, match="planning requires RAW data"):
        plan_orders([target("SPY", 1)], state, context)


@pytest.mark.parametrize("close", [0, -1, float("nan"), float("inf"), True])
def test_rejects_invalid_reference_closes(close):
    portfolio = PortfolioLedger(100).snapshot()
    event = market_event("SPY", 10)
    invalid = replace(event, bar=replace(event.bar, close=close))
    context = SessionContext(SESSION, (invalid,))

    with pytest.raises(ValueError, match="reference close for 'SPY'"):
        plan_orders([target("SPY", 1)], portfolio, context)


def test_duplicate_reference_bars_are_rejected():
    portfolio = PortfolioLedger(100).snapshot()
    context = SessionContext(
        SESSION, (market_event("SPY", 10), market_event("SPY", 40))
    )

    with pytest.raises(ValueError, match="duplicate current-session bar"):
        plan_orders([target("SPY", 1)], portfolio, context)


@pytest.mark.parametrize(
    ("cash", "message"),
    [
        (Decimal(0), "portfolio equity must be finite and positive"),
        (Decimal(-1), "portfolio cash must be a finite nonnegative Decimal"),
        (Decimal("NaN"), "portfolio cash must be a finite nonnegative"),
        (Decimal("Infinity"), "portfolio cash must be a finite nonnegative"),
    ],
)
def test_rejects_nonpositive_or_invalid_equity(cash, message):
    portfolio = replace(PortfolioLedger(100).snapshot(), cash=cash)
    context = SessionContext(SESSION, (market_event("SPY", 10),))

    with pytest.raises(ValueError, match=message):
        plan_orders([target("SPY", 1)], portfolio, context)


def test_exact_decimal_share_boundary_keeps_the_last_share():
    portfolio = PortfolioLedger(Decimal("0.30")).snapshot()
    context = SessionContext(SESSION, (market_event("SPY", 0.1),))

    orders = plan_orders([target("SPY", 1)], portfolio, context)

    assert orders[0].quantity == 3


@pytest.mark.parametrize("quantity", [0, -1, True, 1.5])
def test_rejects_invalid_current_position_quantities(quantity):
    portfolio = replace(
        PortfolioLedger(100).snapshot(), positions={"SPY": quantity}
    )
    context = SessionContext(SESSION, (market_event("SPY", 10),))

    with pytest.raises(ValueError, match="position must be positive whole shares"):
        plan_orders([target("SPY", 0)], portfolio, context)
