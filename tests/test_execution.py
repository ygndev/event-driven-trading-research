from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime
from decimal import Decimal, Inexact, localcontext

import pytest

from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.execution import NextOpenExecution
from trading_research.orders import Fill, MarketOrder, OrderLedger, OrderSide
from trading_research.portfolio import PortfolioLedger

FRIDAY = date(2025, 1, 3)
MONDAY = date(2025, 1, 6)
TUESDAY = date(2025, 1, 7)
WEDNESDAY = date(2025, 1, 8)
THURSDAY = date(2025, 1, 9)


@pytest.fixture
def metadata():
    return DatasetMetadata(
        source="Synthetic",
        symbol="SPY",
        adjustment=PriceAdjustment.RAW,
        retrieved_at=datetime(2025, 1, 10, tzinfo=UTC),
    )


@pytest.fixture
def order():
    return MarketOrder("order-1", "SPY", OrderSide.BUY, 3, FRIDAY)


def bar(session, opening, *, symbol="SPY", close=None):
    close = opening if close is None else close
    return DailyBar(
        symbol, session, opening, max(opening, close), min(opening, close),
        close, 100,
    )


@pytest.mark.parametrize(
    ("side", "expected"),
    [(OrderSide.BUY, "110.275"), (OrderSide.SELL, "109.725")],
)
def test_prices_both_sides_at_the_first_eligible_open(
    order, metadata, side, expected
):
    order = replace(order, side=side)
    data = [
        bar(FRIDAY, 98, close=100),
        bar(MONDAY, 110, close=125),
        bar(TUESDAY, 90),
    ]
    model = NextOpenExecution(adjustment_bps=25, commission=1.25)

    result = model.propose(order, data, metadata)

    assert result.fill == Fill(
        "order-1", "SPY", MONDAY, side, 3, Decimal(expected)
    )
    assert result.opening_price == Decimal("110")
    assert result.adjustment_bps == Decimal("25")
    assert result.commission == Decimal("1.25")


def test_zero_adjustment_keeps_the_overnight_opening_gap(order, metadata):
    friday = bar(FRIDAY, 98, close=100)
    monday = bar(MONDAY, 110, close=108)
    model = NextOpenExecution(adjustment_bps=0, commission=0)

    result = model.propose(order, [friday, monday], metadata)

    assert result.fill.execution_session == MONDAY
    assert result.fill.price == Decimal("110")
    assert result.fill.price - Decimal(friday.close) == Decimal("10")
    assert result.commission == 0


@pytest.mark.parametrize(
    "data",
    [[], [bar(FRIDAY, 110)], [bar(date(2025, 1, 2), 99), bar(FRIDAY, 110)]],
)
def test_missing_eligible_observation_leaves_order_available(
    order, metadata, data
):
    model = NextOpenExecution(adjustment_bps=0, commission=0)

    assert model.propose(order, data, metadata) is None

    result = model.propose(order, [*data, bar(THURSDAY, 112)], metadata)
    assert result.fill.execution_session == THURSDAY
    assert result.fill.price == Decimal("112")


def test_honors_eligibility_bound_with_or_without_a_bar_on_that_date(
    order, metadata
):
    delayed = replace(order, earliest_execution_session=WEDNESDAY)
    before = [bar(MONDAY, 102), bar(TUESDAY, 105)]
    model = NextOpenExecution(adjustment_bps=0, commission=0)

    assert model.propose(delayed, before, metadata) is None

    result = model.propose(delayed, [*before, bar(THURSDAY, 120)], metadata)
    assert result.fill.execution_session == THURSDAY
    assert result.fill.price == Decimal("120")
    at_bound = NextOpenExecution(adjustment_bps=0, commission=0)
    on_bound = [*before, bar(WEDNESDAY, 113)]
    result = at_bound.propose(delayed, on_bound, metadata)
    assert result.fill.execution_session == WEDNESDAY


def test_rejects_wrong_dataset_and_wrong_bar_symbol(order, metadata):
    model = NextOpenExecution(adjustment_bps=0, commission=0)
    qqq = bar(MONDAY, 110, symbol="QQQ")

    with pytest.raises(
        ValueError, match="does not match dataset symbol 'QQQ'"
    ):
        model.propose(order, [qqq], replace(metadata, symbol="QQQ"))
    with pytest.raises(ValueError, match="expected symbol 'SPY', got 'QQQ'"):
        model.propose(order, [qqq], metadata)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([bar(TUESDAY, 110), bar(MONDAY, 100)], "out of order"),
        ([bar(MONDAY, 110), bar(MONDAY, 115)], "duplicate session"),
        (
            [replace(bar(MONDAY, 110), volume=-1), bar(TUESDAY, 115)],
            "volume must",
        ),
        ([replace(bar(MONDAY, 110), open=0), bar(TUESDAY, 115)], "open must"),
    ],
)
def test_rejects_malformed_history_without_skipping_rows(
    order, metadata, data, message
):
    model = NextOpenExecution(adjustment_bps=0, commission=0)

    with pytest.raises(ValueError, match=message):
        model.propose(order, data, metadata)


@pytest.mark.parametrize("field", ["adjustment_bps", "commission"])
@pytest.mark.parametrize(
    "value",
    [-1, True, False, float("nan"), float("inf"), Decimal("-Infinity"), "1"],
)
def test_rejects_invalid_adjustment_and_commission(field, value):
    config = {"adjustment_bps": 0, "commission": 0}
    config[field] = value
    with pytest.raises(ValueError, match=field):
        NextOpenExecution(**config)


@pytest.mark.parametrize("bps", [10000, 12500])
def test_rejects_nonpositive_sell_execution_price(order, metadata, bps):
    model = NextOpenExecution(adjustment_bps=bps, commission=0)

    sell = replace(order, side=OrderSide.SELL)
    with pytest.raises(ValueError, match="price must be a finite, positive"):
        model.propose(sell, [bar(MONDAY, 100)], metadata)


@pytest.mark.parametrize(
    "price", [Decimal("NaN"), Decimal("Infinity"), Decimal(0)]
)
def test_fill_rejects_invalid_decimal_prices(price):
    with pytest.raises(ValueError, match="price must be a finite, positive"):
        Fill("order-1", "SPY", MONDAY, OrderSide.BUY, 3, price)


def test_proposes_an_order_id_only_once_per_run(order, metadata):
    model = NextOpenExecution(adjustment_bps=0, commission=0)
    initial = model.propose(order, [bar(MONDAY, 110)], metadata)
    other_order = replace(order, order_id="order-2")

    same_id = replace(order, side=OrderSide.SELL)
    with pytest.raises(ValueError, match="order 'order-1' was already proposed"):
        model.propose(same_id, [bar(TUESDAY, 115)], metadata)

    assert initial.fill.execution_session == MONDAY
    other = model.propose(other_order, [bar(TUESDAY, 115)], metadata)
    assert other.fill.order_id == "order-2"
    assert other.fill.execution_session == TUESDAY


@pytest.mark.parametrize("state", ["created", "rejected", "filled"])
def test_rejects_order_snapshots_without_acceptance(order, metadata, state):
    ledger = OrderLedger()
    ledger.register(order)
    if state == "rejected":
        ledger.reject(order.order_id, "Invalid order")
    elif state == "filled":
        ledger.accept(order.order_id)
        ledger.record_fill(
            Fill("order-1", "SPY", MONDAY, OrderSide.BUY, 3, 110)
        )
    history = ledger.history()
    model = NextOpenExecution(adjustment_bps=0, commission=0)

    snapshot = ledger.snapshot(order.order_id)
    with pytest.raises(ValueError, match="requires an ACCEPTED order snapshot"):
        model.propose(snapshot, [bar(TUESDAY, 120)], metadata)

    assert ledger.history() == history


@pytest.mark.parametrize(
    "adjustment",
    [PriceAdjustment.SPLIT_ADJUSTED, PriceAdjustment.TOTAL_RETURN_ADJUSTED],
)
def test_rejects_adjusted_market_data(order, metadata, adjustment):
    model = NextOpenExecution(adjustment_bps=0, commission=0)

    adjusted = replace(metadata, adjustment=adjustment)
    with pytest.raises(ValueError, match="data declared RAW"):
        model.propose(order, [bar(MONDAY, 110)], adjusted)


def test_proposal_is_immutable_and_does_not_book_any_changes(order, metadata):
    orders = OrderLedger()
    orders.register(order)
    orders.accept(order.order_id)
    accepted = orders.snapshot(order.order_id)
    history = orders.history()
    portfolio = PortfolioLedger(1000)
    before = portfolio.snapshot()
    monday = bar(MONDAY, 110)
    data = [monday]
    model = NextOpenExecution(adjustment_bps=25, commission=1.25)

    result = model.propose(accepted, data, metadata)

    assert result.fill.price == Decimal("110.275")
    assert accepted.order is order
    assert order == MarketOrder("order-1", "SPY", OrderSide.BUY, 3, FRIDAY)
    assert data == [DailyBar("SPY", MONDAY, 110, 110, 110, 110, 100)]
    assert orders.snapshot(order.order_id) is accepted
    assert orders.history() == history
    assert portfolio.snapshot() == before
    assert portfolio.history() == ()
    with pytest.raises(FrozenInstanceError):
        setattr(result, "commission", Decimal(0))
    with pytest.raises(FrozenInstanceError):
        setattr(result.fill, "price", Decimal(100))


def test_repeated_runs_retain_exact_decimal_price_and_fee(order, metadata):
    data = [bar(MONDAY, 0.1)]

    def run():
        model = NextOpenExecution(
            adjustment_bps=2.5, commission=Decimal("0.02")
        )
        return model.propose(order, data, metadata)

    result = run()
    with localcontext() as context:
        context.prec = 2
        assert run() == result
    assert isinstance(result.fill.price, Decimal)
    assert result.fill.price == Decimal("0.100025")
    assert result.commission == Decimal("0.02")

    # Only the caller books the separately supplied price and commission.
    orders = OrderLedger()
    orders.register(order)
    orders.accept(order.order_id)
    orders.record_fill(result.fill)
    portfolio = PortfolioLedger(10)
    portfolio.apply_fill(result.fill, result.commission)

    assert portfolio.snapshot().cash == Decimal("9.679925")
    assert portfolio.snapshot().cumulative_commissions == Decimal("0.02")


def test_inexact_price_does_not_consume_the_order(order, metadata):
    model = NextOpenExecution(adjustment_bps=1, commission=0)
    oversized = bar(MONDAY, 10**34 - 1)

    with pytest.raises(Inexact):
        model.propose(order, [oversized], metadata)

    result = model.propose(order, [bar(MONDAY, 100)], metadata)
    assert result.fill.price == Decimal("100.01")
