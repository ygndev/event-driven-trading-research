from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import combinations

import pytest

from trading_research.daily_bars import DailyBar, DatasetMetadata, PriceAdjustment
from trading_research.mean_reversion import MeanReversionExitRejected
from trading_research.orders import OrderSide, OrderState
from trading_research import sensitivity
from trading_research.benchmark import BuyAndHoldEntryError
from trading_research.sensitivity import (
    CostScenarioFailed,
    run_transaction_cost_sensitivity,
)


METADATA = DatasetMetadata(
    "Synthetic", "SPY", PriceAdjustment.RAW, datetime(2025, 4, 1, tzinfo=UTC)
)
INITIAL_CASH = Decimal(1000)


def spy_bars(closes, *, opens):
    bars = []
    for observation, close in enumerate(closes, start=1):
        opening = opens.get(observation, close)
        session = date(2025, 1, 1) + timedelta(days=observation - 1)
        bars.append(
            DailyBar(
                "SPY", session, opening, max(opening, close),
                min(opening, close), close, 1000,
            )
        )
    return bars


def spy_warmup():
    closes = [90] * 63
    closes[21] = 120
    closes[58] = 100
    return closes


@pytest.fixture
def round_trip_bars():
    closes = spy_warmup() + [97, 99, 101, 103, 105, 104, 106] + [104] * 14 + [102, 106]
    return spy_bars(closes, opens={65: 96, 70: 101, 86: 110})


def test_scenario_prices_fees_equity_and_return_differences(round_trip_bars):
    results = run_transaction_cost_sensitivity(
        METADATA, round_trip_bars, initial_cash=INITIAL_CASH
    )
    # At the $97 decision close each trading portfolio orders ten shares.
    # Exit opens are $110 for momentum and $101 for mean reversion.
    expected = (
        ("A", 0, "0.00", ("96", "110", "101"), ("1140", "1050", "1100")),
        ("B", 0, "1.25", ("96", "110", "101"), ("1137.5", "1047.5", "1098.75")),
        (
            "C", 10, "1.25", ("96.096", "109.89", "100.899"),
            ("1135.44", "1045.53", "1097.79"),
        ),
        (
            "D", 25, "1.25", ("96.24", "109.725", "100.7475"),
            ("1132.35", "1042.575", "1096.35"),
        ),
        (
            "E", 50, "1.25", ("96.48", "109.45", "100.495"),
            ("1127.20", "1037.65", "1093.95"),
        ),
        (
            "F", 25, "5.00", ("96.24", "109.725", "100.7475"),
            ("1124.85", "1035.075", "1092.60"),
        ),
    )
    zero_cost_equities = (*map(Decimal, expected[0][-1]), INITIAL_CASH)
    dates = tuple(bar.session_date for bar in round_trip_bars[63:])
    for scenario, row in zip(results, expected, strict=True):
        name, bps, fee, prices, equities = row
        commission = Decimal(fee)
        buy_price, momentum_price, reversion_price = map(Decimal, prices)
        final_equities = (*map(Decimal, equities), INITIAL_CASH)
        comparison = scenario.comparison
        assert scenario.scenario == name
        assert comparison.adjustment_bps == bps
        assert comparison.commission == commission
        assert comparison.start_session == dates[0] == date(2025, 3, 5)
        assert comparison.end_session == dates[-1] == date(2025, 3, 27)
        assert comparison.initial_cash == INITIAL_CASH

        momentum, mean_reversion, holding, cash = (
            comparison.momentum, comparison.mean_reversion,
            comparison.buy_and_hold, comparison.cash,
        )
        runs = (momentum, mean_reversion, holding, cash)
        for run, final in zip(runs, final_equities, strict=True):
            assert tuple(day for day, _ in run.equity_path) == dates
            assert run.equity_path[0] == (dates[0], INITIAL_CASH)
            portfolio = run.backtest.sessions[0].portfolio
            assert portfolio.initial_cash == portfolio.cash == INITIAL_CASH
            assert portfolio.positions == {}
            assert run.equity_path[-1] == (dates[-1], final)
            assert run.metrics.total_return == (final - INITIAL_CASH) / INITIAL_CASH
            assert run.backtest.outstanding_orders == run.backtest.rejected_orders == ()

        bought, sold = momentum.backtest.transactions
        assert mean_reversion.backtest.transactions[0] == bought
        assert holding.backtest.transactions == (bought,)
        assert bought.fill.execution_session == round_trip_bars[64].session_date
        assert bought.fill.side is OrderSide.BUY
        assert bought.fill.quantity == 10
        assert bought.fill.price == buy_price
        assert bought.cash_after == INITIAL_CASH - 10 * buy_price - commission
        assert bought.commission == commission
        for sale, index, price in (
            (sold, 85, momentum_price),
            (mean_reversion.backtest.transactions[1], 69, reversion_price),
        ):
            assert sale.fill.execution_session == round_trip_bars[index].session_date
            assert sale.fill.side is OrderSide.SELL
            assert sale.fill.quantity == 10
            assert sale.fill.price == price
            assert sale.commission == commission

        assert [run.metrics.executed_orders for run in runs] == [2, 2, 1, 0]
        assert [run.metrics.total_commissions for run in runs] == [
            2 * commission, 2 * commission, commission, Decimal(0)
        ]
        assert [
            run.backtest.sessions[-1].portfolio.positions for run in runs
        ] == [{}, {}, {"SPY": 10}, {}]
        assert cash.equity_path == tuple((day, INITIAL_CASH) for day in dates)
        assert cash.backtest.order_history == ()
        assert cash.metrics.sharpe_ratio is None

        for difference, run, final, zero in zip(
            scenario.differences_vs_zero_cost, runs, final_equities,
            zero_cost_equities, strict=True,
        ):
            assert difference.portfolio == run.label
            assert difference.equity_difference_usd == final - zero
            assert difference.return_difference_pp == (final - zero) / 10
            assert difference.executed_order_change == 0
            assert difference.filled_share_change == 0

        assert {
            (difference.left, difference.right): difference.return_difference_pp
            for difference in comparison.pairwise_differences
        } == {
            (left.label, right.label): (left_equity - right_equity) / 10
            for (left, left_equity), (right, right_equity) in combinations(
                zip(runs, final_equities, strict=True), 2
            )
        }


def test_costs_change_whole_share_sizing_on_a_later_entry():
    bars = spy_bars(
        spy_warmup() + [97, 110, 105, 104, 103, 104, 104.8, 80],
        opens={65: 96, 70: 101, 71: 100},
    )
    a, _, _, d, _, _ = run_transaction_cost_sensitivity(
        METADATA, bars, initial_cash=INITIAL_CASH
    )

    zero = a.comparison.mean_reversion
    reference = d.comparison.mean_reversion
    assert zero.backtest.signals == reference.backtest.signals
    # Both reenter at the $104.80 close. Only A has the $1048 needed to size ten.
    for run, quantities, cash_at_decision, final_cash, final_equity in (
        (zero, (10, 10, 10), "1050", "50", "850"),
        (reference, (10, 10, 9), "1042.575", "139.075", "859.075"),
    ):
        assert tuple(fill.quantity for fill in run.backtest.fills) == quantities
        assert run.backtest.orders[-1].decision_session == bars[69].session_date
        assert run.backtest.sessions[-2].portfolio.cash == Decimal(cash_at_decision)
        assert run.backtest.sessions[-1].portfolio.cash == Decimal(final_cash)
        assert run.equity_path[-1][1] == Decimal(final_equity)

    difference = d.differences_vs_zero_cost[1]
    assert difference.portfolio == reference.label
    assert difference.equity_difference_usd == Decimal("9.075")
    assert difference.return_difference_pp == Decimal("0.9075")
    assert difference.executed_order_change == 0
    assert difference.filled_share_change == -1
    assert reference.metrics.total_commissions == Decimal("3.75")


def test_five_dollar_commission_reports_scenario_and_invalid_benchmark(monkeypatch):
    bars = spy_bars(spy_warmup() + [97, 99], opens={65: 99.3})
    completed = []
    compare = sensitivity.compare_spy_common_window

    def record(*args, **kwargs):
        comparison = compare(*args, **kwargs)
        completed.append(comparison)
        return comparison

    monkeypatch.setattr(sensitivity, "compare_spy_common_window", record)
    with pytest.raises(CostScenarioFailed, match=r"scenario F .*insufficient cash") as failure:
        run_transaction_cost_sensitivity(METADATA, bars, initial_cash=INITIAL_CASH)

    error = failure.value
    assert (error.scenario, error.adjustment_bps, error.commission) == (
        "F", Decimal(25), Decimal("5.00"),
    )
    cause = error.__cause__
    assert isinstance(cause, BuyAndHoldEntryError)
    assert cause.anchor == bars[63].session_date
    assert cause.status == "rejected"
    assert cause.entry.order.quantity == 10
    assert cause.entry.state is OrderState.REJECTED
    assert cause.entry.fill is None
    assert "insufficient cash" in cause.entry.rejection_reason

    assert len(completed) == 5
    for comparison in completed:
        for run in (
            comparison.momentum, comparison.mean_reversion, comparison.buy_and_hold,
        ):
            fill, = run.backtest.fills
            assert fill.quantity == 10
            assert run.backtest.rejected_orders == ()
    e_purchase, = completed[-1].momentum.backtest.transactions
    assert e_purchase.fill.price == Decimal("99.7965")
    assert e_purchase.cash_after == Decimal("0.785")


def test_pending_exit_stays_outstanding_under_all_cost_scenarios(round_trip_bars):
    results = run_transaction_cost_sensitivity(
        METADATA, round_trip_bars[:69], initial_cash=INITIAL_CASH
    )

    for scenario in results:
        comparison = scenario.comparison
        mean_reversion = comparison.mean_reversion
        outstanding, = mean_reversion.backtest.outstanding_orders
        assert outstanding is comparison.outstanding_exit
        assert outstanding.order.side is OrderSide.SELL
        assert outstanding.order.decision_session == comparison.end_session
        assert outstanding.state is OrderState.ACCEPTED
        assert outstanding.fill is None
        assert mean_reversion.metrics.executed_orders == 1
        assert mean_reversion.metrics.total_commissions == comparison.commission
        assert mean_reversion.backtest.sessions[-1].portfolio.positions == {"SPY": 10}


def test_failed_exit_reports_the_cost_scenario_and_preserves_the_cause(round_trip_bars):
    bars = round_trip_bars[:70]
    bars[-1] = replace(bars[-1], open=0.25, low=0.25)

    with pytest.raises(CostScenarioFailed, match=r"scenario F .*25 bps") as failure:
        run_transaction_cost_sensitivity(METADATA, bars, initial_cash=INITIAL_CASH)

    error = failure.value
    assert (error.scenario, error.adjustment_bps, error.commission) == (
        "F", Decimal(25), Decimal("5.00")
    )
    assert isinstance(error.__cause__, MeanReversionExitRejected)
    rejection = error.__cause__.rejection
    assert rejection.order.side is OrderSide.SELL
    assert rejection.order.decision_session == bars[-2].session_date
    assert rejection.state is OrderState.REJECTED
    assert rejection.fill is None
    assert "commission exceeds proceeds" in rejection.rejection_reason


def test_validation_errors_propagate(round_trip_bars):
    adjusted = replace(METADATA, adjustment=PriceAdjustment.TOTAL_RETURN_ADJUSTED)

    with pytest.raises(ValueError, match="backtest requires RAW data"):
        run_transaction_cost_sensitivity(
            adjusted, round_trip_bars, initial_cash=INITIAL_CASH
        )


def test_analysis_is_deterministic_immutable_and_preserves_input(round_trip_bars):
    original = [replace(bar) for bar in round_trip_bars]

    results = run_transaction_cost_sensitivity(
        METADATA, round_trip_bars, initial_cash=INITIAL_CASH
    )

    assert results == run_transaction_cost_sensitivity(
        METADATA, round_trip_bars, initial_cash=INITIAL_CASH
    )
    assert isinstance(results, tuple)
    assert round_trip_bars == original
    with pytest.raises(FrozenInstanceError):
        setattr(results[0], "scenario", "changed")
    with pytest.raises(FrozenInstanceError):
        setattr(results[0].differences_vs_zero_cost[0], "filled_share_change", 1)
