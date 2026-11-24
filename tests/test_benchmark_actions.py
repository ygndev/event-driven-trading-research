from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

import pytest

from trading_research import benchmark
from trading_research.backtest import PendingOrderSplitError
from trading_research.benchmark import (
    BuyAndHoldEntryError, compare_spy_benchmarks,
    compare_spy_common_window, compare_spy_mean_reversion_benchmarks,
)
from trading_research.chronology import ResearchPeriod, evaluate_spy_research_periods
from trading_research.corporate_actions import CashDividend, StockSplit
from trading_research.daily_bars import DailyBar, DatasetMetadata, PriceAdjustment
from trading_research.signals import SignalEvent

META = DatasetMetadata("Synthetic", "SPY", PriceAdjustment.RAW, datetime(2026, 1, 1, tzinfo=UTC))
COSTS = dict(initial_cash=1000, adjustment_bps=0, commission=0)


def tape(count=70):
    return tuple(
        DailyBar("SPY", date(2020, 1, 1) + timedelta(days=i), 100, 100, 100, 100, 1000)
        for i in range(count)
    )


def cash(bars, ex=66, pay=68, amount=1, action_id="dividend"):
    return CashDividend(action_id, "SPY", bars[ex].session_date,
                        bars[pay].session_date if pay < len(bars) else bars[-1].session_date + timedelta(days=pay),
                        amount)


def comparison(bars, actions=(), **costs):
    return compare_spy_common_window(META, bars, corporate_actions=actions, **(COSTS | costs))


def portfolios(result):
    return result.momentum, result.mean_reversion, result.buy_and_hold, result.cash


@pytest.mark.parametrize("api", [
    compare_spy_benchmarks, compare_spy_mean_reversion_benchmarks, compare_spy_common_window,
])
def test_optional_actions_preserve_price_only_results_and_pay_holding(api):
    bars = tape()
    assert api(META, bars, **COSTS) == api(META, bars, corporate_actions=(), **COSTS)
    result = api(META, bars, corporate_actions=(cash(bars),), **COSTS)
    holding = result.buy_and_hold.backtest
    assert holding.sessions[-1].portfolio.positions == {"SPY": 10}
    assert holding.sessions[-1].portfolio.cash == 10
    assert holding.sessions[-1].equity == 1010
    assert holding.dividend_payments[0].receivable.entitled_shares == 10
    assert all(s.portfolio.cash == 1000 and not s.portfolio.positions
               and not s.portfolio.unpaid_receivables for s in result.cash.backtest.sessions)
    assert result.cash.metrics.total_return == 0


def test_exact_cash_receivable_and_post_window_payment_arithmetic():
    bars = tape()
    action = cash(bars, pay=80, amount=Decimal("1.25"))
    result = comparison(bars, (action,))
    holding = result.buy_and_hold.backtest
    ex = next(s for s in holding.sessions if s.session_date == action.ex_date)
    assert ex.portfolio.cash == 0
    claim, = ex.portfolio.unpaid_receivables
    assert claim.entitled_shares == 10 and claim.total_amount == Decimal("12.50")
    assert claim.prior_close_session == bars[65].session_date
    assert ex.equity == holding.sessions[-1].equity == Decimal("1012.50")
    assert result.buy_and_hold.metrics.total_return == Decimal("0.0125")
    payment, = holding.dividend_payments
    assert payment.session_date == action.payment_date
    changed = comparison(bars, (replace(action, payment_date=action.payment_date + timedelta(days=20)),))
    for before, after in zip(portfolios(result), portfolios(changed)):
        assert before.metrics == after.metrics
        assert before.backtest.signals == after.backtest.signals
        assert before.backtest.transactions == after.backtest.transactions
        assert [(s.equity, s.portfolio.cash, dict(s.portfolio.positions))
                for s in before.backtest.sessions] == [
                    (s.equity, s.portfolio.cash, dict(s.portfolio.positions))
                    for s in after.backtest.sessions]


def test_ex_date_purchase_has_no_entitlement():
    bars = tape()
    result = comparison(bars, (cash(bars, ex=64),))
    assert result.buy_and_hold.backtest.dividend_payments == ()
    assert result.buy_and_hold.backtest.sessions[-1].equity == 1000


@pytest.mark.parametrize(("ratio", "new_price", "shares"), [
    (Fraction(2), 50, 20), (Fraction(1, 2), 200, 5),
])
def test_split_changes_units_without_wealth(ratio, new_price, shares):
    bars = tape()
    bars = bars[:66] + tuple(replace(b, open=new_price, high=new_price, low=new_price, close=new_price)
                            for b in bars[66:])
    result = comparison(bars, (StockSplit("SPY", bars[66].session_date, ratio),))
    final = result.buy_and_hold.backtest.sessions[-1]
    assert final.portfolio.positions == {"SPY": shares}
    assert final.portfolio.cash == 0 and final.equity == 1000
    assert result.buy_and_hold.metrics.total_return == 0


def test_fractional_split_and_pending_split_keep_existing_failures():
    bars = tape()
    with pytest.raises(ValueError, match="fractional"):
        comparison(bars, (StockSplit("SPY", bars[66].session_date, Fraction(1, 3)),))
    with pytest.raises(PendingOrderSplitError):
        comparison(bars, (StockSplit("SPY", bars[64].session_date, 2),))


def test_identical_raw_tape_costs_actions_and_anchor_for_every_run(monkeypatch):
    bars = tape()
    actions = (cash(bars),)
    calls = []
    real = benchmark.run_backtest

    def record(datasets, strategy, **kwargs):
        calls.append((datasets, kwargs))
        return real(datasets, strategy, **kwargs)

    monkeypatch.setattr(benchmark, "run_backtest", record)
    bars = bars[:64] + (replace(bars[64], open=99, low=99),) + bars[65:]
    result = comparison(bars, iter(actions), adjustment_bps=10, commission=1)
    assert len(calls) == 4
    for datasets, kwargs in calls:
        assert datasets == ((META, bars),)
        assert kwargs == dict(initial_cash=1000, adjustment_bps=10, commission=1, corporate_actions=actions)
    for run in portfolios(result):
        assert run.equity_path[0] == (bars[63].session_date, 1000)
        assert run.equity_path[-1][0] == bars[-1].session_date
        assert all(s.decision_session >= bars[63].session_date for s in run.backtest.signals)
        assert all(f.execution_session > bars[63].session_date for f in run.backtest.fills)


def test_held_strategies_and_holding_receive_the_same_dividend(monkeypatch):
    bars = tape()

    def enter(context):
        return (SignalEvent("SPY", context.session_date, 1),) if context.session_date == bars[63].session_date else ()

    monkeypatch.setattr(benchmark, "time_series_momentum", enter)
    monkeypatch.setattr(benchmark, "short_term_mean_reversion", enter)
    result = comparison(bars, (cash(bars),))
    for run in (result.momentum, result.mean_reversion, result.buy_and_hold):
        assert run.backtest.sessions[-1].equity == 1010
        assert run.backtest.dividend_payments[0].receivable.total_amount == 10


@pytest.mark.parametrize("actions", [
    (object(),),
    (StockSplit("QQQ", date(2030, 1, 1), 2),),
    (StockSplit("SPY", date(2020, 3, 7), 2),) * 2,
    (CashDividend("a", "SPY", date(2020, 3, 7), date(2020, 3, 8), 1),) * 2,
    (CashDividend("first", "SPY", date(2020, 1, 1), date(2020, 1, 2), 1),),
    (StockSplit("SPY", date(2019, 12, 31), 2),),
])
def test_invalid_relevant_or_future_identity_actions_fail(actions):
    with pytest.raises(ValueError):
        comparison(tape(), actions)


def period(bars, start=63, end=69, name="test"):
    return ResearchPeriod(name, bars[start].session_date, bars[end].session_date)


def evaluate(bars, actions=(), periods=None, **kwargs):
    return evaluate_spy_research_periods(
        META, bars, corporate_actions=actions, periods=periods or (period(bars),), **COSTS, **kwargs,
    )


def test_missing_preperiod_warmup_rejected_with_explicit_synthetic_opt_out():
    bars = tape()
    window = period(bars, 61)
    with pytest.raises(ValueError, match="before the first observed"):
        evaluate(bars, periods=(window,))
    delayed, = evaluate(bars, periods=(window,), strict_warmup=False)
    assert delayed.comparison.start_session == bars[63].session_date


def test_independent_periods_preserve_receivable_and_discard_future_events():
    bars = tape(80)
    windows = (period(bars), period(bars, 70, 79, "second"))
    action = cash(bars, ex=66, pay=75)
    results = evaluate(bars, (action,), windows)
    first, second = (r.comparison for r in results)
    assert first.buy_and_hold.backtest.sessions[-1].equity == 1010
    assert first.buy_and_hold.backtest.sessions[-1].portfolio.unpaid_receivables
    assert second.buy_and_hold.backtest.dividend_payments == ()
    for row in results:
        for run in portfolios(row.comparison):
            anchor = run.backtest.sessions[0].portfolio
            assert anchor.cash == anchor.initial_cash == 1000
            assert not anchor.positions and not anchor.unpaid_receivables
            assert all(o.decision_session >= row.comparison.start_session for o in run.backtest.orders)
    original, = evaluate(bars, (action,), windows[:1])
    later = (
        StockSplit("SPY", bars[74].session_date, 2),
        cash(bars, ex=76, pay=78, action_id="future"),
    )
    assert evaluate(bars, (action, *later), windows[:1]) == (original,)


def test_future_events_outside_tape_do_not_change_comparison():
    bars = tape()
    actions = (StockSplit("SPY", date(2030, 1, 1), 2),
               CashDividend("later", "SPY", date(2030, 2, 1), date(2030, 2, 2), 2))
    assert comparison(bars, actions) == comparison(bars)


def test_warmup_split_reaches_indicator_history_only(monkeypatch):
    bars = tape()
    bars = tuple(replace(b, open=50, high=50, low=50, close=50) if i >= 60 else b
                 for i, b in enumerate(bars))
    observed = []
    real = benchmark.time_series_momentum

    def inspect(context):
        observed.append(context.strategy_close(context.history("SPY")[0]))
        return real(context)

    monkeypatch.setattr(benchmark, "time_series_momentum", inspect)
    result, = evaluate(bars, (StockSplit("SPY", bars[60].session_date, 2),))
    assert observed[0] == 50
    assert result.comparison.momentum.backtest.sessions[0].portfolio.cash == 1000


def test_buy_and_hold_failed_entry_keeps_its_error_with_actions():
    bars = tape()
    bars = bars[:64] + (replace(bars[64], open=101, high=101),) + bars[65:]
    with pytest.raises(BuyAndHoldEntryError):
        comparison(bars, (cash(bars),))


def test_unpaid_claim_cannot_fund_the_strategy_purchase(monkeypatch):
    bars = tape()

    def target(context):
        if context.session_date == bars[63].session_date:
            return (SignalEvent("SPY", context.session_date, 0.5),)
        if context.session_date == bars[65].session_date:
            return (SignalEvent("SPY", context.session_date, 1),)
        return ()

    monkeypatch.setattr(benchmark, "time_series_momentum", target)
    action = cash(bars, ex=65, pay=80, amount=100)
    result = comparison(bars, (action,))
    run = result.momentum.backtest
    assert [fill.quantity for fill in run.fills] == [5, 5]
    assert not run.rejected_orders
    final = run.sessions[-1]
    assert final.portfolio.cash == 0
    assert final.portfolio.positions == {"SPY": 10}
    assert final.portfolio.unpaid_receivables[0].total_amount == 500
    assert final.equity == 1500
    changed = comparison(bars, (replace(action, payment_date=action.payment_date + timedelta(days=20)),))
    assert changed.momentum.metrics == result.momentum.metrics
    assert changed.momentum.backtest.transactions == run.transactions


def test_action_requires_observed_session_even_during_warmup():
    bars = tape()
    missing = bars[10].session_date
    with pytest.raises(ValueError, match="observed session"):
        comparison(bars[:10] + bars[11:], (StockSplit("SPY", missing, 2),))


def test_invalid_ratio_is_revalidated_without_mutating_supplied_action():
    bars = tape()
    split = StockSplit("SPY", bars[66].session_date, 2)
    object.__setattr__(split, "ratio", 0)
    with pytest.raises(ValueError, match="split ratio"):
        comparison(bars, (split,))
    assert split.ratio == 0


@pytest.fixture
def april_history():
    # Synthetic observations, with a deliberate future gap on April 14.
    return tuple(bar for bar in tape(115) if bar.session_date != date(2020, 4, 14))


def april_actions():
    return (
        StockSplit("SPY", date(2020, 4, 14), 2),
        CashDividend("future-april", "SPY", date(2020, 4, 14), date(2020, 4, 20), 1),
    )


@pytest.mark.parametrize("indices", [(0,), (1,), (0, 1)])
def test_future_action_missing_its_bar_cannot_block_april_7(april_history, indices):
    window = ResearchPeriod("early", date(2020, 4, 1), date(2020, 4, 7))
    actions = tuple(april_actions()[i] for i in indices)
    expected = evaluate(april_history, periods=(window,))
    actual = evaluate(april_history, iter(actions), periods=(window,))
    assert actual == expected
    for run in portfolios(actual[0].comparison):
        assert not run.backtest.split_adjustments
        assert not run.backtest.dividend_payments
        assert all(not row.portfolio.unpaid_receivables for row in run.backtest.sessions)


@pytest.mark.parametrize("action", april_actions())
def test_missing_action_bar_inside_selected_period_still_fails(april_history, action):
    window = ResearchPeriod("relevant", date(2020, 4, 1), date(2020, 4, 20))
    with pytest.raises(ValueError, match="requires an observed session"):
        evaluate(april_history, (action,), periods=(window,))


@pytest.mark.parametrize("action", april_actions())
def test_cutoff_does_not_hide_global_duplicate_identities(april_history, action):
    window = ResearchPeriod("early", date(2020, 4, 1), date(2020, 4, 7))
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        evaluate(april_history, (action, action), periods=(window,))


def test_each_period_filters_before_session_validation_and_keeps_actual_pay_date(april_history):
    windows = (
        ResearchPeriod("early", date(2020, 4, 1), date(2020, 4, 7)),
        ResearchPeriod("later", date(2020, 4, 15), date(2020, 4, 22)),
    )
    action = CashDividend("in-window", "SPY", date(2020, 4, 6), date(2020, 5, 1), Decimal("1.25"))
    # A missing action session beyond every selected window is irrelevant to both.
    future = StockSplit("SPY", date(2020, 4, 23), 2)
    bars = tuple(bar for bar in april_history if bar.session_date != future.effective_session)
    expected = evaluate(bars, (action,), periods=windows)
    actual = evaluate(bars, (action, future), periods=windows)
    assert actual == expected
    early, later = (row.comparison for row in actual)
    holding = early.buy_and_hold.backtest
    assert holding.sessions[-1].equity == Decimal("1012.50")
    claim, = holding.sessions[-1].portfolio.unpaid_receivables
    assert claim.total_amount == Decimal("12.50") and claim.entitled_shares == 10
    payment, = holding.dividend_payments
    assert payment.session_date == action.payment_date
    for run in portfolios(later):
        anchor = run.backtest.sessions[0].portfolio
        assert anchor.cash == 1000 and not anchor.positions and not anchor.unpaid_receivables
    assert later.buy_and_hold.backtest.dividend_payments == ()


@pytest.mark.parametrize("api", [
    compare_spy_benchmarks, compare_spy_mean_reversion_benchmarks, compare_spy_common_window,
])
def test_malformed_final_bar_uses_same_value_error_with_and_without_actions(api):
    bars = tape()
    malformed = (*bars[:-1], object())
    errors = []
    for actions in ((), (cash(bars),)):
        with pytest.raises(ValueError, match="bar 70: expected a DailyBar") as error:
            api(META, malformed, corporate_actions=actions, **COSTS)
        errors.append(str(error.value))
    assert errors[0] == errors[1]
