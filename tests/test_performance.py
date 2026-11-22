from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime
from decimal import Decimal, localcontext

import pytest

from trading_research.backtest import (
    BacktestResult,
    SessionValuation,
    run_backtest,
)
from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
)
from trading_research.orders import Fill, OrderSide
from trading_research.performance import evaluate_performance
from trading_research.portfolio import PortfolioLedger, PortfolioSnapshot
from trading_research.signals import SignalEvent

CAPITAL_START = date(2025, 1, 2)
FIRST = date(2025, 1, 3)
SECOND = date(2025, 1, 6)
THIRD = date(2025, 1, 8)


def dataset(symbol, *observations):
    metadata = DatasetMetadata(
        "Synthetic", symbol, PriceAdjustment.RAW,
        datetime(2025, 1, 9, tzinfo=UTC),
    )
    bars = tuple(
        DailyBar(
            symbol, session, opening, max(opening, close),
            min(opening, close), close, 1_000,
        )
        for session, opening, close in observations
    )
    return metadata, bars


def run(datasets, strategy, commission=0):
    return run_backtest(
        datasets, strategy, initial_cash=100, adjustment_bps=0,
        commission=commission,
    )


def equity_path(equities, dates=(FIRST, SECOND, THIRD)):
    # A fixed share and cash balance isolate the effect of changing marks.
    portfolio = PortfolioSnapshot(
        Decimal(100), Decimal(50), {"SPY": 1}, Decimal(0)
    )
    sessions = tuple(
        SessionValuation(session, portfolio, Decimal(equity))
        for session, equity in zip(dates, equities, strict=True)
    )
    return BacktestResult(sessions, (), (), ())


def near(expected):
    return pytest.approx(Decimal(expected), rel=Decimal("1e-12"))


@pytest.mark.parametrize("valuation_count", [1, 2, 3])
def test_cash_volatility_requires_two_elapsed_intervals(valuation_count):
    bars = [(FIRST, 10, 12), (SECOND, 11, 10), (THIRD, 8, 9)][:valuation_count]
    spy = dataset("SPY", *bars)
    report = evaluate_performance(run([spy], lambda context: ()))

    assert [entry.session_date for entry in report.session_returns] == [
        day for day, _, _ in bars[1:]
    ]
    returns = [entry.simple_return for entry in report.session_returns]
    assert returns == [0] * (valuation_count - 1)
    assert report.total_return == report.max_drawdown == report.turnover == 0
    assert report.average_long_exposure == 0
    if valuation_count < 3:
        assert report.annualized_volatility is None
    else:
        assert report.annualized_volatility == 0
    assert report.total_commissions == report.executed_notional == 0
    assert report.executed_orders == 0
    assert report.sharpe_ratio is None
    assert report.risk_free_rate_per_session == 0
    assert report.annualized_return is None
    assert "explicit start and end dates" in report.annualized_return_reason


@pytest.mark.parametrize(
    ("sale_open", "cash", "total_return", "drawdown", "notional", "turnover"),
    [
        (12, "107.50", "0.075", "0.0025", "88", "0.8592351505288853"),
        (8, "91.50", "-0.085", "0.085", "72", "0.7416309012875536"),
    ],
)
def test_profitable_and_losing_round_trips(
    sale_open, cash, total_return, drawdown, notional, turnover
):
    spy = dataset(
        "SPY", (FIRST, 10, 10), (SECOND, 10, 10), (THIRD, sale_open, 11)
    )

    def strategy(context):
        if context.session_date == FIRST:
            return (SignalEvent("SPY", FIRST, 0.4),)
        if context.session_date == SECOND:
            return (SignalEvent("SPY", SECOND, 0),)
        return ()

    result = run([spy], strategy, commission=Decimal("0.25"))
    report = evaluate_performance(result)

    assert [fill.quantity for fill in result.fills] == [4, 4]
    assert result.sessions[1].equity == Decimal("99.75")
    assert result.sessions[-1].portfolio.cash == Decimal(cash)
    assert report.total_return == Decimal(total_return)
    assert report.max_drawdown == Decimal(drawdown)
    assert report.total_commissions == Decimal("0.50")
    assert report.executed_orders == 2
    assert report.executed_notional == Decimal(notional)
    assert report.turnover == near(turnover)
    assert report.average_long_exposure == near("0.1336675020885547")


def test_sample_volatility_and_sharpe_against_known_returns():
    # Returns +15%, +5%, -5% have mean 5% and sample deviation 10%.
    result = equity_path(["115", "120.75", "114.7125"])

    dates = {"start_date": CAPITAL_START, "end_date": THIRD}
    report = evaluate_performance(result, **dates)
    excess = evaluate_performance(result, risk_free_rate_per_session=0.02, **dates)
    with localcontext() as ambient:
        ambient.prec = 2
        assert evaluate_performance(result, **dates) == report

    assert [item.simple_return for item in report.session_returns] == [
        Decimal("0.15"), Decimal("0.05"), Decimal("-0.05")
    ]
    assert report.total_return == Decimal("0.147125")
    assert report.max_drawdown == Decimal("0.05")
    assert report.annualized_volatility == near("1.5874507866387544")
    assert report.sharpe_ratio == near("7.937253933193772")
    assert excess.annualized_volatility == report.annualized_volatility
    assert excess.sharpe_ratio == near("4.762352359916263")
    assert excess.risk_free_rate_per_session == Decimal("0.02")


def test_equal_nonzero_returns_have_zero_volatility_and_undefined_sharpe():
    report = evaluate_performance(equity_path(["110", "121", "133.1"]))

    assert [row.simple_return for row in report.session_returns] == [Decimal("0.10")] * 2
    assert report.total_return == Decimal("0.331")
    assert report.annualized_volatility == 0
    assert report.sharpe_ratio is None


def test_two_negative_elapsed_returns_keep_negative_sharpe():
    report = evaluate_performance(equity_path([100, 90, 72]))
    assert [row.simple_return for row in report.session_returns] == [
        Decimal("-0.10"), Decimal("-0.20"),
    ]
    # Mean -0.15 and sample variance 0.005 give these annualized values.
    assert report.annualized_volatility == near("1.1224972160321824")
    assert report.sharpe_ratio == near("-33.67491648096547")
    assert report.total_return == Decimal("-0.28")


def test_drawdown_restarts_from_new_peaks():
    dates = (FIRST, SECOND, THIRD, date(2025, 1, 9), date(2025, 1, 10))
    result = equity_path([120, 90, 150, 105, 130], dates)

    report = evaluate_performance(result)

    assert report.max_drawdown == Decimal("0.30")
    assert report.total_return == Decimal("0.30")
    returns_by_date = {row.session_date: row.simple_return for row in report.session_returns}
    assert tuple(returns_by_date) == dates[1:]
    assert returns_by_date[SECOND] == Decimal("-0.25")
    assert returns_by_date[dates[3]] == Decimal("-0.30")


def test_first_snapshot_drawdown_uses_initial_capital_as_the_peak():
    report = evaluate_performance(equity_path([70, 90, 80]))

    assert [row.session_date for row in report.session_returns] == [SECOND, THIRD]
    assert report.session_returns[0].simple_return == near("0.2857142857142857")
    assert report.max_drawdown == Decimal("0.30")
    assert report.total_return == Decimal("-0.20")


def test_same_day_anchor_has_total_return_and_drawdown_but_no_elapsed_return():
    result = equity_path([95], (FIRST,))

    report = evaluate_performance(result, start_date=FIRST, end_date=FIRST)

    assert report.session_returns == ()
    assert report.total_return == Decimal("-0.05")
    assert report.max_drawdown == Decimal("0.05")
    assert report.annualized_return is None
    assert report.annualized_return_reason == (
        "CAGR requires at least 365 days and two observations"
    )
    assert report.annualized_volatility is report.sharpe_ratio is None


def test_earlier_dated_initial_cash_can_start_one_elapsed_interval():
    result = equity_path([110], (FIRST,))

    report = evaluate_performance(result, start_date=CAPITAL_START, end_date=FIRST)

    interval, = report.session_returns
    assert (interval.session_date, interval.simple_return) == (FIRST, Decimal("0.10"))
    assert report.total_return == Decimal("0.10")
    assert report.annualized_volatility is report.sharpe_ratio is None


def test_short_dated_sample_has_volatility_but_no_cagr():
    report = evaluate_performance(
        equity_path([100, 110, 105]), start_date=FIRST, end_date=THIRD
    )
    assert report.annualized_return is None
    assert "at least 365 days" in report.annualized_return_reason
    assert report.annualized_volatility > 0


@pytest.mark.parametrize(
    ("final", "expected"), [(121, "0.10"), (81, "-0.10")]
)
def test_calendar_cagr_over_730_days(final, expected):
    start = date(2023, 1, 1)
    first = date(2023, 1, 3)
    end = date(2024, 12, 31)
    result = equity_path([110, final], (first, end))

    report = evaluate_performance(result, start_date=start, end_date=end)

    assert (end - start).days == 730
    assert report.annualized_return == Decimal(expected)
    assert report.annualized_return_reason is None


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        ({"start_date": FIRST}, "start_date and end_date must both be dates"),
        ({"end_date": THIRD}, "start_date and end_date must both be dates"),
        (
            {"start_date": SECOND, "end_date": THIRD},
            "start_date must be on or before the first session",
        ),
        (
            {"start_date": FIRST, "end_date": date(2025, 1, 9)},
            "end_date must match the final session",
        ),
    ],
)
def test_cagr_bounds_must_match_the_evaluated_period(dates, message):
    with pytest.raises(ValueError, match=message):
        evaluate_performance(equity_path([100, 110, 105]), **dates)


def test_turnover_excludes_rejected_and_outstanding_orders():
    spy = dataset("SPY", (FIRST, 50, 50), (SECOND, 60, 50))
    qqq = dataset("QQQ", (FIRST, 50, 50), (SECOND, 60, 55))

    def strategy(context):
        if context.session_date == FIRST:
            return (
                SignalEvent("SPY", FIRST, 0.5), SignalEvent("QQQ", FIRST, 0.5)
            )
        return (SignalEvent("QQQ", SECOND, 0),)

    result = run([spy, qqq], strategy, commission=Decimal("0.50"))
    assert len(result.rejected_orders) == len(result.outstanding_orders) == 1
    assert len(result.orders) == 3

    report = evaluate_performance(result)

    assert report.executed_orders == 1
    assert report.executed_notional == Decimal("60")
    assert report.total_commissions == Decimal("0.50")
    assert report.turnover == near("0.6169665809768637")
    assert report.average_long_exposure == near("0.291005291005291")


def test_multiple_instruments_can_be_fully_invested_at_the_close():
    spy = dataset("SPY", (FIRST, 10, 10), (SECOND, 10, 12))
    qqq = dataset("QQQ", (FIRST, 10, 10), (SECOND, 10, 9))

    def strategy(context):
        if context.session_date == FIRST:
            return (
                SignalEvent("SPY", FIRST, 0.6), SignalEvent("QQQ", FIRST, 0.4)
            )
        return ()

    result = run([spy, qqq], strategy)
    report = evaluate_performance(result)

    assert result.sessions[-1].portfolio.positions == {"QQQ": 4, "SPY": 6}
    assert result.sessions[-1].portfolio.cash == 0
    assert result.sessions[-1].equity == Decimal("108")
    exposures = [item.long_exposure for item in report.session_returns]
    assert exposures == [1]
    assert report.average_long_exposure == Decimal("0.5")
    assert report.total_return == Decimal("0.08")
    assert report.executed_notional == Decimal("100")


@pytest.mark.parametrize(
    "equity", [0, -1, True, Decimal("NaN"), Decimal("Infinity")]
)
def test_rejects_nonpositive_or_nonfinite_equity(equity):
    result = equity_path([100], (FIRST,))
    invalid = replace(result.sessions[0], equity=equity)

    with pytest.raises(ValueError, match="equity on 2025-01-03"):
        evaluate_performance(replace(result, sessions=(invalid,)))


@pytest.mark.parametrize(
    ("cash", "message"),
    [
        (Decimal(-1), "cash on 2025-01-03"),
        (Decimal("NaN"), "cash on 2025-01-03"),
        (Decimal("Infinity"), "cash on 2025-01-03"),
        (Decimal("101"), "cash exceeds equity"),
    ],
)
def test_rejects_inconsistent_cash(cash, message):
    result = equity_path([100], (FIRST,))
    session = result.sessions[0]
    invalid = replace(session, portfolio=replace(session.portfolio, cash=cash))

    with pytest.raises(ValueError, match=message):
        evaluate_performance(replace(result, sessions=(invalid,)))


@pytest.mark.parametrize("dates", [(SECOND, FIRST), (FIRST, FIRST)])
def test_rejects_unsorted_or_duplicate_sessions(dates):
    with pytest.raises(
        ValueError, match="session dates must be strictly increasing"
    ):
        evaluate_performance(equity_path([100, 110], dates))


def test_rejects_empty_session_history():
    with pytest.raises(ValueError, match="at least one session valuation"):
        evaluate_performance(BacktestResult((), (), (), ()))


@pytest.mark.parametrize("rate", [True, "0", Decimal("NaN"), float("inf"), -1])
def test_rejects_invalid_risk_free_returns(rate):
    with pytest.raises(ValueError, match="risk-free rate"):
        evaluate_performance(
            equity_path([100, 110, 105]), risk_free_rate_per_session=rate
        )


def test_evaluation_is_immutable_and_leaves_source_ledger_unchanged():
    ledger = PortfolioLedger(100)
    fill = Fill("buy", "SPY", FIRST, OrderSide.BUY, 2, Decimal("10.01"))
    ledger.apply_fill(fill, Decimal("0.015"))
    snapshot = ledger.snapshot()
    history = ledger.history()
    session = SessionValuation(FIRST, snapshot, Decimal("103.965"))
    source = BacktestResult((session,), (), (), history)

    dates = {"start_date": CAPITAL_START, "end_date": FIRST}
    report = evaluate_performance(source, **dates)
    with localcontext() as ambient:
        ambient.prec = 2
        assert evaluate_performance(source, **dates) == report

    assert report.total_commissions == Decimal("0.015")
    assert report.executed_notional == Decimal("20.02")
    assert ledger.snapshot() == snapshot
    assert ledger.history() == history
    assert source.sessions == (session,)
    with pytest.raises(FrozenInstanceError):
        setattr(report, "total_return", Decimal(0))
    with pytest.raises(FrozenInstanceError):
        setattr(report.session_returns[0], "long_exposure", Decimal(0))


def test_long_elapsed_period_still_needs_two_observations_for_cagr():
    result = equity_path([121], (THIRD,))

    report = evaluate_performance(
        result, start_date=date(2023, 1, 1), end_date=THIRD
    )

    assert report.annualized_return is None


def test_positive_equity_above_cash_requires_an_open_holding():
    result = equity_path([100], (FIRST,))
    session = result.sessions[0]
    invalid = replace(session, portfolio=replace(session.portfolio, positions={}))

    with pytest.raises(ValueError, match="equity and cash disagree with positions"):
        evaluate_performance(replace(result, sessions=(invalid,)))
