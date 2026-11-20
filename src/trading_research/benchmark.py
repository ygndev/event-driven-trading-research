"""Compare fixed SPY momentum with executable buy-and-hold and cash."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, localcontext

from ._money import USD_CONTEXT, Money, decimal_amount
from .backtest import BacktestResult, run_backtest
from .daily_bars import DailyBar, DatasetMetadata
from .momentum import _LOOKBACK_INTERVALS, time_series_momentum
from .performance import (
    _STATISTICS_CONTEXT,
    PerformanceResult,
    evaluate_performance,
)
from .signals import SessionContext, SignalEvent


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    label: str
    backtest: BacktestResult
    metrics: PerformanceResult
    return_difference_pp_vs_buy_and_hold: Decimal
    equity_difference_usd_vs_buy_and_hold: Decimal

    @property
    def equity_path(self) -> tuple[tuple[date, Decimal], ...]:
        return tuple(
            (session.session_date, session.equity)
            for session in self.backtest.sessions
        )


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    start_session: date
    end_session: date
    initial_cash: Decimal
    adjustment_bps: Decimal
    commission: Decimal
    momentum: BenchmarkRun
    buy_and_hold: BenchmarkRun
    cash: BenchmarkRun


def _comparison_window(
    result: BacktestResult, dates: tuple[date, ...], initial_cash: Decimal
) -> BacktestResult:
    observed = tuple(session.session_date for session in result.sessions)
    if observed != dates:
        raise ValueError(
            "backtest valuation dates must match the SPY observations"
        )

    sessions = result.sessions[_LOOKBACK_INTERVALS:]
    anchor = sessions[0]
    start = anchor.session_date
    if (
        anchor.equity != initial_cash
        or anchor.portfolio.initial_cash != initial_cash
        or anchor.portfolio.cash != initial_cash
        or anchor.portfolio.positions
        or anchor.portfolio.cumulative_commissions
    ):
        raise ValueError(
            "comparison must start with the same uninvested capital"
        )
    if (
        any(signal.decision_session < start for signal in result.signals)
        or any(
            entry.order.decision_session < start
            for entry in result.order_history
        )
    ):
        raise ValueError(
            "comparison contains decisions before the common anchor"
        )
    eligible_dates = set(dates[_LOOKBACK_INTERVALS + 1:])
    if any(
        trade.fill.execution_session not in eligible_dates
        for trade in result.transactions
    ):
        raise ValueError(
            "comparison contains fills outside the eligible window"
        )

    return replace(result, sessions=sessions)


def compare_spy_benchmarks(
    metadata: DatasetMetadata,
    bars: Sequence[DailyBar],
    *,
    initial_cash: Money,
    adjustment_bps: Money,
    commission: Money,
) -> BenchmarkComparison:
    """Compare all three approaches from the first scheduled momentum close."""
    if not isinstance(metadata, DatasetMetadata) or metadata.symbol != "SPY":
        raise ValueError("benchmark comparison requires a SPY dataset")
    if not isinstance(bars, Sequence):
        raise ValueError("bars must be a sequence of DailyBar records")
    bars = tuple(bars)
    minimum_closes = _LOOKBACK_INTERVALS + 1
    if len(bars) < minimum_closes:
        raise ValueError(
            f"benchmark comparison requires {minimum_closes} SPY closes"
        )

    capital = decimal_amount(initial_cash, "initial cash")
    bps = decimal_amount(adjustment_bps, "adjustment_bps", allow_zero=True)
    fee = decimal_amount(commission, "commission", allow_zero=True)
    datasets = ((metadata, bars),)

    def run(
        strategy: Callable[[SessionContext], Iterable[SignalEvent]],
    ) -> BacktestResult:
        return run_backtest(
            datasets, strategy, initial_cash=capital,
            adjustment_bps=bps, commission=fee,
        )

    momentum_result = run(time_series_momentum)
    dates = tuple(bar.session_date for bar in bars)
    start, end = dates[_LOOKBACK_INTERVALS], dates[-1]
    momentum = _comparison_window(momentum_result, dates, capital)

    def buy_and_hold(context: SessionContext) -> tuple[SignalEvent, ...]:
        if context.session_date == start:
            return (SignalEvent("SPY", start, 1),)
        return ()

    holding = _comparison_window(run(buy_and_hold), dates, capital)
    baseline_equity = holding.sessions[-1].equity

    def measure(label: str, result: BacktestResult) -> BenchmarkRun:
        metrics = evaluate_performance(result, start_date=start, end_date=end)
        with localcontext(USD_CONTEXT):
            difference_usd = result.sessions[-1].equity - baseline_equity
        with localcontext(_STATISTICS_CONTEXT):
            difference_pp = difference_usd / capital * 100
        return BenchmarkRun(
            label, result, metrics, difference_pp, difference_usd
        )

    return BenchmarkComparison(
        start_session=start,
        end_session=end,
        initial_cash=capital,
        adjustment_bps=bps,
        commission=fee,
        momentum=measure("SPY momentum", momentum),
        buy_and_hold=measure("SPY buy-and-hold", holding),
        cash=measure(
            "Cash (zero interest)",
            _comparison_window(run(lambda context: ()), dates, capital),
        ),
    )
