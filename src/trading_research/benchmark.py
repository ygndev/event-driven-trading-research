"""Compare fixed SPY research rules with buy-and-hold and cash."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, localcontext

from ._money import USD_CONTEXT, Money, decimal_amount
from .backtest import BacktestResult, run_backtest
from .daily_bars import DailyBar, DatasetMetadata
from .mean_reversion import _LOOKBACK_INTERVALS as _MEAN_REVERSION_LOOKBACK
from .mean_reversion import short_term_mean_reversion
from .momentum import _LOOKBACK_INTERVALS as _MOMENTUM_LOOKBACK
from .momentum import time_series_momentum
from .orders import OrderSide, OrderSnapshot
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


@dataclass(frozen=True, slots=True)
class MeanReversionBenchmarkComparison:
    start_session: date
    end_session: date
    initial_cash: Decimal
    adjustment_bps: Decimal
    commission: Decimal
    mean_reversion: BenchmarkRun
    buy_and_hold: BenchmarkRun
    cash: BenchmarkRun

    @property
    def return_difference_pp_vs_cash(self) -> Decimal:
        with localcontext(_STATISTICS_CONTEXT):
            return (
                self.mean_reversion.metrics.total_return
                - self.cash.metrics.total_return
            ) * 100

    @property
    def outstanding_exit(self) -> OrderSnapshot | None:
        """The accepted SELL, if the round trip is incomplete at the end."""
        orders = self.mean_reversion.backtest.outstanding_orders
        return next(
            (entry for entry in orders if entry.order.side is OrderSide.SELL),
            None,
        )


def _comparison_window(
    result: BacktestResult,
    dates: tuple[date, ...],
    anchor_index: int,
    initial_cash: Decimal,
) -> BacktestResult:
    observed = tuple(session.session_date for session in result.sessions)
    if observed != dates:
        raise ValueError(
            "backtest valuation dates must match the SPY observations"
        )

    sessions = result.sessions[anchor_index:]
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
    eligible_dates = set(dates[anchor_index + 1:])
    if any(
        trade.fill.execution_session not in eligible_dates
        for trade in result.transactions
    ):
        raise ValueError(
            "comparison contains fills outside the eligible window"
        )

    return replace(result, sessions=sessions)


def _comparison_inputs(
    metadata: DatasetMetadata,
    bars: Sequence[DailyBar],
    anchor_index: int,
    initial_cash: Money,
    adjustment_bps: Money,
    commission: Money,
) -> tuple[tuple[DailyBar, ...], Decimal, Decimal, Decimal]:
    if not isinstance(metadata, DatasetMetadata) or metadata.symbol != "SPY":
        raise ValueError("benchmark comparison requires a SPY dataset")
    if not isinstance(bars, Sequence):
        raise ValueError("bars must be a sequence of DailyBar records")
    observations = tuple(bars)
    minimum_closes = anchor_index + 1
    if len(observations) < minimum_closes:
        raise ValueError(
            f"benchmark comparison requires {minimum_closes} SPY closes"
        )

    return (
        observations,
        decimal_amount(initial_cash, "initial cash"),
        decimal_amount(adjustment_bps, "adjustment_bps", allow_zero=True),
        decimal_amount(commission, "commission", allow_zero=True),
    )


def _compare_spy_runs(
    metadata: DatasetMetadata,
    bars: tuple[DailyBar, ...],
    strategy: Callable[[SessionContext], Iterable[SignalEvent]],
    *,
    label: str,
    anchor_index: int,
    initial_cash: Decimal,
    adjustment_bps: Decimal,
    commission: Decimal,
) -> tuple[BenchmarkRun, BenchmarkRun, BenchmarkRun]:
    datasets = ((metadata, bars),)

    def run(
        strategy: Callable[[SessionContext], Iterable[SignalEvent]],
    ) -> BacktestResult:
        return run_backtest(
            datasets, strategy, initial_cash=initial_cash,
            adjustment_bps=adjustment_bps, commission=commission,
        )

    strategy_result = run(strategy)
    dates = tuple(bar.session_date for bar in bars)
    start, end = dates[anchor_index], dates[-1]
    primary = _comparison_window(
        strategy_result, dates, anchor_index, initial_cash
    )

    def buy_and_hold(context: SessionContext) -> tuple[SignalEvent, ...]:
        if context.session_date == start:
            return (SignalEvent("SPY", start, 1),)
        return ()

    holding = _comparison_window(
        run(buy_and_hold), dates, anchor_index, initial_cash
    )
    baseline_equity = holding.sessions[-1].equity

    def measure(label: str, result: BacktestResult) -> BenchmarkRun:
        metrics = evaluate_performance(result, start_date=start, end_date=end)
        with localcontext(USD_CONTEXT):
            difference_usd = result.sessions[-1].equity - baseline_equity
        with localcontext(_STATISTICS_CONTEXT):
            difference_pp = difference_usd / initial_cash * 100
        return BenchmarkRun(
            label, result, metrics, difference_pp, difference_usd
        )

    return (
        measure(label, primary),
        measure("SPY buy-and-hold", holding),
        measure(
            "Cash (zero interest)",
            _comparison_window(
                run(lambda context: ()), dates, anchor_index, initial_cash
            ),
        ),
    )


def compare_spy_benchmarks(
    metadata: DatasetMetadata,
    bars: Sequence[DailyBar],
    *,
    initial_cash: Money,
    adjustment_bps: Money,
    commission: Money,
) -> BenchmarkComparison:
    """Compare all three approaches from the first scheduled momentum close."""
    observations, capital, bps, fee = _comparison_inputs(
        metadata, bars, _MOMENTUM_LOOKBACK, initial_cash,
        adjustment_bps, commission,
    )
    momentum, holding, cash = _compare_spy_runs(
        metadata, observations, time_series_momentum, label="SPY momentum",
        anchor_index=_MOMENTUM_LOOKBACK, initial_cash=capital,
        adjustment_bps=bps, commission=fee,
    )
    return BenchmarkComparison(
        start_session=observations[_MOMENTUM_LOOKBACK].session_date,
        end_session=observations[-1].session_date,
        initial_cash=capital,
        adjustment_bps=bps,
        commission=fee,
        momentum=momentum,
        buy_and_hold=holding,
        cash=cash,
    )


def compare_spy_mean_reversion_benchmarks(
    metadata: DatasetMetadata,
    bars: Sequence[DailyBar],
    *,
    initial_cash: Money,
    adjustment_bps: Money,
    commission: Money,
) -> MeanReversionBenchmarkComparison:
    """Score mean reversion and its benchmarks from the sixth SPY close."""
    observations, capital, bps, fee = _comparison_inputs(
        metadata, bars, _MEAN_REVERSION_LOOKBACK, initial_cash,
        adjustment_bps, commission,
    )
    mean_reversion, holding, cash = _compare_spy_runs(
        metadata, observations, short_term_mean_reversion,
        label="SPY mean reversion", anchor_index=_MEAN_REVERSION_LOOKBACK,
        initial_cash=capital, adjustment_bps=bps, commission=fee,
    )
    return MeanReversionBenchmarkComparison(
        start_session=observations[_MEAN_REVERSION_LOOKBACK].session_date,
        end_session=observations[-1].session_date,
        initial_cash=capital,
        adjustment_bps=bps,
        commission=fee,
        mean_reversion=mean_reversion,
        buy_and_hold=holding,
        cash=cash,
    )
