"""Compare fixed SPY research rules with buy-and-hold and cash."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, localcontext
from itertools import combinations

from ._money import USD_CONTEXT, Money, decimal_amount
from .backtest import BacktestResult, run_backtest
from .daily_bars import DailyBar, DatasetMetadata
from .mean_reversion import _LOOKBACK_INTERVALS as _MEAN_REVERSION_LOOKBACK
from .mean_reversion import short_term_mean_reversion
from .momentum import _LOOKBACK_INTERVALS as _MOMENTUM_LOOKBACK
from .momentum import time_series_momentum
from .orders import OrderSide, OrderSnapshot, OrderState
from .performance import (
    _STATISTICS_CONTEXT,
    PerformanceResult,
    evaluate_performance,
)
from .signals import SessionContext, SignalEvent


class BuyAndHoldEntryError(RuntimeError):
    """The evaluation window could not establish an invested benchmark."""

    def __init__(self, symbol: str, anchor: date, entry: OrderSnapshot | None) -> None:
        self.symbol = symbol
        self.anchor = anchor
        self.entry = entry
        if entry is None:
            self.status = "zero_shares"
            detail = "whole-share sizing produced zero shares"
        elif entry.state is OrderState.ACCEPTED:
            self.status = "outstanding"
            detail = f"order {entry.order.order_id!r} is outstanding at the final close"
        else:
            self.status = entry.state.value
            detail = f"order {entry.order.order_id!r} is {self.status}"
            if entry.rejection_reason:
                detail += f" ({entry.rejection_reason})"
        super().__init__(f"{symbol} buy-and-hold entry at anchor {anchor}: {detail}")


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """A scored portfolio with its entry record for buy-and-hold."""

    label: str
    backtest: BacktestResult
    metrics: PerformanceResult
    return_difference_pp_vs_buy_and_hold: Decimal
    equity_difference_usd_vs_buy_and_hold: Decimal
    entry: OrderSnapshot | None = None

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
        return _outstanding_exit(self.mean_reversion)


@dataclass(frozen=True, slots=True)
class BenchmarkDifference:
    left: str
    right: str
    return_difference_pp: Decimal
    equity_difference_usd: Decimal


@dataclass(frozen=True, slots=True)
class CommonWindowBenchmarkComparison:
    start_session: date
    end_session: date
    initial_cash: Decimal
    adjustment_bps: Decimal
    commission: Decimal
    momentum: BenchmarkRun
    mean_reversion: BenchmarkRun
    buy_and_hold: BenchmarkRun
    cash: BenchmarkRun

    @property
    def pairwise_differences(self) -> tuple[BenchmarkDifference, ...]:
        """Return left-minus-right differences for each of the six pairs."""
        runs = (self.momentum, self.mean_reversion, self.buy_and_hold, self.cash)
        return tuple(
            BenchmarkDifference(
                left.label,
                right.label,
                *_equity_difference(
                    left.backtest.sessions[-1].equity,
                    right.backtest.sessions[-1].equity,
                    self.initial_cash,
                ),
            )
            for left, right in combinations(runs, 2)
        )

    @property
    def outstanding_exit(self) -> OrderSnapshot | None:
        """The accepted mean-reversion SELL, if the round trip is incomplete."""
        return _outstanding_exit(self.mean_reversion)


def _outstanding_exit(run: BenchmarkRun) -> OrderSnapshot | None:
    return next(
        (
            entry for entry in run.backtest.outstanding_orders
            if entry.order.side is OrderSide.SELL
        ),
        None,
    )


def _equity_difference(
    left: Decimal, right: Decimal, initial_cash: Decimal,
) -> tuple[Decimal, Decimal]:
    # Equal initial capital turns the final equity difference into a return gap.
    with localcontext(USD_CONTEXT):
        difference_usd = left - right
    with localcontext(_STATISTICS_CONTEXT):
        difference_pp = difference_usd / initial_cash * 100
    return difference_pp, difference_usd


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
    strategies: tuple[
        tuple[str, Callable[[SessionContext], Iterable[SignalEvent]]], ...
    ],
    *,
    anchor_index: int,
    initial_cash: Decimal,
    adjustment_bps: Decimal,
    commission: Decimal,
) -> tuple[tuple[BenchmarkRun, ...], BenchmarkRun, BenchmarkRun]:
    datasets = ((metadata, bars),)
    anchor_bar = bars[anchor_index]

    def run(
        strategy: Callable[[SessionContext], Iterable[SignalEvent]],
    ) -> BacktestResult:
        def active(context: SessionContext) -> Iterable[SignalEvent]:
            if context.session_date < anchor_bar.session_date:
                return ()
            return strategy(context)

        return run_backtest(
            datasets, active, initial_cash=initial_cash,
            adjustment_bps=adjustment_bps, commission=commission,
        )

    strategy_results = tuple(
        (label, run(strategy)) for label, strategy in strategies
    )
    dates = tuple(bar.session_date for bar in bars)
    start, end = dates[anchor_index], dates[-1]
    strategy_windows = tuple(
        (label, _comparison_window(result, dates, anchor_index, initial_cash))
        for label, result in strategy_results
    )

    def buy_and_hold(context: SessionContext) -> tuple[SignalEvent, ...]:
        if context.session_date == start:
            return (SignalEvent("SPY", start, 1),)
        return ()

    holding = _comparison_window(
        run(buy_and_hold), dates, anchor_index, initial_cash
    )
    entry = holding.order_history[-1] if holding.order_history else None
    if entry is None or entry.state is not OrderState.FILLED:
        raise BuyAndHoldEntryError(metadata.symbol, start, entry)
    baseline_equity = holding.sessions[-1].equity

    def measure(
        label: str, result: BacktestResult, *, entry: OrderSnapshot | None = None,
    ) -> BenchmarkRun:
        metrics = evaluate_performance(result, start_date=start, end_date=end)
        difference_pp, difference_usd = _equity_difference(
            result.sessions[-1].equity, baseline_equity, initial_cash
        )
        return BenchmarkRun(
            label, result, metrics, difference_pp, difference_usd, entry=entry,
        )

    return (
        tuple(measure(label, result) for label, result in strategy_windows),
        measure("SPY buy-and-hold", holding, entry=entry),
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
    (momentum,), holding, cash = _compare_spy_runs(
        metadata, observations, (("SPY momentum", time_series_momentum),),
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
    (mean_reversion,), holding, cash = _compare_spy_runs(
        metadata, observations,
        (("SPY mean reversion", short_term_mean_reversion),),
        anchor_index=_MEAN_REVERSION_LOOKBACK,
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


def compare_spy_common_window(
    metadata: DatasetMetadata,
    bars: Sequence[DailyBar],
    *,
    initial_cash: Money,
    adjustment_bps: Money,
    commission: Money,
) -> CommonWindowBenchmarkComparison:
    """Compare both fixed SPY strategies from the 64th observed close."""
    return _compare_spy_window(
        metadata, bars, anchor_index=_MOMENTUM_LOOKBACK,
        initial_cash=initial_cash, adjustment_bps=adjustment_bps,
        commission=commission,
    )


def _compare_spy_window(
    metadata: DatasetMetadata,
    bars: Sequence[DailyBar],
    *,
    anchor_index: int,
    initial_cash: Money,
    adjustment_bps: Money,
    commission: Money,
) -> CommonWindowBenchmarkComparison:
    observations, capital, bps, fee = _comparison_inputs(
        metadata, bars, anchor_index, initial_cash, adjustment_bps, commission,
    )
    (momentum, mean_reversion), holding, cash = _compare_spy_runs(
        metadata, observations,
        (
            ("SPY momentum", time_series_momentum),
            ("SPY mean reversion", short_term_mean_reversion),
        ),
        anchor_index=anchor_index, initial_cash=capital,
        adjustment_bps=bps, commission=fee,
    )
    return CommonWindowBenchmarkComparison(
        start_session=observations[anchor_index].session_date,
        end_session=observations[-1].session_date,
        initial_cash=capital,
        adjustment_bps=bps,
        commission=fee,
        momentum=momentum,
        mean_reversion=mean_reversion,
        buy_and_hold=holding,
        cash=cash,
    )
