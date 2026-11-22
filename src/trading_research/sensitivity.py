"""Rerun the common-window SPY comparison under fixed transaction costs."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from ._money import Money
from .benchmark import (
    BenchmarkRun,
    BuyAndHoldEntryError,
    CommonWindowBenchmarkComparison,
    _equity_difference,
    compare_spy_common_window,
)
from .daily_bars import DailyBar, DatasetMetadata
from .mean_reversion import MeanReversionExitRejected

_SCENARIOS = (
    ("A", Decimal(0), Decimal("0.00")),
    ("B", Decimal(0), Decimal("1.25")),
    ("C", Decimal(10), Decimal("1.25")),
    ("D", Decimal(25), Decimal("1.25")),
    ("E", Decimal(50), Decimal("1.25")),
    ("F", Decimal(25), Decimal("5.00")),
)


class CostScenarioFailed(RuntimeError):
    """An invalid buy-and-hold entry or rejected exit stopped a cost scenario."""

    def __init__(
        self,
        scenario: str,
        adjustment_bps: Decimal,
        commission: Decimal,
        cause: BuyAndHoldEntryError | MeanReversionExitRejected,
    ) -> None:
        self.scenario = scenario
        self.adjustment_bps = adjustment_bps
        self.commission = commission
        super().__init__(
            f"cost scenario {scenario} ({adjustment_bps} bps, "
            f"${commission} commission per fill) failed: {cause}"
        )


@dataclass(frozen=True, slots=True)
class ZeroCostDifference:
    """Scenario minus zero cost for one portfolio, counting shares on both sides."""

    portfolio: str
    equity_difference_usd: Decimal
    return_difference_pp: Decimal
    executed_order_change: int
    filled_share_change: int


@dataclass(frozen=True, slots=True)
class CostScenarioResult:
    """One illustrative cost scenario with its complete common-window comparison."""

    scenario: str
    comparison: CommonWindowBenchmarkComparison
    differences_vs_zero_cost: tuple[ZeroCostDifference, ...]


def _portfolios(
    comparison: CommonWindowBenchmarkComparison,
) -> tuple[BenchmarkRun, ...]:
    return (
        comparison.momentum, comparison.mean_reversion,
        comparison.buy_and_hold, comparison.cash,
    )


def _difference(
    run: BenchmarkRun, baseline: BenchmarkRun, initial_cash: Decimal,
) -> ZeroCostDifference:
    return_pp, equity_usd = _equity_difference(
        run.backtest.sessions[-1].equity,
        baseline.backtest.sessions[-1].equity,
        initial_cash,
    )
    return ZeroCostDifference(
        portfolio=run.label,
        equity_difference_usd=equity_usd,
        return_difference_pp=return_pp,
        executed_order_change=(
            run.metrics.executed_orders - baseline.metrics.executed_orders
        ),
        filled_share_change=(
            sum(fill.quantity for fill in run.backtest.fills)
            - sum(fill.quantity for fill in baseline.backtest.fills)
        ),
    )


def run_transaction_cost_sensitivity(
    metadata: DatasetMetadata,
    bars: Sequence[DailyBar],
    *,
    initial_cash: Money,
) -> tuple[CostScenarioResult, ...]:
    """Run the six illustrative scenarios in order on one synthetic SPY dataset.

    An invalid buy-and-hold entry or rejected mean-reversion exit raises
    CostScenarioFailed with the original exception as its cause.
    """
    if not isinstance(bars, Sequence):
        raise ValueError("bars must be a sequence of DailyBar records")
    observations = tuple(bars)
    results: list[CostScenarioResult] = []
    for scenario, adjustment_bps, commission in _SCENARIOS:
        try:
            comparison = compare_spy_common_window(
                metadata, observations, initial_cash=initial_cash,
                adjustment_bps=adjustment_bps, commission=commission,
            )
        except (BuyAndHoldEntryError, MeanReversionExitRejected) as error:
            raise CostScenarioFailed(
                scenario, adjustment_bps, commission, error
            ) from error

        zero_cost = results[0].comparison if results else comparison
        differences = tuple(
            _difference(run, baseline, comparison.initial_cash)
            for run, baseline in zip(
                _portfolios(comparison), _portfolios(zero_cost), strict=True
            )
        )
        results.append(CostScenarioResult(scenario, comparison, differences))
    return tuple(results)
