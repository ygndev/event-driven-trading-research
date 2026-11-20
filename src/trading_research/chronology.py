"""Evaluate fixed SPY strategies in independently initialized date windows."""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from ._money import Money
from .benchmark import CommonWindowBenchmarkComparison, _compare_spy_window
from .daily_bars import DailyBar, DatasetMetadata, PriceAdjustment, validate_daily_bars
from .momentum import _LOOKBACK_INTERVALS as _MOMENTUM_LOOKBACK


@dataclass(frozen=True, slots=True)
class ResearchPeriod:
    """A named range of inclusive session dates."""

    name: str
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("research period requires a name")
        if any(
            not isinstance(day, date) or isinstance(day, datetime)
            for day in (self.start_date, self.end_date)
        ):
            raise ValueError(f"{self.name}: research boundaries must be dates")
        if self.start_date > self.end_date:
            raise ValueError(f"{self.name}: research start must not follow its end")


RESEARCH_PERIODS = (
    ResearchPeriod("development", date(2010, 1, 1), date(2018, 12, 31)),
    ResearchPeriod("validation", date(2019, 1, 1), date(2022, 12, 31)),
    ResearchPeriod("final_test", date(2023, 1, 1), date(2025, 12, 31)),
)


@dataclass(frozen=True, slots=True)
class ResearchPeriodResult:
    period: ResearchPeriod
    warmup_observation_count: int
    comparison: CommonWindowBenchmarkComparison


def evaluate_spy_research_periods(
    metadata: DatasetMetadata,
    bars: Sequence[DailyBar],
    *,
    initial_cash: Money,
    adjustment_bps: Money,
    commission: Money,
    periods: Sequence[ResearchPeriod] = RESEARCH_PERIODS,
) -> tuple[ResearchPeriodResult, ...]:
    """Score each window from its first observed close with 63 earlier bars.

    Replay starts at the beginning of the supplied history to retain the
    momentum schedule. Trading stays inactive until the evaluation anchor.
    """
    if not isinstance(periods, Sequence):
        raise ValueError("periods must be an ordered sequence of ResearchPeriod values")
    windows = tuple(periods)
    if not windows or any(not isinstance(period, ResearchPeriod) for period in windows):
        raise ValueError("periods must contain at least one ResearchPeriod")
    for earlier, later in zip(windows, windows[1:]):
        if later.start_date <= earlier.end_date:
            raise ValueError(
                "research periods must be chronological and non-overlapping"
            )

    if not isinstance(metadata, DatasetMetadata) or metadata.symbol != "SPY":
        raise ValueError("chronological evaluation requires a SPY dataset")
    if metadata.adjustment is not PriceAdjustment.RAW:
        raise ValueError("chronological evaluation requires RAW data")
    if not isinstance(bars, Sequence):
        raise ValueError("bars must be a sequence of DailyBar records")
    observations = tuple(bars)
    validate_daily_bars(observations, metadata)
    dates = tuple(bar.session_date for bar in observations)

    results: list[ResearchPeriodResult] = []
    for period in windows:
        start = bisect_left(dates, period.start_date)
        stop = bisect_right(dates, period.end_date)
        if start == stop:
            raise ValueError(f"{period.name}: no SPY observations in the period")
        anchor_index = max(start, _MOMENTUM_LOOKBACK)
        if anchor_index >= stop:
            raise ValueError(
                f"{period.name}: evaluation requires {_MOMENTUM_LOOKBACK} "
                "earlier SPY closes before a scored session within the period"
            )

        comparison = _compare_spy_window(
            metadata, observations[:stop],
            anchor_index=anchor_index,
            initial_cash=initial_cash, adjustment_bps=adjustment_bps,
            commission=commission,
        )
        results.append(ResearchPeriodResult(period, anchor_index, comparison))
    return tuple(results)
