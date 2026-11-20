"""Evaluate research signals at completed daily session boundaries."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from .daily_bars import DailyBar, DatasetMetadata
from .replay import MarketEvent, replay_daily_bars


@dataclass(frozen=True, slots=True, eq=False)
class SessionContext:
    """Market observations available at a completed session."""

    session_date: date
    events: tuple[MarketEvent, ...]
    _previous: "SessionContext | None" = field(default=None, repr=False)

    def history(self, symbol: str) -> tuple[MarketEvent, ...]:
        """Return a symbol's events through this session, oldest first."""
        events = []
        context: SessionContext | None = self
        while context is not None:
            events.extend(
                event for event in context.events if event.bar.symbol == symbol
            )
            context = context._previous
        return tuple(reversed(events))


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """Long-only target exposure as research intent for a completed session."""

    symbol: str
    decision_session: date
    target_exposure: int | float

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("signal symbol must be a nonempty string")
        session = self.decision_session
        if not isinstance(session, date) or isinstance(session, datetime):
            raise ValueError("decision_session must be a date")
        exposure = self.target_exposure
        if (
            isinstance(exposure, bool)
            or not isinstance(exposure, (int, float))
            or not 0 <= exposure <= 1
        ):
            raise ValueError(
                "target_exposure must be a finite number from 0 to 1"
            )


def _evaluate_strategy(
    strategy: Callable[[SessionContext], Iterable[SignalEvent]],
    context: SessionContext,
    observed_symbols: set[str],
) -> tuple[SignalEvent, ...]:
    session = context.session_date
    observed_symbols.update(event.bar.symbol for event in context.events)
    emitted = strategy(context)
    if not isinstance(emitted, Iterable):
        raise ValueError(f"{session}: strategy must return iterable signals")

    batch: dict[str, SignalEvent] = {}
    for signal in emitted:
        if not isinstance(signal, SignalEvent):
            raise ValueError(f"{session}: expected a SignalEvent")
        symbol = signal.symbol
        if signal.decision_session != session:
            raise ValueError(
                f"{symbol}: decision session {signal.decision_session} "
                f"must match completed session {session}"
            )
        if symbol not in observed_symbols:
            raise ValueError(f"{session}: unknown instrument {symbol!r}")
        if symbol in batch:
            raise ValueError(f"{session}: duplicate signal for {symbol!r}")
        batch[symbol] = signal

    return tuple(batch[symbol] for symbol in sorted(batch))


def generate_signals(
    datasets: Iterable[tuple[DatasetMetadata, Sequence[DailyBar]]],
    strategy: Callable[[SessionContext], Iterable[SignalEvent]],
) -> tuple[SignalEvent, ...]:
    """Evaluate completed sessions and return signals by date and symbol."""
    pending: list[MarketEvent] = []
    context: SessionContext | None = None
    observed_symbols: set[str] = set()
    signals: list[SignalEvent] = []

    def evaluate(session: date) -> None:
        nonlocal context
        context = SessionContext(session, tuple(pending), context)
        pending.clear()
        signals.extend(_evaluate_strategy(strategy, context, observed_symbols))

    replay_daily_bars(datasets, pending.append, on_session_end=evaluate)
    return tuple(signals)
