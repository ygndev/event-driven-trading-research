"""Run a deterministic daily backtest from observed opens and closes."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from ._money import Money
from .coordinator import ExecutionCoordinator
from .daily_bars import DailyBar, DatasetMetadata, PriceAdjustment
from .execution import NextOpenExecution
from .orders import (
    Fill,
    MarketOrder,
    OrderLedger,
    OrderSide,
    OrderSnapshot,
    OrderState,
)
from .planning import plan_orders
from .portfolio import (
    PortfolioLedger,
    PortfolioSnapshot,
    PortfolioTransaction,
    mark_to_market,
)
from .replay import MarketEvent, MarketOpen, replay_daily_bars
from .signals import (
    ExecutionFeedback,
    SessionContext,
    SignalEvent,
    _evaluate_strategy,
)


@dataclass(frozen=True, slots=True)
class SessionValuation:
    session_date: date
    portfolio: PortfolioSnapshot
    equity: Decimal


@dataclass(frozen=True, slots=True)
class BacktestResult:
    sessions: tuple[SessionValuation, ...]
    signals: tuple[SignalEvent, ...]
    order_history: tuple[OrderSnapshot, ...]
    transactions: tuple[PortfolioTransaction, ...]

    @property
    def orders(self) -> tuple[MarketOrder, ...]:
        return tuple(
            entry.order
            for entry in self.order_history
            if entry.state is OrderState.CREATED
        )

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(transaction.fill for transaction in self.transactions)

    @property
    def rejected_orders(self) -> tuple[OrderSnapshot, ...]:
        return tuple(
            entry for entry in self.order_history
            if entry.state is OrderState.REJECTED
        )

    @property
    def outstanding_orders(self) -> tuple[OrderSnapshot, ...]:
        latest = {entry.order.order_id: entry for entry in self.order_history}
        return tuple(
            entry for entry in latest.values()
            if entry.state is OrderState.ACCEPTED
        )


def run_backtest(
    datasets: Iterable[tuple[DatasetMetadata, Sequence[DailyBar]]],
    strategy: Callable[[SessionContext], Iterable[SignalEvent]],
    *,
    initial_cash: Money,
    adjustment_bps: Money,
    commission: Money,
) -> BacktestResult:
    """Process observed opens, completed bars, decisions, then valuation."""
    datasets = tuple(datasets)
    for metadata, _ in datasets:
        if not isinstance(metadata, DatasetMetadata):
            raise ValueError("backtest inputs require DatasetMetadata")
        if metadata.adjustment is not PriceAdjustment.RAW:
            raise ValueError(
                f"{metadata.symbol!r}: backtest requires RAW data"
            )

    coordinator = ExecutionCoordinator(
        OrderLedger(), PortfolioLedger(initial_cash)
    )
    execution = NextOpenExecution(
        adjustment_bps=adjustment_bps, commission=commission
    )
    pending: dict[str, MarketOrder] = {}
    close_events: list[MarketEvent] = []
    opening_outcomes: list[OrderSnapshot] = []
    context: SessionContext | None = None
    observed_symbols: set[str] = set()
    target_weights: dict[str, int | float] = {}
    signals: list[SignalEvent] = []
    sessions: list[SessionValuation] = []

    def execute_open(_session: date, opens: tuple[MarketOpen, ...]) -> None:
        by_symbol = {opening.symbol: opening for opening in opens}
        ordered = sorted(
            pending.values(),
            key=lambda order: (
                order.side is OrderSide.BUY, order.symbol,
                order.decision_session, order.order_id,
            ),
        )
        for order in ordered:
            opening = by_symbol.get(order.symbol)
            if opening is None:
                continue
            snapshot = coordinator.order_snapshot(order.order_id)
            proposal = execution.propose_at_open(snapshot, opening)
            if proposal is not None:
                outcome = coordinator.execute(order.order_id, proposal)
                opening_outcomes.append(outcome)
                del pending[order.order_id]

    def finish_session(session: date) -> None:
        nonlocal context
        portfolio = coordinator.portfolio_snapshot()
        feedback = ExecutionFeedback(
            portfolio, tuple(pending.values()), tuple(opening_outcomes)
        )
        context = SessionContext(
            session, tuple(close_events), context, execution=feedback
        )
        close_events.clear()
        opening_outcomes.clear()
        emitted = _evaluate_strategy(strategy, context, observed_symbols)
        signals.extend(emitted)

        if emitted:
            target_weights.update(
                (signal.symbol, signal.target_exposure) for signal in emitted
            )
            allocation = (
                SignalEvent(symbol, session, weight)
                for symbol, weight in target_weights.items()
            )
            orders = plan_orders(allocation, portfolio, context)
            coordinator.submit_orders(orders)
            pending.update((order.order_id, order) for order in orders)

        closes = {
            event.bar.symbol: event.bar.close for event in context.events
        }
        equity = mark_to_market(portfolio, closes)
        sessions.append(SessionValuation(session, portfolio, equity))

    replay_daily_bars(
        datasets,
        close_events.append,
        on_session_open=execute_open,
        on_session_end=finish_session,
    )
    return BacktestResult(
        tuple(sessions), tuple(signals),
        coordinator.order_history(), coordinator.portfolio_history(),
    )
