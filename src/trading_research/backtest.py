"""Run a deterministic daily backtest from observed opens and closes."""

from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from ._money import Money
from .coordinator import ExecutionCoordinator
from .corporate_actions import CashDividend, StockSplit
from .daily_bars import DailyBar, DatasetMetadata, PriceAdjustment, validate_daily_bars
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
    DividendPayment,
    PortfolioLedger,
    PortfolioSnapshot,
    PortfolioTransaction,
    SplitAdjustment,
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
    split_adjustments: tuple[SplitAdjustment, ...] = ()
    dividend_payments: tuple[DividendPayment, ...] = ()

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


class PendingOrderSplitError(ValueError):
    """An accepted order still uses the pre-split share basis."""

    def __init__(self, order: MarketOrder, split: StockSplit) -> None:
        self.order = order
        self.split = split
        super().__init__(
            f"pending order {order.order_id!r} for {split.symbol} crosses "
            f"the split effective {split.effective_session}"
        )


def _schedule_actions(
    datasets: tuple[tuple[DatasetMetadata, Sequence[DailyBar]], ...],
    actions: tuple[StockSplit | CashDividend, ...],
) -> tuple[
    dict[date, tuple[StockSplit, ...]],
    dict[date, tuple[tuple[CashDividend, date], ...]],
    tuple[date, ...],
]:
    dates_by_symbol = {}
    for metadata, bars in datasets:
        validate_daily_bars(bars, metadata)
        symbol = metadata.symbol
        if symbol in dates_by_symbol:
            raise ValueError(f"multiple datasets for symbol {symbol!r}")
        dates_by_symbol[symbol] = tuple(bar.session_date for bar in bars)

    splits: dict[date, list[StockSplit]] = {}
    entitlements: dict[date, list[tuple[CashDividend, date]]] = {}
    split_keys: set[tuple[str, date]] = set()
    dividend_keys: set[tuple[str, date]] = set()
    dividend_ids: set[str] = set()
    payments: set[date] = set()
    for action in actions:
        if not isinstance(action, (StockSplit, CashDividend)):
            raise ValueError("corporate actions must be StockSplit or CashDividend records")
        is_split = isinstance(action, StockSplit)
        session = action.effective_session if is_split else action.ex_date
        symbol = action.symbol
        dates = dates_by_symbol.get(symbol)
        if dates is None:
            raise ValueError(f"corporate action for unknown symbol {symbol!r}")
        index = bisect_left(dates, session)
        if index == len(dates) or dates[index] != session:
            raise ValueError(f"{symbol} {session}: corporate action requires an observed session")
        key = symbol, session
        if isinstance(action, StockSplit):
            if key in split_keys:
                raise ValueError(f"{symbol} {session}: duplicate or conflicting split")
            split_keys.add(key)
            splits.setdefault(session, []).append(action)
        else:
            if action.action_id in dividend_ids or key in dividend_keys:
                raise ValueError(f"{symbol} {session}: duplicate or conflicting dividend")
            if index == 0:
                raise ValueError(f"{symbol} {session}: dividend requires a preceding observed close")
            dividend_ids.add(action.action_id)
            dividend_keys.add(key)
            entitlements.setdefault(session, []).append((action, dates[index - 1]))
            payments.add(action.payment_date)

    return (
        {day: tuple(sorted(rows, key=lambda split: split.symbol))
         for day, rows in splits.items()},
        {day: tuple(sorted(rows, key=lambda row: row[0].symbol))
         for day, rows in entitlements.items()},
        tuple(sorted(payments)),
    )


def run_backtest(
    datasets: Iterable[tuple[DatasetMetadata, Sequence[DailyBar]]],
    strategy: Callable[[SessionContext], Iterable[SignalEvent]],
    *,
    initial_cash: Money,
    adjustment_bps: Money,
    commission: Money,
    corporate_actions: Iterable[StockSplit | CashDividend] = (),
) -> BacktestResult:
    """Process supplied actions, observed opens, completed bars, and decisions."""
    datasets = tuple(datasets)
    for metadata, _ in datasets:
        if not isinstance(metadata, DatasetMetadata):
            raise ValueError("backtest inputs require DatasetMetadata")
        if metadata.adjustment is not PriceAdjustment.RAW:
            raise ValueError(
                f"{metadata.symbol!r}: backtest requires RAW data"
            )

    actions = tuple(corporate_actions)
    splits, entitlements, payments = {}, {}, ()
    if actions:
        datasets = tuple(
            (metadata, tuple(bars) if isinstance(bars, Sequence) else bars)
            for metadata, bars in datasets
        )
        splits, entitlements, payments = _schedule_actions(datasets, actions)
    closing_dates = {
        preceding for rows in entitlements.values() for _, preceding in rows
    }
    payment_dates = deque(payments)

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
    replacements_without_close: set[str] = set()
    signals: list[SignalEvent] = []
    sessions: list[SessionValuation] = []

    def execute_open(session: date, opens: tuple[MarketOpen, ...]) -> None:
        effective_splits = splits.get(session, ())
        for split in effective_splits:
            for order in pending.values():
                if order.symbol == split.symbol:
                    raise PendingOrderSplitError(order, split)
        while payment_dates and payment_dates[0] < session:
            coordinator._apply_corporate_actions(payment_dates.popleft())
        ex_dividends = entitlements.get(session, ())
        payment_due = bool(payment_dates and payment_dates[0] == session)
        if effective_splits or ex_dividends or payment_due:
            coordinator._apply_corporate_actions(
                session, splits=effective_splits, entitlements=ex_dividends
            )
            if payment_due:
                payment_dates.popleft()

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
        nonlocal context, target_weights, replacements_without_close
        portfolio = coordinator.portfolio_snapshot()
        feedback = ExecutionFeedback(
            portfolio, tuple(pending.values()), tuple(opening_outcomes)
        )
        context = SessionContext(
            session, tuple(close_events), context, execution=feedback,
            _effective_splits=splits.get(session, ()),
        )
        close_events.clear()
        opening_outcomes.clear()
        emitted = _evaluate_strategy(strategy, context, observed_symbols)
        closes = {
            event.bar.symbol: event.bar.close for event in context.events
        }
        if emitted:
            updates = {signal.symbol: signal.target_exposure for signal in emitted}
            next_weights = target_weights | updates
            cancelled = {
                order_id: order.symbol
                for order_id, order in pending.items()
                if order.symbol in updates
                and updates[order.symbol] != target_weights[order.symbol]
            }
            retained_pending = {
                order.symbol for order_id, order in pending.items()
                if order_id not in cancelled
            }
            # Cancellation needs no price. Replacement sizing still needs a current close.
            waiting_for_close = replacements_without_close.union(
                cancelled.values()
            ).difference(closes)
            allocation = (
                SignalEvent(symbol, session, weight)
                for symbol, weight in next_weights.items()
            )
            orders = plan_orders(
                allocation, portfolio, context,
                skip_symbols=retained_pending | waiting_for_close,
            )
            coordinator.submit_orders(
                orders, cancellations=((order_id, session) for order_id in cancelled)
            )
            for order_id in cancelled:
                del pending[order_id]
            pending.update((order.order_id, order) for order in orders)
            target_weights = next_weights
            replacements_without_close = waiting_for_close
            signals.extend(emitted)

        equity = mark_to_market(portfolio, closes, include_receivables=True)
        sessions.append(SessionValuation(session, portfolio, equity))
        if session in closing_dates:
            coordinator._record_close(session)

    replay_daily_bars(
        datasets,
        close_events.append,
        on_session_open=execute_open,
        on_session_end=finish_session,
    )
    # Post-close dates update the payment audit without adding strategy sessions.
    for payment_date in payment_dates:
        coordinator._apply_corporate_actions(payment_date)
    return BacktestResult(
        tuple(sessions), tuple(signals),
        coordinator.order_history(), coordinator.portfolio_history(),
        coordinator.split_history(), coordinator.dividend_payments(),
    )
