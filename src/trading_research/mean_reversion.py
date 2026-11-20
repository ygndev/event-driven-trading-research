"""Fixed long-only SPY mean reversion with a fill-based holding period."""

from decimal import Decimal, localcontext

from ._money import USD_CONTEXT, decimal_amount
from .daily_bars import PriceAdjustment
from .orders import OrderSide, OrderSnapshot, OrderState
from .replay import MarketEvent
from .signals import SessionContext, SignalEvent

_SYMBOL = "SPY"
_LOOKBACK_INTERVALS = 5
_HOLDING_SESSIONS = 5
_ENTRY_PRICE_RATIO = Decimal("0.97")


class MeanReversionExitRejected(RuntimeError):
    """A rejected exit left the mean-reversion position open."""

    def __init__(self, rejection: OrderSnapshot) -> None:
        self.rejection = rejection
        order = rejection.order
        reason = rejection.rejection_reason or "reason unavailable"
        super().__init__(
            f"{order.symbol} mean-reversion exit {order.order_id!r} "
            f"from decision session {order.decision_session} was rejected "
            f"and left the position open ({reason})"
        )


def _entry_triggered(current: MarketEvent, reference: MarketEvent) -> bool:
    if (
        current.adjustment is not PriceAdjustment.RAW
        or reference.adjustment is not PriceAdjustment.RAW
    ):
        raise ValueError("SPY mean reversion requires RAW closes")

    current_close = decimal_amount(current.bar.close, "current SPY close")
    reference_close = decimal_amount(reference.bar.close, "lookback SPY close")
    # With a positive denominator this is equivalent to a return <= -3%.
    with localcontext(USD_CONTEXT):
        return current_close <= reference_close * _ENTRY_PRICE_RATIO


def short_term_mean_reversion(
    context: SessionContext,
) -> tuple[SignalEvent, ...]:
    """Enter on a five-observation decline and exit after five held closes."""
    execution = context.execution
    if execution is None:
        raise ValueError("mean reversion requires backtest execution feedback")
    if not any(
        event.bar.symbol == _SYMBOL
        and event.bar.session_date == context.session_date
        for event in context.events
    ):
        return ()
    if any(order.symbol == _SYMBOL for order in execution.pending_orders):
        return ()

    session = context.session_date
    history = context.history(_SYMBOL)
    latest = context.last_order_outcome(_SYMBOL)
    if execution.portfolio.positions.get(_SYMBOL, 0):
        if (
            latest is not None
            and latest.order.side is OrderSide.SELL
            and latest.state is OrderState.REJECTED
        ):
            raise MeanReversionExitRejected(latest)
        if (
            latest is None
            or latest.order.side is not OrderSide.BUY
            or latest.state is not OrderState.FILLED
            or latest.fill is None
        ):
            raise ValueError(
                "a held SPY position requires a committed entry fill"
            )

        entry_session = latest.fill.execution_session
        completed = sum(
            event.bar.session_date >= entry_session for event in history
        )
        if completed >= _HOLDING_SESSIONS:
            return (SignalEvent(_SYMBOL, session, 0),)
        return ()

    if len(history) <= _LOOKBACK_INTERVALS:
        return ()
    if not _entry_triggered(history[-1], history[-1 - _LOOKBACK_INTERVALS]):
        return ()

    if (
        latest is not None
        and latest.order.side is OrderSide.BUY
        and latest.state is OrderState.REJECTED
    ):
        if (
            latest in execution.outcomes
            or len(history) <= _LOOKBACK_INTERVALS + 1
        ):
            return ()
        if _entry_triggered(history[-2], history[-2 - _LOOKBACK_INTERVALS]):
            return ()

    return (SignalEvent(_SYMBOL, session, 1),)
