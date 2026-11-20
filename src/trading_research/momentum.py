"""Fixed long-only SPY time-series momentum research rule."""

from ._money import decimal_amount
from .daily_bars import PriceAdjustment
from .signals import SessionContext, SignalEvent

_SYMBOL = "SPY"
_LOOKBACK_INTERVALS = 63
_REBALANCE_INTERVAL = 21


def time_series_momentum(context: SessionContext) -> tuple[SignalEvent, ...]:
    """Target full SPY exposure on positive trailing price momentum."""
    if not any(
        event.bar.symbol == _SYMBOL
        and event.bar.session_date == context.session_date
        for event in context.events
    ):
        return ()

    history = context.history(_SYMBOL)
    intervals = len(history) - 1
    if intervals < _LOOKBACK_INTERVALS:
        return ()
    if (intervals - _LOOKBACK_INTERVALS) % _REBALANCE_INTERVAL:
        return ()

    current = history[-1]
    reference = history[-1 - _LOOKBACK_INTERVALS]
    if (
        current.adjustment is not PriceAdjustment.RAW
        or reference.adjustment is not PriceAdjustment.RAW
    ):
        raise ValueError("SPY momentum requires RAW closes")

    current_close = decimal_amount(current.bar.close, "current SPY close")
    reference_close = decimal_amount(reference.bar.close, "lookback SPY close")
    # For a positive reference, current / reference - 1 has this exact sign.
    target = 1 if current_close > reference_close else 0
    return (SignalEvent(_SYMBOL, context.session_date, target),)
