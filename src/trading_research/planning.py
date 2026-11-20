"""Plan whole-share orders from completed-session target weights."""

from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, localcontext

from ._money import USD_CONTEXT, decimal_amount, to_decimal
from .daily_bars import DailyBar, PriceAdjustment
from .orders import MarketOrder, OrderSide
from .portfolio import PortfolioSnapshot, mark_to_market
from .replay import MarketEvent
from .signals import SessionContext, SignalEvent


def _reference_prices(
    context: SessionContext, required: set[str]
) -> dict[str, Decimal]:
    prices: dict[str, Decimal] = {}
    for event in context.events:
        if not isinstance(event, MarketEvent) or not isinstance(
            event.bar, DailyBar
        ):
            raise ValueError("session context requires daily market events")
        bar = event.bar
        if bar.session_date != context.session_date:
            raise ValueError(
                f"{bar.symbol!r}: observation must be from "
                "the decision session"
            )
        if bar.symbol not in required:
            continue
        if bar.symbol in prices:
            raise ValueError(
                f"duplicate current-session bar for {bar.symbol!r}"
            )
        if event.adjustment is not PriceAdjustment.RAW:
            raise ValueError(f"{bar.symbol!r}: planning requires RAW data")
        prices[bar.symbol] = decimal_amount(
            bar.close, f"reference close for {bar.symbol!r}"
        )

    missing = required - prices.keys()
    if missing:
        symbols = ", ".join(sorted(missing))
        raise ValueError(
            f"missing current-session reference close for {symbols}"
        )
    return prices


def plan_orders(
    targets: Iterable[SignalEvent],
    portfolio: PortfolioSnapshot,
    context: SessionContext,
) -> tuple[MarketOrder, ...]:
    """Size a complete allocation at current closes without booking orders.

    A nonempty allocation must cover all open holdings. Callers include
    previously retained targets themselves. Empty targets request no changes.
    """
    if not isinstance(portfolio, PortfolioSnapshot):
        raise ValueError("planning requires a PortfolioSnapshot")
    if not isinstance(context, SessionContext):
        raise ValueError("planning requires a SessionContext")
    session = context.session_date
    if not isinstance(session, date) or isinstance(session, datetime):
        raise ValueError("decision session must be a date")

    weights: dict[str, Decimal] = {}
    for target in targets:
        if not isinstance(target, SignalEvent):
            raise ValueError("targets must contain SignalEvent records")
        symbol = target.symbol
        if target.decision_session != session:
            raise ValueError(f"{symbol!r}: target must match decision session")
        if symbol in weights:
            raise ValueError(f"duplicate target for {symbol!r}")
        weights[symbol] = to_decimal(target.target_exposure)

    if not weights:
        return ()

    cash = portfolio.cash
    if not isinstance(cash, Decimal) or not cash.is_finite() or cash < 0:
        raise ValueError("portfolio cash must be a finite nonnegative Decimal")
    for symbol, quantity in portfolio.positions.items():
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity <= 0
        ):
            raise ValueError(
                f"{symbol!r}: position must be positive whole shares"
            )

    omitted = portfolio.positions.keys() - weights.keys()
    if omitted:
        symbols = ", ".join(sorted(omitted))
        raise ValueError(
            f"complete allocation requires targets for holdings: {symbols}"
        )

    symbols = sorted(weights)
    with localcontext(USD_CONTEXT):
        if sum((weights[symbol] for symbol in symbols), Decimal(0)) > 1:
            raise ValueError("aggregate target weights must not exceed 1")

    required = set(portfolio.positions) | {
        symbol for symbol, weight in weights.items() if weight > 0
    }
    prices = _reference_prices(context, required)
    equity = mark_to_market(portfolio, prices)
    if not equity.is_finite() or equity <= 0:
        raise ValueError("portfolio equity must be finite and positive")

    sells: list[MarketOrder] = []
    buys: list[MarketOrder] = []
    with localcontext(USD_CONTEXT):
        for symbol in symbols:
            target_notional = equity * weights[symbol]
            # Decimal // floors a nonnegative quotient without rounding it.
            target_shares = (
                int(target_notional // prices[symbol]) if target_notional else 0
            )
            delta = target_shares - portfolio.positions.get(symbol, 0)
            if delta == 0:
                continue
            side = OrderSide.SELL if delta < 0 else OrderSide.BUY
            order = MarketOrder(
                order_id=f"{session.isoformat()}:{symbol}",
                symbol=symbol,
                side=side,
                quantity=abs(delta),
                decision_session=session,
            )
            (sells if delta < 0 else buys).append(order)

    return tuple(sells + buys)
