"""Propose whole-share market executions at the next eligible observed open."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext

from ._money import USD_CONTEXT, Money, decimal_amount, to_decimal
from .daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
    validate_daily_bars,
)
from .orders import Fill, MarketOrder, OrderSide, OrderSnapshot, OrderState


@dataclass(frozen=True, slots=True)
class ProposedExecution:
    """A priced execution and its fixed USD commission."""

    fill: Fill
    commission: Decimal
    opening_price: Decimal
    adjustment_bps: Decimal


class NextOpenExecution:
    """Price each order at most once per sequential simulation run."""

    def __init__(self, *, adjustment_bps: Money, commission: Money) -> None:
        self._adjustment_bps = decimal_amount(
            adjustment_bps, "adjustment_bps", allow_zero=True
        )
        self._commission = decimal_amount(
            commission, "commission", allow_zero=True
        )
        self._proposed: set[str] = set()

    def propose(
        self,
        order: MarketOrder | OrderSnapshot,
        bars: Sequence[DailyBar],
        metadata: DatasetMetadata,
    ) -> ProposedExecution | None:
        """Return a proposal for the first eligible bar, or None if absent.

        A bare order represents acceptance confirmed by the caller. A supplied
        ledger snapshot must be ACCEPTED.
        """
        if isinstance(order, OrderSnapshot):
            if order.state is not OrderState.ACCEPTED:
                raise ValueError(
                    "execution requires an ACCEPTED order snapshot"
                )
            order = order.order
        if not isinstance(order, MarketOrder):
            raise ValueError("execution requires a MarketOrder")
        if order.order_id in self._proposed:
            raise ValueError(f"order {order.order_id!r} was already proposed")

        if not isinstance(metadata, DatasetMetadata):
            raise ValueError("execution requires DatasetMetadata")
        if metadata.symbol != order.symbol:
            raise ValueError(
                f"order symbol {order.symbol!r} does not match "
                f"dataset symbol {metadata.symbol!r}"
            )
        if metadata.adjustment is not PriceAdjustment.RAW:
            raise ValueError("execution requires data declared RAW")
        if not isinstance(bars, Sequence) or isinstance(
            bars, (str, bytes, bytearray)
        ):
            raise ValueError("bars must be a sequence of DailyBar records")
        observations = tuple(bars)
        if not observations:
            return None
        validate_daily_bars(observations, metadata)

        earliest = order.earliest_execution_session
        for bar in observations:
            session = bar.session_date
            if session <= order.decision_session:
                continue
            if earliest is not None and session < earliest:
                continue

            opening_price = to_decimal(bar.open)
            with localcontext(USD_CONTEXT):
                adjustment = self._adjustment_bps / 10_000
                multiplier = (
                    1 + adjustment
                    if order.side is OrderSide.BUY
                    else 1 - adjustment
                )
                price = opening_price * multiplier

            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                execution_session=session,
                side=order.side,
                quantity=order.quantity,
                price=price,
            )
            proposal = ProposedExecution(
                fill, self._commission, opening_price, self._adjustment_bps
            )
            self._proposed.add(order.order_id)
            return proposal

        return None
