"""Whole-share market orders and their in-memory lifecycle."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from math import isfinite

from ._money import Money


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderState(Enum):
    CREATED = "created"
    ACCEPTED = "accepted"
    FILLED = "filled"
    REJECTED = "rejected"


_TRANSITIONS = {
    OrderState.CREATED: (OrderState.ACCEPTED, OrderState.REJECTED),
    OrderState.ACCEPTED: (OrderState.FILLED, OrderState.REJECTED),
    OrderState.FILLED: (),
    OrderState.REJECTED: (),
}


def _validate_order_fields(
    order_id: str, symbol: str, side: OrderSide, quantity: int
) -> None:
    if (
        not isinstance(order_id, str)
        or not order_id
        or order_id != order_id.strip()
    ):
        raise ValueError(
            "order_id must be nonempty without surrounding whitespace"
        )
    if (
        not isinstance(symbol, str)
        or not symbol
        or not symbol.isprintable()
        or any(char.isspace() for char in symbol)
    ):
        raise ValueError(
            "symbol must be a nonempty printable string without whitespace"
        )
    if not isinstance(side, OrderSide):
        raise ValueError("side must be OrderSide.BUY or OrderSide.SELL")
    if (
        isinstance(quantity, bool)
        or not isinstance(quantity, int)
        or quantity <= 0
    ):
        raise ValueError("quantity must be a positive integer")


def _validate_session(session: date, field: str) -> None:
    if not isinstance(session, date) or isinstance(session, datetime):
        raise ValueError(f"{field} must be a date")


@dataclass(frozen=True, slots=True)
class MarketOrder:
    """A whole-share order decided after the decision session closes."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    decision_session: date
    earliest_execution_session: date | None = None

    def __post_init__(self) -> None:
        _validate_order_fields(
            self.order_id, self.symbol, self.side, self.quantity
        )
        _validate_session(self.decision_session, "decision_session")
        earliest = self.earliest_execution_session
        if earliest is not None:
            _validate_session(earliest, "earliest_execution_session")
            if earliest <= self.decision_session:
                raise ValueError(
                    "earliest_execution_session must be later "
                    "than decision_session"
                )


@dataclass(frozen=True, slots=True)
class Fill:
    """An externally supplied simulated execution result."""

    order_id: str
    symbol: str
    execution_session: date
    side: OrderSide
    quantity: int
    price: Money

    def __post_init__(self) -> None:
        _validate_order_fields(
            self.order_id, self.symbol, self.side, self.quantity
        )
        _validate_session(self.execution_session, "execution_session")
        price = self.price
        if (
            isinstance(price, bool)
            or not isinstance(price, (int, float, Decimal))
            or (isinstance(price, float) and not isfinite(price))
            or (isinstance(price, Decimal) and not price.is_finite())
            or price <= 0
        ):
            raise ValueError("price must be a finite, positive number")


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    """The state recorded after one successful ledger operation."""

    order: MarketOrder
    state: OrderState
    fill: Fill | None = None
    rejection_reason: str | None = None


class OrderLedger:
    """Track orders and successful transitions in processing order."""

    def __init__(self) -> None:
        self._orders: dict[str, OrderSnapshot] = {}
        self._history: list[OrderSnapshot] = []

    def _copy(self) -> "OrderLedger":
        ledger = OrderLedger()
        ledger._orders = self._orders.copy()
        ledger._history = self._history.copy()
        return ledger

    def register(self, order: MarketOrder) -> None:
        if not isinstance(order, MarketOrder):
            raise ValueError("expected a MarketOrder")
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order identifier {order.order_id!r}")
        self._record(OrderSnapshot(order, OrderState.CREATED))

    def accept(self, order_id: str) -> None:
        order = self._order_for_transition(order_id, OrderState.ACCEPTED)
        self._record(OrderSnapshot(order, OrderState.ACCEPTED))

    def reject(self, order_id: str, reason: str) -> None:
        order = self._order_for_transition(order_id, OrderState.REJECTED)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("rejection reason must be a nonempty string")
        self._record(
            OrderSnapshot(order, OrderState.REJECTED, rejection_reason=reason)
        )

    def record_fill(self, fill: Fill) -> None:
        if not isinstance(fill, Fill):
            raise ValueError("expected a Fill")
        order_id = fill.order_id
        order = self._order_for_transition(order_id, OrderState.FILLED)
        if fill.symbol != order.symbol:
            raise ValueError(f"order {order_id!r}: fill symbol does not match")
        if fill.side is not order.side:
            raise ValueError(f"order {order_id!r}: fill side does not match")
        if fill.quantity != order.quantity:
            raise ValueError(f"order {order_id!r}: fill quantity does not match")
        if fill.execution_session <= order.decision_session:
            raise ValueError(
                f"order {order_id!r}: fill session must be later "
                "than the decision session"
            )
        earliest = order.earliest_execution_session
        if earliest is not None and fill.execution_session < earliest:
            raise ValueError(
                f"order {order_id!r}: fill session is before "
                f"the earliest eligible session {earliest}"
            )
        self._record(OrderSnapshot(order, OrderState.FILLED, fill=fill))

    def snapshot(self, order_id: str) -> OrderSnapshot:
        """Return the order's current immutable record."""
        if not isinstance(order_id, str) or order_id not in self._orders:
            raise ValueError(f"unknown order identifier {order_id!r}")
        return self._orders[order_id]

    def history(self) -> tuple[OrderSnapshot, ...]:
        """Return successful ledger operations in call order."""
        return tuple(self._history)

    def _order_for_transition(
        self, order_id: str, target: OrderState
    ) -> MarketOrder:
        current = self.snapshot(order_id)
        if target not in _TRANSITIONS[current.state]:
            raise ValueError(
                f"order {order_id!r}: cannot transition from "
                f"{current.state.name} to {target.name}"
            )
        return current.order

    def _record(self, snapshot: OrderSnapshot) -> None:
        self._orders[snapshot.order.order_id] = snapshot
        self._history.append(snapshot)
