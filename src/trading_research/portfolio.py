"""USD cash and whole-share position accounting for supplied fills."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from types import MappingProxyType

from ._money import USD_CONTEXT, Money, decimal_amount, to_decimal
from .orders import Fill, OrderSide


@dataclass(frozen=True, slots=True)
class PortfolioTransaction:
    """A fill applied once, with its USD fee and resulting cash balance."""

    fill: Fill
    commission: Decimal
    cash_after: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Cash, open holdings, and cumulative commissions at one ledger state."""

    initial_cash: Decimal
    cash: Decimal
    positions: Mapping[str, int]
    cumulative_commissions: Decimal

    def __post_init__(self) -> None:
        positions = MappingProxyType(dict(self.positions))
        object.__setattr__(self, "positions", positions)


class PortfolioLedger:
    """Book supplied fills against USD cash and long-only holdings."""

    def __init__(self, initial_cash: Money) -> None:
        self._initial_cash = decimal_amount(initial_cash, "initial cash")
        self._cash = self._initial_cash
        self._positions: dict[str, int] = {}
        self._commissions = Decimal(0)
        # Order IDs are unique because the order ledger permits one full fill.
        self._transactions: dict[str, PortfolioTransaction] = {}

    def apply_fill(self, fill: Fill, commission: Money = 0) -> None:
        """Book a fill and its supplied commission in this ledger."""
        if not isinstance(fill, Fill):
            raise ValueError("expected a Fill")
        order_id = fill.order_id
        if order_id in self._transactions:
            raise ValueError(
                f"fill for order {order_id!r} was already applied"
            )

        fee = decimal_amount(commission, "commission", allow_zero=True)
        # Fill already validates the price, quantity, side, and identifiers.
        price = to_decimal(fill.price)
        held = self._positions.get(fill.symbol, 0)

        with localcontext(USD_CONTEXT):
            notional = price * fill.quantity
            if fill.side is OrderSide.BUY:
                debit = notional + fee
                if debit > self._cash:
                    raise ValueError(f"order {order_id!r}: insufficient cash")
                cash = self._cash - debit
                quantity = held + fill.quantity
            else:
                if fill.quantity > held:
                    raise ValueError(
                        f"order {order_id!r}: insufficient holdings"
                    )
                if fee > notional:
                    raise ValueError(
                        f"order {order_id!r}: commission exceeds proceeds"
                    )
                cash = self._cash + (notional - fee)
                quantity = held - fill.quantity
            commissions = self._commissions + fee

        transaction = PortfolioTransaction(fill, fee, cash)
        if quantity:
            self._positions[fill.symbol] = quantity
        else:
            del self._positions[fill.symbol]
        self._cash = cash
        self._commissions = commissions
        self._transactions[order_id] = transaction

    def snapshot(self) -> PortfolioSnapshot:
        """Return a detached, read-only snapshot of the current state."""
        return PortfolioSnapshot(
            self._initial_cash, self._cash, self._positions, self._commissions
        )

    def history(self) -> tuple[PortfolioTransaction, ...]:
        """Return the applied fills in transaction order."""
        return tuple(self._transactions.values())


def mark_to_market(
    portfolio: PortfolioSnapshot, prices: Mapping[str, Money] | None = None
) -> Decimal:
    """Value a snapshot in USD using explicit prices for all open holdings."""
    equity = portfolio.cash
    with localcontext(USD_CONTEXT):
        for symbol, quantity in portfolio.positions.items():
            if prices is None or symbol not in prices:
                raise ValueError(f"missing valuation price for {symbol!r}")
            price = decimal_amount(
                prices[symbol], f"valuation price for {symbol!r}"
            )
            equity += quantity * price
    return equity
