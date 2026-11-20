"""Declared stock splits and ordinary USD cash dividends."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from fractions import Fraction

from ._money import Money, decimal_amount
from .daily_bars import _validate_text
from .orders import _validate_session


def _validate_symbol(symbol: str) -> None:
    if (
        not isinstance(symbol, str)
        or not symbol
        or not symbol.isprintable()
        or any(character.isspace() for character in symbol)
    ):
        raise ValueError(
            "symbol must be a nonempty printable string without whitespace"
        )


@dataclass(frozen=True, slots=True)
class StockSplit:
    """A split ratio of new shares per old share, stored as a Fraction."""

    symbol: str
    effective_session: date
    ratio: Fraction | Decimal | int

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _validate_session(self.effective_session, "effective_session")
        ratio = self.ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (Fraction, Decimal, int))
            or (isinstance(ratio, Decimal) and not ratio.is_finite())
            or ratio <= 0
        ):
            raise ValueError(
                "split ratio must be a positive Fraction, Decimal, or integer"
            )
        object.__setattr__(self, "ratio", Fraction(ratio))


@dataclass(frozen=True, slots=True)
class CashDividend:
    """An ordinary USD distribution on the prior-close share basis."""

    action_id: str
    symbol: str
    ex_date: date
    payment_date: date
    amount_per_share: Money

    def __post_init__(self) -> None:
        _validate_text(self.action_id, "action_id")
        _validate_symbol(self.symbol)
        _validate_session(self.ex_date, "ex_date")
        _validate_session(self.payment_date, "payment_date")
        if self.payment_date < self.ex_date:
            raise ValueError("payment_date must be on or after ex_date")
        amount = decimal_amount(self.amount_per_share, "dividend amount per share")
        object.__setattr__(self, "amount_per_share", amount)
