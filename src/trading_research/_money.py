"""Exact decimal arithmetic shared by execution pricing and accounting."""

from decimal import Context, Decimal, Inexact, InvalidOperation, Overflow

Money = int | float | Decimal

USD_CONTEXT = Context(prec=34, traps=[Inexact, InvalidOperation, Overflow])


def to_decimal(value: Money) -> Decimal:
    return Decimal(str(value)) if isinstance(value, float) else Decimal(value)


def decimal_amount(
    value: Money, field: str, *, allow_zero: bool = False
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{field} must be a numeric amount")
    amount = to_decimal(value)
    if (
        not amount.is_finite()
        or amount < 0
        or (amount == 0 and not allow_zero)
    ):
        sign = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field} must be finite and {sign}")
    return amount
