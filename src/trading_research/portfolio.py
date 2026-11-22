"""Standalone USD accounting for whole-share fills and corporate actions."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from types import MappingProxyType

from ._money import USD_CONTEXT, Money, decimal_amount, to_decimal
from .corporate_actions import CashDividend, StockSplit
from .orders import Fill, OrderSide, _validate_session


@dataclass(frozen=True, slots=True)
class PortfolioTransaction:
    """A fill applied once, with its USD fee and resulting cash balance."""

    fill: Fill
    commission: Decimal
    cash_after: Decimal


@dataclass(frozen=True, slots=True)
class SplitAdjustment:
    split: StockSplit
    shares_before: int
    shares_after: int


@dataclass(frozen=True, slots=True)
class DividendReceivable:
    dividend: CashDividend
    prior_close_session: date
    entitled_shares: int
    total_amount: Decimal


@dataclass(frozen=True, slots=True)
class DividendPayment:
    receivable: DividendReceivable
    session_date: date
    cash_after: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Available cash, open holdings, commissions, and unpaid dividends."""

    initial_cash: Decimal
    cash: Decimal
    positions: Mapping[str, int]
    cumulative_commissions: Decimal
    unpaid_receivables: tuple[DividendReceivable, ...] = ()

    def __post_init__(self) -> None:
        positions = MappingProxyType(dict(self.positions))
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "unpaid_receivables", tuple(self.unpaid_receivables))


class PortfolioRejection(ValueError):
    """A booking conflicts with the portfolio's accounting constraints."""


class PortfolioLedger:
    """Book supplied fills and explicit actions against long-only holdings."""

    def __init__(self, initial_cash: Money) -> None:
        self._initial_cash = decimal_amount(initial_cash, "initial cash")
        self._cash = self._initial_cash
        self._positions: dict[str, int] = {}
        self._commissions = Decimal(0)
        # Order IDs are unique because the order ledger permits one full fill.
        self._transactions: dict[str, PortfolioTransaction] = {}
        self._splits: dict[tuple[str, date], SplitAdjustment] = {}
        self._closing_positions: dict[date, Mapping[str, int]] = {}
        self._dividends: dict[str, DividendReceivable | None] = {}
        self._payments: dict[str, DividendPayment] = {}

    def _copy(self) -> "PortfolioLedger":
        ledger = PortfolioLedger(self._initial_cash)
        ledger._cash = self._cash
        ledger._positions = self._positions.copy()
        ledger._commissions = self._commissions
        ledger._transactions = self._transactions.copy()
        ledger._splits = self._splits.copy()
        ledger._closing_positions = self._closing_positions.copy()
        ledger._dividends = self._dividends.copy()
        ledger._payments = self._payments.copy()
        return ledger

    def _last_close(self) -> date:
        return next(reversed(self._closing_positions), date.min)

    def apply_fill(self, fill: Fill, commission: Money = 0) -> None:
        """Book a fill and its supplied commission in this ledger."""
        if not isinstance(fill, Fill):
            raise ValueError("expected a Fill")
        order_id = fill.order_id
        if order_id in self._transactions:
            raise ValueError(
                f"fill for order {order_id!r} was already applied"
            )

        session = fill.execution_session
        if self._closing_positions and session <= self._last_close():
            raise ValueError("cannot book a fill on or before the recorded close")
        if self._splits and any(
            symbol == fill.symbol and effective > session
            for symbol, effective in self._splits
        ):
            raise ValueError("cannot book a fill before an applied split")
        if self._payments and any(
            payment.session_date > session for payment in self._payments.values()
        ):
            raise ValueError("cannot book a fill before an applied dividend payment")

        fee = decimal_amount(commission, "commission", allow_zero=True)
        # Fill already validates the price, quantity, side, and identifiers.
        price = to_decimal(fill.price)
        held = self._positions.get(fill.symbol, 0)

        with localcontext(USD_CONTEXT):
            notional = price * fill.quantity
            if fill.side is OrderSide.BUY:
                debit = notional + fee
                if debit > self._cash:
                    raise PortfolioRejection(
                        f"order {order_id!r}: insufficient cash"
                    )
                cash = self._cash - debit
                quantity = held + fill.quantity
            else:
                if fill.quantity > held:
                    raise PortfolioRejection(
                        f"order {order_id!r}: insufficient holdings"
                    )
                if fee > notional:
                    raise PortfolioRejection(
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

    def apply_split(self, split: StockSplit) -> SplitAdjustment:
        """Book a split before any fill in the symbol on its effective date."""
        if not isinstance(split, StockSplit):
            raise ValueError("expected a StockSplit")
        symbol, session = split.symbol, split.effective_session
        key = (symbol, session)
        if key in self._splits:
            raise ValueError(f"{symbol} {session}: split already applied")
        if self._closing_positions and session <= self._last_close():
            raise ValueError("cannot apply a split on or before the recorded close")
        if any(
            trade.fill.symbol == symbol and trade.fill.execution_session >= session
            for trade in self._transactions.values()
        ):
            raise ValueError(
                f"{symbol} {session}: split must precede same-day or later fills"
            )
        if any(other == symbol and day > session for other, day in self._splits):
            raise ValueError(f"{symbol} {session}: a later split is already applied")

        before = self._positions.get(symbol, 0)
        quantity = before * split.ratio
        after = int(quantity)
        if quantity != after:
            raise PortfolioRejection(
                f"{symbol} {session}: split would create fractional shares"
            )

        adjustment = SplitAdjustment(split, before, after)
        if after:
            self._positions[symbol] = after
        self._splits[key] = adjustment
        return adjustment

    def record_close(self, session: date) -> None:
        """Freeze the holdings at a caller-supplied observed close."""
        _validate_session(session, "session")
        if self._closing_positions and session <= self._last_close():
            raise ValueError("recorded closes must be strictly increasing")
        if (
            any(
                trade.fill.execution_session > session
                for trade in self._transactions.values()
            )
            or any(day > session for _, day in self._splits)
        ):
            raise ValueError("cannot record a close before an applied position change")
        self._closing_positions[session] = MappingProxyType(self._positions.copy())

    def accrue_dividend(
        self, dividend: CashDividend, *, session: date, preceding_session: date,
    ) -> DividendReceivable | None:
        """Use the captured holdings at the caller-supplied preceding close.

        A known zero position consumes the action ID and creates no receivable.
        """
        if not isinstance(dividend, CashDividend):
            raise ValueError("expected a CashDividend")
        _validate_session(session, "session")
        _validate_session(preceding_session, "preceding_session")
        if session != dividend.ex_date:
            raise ValueError("dividend entitlement must be recorded on ex_date")
        if dividend.action_id in self._dividends:
            raise ValueError(
                f"dividend {dividend.action_id!r}: entitlement already processed"
            )
        if preceding_session >= session:
            raise ValueError("preceding_session must be before ex_date")
        if self._closing_positions and session <= self._last_close():
            raise ValueError("cannot accrue a dividend on or before the recorded close")
        positions = self._closing_positions.get(preceding_session)
        if positions is None:
            raise ValueError(
                f"missing recorded holdings for preceding session {preceding_session}"
            )

        shares = positions.get(dividend.symbol, 0)
        receivable = None
        if shares:
            with localcontext(USD_CONTEXT):
                amount = shares * to_decimal(dividend.amount_per_share)
            receivable = DividendReceivable(dividend, preceding_session, shares, amount)
        self._dividends[dividend.action_id] = receivable
        return receivable

    def pay_dividend(self, action_id: str, *, session: date) -> DividendPayment:
        """Move an unpaid entitlement into cash exactly once on its payment date."""
        _validate_session(session, "session")
        if not isinstance(action_id, str):
            raise ValueError("action_id must be a string")
        receivable = self._dividends.get(action_id)
        if receivable is None:
            raise ValueError(f"dividend {action_id!r}: no entitlement")
        if action_id in self._payments:
            raise ValueError(f"dividend {action_id!r}: already paid")
        if session != receivable.dividend.payment_date:
            raise ValueError(
                f"dividend {action_id!r}: payment must occur on payment_date"
            )
        if self._closing_positions and session <= self._last_close():
            raise ValueError("cannot pay a dividend on or before the recorded close")

        with localcontext(USD_CONTEXT):
            cash = self._cash + receivable.total_amount
        payment = DividendPayment(receivable, session, cash)
        self._payments[action_id] = payment
        self._cash = cash
        return payment

    def split_history(self) -> tuple[SplitAdjustment, ...]:
        return tuple(self._splits.values())

    def dividend_payments(self) -> tuple[DividendPayment, ...]:
        return tuple(self._payments.values())

    def snapshot(self) -> PortfolioSnapshot:
        """Return a detached, read-only snapshot of the current state."""
        unpaid = tuple(
            receivable for action_id, receivable in self._dividends.items()
            if receivable is not None and action_id not in self._payments
        )
        return PortfolioSnapshot(
            self._initial_cash, self._cash, self._positions, self._commissions,
            unpaid,
        )

    def history(self) -> tuple[PortfolioTransaction, ...]:
        """Return the applied fills in transaction order."""
        return tuple(self._transactions.values())


def mark_to_market(
    portfolio: PortfolioSnapshot,
    prices: Mapping[str, Money] | None = None,
    *,
    include_receivables: bool = False,
) -> Decimal:
    """Mark holdings at supplied prices and optionally include unpaid dividends."""
    equity = portfolio.cash
    with localcontext(USD_CONTEXT):
        for symbol, quantity in portfolio.positions.items():
            if prices is None or symbol not in prices:
                raise ValueError(f"missing valuation price for {symbol!r}")
            price = decimal_amount(
                prices[symbol], f"valuation price for {symbol!r}"
            )
            equity += quantity * price
        if include_receivables:
            for receivable in portfolio.unpaid_receivables:
                equity += receivable.total_amount
    return equity
