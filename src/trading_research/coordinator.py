"""Commit proposed fills to a consistent pair of in-memory ledgers."""

from collections.abc import Iterable
from datetime import date

from .corporate_actions import CashDividend, StockSplit
from .execution import ProposedExecution, _validate_proposal_price
from .orders import MarketOrder, OrderLedger, OrderSnapshot, OrderState
from .portfolio import (
    DividendPayment,
    PortfolioLedger,
    PortfolioRejection,
    PortfolioSnapshot,
    PortfolioTransaction,
    SplitAdjustment,
)


class ExecutionCoordinator:
    """Own the order and portfolio state for one sequential run."""

    def __init__(
        self, orders: OrderLedger, portfolio: PortfolioLedger
    ) -> None:
        orders = orders._copy()
        portfolio = portfolio._copy()
        order_fills = tuple(
            entry.fill
            for entry in orders.history()
            if entry.state is OrderState.FILLED
        )
        booked_fills = tuple(entry.fill for entry in portfolio.history())
        if order_fills != booked_fills:
            raise ValueError("order and portfolio histories disagree on fills")
        self._ledgers = (orders, portfolio)

    def submit_orders(
        self,
        orders: Iterable[MarketOrder],
        *,
        cancellations: Iterable[tuple[str, date]] = (),
    ) -> None:
        """Stage cancellations and new acceptances together at the decision close."""
        order_ledger, portfolio = self._ledgers
        staged = order_ledger._copy()
        for order_id, session in cancellations:
            staged.cancel(order_id, session=session)
        for order in orders:
            staged.register(order)
            staged.accept(order.order_id)
        self._ledgers = (staged, portfolio)

    def execute(
        self, order_id: str, proposal: ProposedExecution
    ) -> OrderSnapshot:
        """Commit a matched fill or record an economic rejection."""
        _validate_proposal_price(proposal)
        fill = proposal.fill
        if fill.order_id != order_id:
            raise ValueError(
                "proposal fill order ID does not match requested order"
            )

        orders, portfolio = self._ledgers
        next_orders = orders._copy()
        next_orders.record_fill(fill)
        next_portfolio = portfolio._copy()
        try:
            next_portfolio.apply_fill(fill, proposal.commission)
        except PortfolioRejection as rejection:
            next_orders = orders._copy()
            next_orders.reject(order_id, str(rejection))
            next_portfolio = portfolio

        outcome = next_orders.snapshot(order_id)
        # Both transitions are staged before publishing the new pair.
        self._ledgers = (next_orders, next_portfolio)
        return outcome

    def _apply_corporate_actions(
        self,
        session: date,
        *,
        splits: Iterable[StockSplit] = (),
        entitlements: Iterable[tuple[CashDividend, date]] = (),
    ) -> None:
        """Publish this date's split, entitlement, and payment bookings together."""
        orders, portfolio = self._ledgers
        staged = portfolio._copy()
        for split in splits:
            staged.apply_split(split)
        for dividend, preceding_session in entitlements:
            staged.accrue_dividend(
                dividend, session=session, preceding_session=preceding_session
            )
        for receivable in staged.snapshot().unpaid_receivables:
            if receivable.dividend.payment_date == session:
                staged.pay_dividend(receivable.dividend.action_id, session=session)
        self._ledgers = (orders, staged)

    def _record_close(self, session: date) -> None:
        self._ledgers[1].record_close(session)

    def split_history(self) -> tuple[SplitAdjustment, ...]:
        return self._ledgers[1].split_history()

    def dividend_payments(self) -> tuple[DividendPayment, ...]:
        return self._ledgers[1].dividend_payments()

    def order_snapshot(self, order_id: str) -> OrderSnapshot:
        return self._ledgers[0].snapshot(order_id)

    def order_history(self) -> tuple[OrderSnapshot, ...]:
        return self._ledgers[0].history()

    def portfolio_snapshot(self) -> PortfolioSnapshot:
        return self._ledgers[1].snapshot()

    def portfolio_history(self) -> tuple[PortfolioTransaction, ...]:
        return self._ledgers[1].history()
