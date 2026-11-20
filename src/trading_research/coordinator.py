"""Commit proposed fills to a consistent pair of in-memory ledgers."""

from .execution import ProposedExecution, _validate_proposal_price
from .orders import OrderLedger, OrderSnapshot, OrderState
from .portfolio import (
    PortfolioLedger,
    PortfolioRejection,
    PortfolioSnapshot,
    PortfolioTransaction,
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

    def order_snapshot(self, order_id: str) -> OrderSnapshot:
        return self._ledgers[0].snapshot(order_id)

    def order_history(self) -> tuple[OrderSnapshot, ...]:
        return self._ledgers[0].history()

    def portfolio_snapshot(self) -> PortfolioSnapshot:
        return self._ledgers[1].snapshot()

    def portfolio_history(self) -> tuple[PortfolioTransaction, ...]:
        return self._ledgers[1].history()
