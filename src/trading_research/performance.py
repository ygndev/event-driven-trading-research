"""Measure performance from committed trades and observed portfolio equity."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import (
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from statistics import mean, stdev

from ._money import USD_CONTEXT, Money, decimal_amount, to_decimal
from .backtest import BacktestResult, SessionValuation

_SESSIONS_PER_YEAR = 252
_DAYS_PER_YEAR = 365
_STATISTICS_CONTEXT = Context(
    prec=34, traps=[InvalidOperation, DivisionByZero, Overflow]
)


@dataclass(frozen=True, slots=True)
class SessionReturn:
    """Simple return and closing long exposure for one observed session."""

    session_date: date
    simple_return: Decimal
    long_exposure: Decimal


@dataclass(frozen=True, slots=True)
class PerformanceResult:
    session_returns: tuple[SessionReturn, ...]
    total_return: Decimal
    annualized_return: Decimal | None
    annualized_return_reason: str | None
    annualized_volatility: Decimal | None
    sharpe_ratio: Decimal | None
    max_drawdown: Decimal
    turnover: Decimal
    total_commissions: Decimal
    executed_orders: int
    executed_notional: Decimal
    average_long_exposure: Decimal
    risk_free_rate_per_session: Decimal


def _calendar_cagr(
    initial: Decimal,
    final: Decimal,
    sessions: tuple[SessionValuation, ...],
    start_date: date | None,
    end_date: date | None,
) -> tuple[Decimal | None, str | None]:
    if start_date is None and end_date is None:
        return None, "CAGR requires explicit start and end dates"
    if (
        not isinstance(start_date, date)
        or isinstance(start_date, datetime)
        or not isinstance(end_date, date)
        or isinstance(end_date, datetime)
    ):
        raise ValueError("start_date and end_date must both be dates")
    if start_date > sessions[0].session_date:
        raise ValueError("start_date must be on or before the first session")
    if end_date != sessions[-1].session_date:
        raise ValueError("end_date must match the final session")

    elapsed_days = (end_date - start_date).days
    if elapsed_days < _DAYS_PER_YEAR or len(sessions) < 2:
        return None, "CAGR requires at least 365 days and two observations"

    annualizing_exponent = Decimal(_DAYS_PER_YEAR) / elapsed_days
    return (final / initial) ** annualizing_exponent - 1, None


def evaluate_performance(
    result: BacktestResult,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    risk_free_rate_per_session: Money = 0,
) -> PerformanceResult:
    """Use initial cash as the baseline for the first observed return."""
    sessions = result.sessions
    if not sessions:
        raise ValueError("performance requires at least one session valuation")
    if isinstance(risk_free_rate_per_session, bool) or not isinstance(
        risk_free_rate_per_session, (int, float, Decimal)
    ):
        raise ValueError("risk-free rate must be a numeric session return")
    risk_free = to_decimal(risk_free_rate_per_session)
    if not risk_free.is_finite() or risk_free <= -1:
        raise ValueError("risk-free rate must be finite and greater than -1")

    initial = decimal_amount(
        sessions[0].portfolio.initial_cash, "initial cash"
    )
    previous = peak = initial
    max_drawdown = Decimal(0)
    last_date: date | None = None
    equities: list[Decimal] = []
    observations: list[SessionReturn] = []

    with localcontext(USD_CONTEXT):
        commissions = sum(
            (trade.commission for trade in result.transactions), Decimal(0)
        )
        executed_notional = sum(
            (
                abs(to_decimal(trade.fill.price) * trade.fill.quantity)
                for trade in result.transactions
            ),
            Decimal(0),
        )

    with localcontext(_STATISTICS_CONTEXT):
        for session in sessions:
            current_date = session.session_date
            if not isinstance(current_date, date) or isinstance(
                current_date, datetime
            ):
                raise ValueError("session_date must be a date")
            if last_date is not None and current_date <= last_date:
                raise ValueError("session dates must be strictly increasing")
            if session.portfolio.initial_cash != initial:
                raise ValueError(
                    "initial cash must be consistent across sessions"
                )

            equity = decimal_amount(
                session.equity, f"equity on {current_date}"
            )
            cash = decimal_amount(
                session.portfolio.cash, f"cash on {current_date}",
                allow_zero=True,
            )
            if cash > equity:
                raise ValueError(f"cash exceeds equity on {current_date}")
            if bool(session.portfolio.positions) != (cash < equity):
                raise ValueError(
                    f"equity and cash disagree with positions on {current_date}"
                )

            observations.append(
                SessionReturn(
                    current_date,
                    (equity - previous) / previous,
                    (equity - cash) / equity,
                )
            )
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
            equities.append(equity)
            previous = equity
            last_date = current_date

        total_return = (previous - initial) / initial
        annualized_return, annualized_reason = _calendar_cagr(
            initial, previous, sessions, start_date, end_date
        )
        annualized_volatility = sharpe = None
        if len(observations) >= 2:
            returns = [item.simple_return for item in observations]
            sample_volatility = stdev(returns)
            annual_scale = Decimal(_SESSIONS_PER_YEAR).sqrt()
            annualized_volatility = sample_volatility * annual_scale
            if sample_volatility:
                sharpe = (
                    (mean(returns) - risk_free) / sample_volatility
                    * annual_scale
                )

        turnover = executed_notional / mean(equities)
        average_exposure = mean(item.long_exposure for item in observations)

    return PerformanceResult(
        session_returns=tuple(observations),
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_return_reason=annualized_reason,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        turnover=turnover,
        total_commissions=commissions,
        executed_orders=len(result.transactions),
        executed_notional=executed_notional,
        average_long_exposure=average_exposure,
        risk_free_rate_per_session=risk_free,
    )
