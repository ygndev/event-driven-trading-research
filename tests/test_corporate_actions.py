from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
from decimal import Decimal, Inexact, localcontext
from fractions import Fraction

import pytest

from trading_research.corporate_actions import CashDividend, StockSplit
from trading_research.orders import Fill, OrderSide
from trading_research.portfolio import (
    PortfolioLedger,
    PortfolioRejection,
    mark_to_market,
)


PREVIOUS = date(2025, 1, 3)
EX_DATE = date(2025, 1, 6)
PAY_DATE = date(2025, 1, 10)


def buy(ledger, quantity, price, *, session=PREVIOUS, order_id="before"):
    fill = Fill(order_id, "SPY", session, OrderSide.BUY, quantity, price)
    ledger.apply_fill(fill)
    return fill


def dividend(amount=Decimal("1.25"), *, action_id="distribution", payment=PAY_DATE):
    return CashDividend(action_id, "SPY", EX_DATE, payment, amount)


def accrue(ledger, action):
    return ledger.accrue_dividend(
        action, session=EX_DATE, preceding_session=PREVIOUS
    )


@pytest.mark.parametrize(
    "ratio",
    [0, -2, True, "2", 0.5, float("inf"), Decimal("NaN"), Decimal("-Infinity")],
)
def test_split_ratio_requires_a_positive_exact_value(ratio):
    with pytest.raises(ValueError, match="split ratio"):
        StockSplit("SPY", EX_DATE, ratio)


@pytest.mark.parametrize(
    "amount",
    [0, -1, True, "1", float("nan"), float("inf"), Decimal("sNaN")],
)
def test_cash_dividend_requires_a_positive_finite_amount(amount):
    with pytest.raises(ValueError, match="dividend amount per share"):
        dividend(amount)


def test_action_dates_and_identifiers_are_validated():
    with pytest.raises(ValueError, match="effective_session must be a date"):
        StockSplit("SPY", datetime(2025, 1, 6), 2)
    with pytest.raises(ValueError, match="symbol"):
        StockSplit(" ", EX_DATE, 2)
    with pytest.raises(ValueError, match="action_id"):
        dividend(action_id="")
    with pytest.raises(ValueError, match="payment_date must be on or after ex_date"):
        dividend(payment=PREVIOUS)


def test_forward_and_reverse_splits_preserve_value_with_matching_raw_marks():
    ledger = PortfolioLedger(2500)
    original_fill = buy(ledger, 10, 200)
    before = ledger.snapshot()
    forward = StockSplit("SPY", EX_DATE, 2)
    reverse = StockSplit("SPY", PAY_DATE, Decimal("0.5"))

    doubled = ledger.apply_split(forward)
    after_forward = ledger.snapshot()
    halved = ledger.apply_split(reverse)
    after_reverse = ledger.snapshot()

    assert forward.ratio == Fraction(2)
    assert isinstance(forward.ratio, Fraction)
    assert reverse.ratio == Fraction(1, 2)
    assert (doubled.shares_before, doubled.shares_after) == (10, 20)
    assert (halved.shares_before, halved.shares_after) == (20, 10)
    assert [state.positions for state in (before, after_forward, after_reverse)] == [
        {"SPY": 10}, {"SPY": 20}, {"SPY": 10}
    ]
    for state, price in ((before, 200), (after_forward, 100), (after_reverse, 200)):
        assert state.cash == 500
        assert state.cumulative_commissions == 0
        assert mark_to_market(state, {"SPY": price}) == 2500
    assert ledger.split_history() == (doubled, halved)
    trade, = ledger.history()
    assert trade.fill is original_fill


def test_split_for_an_unheld_symbol_records_a_zero_quantity_once():
    ledger = PortfolioLedger(10)
    buy(ledger, 1, 10)
    before = ledger.snapshot()
    split = StockSplit("QQQ", EX_DATE, 2)

    applied = ledger.apply_split(split)

    assert applied.shares_before == applied.shares_after == 0
    assert ledger.snapshot() == before
    assert ledger.split_history() == (applied,)
    with pytest.raises(ValueError, match="split already applied"):
        ledger.apply_split(replace(split, ratio=3))
    assert ledger.snapshot() == before
    assert ledger.split_history() == (applied,)


def test_fractional_split_is_atomic_and_exact_thirds_can_be_booked():
    ledger = PortfolioLedger(100)
    buy(ledger, 3, 10)
    before, trades = ledger.snapshot(), ledger.history()

    with pytest.raises(PortfolioRejection, match="fractional shares"):
        ledger.apply_split(StockSplit("SPY", EX_DATE, Fraction(1, 2)))

    assert ledger.snapshot() == before
    assert ledger.history() == trades
    assert ledger.split_history() == ()
    ledger.apply_split(StockSplit("SPY", EX_DATE, Fraction(1, 3)))
    assert ledger.snapshot().positions == {"SPY": 1}
    assert mark_to_market(ledger.snapshot(), {"SPY": 30}) == 100


def test_split_cannot_be_applied_after_the_symbols_effective_session_fill():
    ledger = PortfolioLedger(100)
    buy(ledger, 2, 10, session=EX_DATE)
    before = ledger.snapshot()

    with pytest.raises(ValueError, match="split must precede"):
        ledger.apply_split(StockSplit("SPY", EX_DATE, 2))

    assert ledger.snapshot() == before
    assert ledger.split_history() == ()


def test_fill_cannot_be_backdated_across_an_applied_split():
    ledger = PortfolioLedger(100)
    buy(ledger, 2, 10)
    ledger.apply_split(StockSplit("SPY", EX_DATE, 2))
    before, trades = ledger.snapshot(), ledger.history()

    with pytest.raises(ValueError, match="before an applied split"):
        buy(ledger, 1, 10, order_id="late")

    assert ledger.snapshot() == before
    assert ledger.history() == trades


def test_dividend_entitlement_and_payment_conserve_equity_at_unchanged_marks():
    ledger = PortfolioLedger(2500)
    buy(ledger, 10, 200)
    ledger.record_close(PREVIOUS)
    action = dividend()
    previous = ledger.snapshot()
    assert mark_to_market(previous, {"SPY": 200}) == 2500

    buy(ledger, 2, Decimal("198.75"), session=EX_DATE, order_id="at-open")
    available_before_accrual = ledger.snapshot().cash
    receivable = accrue(ledger, action)
    ex = ledger.snapshot()

    assert isinstance(action.amount_per_share, Decimal)
    assert receivable.dividend is action
    assert receivable.prior_close_session == PREVIOUS
    assert receivable.entitled_shares == 10
    assert receivable.total_amount == Decimal("12.50")
    assert ex.positions == {"SPY": 12}
    assert ex.cash == available_before_accrual == Decimal("102.50")
    assert ex.unpaid_receivables == (receivable,)
    assert mark_to_market(ex, {"SPY": Decimal("198.75")}) == Decimal("2487.50")
    assert mark_to_market(
        ex, {"SPY": Decimal("198.75")}, include_receivables=True
    ) == 2500

    trades = ledger.history()
    payment = ledger.pay_dividend(action.action_id, session=PAY_DATE)
    paid = ledger.snapshot()

    assert paid.cash == payment.cash_after == Decimal("115.00")
    assert paid.positions == ex.positions
    assert paid.unpaid_receivables == ()
    assert ex.unpaid_receivables == (receivable,)
    assert ledger.dividend_payments() == (payment,)
    assert payment.session_date == PAY_DATE
    assert payment.receivable is receivable
    assert ledger.history() == trades
    assert paid.cumulative_commissions == ex.cumulative_commissions == 0
    assert mark_to_market(
        paid, {"SPY": Decimal("198.75")}, include_receivables=True
    ) == mark_to_market(paid, {"SPY": Decimal("198.75")}) == 2500


def test_buying_the_ex_date_open_cannot_create_dividend_entitlement():
    ledger = PortfolioLedger(100)
    ledger.record_close(PREVIOUS)
    buy(ledger, 2, 10, session=EX_DATE)
    before = ledger.snapshot()
    action = dividend()

    assert accrue(ledger, action) is None
    with pytest.raises(ValueError, match="entitlement already processed"):
        accrue(ledger, action)
    with pytest.raises(ValueError, match="no entitlement"):
        ledger.pay_dividend(action.action_id, session=PAY_DATE)
    assert ledger.snapshot() == before
    assert ledger.dividend_payments() == ()


def test_dividend_is_exact_and_independent_of_ex_date_sale_and_ambient_precision():
    ledger = PortfolioLedger(100)
    buy(ledger, 3, 30)
    ledger.record_close(PREVIOUS)
    ledger.apply_fill(Fill("sell", "SPY", EX_DATE, OrderSide.SELL, 3, 29))
    action = dividend(Decimal("0.1234"))

    with localcontext() as context:
        context.prec = 2
        receivable = accrue(ledger, action)
        ex = ledger.snapshot()
        assert mark_to_market(ex, include_receivables=True) == Decimal("97.3702")
        ledger.pay_dividend(action.action_id, session=PAY_DATE)

    assert receivable.entitled_shares == 3
    assert receivable.total_amount == Decimal("0.3702")
    assert ex.positions == {}
    assert ex.cash == 97
    assert ledger.snapshot().cash == Decimal("97.3702")


def test_missing_or_mismatched_close_is_rejected_without_consuming_the_action():
    ledger = PortfolioLedger(100)
    buy(ledger, 2, 10)
    action = dividend()
    before = ledger.snapshot()

    with pytest.raises(ValueError, match="missing recorded holdings"):
        accrue(ledger, action)
    ledger.record_close(PREVIOUS)
    with pytest.raises(ValueError, match="missing recorded holdings"):
        ledger.accrue_dividend(
            action, session=EX_DATE, preceding_session=date(2025, 1, 2)
        )
    with pytest.raises(ValueError, match="preceding_session must be before ex_date"):
        ledger.accrue_dividend(action, session=EX_DATE, preceding_session=EX_DATE)
    with pytest.raises(ValueError, match="recorded on ex_date"):
        ledger.accrue_dividend(
            action, session=PREVIOUS, preceding_session=date(2025, 1, 2)
        )

    assert ledger.snapshot() == before
    assert accrue(ledger, action).entitled_shares == 2


def test_close_capture_cannot_relabel_ex_date_holdings_as_prior_or_accept_late_fills():
    ledger = PortfolioLedger(100)
    buy(ledger, 2, 10, session=EX_DATE)

    with pytest.raises(ValueError, match="before an applied position change"):
        ledger.record_close(PREVIOUS)
    ledger.record_close(EX_DATE)
    before, trades = ledger.snapshot(), ledger.history()
    with pytest.raises(ValueError, match="strictly increasing"):
        ledger.record_close(EX_DATE)
    with pytest.raises(ValueError, match="on or before the recorded close"):
        buy(ledger, 1, 10, session=EX_DATE, order_id="after-close")

    assert ledger.snapshot() == before
    assert ledger.history() == trades


def test_wrong_payment_dates_and_duplicate_operations_are_atomic():
    ledger = PortfolioLedger(100)
    buy(ledger, 4, 10)
    ledger.record_close(PREVIOUS)
    action = dividend()
    receivable = accrue(ledger, action)
    ex = ledger.snapshot()

    with pytest.raises(ValueError, match="entitlement already processed"):
        accrue(ledger, replace(action, amount_per_share=2))
    with pytest.raises(ValueError, match="no entitlement"):
        ledger.pay_dividend("unknown", session=PAY_DATE)
    for day in (EX_DATE, date(2025, 1, 13)):
        with pytest.raises(ValueError, match="payment must occur on payment_date"):
            ledger.pay_dividend(action.action_id, session=day)
        assert ledger.snapshot() == ex
        assert ledger.dividend_payments() == ()

    payment = ledger.pay_dividend(action.action_id, session=PAY_DATE)
    paid = ledger.snapshot()
    with pytest.raises(ValueError, match="already paid"):
        ledger.pay_dividend(action.action_id, session=PAY_DATE)

    assert payment.receivable is receivable
    assert ledger.snapshot() == paid
    assert paid.cash == 65
    assert ledger.dividend_payments() == (payment,)


def test_multiple_receivables_pay_independently_including_same_day():
    ledger = PortfolioLedger(100)
    buy(ledger, 2, 10)
    ledger.record_close(PREVIOUS)
    early = dividend(payment=EX_DATE, action_id="same-day")
    later = dividend(2, action_id="later")
    first, second = accrue(ledger, early), accrue(ledger, later)
    assert ledger.snapshot().unpaid_receivables == (first, second)

    ledger.pay_dividend(early.action_id, session=EX_DATE)

    state = ledger.snapshot()
    assert state.cash == Decimal("82.50")
    assert state.unpaid_receivables == (second,)
    assert mark_to_market(
        state, {"SPY": 10}, include_receivables=True
    ) == Decimal("106.50")


def test_receivable_becomes_spendable_on_payment_date_only():
    ledger = PortfolioLedger(100)
    buy(ledger, 10, 10)
    ledger.record_close(PREVIOUS)
    action = dividend()
    accrue(ledger, action)
    unpaid = ledger.snapshot()

    with pytest.raises(PortfolioRejection, match="insufficient cash"):
        buy(ledger, 1, 10, session=EX_DATE, order_id="new")
    assert ledger.snapshot() == unpaid

    ledger.pay_dividend(action.action_id, session=PAY_DATE)
    paid = ledger.snapshot()
    with pytest.raises(ValueError, match="before an applied dividend payment"):
        buy(ledger, 1, 10, session=EX_DATE, order_id="new")
    assert ledger.snapshot() == paid

    buy(ledger, 1, 10, session=PAY_DATE, order_id="new")
    assert ledger.snapshot().cash == Decimal("2.50")
    assert ledger.snapshot().positions == {"SPY": 11}


def test_inexact_accrual_leaves_the_action_available_for_retry():
    ledger = PortfolioLedger(100)
    buy(ledger, 3, 10)
    ledger.record_close(PREVIOUS)
    before = ledger.snapshot()

    with pytest.raises(Inexact):
        accrue(ledger, dividend(Decimal("0." + "9" * 34)))

    assert ledger.snapshot() == before
    assert accrue(ledger, dividend()).total_amount == Decimal("3.75")


def test_inexact_payment_preserves_unpaid_receivable_and_audit():
    ledger = PortfolioLedger(Decimal("1e35"))
    buy(ledger, 1, 10**34)
    ledger.record_close(PREVIOUS)
    action = dividend(1)
    receivable = accrue(ledger, action)
    before = ledger.snapshot()

    with pytest.raises(Inexact):
        ledger.pay_dividend(action.action_id, session=PAY_DATE)

    assert ledger.snapshot() == before
    assert before.unpaid_receivables == (receivable,)
    assert ledger.dividend_payments() == ()


def test_ledger_copy_preserves_action_records_and_copies_are_independent():
    ledger = PortfolioLedger(100)
    buy(ledger, 2, 10, session=date(2025, 1, 2))
    split = ledger.apply_split(StockSplit("SPY", PREVIOUS, 2))
    ledger.record_close(PREVIOUS)
    action = dividend()
    receivable = accrue(ledger, action)
    extended = dividend(2, action_id="extended")
    accrue(ledger, extended)
    payment = ledger.pay_dividend(action.action_id, session=PAY_DATE)

    staged = ledger._copy()
    assert staged.snapshot() == ledger.snapshot()
    assert staged.split_history() == (split,)
    assert staged.dividend_payments() == (payment,)
    assert staged.history() == ledger.history()
    with pytest.raises(ValueError, match="already paid"):
        staged.pay_dividend(action.action_id, session=PAY_DATE)
    staged.pay_dividend(extended.action_id, session=PAY_DATE)
    assert staged.snapshot().unpaid_receivables == ()
    assert len(ledger.snapshot().unpaid_receivables) == 1
    assert ledger.dividend_payments() == (payment,)

    for record, field in (
        (split.split, "ratio"), (split, "shares_after"),
        (action, "amount_per_share"), (receivable, "total_amount"),
        (payment, "cash_after"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, 0)
