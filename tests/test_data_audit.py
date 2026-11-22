from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json

import pytest

from trading_research.corporate_actions import CashDividend, StockSplit
from trading_research.daily_bars import DailyBar, DatasetMetadata, PriceAdjustment
from trading_research.data_audit import AuditDataset, ObservedCoverage, audit_etf_data


FIRST = date(2025, 1, 17)
HOLIDAY = date(2025, 1, 20)
SECOND = date(2025, 1, 21)
LAST = date(2025, 1, 22)
LATER = date(2025, 1, 23)
EXPECTED = (FIRST, SECOND, LAST)
RANGE = (FIRST, LAST)


def metadata(symbol="SPY", *, adjustment=PriceAdjustment.RAW):
    return DatasetMetadata(
        "Synthetic", symbol, adjustment, datetime(2025, 2, 1, tzinfo=UTC),
        source_version="fixture-1",
    )


def bar(day, *, symbol="SPY", opened=100, close=100, volume=1000):
    return DailyBar(symbol, day, opened, max(opened, close), min(opened, close), close, volume)


def dataset(*, symbol="SPY", days=EXPECTED, adjustment=PriceAdjustment.RAW):
    return AuditDataset(
        metadata(symbol, adjustment=adjustment),
        tuple(bar(day, symbol=symbol) for day in days),
        requested_range=RANGE,
        snapshot_id=f"synthetic-{symbol}",
    )


def signatures(report):
    return [
        (row.severity, row.code, row.symbol, row.session) for row in report.findings
    ]


def test_complete_calendar_gap_and_compatible_declarations():
    spy = replace(dataset(), bars=(
        bar(FIRST), bar(SECOND, opened=50, close=50),
        bar(LAST, opened=49.70, close=49.70),
    ))
    result = audit_etf_data(
        (spy,), expected_sessions=EXPECTED, calendar_range=RANGE,
        corporate_actions=(
            StockSplit("SPY", SECOND, 2),
            CashDividend("div", "SPY", LAST, LATER, Decimal("0.30")),
        )
    )

    assert result.coverage == (ObservedCoverage("SPY", FIRST, LAST, 3),)
    assert signatures(result) == [
        ("passed", "ohlcv_valid", "SPY", None),
        ("unverified", "raw_provenance_unverified", "SPY", None),
        ("passed", "metadata_consistent", "SPY", None),
        ("passed", "calendar_matches", "SPY", None),
        ("warning", "payment_after_coverage", "SPY", LATER),
        ("passed", "actions_consistent", None, None),
    ]
    assert all(row.session != HOLIDAY for row in result.findings)


def test_multi_instrument_fixture_reports_exact_planted_defects():
    spy = replace(dataset(), bars=[
        bar(FIRST), bar(LAST), bar(LAST, volume=-1)
    ])
    qqq = dataset(
        symbol="QQQ", days=(FIRST, SECOND, HOLIDAY),
        adjustment=PriceAdjustment.SPLIT_ADJUSTED,
    )
    split = StockSplit("SPY", SECOND, 2)
    actions = (
        split, replace(split, ratio=3),
        CashDividend("qqq-first", "QQQ", FIRST, LATER, 1),
        CashDividend("spy-late", "SPY", LATER, LATER, 1),
    )
    bar_inputs = tuple(spy.bars), tuple(qqq.bars)

    result = audit_etf_data(
        (spy, qqq), expected_sessions=EXPECTED, calendar_range=RANGE,
        corporate_actions=actions,
    )

    assert signatures(result) == [
        ("blocking", "duplicate_session", "SPY", LAST),
        ("blocking", "invalid_ohlcv", "SPY", LAST),
        ("unverified", "raw_provenance_unverified", "SPY", None),
        ("passed", "metadata_consistent", "SPY", None),
        ("blocking", "missing_session", "SPY", SECOND),
        ("blocking", "unsorted_sessions", "QQQ", HOLIDAY),
        ("blocking", "incompatible_adjustment", "QQQ", None),
        ("blocking", "missing_session", "QQQ", LAST),
        ("blocking", "unexpected_session", "QQQ", HOLIDAY),
        ("blocking", "action_without_bar", "SPY", SECOND),
        ("blocking", "duplicate_split", "SPY", SECOND),
        ("blocking", "action_without_bar", "SPY", SECOND),
        ("blocking", "dividend_without_preceding_close", "QQQ", FIRST),
        ("warning", "payment_after_coverage", "QQQ", LATER),
        ("blocking", "action_outside_coverage", "SPY", LATER),
    ]
    assert result.coverage[-1] == ObservedCoverage("QQQ", FIRST, SECOND, 3)
    assert (tuple(spy.bars), tuple(qqq.bars)) == bar_inputs
    assert tuple(action.symbol for action in actions) == ("SPY", "SPY", "QQQ", "SPY")
    assert result == audit_etf_data(
        (spy, qqq), expected_sessions=EXPECTED, calendar_range=RANGE,
        corporate_actions=actions,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(result.findings[0], "severity", "passed")
    with pytest.raises(FrozenInstanceError):
        setattr(result, "coverage", ())


def test_no_calendar_surfaces_unverified_completeness_without_guessing_weekdays():
    report = audit_etf_data((dataset(days=(FIRST, LAST)),))

    assert ("unverified", "calendar_unverified", "SPY", None) in signatures(report)
    assert not any(row.severity == "blocking" for row in report.findings)
    assert all(row.code != "missing_session" for row in report.findings)


def test_partial_calendar_checks_known_dates_without_inventing_the_edges():
    report = audit_etf_data(
        (dataset(days=(FIRST, LAST)),), expected_sessions=(SECOND, LAST)
    )

    assert ("unverified", "calendar_partial", "SPY", None) in signatures(report)
    assert ("blocking", "missing_session", "SPY", SECOND) in signatures(report)
    assert all(row.code not in ("unexpected_session", "calendar_matches")
               for row in report.findings)


@pytest.mark.parametrize("days", [(FIRST, FIRST), (LAST, FIRST), (FIRST, datetime(2025, 1, 22))])
def test_malformed_calendar_is_a_blocking_input(days):
    result = audit_etf_data((dataset(),), expected_sessions=days, calendar_range=RANGE)

    assert ("blocking", "invalid_calendar", None, None) in signatures(result)
    assert ("unverified", "calendar_unverified", "SPY", None) in signatures(result)


@pytest.mark.parametrize(("changes", "fragment"), [
    ({"open": None}, "open must be"),
    ({"close": float("nan")}, "close must be"),
    ({"low": 0}, "low must be"),
    ({"high": float("inf")}, "high must be"),
    ({"high": 99, "low": 98}, "high is below open"),
    ({"low": 101}, "low exceeds high"),
    ({"volume": 10.5}, "volume must be"),
    ({"volume": None}, "volume must be"),
])
def test_reports_existing_ohlcv_contract_errors_with_session(changes, fragment):
    spy = dataset()
    broken = replace(spy.bars[1], **changes)

    report = audit_etf_data((replace(spy, bars=(spy.bars[0], broken, spy.bars[2])),))

    error, = [row for row in report.findings if row.code == "invalid_ohlcv"]
    assert (error.severity, error.symbol, error.session) == ("blocking", "SPY", SECOND)
    assert fragment in error.message


def test_empty_and_mislabeled_datasets_have_explicit_findings():
    empty = replace(dataset(), bars=())
    wrong_symbol = replace(dataset(symbol="QQQ"), bars=(bar(FIRST, symbol="IWM"),))

    report = audit_etf_data((empty, wrong_symbol))

    assert ("blocking", "empty_dataset", "SPY", None) in signatures(report)
    assert ("blocking", "symbol_mismatch", "QQQ", FIRST) in signatures(report)
    assert report.coverage[0] == ObservedCoverage("SPY", None, None, 0)
    assert report.coverage[1] == ObservedCoverage("QQQ", None, None, 1)


def test_metadata_missing_and_inconsistent_dates_are_blocking():
    spy = replace(dataset(), metadata=None)
    non_utc = datetime(2025, 1, 22, 1, tzinfo=timezone(timedelta(hours=2)))
    qqq = replace(
        dataset(symbol="QQQ"),
        metadata=replace(metadata("QQQ"), retrieved_at=non_utc),
        requested_range=(SECOND, SECOND),
    )

    report = audit_etf_data((spy, qqq))

    assert ("blocking", "missing_metadata", "SPY", None) in signatures(report)
    assert ("blocking", "retrieval_not_utc", "QQQ", None) in signatures(report)
    assert ("blocking", "retrieval_before_session", "QQQ", LAST) in signatures(report)
    assert ("blocking", "outside_requested_range", "QQQ", FIRST) in signatures(report)
    assert ("blocking", "outside_requested_range", "QQQ", LAST) in signatures(report)
    assert all(row.code != "metadata_consistent" for row in report.findings)


def test_total_return_adjusted_declaration_is_also_blocking():
    report = audit_etf_data((dataset(adjustment=PriceAdjustment.TOTAL_RETURN_ADJUSTED),))
    assert ("blocking", "incompatible_adjustment", "SPY", None) in signatures(report)


@pytest.mark.parametrize(("field", "bad_value", "message"), [
    ("ratio", 0, "split ratio"),
    ("amount_per_share", Decimal("NaN"), "dividend amount"),
    ("payment_date", FIRST, "on or after ex_date"),
])
def test_rechecks_malformed_corporate_action_records(field, bad_value, message):
    action = (
        StockSplit("SPY", SECOND, 2) if field == "ratio"
        else CashDividend("invalid", "SPY", SECOND, LAST, 1)
    )
    object.__setattr__(action, field, bad_value)

    result = audit_etf_data((dataset(),), corporate_actions=(action,))

    invalid, = [row for row in result.findings if row.code == "invalid_action"]
    assert (invalid.severity, invalid.symbol, invalid.session) == ("blocking", "SPY", SECOND)
    assert message in invalid.message


def test_action_symbol_and_dividend_id_checks_follow_backtest_policy():
    actions = (
        CashDividend("d", "SPY", SECOND, LAST, 1),
        CashDividend("d", "QQQ", LAST, LATER, 1),
        StockSplit("IWM", SECOND, 2),
        CashDividend("ok", "QQQ", SECOND, LAST, 1),
    )

    result = audit_etf_data(
        (dataset(), dataset(symbol="QQQ")), corporate_actions=actions,
        expected_sessions=EXPECTED,
    )

    assert [row for row in signatures(result) if row[0] == "blocking"] == [
        ("blocking", "duplicate_dividend", "QQQ", LAST),
        ("blocking", "action_without_dataset", "IWM", SECOND),
    ]


def test_no_auto_repair_of_large_overnight_or_closing_moves():
    spy = replace(dataset(), bars=(
        bar(FIRST), bar(SECOND, opened=50, close=140),
        bar(LAST, opened=140, close=140),
    ))

    result = audit_etf_data((spy,), expected_sessions=EXPECTED, price_move_threshold=Decimal("0.30"))

    assert [row for row in signatures(result) if row[0] == "warning"] == [
        ("warning", "overnight_move", "SPY", SECOND),
        ("warning", "close_move", "SPY", SECOND),
    ]
    assert not any(row.severity == "blocking" for row in result.findings)
    assert spy.bars[1].open == 50 and spy.bars[1].close == 140
    unchecked = audit_etf_data((spy,))
    assert not any(row.severity == "warning" for row in unchecked.findings)


def test_existing_manifest_digest_checks_source_bytes_independently_of_bars(tmp_path):
    source = tmp_path / "provider.csv"
    content = b"source bytes\r\n"
    source.write_bytes(content)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "source_file": source.name, "source_sha256": sha256(content).hexdigest()
    }), encoding="utf-8")
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    spy = replace(
        dataset(), source_path=str(manifest.parent / stored["source_file"]),
        source_sha256=stored["source_sha256"],
    )

    good = audit_etf_data((spy,))
    assert ("passed", "source_hash_matches", "SPY", None) in signatures(good)
    assert ("unverified", "raw_provenance_unverified", "SPY", None) in signatures(good)
    source.write_bytes(b"source bytes\n")
    changed = audit_etf_data((spy,))
    assert ("blocking", "source_hash_mismatch", "SPY", None) in signatures(changed)
    assert ("passed", "ohlcv_valid", "SPY", None) in signatures(changed)
    assert source.read_bytes() == b"source bytes\n"
