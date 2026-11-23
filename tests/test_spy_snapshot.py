from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
import socket
from zoneinfo import ZoneInfo

import pytest

from trading_research.massive_history import parse_dividend
from trading_research.spy_snapshot import (
    ReviewedEvidence,
    SavedResponse,
    SnapshotFile,
    SnapshotReview,
    SpySnapshot,
    preflight_spy_snapshot,
)

HOST = "https://api.massive.com"
ZONE = "America/New_York"
START, END = date(2024, 3, 8), date(2024, 3, 13)
CAPTURED = datetime(2024, 3, 15, tzinfo=UTC)
FIRST = date(2024, 3, 11)
NEXT = f"{HOST}/stocks/v1/dividends?cursor=page-two"


def save(root, path, payload):
    content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    (root / path).write_bytes(content)
    return SnapshotFile(path, sha256(content).hexdigest())


def bar(day):
    midnight = datetime.combine(day, time.min, ZoneInfo(ZONE))
    return dict(t=int(midnight.timestamp()) * 1000, o=100, h=105, l=98, c=102, v=601)


def bars_payload(days):
    return dict(status="OK", ticker="SPY", adjusted=False, request_id="test-bars",
                queryCount=len(days), resultsCount=len(days), results=[bar(day) for day in days])


def action_payload(results, request_id="test-page", **extra):
    return dict(status="OK", request_id=request_id, results=results, **extra)


def dividend(**changes):
    row = dict(ticker="SPY", id="cash-1", distribution_type="recurring", currency="USD",
               ex_dividend_date="2024-03-12", pay_date="2024-03-16", record_date="2024-03-13",
               cash_amount=0.12345, split_adjusted_cash_amount=0.25, frequency=4)
    row.update(changes)
    return row


def stream(root, kind, rows, **extra):
    field = "execution_date" if kind == "splits" else "ex_dividend_date"
    url = (f"{HOST}/stocks/v1/{kind}?ticker=SPY&{field}.gte={START}&{field}.lte={END}"
           f"&sort={field}.asc")
    return (SavedResponse(save(root, f"{kind}.json", action_payload(rows, **extra)), url, CAPTURED),)


def attest(root, manifest, *, ids=frozenset({"cash-1"}), split_count=1, dividend_count=1):
    # Stand-ins for externally reviewed records, never an authentic data review.
    records = {
        "rights": {},
        "calendar": {},
        "raw_prices": {},
        "actions": {"split_count": split_count, "dividend_count": dividend_count},
        "availability": {"timing": "fixture timing", "revisions": "fixture revision policy"},
    }
    return SnapshotReview(
        manifest.fingerprint(), "fixture reviewer",
        tuple(ReviewedEvidence(gate, save(root, f"{gate}-review.json",
                                        dict(source=f"fixture {gate}", conclusion="confirmed", **record)))
              for gate, record in records.items()),
        ids, "new_per_old", "prior_close",
    )


@pytest.fixture
def capture(tmp_path):
    days = [START, FIRST, date(2024, 3, 12), END]
    reference = save(tmp_path, "bars.json", bars_payload(days))
    url = f"{HOST}/v2/aggs/ticker/SPY/range/1/day/{START}/{END}?adjusted=false&sort=asc"
    calendar = save(tmp_path, "calendar.json", dict(
        source="fixture calendar", version="fixture-1", session_zone=ZONE,
        start=str(START), end=str(END), sessions=[str(day) for day in days],
    ))
    split = dict(ticker="SPY", id="split-1", adjustment_type="forward_split",
                 execution_date=str(FIRST), split_from=0.3, split_to=0.7)
    manifest = SpySnapshot(
        "synthetic-spy", "fixture-1", (START, END), (START, END), (FIRST, END),
        SavedResponse(reference, url, CAPTURED), calendar, stream(tmp_path, "splits", [split]),
        stream(tmp_path, "dividends", [dividend()]), warmup_sessions=1,
    )
    return tmp_path, manifest, days


def assert_blocked(result, code):
    assert result.status == "BLOCKED"
    assert result.dataset is None and not result.actions
    assert any(row.severity == "blocking" and row.code == code for row in result.findings)


def test_complete_synthetic_snapshot_reproduces_without_empirical_approval(capture, monkeypatch):
    root, manifest, _ = capture
    review = attest(root, manifest)

    def reject(*args, **kwargs):
        pytest.fail("network access during local preflight")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "getaddrinfo", reject)
    result = preflight_spy_snapshot(root, manifest, review=review)
    assert result == preflight_spy_snapshot(root, manifest, review=review)
    assert_blocked(result, "synthetic")
    assert len(result.evidence) == 5
    assert result.input_hashes[0] == ("bars", manifest.bars.file.sha256)
    assert {row.code for row in result.findings} >= {
        "source_hash_matches", "calendar_matches", "actions_consistent", "raw_provenance_unverified",
    }


@pytest.mark.parametrize(("kind", "code"), [
    ("corrupt", "hash_mismatch"), ("invalid_json", "invalid_json"), ("duplicate_field", "invalid_json"),
    ("adjusted_request", "raw_request"), ("adjusted_response", "invalid_bars"),
    ("duplicate_bar", "invalid_bars"), ("missing_bar", "missing_session"), ("extra_bar", "unexpected_session"),
])
def test_invalid_bars_block_before_exposing_inputs(capture, kind, code):
    root, manifest, days = capture
    ref = manifest.bars
    payload = bars_payload(days)
    if kind == "corrupt":
        (root / ref.file.path).write_bytes(b"changed")
    elif kind == "adjusted_request":
        ref = replace(ref, request_url=ref.request_url.replace("adjusted=false", "adjusted=true"))
    else:
        if kind == "invalid_json":
            payload = b"{"
        elif kind == "duplicate_field":
            payload = b'{"adjusted": true, "adjusted": false}'
        elif kind == "adjusted_response":
            payload["adjusted"] = True
        elif kind == "duplicate_bar":
            payload = bars_payload([days[0], days[1], days[1], *days[2:]])
        elif kind == "missing_bar":
            payload = bars_payload([days[0], days[1], days[-1]])
        elif kind == "extra_bar":
            payload = bars_payload([days[0], date(2024, 3, 9), *days[1:]])
        ref = replace(ref, file=save(root, "bars.json", payload))
    manifest = replace(manifest, bars=ref)
    assert_blocked(preflight_spy_snapshot(root, manifest, review=attest(root, manifest)), code)


@pytest.mark.parametrize(("key", "value", "code"), [
    ("source", "", "calendar_evidence"), ("version", None, "calendar_evidence"),
    ("start", "2024-03-11", "calendar_partial"), ("session_zone", "UTC", "calendar_evidence"),
    ("sessions", [str(START), str(FIRST), str(FIRST), str(END)], "invalid_calendar"),
    ("sessions", [str(FIRST), str(END)], "warmup"),
])
def test_calendar_must_be_complete_and_versioned(capture, key, value, code):
    root, manifest, _ = capture
    payload = json.loads((root / "calendar.json").read_bytes())
    payload[key] = value
    manifest = replace(manifest, calendar=save(root, "calendar.json", payload))
    assert_blocked(preflight_spy_snapshot(root, manifest, review=attest(root, manifest)), code)


@pytest.mark.parametrize("scenario", ["no_pages", "truncated", "wrong_cursor", "bad_count", "extra_page"])
def test_action_chain_requires_every_page(capture, scenario):
    root, manifest, _ = capture
    pages = stream(root, "dividends", [], next_url=NEXT)
    expected = "pagination"
    if scenario == "no_pages":
        pages = ()
    elif scenario == "bad_count":
        pages = stream(root, "dividends", [dividend()], resultsCount=3)
        expected = "action_count"
    elif scenario in {"wrong_cursor", "extra_page"}:
        if scenario == "extra_page":
            pages = stream(root, "dividends", [])
        pages += (SavedResponse(save(root, "page.json", action_payload([dividend()], "second")),
                                NEXT.replace("page-two", "wrong"), CAPTURED),)
    manifest = replace(manifest, dividends=pages)
    assert_blocked(preflight_spy_snapshot(root, manifest, review=attest(root, manifest)), expected)


def test_complete_action_pagination(capture):
    root, manifest, _ = capture
    first = stream(root, "dividends", [], next_url=NEXT)
    second = SavedResponse(save(root, "page.json", action_payload([dividend()], "second")), NEXT, CAPTURED)
    manifest = replace(manifest, dividends=(*first, second))
    result = preflight_spy_snapshot(root, manifest, review=attest(root, manifest))
    assert_blocked(result, "synthetic")
    assert ("dividends page 2", second.file.sha256) in result.input_hashes


@pytest.mark.parametrize("change", [
    {"pay_date": None}, {"cash_amount": None}, {"currency": "CAD"},
    {"distribution_type": "special"}, {"distribution_type": "supplemental"},
    {"distribution_type": "unknown"}, {"distribution_type": None},
    {"ex_dividend_date": "2024-03-31"}, {"cash_amount": -1},
])
def test_unsupported_dividend_never_becomes_an_action(capture, change):
    root, manifest, _ = capture
    manifest = replace(manifest, dividends=stream(root, "dividends", [dividend(**change)]))
    code = "action_coverage" if "ex_dividend_date" in change else "unsupported_dividend"
    assert_blocked(preflight_spy_snapshot(root, manifest, review=attest(root, manifest)), code)


def test_dividend_uses_exact_original_amount_and_actual_payment():
    record = dividend(cash_amount=Decimal("0.123456789012345678901"))
    cash = parse_dividend(record, expected_symbol="SPY", ordinary_confirmed=True, share_basis="prior_close")
    assert cash.amount_per_share == Decimal("0.123456789012345678901")
    assert cash.payment_date == date(2024, 3, 16)
    assert cash.action_id == "cash-1"
    with pytest.raises(ValueError, match="prior-close"):
        parse_dividend(record, expected_symbol="SPY", ordinary_confirmed=True, share_basis="adjusted")
    with pytest.raises(ValueError, match="classification"):
        parse_dividend(record, expected_symbol="SPY", ordinary_confirmed=False, share_basis="prior_close")


def test_conflicting_revisions_block(capture):
    root, manifest, _ = capture
    rows = [dividend(), dividend(cash_amount=5)]
    manifest = replace(manifest, dividends=stream(root, "dividends", rows))
    assert_blocked(preflight_spy_snapshot(root, manifest, review=attest(root, manifest)), "action_revision")


def test_empty_action_streams_need_a_bound_review(capture):
    root, manifest, _ = capture
    manifest = replace(manifest, splits=stream(root, "splits", []), dividends=stream(root, "dividends", []))
    assert_blocked(preflight_spy_snapshot(root, manifest), "review_required")
    review = attest(root, manifest, ids=frozenset(), split_count=0, dividend_count=0)
    assert_blocked(preflight_spy_snapshot(root, manifest, review=review), "synthetic")
    assert_blocked(preflight_spy_snapshot(root, replace(manifest, vintage="changed"),
                                         review=review), "review_required")


def test_review_requires_verified_share_basis_and_matching_counts(capture):
    root, manifest, _ = capture
    review = attest(root, manifest)
    assert_blocked(preflight_spy_snapshot(root, manifest, review=replace(review, split_share_basis="old")),
                   "share_basis")
    assert_blocked(preflight_spy_snapshot(root, manifest, review=replace(review, evidence=review.evidence[:-1])),
                   "review_evidence")
    assert_blocked(preflight_spy_snapshot(root, manifest, review=attest(root, manifest, dividend_count=0)),
                   "action_review")


def test_request_diagnostics_do_not_reveal_credentials(capture):
    root, manifest, _ = capture
    token = "sample-secret-do-not-print"
    manifest = replace(manifest, bars=replace(manifest.bars, request_url=manifest.bars.request_url + "&apiKey=" + token))
    result = preflight_spy_snapshot(root, manifest)
    assert_blocked(result, "raw_request")
    assert token not in repr(result)


def test_ready_contract_with_test_doubles_for_a_full_review(capture):
    root, manifest, _ = capture
    # Exercise the empirical path with artificial inputs and attestation; this is not a real review.
    warmup = [START + timedelta(days=i) for i in range(63)]
    scoring = START + timedelta(days=63)
    last = scoring + timedelta(days=2)
    days = [*warmup, scoring, last]
    bar_ref = save(root, "bars.json", bars_payload(days))
    calendar = save(root, "calendar.json", dict(
        source="test schedule", version="test version", session_zone=ZONE,
        start=str(START), end=str(last), sessions=[str(day) for day in days],
    ))
    retrieved = CAPTURED + timedelta(days=100)
    split = dict(ticker="SPY", id="split-1", adjustment_type="forward_split",
                 execution_date=str(scoring), split_from=0.3, split_to=0.7)
    cash = dividend(ex_dividend_date=str(last), pay_date=str(last + timedelta(days=1)))
    splits, = stream(root, "splits", [split])
    dividends, = stream(root, "dividends", [cash])
    manifest = replace(
        manifest, requested_range=(START, last), observed_range=(START, last),
        scoring_range=(scoring, last), calendar=calendar, synthetic=False, warmup_sessions=63,
        bars=SavedResponse(bar_ref, manifest.bars.request_url.replace(str(END), str(last)), retrieved),
        splits=(replace(splits, request_url=splits.request_url.replace(str(END), str(last)), retrieved_at=retrieved),),
        dividends=(replace(dividends, request_url=dividends.request_url.replace(str(END), str(last)),
                           retrieved_at=retrieved),),
    )
    result = preflight_spy_snapshot(root, manifest, review=attest(root, manifest))
    assert result.status == "READY"
    assert len(result.dataset.bars) == 65
    assert result.actions[0].ratio == Fraction(7, 3)
    assert result.actions[1].amount_per_share == Decimal("0.12345")


def test_split_distribution_and_missing_ratio_block(capture):
    root, manifest, _ = capture
    for adjustment_type, ratio in (("stock_dividend", 2), ("forward_split", None)):
        event = dict(ticker="SPY", execution_date=str(FIRST), split_from=1,
                     split_to=ratio, adjustment_type=adjustment_type)
        broken = replace(manifest, splits=stream(root, "splits", [event]))
        assert_blocked(preflight_spy_snapshot(root, broken, review=attest(root, broken)),
                       "unsupported_split")


def test_explicit_unverified_attestation_does_not_qualify(capture):
    root, manifest, _ = capture
    review = attest(root, manifest)
    wrong = ReviewedEvidence("rights", save(root, "not-cleared.json",
                             {"source": "fixture rights", "conclusion": "unverified"}))
    review = replace(review, evidence=(wrong, *review.evidence[1:]))
    assert_blocked(preflight_spy_snapshot(root, manifest, review=review), "review_evidence")


def test_missing_response_file_blocks(capture):
    root, manifest, _ = capture
    (root / manifest.bars.file.path).unlink()
    assert_blocked(preflight_spy_snapshot(root, manifest), "file_reference")


def test_expected_hash_cannot_authorize_an_outside_path(capture):
    root, manifest, _ = capture
    response = replace(manifest.bars, file=replace(manifest.bars.file, path="../bars.json"))
    assert_blocked(preflight_spy_snapshot(root, replace(manifest, bars=response)), "file_reference")
