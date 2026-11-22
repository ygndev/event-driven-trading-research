from copy import deepcopy
from datetime import UTC, date, datetime, timedelta, timezone
from fractions import Fraction

import pytest

from trading_research.corporate_actions import StockSplit
from trading_research.daily_bars import DailyBar, PriceAdjustment
from trading_research.data_audit import audit_etf_data
from trading_research.massive_history import parse_daily_aggregates, parse_split


MARCH_8 = date(2024, 3, 8)
MARCH_11 = date(2024, 3, 11)
RETRIEVED = datetime(2024, 3, 13, tzinfo=UTC)
REQUEST = "https://api.massive.com/v2/aggs/ticker/SPY/range/1/day/2024-03-08/2024-03-11"


def aggregate(timestamp, **fields):
    return {"t": timestamp, "o": 100.25, "h": 105, "l": 99, "c": 103, "v": 1200, **fields}


def payload(*rows, adjusted=False):
    return {
        "status": "OK",
        "ticker": "SPY",
        "request_id": "synthetic-request",
        "adjusted": adjusted,
        "queryCount": len(rows),
        "resultsCount": len(rows),
        "results": list(rows),
    }


@pytest.fixture
def response():
    return payload(
        aggregate(1709874000000),                    # 05:00 UTC before spring DST
        aggregate(1710129600000, o=103, v=900.0),     # 04:00 UTC after spring DST
    )


def parse(response, **overrides):
    params = {
        "request_url": REQUEST + "?adjusted=false&sort=asc",
        "expected_symbol": "SPY",
        "adjustment": PriceAdjustment.RAW,
        "retrieved_at": RETRIEVED,
    }
    return parse_daily_aggregates(response, **(params | overrides))


def test_raw_bars_preserve_values_request_and_inclusive_session_dates(response):
    original = deepcopy(response)
    request_url = REQUEST + "?adjusted=false&sort=asc&limit=2"

    parsed = parse(response, request_url=request_url)

    assert parsed.dataset.bars == (
        DailyBar("SPY", MARCH_8, 100.25, 105, 99, 103, 1200),
        DailyBar("SPY", MARCH_11, 103, 105, 99, 103, 900),
    )
    assert type(parsed.dataset.bars[1].volume) is int
    assert parsed.dataset.requested_range == (MARCH_8, MARCH_11)
    assert parsed.dataset.metadata.adjustment is PriceAdjustment.RAW
    assert parsed.dataset.metadata.retrieved_at == RETRIEVED
    assert parsed.dataset.metadata.source == "Massive"
    assert (parsed.request_url, parsed.request_id, parsed.query_count) == (
        request_url, "synthetic-request", 2,
    )
    assert parsed == parse(response, request_url=request_url)
    assert response == original


def test_preflight_keeps_provider_reported_raw_unverified(response):
    dataset = parse(response).dataset

    report = audit_etf_data((dataset,), expected_sessions=(MARCH_8, MARCH_11))

    assert [(finding.severity, finding.code) for finding in report.findings] == [
        ("passed", "ohlcv_valid"),
        ("unverified", "raw_provenance_unverified"),
        ("passed", "metadata_consistent"),
        ("passed", "calendar_matches"),
    ]


def test_split_adjusted_response_retains_its_basis_and_preflight_blocks_execution(response):
    dataset = parse(
        {**response, "adjusted": True},
        request_url=REQUEST + "?adjusted=true",
        adjustment=PriceAdjustment.SPLIT_ADJUSTED,
    ).dataset

    assert dataset.metadata.adjustment is PriceAdjustment.SPLIT_ADJUSTED
    assert dataset.bars[0].open == response["results"][0]["o"]
    report = audit_etf_data((dataset,))
    assert ("blocking", "incompatible_adjustment") in {
        (finding.severity, finding.code) for finding in report.findings
    }


@pytest.mark.parametrize(("query", "message"), [
    ("", "explicitly specify adjusted=false"),
    ("?adjusted=true", "explicitly specify adjusted=false"),
    ("?adjusted=0", "explicitly specify adjusted=false"),
    ("?adjusted=false&adjusted=true", "unique adjusted"),
    ("?adjusted=false&cursor=synthetic-cursor", "unique adjusted"),
    ("?adjusted=false&sort=desc", "ascending"),
    ("?adjusted=false&limit=50001", "request limit"),
])
def test_raw_requires_the_saved_unadjusted_daily_request(response, query, message):
    with pytest.raises(ValueError, match=message):
        parse(response, request_url=REQUEST + query)


@pytest.mark.parametrize(("change", "message"), [
    ({"adjusted": True}, "response adjusted"),
    ({"adjusted": 0}, "response adjusted"),
    ({"ticker": "spy"}, "response ticker"),
    ({"request_id": ""}, "request_id"),
    ({"resultsCount": 1}, "resultsCount"),
    ({"queryCount": 1}, "queryCount"),
    ({"queryCount": 5001}, "queryCount"),
    ({"next_url": "https://api.massive.com/v2/aggs/example?cursor=synthetic"}, "paginated"),
    ({"status": "DELAYED"}, "status OK"),
])
def test_rejects_inconsistent_or_unsupported_response_envelopes(response, change, message):
    with pytest.raises(ValueError, match=message):
        parse(response | change)


@pytest.mark.parametrize("field", ("adjusted", "queryCount", "resultsCount", "request_id"))
def test_missing_required_response_fields(response, field):
    del response[field]

    with pytest.raises(ValueError, match=field):
        parse(response)


@pytest.mark.parametrize("suffix", ("?adjusted=false", "?adjusted=true"))
def test_total_return_adjustment_is_unsupported(response, suffix):
    with pytest.raises(ValueError, match="RAW or SPLIT_ADJUSTED"):
        parse(
            response, request_url=REQUEST + suffix,
            adjustment=PriceAdjustment.TOTAL_RETURN_ADJUSTED,
        )


def test_provider_errors_are_not_empty_history():
    with pytest.raises(ValueError, match="NOT_AUTHORIZED"):
        parse({"status": "NOT_AUTHORIZED", "message": "synthetic access error"})


@pytest.mark.parametrize("include_results", (True, False))
def test_empty_success_responses_cannot_form_a_dataset(include_results):
    response = payload()
    if not include_results:
        del response["results"]

    with pytest.raises(ValueError, match="no daily bars"):
        parse(response)


@pytest.mark.parametrize(("url", "message"), [
    (REQUEST.replace("/SPY/", "/QQQ/"), "request ticker"),
    (REQUEST.replace("/1/day/", "/1/minute/"), "1/day"),
    (REQUEST.replace("2024-03-08", "20240308"), "YYYY-MM-DD"),
    (REQUEST.replace("2024-03-08", "2024-03-12"), "on or before"),
    (REQUEST.replace("2024-03-08", "2024-03-11"), "outside the inclusive"),
    (REQUEST.replace("2024-03-11", "2024-03-08"), "outside the inclusive"),
])
def test_request_instrument_resolution_and_boundaries(response, url, message):
    with pytest.raises(ValueError, match=message):
        parse(response, request_url=url + "?adjusted=false")


def test_same_day_request_can_be_valid_and_retrieval_uses_utc(response):
    url = REQUEST.replace("2024-03-11", "2024-03-08") + "?adjusted=false"
    single = payload(response["results"][0])

    assert parse(single, request_url=url).dataset.bars[0].session_date == MARCH_8
    with pytest.raises(ValueError, match="in UTC"):
        parse(
            single, request_url=url,
            retrieved_at=datetime(2024, 3, 13, tzinfo=timezone(timedelta(hours=-4))),
        )
    with pytest.raises(ValueError, match="follows the retrieval"):
        parse(single, request_url=url, retrieved_at=datetime(2024, 3, 8, 3, tzinfo=UTC))


@pytest.mark.parametrize(("changes", "message"), [
    ({"t": 1709874000}, "midnight America/New_York"),
    ({"t": 1709874000000.0}, "integer Unix milliseconds"),
    ({"t": True}, "integer Unix milliseconds"),
    ({"t": 10**30}, "outside the supported date range"),
    ({"t": 1709856000000}, "midnight America/New_York"),
    ({"o": 0}, "open must be"),
    ({"c": float("inf")}, "close must be"),
    ({"h": 100}, "high is below open"),
    ({"v": -1}, "volume must be"),
    ({"v": 1.25}, "volume must be"),
    ({"v": True}, "volume must be"),
])
def test_rejects_bad_timestamp_price_or_volume(changes, message):
    with pytest.raises(ValueError, match=message):
        parse(payload(aggregate(1709874000000, **changes)))


@pytest.mark.parametrize("field", ("t", "o", "h", "l", "c", "v"))
def test_missing_aggregate_fields_are_not_imputed(field):
    row = aggregate(1709874000000)
    del row[field]

    with pytest.raises(ValueError, match=f"missing field '{field}'"):
        parse(payload(row))


def test_rejects_nonobject_aggregate():
    with pytest.raises(ValueError, match="aggregate object"):
        parse(payload(None))


def test_rejects_out_of_order_and_duplicate_sessions(response):
    first, last = response["results"]
    with pytest.raises(ValueError, match="out of order"):
        parse(payload(last, first))
    with pytest.raises(ValueError, match="duplicate session"):
        parse(payload(first, first))


@pytest.mark.parametrize(("start", "end", "first", "last"), [
    ("2006-03-31", "2006-04-03", 1143781200000, 1144036800000),
    ("2007-03-09", "2007-03-12", 1173416400000, 1173672000000),
])
def test_historical_eastern_dst_rules_not_current_fixed_offsets(start, end, first, last):
    url = f"https://api.massive.com/v2/aggs/ticker/SPY/range/1/day/{start}/{end}?adjusted=false"

    parsed = parse(payload(aggregate(first), aggregate(last)), request_url=url)

    assert [bar.session_date for bar in parsed.dataset.bars] == [
        date.fromisoformat(start), date.fromisoformat(end),
    ]


def test_official_midnight_anchor_with_synthetic_prices():
    # Massive's discrepancy guide identifies this timestamp as midnight ET on September 10.
    start = "2026-09-10"
    url = f"https://api.massive.com/v2/aggs/ticker/SPY/range/1/day/{start}/{start}?adjusted=false"
    parsed = parse(
        payload(aggregate(1789012800000)), request_url=url,
        retrieved_at=datetime(2026, 9, 11, tzinfo=UTC),
    )

    assert parsed.dataset.bars[0].session_date == date(2026, 9, 10)


def split_record(**changes):
    return {
        "ticker": "SPY",
        "execution_date": "2024-03-11",
        "adjustment_type": "forward_split",
        "split_from": 2.0,
        "split_to": 3.0,
        "historical_adjustment_factor": 0.1,
        **changes,
    }


def test_split_uses_event_ratio_and_effective_session(response):
    split = parse_split(split_record(), expected_symbol="SPY")
    reverse = parse_split(
        split_record(adjustment_type="reverse_split", split_from=5, split_to=2),
        expected_symbol="SPY",
    )

    assert split == StockSplit("SPY", MARCH_11, Fraction(3, 2))
    assert reverse == StockSplit("SPY", MARCH_11, Fraction(2, 5))
    report = audit_etf_data((parse(response).dataset,), corporate_actions=(split,))
    assert any(finding.code == "actions_consistent" for finding in report.findings)


@pytest.mark.parametrize(("changes", "message"), [
    ({"ticker": "QQQ"}, "split ticker"),
    ({"execution_date": None}, "execution_date"),
    ({"split_from": None}, "split_from"),
    ({"split_to": 0}, "split_to"),
    ({"split_to": 2}, "conflicts with adjustment_type"),
    ({"adjustment_type": "reverse_split"}, "conflicts with adjustment_type"),
    ({"adjustment_type": "stock_dividend"}, "forward_split or reverse_split"),
])
def test_rejects_split_records_outside_the_contract(changes, message):
    with pytest.raises(ValueError, match=message):
        parse_split(split_record(**changes), expected_symbol="SPY")
