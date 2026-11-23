"""Convert saved Massive daily aggregate responses without network access."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
from re import fullmatch
from urllib.parse import parse_qs, unquote, urlsplit
from zoneinfo import ZoneInfo

from ._money import decimal_amount
from .corporate_actions import CashDividend, StockSplit
from .daily_bars import (
    DailyBar, DatasetMetadata, PriceAdjustment, _validate_text, validate_daily_bars,
)
from .data_audit import AuditDataset


@dataclass(frozen=True, slots=True)
class MassiveDaily:
    """Validated bars and request evidence retained from a single retrieval."""

    dataset: AuditDataset
    request_url: str
    request_id: str
    query_count: int


def _date(value: object, field: str) -> date:
    if isinstance(value, str):
        try:
            day = date.fromisoformat(value)
        except ValueError:
            pass
        else:
            if day.isoformat() == value:
                return day
    raise ValueError(f"{field} must be a YYYY-MM-DD date")


def _daily_request(
    request_url: str, symbol: str, adjusted: bool,
) -> tuple[date, date, int]:
    _validate_text(request_url, "request_url")
    url = urlsplit(request_url)
    path = fullmatch(r"/v2/aggs/ticker/([^/]+)/range/1/day/([^/]+)/([^/]+)", url.path)
    if url.scheme != "https" or url.netloc != "api.massive.com" or url.fragment or path is None:
        raise ValueError("request_url must be the saved Massive 1/day stocks aggregate request")
    if unquote(path[1]) != symbol:
        raise ValueError(f"request ticker must match {symbol!r}")

    try:
        params = parse_qs(url.query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        raise ValueError("request_url has a malformed query") from None
    if params.keys() - {"adjusted", "sort", "limit"} or any(len(v) != 1 for v in params.values()):
        raise ValueError("request query must contain only unique adjusted, sort, or limit fields")
    flag = "true" if adjusted else "false"
    if params.get("adjusted") != [flag]:
        raise ValueError(f"the saved request must explicitly specify adjusted={flag}")
    if params.get("sort", ["asc"]) != ["asc"]:
        raise ValueError("only ascending aggregate requests are supported")
    limit = params.get("limit", ["5000"])[0]
    if fullmatch(r"[0-9]{1,5}", limit) is None or not 1 <= int(limit) <= 50_000:
        raise ValueError("request limit must be an integer from 1 to 50000")

    start, end = _date(path[2], "request from"), _date(path[3], "request to")
    if start > end:
        raise ValueError("request from must be on or before to")
    return start, end, int(limit)


def _daily_bar(
    row: object, index: int, symbol: str, eastern: ZoneInfo,
) -> DailyBar:
    if not isinstance(row, Mapping):
        raise ValueError(f"result {index}: expected an aggregate object")
    try:
        timestamp, opened, high, low, close, volume = (
            row[field] for field in ("t", "o", "h", "l", "c", "v")
        )
    except KeyError as exc:
        raise ValueError(f"result {index}: missing field {exc.args[0]!r}") from exc
    if type(timestamp) is not int:
        raise ValueError(f"result {index}: t must be integer Unix milliseconds")
    try:
        utc_start = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=timestamp)
        local_start = utc_start.astimezone(eastern)
    except OverflowError as exc:
        raise ValueError(f"result {index}: t is outside the supported date range") from exc
    if local_start.time() != time.min:
        raise ValueError(f"result {index}: daily window must start at midnight America/New_York")

    # The provider calls v numeric. Whole-valued numbers fit our integer contract.
    if isinstance(volume, float) and volume.is_integer():
        volume = int(volume)
    return DailyBar(symbol, local_start.date(), opened, high, low, close, volume)


def parse_daily_aggregates(
    response: Mapping[str, object],
    *,
    request_url: str,
    expected_symbol: str,
    adjustment: PriceAdjustment,
    retrieved_at: datetime,
) -> MassiveDaily:
    """Validate one saved response against its actual, credential-free request URL."""
    if adjustment is PriceAdjustment.RAW:
        adjusted = False
    elif adjustment is PriceAdjustment.SPLIT_ADJUSTED:
        adjusted = True
    else:
        raise ValueError("Massive daily aggregates support RAW or SPLIT_ADJUSTED")
    metadata = DatasetMetadata("Massive", expected_symbol, adjustment, retrieved_at, "v2/aggs")
    if retrieved_at.utcoffset() != timedelta(0):
        raise ValueError("retrieved_at must be in UTC")
    start, end, limit = _daily_request(request_url, expected_symbol, adjusted)

    if not isinstance(response, Mapping):
        raise ValueError("Massive response must be a JSON object")
    if response.get("status") != "OK":
        raise ValueError(f"expected historical status OK, got {response.get('status')!r}")
    if response.get("ticker") != expected_symbol:
        raise ValueError(f"response ticker must match {expected_symbol!r}")
    if response.get("adjusted") is not adjusted:
        raise ValueError(f"response adjusted must be {adjusted!r} to match the saved request")
    if "next_url" in response:
        raise ValueError("paginated responses are outside the single-response adapter")
    request_id = response.get("request_id")
    _validate_text(request_id, "response request_id")
    rows = response.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("response results must be a JSON array")
    if not rows:
        raise ValueError("response contains no daily bars")
    results_count = response.get("resultsCount")
    if type(results_count) is not int or results_count != len(rows):
        raise ValueError("response resultsCount must match the number of results")
    query_count = response.get("queryCount")
    if type(query_count) is not int or not results_count <= query_count <= limit:
        raise ValueError("response queryCount must be between resultsCount and the request limit")

    eastern = ZoneInfo("America/New_York")
    bars = tuple(
        _daily_bar(row, index, expected_symbol, eastern)
        for index, row in enumerate(rows, start=1)
    )
    validate_daily_bars(bars, metadata)
    if bars[0].session_date < start or bars[-1].session_date > end:
        raise ValueError("observed sessions fall outside the inclusive request dates")
    if bars[-1].session_date > retrieved_at.astimezone(eastern).date():
        raise ValueError("last observed session follows the retrieval date")

    return MassiveDaily(
        AuditDataset(metadata, bars, requested_range=(start, end)),
        request_url,
        request_id,
        query_count,
    )


def parse_split(record: Mapping[str, object], *, expected_symbol: str) -> StockSplit:
    """Map one current /stocks/v1/splits result to the existing split contract."""
    if not isinstance(record, Mapping):
        raise ValueError("split result must be a JSON object")
    _validate_text(expected_symbol, "expected_symbol")
    if record.get("ticker") != expected_symbol:
        raise ValueError(f"split ticker must match {expected_symbol!r}")
    kind = record.get("adjustment_type")
    if kind not in ("forward_split", "reverse_split"):
        raise ValueError("split adjustment_type must be forward_split or reverse_split")
    day = _date(record.get("execution_date"), "execution_date")
    old_shares = decimal_amount(record.get("split_from"), "split_from")
    new_shares = decimal_amount(record.get("split_to"), "split_to")
    if old_shares == new_shares or (new_shares > old_shares) != (kind == "forward_split"):
        raise ValueError("split ratio conflicts with adjustment_type")
    return StockSplit(expected_symbol, day, Fraction(new_shares) / Fraction(old_shares))


def parse_dividend(
    record: Mapping[str, object],
    *,
    expected_symbol: str,
    ordinary_confirmed: bool,
    share_basis: str,
) -> CashDividend:
    """Map an independently classified ordinary USD event on prior-close shares."""
    if not isinstance(record, Mapping) or record.get("ticker") != expected_symbol:
        raise ValueError("Dividend ticker must match the snapshot instrument.")
    if record.get("distribution_type") != "recurring" or ordinary_confirmed is not True:
        raise ValueError("Dividend requires a supported classification and ordinary-event review.")
    if record.get("currency") != "USD":
        raise ValueError("Dividend currency must be USD.")
    if share_basis != "prior_close":
        raise ValueError("Dividend original cash amount must be verified on prior-close shares.")
    amount = record.get("cash_amount")
    if type(amount) not in (int, Decimal):
        raise ValueError("Dividend requires an exact original cash_amount.")
    ex_date = _date(record.get("ex_dividend_date"), "ex_dividend_date")
    payment_date = _date(record.get("pay_date"), "pay_date")
    return CashDividend(record.get("id"), expected_symbol, ex_date, payment_date, amount)
