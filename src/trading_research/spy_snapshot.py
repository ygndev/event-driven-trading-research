"""Fail-closed preflight for one offline Massive SPY snapshot."""

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from re import fullmatch
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

from .corporate_actions import CashDividend, StockSplit
from .daily_bars import PriceAdjustment
from .data_audit import AuditDataset, AuditFinding, DateRange, _ordered_range, audit_etf_data
from .massive_history import _date, parse_daily_aggregates, parse_dividend, parse_split

_SESSION = "America/New_York"
_GATES = frozenset({"rights", "calendar", "raw_prices", "actions", "availability"})


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SavedResponse:
    file: SnapshotFile
    request_url: str
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class SpySnapshot:
    snapshot_id: str
    vintage: str
    requested_range: DateRange
    observed_range: DateRange
    scoring_range: DateRange
    bars: SavedResponse
    calendar: SnapshotFile
    splits: tuple[SavedResponse, ...]
    dividends: tuple[SavedResponse, ...]
    synthetic: bool = True
    warmup_sessions: int = 63
    symbol: str = "SPY"
    provider: str = "Massive"
    session_zone: str = _SESSION
    adjustment: PriceAdjustment = PriceAdjustment.RAW

    def fingerprint(self) -> str:
        """Identify this manifest, including all declared request and input hashes."""
        encoded = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewedEvidence:
    gate: str
    file: SnapshotFile


@dataclass(frozen=True, slots=True)
class SnapshotReview:
    """An operator's separate review of rights and independent source evidence."""

    manifest_sha256: str
    reviewer: str
    evidence: tuple[ReviewedEvidence, ...]
    ordinary_dividend_ids: frozenset[str]
    split_share_basis: str
    dividend_share_basis: str


@dataclass(frozen=True, slots=True)
class PreflightResult:
    status: Literal["READY", "BLOCKED"]
    findings: tuple[AuditFinding, ...]
    input_hashes: tuple[tuple[str, str], ...]
    evidence: tuple[ReviewedEvidence, ...] = ()
    dataset: AuditDataset | None = None
    actions: tuple[StockSplit | CashDividend, ...] = ()


class _Blocked(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read(
    root: Path, reference: SnapshotFile, label: str, hashes: list[tuple[str, str]],
) -> bytes:
    if not isinstance(reference, SnapshotFile) or not isinstance(reference.path, str):
        raise _Blocked("file_reference", f"{label}: a local file reference is required.")
    parts = reference.path.split("/")
    if not all(fullmatch(r"[A-Za-z0-9_.-]+", part) and part not in {".", ".."} for part in parts):
        raise _Blocked("file_reference", f"{label}: use a relative path within the snapshot.")
    if not isinstance(reference.sha256, str) or not fullmatch(r"[0-9a-f]{64}", reference.sha256):
        raise _Blocked("file_digest", f"{label}: a lowercase SHA-256 digest is required.")
    try:
        path = root.joinpath(*parts).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise _Blocked("file_reference", f"{label}: expected a file within the snapshot.")
        content = path.read_bytes()
    except (OSError, RuntimeError):
        raise _Blocked("unreadable_file", f"{label}: could not read the local file.") from None
    digest = sha256(content).hexdigest()
    hashes.append((label, digest))
    if digest != reference.sha256:
        raise _Blocked("hash_mismatch", f"{label}: SHA-256 differs from the manifest.")
    return content


def _unique_fields(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON field")
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("nonfinite JSON constant")


def _json(content: bytes, label: str, *, exact: bool = False) -> dict:
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_fields,
                           parse_constant=_invalid_constant, parse_float=Decimal if exact else float)
    except (ValueError, ArithmeticError, RecursionError):
        raise _Blocked("invalid_json", f"{label}: expected unambiguous UTF-8 JSON.") from None
    if not isinstance(value, dict):
        raise _Blocked("invalid_json", f"{label}: expected a JSON object.")
    return value


def _request(url: str, endpoint: str, label: str) -> dict[str, str]:
    try:
        if not isinstance(url, str) or not url.isascii() or any(c.isspace() for c in url):
            raise ValueError
        parts = urlsplit(url)
        pairs = parse_qsl(parts.query, strict_parsing=True, keep_blank_values=True)
        params = dict(pairs)
        if (parts.scheme != "https" or parts.netloc != "api.massive.com"
                or parts.path != endpoint or parts.fragment or len(params) != len(pairs)):
            raise ValueError
    except ValueError:
        raise _Blocked("request", f"{label}: expected the credential-free Massive endpoint.") from None
    return params


def _utc(response: SavedResponse, end: date, label: str) -> None:
    if (not isinstance(response, SavedResponse)
            or not isinstance(response.retrieved_at, datetime)
            or response.retrieved_at.utcoffset() != timedelta(0)
            or response.retrieved_at.date() <= end):
        raise _Blocked("acquisition_time", f"{label}: require a UTC acquisition after the requested end day.")


def _calendar(root: Path, manifest: SpySnapshot, hashes) -> tuple[tuple[date, ...], DateRange]:
    payload = _json(_read(root, manifest.calendar, "calendar", hashes), "calendar")
    if (not _text(payload.get("source")) or not _text(payload.get("version"))
            or payload.get("session_zone") != _SESSION or not isinstance(payload.get("sessions"), list)):
        raise _Blocked("calendar_evidence", "Calendar requires source, version, zone and explicit sessions.")
    try:
        bounds = (_date(payload.get("start"), "calendar start"), _date(payload.get("end"), "calendar end"))
        days = tuple(_date(day, "session") for day in payload["sessions"])
    except ValueError:
        raise _Blocked("calendar_evidence", "Calendar bounds and sessions must be ISO dates.") from None
    start, end = manifest.requested_range
    if bounds[0] > start or bounds[1] < end:
        raise _Blocked("calendar_partial", "Calendar must cover the full requested range.")
    scoring_start, scoring_end = manifest.scoring_range
    if (sum(start <= day < scoring_start for day in days) < manifest.warmup_sessions
            or not any(scoring_start <= day <= scoring_end for day in days)):
        raise _Blocked("warmup", "Calendar lacks the required warm-up or scoring sessions.")
    return days, bounds


def _action_rows(root: Path, pages: tuple[SavedResponse, ...], stream: str,
                 date_field: str, bounds: DateRange, hashes) -> list[dict]:
    if not isinstance(pages, tuple) or not pages:
        raise _Blocked("pagination", f"{stream}: expected at least the first saved page.")
    endpoint = f"/stocks/v1/{stream}"
    start, end = bounds
    rows = []
    request_ids = set()
    cursors = set()
    next_request = None
    retrieved_at = None
    for position, page in enumerate(pages, start=1):
        label = f"{stream} page {position}"
        _utc(page, end, label)
        params = _request(page.request_url, endpoint, label)
        if position == 1:
            filters = {"ticker": "SPY", f"{date_field}.gte": str(start),
                       f"{date_field}.lte": str(end), "sort": f"{date_field}.asc"}
            limit = params.pop("limit", "100")
            if params != filters or not fullmatch(r"[0-9]{1,4}", limit) or not 1 <= int(limit) <= 5000:
                raise _Blocked("action_coverage", f"{label}: expected an ascending full-range SPY request.")
        elif (next_request is None or params != next_request
              or page.retrieved_at < retrieved_at):
            raise _Blocked("pagination", f"{label}: missing or out-of-order continuation.")
        retrieved_at = page.retrieved_at
        payload = _json(_read(root, page.file, label, hashes), label, exact=True)
        page_rows = payload.get("results")
        request_id = payload.get("request_id")
        if (payload.get("status") != "OK" or not _text(request_id)
                or request_id in request_ids or not isinstance(page_rows, list)
                or not all(isinstance(row, dict) for row in page_rows)):
            raise _Blocked("action_envelope", f"{label}: invalid status, request ID or results.")
        request_ids.add(request_id)
        for count in ("count", "resultsCount"):
            if count in payload and (type(payload[count]) is not int or payload[count] != len(page_rows)):
                raise _Blocked("action_count", f"{label}: returned count disagrees with page rows.")
        rows.extend(page_rows)
        next_request = None
        if "next_url" in payload:
            next_request = _request(payload["next_url"], endpoint, label)
            cursor = next_request.get("cursor", "")
            if next_request.keys() != {"cursor"} or not cursor or cursor in cursors:
                raise _Blocked("pagination", f"{label}: missing, repeated or unexpected continuation.")
            cursors.add(cursor)
    if next_request is not None:
        raise _Blocked("pagination", f"{stream}: the last saved page advertises another page.")

    previous = start
    ids = set()
    for row in rows:
        try:
            action_date = _date(row.get(date_field), date_field)
        except ValueError:
            raise _Blocked("action_date", f"{stream}: expected a valid action date.") from None
        if not previous <= action_date <= end:
            raise _Blocked("action_coverage", f"{stream}: dates are unordered or outside the request.")
        previous = action_date
        if "id" in row:
            event_id = row["id"]
            if not _text(event_id) or event_id in ids:
                raise _Blocked("action_revision", f"{stream}: invalid or duplicate provider event ID.")
            ids.add(event_id)
    return rows


def _review(
    root: Path,
    manifest: SpySnapshot,
    review: SnapshotReview | None,
    hashes: list[tuple[str, str]],
    *,
    split_count: int,
    dividend_count: int,
) -> SnapshotReview:
    if (not isinstance(review, SnapshotReview) or not _text(review.reviewer)
            or review.manifest_sha256 != manifest.fingerprint()):
        raise _Blocked("review_required", "Require a separate review bound to this exact manifest.")
    if (not isinstance(review.evidence, tuple)
            or not all(isinstance(item, ReviewedEvidence) and isinstance(item.gate, str)
                       for item in review.evidence)
            or len(review.evidence) != len(_GATES)
            or {item.gate for item in review.evidence} != _GATES):
        raise _Blocked("review_evidence", "Require rights, calendar, RAW, action and availability evidence.")
    if (review.split_share_basis != "new_per_old" or review.dividend_share_basis != "prior_close"
            or not isinstance(review.ordinary_dividend_ids, frozenset)
            or not all(_text(i) for i in review.ordinary_dividend_ids)):
        raise _Blocked("share_basis", "Review must confirm split units and the prior-close cash share basis.")
    evidence_hashes = set()
    input_digests = {digest for _, digest in hashes}
    for item in sorted(review.evidence, key=lambda row: row.gate):
        content = _read(root, item.file, f"review {item.gate}", hashes)
        digest = item.file.sha256
        record = _json(content, f"review {item.gate}")
        if (not _text(record.get("source")) or record.get("conclusion") != "confirmed"
                or digest in input_digests or digest in evidence_hashes):
            raise _Blocked("review_evidence", "Require a separate nonempty record for each reviewed gate.")
        if item.gate == "actions" and any(
            type(record.get(key)) is not int or record[key] != count
            for key, count in (("split_count", split_count), ("dividend_count", dividend_count))
        ):
            raise _Blocked("action_review", "Reviewed action counts must match both captured streams.")
        if item.gate == "availability" and not all(_text(record.get(key)) for key in ("timing", "revisions")):
            raise _Blocked("review_evidence", "Record timing evidence and revision limitations.")
        evidence_hashes.add(digest)
    return review


def preflight_spy_snapshot(
    root: Path, manifest: SpySnapshot, *, review: SnapshotReview | None = None,
) -> PreflightResult:
    """Verify local integrity and a supplied review; expose data only if READY."""
    hashes: list[tuple[str, str]] = []
    findings: list[AuditFinding] = []
    evidence = ()
    try:
        if (not isinstance(manifest, SpySnapshot) or not _text(manifest.snapshot_id)
                or not _text(manifest.vintage) or manifest.symbol != "SPY" or manifest.provider != "Massive"
                or manifest.session_zone != _SESSION or manifest.adjustment is not PriceAdjustment.RAW
                or type(manifest.synthetic) is not bool or type(manifest.warmup_sessions) is not int
                or manifest.warmup_sessions < (0 if manifest.synthetic else 63)
                or not all(_ordered_range(span) for span in
                           (manifest.requested_range, manifest.observed_range, manifest.scoring_range))):
            raise _Blocked("manifest", "Require a complete SPY Massive RAW manifest.")
        start, end = manifest.requested_range
        if not start <= manifest.scoring_range[0] <= manifest.scoring_range[1] <= end:
            raise _Blocked("manifest", "The scoring range must fit the requested range.")
        root = root.resolve()
        _utc(manifest.bars, end, "bars")
        params = _request(manifest.bars.request_url, f"/v2/aggs/ticker/SPY/range/1/day/{start}/{end}", "bars")
        if params.keys() - {"adjusted", "sort", "limit"} or params.get("adjusted") != "false":
            raise _Blocked("raw_request", "Bars require an explicit credential-free adjusted=false request.")
        payload = _json(_read(root, manifest.bars.file, "bars", hashes), "bars")
        try:
            capture = parse_daily_aggregates(
                payload, request_url=manifest.bars.request_url, expected_symbol="SPY",
                adjustment=PriceAdjustment.RAW, retrieved_at=manifest.bars.retrieved_at,
            )
        except (ValueError, OverflowError):
            raise _Blocked("invalid_bars", "Check RAW envelope, counts, session order and OHLCV.") from None
        bars = capture.dataset.bars
        if (bars[0].session_date, bars[-1].session_date) != manifest.observed_range:
            raise _Blocked("observed_range", "Observed endpoints differ from the manifest.")
        days, calendar_range = _calendar(root, manifest, hashes)
        split_rows = _action_rows(root, manifest.splits, "splits", "execution_date", (start, end), hashes)
        dividend_rows = _action_rows(root, manifest.dividends, "dividends", "ex_dividend_date", (start, end), hashes)
        # Validate types before allowing review claims to qualify or normalize the inputs.
        try:
            splits = tuple(parse_split(row, expected_symbol="SPY") for row in split_rows)
        except ValueError:
            raise _Blocked("unsupported_split", "A split has an unsupported type, date or ratio.") from None
        dataset = replace(capture.dataset, snapshot_id=manifest.snapshot_id,
                          source_path=root / manifest.bars.file.path, source_sha256=manifest.bars.file.sha256)
        review = _review(
            root, manifest, review, hashes,
            split_count=len(split_rows), dividend_count=len(dividend_rows),
        )
        evidence = tuple(sorted(review.evidence, key=lambda row: row.gate))
        try:
            dividends = tuple(
                parse_dividend(row, expected_symbol="SPY",
                               ordinary_confirmed=isinstance(row.get("id"), str)
                               and row["id"] in review.ordinary_dividend_ids,
                               share_basis=review.dividend_share_basis)
                for row in dividend_rows
            )
        except ValueError as exc:
            raise _Blocked("unsupported_dividend", str(exc)) from None
        if {row.action_id for row in dividends} != review.ordinary_dividend_ids:
            raise _Blocked("action_review", "Reviewed ordinary dividend IDs must match captured events.")
        actions = (*splits, *dividends)
        report = audit_etf_data(
            [dataset], corporate_actions=actions, expected_sessions=days, calendar_range=calendar_range,
        )
        findings.extend(report.findings)
        if any(f.severity == "blocking" or
               (f.severity == "unverified" and f.code != "raw_provenance_unverified") for f in findings):
            return PreflightResult("BLOCKED", tuple(findings), tuple(hashes), evidence)
        if manifest.synthetic:
            raise _Blocked("synthetic", "Synthetic inputs cannot qualify an empirical snapshot.")
        findings.append(AuditFinding("passed", "review_recorded",
                                     "The supplied review covers all gates for this manifest.", "SPY"))
        return PreflightResult("READY", tuple(findings), tuple(hashes), evidence, dataset, actions)
    except _Blocked as error:
        findings.append(AuditFinding("blocking", error.code, str(error), "SPY"))
        return PreflightResult("BLOCKED", tuple(findings), tuple(hashes), evidence)
