"""Offline integrity and provenance checks for daily ETF research inputs."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
from hashlib import file_digest
from pathlib import Path
from re import fullmatch
from typing import Literal, TypeGuard

from ._money import Money, decimal_amount, to_decimal
from .corporate_actions import CashDividend, StockSplit
from .daily_bars import DailyBar, DatasetMetadata, PriceAdjustment, _validate_ohlcv

DateRange = tuple[date, date]
Severity = Literal["blocking", "warning", "unverified", "passed"]


@dataclass(frozen=True, slots=True)
class AuditDataset:
    metadata: DatasetMetadata | None
    bars: Sequence[DailyBar]
    requested_range: DateRange | None = None
    snapshot_id: str | None = None
    source_path: str | Path | None = None
    source_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class AuditFinding:
    severity: Severity
    code: str
    message: str
    symbol: str | None = None
    session: date | None = None


@dataclass(frozen=True, slots=True)
class ObservedCoverage:
    symbol: str | None
    first_session: date | None
    last_session: date | None
    observations: int


@dataclass(frozen=True, slots=True)
class AuditReport:
    coverage: tuple[ObservedCoverage, ...]
    findings: tuple[AuditFinding, ...]


def _session(value: object) -> TypeGuard[date]:
    return isinstance(value, date) and not isinstance(value, datetime)


def _ordered_range(bounds: DateRange) -> bool:
    return (
        isinstance(bounds, tuple) and len(bounds) == 2
        and all(_session(day) for day in bounds) and bounds[0] <= bounds[1]
    )


def _audit_rows(
    bars: Sequence[DailyBar], symbol: str | None, findings: list[AuditFinding],
) -> set[date]:
    if not bars:
        findings.append(AuditFinding("blocking", "empty_dataset", "No daily bars.", symbol))
        return set()

    first_finding = len(findings)
    observed: set[date] = set()
    previous = None
    for row, bar in enumerate(bars, start=1):
        if not isinstance(bar, DailyBar):
            findings.append(AuditFinding(
                "blocking", "invalid_bar", f"Row {row} must be a DailyBar.", symbol
            ))
            continue
        day = bar.session_date if _session(bar.session_date) else None
        matches = (
            isinstance(bar.symbol, str) and bool(bar.symbol)
            and bar.symbol == bar.symbol.strip()
            and (symbol is None or bar.symbol == symbol)
        )
        if not matches:
            findings.append(AuditFinding(
                "blocking", "symbol_mismatch",
                f"Row {row} has symbol {bar.symbol!r}; expected {symbol!r}.", symbol, day
            ))
        if day is None:
            findings.append(AuditFinding(
                "blocking", "invalid_session", f"Row {row} requires a session date.", symbol
            ))
        elif matches:
            if day in observed:
                findings.append(AuditFinding(
                    "blocking", "duplicate_session", f"Duplicate at row {row}.", symbol, day
                ))
            if previous is not None and day < previous:
                findings.append(AuditFinding(
                    "blocking", "unsorted_sessions", f"Row {row} follows {previous}.",
                    symbol, day,
                ))
            observed.add(day)
            previous = day
        try:
            _validate_ohlcv(bar, f"row {row}")
        except ValueError as exc:
            findings.append(AuditFinding("blocking", "invalid_ohlcv", str(exc), symbol, day))

    if len(findings) == first_finding:
        findings.append(AuditFinding("passed", "ohlcv_valid", "Daily bars passed validation.", symbol))
    return observed


def _audit_metadata(
    dataset: AuditDataset, coverage: ObservedCoverage, findings: list[AuditFinding],
) -> DateRange | None:
    symbol = coverage.symbol
    metadata = dataset.metadata
    start = len(findings)
    if metadata is None:
        findings.append(AuditFinding(
            "blocking", "missing_metadata", "Supply the dataset's provenance metadata.", symbol
        ))
    elif not isinstance(metadata, DatasetMetadata):
        findings.append(AuditFinding(
            "blocking", "invalid_metadata", "Expected a DatasetMetadata record.", symbol
        ))
    else:
        try:
            replace(metadata)
        except ValueError as exc:
            findings.append(AuditFinding("blocking", "invalid_metadata", str(exc), symbol))
        else:
            if metadata.retrieved_at.utcoffset() != timedelta(0):
                findings.append(AuditFinding(
                    "blocking", "retrieval_not_utc", "Record retrieval time in UTC.", symbol
                ))
            retrieved_date = metadata.retrieved_at.astimezone(UTC).date()
            if coverage.last_session is not None and retrieved_date < coverage.last_session:
                findings.append(AuditFinding(
                    "blocking", "retrieval_before_session",
                    "Retrieval date precedes the last observation.", symbol, coverage.last_session,
                ))
            if metadata.adjustment is not PriceAdjustment.RAW:
                findings.append(AuditFinding(
                    "blocking", "incompatible_adjustment",
                    f"Execution requires RAW; supplied {metadata.adjustment.value}.", symbol,
                ))
            else:
                findings.append(AuditFinding(
                    "unverified", "raw_provenance_unverified",
                    "RAW is declared. Obtain independent as-traded price provenance.", symbol,
                ))

    if dataset.snapshot_id is not None and (
        not isinstance(dataset.snapshot_id, str) or not dataset.snapshot_id.strip()
    ):
        findings.append(AuditFinding(
            "blocking", "invalid_snapshot_id", "Supply a nonempty snapshot identifier.", symbol
        ))
    requested = dataset.requested_range
    if requested is not None:
        if not _ordered_range(requested):
            findings.append(AuditFinding(
                "blocking", "invalid_requested_range",
                "Use ordered inclusive request dates.", symbol,
            ))
            requested = None
        elif coverage.first_session is not None and coverage.last_session is not None:
            for edge in dict.fromkeys((coverage.first_session, coverage.last_session)):
                if not requested[0] <= edge <= requested[1]:
                    findings.append(AuditFinding(
                        "blocking", "outside_requested_range",
                        "Observed data extends outside the requested date range.", symbol, edge,
                    ))

    if not any(row.severity == "blocking" for row in findings[start:]):
        findings.append(AuditFinding(
            "passed", "metadata_consistent",
            "Supplied provenance fields passed internal checks.", symbol,
        ))
    return requested


def _audit_source(dataset: AuditDataset, symbol: str | None, findings: list[AuditFinding]) -> None:
    path, expected = dataset.source_path, dataset.source_sha256
    if path is None and expected is None:
        return
    if expected is not None and (
        not isinstance(expected, str) or fullmatch(r"[0-9a-fA-F]{64}", expected) is None
    ):
        findings.append(AuditFinding(
            "blocking", "invalid_source_hash", "Supply a 64-digit SHA-256 hex digest.", symbol
        ))
        return
    if path is None:
        findings.append(AuditFinding(
            "blocking", "missing_source_path", "Supply the file covered by the SHA-256.", symbol
        ))
        return
    try:
        with Path(path).open("rb") as source:
            actual = file_digest(source, "sha256").hexdigest() if expected else None
    except (OSError, ValueError) as exc:
        findings.append(AuditFinding(
            "blocking", "source_unreadable", f"Could not open source bytes: {exc}.", symbol
        ))
        return
    if expected is None:
        findings.append(AuditFinding(
            "unverified", "source_hash_unverified", "No reference digest supplied.", symbol
        ))
    elif actual == expected.lower():
        findings.append(AuditFinding(
            "passed", "source_hash_matches", "Source bytes match the supplied SHA-256.", symbol
        ))
    else:
        findings.append(AuditFinding(
            "blocking", "source_hash_mismatch",
            f"Expected SHA-256 {expected.lower()} but read {actual}.", symbol,
        ))


def _calendar(
    sessions: Sequence[date] | None, bounds: DateRange | None, findings: list[AuditFinding],
) -> tuple[frozenset[date], DateRange] | None:
    if sessions is None and bounds is None:
        return None
    if isinstance(sessions, Sequence):
        days = tuple(sessions)
        if bounds is None and days:
            bounds = (days[0], days[-1])
        if (
            bounds is not None and _ordered_range(bounds)
            and all(_session(day) and bounds[0] <= day <= bounds[1] for day in days)
            and all(left < right for left, right in zip(days, days[1:]))
        ):
            return frozenset(days), bounds
    findings.append(AuditFinding(
        "blocking", "invalid_calendar",
        "Supply ordered unique session dates within an inclusive calendar range.",
    ))
    return None


def _audit_calendar(
    observed: set[date], requested: DateRange | None,
    calendar: tuple[frozenset[date], DateRange] | None,
    symbol: str | None, findings: list[AuditFinding],
) -> None:
    if calendar is None:
        findings.append(AuditFinding(
            "unverified", "calendar_unverified",
            "Exchange-session completeness requires a usable expected calendar.", symbol,
        ))
        return

    expected, (first, last) = calendar
    start, end = requested or (first, last)
    partial = start < first or end > last or any(not first <= day <= last for day in observed)
    if partial:
        findings.append(AuditFinding(
            "unverified", "calendar_partial",
            f"Completeness outside {first} through {last} cannot be checked.", symbol,
        ))
    missing = {day for day in expected if start <= day <= end} - observed
    unexpected = {day for day in observed if first <= day <= last} - expected
    for day in sorted(missing):
        findings.append(AuditFinding(
            "blocking", "missing_session", "Expected exchange session is absent.", symbol, day
        ))
    for day in sorted(unexpected):
        findings.append(AuditFinding(
            "blocking", "unexpected_session", "Observation falls on a non-session date.", symbol, day
        ))
    if not (partial or missing or unexpected):
        findings.append(AuditFinding(
            "passed", "calendar_matches", "Observed dates match the supplied calendar.", symbol
        ))


def _audit_actions(
    actions: Iterable[StockSplit | CashDividend],
    observed: dict[str, set[date]], findings: list[AuditFinding],
) -> None:
    split_dates: set[tuple[str, date]] = set()
    ex_dates: set[tuple[str, date]] = set()
    dividend_ids: set[str] = set()
    supplied = False
    start = len(findings)
    for action in actions:
        supplied = True
        if not isinstance(action, (StockSplit, CashDividend)):
            findings.append(AuditFinding(
                "blocking", "invalid_action", "Expected a StockSplit or CashDividend."
            ))
            continue
        is_split = isinstance(action, StockSplit)
        day = action.effective_session if is_split else action.ex_date
        symbol = action.symbol if isinstance(action.symbol, str) else None
        try:
            replace(action)
        except ValueError as exc:
            findings.append(AuditFinding(
                "blocking", "invalid_action", str(exc), symbol, day if _session(day) else None
            ))
            continue

        key = action.symbol, day
        if isinstance(action, StockSplit):
            if key in split_dates:
                findings.append(AuditFinding(
                    "blocking", "duplicate_split", "Duplicate or conflicting split.", symbol, day
                ))
            split_dates.add(key)
        else:
            if key in ex_dates or action.action_id in dividend_ids:
                findings.append(AuditFinding(
                    "blocking", "duplicate_dividend", "Duplicate dividend ID or ex-date.",
                    symbol, day,
                ))
            ex_dates.add(key)
            dividend_ids.add(action.action_id)

        dates = observed.get(action.symbol)
        if dates is None:
            findings.append(AuditFinding(
                "blocking", "action_without_dataset", "Action has no matching dataset.",
                symbol, day,
            ))
            continue
        if day not in dates:
            outside = not dates or day < min(dates) or day > max(dates)
            findings.append(AuditFinding(
                "blocking", "action_outside_coverage" if outside else "action_without_bar",
                "Supply the as-traded bar for this action's effective or ex-date.", symbol, day,
            ))
        if isinstance(action, CashDividend) and dates:
            if day in dates and day == min(dates):
                findings.append(AuditFinding(
                    "blocking", "dividend_without_preceding_close",
                    "Supply a preceding observed close for entitlement.", symbol, day,
                ))
            if day in dates and action.payment_date > max(dates):
                findings.append(AuditFinding(
                    "warning", "payment_after_coverage",
                    "Payment is dated after the last observed close.", symbol, action.payment_date,
                ))
    if supplied and not any(row.severity == "blocking" for row in findings[start:]):
        findings.append(AuditFinding(
            "passed", "actions_consistent", "Supplied actions passed consistency checks."
        ))


def _audit_moves(
    bars: Sequence[DailyBar], threshold: Fraction, findings: list[AuditFinding],
) -> None:
    for previous, current in zip(bars, bars[1:]):
        reference = Fraction(to_decimal(previous.close))
        for code, price in (("overnight_move", current.open), ("close_move", current.close)):
            if abs(Fraction(to_decimal(price)) - reference) > threshold * reference:
                findings.append(AuditFinding(
                    "warning", code,
                    f"Price {price} versus {previous.close} on {previous.session_date} "
                    "exceeds the supplied movement threshold.",
                    current.symbol, current.session_date,
                ))


def audit_etf_data(
    datasets: Iterable[AuditDataset],
    *,
    corporate_actions: Iterable[StockSplit | CashDividend] = (),
    expected_sessions: Sequence[date] | None = None,
    calendar_range: DateRange | None = None,
    price_move_threshold: Money | None = None,
) -> AuditReport:
    """Check supplied data and source bytes without repair or external access."""
    threshold = (
        Fraction(decimal_amount(price_move_threshold, "price_move_threshold"))
        if price_move_threshold is not None else None
    )
    findings: list[AuditFinding] = []
    coverage: list[ObservedCoverage] = []
    sessions_by_symbol: dict[str, set[date]] = {}
    duplicate_symbols: set[str] = set()
    calendar = _calendar(expected_sessions, calendar_range, findings)
    for dataset in datasets:
        bars = tuple(dataset.bars)
        metadata = dataset.metadata
        symbol = (
            metadata.symbol
            if isinstance(metadata, DatasetMetadata) and isinstance(metadata.symbol, str)
            else next(
                (bar.symbol for bar in bars
                 if isinstance(bar, DailyBar) and isinstance(bar.symbol, str)), None
            )
        )
        row_start = len(findings)
        observed = _audit_rows(bars, symbol, findings)
        valid_bars = not any(row.severity == "blocking" for row in findings[row_start:])
        span = ObservedCoverage(
            symbol, min(observed, default=None), max(observed, default=None), len(bars)
        )
        coverage.append(span)
        if symbol is not None:
            if symbol in sessions_by_symbol:
                duplicate_symbols.add(symbol)
                findings.append(AuditFinding(
                    "blocking", "duplicate_dataset", "Provide one dataset per symbol.", symbol
                ))
            else:
                sessions_by_symbol[symbol] = observed

        requested = _audit_metadata(dataset, span, findings)
        _audit_calendar(observed, requested, calendar, symbol, findings)
        _audit_source(dataset, symbol, findings)
        if threshold is not None and valid_bars:
            _audit_moves(bars, threshold, findings)

    if not coverage:
        findings.append(AuditFinding("blocking", "no_datasets", "Supply at least one dataset."))
    for symbol in duplicate_symbols:
        del sessions_by_symbol[symbol]
    _audit_actions(corporate_actions, sessions_by_symbol, findings)
    return AuditReport(tuple(coverage), tuple(findings))
