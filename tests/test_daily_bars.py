from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta, timezone
from math import nextafter

import pytest

from trading_research.daily_bars import (
    DailyBar,
    DatasetMetadata,
    PriceAdjustment,
    validate_daily_bars,
)


@pytest.fixture
def metadata() -> DatasetMetadata:
    return DatasetMetadata(
        source="Synthetic provider",
        symbol="SPY",
        adjustment=PriceAdjustment.RAW,
        retrieved_at=datetime(2024, 9, 5, 12, tzinfo=UTC),
    )


@pytest.fixture
def bar() -> DailyBar:
    return DailyBar(
        symbol="SPY",
        session_date=date(2024, 8, 30),
        open=100,
        high=102.5,
        low=99,
        close=101.5,
        volume=1000,
    )


def test_accepts_gaps_flat_prices_and_zero_volume(bar, metadata):
    later = replace(
        bar,
        session_date=date(2024, 9, 3),
        open=101,
        high=101,
        low=101,
        close=101,
        volume=0,
    )

    validate_daily_bars((bar, later), metadata)


def test_bar_and_metadata_are_immutable(bar, metadata):
    with pytest.raises(FrozenInstanceError):
        setattr(bar, "close", 101)
    with pytest.raises(FrozenInstanceError):
        setattr(metadata, "adjustment", PriceAdjustment.SPLIT_ADJUSTED)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"high": 99.5}, "high is below open"),
        ({"high": nextafter(101.5, 100.0)}, "high is below close"),
        ({"low": 100.5}, "low is above open"),
        ({"open": 101.8, "low": 101.6}, "low is above close"),
        ({"low": 103}, "low exceeds high"),
    ],
)
def test_rejects_invalid_ohlc(bar, metadata, fields, message):
    with pytest.raises(ValueError, match=rf"bar 1 .*: {message}"):
        validate_daily_bars([replace(bar, **fields)], metadata)


@pytest.mark.parametrize("field", ["open", "high", "low", "close"])
def test_rejects_zero_prices(bar, metadata, field):
    with pytest.raises(ValueError, match=rf"{field} must be a finite"):
        validate_daily_bars([replace(bar, **{field: 0})], metadata)


@pytest.mark.parametrize(
    "price",
    [-1, float("nan"), float("inf"), float("-inf"), True, "101.5", None, 1j],
)
def test_rejects_invalid_prices(bar, metadata, price):
    with pytest.raises(ValueError, match="close must be a finite, positive"):
        validate_daily_bars([replace(bar, close=price)], metadata)


@pytest.mark.parametrize(
    "volume", [-1, 1.0, False, float("nan"), float("inf"), "100", None]
)
def test_rejects_invalid_volume(bar, metadata, volume):
    with pytest.raises(
        ValueError, match="volume must be a nonnegative integer"
    ):
        validate_daily_bars([replace(bar, volume=volume)], metadata)


@pytest.mark.parametrize(
    "session_date", [datetime(2024, 8, 30), "2024-08-30", None]
)
def test_rejects_values_other_than_a_session_date(bar, metadata, session_date):
    bar = replace(bar, session_date=session_date)
    with pytest.raises(ValueError, match="bar 1: session_date must be a date"):
        validate_daily_bars([bar], metadata)


def test_rejects_nonconsecutive_duplicate_session_dates(bar, metadata):
    later = replace(bar, session_date=date(2024, 9, 3))

    with pytest.raises(
        ValueError, match=r"bar 3 .*: duplicate session date for 'SPY'"
    ):
        validate_daily_bars([bar, later, bar], metadata)


def test_rejects_out_of_order_session_dates(bar, metadata):
    later = replace(bar, session_date=date(2024, 9, 3))

    with pytest.raises(
        ValueError, match=r"bar 2 .*: session date is out of order"
    ):
        validate_daily_bars([later, bar], metadata)


@pytest.mark.parametrize("symbol", ["QQQ", "spy", "SPY "])
def test_rejects_unexpected_symbols(bar, metadata, symbol):
    later = replace(bar, session_date=date(2024, 9, 3), symbol=symbol)

    with pytest.raises(
        ValueError, match=rf"bar 2: expected symbol 'SPY', got {symbol!r}"
    ):
        validate_daily_bars([bar, later], metadata)


def test_rejects_empty_datasets(metadata):
    with pytest.raises(ValueError, match="daily bar dataset is empty"):
        validate_daily_bars([], metadata)


def test_rejects_non_sequence_inputs(metadata):
    with pytest.raises(ValueError, match="bars must be a sequence"):
        validate_daily_bars(iter(()), metadata)


def test_rejects_non_bar_records(metadata):
    with pytest.raises(ValueError, match="bar 1: expected a DailyBar"):
        validate_daily_bars([object()], metadata)


def test_requires_dataset_metadata(bar):
    with pytest.raises(ValueError, match="metadata must be DatasetMetadata"):
        validate_daily_bars([bar], None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", ""),
        ("source", 42),
        ("symbol", " "),
        ("symbol", "SPY "),
        ("source_version", ""),
        ("source_version", 42),
        ("adjustment", "raw"),
        ("adjustment", None),
        ("retrieved_at", datetime(2024, 9, 5)),
        ("retrieved_at", date(2024, 9, 5)),
    ],
)
def test_rejects_invalid_metadata(metadata, field, value):
    with pytest.raises(ValueError, match=rf"^{field} must be"):
        replace(metadata, **{field: value})


def test_accepts_source_version_and_aware_retrieval_time(bar, metadata):
    offset = timezone(-timedelta(hours=4))
    retrieved_at = datetime(2024, 9, 5, 9, tzinfo=offset)
    versioned = replace(
        metadata, retrieved_at=retrieved_at, source_version="snapshot-2"
    )

    validate_daily_bars([bar], versioned)

    assert versioned.retrieved_at is retrieved_at
    assert versioned.source_version == "snapshot-2"
    assert metadata.source_version is None


@pytest.mark.parametrize(
    ("adjustment", "label"),
    [
        (PriceAdjustment.RAW, "raw"),
        (PriceAdjustment.SPLIT_ADJUSTED, "split_adjusted"),
        (PriceAdjustment.TOTAL_RETURN_ADJUSTED, "total_return_adjusted"),
    ],
)
def test_adjustment_conventions_preserve_prices(
    bar, metadata, adjustment, label
):
    dataset_metadata = replace(metadata, adjustment=adjustment)

    validate_daily_bars([bar], dataset_metadata)

    assert dataset_metadata.adjustment.value == label
    assert bar.close == 101.5
