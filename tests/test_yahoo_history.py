import csv
import json
from datetime import UTC, date, datetime
from hashlib import sha256
from unittest.mock import Mock, call

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from trading_research import yahoo_history

START = date(2010, 1, 1)
END = date(2025, 12, 31)


@pytest.fixture
def history():
    return pd.DataFrame(
        {
            "Close": [101.5, 102.0],
            "Volume": [1000, 0],
            "Adj Close": [97.25, 99.50],
            "High": [103.0, 103.0],
            "Open": [100.0, 101.5],
            "Low": [99.5, 101.25],
            "Dividends": [1.25, 0.0],
            "Stock Splits": [0.0, 2.0],
            "Capital Gains": [0.25, None],
        },
        index=pd.DatetimeIndex(
            ["2024-08-30", "2024-09-03"], tz="America/New_York", name="Date"
        ),
    )


@pytest.fixture
def ticker(monkeypatch, history):
    ticker = Mock()
    ticker.return_value.history.return_value = history
    monkeypatch.setattr(yahoo_history.yf, "Ticker", ticker)
    return ticker


def test_snapshot_maps_fields_and_preserves_source(ticker, history, tmp_path):
    # Late exchange-local labels fall on the following UTC day.
    history.index += pd.Timedelta(hours=23)
    snapshot, quality = yahoo_history._ingest_symbol("SPY", START, END, tmp_path)

    ticker.assert_called_once_with("SPY")
    ticker.return_value.history.assert_called_once_with(
        start="2010-01-01",
        end="2026-01-01",
        interval="1d",
        prepost=False,
        auto_adjust=False,
        back_adjust=False,
        actions=True,
        repair=False,
        keepna=True,
        rounding=False,
        timeout=10,
    )

    source = snapshot / "provider.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["provider_timestamp", *history.columns]
        original = list(reader)
    with (snapshot / "daily_bars.csv").open(encoding="utf-8", newline="") as stream:
        normalized = list(csv.DictReader(stream))
    assert original[0]["provider_timestamp"] == "2024-08-30 23:00:00-04:00"
    assert original[0]["Adj Close"] == "97.25"
    assert original[0]["Dividends"] == "1.25"
    assert original[1]["Stock Splits"] == "2.0"
    assert original[0]["Capital Gains"] == "0.25"
    assert original[1]["Capital Gains"] == ""
    assert normalized[0] == {
        "symbol": "SPY",
        "session_date": "2024-08-30",
        "open": "100.0",
        "high": "103.0",
        "low": "99.5",
        "close": "101.5",
        "volume": "1000",
    }
    assert normalized[-1]["session_date"] == "2024-09-03"
    assert normalized[-1]["volume"] == "0"

    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "Yahoo Finance"
    assert manifest["symbol"] == "SPY"
    assert manifest["request"]["start_inclusive"] == "2010-01-01"
    assert manifest["request"]["end_inclusive"] == "2025-12-31"
    assert manifest["request"]["api_end_exclusive"] == "2026-01-01"
    assert manifest["yfinance_version"] == yahoo_history.yf.__version__
    assert datetime.fromisoformat(manifest["retrieved_at_utc"]).tzinfo is UTC
    assert manifest["price_adjustment"] == {
        "convention": "split_adjusted",
        "basis": "provider-reported",
        "independently_verified": False,
    }
    assert manifest["source_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert quality == manifest["quality"] == {
        "observations": len(original),
        "first_session": normalized[0]["session_date"],
        "last_session": normalized[-1]["session_date"],
        "duplicate_dates": 0,
        "missing_ohlcv_values": 0,
        "invalid_ohlcv_rows": 0,
        "zero_volume_rows": 1,
        "adjusted_close_values": 2,
        "corporate_actions": {
            "dividends": {"event_rows": 1, "missing_values": 0},
            "stock_splits": {"event_rows": 1, "missing_values": 0},
            "capital_gains": {"event_rows": 1, "missing_values": 1},
        },
    }
    assert len(original) == len(normalized) == 2


def test_absent_optional_fields_remain_absent(ticker, history, tmp_path):
    ticker.return_value.history.return_value = history.drop(
        columns=["Adj Close", "Dividends", "Stock Splits", "Capital Gains"]
    )
    snapshot, quality = yahoo_history._ingest_symbol("IWM", START, END, tmp_path)

    assert quality["adjusted_close_values"] == 0
    assert quality["corporate_actions"] == {
        "dividends": None, "stock_splits": None, "capital_gains": None
    }
    with (snapshot / "provider.csv").open(encoding="utf-8", newline="") as stream:
        assert csv.DictReader(stream).fieldnames == [
            "provider_timestamp", "Close", "Volume", "High", "Open", "Low"
        ]


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda frame: None, "expected a yfinance DataFrame"),
        (lambda frame: frame.iloc[:0], "no daily observations"),
        (lambda frame: frame.drop(columns="Open"), "missing required Yahoo columns: Open"),
        (lambda frame: frame.rename(columns={"High": "Open"}), "unique, nonempty"),
        (lambda frame: frame.tz_localize(None), "expected timezone-aware Yahoo dates"),
        (lambda frame: frame.set_axis(["2024-08-30", "2024-09-03"]), "expected timezone-aware"),
        (lambda frame: frame.rename(index=lambda timestamp: pd.NaT), "without missing values"),
        (lambda frame: frame.set_axis(frame.index + pd.Timedelta(days=500)), "exceed"),
        (lambda frame: frame.assign(**{"Adj Close": [float("inf"), 100.0]}), "invalid numeric"),
        (lambda frame: frame.assign(**{"Stock Splits": [-1.0, 0.0]}), "invalid value"),
    ],
)
def test_rejects_malformed_history(ticker, history, tmp_path, transform, message):
    ticker.return_value.history.return_value = transform(history)

    with pytest.raises(ValueError, match=message):
        yahoo_history._ingest_symbol("SPY", START, END, tmp_path)

    assert not (tmp_path / "yahoo").exists()


@pytest.mark.parametrize(
    ("order", "message", "duplicates"),
    [
        ([0, 0, 1], "duplicate session date", 1),
        ([1, 0], "out of order", 0),
    ],
)
def test_rejects_duplicate_and_unsorted_dates(
    ticker, history, tmp_path, order, message, duplicates
):
    ticker.return_value.history.return_value = history.iloc[order]

    with pytest.raises(ValueError, match=message) as exc:
        yahoo_history._ingest_symbol("QQQ", START, END, tmp_path)

    quality = json.loads(str(exc.value).split("Quality: ", 1)[1])
    assert quality["duplicate_dates"] == duplicates
    assert quality["invalid_ohlcv_rows"] == 0
    assert not (tmp_path / "yahoo").exists()


def test_counts_missing_values_and_invalid_ohlcv(ticker, history, tmp_path):
    ticker.return_value.history.return_value = history.assign(
        Open=[float("nan"), 101.5], High=[None, 100.0]
    )

    with pytest.raises(ValueError, match="bar 1 .*open must be") as exc:
        yahoo_history._ingest_symbol("SPY", START, END, tmp_path)

    quality = json.loads(str(exc.value).split("Quality: ", 1)[1])
    assert quality["missing_ohlcv_values"] == 2
    assert quality["invalid_ohlcv_rows"] == 2
    assert not (tmp_path / "yahoo").exists()


def test_serialization_is_deterministic(ticker, monkeypatch, tmp_path):
    clock = Mock()
    clock.now.return_value = datetime(2026, 1, 5, 10, 30, tzinfo=UTC)
    monkeypatch.setattr(yahoo_history, "datetime", clock)
    first, _ = yahoo_history._ingest_symbol("SPY", START, END, tmp_path / "first")
    second, _ = yahoo_history._ingest_symbol("SPY", START, END, tmp_path / "second")

    clock.now.assert_has_calls([call(UTC), call(UTC)])
    assert {path.name for path in first.iterdir()} == {
        "provider.csv", "daily_bars.csv", "manifest.json"
    }
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_storage_failure_does_not_publish_partial_snapshot(ticker, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("cannot write CSV")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="cannot write CSV"):
        yahoo_history._ingest_symbol("SPY", START, END, tmp_path)

    assert list((tmp_path / "yahoo" / "SPY").iterdir()) == []


def test_cli_attempts_other_symbols_after_a_bad_response(
    ticker, history, monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    set_cache = Mock()
    monkeypatch.setattr(yahoo_history.yf, "set_tz_cache_location", set_cache)
    ticker.return_value.history.side_effect = [history, KeyError("chart"), history]

    code = yahoo_history.main(
        ["SPY", "QQQ", "IWM", "--start", "2010-01-01", "--end", "2025-12-31"]
    )

    assert code == 1
    assert ticker.call_args_list == [call("SPY"), call("QQQ"), call("IWM")]
    set_cache.assert_called_once_with(str(tmp_path / "data" / ".yfinance-cache"))
    assert {path.name for path in (tmp_path / "data" / "yahoo").iterdir()} == {"SPY", "IWM"}
    output = capsys.readouterr()
    assert "QQQ (2010-01-01 through 2025-12-31)" in output.err
    assert "could not parse Yahoo's response" in output.err


def test_cli_stops_on_rate_limit(ticker, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(yahoo_history.yf, "set_tz_cache_location", Mock())
    ticker.return_value.history.side_effect = YFRateLimitError()

    code = yahoo_history.main(
        ["SPY", "QQQ", "IWM", "--start", "2010-01-01", "--end", "2025-12-31"]
    )

    assert code == 1
    ticker.assert_called_once_with("SPY")
    assert "rate limit reached" in capsys.readouterr().err
    assert not (tmp_path / "data" / "yahoo").exists()
