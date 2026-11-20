# Event-Driven Quantitative Trading Research & Execution Simulator

The research goal is to study how the timing of market events, order execution, and trading costs affect simulated strategy results.

Planned work includes historical market data ingestion and replay, event-driven backtesting, simulated order fills and transaction costs, and reproducible experiment reporting.

The current repository contains a daily market data contract, a Yahoo Finance ingestion workflow, and offline tests. Strategies, backtesting, and execution simulation are pending.

## Daily market data

Import `DailyBar`, `DatasetMetadata`, `PriceAdjustment`, and `validate_daily_bars` from `trading_research.daily_bars`.

- `DailyBar` is an immutable US equity or ETF OHLCV record. Its `session_date` labels an exchange trading session. Prices accept positive integers and finite positive floats. Volume accepts nonnegative integers, including zero. Booleans are invalid for prices and volume.
- `DatasetMetadata` records the source, symbol, adjustment convention, timezone-aware retrieval time, and optional source version. Retrieval time describes when the data was obtained. The contract makes no claim about when historical data became available.
- `PriceAdjustment` distinguishes raw (unadjusted), split-adjusted, and total-return-adjusted prices. Use a separate dataset for each convention. The value records the source's claim. Validation cannot infer or verify the adjustment convention from OHLCV values. Adjusted prices should not be treated as directly executable.
- Call `validate_daily_bars(bars, metadata)` on parsed records before use. It requires a nonempty sequence for the exact metadata symbol, checks OHLCV values and price relationships, and requires unique session dates in ascending order. It raises `ValueError` on the first problem, with the row number for row errors. It accepts gaps without consulting a market calendar and leaves records as supplied.

## Yahoo Finance daily history

Use this command from the repository root once you have permission for automated acquisition and local storage. The requested range is inclusive, so the command sends an exclusive Yahoo end date of 2026-01-01.

```powershell
.\.venv\Scripts\python.exe -m trading_research.yahoo_history --start 2010-01-01 --end 2025-12-31 SPY QQQ IWM
```

Each validated retrieval creates a timestamped directory under `data/yahoo/<symbol>/`.

- `provider.csv` preserves the columns and row order returned by yfinance, including the timestamp offset, `Adj Close`, cash dividends, split ratios, and capital gains when present.
- `daily_bars.csv` contains validated OHLCV only. The session label is taken from the exchange-local date. The adapter preserves Yahoo's reported OHLC and volume without adjustment, sorting, or imputation.
- `manifest.json` records request flags, UTC retrieval time, yfinance version, the source CSV's SHA-256, the declared adjustment basis, and the data quality report. The report compares requested and observed endpoints and counts duplicate dates, missing OHLCV cells, invalid OHLCV rows, zero-volume rows, and rows with nonzero corporate actions. Missing OHLCV cells also make their rows invalid. Missing action columns have a null report entry.

Yahoo describes historical Close as split-adjusted and publishes a separate Adj Close that reflects splits and distributions. The manifest records the ordinary price basis as provider-reported split-adjusted, with the adjustment factors unverified. `auto_adjust=False` and `back_adjust=False` disable yfinance's own price adjustments. They do not recover original as-traded prices. [Yahoo historical price labels](https://finance.yahoo.com/quote/SPY/history/) · [yfinance maintainer clarification](https://github.com/ranaroussi/yfinance/issues/2929)

The stored source is the table returned by yfinance. Even with `repair=False` and `keepna=True`, yfinance 1.8.0 can sort and deduplicate upstream rows, merge events, replace missing volume with zero, and substitute Close when Yahoo omits Adj Close. The quality report cannot recover those upstream details, determine exchange-calendar completeness, or establish historical availability. [yfinance 1.8.0 history processing](https://github.com/ranaroussi/yfinance/blob/1.8.0/yfinance/scrapers/history.py) · [yfinance quote parsing](https://github.com/ranaroussi/yfinance/blob/1.8.0/yfinance/utils.py)

Invalid responses fail without publishing a snapshot. The command reports other symbols independently and stops subsequent requests if Yahoo rate-limits it. Data and yfinance's local cache stay under the Git-ignored `data/` directory. Yahoo requires express prior permission for automated access or collection, and downloaded quotes may carry additional redistribution restrictions. [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) · [Yahoo data permissions](https://legal.yahoo.com/us/en/yahoo/permissions/requests/index.html)

## Local setup on Windows

Run these PowerShell commands from the repository root with Python 3.11 or newer. The environment variables apply to the current PowerShell session.

```powershell
New-Item -ItemType Directory -Force -Path .tmp, .cache\pip, .cache\python | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = (Resolve-Path .cache\pip).Path
$env:PYTHONPYCACHEPREFIX = (Resolve-Path .cache\python).Path

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest
```

The virtual environment, caches, and temporary files are local to the repository. Keep local datasets under `data/` and generated results under `outputs/`.
