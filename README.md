# Event-Driven Quantitative Trading Research & Execution Simulator

The research goal is to study how the timing of market events, order execution, and trading costs affect simulated strategy results.

Planned work includes historical market data ingestion and replay, event-driven backtesting, simulated order fills and transaction costs, and reproducible experiment reporting.

The current repository contains an installable Python package, a daily market data contract, and offline tests. Market data ingestion, strategies, backtesting, and execution simulation are pending.

## Daily market data

Import `DailyBar`, `DatasetMetadata`, `PriceAdjustment`, and `validate_daily_bars` from `trading_research.daily_bars`.

- `DailyBar` is an immutable US equity or ETF OHLCV record. Its `session_date` labels an exchange trading session. Prices accept positive integers and finite positive floats. Volume accepts nonnegative integers, including zero. Booleans are invalid for prices and volume.
- `DatasetMetadata` records the source, symbol, adjustment convention, timezone-aware retrieval time, and optional source version. Retrieval time describes when the data was obtained. The contract makes no claim about when historical data became available.
- `PriceAdjustment` distinguishes raw (unadjusted), split-adjusted, and total-return-adjusted prices. Use a separate dataset for each convention. The value records the source's claim. Validation cannot infer or verify the adjustment convention from OHLCV values. Adjusted prices should not be treated as directly executable.
- Call `validate_daily_bars(bars, metadata)` on parsed records before use. It requires a nonempty sequence for the exact metadata symbol, checks OHLCV values and price relationships, and requires unique session dates in ascending order. It raises `ValueError` on the first problem, with the row number for row errors. It accepts gaps without consulting a market calendar and leaves records as supplied.

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
