# Event-Driven Quantitative Trading Research & Execution Simulator

The research goal is to study how the timing of market events, order execution, and trading costs affect simulated strategy results.

Planned work includes historical market data ingestion and replay, event-driven backtesting, simulated order fills and transaction costs, and reproducible experiment reporting.

The current repository contains an installable Python package and a packaging smoke test. Market data, strategies, backtesting, and execution simulation are pending.

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
