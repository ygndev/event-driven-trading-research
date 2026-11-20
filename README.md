# Event-Driven Quantitative Trading Research & Execution Simulator

The research goal is to study how the timing of market events, order execution, and trading costs affect simulated strategy results.

Planned work includes historical market data ingestion and replay, event-driven backtesting, simulated order fills and transaction costs, and reproducible experiment reporting.

The current repository contains a daily market data contract, a Yahoo Finance ingestion workflow, deterministic daily replay, a session-level signal interface, an order ledger, portfolio accounting, target-weight order planning, and coordinated open-price execution. Strategies and backtest integration are pending.

## Daily market data

Import `DailyBar`, `DatasetMetadata`, `PriceAdjustment`, and `validate_daily_bars` from `trading_research.daily_bars`.

- `DailyBar` is an immutable US equity or ETF OHLCV record. Its `session_date` labels an exchange trading session. Prices accept positive integers and finite positive floats. Volume accepts nonnegative integers, including zero. Booleans are invalid for prices and volume.
- `DatasetMetadata` records the source, symbol, adjustment convention, timezone-aware retrieval time, and optional source version. Retrieval time describes when the data was obtained. The contract makes no claim about when historical data became available.
- `PriceAdjustment` distinguishes raw (unadjusted), split-adjusted, and total-return-adjusted prices. Use a separate dataset for each convention. The value records the source's claim. Validation cannot infer or verify the adjustment convention from OHLCV values. Adjusted prices should not be treated as directly executable.
- Call `validate_daily_bars(bars, metadata)` on parsed records before use. It requires a nonempty sequence for the exact metadata symbol, checks OHLCV values and price relationships, and requires unique session dates in ascending order. It raises `ValueError` on the first problem, with the row number for row errors. It accepts gaps without consulting a market calendar and leaves records as supplied.

## Daily replay

Call `replay_daily_bars(datasets, on_event)` from `trading_research.replay`. Pass one `(DatasetMetadata, sequence_of_bars)` pair per symbol. Replay snapshots and validates every sequence before invoking the callback. Duplicate symbols, duplicate sessions, invalid bars, and out-of-order input fail before delivery. Gaps are allowed.

Replay advances by exchange session date. After the conceptual end of each session, it delivers each available bar in lexicographic symbol order. The tie-break is a replay convention and conveys no intraday ordering. Each callback receives only an immutable `MarketEvent` with the bar and its declared price adjustment. Replay advances after the callback returns and provides no history or lookahead interface.

The session boundary models when a completed daily bar can first be used in this replay. It supplies no exact close or delivery time and makes no point-in-time availability claim about a provider. Retrieval timestamps remain dataset provenance.

## Session signals

Pass the same datasets and a callable to `generate_signals(datasets, strategy)` from `trading_research.signals`. The strategy is called once per observed session, after all of that session's replay events have been delivered. The optional `on_session_end` callback on `replay_daily_bars` marks this boundary for callers that handle market events directly.

The strategy receives an immutable `SessionContext` with `session_date`, the current session's `events`, and `history(symbol)`. History includes observed events through that session, oldest first. A retained context keeps the same history as replay advances. Missing bars stay absent.

Return an iterable of `SignalEvent(symbol, decision_session, target_exposure)`, or an empty iterable. The exposure is a long-only research target from 0 to 1. It expresses research intent. Execution requires a separate order decision. The decision date must match the context, and the symbol must have been observed by that session. At most one signal per symbol is accepted per session. Results are ordered by decision session and symbol. Repeated output depends on the strategy being deterministic.

## Target-weight order planning

Call `plan_orders(targets, portfolio_snapshot, session_context)` from `trading_research.planning`. Each target is a `SignalEvent` for the completed decision session, and its exposure is a fraction of total portfolio equity. A nonempty batch is a complete allocation and must explicitly cover every open holding. When a new signal is missing, carry its previous weight forward into that allocation. Omitting an open holding from a nonempty batch is an error. An empty batch requests no changes. A zero target liquidates an open position. Duplicate targets and aggregate weights above 100% are rejected.

Equity and sizing use only RAW closing observations from the supplied decision session. Every holding and positive target needs an observed finite positive close. A zero target for an unheld symbol requires no mark. Missing marks cause an error and history is never used to fill the gap.

The target share count is `floor(equity × weight / close)`. The planner uses the shared Decimal policy for exact arithmetic and floors the nonnegative quotient to whole shares without rounding the notional. It proposes the difference from current holdings, with SELL orders first and symbols in lexicographic order within each side. Order IDs are stable for each decision session and symbol. Eligibility starts strictly after the decision session and no later session date is inferred.

The allocation is bounded by equity at the reference closes before fees. The planner does not spend or reserve cash or credit sale proceeds. BUY proposals may depend on earlier sales booking successfully and remain subject to execution-time cash and holdings checks. Overnight gaps, execution adjustments, and commissions can make the proposed allocation infeasible. Planning returns immutable order specifications. Registration, acceptance, and coordinated execution are separate steps.

## Market order lifecycle

Create a `MarketOrder` with a caller-supplied ID, symbol, BUY or SELL side, whole-share quantity, and decision session. The ID must be unique within an `OrderLedger`. Symbols are matched exactly and must be printable and free of whitespace. The optional `earliest_execution_session` is a caller-supplied lower bound. It must be strictly after the decision session.

Registering an order records CREATED. A CREATED order can move to ACCEPTED or REJECTED. An ACCEPTED order can move to FILLED or REJECTED. FILLED and REJECTED are terminal. Rejections require a nonblank reason. `snapshot(order_id)` returns the current immutable record. `history()` returns an immutable sequence of successful operations in call order, including registration, intermediate states, fills, and rejection reasons.

`record_fill(fill)` takes an externally supplied simulated execution result with the order ID, symbol, execution session, side, quantity, and price. It requires an accepted order, matching symbol and side, the full ordered quantity, and a finite positive price. The execution session must be strictly later than the decision session and on or after the supplied eligibility bound, if present.

The ledger compares session dates without inferring an exchange calendar. It keeps history in memory and validates the order record and timing constraints. Assessing whether a fill was economically realistic or whether a SELL was covered by an existing long position requires additional execution and holdings data.

## Portfolio accounting

Create a `PortfolioLedger(initial_cash)` from `trading_research.portfolio` with positive finite USD capital. Use `apply_fill(fill, commission=0)` to book a whole-share fill and its supplied nonnegative commission. BUY debits the full cost plus commission and requires sufficient cash. SELL credits proceeds less commission and requires sufficient shares and proceeds to cover the fee. Closed positions are removed. Successful fills are booked once per order ID, and validation failures leave the ledger unchanged.

Balances, fees, and values use `Decimal`. Integers and decimals retain their values. Float inputs, including existing fill prices, are interpreted via their shortest decimal string. Calculations use a private 34-significant-digit context and raise a decimal arithmetic error if a result cannot be represented exactly. There is no rounding to cents. At unchanged marks equal to a fill's price for the traded instrument, a fill moves value between cash and holdings and reduces equity only by the commission.

`snapshot()` returns detached, read-only cash, holdings, and cumulative fees. `history()` returns immutable transaction records in application order, each with its original fill, commission, and resulting cash. `mark_to_market(snapshot, prices)` adds cash and the marked value of all open positions. It requires positive finite caller-supplied prices for every holding and never fetches or carries forward quotes. An empty portfolio values to cash.

Direct calls to the standalone ledgers are independent and non-atomic. An order ledger can record a fill that portfolio accounting subsequently rejects. Use the coordinator for consistent updates across both ledgers. Fill validation establishes lifecycle and accounting consistency within each ledger. Economic realism and the quality of valuation marks depend on the caller's inputs. This in-memory ledger covers single-currency cash and shares. Settlement, cost basis, P&L reporting, and corporate actions are outside its scope.

## Next-session open execution

Create one `NextOpenExecution(adjustment_bps=..., commission=...)` per sequential simulation run. Call `propose(order, bars, metadata)` with an accepted `OrderSnapshot`, or a `MarketOrder` whose acceptance the caller has confirmed. Supply the instrument's daily bars in ascending session order. The model uses the reported open of the first observed bar strictly after the decision session and on or after any explicit eligibility date. No eligible bar means no proposal. Calendar gaps require no date inference. Each order ID can yield one proposal per model instance.

Only data declared RAW is eligible. The model rejects both split-adjusted and total-return-adjusted data, including split-adjusted Yahoo ingestion output. A RAW declaration still depends on source provenance. The model never reconstructs as-traded prices.

The modeled price is `open × (1 + bps / 10000)` for BUY and `open × (1 − bps / 10000)` for SELL. Basis points must be finite and nonnegative, and the result must be positive. The adjustment is a hypothetical execution disadvantage. A separate fixed USD commission is reported once per proposed full fill. Prices, basis points, and fees use the accounting decimal policy, with exact arithmetic checked at 34 significant digits and no tick-size rounding.

A `ProposedExecution` contains the `Fill`, commission, reference open, and basis points. Use the coordinator below to validate and commit a proposal to both ledgers. This model uses daily observations and cannot assess liquidity, queue position, spread, market impact, or exchange matching.

## Coordinated execution

Initialize an `ExecutionCoordinator(orders, portfolio)` from `trading_research.coordinator` with the standalone ledgers. Register and accept the intended orders before taking this initial copy. Existing fills must agree across both histories in transaction order. The coordinator owns its ledger copies and exposes their state through `order_snapshot`, `order_history`, `portfolio_snapshot`, and `portfolio_history`.

Use the current order snapshot to price a proposal with `NextOpenExecution`, then pass it to `execute(order_id, proposal)`. The coordinator checks that the proposed price follows the model formula and delegates lifecycle, identity, timing, and funds checks to the existing ledgers. It stages both updates on detached copies and publishes the pair together. One portfolio transaction carries the fill and the supplied commission. The proposal's commission is informational.

Insufficient cash or holdings and commissions exceeding sale proceeds produce a terminal REJECTED order with a reason. The portfolio is unchanged. Invalid proposals or unexpected exceptions preserve both ledgers and propagate to the caller. An unexpected staging failure can be retried using the same proposal.

Calls are processed sequentially in the order supplied. Earlier bookings change cash and holdings available to later orders. Atomicity applies to this in-memory coordinator under sequential access. Its state is lost on process exit, and callers must serialize access. The coordinator uses the proposal's supplied opening price and execution session, and relies on the execution model for observed-bar selection. Order planning is separate from the coordinator. Settlement, liquidity, and live brokerage remain outside this workflow.

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
