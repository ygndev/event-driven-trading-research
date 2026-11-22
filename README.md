# Event-Driven Quantitative Trading Research & Execution Simulator

The research goal is to study how the timing of market events, order execution, and trading costs affect simulated strategy results.

Planned work includes historical market data ingestion and replay, event-driven backtesting, simulated order fills and transaction costs, and reproducible experiment reporting.

The current repository contains a daily market data contract, a Yahoo Finance ingestion workflow, deterministic daily replay, a session-level signal interface, an order ledger, portfolio accounting, target-weight order planning, and coordinated open-price execution. A deterministic daily backtest connects these components for synthetic research, with performance evaluation based on the resulting sessions and committed fills.

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

Registering an order records CREATED. A CREATED order can move to ACCEPTED or REJECTED. An ACCEPTED order can move to FILLED, REJECTED, or CANCELLED. All three are terminal. Rejections require a nonblank reason. `cancel(order_id, session=...)` records a later decision close in `cancellation_session` on a new history entry for the original order. `snapshot(order_id)` returns the current immutable record. `history()` returns an immutable sequence of successful operations in call order, including registration, intermediate states, fills, rejection reasons, and cancellation dates.

`record_fill(fill)` takes an externally supplied simulated execution result with the order ID, symbol, execution session, side, quantity, and price. It requires an accepted order, matching symbol and side, the full ordered quantity, and a finite positive price. The execution session must be strictly later than the decision session and on or after the supplied eligibility bound, if present.

The ledger compares session dates without inferring an exchange calendar. It keeps history in memory and validates the order record and timing constraints. Assessing whether a fill was economically realistic or whether a SELL was covered by an existing long position requires additional execution and holdings data.

## Portfolio accounting

Create a `PortfolioLedger(initial_cash)` from `trading_research.portfolio` with positive finite USD capital. Use `apply_fill(fill, commission=0)` to book a whole-share fill and its supplied nonnegative commission. BUY debits the full cost plus commission and requires sufficient cash. SELL credits proceeds less commission and requires sufficient shares and proceeds to cover the fee. Closed positions are removed. Successful fills are booked once per order ID, and validation failures leave the ledger unchanged.

Balances, fees, and values use `Decimal`. Integers and decimals retain their values. Float inputs, including existing fill prices, are interpreted via their shortest decimal string. Calculations use a private 34-significant-digit context and raise a decimal arithmetic error if a result cannot be represented exactly. There is no rounding to cents. At unchanged marks equal to a fill's price for the traded instrument, a fill moves value between cash and holdings and reduces equity only by the commission.

`snapshot()` returns detached, read-only cash, holdings, and cumulative fees. `history()` returns immutable transaction records in application order, each with its original fill, commission, and resulting cash. `mark_to_market(snapshot, prices)` adds cash and the marked value of all open positions. It requires positive finite caller-supplied prices for every holding and never fetches or carries forward quotes. An empty portfolio values to cash.

Direct calls to the standalone ledgers are independent and non-atomic. An order ledger can record a fill that portfolio accounting subsequently rejects. Use the coordinator for consistent updates across both ledgers. Fill validation establishes lifecycle and accounting consistency within each ledger. Economic realism and the quality of valuation marks depend on the caller's inputs. This in-memory ledger covers single-currency cash and shares. Settlement, cost basis, and P&L reporting are outside the fill-accounting scope.

**Fill chronology.** Portfolio fills must be booked in nondecreasing execution-session order across all symbols. Fills on the same date use the supplied order and each booking sees the cash and holdings from preceding bookings. The daily runner supplies SELL before BUY, with the documented tie-breaks. The date check uses committed fills. Order registration and acceptance do not advance it. A backdated fill raises `ValueError` with the order ID, symbol, attempted date and last committed fill date. The coordinator keeps the accepted order and both committed histories intact.

With $100 and a flat SPY position, a January 5 BUY of one share at $100 leaves $0 and one share. A submission to sell that share for $120 dated January 4 raises the chronology error and leaves the January 5 BUY as the only booked fill. A valid January 8 sale of that share at $120 leaves $120 and a flat position with two committed fills. Both examples use zero fees.

The guarantee is at daily-session granularity. Recorded closes, effective splits and dividend payments have their own date checks. Standalone callers still need to sequence those actions. For example, a payment dated January 6 can be booked after a January 8 fill when the recorded-close checks allow it. The daily runner sequences declared payments, including non-session dates, in its own replay.

## Standalone corporate-action accounting

`StockSplit` and `CashDividend` in `trading_research.corporate_actions` describe declared events. These records establish the supplied dates and amounts. Historical announcement timing and data provenance require separate evidence.

- Call `PortfolioLedger.apply_split(split)` before the effective session's fills for the symbol. The ratio is **new shares divided by old shares** and is stored as a `Fraction`. Use `2` for 2-for-1 and `Fraction(1, 2)` or `Decimal("0.5")` for 1-for-2. A split changes share quantity and conserves cash. It leaves prices to the caller. For example, 10 shares at an as-traded $200 become 20 shares at an as-traded $100, with $2,000 of position value either way. The ledger accepts one split per symbol and effective session and rejects fractional results before booking. An unheld symbol records a zero-quantity adjustment. See `split_history()` for the immutable adjustments.
- For an ordinary USD dividend, record the actual preceding eligible close with `record_close(session)` after booking that session's trades. The ledger retains the captured holdings by session. Call `accrue_dividend(dividend, session=ex_date, preceding_session=...)` on ex-date before capturing that day's close. Under the settlement simplification, entitlement follows the preceding close's holdings, even if an ex-date trade has already booked. The caller identifies the preceding eligible session. Missing or mismatched close history raises an error. A known zero holding produces no receivable and consumes the event ID. Use a stable action identifier and a positive finite USD amount per share on the preceding close's share basis. Zero-amount records are rejected.
- Payment may occur on or after ex-date as declared by the action. Call `pay_dividend(action_id, session=payment_date)` on the exact payment date before recording its close. This credits available cash once and records an immutable payment with its original receivable. Missing entitlement, the wrong payment date, and repeat payment raise errors. See `dividend_payments()` for the audit history.

`PortfolioSnapshot.unpaid_receivables` contains the unpaid assets, separate from available cash. Use `mark_to_market(snapshot, prices, include_receivables=True)` for equity including those assets. The default values cash and holdings. On payment, the unpaid asset leaves the current snapshot and the same amount enters cash. At fixed marks, receivable-inclusive equity is unchanged by payment. Dividend accrual and payment use the ledger's exact Decimal context. Fill history and commission accounting remain separate.

These methods also support direct ledger use. Record the required closes in chronological order. Use as-traded marks and opens on each date and reconcile simultaneous events to the correct share basis. The scope covers ordinary USD cash dividends and whole-share splits. Fractional settlement, cash-in-lieu, stock distributions, mergers, spin-offs, rights issues, withholding, foreign exchange, broker-specific settlement, and reinvestment remain outside this model.

## Next-session open execution

Create one `NextOpenExecution(adjustment_bps=..., commission=...)` per sequential simulation run. Call `propose(order, bars, metadata)` with an accepted `OrderSnapshot`, or a `MarketOrder` whose acceptance the caller has confirmed. Supply the instrument's daily bars in ascending session order. The model uses the reported open of the first observed bar strictly after the decision session and on or after any explicit eligibility date. No eligible bar means no proposal. Calendar gaps require no date inference. Each order ID can yield one proposal per model instance.

Only data declared RAW is eligible. The model rejects both split-adjusted and total-return-adjusted data, including split-adjusted Yahoo ingestion output. A RAW declaration still depends on source provenance. The model never reconstructs as-traded prices.

The modeled price is `open × (1 + bps / 10000)` for BUY and `open × (1 − bps / 10000)` for SELL. Basis points must be finite and nonnegative, and the result must be positive. The adjustment is a hypothetical execution disadvantage. A separate fixed USD commission is reported once per proposed full fill. Prices, basis points, and fees use the accounting decimal policy, with exact arithmetic checked at 34 significant digits and no tick-size rounding.

A `ProposedExecution` contains the `Fill`, commission, reference open, and basis points. Use the coordinator below to validate and commit a proposal to both ledgers. This model uses daily observations and cannot assess liquidity, queue position, spread, market impact, or exchange matching.

## Coordinated execution

Initialize an `ExecutionCoordinator(orders, portfolio)` from `trading_research.coordinator` with the standalone ledgers. Seed it with existing orders or use `submit_orders(orders)` to register and accept a batch during a run. Pass `cancellations` as `(order_id, session)` pairs to stage cancellations and new orders together. An invalid batch leaves both ledgers as they were before the call. Existing fills must agree across both histories in transaction order. The coordinator owns its ledger copies and exposes their state through `order_snapshot`, `order_history`, `portfolio_snapshot`, and `portfolio_history`.

Use the current order snapshot to price a proposal with `NextOpenExecution`, then pass it to `execute(order_id, proposal)`. The coordinator checks that the proposed price follows the model formula and delegates lifecycle, identity, timing, and funds checks to the existing ledgers. It stages both updates on detached copies and publishes the pair together. One portfolio transaction carries the fill and the supplied commission. The proposal's commission is informational.

Insufficient cash or holdings and commissions exceeding sale proceeds produce a terminal REJECTED order with a reason. The portfolio is unchanged. Invalid proposals or unexpected exceptions preserve both ledgers and propagate to the caller. An unexpected staging failure can be retried using the same proposal.

Calls are processed sequentially in the order supplied. Earlier bookings change cash and holdings available to later orders. Atomicity applies to this in-memory coordinator under sequential access. Its state is lost on process exit, and callers must serialize access. The coordinator uses the proposal's supplied opening price and execution session, and relies on the execution model for observed-bar selection. Order planning is separate from the coordinator. Settlement, liquidity, and live brokerage remain outside this workflow.

## Daily backtest

Call `run_backtest(datasets, strategy, initial_cash=..., adjustment_bps=..., commission=...)` from `trading_research.backtest`. Supply validated RAW-declared daily-bar datasets and a callable that returns signals for each completed session. The test fixtures use synthetic RAW bars as a controlled assumption about the price basis.

For every observed session, replay runs these phases in order.

1. **Open.** Replay delivers opening observations containing only symbol, session, price, and adjustment basis. Pending orders with a matching eligible open are proposed at that price using the configured adjustment and commission. The coordinator commits SELL before BUY, then by symbol and decision date. Each fill and fee is recorded once across the coordinated histories. A terminal rejection leaves the portfolio unchanged.
2. **Completed observations.** Replay delivers all completed bars for the session in symbol order. The strategy context can access that session and completed history through that date.
3. **Decision.** The strategy runs once. A nonempty signal batch reconciles pending orders against the updated desired weights and sizes eligible orders at current closes. Omitted symbols retain their prior weights, and an empty batch triggers no replanning. New orders are registered and accepted without reserving cash. They are eligible only after the decision session.
4. **Valuation.** Record immutable cash, holdings, fees, and equity using current closes. A missing close for any holding raises an error. History never supplies a substitute mark.

A later explicit target with a different weight cancels a still-accepted order for that symbol at the decision close. An identical weight or an omitted symbol preserves the accepted order and its original quantity. The full allocation still counts toward the weight limit. Cancellation can occur when the symbol has no bar on that session. When a replacement needs a current close, the updated target is retained for the next nonempty decision batch with that close. Open holdings still require current marks. A flat target with no holdings requires no replacement.

At each opening the simulator considers only orders still accepted after earlier decisions. A target from that day's close takes effect after its opening fills. This idealized daily-bar cancellation happens at the decision boundary. Broker acknowledgements and intraday cancellation races fall outside the model.

Cancelled orders leave the pending set and retain their original IDs and cancellation dates in `order_history`. Cancellation produces no fill, cash movement, holdings change, or commission. Strategy feedback continues to report opening fills and rejections and the pending orders at the start of its close callback. Only orders still accepted at dataset end appear in `outstanding_orders`. There is no forced liquidation.

The runner uses the union of observed session dates. It assumes supplied opens are usable references and uses an idealized full-fill price, with no exchange calendar, liquidity or settlement model. Price gaps, fees, and whole-share constraints can leave actual holdings away from the desired allocation. Coordination is in-memory and sequential. An unexpected error aborts the run, and the failed coordinated fill leaves its ledger pair as it was before the attempt.

### Corporate actions in daily replay

Pass a collection of declared `StockSplit` and `CashDividend` records as `corporate_actions=...` to `run_backtest`. The runner validates the collection before replay. Each symbol needs a bar on its split effective date or dividend ex-date. Dividends also need a preceding observed close for that symbol. Input order is flexible. The runner rejects repeated dividend IDs and multiple events of the same type for one symbol and session.

For each observed session, the runner checks for pending orders crossing any effective split. A conflict raises `PendingOrderSplitError` before booking actions or fills at that opening. The order stays accepted at its original quantity. Fractional split results reject the date's staged corporate-action batch. After preflight, the runner books any earlier payment-only dates at their actual dates, then applies effective splits, accrues ex-date entitlements, credits payments due that date, and executes opening orders. Splits and entitlements on the same date are processed in symbol order. Their dated batch is published together. This is an in-memory consistency guarantee for the action batch.

Entitlement uses the affected symbol's preceding observed close, including when another instrument has an intervening bar. It survives an ex-date sale and excludes ex-date purchases. When a split and dividend share a date, supply the dividend amount on the preceding close's share basis. The model assumes a payment becomes spendable before opening orders on its declared date. Actual availability time is unknown. Payment-only dates create no strategy callbacks or equity snapshots. A payment dated after the last bar appears only in the payment audit, and final close cash and receivables retain their as-of balances.

The closing mark includes available cash, current as-traded holdings, and unpaid receivables. Planning sizes with cash and holdings, excluding unpaid receivables. Payment moves the asset to cash exactly once. The result exposes `split_adjustments` and `dividend_payments` separately from fills. Performance long exposure measures marked holdings as a share of total equity, excluding receivables from the holdings numerator.

RAW metadata declares the data's provenance. A real ETF study still needs verified as-traded bars, corporate-action terms and dates, coverage checks against the exchange calendar, and documented historical availability. The supplied raw closes remain available to strategies through the session context. This runner books supplied events at the stated boundaries and infers no declaration or publication times. Its supported actions and settlement restrictions are described above.

## Performance evaluation

Call `evaluate_performance(result, start_date=..., end_date=..., risk_free_rate_per_session=0)` from `trading_research.performance`. The evaluator reads the immutable backtest result. It uses initial cash as the starting equity and high-water mark, so the initial-to-first-snapshot return and drawdown are included. Returns are simple fractions between observed snapshots. Missing dates remain gaps, and external capital flows are outside this model.

Calendar CAGR uses the supplied dates and a 365-day year. The start date must be on or before the first observed session, and the end date must match the final session. Provide both dates or omit both. CAGR requires at least 365 elapsed days and two observations. The result records the reason when dates are omitted or the sample is too short.

Annualized volatility uses the sample standard deviation of returns and a multiplier of `sqrt(252)`. Sharpe uses the same scale with a constant per-observation risk-free return, zero by default. Both require two returns, and Sharpe is undefined at zero volatility. Undefined statistics are `None`. The 252-session scale is an assumption. A return spanning a gap can cover more than one trading session.

Drawdown is the largest peak-to-trough decline, expressed as a nonnegative fraction. Long exposure is `(equity − cash − unpaid dividend receivables) / equity` at each close and is averaged over observed snapshots. Turnover is total gross committed fill notional divided by mean observed closing equity, excluding initial capital from the denominator. Each committed fill counts as one executed order. Commissions and executed notional are in USD. Simulated price adjustments enter through fill prices and are not measured market costs.

USD totals use the existing exact Decimal accounting policy. Ratios and statistical calculations use a separate local 34-digit Decimal context that permits rounding for division, roots, and powers. The evaluator rejects empty snapshots, nonpositive equity, inconsistent cash and holdings, or unordered session dates.

## Split-aware strategy price history

`SessionContext.history(symbol)` still returns the completed raw observations. Use `context.strategy_close(observation)` to express a raw close on the current decision session's share basis. The method also accepts `as_of=` for an earlier comparison date bounded by the observation and the decision. Raw bars continue to supply opening fills, order sizing, closing marks, and accounting.

For an observed close at session `S` compared as of `T`, divide its raw close by the product of the same symbol's **new shares / old shares** split ratios for effective dates `S < date <= T`. The bar on a split's effective date already uses the new basis. A $200 pre-split close becomes comparable at $100 after a 2-for-1 split, and a current $100 close stays $100. Comparing the raw quotes would imply a 50% loss and can generate a false mean-reversion entry or momentum exit. The reverse-split rule follows the same formula, and missing observations remain gaps.

Each decision context receives only that session's successfully applied splits and links to previous completed contexts. Retaining an earlier context freezes its view. `strategy_close` uses the existing Decimal input convention and returns a Decimal when the cumulative ratio is one. An adjusted value uses an exact `Fraction`, including values such as $100/3 that have no finite Decimal representation. The strategies compare those fractions exactly. Ordinary Decimal accounting and unadjusted Decimal threshold comparisons retain the 34-digit exact-arithmetic policy.

This is split-adjusted price history. Cash dividends can still affect signals through the ex-date price move. The view requires RAW-declared observations and rejects provider-adjusted inputs. An effective session identifies when the share basis changes. Historical announcement availability and raw price provenance still need independent evidence for empirical research.

## SPY time-series momentum

Pass `time_series_momentum` from `trading_research.momentum` as the strategy callable. The research hypothesis is that positive trailing price returns may indicate persistent trends. The 63-interval lookback and 21-observation rebalance cadence are fixed initial research assumptions.

Momentum is `current_close / close_63_observations_ago − 1`. The rule compares the two positive finite closes on the same share basis to determine the sign without dividing. It needs 64 observed SPY closes for its first decision, then evaluates at observations 85, 106, and every 21 thereafter. Missing SPY observations do not advance the count. Warm-up and between-decision sessions emit no signal. A scheduled decision emits 100% SPY for positive momentum or an explicit 0% for zero or negative momentum. Sizing runs after the decision close, and execution is eligible at the next observed SPY open.

In the synthetic round trip, the 64th close was $110 versus a $90 lookback close, so a $1,000 portfolio targeted nine shares. The next open was $108, and a 25 bp adjustment gave a $108.27 BUY fill. The 85th close was $100 versus $120 at the lookback, so the strategy targeted liquidation. The next open of $105 gave a $104.7375 SELL fill. With $1.25 commission per fill, the run ended with $965.7075 cash and equity and zero shares.

This is a price-only rule, so its signal omits dividend returns. Real ETF studies require a defensible policy for splits, dividends, and price provenance. A 100% target can also be rejected when the next-open fill cost plus commission exceeds available cash. Synthetic tests exercise the implementation and do not establish historical profitability.

## SPY benchmark comparison

Call `compare_spy_benchmarks(metadata, bars, initial_cash=..., adjustment_bps=..., commission=...)` from `trading_research.benchmark` with a single RAW-declared SPY dataset. Each run starts with the same capital and uses the same observed bars, fixed commission, adverse execution adjustment, and final session. The 64th observed close is the common decision anchor, even when momentum's scheduled target is zero. The runs process warm-up history and the comparison reports the aligned window from that close through the last observed close, including the zero initial-to-anchor return.

Cash earns zero interest and generates no orders. Buy-and-hold submits a single 100% SPY target after the anchor close. The planner sizes whole shares at that close. A valid benchmark requires that entry to fill at the next observed open under the specified execution price and commission. The committed order snapshot is available as `buy_and_hold.entry`. Residual cash stays in the portfolio and shares are marked at the final raw close without a liquidation order or exit fee.

An invalid entry raises `BuyAndHoldEntryError` before the experiment returns metrics or relative results. The exception reports the symbol, anchor, status, and the original order snapshot in `entry` when an order exists. The status is `rejected` when execution rejects the full order, such as when an opening gap or fee makes it unaffordable. It is `outstanding` when the final observation leaves the entry accepted, including an anchor-only window, and `zero_shares` when the anchor-close budget cannot size a whole share. The capital, costs, observations and execution assumptions determine whether the experiment can establish its invested reference. Cash remains a separate comparator. An invalid entry leaves the experiment unranked.

Each labeled result includes the common-window equity path and the existing performance metrics. The return difference versus buy-and-hold is expressed in percentage points, and the final equity difference is in USD. Both describe this run's relative outcome. CAGR and Sharpe follow the evaluator's availability rules, including the reason recorded for unavailable CAGR.

Using the synthetic momentum example above, buy-and-hold also bought nine shares at $108.27 and retained $24.32 cash. Marking those shares at the final $106 close produced $978.32 equity with $1.25 in commissions. Momentum ended at $965.7075 with $2.50 in commissions, trailing buy-and-hold by $12.6125 and 1.26125 percentage points. Cash stayed at $1,000.

This comparison uses price-only SPY bars. Historical ETF total returns require dividend and split treatment and defensible price provenance. Yahoo's split-adjusted data remains outside the RAW execution contract. Synthetic results verify the implementation and offer no evidence of historical outperformance or statistical significance.

## SPY short-term mean reversion

Pass `short_term_mean_reversion` from `trading_research.mean_reversion` to `run_backtest`. The research hypothesis is that a large short-term decline may be followed by a partial reversal. The entry rule uses six observed SPY closes and targets 100% SPY when `current_close / close_5_observations_ago − 1 <= -0.03`. It tests the equivalent exact price inequality using positive finite closes on the same share basis.

At each backtest close, `SessionContext.execution` contains an immutable portfolio snapshot, accepted pending orders, and that session's opening outcomes. `last_order_outcome(symbol)` searches only outcomes known through that close. Market-only `generate_signals` contexts have no execution feedback. The mean-reversion rule requires the backtest context.

A committed SPY BUY starts the clock. Its session counts as the first held close. The fifth observed SPY close while holding emits an explicit zero target, eligible at the next observed SPY open. Missing SPY bars do not advance the count, and pending entry or exit orders block new signals. The existing valuation policy still requires a current close for every held instrument when another instrument creates a backtest session.

An entry rejection leaves the portfolio flat. Starting after the rejection session, a new entry requires the trailing return to cross from above −3% back to −3% or below. A pending exit remains accepted until an eligible open appears. If the dataset ends first, the order is reported as outstanding and the position remains open. When an exit is rejected, the coordinator records the rejection and preserves the portfolio. At that session's close, the strategy raises `MeanReversionExitRejected`, which carries the order snapshot, and the backtest stops. Continuing to hold after that rejection would measure a different holding period. A successful exit returns the strategy to its flat entry rule. This policy is limited to research backtests.

In a synthetic run starting at $1,000, a $97 sixth close versus $100 five observations earlier triggered a BUY for ten shares. The next observed open was $96, giving a $96.24 fill with a 25 bp adjustment. After five held closes, a zero target led to a $100.7475 SELL at the following observed open. With $1.25 per fill, final cash and equity were $1,042.575, with zero shares.

The −3% threshold and five-session holding period are uncalibrated research assumptions. Synthetic inputs establish no profitability or transaction-cost advantage. Selection and overfitting risk remain. Real ETF research still requires defensible corporate-action and dividend treatment.

## SPY mean-reversion benchmark comparison

Call `compare_spy_mean_reversion_benchmarks(metadata, bars, initial_cash=..., adjustment_bps=..., commission=...)` from `trading_research.benchmark` with one RAW-declared SPY dataset. The sixth observed close is the common evaluation anchor, including when the fixed rule stays flat. All three runs share the observations, initial USD capital, execution costs, and final observed session. The scored equity paths begin with uninvested capital at the anchor close. The first five observations supply warm-up history and contribute no trades or metrics. Missing observations remain gaps.

Buy-and-hold sets one 100% target at the anchor close. Both trading runs size whole shares from the decision close and use the same adjusted next eligible observed open and fixed commission per committed fill. Buy-and-hold keeps residual cash and marks its shares at the final close without a sale or exit commission. Cash earns zero interest and places no orders. The same `BuyAndHoldEntryError` gate applies before comparison.

Each run exposes the existing performance evaluation. Return and drawdown describe the marked equity path. Turnover uses committed notional over mean closing equity, and average long exposure includes the uninvested anchor. The evaluator retains its rules for undefined CAGR, volatility, and Sharpe. Return differences are in percentage points versus buy-and-hold and cash. Read the cash comparison from `return_difference_pp_vs_cash`. The final equity difference versus buy-and-hold is in USD.

The comparison preserves each run's `backtest.outstanding_orders` and final holdings. When `outstanding_exit` contains an accepted SELL, the equity and metrics include the marked open position and the round trip is incomplete. A rejected exit raises `MeanReversionExitRejected` and produces no benchmark comparison. Rejected entries follow the strategy's fresh-crossing rule.

Momentum's separate comparison starts at observation 64. A future strategy-to-strategy study needs an identical out-of-sample period. Synthetic cases check timing, costs, and accounting and establish no historical edge or statistical significance. ETF total-return research still needs defensible corporate-action, dividend, and price-provenance policies.

## SPY common-window strategy comparison

Call `compare_spy_common_window(metadata, bars, initial_cash=..., adjustment_bps=..., commission=...)` from `trading_research.benchmark` with one RAW-declared SPY dataset. Use the same synthetic observations for momentum, mean reversion, buy-and-hold, and zero-interest cash. Scoring starts with identical uninvested capital at the 64th observed close and ends at the last observed close. The first 63 observations supply completed-bar history and generate no orders or scored returns. Missing observations remain gaps.

Momentum uses its existing decisions at observations 64, 85, 106, and every 21 thereafter. The comparison starts calling mean reversion at observation 64, when the strategy can evaluate the return from close 59 to close 64. Its portfolio and order ledger have had no trading activity. The first committed BUY starts the existing five-held-close count. Buy-and-hold submits one full target at the anchor, and cash earns no interest and submits no orders.

All trading decisions size whole shares at the decision close. Fills use the next eligible observed open with the same adverse price adjustment and fixed commission per fill. The final close marks open positions, and each run's `backtest.sessions[-1].portfolio` reports final holdings and cash. Accepted orders still waiting for an open remain in `backtest.outstanding_orders`. The comparison's `outstanding_exit` identifies an incomplete mean-reversion round trip with an accepted SELL. A rejected mean-reversion entry follows the strategy's fresh-crossing rule. A rejected mean-reversion exit raises `MeanReversionExitRejected` and aborts the comparison. An invalid buy-and-hold entry raises `BuyAndHoldEntryError`.

The four runs share the performance evaluator and retain its undefined statistics. Each row of `pairwise_differences` subtracts the right portfolio from the left in total-return percentage points and final-equity USD. These differences describe one aligned window. Synthetic runs verify timing, accounting, and scoring and establish no statistical significance or historical profitability. The RAW declaration in a synthetic fixture describes the simulation contract. Historical ETF studies still need verified price provenance, defensible split and dividend treatment, and a common out-of-sample protocol.

## SPY transaction-cost sensitivity

Call `run_transaction_cost_sensitivity(metadata, bars, initial_cash=...)` from `trading_research.sensitivity` with one RAW-declared synthetic SPY dataset. The analysis independently reruns momentum, mean reversion, buy-and-hold, and zero-interest cash for each scenario. All runs use the same observations, starting capital, and fixed strategy rules. The comparison scores from the 64th observed close through the last observed close.

| Scenario | Adverse adjustment | USD commission per fill |
|---|---:|---:|
| A | 0 bps | $0.00 |
| B | 0 bps | $1.25 |
| C | 10 bps | $1.25 |
| D (synthetic reference) | 25 bps | $1.25 |
| E | 50 bps | $1.25 |
| F | 25 bps | $5.00 |

The adjustment raises each eligible opening BUY price and lowers each eligible opening SELL price by the stated basis-point fraction. The commission is a separate cash charge on each committed fill. Fixed USD fees have greater proportional impact on smaller accounts and smaller orders. These illustrative assumptions do not estimate spread, market impact, or liquidity from daily OHLCV.

Results are returned in scenario order. Each `comparison` preserves equity paths, performance metrics, fills, orders, and pairwise differences. Each row of `differences_vs_zero_cost` subtracts that portfolio's scenario A outcome from the current scenario in final USD equity, return percentage points, executed orders, and gross filled shares across BUYs and SELLs.

Orders are sized in whole shares using decision-close equity. Subsequent affordability is checked at the adjusted execution price plus commission. A higher-cost scenario can reject a full BUY or change cash and the size of a later trade. Final-equity differences therefore include changes in exposure and portfolio path as well as modeled execution prices and direct commissions. Returns need not move monotonically with the cost assumptions.

Outstanding active strategy orders stay visible in each completed comparison. An invalid buy-and-hold entry or rejected mean-reversion exit raises `CostScenarioFailed` with the scenario, costs, and the original exception as its cause. The analysis returns its scenario collection only after all six comparisons succeed.

The daily-bar simulation assumes full fills at modeled next-open prices and marks open positions at the final close. These tests establish simulation behavior on synthetic prices. They provide no evidence of statistical significance or real-world profitability.

## SPY chronological research protocol

Call `evaluate_spy_research_periods(metadata, bars, initial_cash=..., adjustment_bps=..., commission=...)` from `trading_research.chronology`. The predefined research windows have inclusive session-date boundaries.

| Research period | Start | End |
|---|---|---|
| Development | 2010-01-01 | 2018-12-31 |
| Validation | 2019-01-01 | 2022-12-31 |
| Final test | 2023-01-01 | 2025-12-31 |

Each period uses the ordered SPY history through its last available permitted close. The anchor is the first observed close within the period with at least 63 earlier completed observations. If the supplied history starts late, early in-period observations can supply warm-up. A period with no observations or no eligible anchor raises an error. Session dates come from the input, so weekends, holidays, and missing sessions do not generate synthetic observations. The optional `periods` argument accepts explicit chronological, non-overlapping ranges.

Replay builds indicator history during warm-up. Trading and scoring begin at the same anchor with fresh ledgers and identical uninvested capital across all four portfolios. Momentum keeps its 64th, 85th, 106th observation schedule from the start of the supplied history and stays in cash until an in-period scheduled decision. Mean reversion can act at the anchor using its five-interval history. Buy-and-hold makes its one allocation decision at the anchor, and cash stays uninvested. Orders size at a decision close and can first fill at the next eligible observed opening.

The period ends with a mark at its last observed close. Cash and holdings are retained in the result and each later period starts independently. A mean-reversion position can be open because fewer than five held closes have elapsed or because its exit order still awaits an open. Both are incomplete round trips. The result exposes the marked position and any outstanding order without creating a closing transaction. A rejected exit within the period raises `MeanReversionExitRejected`. An invalid buy-and-hold entry propagates `BuyAndHoldEntryError` with that period's actual anchor. The entry must fill inside the period.

Each result exposes the configured `period`, `warmup_observation_count`, and `comparison`. The count includes every supplied observation before the anchor. The comparison's `start_session` and `end_session` are the actual scored dates. It retains execution assumptions, order histories, and performance metrics, including undefined values and the reason when CAGR is unavailable. Returns are evaluated separately for each period.

The windows are fixed research choices. Keep the final-test window reserved for evaluation after rules and assumptions are fixed, since using it for selection compromises the holdout. Executable tests use fabricated prices and do not establish economic out-of-sample performance. A RAW declaration alone cannot verify real-market provenance. Historical ETF research also requires defensible treatment of splits and dividends.

## Historical ETF data preflight

Call `audit_etf_data` from `trading_research.data_audit` before empirical research. Supply one `AuditDataset` per symbol. Each wraps the existing `DatasetMetadata` and ordered `DailyBar` records. `requested_range=(start, end)` is inclusive. `snapshot_id` can identify a provider dataset or stored retrieval. Supply a UTC retrieval timestamp and the provider's declared adjustment convention.

```python
from trading_research.data_audit import AuditDataset, audit_etf_data

report = audit_etf_data(
    [AuditDataset(metadata, bars, requested_range=(start, end))],
    corporate_actions=actions,
    expected_sessions=exchange_sessions,
    calendar_range=(calendar_start, calendar_end),
)
for finding in report.findings:
    print(finding.severity, finding.symbol, finding.session, finding.code, finding.message)
```

The audit reuses the daily OHLCV checks and gathers row, symbol, session order, duplicate, and coverage findings without repairing the source. Missing or inconsistent provenance and a declared split- or total-return-adjusted basis block RAW execution. A consistent RAW declaration remains unverified until independent price provenance is obtained.

Supply the expected-session sequence from a credible exchange calendar. With an explicit inclusive `calendar_range`, the sequence covers that entire range, including holidays and closures. Without bounds, only the span from the first to the last expected session is established. The requested range limits which sessions are required for each symbol. Without a requested range, the supplied calendar span is the expected coverage. Missing and unexpected sessions within checked coverage block. Missing or partial calendar evidence produces an unverified finding.

Supply declared `StockSplit` and `CashDividend` records together. Duplicate or conflicting records, invalid terms, unknown symbols, missing split or ex-date bars, and missing preceding dividend closes block. A payment after the last observed close is a warning. Payment dates need no market bar. Optional `price_move_threshold=Decimal("0.30")` flags absolute overnight or close-to-close moves strictly above 30%. These warnings prompt investigation and do not infer missing corporate actions.

For file integrity, pass `source_path` and `source_sha256`. With an existing Yahoo manifest, use its `source_file` relative to the manifest and its `source_sha256`. Hash checks cover the supplied source bytes. The normalized observations are validated separately, and a matching digest does not establish how those observations were derived. The audit never edits files.

The immutable report records blocking findings, warnings, unverified checks, and checks that passed. A clean schema and a matching calendar do not establish provider licensing, independent as-traded provenance, action completeness, or historical availability. Those remain separate requirements before licensed empirical backtesting.

## Offline Massive daily history

`parse_daily_aggregates` accepts parsed JSON from the Massive Stocks custom bars endpoint `/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}`. Pass the credential-free URL captured with the response, the expected ticker, the requested price convention, and the actual UTC retrieval time. The URL must use inclusive `YYYY-MM-DD` dates and explicitly specify `adjusted=false` for provider-reported RAW or `adjusted=true` for split-adjusted bars. The response's Boolean `adjusted` flag and case-sensitive `ticker` must agree with the request. Massive's flag concerns splits. The endpoint supplies no dividend-adjusted series. [Custom Bars](https://massive.com/docs/rest/stocks/aggregates/custom-bars) · [Stocks REST overview](https://massive.com/docs/rest/stocks/overview#aggregate-bars-ohlc) · [Massive adjustment FAQ](https://massive.com/knowledge-base/article/is-massives-stock-data-adjusted-for-splits-or-dividends)

```python
from trading_research.daily_bars import PriceAdjustment
from trading_research.data_audit import audit_etf_data
from trading_research.massive_history import parse_daily_aggregates

capture = parse_daily_aggregates(
    saved_payload,
    request_url=saved_request_url,
    expected_symbol="SPY",
    adjustment=PriceAdjustment.RAW,
    retrieved_at=saved_retrieval_utc,
)
report = audit_etf_data(
    [capture.dataset], expected_sessions=exchange_sessions, calendar_range=calendar_bounds,
)
```

The adapter maps `o`, `h`, `l`, `c`, and `v` to OHLCV and checks the existing bar contract. Volume must represent an exact nonnegative whole number. The `t` field is the integer Unix millisecond start of the daily window. The adapter requires midnight in `America/New_York` and uses that local date as the session label. Historical daylight-saving rules come from the IANA time zone. The window start supplies no trade time or information-availability timestamp. [Massive timestamp and request guide](https://massive.com/knowledge-base/article/how-should-i-go-about-reporting-a-possible-massive-data-discrepancy)

The adapter handles one historical response with `status="OK"`. An empty response, a provider error, a `DELAYED` status, or a response with `next_url` raises an error. The retained request ID, request URL, retrieval time, and base aggregate count help identify the capture. `resultsCount` must match the row count, and `queryCount` must fit the base aggregate limit. A response within the limit can still have gaps. Pass a credible exchange calendar to the audit to check completeness. For stored files, pass the original source path and reference SHA-256 on the `AuditDataset` using `dataclasses.replace` before auditing. A consistent request and response establish the provider's declaration. Independent as-traded provenance remains unresolved. [Custom Bars response and limits](https://massive.com/docs/rest/stocks/aggregates/custom-bars) · [Massive status guide](https://massive.com/knowledge-base/article/how-quickly-can-i-access-massives-market-date)

`parse_split` converts one current `/stocks/v1/splits` result. It maps `ticker` and `execution_date` to the symbol and effective session, and uses `split_to / split_from` for the exact new-to-old share ratio. It accepts forward and reverse splits and rejects stock dividends. Keep unresolved share changes visible when auditing coverage. [Massive Splits](https://massive.com/docs/rest/stocks/corporate-actions/splits)

Cash dividends still require review before constructing `CashDividend`. The candidate fields are `id`, `ticker`, `ex_dividend_date`, `pay_date`, and USD `cash_amount`. Massive reports the as-paid amount, and `distribution_type` describes recurrence. Those fields alone leave the prior-close share basis unresolved when a split shares the ex-date, and a recurring fund distribution can include other cash components. Confirm the event is an ordinary USD distribution, verify both dates, and reconcile the amount to the prior-close basis before supplying it to the simulator. [Massive Dividends](https://massive.com/docs/rest/stocks/corporate-actions/dividends) · [Massive dividend types](https://massive.com/knowledge-base/article/what-are-the-different-types-of-dividends-that-massive-supports) · [Massive as-paid amounts](https://massive.com/knowledge-base/article/does-massive-adjust-historic-dividends-for-splits)

Verification uses synthetic payloads only. Live response compatibility, per-symbol coverage, action completeness, the eligible-trade rules for daily open and close, and historical publication timing remain unverified. Confirm entitlement and rights to acquire, retain, and use snapshots and publish the intended research outputs before acquiring real data. [Provider due diligence](docs/market_data_provider_decision.md)

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
