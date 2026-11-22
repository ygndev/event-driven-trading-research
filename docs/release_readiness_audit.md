# v0.1.0 release readiness audit

**Verdict — BLOCKED for public release.** A newer flat target can leave an accepted buy executable, so the backtest can silently end with a position the strategy no longer requested. The clean installation and green suite establish a reproducible test baseline. This order state transition falls outside the existing suite.

Audited source commit `aeaeb507c41968e43772845e1ce6cd435f6f6209` on `main` on 2026-11-22. The target is an offline research infrastructure release. The review traced bar contracts and adapters through replay, signals, planning, execution, both ledgers, valuation, and research evaluation. Reproductions use fabricated bars and fills.

## Findings

### Blocker — A superseded buy can still fill

**Reproduced** in [`backtest.run_backtest`](../src/trading_research/backtest.py#L156), in its `finish_session` and `execute_open` callbacks. `finish_session` updates retained target weights and sizes from booked holdings. A zero target for a flat symbol produces no new order in [`planning.plan_orders`](../src/trading_research/planning.py). The accepted buy remains in `pending` and the next matching open fills it without checking the newer target.

Minimal reproduction in the installed package:

```python
from datetime import UTC, date, datetime

from trading_research.backtest import run_backtest
from trading_research.daily_bars import DailyBar, DatasetMetadata, PriceAdjustment
from trading_research.signals import SignalEvent

d2, d3, d4 = (date(2024, 1, day) for day in (2, 3, 4))

def dataset(symbol, dates):
    metadata = DatasetMetadata(
        "Synthetic", symbol, PriceAdjustment.RAW, datetime(2024, 2, 1, tzinfo=UTC),
    )
    return metadata, tuple(DailyBar(symbol, day, 100, 100, 100, 100, 100) for day in dates)

def strategy(context):
    if context.session_date == d2:
        return (SignalEvent("SPY", d2, 1),)
    if context.session_date == d3:
        return (SignalEvent("SPY", d3, 0),)
    return ()

result = run_backtest(
    [dataset("SPY", (d2, d4)), dataset("QQQ", (d3,))],
    strategy, initial_cash=1000, adjustment_bps=0, commission=0,
)
print(result.fills[0].execution_session, result.fills[0].quantity)
print(dict(result.sessions[-1].portfolio.positions), result.sessions[-1].portfolio.cash)
# 2024-01-04 10
# {'SPY': 10} 0
```

The result commits the full old buy after the newer flat signal. This can distort exposure, cash, and subsequent strategy decisions on valid staggered input. The smallest correction is to reject a target update that conflicts with an accepted order before changing the desired weights. If supersession is supported later, give it an explicit order transition and reconcile the pending quantity. Keep the staggered-session case as a regression when correcting the behavior.

### Important — Standalone booking can use holdings from a future session

**Reproduced** in [`ExecutionCoordinator.execute`](../src/trading_research/coordinator.py#L46) and [`PortfolioLedger.apply_fill`](../src/trading_research/portfolio.py#L95). With $100, accept two one-share SPY orders dated January 2. Execute a January 5 BUY at $100, then present a January 4 SELL at $120. With no registered closes or actions, both commit and cash ends at $120. Per-order eligibility checks pass, but the second operation sells shares that did not exist on its dated session. The main daily replay supplies chronological opens, so the reproduction uses the documented standalone coordinator interface.

Reject a fill session earlier than the latest committed fill before applying any cash or position changes. Preserve same-session ordering and validate already-seeded history when constructing a coordinator.

### Important — Benchmark validity and statistical sample size

- **Reproduced, intended fill policy with a reporting gap.** [`benchmark._compare_spy_runs`](../src/trading_research/benchmark.py#L216) makes a single buy-and-hold allocation and reports numeric benchmark differences even when it never invests. With six observed SPY closes at $100, $1,000 cash, a $101 next opening, and a later $120 close, the ten-share benchmark BUY is rejected. Its reported return is zero, identical to cash. The rejection is visible in `backtest.rejected_orders` and [the existing rejection test](../tests/test_benchmark.py#L228) deliberately preserves the no-retry policy. The smallest correction is to surface benchmark entry status in summaries and treat a never-entered comparator as unavailable for invested benchmark differences. Keep the rejection and fill policy intact.
- **Reproduced, statistical boundary defect.** [`evaluate_performance`](../src/trading_research/performance.py#L80) includes the initial-cash-to-first-snapshot return in the volatility and Sharpe sample. The [benchmark](../src/trading_research/benchmark.py#L264) also uses that first cash snapshot as the start instant. With the same $100 anchor and one later session opening at $100 and closing at $110, the buy-and-hold returns are `[0, 0.1]` and annualized Sharpe is approximately `11.225`. Only one interval elapsed after the anchor. The extra zero makes volatility and Sharpe appear estimable from a single interval. Exclude the zero-time baseline from interval statistics while retaining it for equity and drawdown. Require at least two elapsed returns.

### Important — Code publication terms are unspecified

**Verified absence.** The audited tree contains no root license file and [`pyproject.toml`](../pyproject.toml) contains no license declaration. A public portfolio reader has no stated terms from the author for code reuse. The repository also discusses separate third-party data restrictions. The owner should state the intended code license or explicit rights position before publication and keep it distinct from data permissions.

### Minor — Reproduction records and documentation accuracy

- **Verified.** [`pyproject.toml`](../pyproject.toml) pins yfinance, while pandas, pytest, the build backend, and transitive dependencies can resolve to newer versions. The fresh installation succeeded, but the documented command cannot recreate the exact verified dependency set later. Record the tested Python and resolved dependencies with the release so readers can recreate that environment.
- **Verified.** Root `.env`, `.env.local` and `private.pem` are outside the current [ignore patterns](../.gitignore). No corresponding root files or historical candidates were found. Add narrowly scoped ignores for local credential files if used, and keep checking the staged tree.
- **Reproduced documentation mismatch.** The [standalone dividend instructions](../README.md#standalone-corporate-action-accounting) say mismatched close history raises an error. [`PortfolioLedger.accrue_dividend`](../src/trading_research/portfolio.py#L204) accepts any recorded earlier close that meets its date checks. Record one share on January 2 and two on January 3, then accrue a January 4 dividend with January 2 supplied as the predecessor. The accrual uses one share. In the main backtest the predecessor comes from the symbol's bar sequence. State that selecting the correct predecessor is a standalone caller obligation. The [provider decision's original implementation description](market_data_provider_decision.md#decision-and-next-step) also says the ledger changes shares and cash only through fills. Label that paragraph historical or update it for the implemented action bookings.

## Verified strengths

- Replay validates all streams before delivery and separates open-only data from completed bars. Tests cover future-price perturbation, isolation of future split information, strict next-open eligibility, and deterministic symbol ordering.
- Planning sizes whole shares from completed raw closes. Coordinator tests cover price and fee arithmetic, atomic in-memory staging on error, terminal economic rejection, and cash and holdings consistency. Mean-reversion exit rejection stops the research run. A pending exit stays visible.
- Action tests cover pre-split dividend entitlement, receivable accounting, payment on a nontrading date, fractional split rejection without partial action booking, and pending-order conflicts on split dates. Analytical close adjustment uses splits through the decision date and leaves execution prices raw.
- Chronological runs slice future bars at each period boundary, build warm-up without active orders, and use equal initial cash and costs at the benchmark anchor. Tests cover independent windows and pending exits at a period end. Held positions are marked at the last observed close. Performance uses committed fills for turnover and commissions.
- OHLCV, hashes and explicitly supplied calendars have offline checks. The Massive adapter matches the saved request and adjustment flag. The audit preserves independent as-traded provenance as unresolved. Yahoo's snapshot tests stub acquisition.

## Installation, tests and publication hygiene

| Check | Actual result |
| --- | --- |
| Clean Windows setup | Exported tracked source into a separate D: directory, created a fresh virtual environment with CPython 3.11.9 on Windows build 26300, and installed `-e '.[dev]'` from PyPI. Relative setup paths and editable package discovery worked in the separate copy. |
| Offline suite | **613 passed** in the existing checkout (5.57 s) and **613 passed** in the fresh checkout (5.02 s), with Python socket connection and DNS calls and curl_cffi transfers blocked during test execution. The index was used for dependency installation. |
| Installed package | All 21 package modules imported with `python -I -B` in both environments. Editable metadata pointed to the expected source directory. `pip check` reported no broken requirements in either environment. Verified versions were pandas 3.1.1, numpy 2.4.6, yfinance 1.8.0, pytest 9.2.0 and tzdata 2026.6. |
| Git and local files | At audit start, `main` was clean with 27 commits, no remotes and no tags. A filename and high-signal content scan of 48 tracked paths and 102 historical blobs across refs found no credential, personal home path, committed environment or data-file candidates. The largest historical blob was 53,241 bytes. The D: workspace contained ignored environments, caches, test output and the fresh audit copy. Sampled provider-style files under pytest temporary directories came from synthetic fixtures. |

## Unresolved limitations

Cross-platform execution and other Python versions remain untested. The Git scan is heuristic. Market-data licenses, independent real price provenance, live-response compatibility, archive completeness, and point-in-time availability still require the [licensed pilot](licensed_data_pilot_protocol.md). The comparison and chronological interfaces have no corporate-action input even though the underlying backtest can book supplied actions. Treat those comparisons as synthetic price-only research until that gap is addressed. The accounting and fill model is scoped to sequential, in-memory simulation with supplied action terms, whole shares and date-level timing. The review found no documentation claim of live brokerage readiness or independently established real profitability.

## Three priorities before release

1. Close the stale accepted-order path and enforce monotonic standalone fill booking. Add regression coverage for both.
2. Surface benchmark entry status and make volatility and Sharpe depend on elapsed return intervals. Recheck common-window and chronological summaries.
3. Record the code publication terms and freeze a reproducible reference environment. Review the staged files for sensitive or licensed material before any publication decision.
