# Offline Massive SPY snapshot preflight

[Readiness audit](empirical_data_readiness.md) · [Licensed pilot protocol](licensed_data_pilot_protocol.md)

## Input layout

Use a private, Git-ignored directory on D:. An example layout is `bars.json`, `calendar.json`, `splits/page1.json`, `dividends/page1.json`, and one local review record for each gate under `evidence/`. Each `SnapshotFile` identifies a relative filename and the lowercase SHA-256 of its original bytes. The preflight resolves only files within that directory. It checks hashes before decoding, and returns the observed hashes by input label. Capture metadata and diagnostics must exclude credentials.

Construct a `SpySnapshot` with a snapshot ID, vintage, requested inclusive range, actual first and last bar dates, scoring range, calendar file and response references. The pilot contract uses Massive, SPY, RAW and `America/New_York`. Each response reference carries the actual saved credential-free request URL and its UTC retrieval time after the requested range. Put enough earlier sessions in the requested range for the first scored observation. Empirical mode requires at least 63 earlier closes.

- Save a complete single ascending 1/day response with explicit `adjusted=false`. The existing parser checks the response flag, counts, chronological order, and OHLCV. A paginated or truncated bar capture is outside this pilot contract.
- Save the entire ordered split and dividend responses, including an explicit empty results list when applicable. The first request must filter `ticker=SPY` and use the inclusive `execution_date.gte/lte` or `ex_dividend_date.gte/lte` dates from the bar request, sorted by the same field ascending. Continuations carry a cursor on the same endpoint and must match the previous response. The last saved response must have no continuation. Preserve page order, IDs and original bytes.
- The calendar JSON has `source`, `version`, `session_zone`, `start`, `end` and ordered ISO `sessions`. Supply an independently reconciled schedule for the full capture, including warm-up. Use actual trading dates for exceptional sessions. The published calendar and early-close timing evidence belong in the review.

## Evidence and the decision

Use `preflight_spy_snapshot(root, manifest, review=...)`. A separate `SnapshotReview` identifies the responsible reviewer and the exact `manifest.fingerprint()`. Include a distinct hashed JSON evidence record for each gate below, with a named `source` and `conclusion: "confirmed"`. Keep the actual cited material and rationale with the review according to the provider agreement. The caller is responsible for the authenticity of external evidence. The validator checks the record and its binding to the inputs. It cannot authenticate licenses, independently observe exchange data, or establish historical availability itself.

| Gate | What the reviewer confirms |
| --- | --- |
| `rights` | Entitlement, intended research and comparison use, and lawful retention. Publication rights are a separate decision. |
| `calendar` | The complete expected schedule and exceptional sessions using an independent source. |
| `raw_prices` | Independent as-traded reconciliation and compatible open, close and volume definitions. |
| `actions` | Complete action streams against independent records, with explicit `split_count` and `dividend_count` even when zero. Supply the confirmed ordinary dividend IDs and verify actual share units, including overlapping events. |
| `availability` | Record decision-time availability and correction limitations with explicit `timing` and `revisions` evidence fields. |

Review split basis `new_per_old` and dividend basis `prior_close`. The split adapter uses exact original ratios. Dividends require a reviewed ordinary classification, a provider `recurring` record, USD, stable ID, ex-date, actual pay date, and exact original `cash_amount` on prior-close shares. Decimal decoding preserves the original number. The separately adjusted amount and record date cannot fill missing cash or pay-date fields. Duplicate or conflicting events and unsupported action classes block the capture. Payment dates may fall on a closed day or after the final bar.

**READY** requires structural checks and all reviewed evidence. **BLOCKED** returns findings and observed hashes, with no usable dataset or actions. The original audit's RAW provenance finding remains visible, and the separate review records its disposition. Synthetic inputs always block empirical readiness, even with a complete local test review.

## Before empirical research

Qualify the real data and rights externally, reconcile timing and revisions, and add action-aware benchmark and chronological evaluation in a separate change. Feed the same verified actions to strategies and buy-and-hold, retain each window's independent capital and warm-up, and test that dates and post-window payments cannot affect earlier decisions or scoring. Synthetic tests exercise integrity and normalization only.
