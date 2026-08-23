"""Yahoo Finance ingestion.

Yahoo is a feed, never a database. This module's only job is to move raw bars
into `prices` and to write an honest row into `ingest_log` for every attempt.

Three rules it exists to enforce:

1. ``auto_adjust=False``, always explicit. The yfinance default is True, which
   returns dividend-adjusted closes that will not match a broker screen.
2. Batched downloads at low concurrency, never per-ticker ``Ticker.history()``
   and never ``Ticker.info`` (a separate, slow endpoint that gets us rate
   limited for the sake of a name we already have in `tickers`).
3. No ``except Exception: return None``. A rate limit, a dropped connection and
   a genuine empty response are three different facts and are recorded as three
   different statuses. A network failure must never be reported as a fact about
   a stock.
"""

from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd

SOURCE = "yahoo"

#: Fields we keep from a Yahoo response, mapped to store column names.
_FIELD_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Dividends": "dividend",
    "Stock Splits": "split_factor",
}

# Substrings that identify a rate-limit response. yfinance surfaces these as
# plain strings rather than typed exceptions in several code paths, so text
# matching is unavoidable here; it is deliberately kept in one place.
_RATE_LIMIT_MARKERS = (
    "too many requests",
    "rate limit",
    "429",
    "yfratelimit",
)

_NETWORK_MARKERS = (
    "connection",
    "timed out",
    "timeout",
    "temporary failure",
    "name resolution",
    "unreachable",
    "max retries exceeded",
    "ssl",
    "remote end closed",
    "getaddrinfo",
)


class IngestError(RuntimeError):
    """Base class for ingestion failures that must not be swallowed."""


class CircuitBreakerOpen(IngestError):
    """Raised when too many consecutive batches came back rate-limited.

    Grinding through 500 more failures after Yahoo has started refusing us
    produces 500 misleading log rows and no data. Stop instead.
    """


@dataclass
class BatchConfig:
    """Knobs for a fetch run. Defaults are deliberately gentle."""

    batch_size: int = 40
    max_workers: int = 2          # cap concurrency at 2-3, never 8
    max_attempts: int = 5
    base_delay: float = 2.0       # seconds, doubled per attempt
    max_delay: float = 120.0
    jitter: float = 0.4           # fraction of the delay, applied +/-
    circuit_breaker_threshold: int = 3   # consecutive rate-limited batches
    pause_between_batches: float = 1.0

    def __post_init__(self) -> None:
        if self.max_workers > 3:
            raise ValueError(
                f"max_workers={self.max_workers} exceeds the cap of 3. "
                "High concurrency against Yahoo is what got the draft rate limited."
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def workers_label(self) -> str:
        return "sequential" if self.max_workers <= 1 else f"{self.max_workers} workers"


@dataclass
class TickerOutcome:
    """What happened to one ticker in one batch. Always recorded."""

    ticker: str
    status: str
    rows: int = 0
    error_message: Optional[str] = None
    attempt: int = 1
    started_at: Optional[dt.datetime] = None
    finished_at: Optional[dt.datetime] = None


@dataclass
class FetchResult:
    frame: pd.DataFrame
    outcomes: list[TickerOutcome] = field(default_factory=list)
    aborted: bool = False
    abort_reason: Optional[str] = None

    @property
    def ok_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "ok")

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.outcomes:
            counts[o.status] = counts.get(o.status, 0) + 1
        return counts


def classify_error(exc_or_message: BaseException | str) -> str:
    """Map an exception or an error string to an ingest_log status.

    The distinction is the whole point: `rate_limited` means back off and try
    again, `network_error` means the problem is on our side of the wire, and
    neither says anything at all about whether the company still exists.
    """
    if isinstance(exc_or_message, BaseException):
        text = f"{type(exc_or_message).__name__}: {exc_or_message}"
    else:
        text = str(exc_or_message)
    lowered = text.lower()

    if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
        return "rate_limited"
    if any(marker in lowered for marker in _NETWORK_MARKERS):
        return "network_error"
    # "possibly delisted; no price data found" is Yahoo's catch-all and is
    # emitted for network failures too. It is an empty response, nothing more.
    if "no price data" in lowered or "no data found" in lowered:
        return "empty_response"
    return "parse_error"


def backoff_delay(attempt: int, cfg: BatchConfig, rng: Optional[random.Random] = None) -> float:
    """Exponential backoff with symmetric jitter, capped at cfg.max_delay."""
    rng = rng or random
    raw = min(cfg.base_delay * (2 ** max(0, attempt - 1)), cfg.max_delay)
    spread = raw * cfg.jitter
    return max(0.0, raw + rng.uniform(-spread, spread))


def chunked(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# response normalisation
# ---------------------------------------------------------------------------

def _extract_ticker_frame(raw: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame:
    """Pull one ticker's columns out of a yf.download() result."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if ticker in set(level0):
            sub = raw.xs(ticker, axis=1, level=0)
        else:
            # group_by="ticker" puts the symbol on level 0; some yfinance
            # versions invert that. Try the other level before giving up.
            level1 = raw.columns.get_level_values(1)
            if ticker in set(level1):
                sub = raw.xs(ticker, axis=1, level=1)
            elif single:
                sub = raw.copy()
                sub.columns = sub.columns.get_level_values(-1)
            else:
                return pd.DataFrame()
    else:
        if not single:
            return pd.DataFrame()
        sub = raw.copy()

    return sub


def normalise_frame(
    sub: pd.DataFrame,
    ticker: str,
    *,
    ingested_at: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    """Turn one ticker's raw Yahoo columns into store-shaped rows.

    Raw prices only. `Adj Close` is deliberately dropped: adjustment is
    recomputed on read from split_factor and dividend, so that the store always
    retains what the broker screen showed on the day.
    """
    if sub is None or sub.empty:
        return pd.DataFrame(columns=_store_columns())

    out = pd.DataFrame(index=sub.index)
    for source_col, target_col in _FIELD_MAP.items():
        if source_col in sub.columns:
            out[target_col] = pd.to_numeric(sub[source_col], errors="coerce")
        else:
            out[target_col] = pd.NA

    # Drop rows where the whole bar is missing — Yahoo pads non-trading days
    # when several tickers with different calendars are downloaded together.
    price_cols = ["open", "high", "low", "close"]
    out = out.dropna(subset=price_cols, how="all")
    if out.empty:
        return pd.DataFrame(columns=_store_columns())

    # yfinance reports 0.0 for "no split on this day"; the store's neutral
    # value is 1.0, because split_factor is a multiplier.
    split = pd.to_numeric(out["split_factor"], errors="coerce").fillna(0.0)
    out["split_factor"] = split.where(split > 0, 1.0)
    out["dividend"] = pd.to_numeric(out["dividend"], errors="coerce").fillna(0.0)

    volume = pd.to_numeric(out["volume"], errors="coerce")
    out["volume"] = volume.round().astype("Int64")

    index = pd.to_datetime(out.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out.insert(0, "date", index.date)
    out.insert(0, "ticker", ticker)
    out["source"] = SOURCE
    out["ingested_at"] = ingested_at or dt.datetime.now()

    out = out.reset_index(drop=True)
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    return out.loc[:, _store_columns()]


def _store_columns() -> list[str]:
    return [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "split_factor",
        "dividend",
        "source",
        "ingested_at",
    ]


def _yahoo_reported_errors() -> dict[str, str]:
    """Read yfinance's per-ticker error map for the last download.

    This is a private attribute, so it is read defensively: if it moves, we
    lose error detail but never correctness — tickers with no rows still get
    logged as empty_response rather than silently dropped.
    """
    try:
        import yfinance as yf

        return dict(getattr(yf.shared, "_ERRORS", {}) or {})
    except Exception:  # pragma: no cover - defensive only
        return {}


def _reset_yahoo_errors() -> None:
    try:
        import yfinance as yf

        shared = getattr(yf, "shared", None)
        if shared is not None and hasattr(shared, "_ERRORS"):
            shared._ERRORS.clear()
    except Exception:  # pragma: no cover - defensive only
        pass


# ---------------------------------------------------------------------------
# the fetch loop
# ---------------------------------------------------------------------------

def _default_downloader(
    tickers: Sequence[str],
    start: Optional[dt.date],
    end: Optional[dt.date],
    period: Optional[str],
) -> pd.DataFrame:
    """The one place yf.download is called. Note every explicit keyword."""
    import yfinance as yf

    kwargs = dict(
        tickers=list(tickers),
        interval="1d",
        auto_adjust=False,   # EXPLICIT. The library default is True.
        actions=True,        # we need Dividends and Stock Splits as raw columns
        group_by="ticker",
        threads=False,       # concurrency is managed by this module, not yfinance
        progress=False,
        repair=False,
        rounding=False,
    )
    if period is not None:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    return yf.download(**kwargs)


def fetch_batch(
    tickers: Sequence[str],
    *,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    period: Optional[str] = None,
    cfg: Optional[BatchConfig] = None,
    downloader: Optional[Callable[..., pd.DataFrame]] = None,
    sleeper: Callable[[float], None] = time.sleep,
    rng: Optional[random.Random] = None,
) -> FetchResult:
    """Fetch one batch, retrying with backoff on rate limits.

    Returns a FetchResult carrying both the rows and an outcome per ticker.
    Rate limiting that survives max_attempts is reported as `rate_limited` for
    every ticker in the batch — not as missing data.
    """
    cfg = cfg or BatchConfig()
    download = downloader or _default_downloader
    tickers = list(tickers)
    if not tickers:
        return FetchResult(frame=pd.DataFrame(columns=_store_columns()))

    started_at = dt.datetime.now()
    last_error: Optional[str] = None

    for attempt in range(1, cfg.max_attempts + 1):
        _reset_yahoo_errors()
        try:
            raw = download(tickers, start, end, period)
        except BaseException as exc:  # noqa: BLE001 - classified, never swallowed
            status = classify_error(exc)
            last_error = f"{type(exc).__name__}: {exc}"
            if status == "rate_limited" and attempt < cfg.max_attempts:
                sleeper(backoff_delay(attempt, cfg, rng))
                continue
            return FetchResult(
                frame=pd.DataFrame(columns=_store_columns()),
                outcomes=[
                    TickerOutcome(
                        ticker=t,
                        status=status,
                        error_message=last_error,
                        attempt=attempt,
                        started_at=started_at,
                        finished_at=dt.datetime.now(),
                    )
                    for t in tickers
                ],
            )

        reported = _yahoo_reported_errors()
        rate_limited = {
            t for t, msg in reported.items() if classify_error(msg) == "rate_limited"
        }
        # A whole-batch rate limit is worth another attempt; a single bad
        # ticker is not.
        if rate_limited and len(rate_limited) == len(tickers) and attempt < cfg.max_attempts:
            last_error = next(iter(reported.values()), "rate limited")
            sleeper(backoff_delay(attempt, cfg, rng))
            continue

        return _assemble(
            raw=raw,
            tickers=tickers,
            reported=reported,
            attempt=attempt,
            started_at=started_at,
        )

    # Every attempt was rate limited.
    return FetchResult(
        frame=pd.DataFrame(columns=_store_columns()),
        outcomes=[
            TickerOutcome(
                ticker=t,
                status="rate_limited",
                error_message=last_error or "rate limited on every attempt",
                attempt=cfg.max_attempts,
                started_at=started_at,
                finished_at=dt.datetime.now(),
            )
            for t in tickers
        ],
    )


def _assemble(
    *,
    raw: pd.DataFrame,
    tickers: Sequence[str],
    reported: dict[str, str],
    attempt: int,
    started_at: dt.datetime,
) -> FetchResult:
    single = len(tickers) == 1
    ingested_at = dt.datetime.now()
    frames: list[pd.DataFrame] = []
    outcomes: list[TickerOutcome] = []

    for ticker in tickers:
        sub = _extract_ticker_frame(raw, ticker, single)
        try:
            normalised = normalise_frame(sub, ticker, ingested_at=ingested_at)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            outcomes.append(
                TickerOutcome(
                    ticker=ticker,
                    status="parse_error",
                    error_message=f"{type(exc).__name__}: {exc}",
                    attempt=attempt,
                    started_at=started_at,
                    finished_at=dt.datetime.now(),
                )
            )
            continue

        if normalised.empty:
            message = reported.get(ticker)
            # No rows and no reported reason is an empty response. It is NOT
            # evidence of delisting, and nothing downstream may treat it so.
            status = classify_error(message) if message else "empty_response"
            outcomes.append(
                TickerOutcome(
                    ticker=ticker,
                    status=status,
                    error_message=message,
                    attempt=attempt,
                    started_at=started_at,
                    finished_at=dt.datetime.now(),
                )
            )
            continue

        frames.append(normalised)
        outcomes.append(
            TickerOutcome(
                ticker=ticker,
                status="ok",
                rows=len(normalised),
                attempt=attempt,
                started_at=started_at,
                finished_at=dt.datetime.now(),
            )
        )

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=_store_columns())
    )
    return FetchResult(frame=combined, outcomes=outcomes)


def fetch_many(
    tickers: Sequence[str],
    *,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    period: Optional[str] = None,
    cfg: Optional[BatchConfig] = None,
    downloader: Optional[Callable[..., pd.DataFrame]] = None,
    sleeper: Callable[[float], None] = time.sleep,
    rng: Optional[random.Random] = None,
    on_batch: Optional[Callable[[FetchResult], None]] = None,
) -> FetchResult:
    """Fetch every ticker in batches, tripping a circuit breaker if Yahoo stops answering.

    `on_batch` is called after each batch so the caller can persist rows and log
    outcomes incrementally — an aborted run should keep the data it already got.
    """
    cfg = cfg or BatchConfig()
    batches = chunked(list(tickers), cfg.batch_size)
    if not batches:
        return FetchResult(frame=pd.DataFrame(columns=_store_columns()))

    state = _RunState(cfg=cfg)

    def run_one(index: int, batch: list[str]) -> Optional[FetchResult]:
        if state.tripped:
            return None
        result = fetch_batch(
            batch,
            start=start,
            end=end,
            period=period,
            cfg=cfg,
            downloader=downloader,
            sleeper=sleeper,
            rng=rng,
        )
        state.record(batch, result, on_batch)
        if cfg.pause_between_batches and index < len(batches) - 1:
            sleeper(cfg.pause_between_batches)
        return result

    if cfg.max_workers <= 1:
        for i, batch in enumerate(batches):
            run_one(i, batch)
            if state.tripped:
                state.mark_unattempted(batches[i + 1 :])
                break
    else:
        # Bounded concurrency. Batches are submitted up front and the executor
        # queues them; when the breaker trips, everything not yet started is
        # cancelled so we stop rather than grind through the rest of the
        # universe collecting identical failures.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {
                pool.submit(run_one, i, batch): (i, batch)
                for i, batch in enumerate(batches)
            }
            for future in futures:
                if state.tripped:
                    future.cancel()
            for future, (_, batch) in futures.items():
                if future.cancelled():
                    state.mark_unattempted([batch])
                    continue
                try:
                    if future.result() is None:
                        state.mark_unattempted([batch])
                except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                    state.record_exception(batch, exc)

    return state.finish()


class _RunState:
    """Accumulates outcomes across batches, thread-safely.

    Exists because the circuit breaker and the incremental `on_batch` writes
    both need to be serialised when batches run concurrently — a DuckDB
    connection is not safe to write from two threads at once.
    """

    def __init__(self, cfg: BatchConfig) -> None:
        import threading

        self.cfg = cfg
        self.lock = threading.Lock()
        self.outcomes: list[TickerOutcome] = []
        self.frames: list[pd.DataFrame] = []
        self.consecutive_rate_limited = 0
        self.tripped = False
        self.abort_reason: Optional[str] = None

    def record(
        self,
        batch: list[str],
        result: FetchResult,
        on_batch: Optional[Callable[[FetchResult], None]],
    ) -> None:
        with self.lock:
            self.outcomes.extend(result.outcomes)
            if not result.frame.empty:
                self.frames.append(result.frame)
            if on_batch is not None:
                on_batch(result)

            counts = result.status_counts()
            if batch and counts.get("rate_limited", 0) == len(batch):
                self.consecutive_rate_limited += 1
            elif counts.get("ok", 0) > 0:
                self.consecutive_rate_limited = 0

            if (
                not self.tripped
                and self.consecutive_rate_limited >= self.cfg.circuit_breaker_threshold
            ):
                self.tripped = True
                self.abort_reason = (
                    f"circuit breaker: {self.consecutive_rate_limited} consecutive "
                    "rate-limited batches; remaining tickers not attempted"
                )

    def record_exception(self, batch: list[str], exc: BaseException) -> None:
        with self.lock:
            self.outcomes.extend(
                TickerOutcome(
                    ticker=t,
                    status=classify_error(exc),
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                for t in batch
            )

    def mark_unattempted(self, batches: Iterable[list[str]]) -> None:
        reason = self.abort_reason or "not attempted"
        with self.lock:
            already = {o.ticker for o in self.outcomes}
            for batch in batches:
                for ticker in batch:
                    if ticker not in already:
                        self.outcomes.append(
                            TickerOutcome(
                                ticker=ticker,
                                status="circuit_open",
                                error_message=reason,
                            )
                        )

    def finish(self) -> FetchResult:
        combined = (
            pd.concat(self.frames, ignore_index=True)
            if self.frames
            else pd.DataFrame(columns=_store_columns())
        )
        return FetchResult(
            frame=combined,
            outcomes=self.outcomes,
            aborted=self.tripped,
            abort_reason=self.abort_reason,
        )


def outcomes_to_log_entries(
    outcomes: Iterable[TickerOutcome],
    *,
    run_id: str,
) -> list[dict]:
    """Shape fetch outcomes for db.log_ingest. Every outcome becomes a row."""
    return [
        {
            "run_id": run_id,
            "source": SOURCE,
            "ticker": o.ticker,
            "status": o.status,
            "rows_written": o.rows,
            "error_message": o.error_message,
            "attempt": o.attempt,
            "started_at": o.started_at,
            "finished_at": o.finished_at,
        }
        for o in outcomes
    ]
