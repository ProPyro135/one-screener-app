"""DuckDB access layer.

Python is the only writer of the store. The R/Shiny app opens the same file
read-only. Every write path in the project goes through this module, so that
"one source of truth" is enforced by there being exactly one door.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import duckdb
import pandas as pd


class StoreLocked(RuntimeError):
    """Another process holds the store open."""


SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "1"

#: Where the store lives unless overridden by --db or IDXCORE_DB.
DEFAULT_DB_PATH = Path("data") / "idx.duckdb"

INGEST_STATUSES = frozenset({
    "ok",
    "rate_limited",
    "network_error",
    "empty_response",
    "parse_error",
    "circuit_open",
    "run_started",
    "run_finished",
    "run_aborted",
})

PRICE_COLUMNS = [
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

TICKER_COLUMNS = [
    "ticker",
    "idx_code",
    "name",
    "sector",
    "board",
    "listing_date",
    "source",
]


def resolve_db_path(path: Optional[os.PathLike | str] = None) -> Path:
    """Pick the store path: explicit argument, then $IDXCORE_DB, then default."""
    if path is not None:
        return Path(path)
    env = os.environ.get("IDXCORE_DB")
    if env:
        return Path(env)
    return DEFAULT_DB_PATH


def schema_sql() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


def connect(
    path: Optional[os.PathLike | str] = None,
    *,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open the store.

    A read-only connection against a store that does not exist yet is an error
    worth surfacing loudly: it means someone is reading before `init-db` ran,
    and the alternative is an empty result set masquerading as "no signals".
    """
    db_path = resolve_db_path(path)
    if read_only and not db_path.exists():
        raise FileNotFoundError(
            f"No store at {db_path}. Run `idxcore init-db` first — "
            "reading a missing store would look like an empty universe."
        )
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return duckdb.connect(str(db_path), read_only=read_only)
    except duckdb.IOException as exc:
        # DuckDB allows one writer OR many readers, so a running dashboard
        # blocks every write command. The raw error names a PID and nothing
        # else, which is not much help at a prompt.
        if "being used by another process" in str(exc) or "lock" in str(exc).lower():
            raise StoreLocked(
                f"{db_path} is open in another process.\n"
                "The dashboard holds a read lock while it runs, and DuckDB allows "
                "only one writer at a time. Stop the dashboard (Ctrl+C in its "
                "terminal) and try again."
            ) from exc
        raise


@contextmanager
def open_db(
    path: Optional[os.PathLike | str] = None,
    *,
    read_only: bool = False,
) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def table_names(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def declared_columns() -> dict[str, dict[str, str]]:
    """What schema.sql says each table should look like.

    Built by running the schema into a throwaway in-memory database and asking
    DuckDB what it made, rather than by parsing SQL with a regex. The schema
    file stays the single source of truth and nothing has to be kept in sync
    with it by hand.
    """
    scratch = duckdb.connect(":memory:")
    try:
        scratch.execute(schema_sql())
        rows = scratch.execute(
            "SELECT table_name, column_name, data_type "
            "FROM information_schema.columns WHERE table_schema = 'main'"
        ).fetchall()
    finally:
        scratch.close()

    declared: dict[str, dict[str, str]] = {}
    for table, column, dtype in rows:
        declared.setdefault(table, {})[column] = dtype
    return declared


def migrate_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Add columns that schema.sql declares but the live table lacks.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists,
    so adding a column to schema.sql had no effect on any store that had
    already been created — the write then failed with a binder error pointing
    at the column rather than at the cause. Found exactly that way when
    Ichimoku and Donchian were added.

    Additive only. Columns are never dropped, renamed or retyped here: those
    are decisions with data loss attached, and they should not happen as a side
    effect of running init-db.
    """
    added: list[str] = []
    live_tables = set(table_names(con))

    for table, columns in declared_columns().items():
        if table not in live_tables:
            continue
        existing = {
            r[0]
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ?",
                [table],
            ).fetchall()
        }
        for column, dtype in columns.items():
            if column in existing:
                continue
            con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {dtype}')
            added.append(f"{table}.{column}")
    return added


def init_db(path: Optional[os.PathLike | str] = None) -> dict:
    """Create or update the schema. Idempotent: safe against an existing store.

    Every statement in schema.sql is CREATE ... IF NOT EXISTS, so this never
    drops or truncates anything. Re-running it on a populated store leaves the
    data untouched — and any column added to schema.sql since the store was
    created is added to the live table rather than silently ignored.
    """
    db_path = resolve_db_path(path)
    existed = db_path.exists()

    with open_db(db_path) as con:
        before = set(table_names(con))
        con.execute(schema_sql())
        added_columns = migrate_columns(con)
        after = set(table_names(con))
        con.execute(
            """
            INSERT INTO meta (key, value, updated_at) VALUES ('schema_version', ?, ?)
            ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                            updated_at = excluded.updated_at
            """,
            [SCHEMA_VERSION, dt.datetime.now()],
        )
        counts = {t: con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in sorted(after)}

    return {
        "path": str(db_path),
        "created_file": not existed,
        "created_tables": sorted(after - before),
        "added_columns": added_columns,
        "tables": sorted(after),
        "row_counts": counts,
    }


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------

def upsert_prices(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> int:
    """Upsert raw OHLCV rows keyed on (ticker, date).

    Re-running a day updates in place rather than duplicating, which is what
    makes `backfill` and `daily` safe to re-run.
    """
    if frame is None or frame.empty:
        return 0

    missing = [c for c in PRICE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"price frame is missing columns: {missing}")

    payload = frame.loc[:, PRICE_COLUMNS].copy()
    payload = payload.drop_duplicates(subset=["ticker", "date"], keep="last")

    con.register("_incoming_prices", payload)
    try:
        con.execute(
            """
            INSERT INTO prices
                (ticker, date, open, high, low, close, volume,
                 split_factor, dividend, source, ingested_at)
            SELECT ticker, date, open, high, low, close, volume,
                   split_factor, dividend, source, ingested_at
            FROM _incoming_prices
            ON CONFLICT (ticker, date) DO UPDATE SET
                open         = excluded.open,
                high         = excluded.high,
                low          = excluded.low,
                close        = excluded.close,
                volume       = excluded.volume,
                split_factor = excluded.split_factor,
                dividend     = excluded.dividend,
                source       = excluded.source,
                ingested_at  = excluded.ingested_at
            """
        )
    finally:
        con.unregister("_incoming_prices")

    return len(payload)


def last_price_date(con: duckdb.DuckDBPyConnection, ticker: str) -> Optional[dt.date]:
    row = con.execute(
        "SELECT max(date) FROM prices WHERE ticker = ?", [ticker]
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def last_price_dates(con: duckdb.DuckDBPyConnection) -> dict[str, dt.date]:
    rows = con.execute(
        "SELECT ticker, max(date) FROM prices GROUP BY ticker"
    ).fetchall()
    return {t: d for t, d in rows if d is not None}


# ---------------------------------------------------------------------------
# tickers  (survivorship: never delete)
# ---------------------------------------------------------------------------

def upsert_tickers(
    con: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    *,
    seen_on: dt.date,
) -> int:
    """Insert or refresh ticker rows, stamping last_seen.

    A ticker seen again after being marked delisted is reactivated and its
    delisted_at cleared — a suspension that lifts is a real event, and leaving
    the stamp set would misreport the company as dead.
    """
    if frame is None or frame.empty:
        return 0

    missing = [c for c in TICKER_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"ticker frame is missing columns: {missing}")

    payload = frame.loc[:, TICKER_COLUMNS].copy()
    payload = payload.drop_duplicates(subset=["ticker"], keep="last")
    payload["seen_on"] = seen_on
    payload["updated_at"] = dt.datetime.now()

    con.register("_incoming_tickers", payload)
    try:
        con.execute(
            """
            INSERT INTO tickers
                (ticker, idx_code, name, sector, board, listing_date,
                 is_active, first_seen, last_seen, delisted_at, source, updated_at)
            SELECT ticker, idx_code, name, sector, board, listing_date,
                   TRUE, seen_on, seen_on, NULL, source, updated_at
            FROM _incoming_tickers
            ON CONFLICT (ticker) DO UPDATE SET
                idx_code     = excluded.idx_code,
                name         = coalesce(excluded.name, tickers.name),
                sector       = coalesce(excluded.sector, tickers.sector),
                board        = coalesce(excluded.board, tickers.board),
                listing_date = coalesce(excluded.listing_date, tickers.listing_date),
                is_active    = TRUE,
                last_seen    = excluded.last_seen,
                delisted_at  = NULL,
                source       = excluded.source,
                updated_at   = excluded.updated_at
            """
        )
    finally:
        con.unregister("_incoming_tickers")

    return len(payload)


def mark_absent_tickers_inactive(
    con: duckdb.DuckDBPyConnection,
    present: Sequence[str],
    *,
    seen_on: dt.date,
) -> int:
    """Mark tickers that were NOT in this sync as inactive. Never delete.

    Deleting them instead would rebuild the universe from "what exists today",
    which removes exactly the companies that failed and makes every backtest
    that follows optimistic.
    """
    present_frame = pd.DataFrame({"ticker": list(dict.fromkeys(present))})
    con.register("_present_tickers", present_frame)
    try:
        rows = con.execute(
            """
            UPDATE tickers
               SET is_active   = FALSE,
                   delisted_at = coalesce(delisted_at, ?),
                   updated_at  = ?
             WHERE is_active = TRUE
               AND ticker NOT IN (SELECT ticker FROM _present_tickers)
            RETURNING ticker
            """,
            [seen_on, dt.datetime.now()],
        ).fetchall()
    finally:
        con.unregister("_present_tickers")
    return len(rows)


def universe_counts(con: duckdb.DuckDBPyConnection) -> dict:
    total, active = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE is_active) FROM tickers"
    ).fetchone()
    return {"total": total, "active": active, "inactive": total - active}


def active_tickers(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT ticker FROM tickers WHERE is_active ORDER BY ticker"
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# ingest_log  (no silent failures)
# ---------------------------------------------------------------------------

def new_run_id() -> str:
    return f"{dt.datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


def log_ingest(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    source: str,
    status: str,
    ticker: Optional[str] = None,
    rows_written: int = 0,
    error_message: Optional[str] = None,
    attempt: int = 1,
    started_at: Optional[dt.datetime] = None,
    finished_at: Optional[dt.datetime] = None,
) -> None:
    """Write one row to ingest_log.

    `status` is validated here rather than only by the CHECK constraint so a
    typo fails at the call site with a readable message instead of surfacing as
    a constraint violation three frames away.
    """
    if status not in INGEST_STATUSES:
        raise ValueError(
            f"unknown ingest status {status!r}; expected one of {sorted(INGEST_STATUSES)}"
        )
    now = dt.datetime.now()
    con.execute(
        """
        INSERT INTO ingest_log
            (run_id, ticker, source, status, rows_written,
             error_message, attempt, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            ticker,
            source,
            status,
            rows_written,
            error_message,
            attempt,
            started_at or now,
            finished_at if finished_at is not None else now,
        ],
    )


def log_many(con: duckdb.DuckDBPyConnection, entries: Iterable[dict]) -> int:
    n = 0
    for entry in entries:
        log_ingest(con, **entry)
        n += 1
    return n


def run_status_counts(con: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, int]:
    rows = con.execute(
        "SELECT status, count(*) FROM ingest_log WHERE run_id = ? GROUP BY status",
        [run_id],
    ).fetchall()
    return {status: count for status, count in rows}


def latest_run_id(con: duckdb.DuckDBPyConnection, source: Optional[str] = None) -> Optional[str]:
    if source is None:
        row = con.execute(
            "SELECT run_id FROM ingest_log ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    else:
        row = con.execute(
            "SELECT run_id FROM ingest_log WHERE source = ? ORDER BY started_at DESC LIMIT 1",
            [source],
        ).fetchone()
    return row[0] if row else None
