"""Market benchmarks (IHSG and friends).

Kept deliberately separate from `prices` and `tickers`. IHSG is not a stock:
if it lived in the universe it would appear in every screen, be entered by the
backtest as though it were tradeable, and be folded into the "average stock"
baseline — quietly corrupting the exact comparison it exists to provide.

IVI-81 needs a buy-and-hold benchmark. This is where it comes from.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import duckdb
import pandas as pd

from . import yahoo

#: Indeks Harga Saham Gabungan — the IDX composite, Yahoo's symbol for it.
IHSG = "^JKSE"

KNOWN_INDICES = {
    IHSG: "IHSG (Jakarta Composite Index)",
    "^JKLQ45": "LQ45",
}

INDEX_COLUMNS = [
    "symbol",
    "name",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "ingested_at",
]


def fetch_index(
    symbol: str = IHSG,
    *,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    downloader=None,
) -> pd.DataFrame:
    """Fetch one index through the same guarded path as everything else.

    Reuses the Yahoo worker so an index fetch is subject to the identical
    backoff, error classification and no-silent-failure rules. A benchmark that
    quietly came back empty would make the strategy look good by comparison.
    """
    result = yahoo.fetch_batch(
        [symbol],
        start=start,
        end=end,
        cfg=yahoo.BatchConfig(batch_size=1, max_workers=1),
        downloader=downloader,
    )
    statuses = {o.status for o in result.outcomes}
    if "ok" not in statuses:
        detail = "; ".join(
            f"{o.status}: {o.error_message or 'no detail'}" for o in result.outcomes
        )
        raise yahoo.IngestError(f"index {symbol} did not return data — {detail}")

    frame = result.frame.copy()
    frame = frame.rename(columns={"ticker": "symbol"})
    frame["name"] = KNOWN_INDICES.get(symbol, symbol)
    return frame.loc[:, INDEX_COLUMNS]


def upsert_indices(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> int:
    if frame is None or frame.empty:
        return 0
    payload = frame.loc[:, INDEX_COLUMNS].drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    con.register("_incoming_indices", payload)
    try:
        con.execute(
            """
            INSERT INTO indices
                (symbol, name, date, open, high, low, close, volume, source, ingested_at)
            SELECT symbol, name, date, open, high, low, close, volume, source, ingested_at
            FROM _incoming_indices
            ON CONFLICT (symbol, date) DO UPDATE SET
                name        = excluded.name,
                open        = excluded.open,
                high        = excluded.high,
                low         = excluded.low,
                close       = excluded.close,
                volume      = excluded.volume,
                source      = excluded.source,
                ingested_at = excluded.ingested_at
            """
        )
    finally:
        con.unregister("_incoming_indices")
    return len(payload)


def index_returns(
    con: duckdb.DuckDBPyConnection,
    *,
    symbol: str = IHSG,
    horizon_bars: int = 20,
) -> pd.DataFrame:
    """Forward return of the index over the same horizon, from every bar.

    Built to mirror the trade convention exactly: enter at the *next* bar's
    open, exit at the close `horizon_bars` later. Comparing a
    close-to-close index return against an open-to-close trade return would
    quietly favour one side.
    """
    return con.execute(
        """
        WITH b AS (
          SELECT date, open, close,
                 row_number() OVER (ORDER BY date) AS rn
            FROM indices WHERE symbol = ?
        )
        SELECT b.date,
               f.open  AS fill,
               x.close AS exit_px,
               (x.close / f.open - 1) * 100 AS forward_pct
          FROM b
          JOIN b f ON f.rn = b.rn + 1
          JOIN b x ON x.rn = b.rn + ?
         WHERE f.open > 0
         ORDER BY b.date
        """,
        [symbol, horizon_bars],
    ).df()


def coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT symbol, name, count(*) AS bars, min(date) AS first, max(date) AS last
          FROM indices GROUP BY 1, 2 ORDER BY 1
        """
    ).df()
