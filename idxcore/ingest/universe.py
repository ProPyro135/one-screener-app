"""Universe sync with survivorship tracking.

The rule this module exists to enforce: **never delete a ticker.** A stock that
delists or is suspended gets ``is_active = false`` and a ``delisted_at`` stamp.
It keeps its price history and it keeps appearing in backtests.

Rebuilding the universe from "what exists today" quietly removes exactly the
companies that failed, which makes every backtest that follows optimistic. That
bias is invisible in the output — the numbers just look better — which is why
the protection has to live in code rather than in a habit.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from . import idx_web
from ..store import db


class UniverseShrinkGuard(RuntimeError):
    """The incoming universe was much smaller than the stored one.

    Almost always a broken source rather than a mass delisting event. Marking
    the difference as delisted would corrupt the universe in a way that is hard
    to notice and harder to undo, so the run stops and asks.
    """


@dataclass
class SyncSummary:
    source: str
    fetched: int = 0
    upserted: int = 0
    newly_inactive: int = 0
    reactivated: int = 0
    total: int = 0
    active: int = 0
    inactive: int = 0
    skipped_deactivation: bool = False
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"universe sync ({self.source})",
            f"  fetched from source : {self.fetched}",
            f"  upserted            : {self.upserted}",
            f"  reactivated         : {self.reactivated}",
            f"  newly inactive      : {self.newly_inactive}",
            f"  ---",
            f"  tickers total       : {self.total}",
            f"  active              : {self.active}",
            f"  inactive (delisted) : {self.inactive}",
        ]
        if self.skipped_deactivation:
            lines.append("  NOTE: deactivation skipped (see notes)")
        lines.extend(f"  ! {n}" for n in self.notes)
        return "\n".join(lines)


def load_universe_frame(
    *,
    from_file: Optional[str | Path] = None,
    allow_web: bool = True,
) -> pd.DataFrame:
    """Resolve the incoming universe from the best available source.

    A file, when given, wins: it is the authoritative IDX export and it does
    not depend on the website answering an automated client.
    """
    if from_file is not None:
        return idx_web.load_daftar_saham(from_file)
    if not allow_web:
        raise ValueError("no --from-file given and the web source is disabled")
    return idx_web.fetch_securities()


def _currently_active(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("SELECT ticker FROM tickers WHERE is_active").fetchall()
    return {r[0] for r in rows}


def _reactivation_count(con: duckdb.DuckDBPyConnection, incoming: list[str]) -> int:
    if not incoming:
        return 0
    frame = pd.DataFrame({"ticker": incoming})
    con.register("_incoming_check", frame)
    try:
        row = con.execute(
            """
            SELECT count(*) FROM tickers
             WHERE NOT is_active
               AND ticker IN (SELECT ticker FROM _incoming_check)
            """
        ).fetchone()
    finally:
        con.unregister("_incoming_check")
    return row[0] if row else 0


def sync_universe(
    con: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    *,
    source: str,
    seen_on: Optional[dt.date] = None,
    shrink_threshold: float = 0.10,
    allow_shrink: bool = False,
    deactivate_missing: bool = True,
    run_id: Optional[str] = None,
) -> SyncSummary:
    """Apply an incoming universe to `tickers`. Idempotent.

    Running this twice with the same input is a no-op the second time. Running
    it with a ticker missing marks that ticker inactive; running it again with
    the ticker back reactivates it.
    """
    seen_on = seen_on or dt.date.today()
    run_id = run_id or db.new_run_id()
    summary = SyncSummary(source=source)

    if frame is None or frame.empty:
        raise UniverseShrinkGuard(
            "the source returned zero tickers. Refusing to mark the entire "
            "universe delisted on the strength of an empty response."
        )

    incoming = frame["ticker"].astype(str).tolist()
    summary.fetched = len(incoming)

    active_before = _currently_active(con)
    summary.reactivated = _reactivation_count(con, incoming)

    started_at = dt.datetime.now()
    summary.upserted = db.upsert_tickers(con, frame, seen_on=seen_on)

    # ---- the guard -------------------------------------------------------
    # A source that half-works is more dangerous than one that fails, because
    # its output looks like data. Compare against what we already had.
    missing = active_before - set(incoming)
    shrink_ratio = (len(missing) / len(active_before)) if active_before else 0.0

    if deactivate_missing and missing and shrink_ratio > shrink_threshold and not allow_shrink:
        summary.skipped_deactivation = True
        summary.notes.append(
            f"{len(missing)} of {len(active_before)} active tickers "
            f"({shrink_ratio:.1%}) were absent from this sync, above the "
            f"{shrink_threshold:.0%} threshold. Nothing was marked delisted. "
            "Verify the source, then re-run with --allow-shrink if the "
            "disappearance is real."
        )
        db.log_ingest(
            con,
            run_id=run_id,
            source=source,
            status="parse_error",
            error_message=summary.notes[-1],
            started_at=started_at,
        )
    elif deactivate_missing:
        summary.newly_inactive = db.mark_absent_tickers_inactive(
            con, incoming, seen_on=seen_on
        )

    counts = db.universe_counts(con)
    summary.total = counts["total"]
    summary.active = counts["active"]
    summary.inactive = counts["inactive"]

    db.log_ingest(
        con,
        run_id=run_id,
        source=source,
        status="ok",
        rows_written=summary.upserted,
        started_at=started_at,
    )
    return summary
