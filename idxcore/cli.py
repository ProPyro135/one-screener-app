"""idxcore command line.

    idxcore init-db          create schema (idempotent, never wipes)
    idxcore sync-universe    refresh ticker list, mark delistings
    idxcore backfill         historical OHLCV for the active universe
    idxcore daily            delta fetch -> indicators -> signals
    idxcore compute          recompute indicators/signals only
    idxcore backtest         replay + label + calibration report
    idxcore walkforward      out-of-sample validation of the rule choice
    idxcore verify           cross-check prices against a second source

Exit codes are meaningful, because a scheduled task can only see the exit code:

    0  everything the command attempted succeeded
    1  the run completed but too much of it failed (see ingest_log)
    2  the run was aborted, or the command is not implemented yet

A run in which every ticker failed always exits non-zero. Silence is not success.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Optional, Sequence

from .store import db

EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_ABORTED = 2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def _resolve_tickers(con, explicit: Optional[str], limit: Optional[int]) -> list[str]:
    if explicit:
        tickers = [t.strip().upper() for t in explicit.split(",") if t.strip()]
    else:
        tickers = db.active_tickers(con)
    if limit is not None:
        tickers = tickers[:limit]
    return tickers


def _summarise_run(counts: dict[str, int], rows_written: int) -> None:
    total = sum(counts.values())
    print()
    _print_header("ingest summary")
    print(f"  tickers attempted : {total}")
    for status in sorted(counts):
        print(f"  {status:<18}: {counts[status]}")
    print(f"  rows written      : {rows_written}")


def _failure_rate(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - (counts.get("ok", 0) / total)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_init_db(args: argparse.Namespace) -> int:
    result = db.init_db(args.db)
    _print_header("init-db")
    print(f"  store        : {result['path']}")
    print(f"  file created : {result['created_file']}")
    if result["created_tables"]:
        print(f"  new tables   : {', '.join(result['created_tables'])}")
    else:
        print("  new tables   : none (schema already present, nothing wiped)")
    if result.get("added_columns"):
        print(f"  new columns  : {', '.join(result['added_columns'])}")
    print()
    print("  table                rows")
    for table, n in result["row_counts"].items():
        print(f"  {table:<20} {n}")
    return EXIT_OK


def cmd_sync_universe(args: argparse.Namespace) -> int:
    from .ingest import idx_web, universe

    _print_header("sync-universe")

    try:
        frame = universe.load_universe_frame(
            from_file=args.from_file,
            allow_web=not args.no_web,
        )
    except idx_web.SourceBlockedError as exc:
        print(f"  source blocked: {exc}", file=sys.stderr)
        return EXIT_ABORTED
    except (idx_web.SourceFormatError, FileNotFoundError, ValueError) as exc:
        print(f"  source unusable: {exc}", file=sys.stderr)
        return EXIT_ABORTED

    source = str(frame["source"].iloc[0]) if "source" in frame.columns else "unknown"

    with db.open_db(args.db) as con:
        try:
            summary = universe.sync_universe(
                con,
                frame,
                source=source,
                shrink_threshold=args.shrink_threshold,
                allow_shrink=args.allow_shrink,
                deactivate_missing=not args.no_deactivate,
            )
        except universe.UniverseShrinkGuard as exc:
            print(f"  refused: {exc}", file=sys.stderr)
            return EXIT_ABORTED

    print(summary.render())
    if args.write_seed:
        path = idx_web.write_seed_csv(frame, args.write_seed)
        print(f"  seed written : {path}")
    return EXIT_PARTIAL_FAILURE if summary.skipped_deactivation else EXIT_OK


def _run_fetch(args: argparse.Namespace, *, mode: str) -> int:
    """Shared body for backfill and daily."""
    from .ingest import yahoo

    _print_header(mode)

    cfg = yahoo.BatchConfig(
        batch_size=args.batch_size,
        max_workers=args.workers,
        max_attempts=args.max_attempts,
        circuit_breaker_threshold=args.circuit_breaker,
    )

    with db.open_db(args.db) as con:
        tickers = _resolve_tickers(con, args.tickers, args.limit)
        if not tickers:
            print(
                "  no tickers to fetch. Run `idxcore sync-universe` first — "
                "an empty universe is not an empty market.",
                file=sys.stderr,
            )
            return EXIT_ABORTED

        run_id = db.new_run_id()
        started_at = dt.datetime.now()
        db.log_ingest(
            con,
            run_id=run_id,
            source=yahoo.SOURCE,
            status="run_started",
            error_message=f"{mode}: {len(tickers)} tickers",
            started_at=started_at,
        )

        if mode == "backfill":
            start = args.start or (dt.date.today() - dt.timedelta(days=365 * args.years + 10))
            end = args.end
        else:
            known = db.last_price_dates(con)
            if known:
                earliest = min(known[t] for t in tickers if t in known) if any(
                    t in known for t in tickers
                ) else None
            else:
                earliest = None
            # Re-fetch a short overlap so a corrected bar replaces the stored one.
            start = (
                earliest - dt.timedelta(days=args.overlap_days)
                if earliest
                else dt.date.today() - dt.timedelta(days=args.overlap_days)
            )
            end = args.end

        print(f"  tickers : {len(tickers)}")
        print(f"  window  : {start} -> {end or 'today'}")
        n_batches = len(yahoo.chunked(tickers, cfg.batch_size))
        print(f"  batches : {n_batches} x {cfg.batch_size}, {cfg.workers_label()}")
        print(f"  run_id  : {run_id}")
        print()

        rows_written = 0

        def persist(result: yahoo.FetchResult) -> None:
            nonlocal rows_written
            if not result.frame.empty:
                rows_written += db.upsert_prices(con, result.frame)
            db.log_many(
                con,
                yahoo.outcomes_to_log_entries(result.outcomes, run_id=run_id),
            )
            counts = result.status_counts()
            rendered = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  batch: {rendered}")

        result = yahoo.fetch_many(
            tickers,
            start=start,
            end=end,
            cfg=cfg,
            on_batch=persist,
        )

        counts = result.status_counts()
        _summarise_run(counts, rows_written)

        if result.aborted:
            db.log_ingest(
                con,
                run_id=run_id,
                source=yahoo.SOURCE,
                status="run_aborted",
                error_message=result.abort_reason,
                rows_written=rows_written,
                started_at=started_at,
            )
            print(f"\n  ABORTED: {result.abort_reason}", file=sys.stderr)
            return EXIT_ABORTED

        db.log_ingest(
            con,
            run_id=run_id,
            source=yahoo.SOURCE,
            status="run_finished",
            rows_written=rows_written,
            started_at=started_at,
        )

        rate = _failure_rate(counts)
        if rate > args.max_failure_rate:
            print(
                f"\n  FAILED: {rate:.1%} of tickers did not return data, above "
                f"the {args.max_failure_rate:.0%} threshold. "
                f"Inspect: SELECT status, count(*) FROM ingest_log "
                f"WHERE run_id = '{run_id}' GROUP BY status;",
                file=sys.stderr,
            )
            return EXIT_PARTIAL_FAILURE

    return EXIT_OK


def cmd_backfill(args: argparse.Namespace) -> int:
    return _run_fetch(args, mode="backfill")


def cmd_daily(args: argparse.Namespace) -> int:
    """Fetch the delta, then recompute everything that depends on it.

    One command, because a scheduled run that fetched prices but left the
    indicators behind would show yesterday's signals against today's prices —
    stale in a way nothing on screen would reveal.
    """
    rc = _run_fetch(args, mode="daily")
    if rc != EXIT_OK:
        print(
            "\n  fetch did not succeed, so indicators and signals were NOT "
            "recomputed. The store keeps the previous run's derived tables, "
            "which is the honest state to leave it in.",
            file=sys.stderr,
        )
        return rc

    print()
    compute_rc = _run_compute(
        db_path=args.db,
        tickers=args.tickers,
        limit=args.limit,
        show_distribution=False,
    )
    return compute_rc if compute_rc != EXIT_OK else rc


def _not_implemented(name: str, issue: str) -> int:
    print(
        f"`idxcore {name}` is not implemented yet. Tracked in {issue}.\n"
        "It is listed here so the command surface is stable, not because it works.",
        file=sys.stderr,
    )
    return EXIT_ABORTED


def _run_compute(
    *,
    db_path,
    tickers: Optional[str] = None,
    limit: Optional[int] = None,
    skip_quality: bool = False,
    show_distribution: bool = True,
) -> int:
    """Quality gates -> indicators -> signals, in that order.

    The order is not arbitrary. A bad bar looks like a genuine move, and one
    such bar sits inside every moving average that spans it, so the flags are
    computed before anything reads the prices.

    Shared by `compute` and `daily` so a scheduled run cannot drift from a
    manual one.
    """
    from .compute import indicators as ind
    from .compute import quality as qc
    from .compute import signals as sig

    _print_header("compute")

    with db.open_db(db_path) as con:
        resolved = _resolve_tickers(con, tickers, limit)
        stored = {
            r[0]
            for r in con.execute("SELECT DISTINCT ticker FROM prices").fetchall()
        }
        resolved = [t for t in resolved if t in stored]
        if not resolved:
            print(
                "  no stored prices to compute from. Run `idxcore backfill` first.",
                file=sys.stderr,
            )
            return EXIT_ABORTED

        print(f"  tickers : {len(resolved)}")

        if not skip_quality:
            written = qc.upsert_quality(con, qc.flag_all(con, tickers=resolved))
            print(f"  quality    : {written} rows flagged")

        n_ind = ind.upsert_indicators(con, ind.compute_all(con, tickers=resolved))
        print(f"  indicators : {n_ind} rows")

        n_sig = sig.upsert_signals(con, sig.compute_all(con, tickers=resolved))
        print(f"  signals    : {n_sig} rows")

        if not skip_quality:
            print()
            print(qc.summarise(con).render())

        if show_distribution:
            print()
            _print_header("signal distribution (all history)")
            for depth, gate, n in con.execute(
                """
                SELECT alignment_depth,
                       coalesce(rsi_gate, FALSE) AS gate,
                       count(*) AS n
                  FROM signals
                 GROUP BY 1, 2
                 ORDER BY 1 DESC, 2 DESC
                """
            ).fetchall():
                print(f"  depth {depth}  rsi_gate={str(bool(gate)):<5}  {n}")

    return EXIT_OK


def cmd_compute(args: argparse.Namespace) -> int:
    return _run_compute(
        db_path=args.db,
        tickers=args.tickers,
        limit=args.limit,
        skip_quality=args.skip_quality,
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    """Replay the stored signals and label what happened next.

    Requires an explicit outcome rule and, to report anything, a verified cost
    model. Neither is defaulted: the outcome definition is the owner's call
    (IVI-79), and a backtest without costs is an advertisement rather than a
    measurement.
    """
    from .backtest import costs as cost_mod
    from .backtest import label as lab
    from .backtest import replay as rep

    _print_header("backtest")

    if args.write_example_config:
        path = cost_mod.write_example(args.write_example_config)
        print(f"  wrote {path}")
        print("  Fill in the real broker fee rates and the current IDX fraksi")
        print("  harga bands, verified against published IDX rules, then set")
        print('  "verified": true and pass the file with --costs.')
        return EXIT_OK

    if args.horizon is None:
        print(
            "  missing --horizon.\n\n"
            "  The outcome definition is a deliberate choice, not something this\n"
            "  command should pick for you (IVI-79). The rule decided for this\n"
            "  project is hold-to-horizon:\n\n"
            "    idxcore backtest --hold --horizon 20 --costs config/costs.json\n\n"
            "  Or state a target and stop explicitly:\n\n"
            "    idxcore backtest --target 8 --stop 4 --horizon 20 --costs config/costs.json",
            file=sys.stderr,
        )
        return EXIT_ABORTED

    if not args.hold and args.target is None and args.stop is None:
        print(
            "  no exit rule given. Pass --hold to ride to the horizon, or set\n"
            "  --target and/or --stop. Omitting them by accident would silently\n"
            "  become a hold rule, which is a different measurement entirely.",
            file=sys.stderr,
        )
        return EXIT_ABORTED

    if args.hold and (args.target is not None or args.stop is not None):
        print(
            "  --hold means no target and no stop; do not pass them together.",
            file=sys.stderr,
        )
        return EXIT_ABORTED

    try:
        rule = lab.OutcomeRule(
            horizon_bars=args.horizon,
            target_pct=args.target,
            stop_pct=args.stop,
            tie_break=args.tie_break,
            label=args.label or "",
        )
    except ValueError as exc:
        print(f"  invalid outcome rule: {exc}", file=sys.stderr)
        return EXIT_ABORTED

    model = None
    if args.costs and args.assume_fees:
        try:
            buy_s, sell_s = args.assume_fees.split(",")
            model = cost_mod.from_file_with_assumed_fees(
                args.costs, float(buy_s), float(sell_s)
            )
        except ValueError as exc:
            print(
                f"  --assume-fees must be BUY,SELL as fractions e.g. 0.0015,0.0025 ({exc})",
                file=sys.stderr,
            )
            return EXIT_ABORTED
        except cost_mod.CostConfigInvalid as exc:
            print(f"  cost model unusable: {exc}", file=sys.stderr)
            return EXIT_ABORTED
        print(
            "  NOTE: fees are ASSUMED, not from your broker. Every return below\n"
            "  inherits that assumption. The tick grid is the verified IDX one.\n"
        )
    elif args.costs:
        try:
            model = cost_mod.CostModel.from_file(args.costs)
        except (cost_mod.CostModelNotConfigured, cost_mod.CostConfigInvalid) as exc:
            print(f"  cost model unusable: {exc}", file=sys.stderr)
            return EXIT_ABORTED
    elif not args.dry_run:
        print(
            "  refusing to run without a cost model.\n\n"
            "  Broker fees and the IDX tick grid are large relative to the moves\n"
            "  this strategy trades, so results without them would overstate the\n"
            "  edge. Two ways forward:\n\n"
            "    idxcore backtest ... --costs config/costs.json\n"
            "    idxcore backtest --write-example-config config/costs.json\n\n"
            "  Or pass --dry-run to exercise the replay without producing any\n"
            "  outcome or return figures.",
            file=sys.stderr,
        )
        return EXIT_ABORTED

    print(rule.describe())
    print()
    if model is not None:
        print(model.describe())
    else:
        print("costs: NONE (dry run - no outcomes or returns will be reported)")
    print()

    # A dry run writes nothing, so it takes a read lock and can run while the
    # dashboard is open.
    with db.open_db(args.db, read_only=args.dry_run) as con:
        if con.execute("SELECT count(*) FROM signals").fetchone()[0] == 0:
            print("  no signals stored. Run `idxcore compute` first.", file=sys.stderr)
            return EXIT_ABORTED

        tickers = None
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

        frame, summary = rep.replay(
            con,
            rule,
            costs=model,
            min_depth=args.min_depth,
            require_gate=not args.no_rsi_gate,
            require_above_ma5=not args.no_ma5_filter,
            tickers=tickers,
        )

        if args.dry_run:
            print("replay (dry run)")
            print(f"  rule                 : {summary.rule_version}")
            print(f"  tickers              : {summary.tickers}")
            print(f"  signals seen         : {summary.signals_considered}")
            print(f"  fills possible       : {summary.trades_labelled}")
            print(f"  skipped (no next bar): {summary.skipped_no_next_bar}")
            print(f"  entry range          : {summary.first_entry} -> {summary.last_entry}")
            print(f"  runtime              : {summary.seconds:.1f}s")
            print()
            print(
                "  No outcome counts or returns are shown: without a cost model\n"
                "  they would not be a measurement. Nothing was written."
            )
            return EXIT_OK

        written = lab.upsert_labels(con, frame)
        print(summary.render())
        print(f"  labels written  : {written}")

        if summary.seconds > 180:
            print(
                f"\n  NOTE: replay took {summary.seconds:.0f}s. Worth optimising "
                "before the full universe lands."
            )

        _print_tier_outcomes(con, rule.version)

    return EXIT_OK


def _print_tier_outcomes(con, rule_version: str) -> None:
    rows = con.execute(
        """
        SELECT s.tier,
               count(*)                                     AS trades,
               count(*) FILTER (WHERE l.outcome = 'target') AS targets,
               avg(l.return_pct)                            AS mean_net,
               median(l.return_pct)                         AS median_net,
               avg(l.mae)                                   AS mean_mae
          FROM labels l
          JOIN signals s ON s.ticker = l.ticker AND s.date = l.entry_date
         WHERE l.rule_version = ? AND l.outcome IN ('target', 'stop', 'timeout')
         GROUP BY s.tier
         ORDER BY s.tier
        """,
        [rule_version],
    ).fetchall()

    if not rows:
        print("\n  no resolved trades yet - nothing to summarise.")
        return

    print()
    _print_header("outcomes by tier")
    print(f"  {'tier':<6}{'n':>7}{'hit%':>8}{'mean%':>9}{'median%':>10}{'meanMAE':>10}")
    for tier, n, targets, mean_net, median_net, mean_mae in rows:
        hit = (targets / n * 100) if n else 0.0
        small = "   (too few to conclude)" if n < 30 else ""
        print(
            f"  {tier or '-':<6}{n:>7}{hit:>8.1f}"
            f"{(mean_net or 0):>9.2f}{(median_net or 0):>10.2f}"
            f"{(mean_mae or 0):>10.2f}{small}"
        )

    print()
    print(
        "  These are raw per-tier outcomes, not the calibration report.\n"
        "  IVI-81 adds the baselines that make them mean anything: buy-and-hold\n"
        "  IHSG, random entry over the same holding period, and the universe\n"
        "  average. Until those exist a hit rate is not evidence of an edge, and\n"
        "  none of these numbers may be shown in the dashboard."
    )


def cmd_report(args: argparse.Namespace) -> int:
    """Generate the calibration report (IVI-81) from stored labels."""
    from .backtest import calibration as cal
    from .backtest import costs as cost_mod
    from .backtest import label as lab

    _print_header("report")

    if args.horizon is None:
        print("  --horizon is required, and must match the backtest run.",
              file=sys.stderr)
        return EXIT_ABORTED

    rule = lab.OutcomeRule(
        horizon_bars=args.horizon,
        target_pct=args.target,
        stop_pct=args.stop,
        label=args.label or "",
    )

    buy_fee = sell_fee = 0.0
    cost_line = "costs: none applied"
    if args.costs and args.assume_fees:
        buy_s, sell_s = args.assume_fees.split(",")
        model = cost_mod.from_file_with_assumed_fees(
            args.costs, float(buy_s), float(sell_s)
        )
        buy_fee, sell_fee = model.buy_fee, model.sell_fee
        cost_line = model.describe()
    elif args.costs:
        model = cost_mod.CostModel.from_file(args.costs)
        buy_fee, sell_fee = model.buy_fee, model.sell_fee
        cost_line = model.describe()

    ctx = cal.ReportContext(
        rule_version=rule.version,
        rule_description=rule.describe(),
        cost_description=cost_line,
        horizon_bars=args.horizon,
        generated_at=dt.datetime.now(),
    )

    with db.open_db(args.db, read_only=True) as con:
        try:
            markdown = cal.build(con, ctx, buy_fee=buy_fee, sell_fee=sell_fee)
        except ValueError as exc:
            print(f"  {exc}", file=sys.stderr)
            return EXIT_ABORTED

    path = cal.write(args.out, markdown)
    print(f"  rule    : {rule.version}")
    print(f"  written : {path} ({path.stat().st_size / 1024:.0f} KB)")
    print()
    print("  Nothing in this report may be shown in the dashboard until you have")
    print("  read it and decided the numbers are fit to display.")
    return EXIT_OK


def _resolve_cost_model(args) -> tuple[object, str, int]:
    """Shared by backtest and walkforward: a model, a description, an exit code.

    Returns ``(None, message, EXIT_ABORTED)`` when the run must not proceed, so
    a caller cannot forget to check and quietly continue without costs.
    """
    from .backtest import costs as cost_mod

    if args.costs and args.assume_fees:
        try:
            buy_s, sell_s = args.assume_fees.split(",")
            model = cost_mod.from_file_with_assumed_fees(
                args.costs, float(buy_s), float(sell_s)
            )
        except ValueError as exc:
            return None, (
                "--assume-fees must be BUY,SELL as fractions e.g. "
                f"0.0015,0.0025 ({exc})"
            ), EXIT_ABORTED
        except cost_mod.CostConfigInvalid as exc:
            return None, f"cost model unusable: {exc}", EXIT_ABORTED
        return model, model.describe(), EXIT_OK

    if args.costs:
        try:
            model = cost_mod.CostModel.from_file(args.costs)
        except (cost_mod.CostModelNotConfigured, cost_mod.CostConfigInvalid) as exc:
            return None, f"cost model unusable: {exc}", EXIT_ABORTED
        return model, model.describe(), EXIT_OK

    return None, (
        "refusing to run without a cost model.\n\n"
        "  Walk-forward compares configurations against each other, and the cost\n"
        "  drag differs between them - a 10-day rule pays the round trip twice as\n"
        "  often as a 20-day one. Without costs the comparison would favour\n"
        "  whichever rule trades most, which is not a finding.\n\n"
        "    idxcore walkforward --costs config/costs.json"
    ), EXIT_ABORTED


def cmd_walkforward(args: argparse.Namespace) -> int:
    """Out-of-sample validation: choose on the past, measure on the future.

    The calibration report says plainly that nothing in it is out-of-sample.
    This is the command that answers that, and it is a separate command rather
    than a section of the report because it writes labels for rules nobody
    intends to trade - the losing candidates are the whole point.
    """
    from .backtest import label as lab
    from .backtest import replay as rep
    from .backtest import walkforward as wf

    _print_header("walkforward")

    model, cost_line, rc = _resolve_cost_model(args)
    if rc != EXIT_OK:
        print(f"  {cost_line}", file=sys.stderr)
        return rc
    if args.assume_fees:
        print(
            "  NOTE: fees are ASSUMED, not from your broker. Every return below\n"
            "  inherits that assumption. The tick grid is the verified IDX one.\n"
        )

    try:
        horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    except ValueError:
        print("  --horizons must be a comma-separated list of integers.",
              file=sys.stderr)
        return EXIT_ABORTED

    fixed_tiers = tuple(
        t.strip().upper() for t in args.fixed_tiers.split(",") if t.strip()
    )
    try:
        grid = wf.default_grid(horizons)
        fixed = wf.Candidate(
            rule=lab.OutcomeRule(horizon_bars=args.fixed_horizon), tiers=fixed_tiers
        )
    except ValueError as exc:
        print(f"  invalid configuration: {exc}", file=sys.stderr)
        return EXIT_ABORTED

    rules = {c.rule.version: c.rule for c in grid}
    rules.setdefault(fixed.rule.version, fixed.rule)

    print(f"  grid       : {len(grid)} configurations over {len(rules)} exit rules")
    print(f"  choosing by: {args.select_by} net return on the training window")
    print(f"  fixed rule : {fixed.name}")
    print()
    print(cost_line)
    print()

    # Two-phase open: replaying a missing rule needs the write lock, but a
    # re-run with everything already labelled should not evict the dashboard.
    with db.open_db(args.db, read_only=True) as con:
        if con.execute("SELECT count(*) FROM signals").fetchone()[0] == 0:
            print("  no signals stored. Run `idxcore compute` first.", file=sys.stderr)
            return EXIT_ABORTED
        have = wf.stored_rule_versions(con)
    missing = [rule for version, rule in rules.items() if version not in have]

    if missing:
        print(f"  {len(missing)} of {len(rules)} exit rules have no labels yet; "
              "replaying them in one pass.")
        for rule in missing:
            print(f"    {rule.version}")
        print()

        def tick(done: int, total: int, ticker: str) -> None:
            if done % 100 == 0 or done == total:
                print(f"    {done}/{total} tickers", flush=True)

        with db.open_db(args.db) as con:
            results = rep.replay_multi(
                con,
                missing,
                costs=model,
                min_depth=args.min_depth,
                require_gate=not args.no_rsi_gate,
                require_above_ma5=not args.no_ma5_filter,
                progress=None if args.quiet else tick,
            )
            for version, (frame, summary) in results.items():
                written = lab.upsert_labels(con, frame)
                print(f"    {version}: {written} labels ({summary.seconds:.0f}s total)")
        print()

    with db.open_db(args.db, read_only=True) as con:
        labels_by_rule = {v: wf.load_labels(con, v) for v in rules}
        empty = [v for v, frame in labels_by_rule.items() if frame.empty]
        if empty:
            print(f"  no resolved trades for {empty[0]}. Nothing to validate.",
                  file=sys.stderr)
            return EXIT_ABORTED

        first = min(frame["entry_date"].min() for frame in labels_by_rule.values())
        last = max(frame["entry_date"].max() for frame in labels_by_rule.values())
        try:
            folds = wf.make_folds(
                first, last,
                test_months=args.test_months,
                min_train_months=args.min_train_months,
                train_months=args.train_months,
            )
        except ValueError as exc:
            print(f"  cannot build folds: {exc}", file=sys.stderr)
            return EXIT_ABORTED

        if not folds:
            print(
                f"  the labelled history runs {first} to {last}, which is not long\n"
                f"  enough for a {args.min_train_months}-month training window plus\n"
                f"  a {args.test_months}-month test window. Nothing was measured.",
                file=sys.stderr,
            )
            return EXIT_ABORTED

        window = "anchored" if args.train_months is None else f"rolling {args.train_months}m"
        print(f"  history    : {first} -> {last}")
        print(f"  folds      : {len(folds)} ({window}, {args.test_months}m test windows)")
        print()

        # Printed because a rule left half-labelled by an interrupted run would
        # otherwise just look like a candidate that was never the best.
        print("  resolved trades per exit rule:")
        for version, frame in sorted(labels_by_rule.items()):
            print(f"    {version:<32} {len(frame):>8,}")
        print()

        try:
            result = wf.run(
                con, grid, folds,
                labels_by_rule=labels_by_rule,
                select_by=args.select_by,
                fixed=fixed,
                buy_fee=getattr(model, "buy_fee", 0.0),
                sell_fee=getattr(model, "sell_fee", 0.0),
                anchored=args.train_months is None,
            )
        except ValueError as exc:
            print(f"  {exc}", file=sys.stderr)
            return EXIT_ABORTED

    for f in result.folds:
        print(f"  fold {f.fold.index + 1}  {f.fold.test_start} -> {f.fold.test_end}")
        print(f"    chose        : {f.picked.name if f.picked else '(nothing had enough trades)'}")
        print(f"    in-sample    : {_round(f.in_sample_mean)}  ({f.in_sample_trades} trades)")
        print(f"    out-of-sample: {_round(f.out_of_sample_mean)}  ({f.out_of_sample_trades} trades)")
    print()

    usable = result.usable
    markdown = wf.render_markdown(result, cost_description=cost_line)
    path = wf.write(args.out, markdown)
    print(f"  written : {path} ({path.stat().st_size / 1024:.0f} KB)")

    premium = result.selection_premium
    if premium is not None:
        print(f"  selection premium : {premium:+.2f} points "
              "(the part of the in-sample figure that was hindsight)")
    pooled = result.fold_averaged("out_of_sample_mean")
    if pooled is not None:
        print(f"  out-of-sample mean: {pooled:+.2f}% per trade, one vote per fold")
    per_bar = result.fold_averaged("out_of_sample_per_bar")
    fixed_bar = result.fold_averaged("fixed_per_bar")
    market_bar = result.fold_averaged("universe_per_bar")
    if per_bar is not None:
        print(f"  per bar held      : {per_bar:+.3f}%  "
              f"(fixed rule {_round(fixed_bar, 3)}, market {_round(market_bar, 3)})")

    if not usable:
        print(
            "\n  No fold produced enough out-of-sample trades to conclude "
            "anything.\n  The report was written anyway, saying exactly that.",
            file=sys.stderr,
        )
        return EXIT_PARTIAL_FAILURE
    return EXIT_OK


def _round(value, digits: int = 2):
    return "—" if value is None else round(value, digits)


def cmd_verify(args: argparse.Namespace) -> int:
    """Cross-check stored prices (IVI-85).

    See idxcore/ingest/verify.py for why there is no independent second source:
    every free candidate is either bot-blocked or behind an API key.
    """
    from .ingest import verify as ver

    _print_header("verify")

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    with db.open_db(args.db, read_only=True) as con:
        report = ver.run(
            con,
            tickers=tickers,
            sample=args.sample,
            days=args.days,
            tolerance_pct=args.tolerance,
            skip_refetch=args.no_refetch,
        )

    print(report.render())
    print()
    if report.clean:
        print("  no problems found.")
    else:
        print(
            "  Mismatches are NOT automatically corrected. A difference may be\n"
            "  Yahoo revising a bar, or it may be a bug here - overwriting on\n"
            "  sight would destroy the evidence either way. Inspect, then\n"
            "  re-run `idxcore daily` if the fetched values are right.",
            file=sys.stderr,
        )
    return EXIT_OK if report.clean else EXIT_PARTIAL_FAILURE


def cmd_status(args: argparse.Namespace) -> int:
    """Show what is actually in the store. Data as of, not 'LIVE'."""
    _print_header("status")
    with db.open_db(args.db, read_only=True) as con:
        counts = db.universe_counts(con)
        print(f"  tickers        : {counts['total']} "
              f"({counts['active']} active, {counts['inactive']} inactive)")
        row = con.execute(
            "SELECT count(*), min(date), max(date) FROM prices"
        ).fetchone()
        print(f"  price rows     : {row[0]}")
        print(f"  price coverage : {row[1]} -> {row[2]}")
        run_id = db.latest_run_id(con)
        if run_id:
            print(f"  last run       : {run_id}")
            for status, n in sorted(db.run_status_counts(con, run_id).items()):
                print(f"    {status:<16}: {n}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def _add_fetch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tickers", help="comma-separated symbols; defaults to the active universe")
    p.add_argument("--limit", type=int, help="only the first N tickers (for smoke tests)")
    p.add_argument("--batch-size", type=int, default=40, help="tickers per yf.download call")
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="concurrency cap (max 3; high concurrency is what gets us rate limited)",
    )
    p.add_argument("--max-attempts", type=int, default=5, help="retries per batch on 429")
    p.add_argument(
        "--circuit-breaker",
        type=int,
        default=3,
        help="abort after this many consecutive rate-limited batches",
    )
    p.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.05,
        help="exit non-zero if more than this fraction of tickers failed",
    )
    p.add_argument("--end", type=dt.date.fromisoformat, help="YYYY-MM-DD, exclusive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idxcore",
        description="IDX signal dashboard data core.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"path to the DuckDB store (default: {db.DEFAULT_DB_PATH}, or $IDXCORE_DB)",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("init-db", help="create schema (idempotent, never wipes)")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("sync-universe", help="refresh ticker list, mark delistings")
    p.add_argument(
        "--from-file",
        help="Daftar Saham export (.xlsx/.csv) downloaded from idx.co.id",
    )
    p.add_argument("--no-web", action="store_true", help="do not try the IDX website")
    p.add_argument(
        "--allow-shrink",
        action="store_true",
        help="permit a large drop in the active universe (guard is on by default)",
    )
    p.add_argument(
        "--shrink-threshold",
        type=float,
        default=0.10,
        help="fraction of the active universe that may vanish before the guard trips",
    )
    p.add_argument(
        "--no-deactivate",
        action="store_true",
        help="add new tickers only; do not mark absent ones inactive",
    )
    p.add_argument("--write-seed", help="also write the resolved universe to this CSV")
    p.set_defaults(func=cmd_sync_universe)

    p = sub.add_parser("backfill", help="historical OHLCV for the active universe")
    p.add_argument("--years", type=int, default=10, help="how far back to fetch")
    p.add_argument("--start", type=dt.date.fromisoformat, help="YYYY-MM-DD, overrides --years")
    _add_fetch_args(p)
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("daily", help="delta fetch -> indicators -> signals")
    p.add_argument(
        "--overlap-days",
        type=int,
        default=7,
        help="re-fetch this many days before the last stored bar, so corrections land",
    )
    _add_fetch_args(p)
    p.set_defaults(func=cmd_daily)

    p = sub.add_parser("compute", help="quality gates, indicators and signals")
    p.add_argument("--tickers", help="comma-separated symbols; defaults to the active universe")
    p.add_argument("--limit", type=int, help="only the first N tickers")
    p.add_argument("--skip-quality", action="store_true", help="do not recompute quality flags")
    p.set_defaults(func=cmd_compute)

    p = sub.add_parser("backtest", help="replay signals and label outcomes")
    p.add_argument(
        "--hold", action="store_true",
        help="no target and no stop: hold to the horizon and exit at the close "
             "(the rule decided for this project, IVI-79)",
    )
    p.add_argument("--target", type=float, help="target, percent above the fill (e.g. 8)")
    p.add_argument("--stop", type=float, help="stop, percent below the fill (e.g. 4)")
    p.add_argument(
        "--horizon", type=int,
        help="trading days before an unresolved trade is closed at market",
    )
    p.add_argument(
        "--tie-break",
        choices=["unfavourable", "favourable"],
        default="unfavourable",
        help="same-day target AND stop: which wins (default: unfavourable, the stop)",
    )
    p.add_argument("--label", help="short name for this rule, used in rule_version")
    p.add_argument("--costs", help="path to a verified cost config JSON")
    p.add_argument(
        "--assume-fees",
        help="BUY,SELL as fractions (e.g. 0.0015,0.0025) to use the file's verified "
             "tick bands with fees you are explicitly stating as assumptions",
    )
    p.add_argument(
        "--write-example-config", help="write a blank cost config to this path and exit"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="exercise the replay without reporting outcomes or writing labels",
    )
    p.add_argument("--min-depth", type=int, default=3, help="minimum alignment depth to enter")
    p.add_argument("--no-rsi-gate", action="store_true", help="do not require RSI > 50")
    p.add_argument("--no-ma5-filter", action="store_true", help="do not require close > MA5")
    p.add_argument("--tickers", help="comma-separated subset")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("report", help="calibration report from stored labels")
    p.add_argument("--horizon", type=int, help="must match the backtest run")
    p.add_argument("--target", type=float)
    p.add_argument("--stop", type=float)
    p.add_argument("--label", help="rule label, if the backtest used one")
    p.add_argument("--costs", help="path to the cost config used for the run")
    p.add_argument("--assume-fees", help="BUY,SELL fractions, if the run assumed them")
    p.add_argument("--out", default="reports/calibration.md", help="where to write")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser(
        "walkforward",
        help="out-of-sample validation: choose on the past, measure on the future",
    )
    p.add_argument("--horizons", default="10,20,40",
                   help="comma-separated hold horizons in the candidate grid")
    p.add_argument("--fixed-horizon", type=int, default=20,
                   help="horizon of the fixed comparison rule (default: 20, the "
                        "rule decided in IVI-79)")
    p.add_argument("--fixed-tiers", default="A",
                   help="tiers the fixed comparison rule acts on (default: A)")
    p.add_argument("--select-by", default="return-per-bar",
                   choices=["return-per-bar", "mean", "median", "risk-adjusted"],
                   help="how the training window picks a winner. Default is "
                        "return-per-bar: candidates hold for different lengths "
                        "of time, and per-trade mean simply hands the contest "
                        "to whichever holds longest")
    p.add_argument("--test-months", type=int, default=12,
                   help="length of each out-of-sample window (default: 12)")
    p.add_argument("--min-train-months", type=int, default=24,
                   help="history required before the first test window (default: 24)")
    p.add_argument("--train-months", type=int,
                   help="rolling training window length; omit for anchored/expanding")
    p.add_argument("--costs", help="path to a verified cost config JSON")
    p.add_argument("--assume-fees",
                   help="BUY,SELL fractions to use the file's verified tick bands "
                        "with fees you are explicitly stating as assumptions")
    p.add_argument("--min-depth", type=int, default=3,
                   help="minimum alignment depth to enter")
    p.add_argument("--no-rsi-gate", action="store_true", help="do not require RSI > 50")
    p.add_argument("--no-ma5-filter", action="store_true", help="do not require close > MA5")
    p.add_argument("--quiet", action="store_true",
                   help="no per-ticker progress while replaying missing rules")
    p.add_argument("--out", default="reports/walkforward.md", help="where to write")
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("verify", help="cross-check stored prices")
    p.add_argument("--tickers", help="comma-separated subset; default is a random sample")
    p.add_argument("--sample", type=int, default=25, help="how many tickers to re-fetch")
    p.add_argument("--days", type=int, default=30, help="how far back to compare")
    p.add_argument("--tolerance", type=float, default=0.5, help="percent difference allowed")
    p.add_argument("--no-refetch", action="store_true",
                   help="OHLC integrity only, no network")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("status", help="what is in the store, and how fresh")
    p.set_defaults(func=cmd_status)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return EXIT_OK
    try:
        return args.func(args)
    except db.StoreLocked as exc:
        print(f"\n  {exc}", file=sys.stderr)
        return EXIT_ABORTED


if __name__ == "__main__":
    raise SystemExit(main())
