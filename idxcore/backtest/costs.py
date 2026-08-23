"""Cost model (IVI-80).

**This module ships with no numbers in it, on purpose.**

A backtest without costs is a advertisement, not a measurement: broker fees and
the IDX tick grid are large relative to the moves this strategy trades. So the
rule from the brief is absolute — *do not report backtest results without the
cost model applied* — and the rule that makes it safe is equally absolute: *do
not hardcode the IDX tick-size table from memory.*

The *fraksi harga* bands have been revised more than once. A remembered table
would be wrong in a way that is invisible in the output: every fill would
quietly round to the wrong grid and every return would be off by a little, in a
consistent direction. That is worse than having no backtest.

So the tick table and the fee rates are **configuration**, loaded from a JSON
file the owner fills in from their own broker statement and from IDX's
published rules. Until that file exists and is marked verified, the cost model
refuses to construct and the report refuses to run.

    idxcore backtest --costs config/costs.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

#: IDX trades in lots. This one is safe to state: it is 100 shares and has been
#: since 2014, it is not a banded table, and it is trivially checkable.
LOT_SIZE = 100

EXAMPLE_PATH = Path("config/costs.example.json")


class CostModelNotConfigured(RuntimeError):
    """Raised when a cost model is required but has not been supplied.

    Deliberately fatal. Falling back to "assume zero fees" would produce a
    number that looks like a result.
    """


class CostConfigInvalid(ValueError):
    """The config file exists but cannot be trusted as written."""


@dataclass(frozen=True)
class TickBand:
    """One *fraksi harga* band: prices in [lower, upper) move in `tick` steps."""

    lower: float
    upper: Optional[float]   # None = open-ended top band
    tick: float

    def contains(self, price: float) -> bool:
        if price < self.lower:
            return False
        return self.upper is None or price < self.upper


@dataclass(frozen=True)
class TickTable:
    bands: tuple[TickBand, ...]

    @classmethod
    def from_rows(cls, rows: Sequence[dict]) -> "TickTable":
        if not rows:
            raise CostConfigInvalid("tick_bands is empty")
        bands = tuple(
            TickBand(
                lower=float(r["lower"]),
                upper=None if r.get("upper") in (None, "", "inf") else float(r["upper"]),
                tick=float(r["tick"]),
            )
            for r in rows
        )
        ordered = sorted(bands, key=lambda b: b.lower)
        if ordered[0].lower > 0:
            raise CostConfigInvalid(
                f"tick bands must start at 0; lowest band starts at {ordered[0].lower}"
            )
        for a, b in zip(ordered, ordered[1:]):
            if a.upper is None:
                raise CostConfigInvalid("only the highest band may be open-ended")
            if not math.isclose(a.upper, b.lower):
                raise CostConfigInvalid(
                    f"gap or overlap between bands at {a.upper} and {b.lower}"
                )
        if ordered[-1].upper is not None:
            raise CostConfigInvalid("the highest tick band must be open-ended")
        if any(b.tick <= 0 for b in ordered):
            raise CostConfigInvalid("every tick must be positive")
        return cls(bands=ordered if isinstance(ordered, tuple) else tuple(ordered))

    def tick_for(self, price: float) -> float:
        for band in self.bands:
            if band.contains(price):
                return band.tick
        raise CostConfigInvalid(f"no tick band covers price {price}")

    def round_to_tick(self, price: float, *, direction: str = "nearest") -> float:
        """Snap a price onto the exchange's grid.

        `direction` is 'up' for a buy and 'down' for a sell when you want the
        conservative side — you cannot transact between ticks.
        """
        tick = self.tick_for(price)
        steps = price / tick
        if direction == "up":
            snapped = math.ceil(steps - 1e-9) * tick
        elif direction == "down":
            snapped = math.floor(steps + 1e-9) * tick
        else:
            snapped = round(steps) * tick
        return round(snapped, 6)


@dataclass(frozen=True)
class CostModel:
    """Broker fees plus the exchange tick grid.

    Fees are fractions, not percents: 0.0015 means 0.15%.
    """

    buy_fee: float
    sell_fee: float
    tick_table: TickTable
    lot_size: int = LOT_SIZE
    slippage_ticks: float = 0.0
    source: str = "config"
    notes: str = ""
    tick_bands_verified_on: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "CostModel":
        p = Path(path)
        if not p.exists():
            raise CostModelNotConfigured(
                f"no cost config at {p}.\n"
                f"Copy {EXAMPLE_PATH} to {p}, fill in your broker's actual buy/sell "
                "fee rates and the current IDX fraksi harga bands (verified against "
                "IDX's published rules, not from memory), then set "
                '"verified": true.'
            )
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CostConfigInvalid(f"{p} is not valid JSON: {exc}") from exc
        return cls.from_dict(raw, source=str(p))

    @classmethod
    def from_dict(cls, raw: dict, *, source: str = "dict") -> "CostModel":
        if not raw.get("verified"):
            raise CostModelNotConfigured(
                f"{source} is not marked verified.\n"
                "The IDX tick bands have been revised more than once, so this file "
                "must be checked against IDX's published fraksi harga rules and the "
                'fees against a real broker statement. Then set "verified": true.'
            )
        for key in ("buy_fee", "sell_fee", "tick_bands"):
            if key not in raw:
                raise CostConfigInvalid(f"{source} is missing required key {key!r}")

        buy_fee = float(raw["buy_fee"])
        sell_fee = float(raw["sell_fee"])
        for name, value in (("buy_fee", buy_fee), ("sell_fee", sell_fee)):
            if not 0.0 <= value < 0.05:
                raise CostConfigInvalid(
                    f"{name}={value} is outside a plausible range. "
                    "Fees are fractions: 0.0015 means 0.15%."
                )

        return cls(
            buy_fee=buy_fee,
            sell_fee=sell_fee,
            tick_table=TickTable.from_rows(raw["tick_bands"]),
            lot_size=int(raw.get("lot_size", LOT_SIZE)),
            slippage_ticks=float(raw.get("slippage_ticks", 0.0)),
            source=source,
            notes=str(raw.get("notes", "")),
            tick_bands_verified_on=str(raw.get("tick_bands_verified_on", "")),
        )

    # ---- application ----------------------------------------------------

    def buy_price(self, raw_price: float) -> float:
        """What a buy actually fills at: snapped up a tick grid, plus slippage."""
        price = self.tick_table.round_to_tick(raw_price, direction="up")
        if self.slippage_ticks:
            price += self.slippage_ticks * self.tick_table.tick_for(price)
            price = self.tick_table.round_to_tick(price, direction="up")
        return price

    def sell_price(self, raw_price: float) -> float:
        price = self.tick_table.round_to_tick(raw_price, direction="down")
        if self.slippage_ticks:
            price -= self.slippage_ticks * self.tick_table.tick_for(price)
            price = self.tick_table.round_to_tick(price, direction="down")
        return max(price, 0.0)

    def net_return_pct(self, entry: float, exit_: float) -> float:
        """Round-trip return after tick rounding and both fees, as a percent."""
        buy = self.buy_price(entry)
        sell = self.sell_price(exit_)
        if buy <= 0:
            raise CostConfigInvalid(f"non-positive fill price {buy}")
        paid = buy * (1.0 + self.buy_fee)
        received = sell * (1.0 - self.sell_fee)
        return (received / paid - 1.0) * 100.0

    def round_trip_cost_pct(self, price: float) -> float:
        """How much a flat trade loses to fees and the grid, in percent."""
        return -self.net_return_pct(price, price)

    def describe(self) -> str:
        top = self.tick_table.bands[-1]
        return (
            f"costs from {self.source}\n"
            f"  buy fee   : {self.buy_fee * 100:.4f}%\n"
            f"  sell fee  : {self.sell_fee * 100:.4f}%\n"
            f"  round trip: {(self.buy_fee + self.sell_fee) * 100:.4f}% before tick effects\n"
            f"  lot size  : {self.lot_size} shares\n"
            f"  slippage  : {self.slippage_ticks} tick(s)\n"
            f"  tick bands: {len(self.tick_table.bands)} "
            f"(top band from {top.lower:g}, tick {top.tick:g})\n"
            f"  verified  : {self.tick_bands_verified_on or 'date not recorded'}"
        )


def from_file_with_assumed_fees(
    path: str | Path, buy_fee: float, sell_fee: float
) -> "CostModel":
    """Use a config's *verified tick bands* with fees you are stating as assumptions.

    This exists so work can proceed without a brokerage account, and it is
    deliberately awkward to reach: the caller has to type the numbers, and the
    resulting model records in `source` that they were assumed. There is still
    no path where a missing fee silently becomes zero.

    The tick grid is not assumed — it is whatever the file holds, which for this
    project was read from IDX's published Peraturan II-A.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not raw.get("tick_bands"):
        raise CostConfigInvalid(f"{path} has no tick_bands to borrow")
    raw = dict(raw)
    raw["verified"] = True
    raw["buy_fee"] = buy_fee
    raw["sell_fee"] = sell_fee
    model = CostModel.from_dict(
        raw,
        source=(
            f"{path} tick bands + ASSUMED fees "
            f"{buy_fee * 100:.4f}%/{sell_fee * 100:.4f}% (not a verified broker rate)"
        ),
    )
    return model


def example_config() -> dict:
    """The shape of the file, with every number left blank on purpose."""
    return {
        "verified": False,
        "_README": [
            "Fill this in, then save as config/costs.json and set verified to true.",
            "buy_fee / sell_fee are FRACTIONS: 0.0015 means 0.15%. Take them from a",
            "real broker statement, including levy and VAT, not from a marketing page.",
            "tick_bands are the IDX fraksi harga bands. VERIFY THEM against IDX's",
            "published rules - they have been revised more than once and a remembered",
            "table is wrong in a way you cannot see in the output.",
            "The lowest band must start at 0 and the highest must have upper: null.",
        ],
        "buy_fee": None,
        "sell_fee": None,
        "lot_size": LOT_SIZE,
        "slippage_ticks": 0,
        "tick_bands_verified_on": "",
        "notes": "",
        "tick_bands": [
            {"lower": 0, "upper": None, "tick": None},
        ],
    }


def write_example(path: str | Path = EXAMPLE_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(example_config(), indent=2) + "\n", encoding="utf-8")
    return p
