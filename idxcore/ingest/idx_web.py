"""IDX (Bursa Efek Indonesia) as a universe source.

The exchange is the authoritative list of what is listed. Yahoo is not: it will
happily keep serving a symbol after the company has gone, and it has no notion
of an IDX listing board.

Two ways in, and the difference matters:

``fetch_securities``
    The official JSON endpoint on www.idx.co.id. As of the last check this
    domain answers automated clients with a Cloudflare interstitial (HTTP 403,
    "Just a moment..."), so this path raises `SourceBlockedError` rather than
    returning a partial list. Working around a bot check is out of scope.

``load_daftar_saham``
    Parses the *Daftar Saham* spreadsheet that IDX publishes for free download.
    A human clicks it once from the IDX website; this reads it. Authoritative,
    free, and unaffected by the bot check.

Neither path is allowed to return a short list quietly. A universe that shrinks
without explanation is how survivorship bias gets in, so a blocked or malformed
response is an exception, never an empty DataFrame.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

SOURCE_WEB = "idx_web"
SOURCE_FILE = "idx_file"

SECURITIES_URL = (
    "https://www.idx.co.id/primary/StockData/GetSecuritiesStock"
    "?start=0&length=10000&code=&sector=&board=&language=en-us"
)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    "Referer": "https://www.idx.co.id/en/market-data/stocks-data/stock-list/",
}

# Marks a Cloudflare (or similar) interstitial rather than a real response.
_CHALLENGE_MARKERS = ("just a moment", "cf-browser-verification", "cf_chl", "attention required")

#: IDX writes listing dates with a mix of English and Indonesian month
#: abbreviations in the same column — "09 Dec 1997" next to "12 Agt 2015".
#: pandas parses the English ones and silently returns NaT for the rest, so
#: every August listing would lose its date without a single warning.
_MONTH_FIXES = {
    "jan": "Jan", "feb": "Feb", "peb": "Feb", "mar": "Mar", "apr": "Apr",
    "mei": "May", "may": "May", "jun": "Jun", "jul": "Jul",
    "agt": "Aug", "agu": "Aug", "ags": "Aug", "agustus": "Aug", "aug": "Aug",
    "sep": "Sep", "set": "Sep", "okt": "Oct", "oct": "Oct",
    "nov": "Nov", "nop": "Nov", "des": "Dec", "dec": "Dec",
}

_MONTH_TOKEN = re.compile(r"\b([A-Za-z]{3,9})\b")


def normalise_month_names(value: object) -> object:
    """Rewrite Indonesian month abbreviations to English ones."""
    if not isinstance(value, str):
        return value

    def swap(match: re.Match) -> str:
        return _MONTH_FIXES.get(match.group(1).lower(), match.group(1))

    return _MONTH_TOKEN.sub(swap, value)


#: The "09 Dec 1997" shape IDX uses on the stock-list page.
_DMY_PATTERN = re.compile(r"^\s*\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\s*$")


def parse_listing_dates(values: pd.Series) -> pd.Series:
    """Parse a listing-date column that may mix both languages.

    Rows matching "DD Mon YYYY" are parsed with an explicit format; anything
    else (an ISO date from a differently-shaped export, say) is left to pandas
    to infer. Forcing dayfirst on everything would misread ISO dates and warn.
    """
    cleaned = values.map(normalise_month_names)
    as_text = cleaned.astype("string")
    is_dmy = as_text.str.match(_DMY_PATTERN).fillna(False)

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if is_dmy.any():
        parsed.loc[is_dmy] = pd.to_datetime(
            as_text[is_dmy], format="%d %b %Y", errors="coerce"
        )
    if (~is_dmy).any():
        parsed.loc[~is_dmy] = pd.to_datetime(as_text[~is_dmy], errors="coerce")
    return parsed


#: Column aliases seen in the Daftar Saham file, Indonesian and English.
_COLUMN_ALIASES = {
    "idx_code": ("kode", "kode saham", "code", "stock code", "kode emiten"),
    "name": ("nama perusahaan", "company name", "nama emiten", "name", "nama"),
    "listing_date": ("tanggal pencatatan", "listing date", "tgl pencatatan", "tanggal"),
    "board": ("papan pencatatan", "listing board", "papan", "board"),
    "sector": ("sektor", "sector", "industry", "sub sektor", "sub-sektor"),
}


class SourceBlockedError(RuntimeError):
    """The source answered, but with a bot check instead of data."""


class SourceFormatError(RuntimeError):
    """The source answered with something we could not parse as a stock list."""


def yahoo_symbol(idx_code: str) -> str:
    """IDX code -> Yahoo symbol. 'BBCA' -> 'BBCA.JK'."""
    code = str(idx_code).strip().upper()
    if not code:
        raise ValueError("empty IDX code")
    return code if code.endswith(".JK") else f"{code}.JK"


def idx_code(symbol: str) -> str:
    """Yahoo symbol -> IDX code. 'BBCA.JK' -> 'BBCA'."""
    s = str(symbol).strip().upper()
    return s[:-3] if s.endswith(".JK") else s


def _looks_like_challenge(text: str) -> bool:
    lowered = text[:4000].lower()
    return any(marker in lowered for marker in _CHALLENGE_MARKERS)


def fetch_securities(
    *,
    url: str = SECURITIES_URL,
    timeout: int = 30,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch the listed-stock table from the IDX website.

    Raises SourceBlockedError when the response is a bot check. That is a real,
    reportable condition — not a reason to fall through to an empty universe.
    """
    http = session or requests.Session()
    response = http.get(url, headers=_BROWSER_HEADERS, timeout=timeout)

    if response.status_code == 403 or _looks_like_challenge(response.text):
        raise SourceBlockedError(
            f"IDX returned HTTP {response.status_code} with a bot-check page. "
            "The stock list is still available as a free manual download: "
            "https://www.idx.co.id/en/market-data/stocks-data/stock-list/ "
            "-> download Daftar Saham, then run "
            "`idxcore sync-universe --from-file <path>`."
        )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise SourceFormatError(f"IDX response was not JSON: {exc}") from exc

    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise SourceFormatError(
            "IDX response contained no stock rows. Refusing to treat this as "
            "an empty universe."
        )
    return _records_to_frame(records, source=SOURCE_WEB)


def _records_to_frame(records: list[dict], *, source: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    return normalise_columns(frame, source=source)


def load_daftar_saham(path: str | Path, *, source: str = SOURCE_FILE) -> pd.DataFrame:
    """Parse a Daftar Saham export (.xlsx / .xls / .csv) downloaded from IDX."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"universe file not found: {p}")

    if p.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(p, dtype=str)
    elif p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        frame = pd.read_excel(p, dtype=str)
    else:
        raise SourceFormatError(
            f"unsupported universe file type {p.suffix!r}; expected .xlsx, .xls or .csv"
        )
    if frame.empty:
        raise SourceFormatError(f"{p} contained no rows")
    return normalise_columns(frame, source=source)


def normalise_columns(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Map a source's columns onto the `tickers` shape.

    Only `idx_code` is mandatory. Everything else is best-effort: a missing
    sector is a blank field, not a dropped company.
    """
    lookup = {str(c).strip().lower(): c for c in frame.columns}

    def pick(field: str) -> Optional[str]:
        for alias in _COLUMN_ALIASES[field]:
            if alias in lookup:
                return lookup[alias]
        return None

    code_col = pick("idx_code")
    if code_col is None:
        raise SourceFormatError(
            "could not find a stock-code column; looked for "
            f"{list(_COLUMN_ALIASES['idx_code'])}, saw {list(frame.columns)[:12]}"
        )

    out = pd.DataFrame()
    out["idx_code"] = (
        frame[code_col].astype(str).str.strip().str.upper().str.replace(".JK", "", regex=False)
    )
    out = out[out["idx_code"].str.fullmatch(r"[A-Z0-9]{2,6}")].copy()
    if out.empty:
        raise SourceFormatError(
            f"no rows in the {source} response had a valid IDX code"
        )

    for field in ("name", "sector", "board"):
        col = pick(field)
        out[field] = frame.loc[out.index, col].astype(str).str.strip() if col else None
        if col is not None:
            out[field] = out[field].replace({"": None, "nan": None, "None": None})

    date_col = pick("listing_date")
    if date_col is not None:
        parsed = parse_listing_dates(frame.loc[out.index, date_col])
        out["listing_date"] = parsed.dt.date
    else:
        out["listing_date"] = None

    out["ticker"] = out["idx_code"].map(yahoo_symbol)
    out["source"] = source
    out = out.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return out.loc[:, ["ticker", "idx_code", "name", "sector", "board", "listing_date", "source"]]


def write_seed_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist a resolved universe so a later run can reproduce it offline."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(p, index=False)
    return p


def verify_on_yahoo(
    symbols: list[str],
    *,
    timeout: int = 20,
    session: Optional[requests.Session] = None,
    expected_exchange: str = "JKT",
) -> pd.DataFrame:
    """Cross-check that symbols resolve to Jakarta-listed equities on Yahoo.

    This is a verification tool, never an enumerator: Yahoo has no endpoint
    that lists an exchange's members, and guessing codes would produce a
    universe of unknown completeness presented as complete.
    """
    http = session or requests.Session()
    rows = []
    for symbol in symbols:
        url = (
            f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            "?range=5d&interval=1d"
        )
        try:
            r = http.get(url, headers=_BROWSER_HEADERS, timeout=timeout)
            if r.status_code != 200:
                rows.append({"ticker": symbol, "resolved": False, "detail": f"HTTP {r.status_code}"})
                continue
            meta = (r.json().get("chart", {}).get("result") or [{}])[0].get("meta", {})
            exchange = meta.get("exchangeName")
            rows.append(
                {
                    "ticker": symbol,
                    "resolved": exchange == expected_exchange,
                    "detail": f"exchange={exchange} currency={meta.get('currency')}",
                }
            )
        except Exception as exc:  # noqa: BLE001 - reported per symbol, never swallowed
            rows.append(
                {"ticker": symbol, "resolved": False, "detail": f"{type(exc).__name__}: {exc}"}
            )
    return pd.DataFrame(rows)


def today() -> dt.date:
    return dt.date.today()
