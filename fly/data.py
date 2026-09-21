"""
Marktdaten: was die Fliege riechen kann.

SPY ab 1993 (dividendenbereinigt, also Total Return), dazu VIX und die Rendite
10-jähriger US-Staatsanleihen. Alle drei reichen bis 1993 zurück — Gold oder
Dollar-ETFs gibt es erst ab 2004/2007 und würden die Evolution um zehn Jahre
Marktgeschichte kürzen.

Zusätzlich die neun Sektor-SPDR-ETFs (XLK … XLB, ab Dezember 1998) für die
Mehrmarkt-Variante: dieselbe Fliege riecht jeden Tag alle neun statt nur SPY.
`load_sector_closes()` liefert pro ETF einen closes-Frame mit derselben Form
wie `load_closes()` (Spalten spy/vix/tnx) — nur heißt "spy" hier der jeweilige
ETF, damit `fly/senses.py` unverändert bleibt. Alle neun teilen sich dieselben
Handelstage (Schnittmenge ihrer Kursreihen): sonst könnte ein Kalendertag in
einem Markt "fertig" sein, während er in einem anderen noch nicht existiert.

`Market.until(cutoff)` schneidet die Daten physisch ab. Die Evolution bekommt
nur dieses gekürzte Objekt; der Friedhofs-Zeitraum existiert für sie nicht.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START = "1993-01-01"
TICKERS = {"spy": "SPY", "vix": "^VIX", "tnx": "^TNX"}
SECTOR_TICKERS = {
    "xlk": "XLK", "xlf": "XLF", "xle": "XLE", "xlv": "XLV", "xly": "XLY",
    "xlp": "XLP", "xli": "XLI", "xlu": "XLU", "xlb": "XLB",
}


def _download(ticker: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Yahoo lieferte keine Daten für {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # neuere yfinance-Versionen: MultiIndex
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename(ticker).dropna()


def _load_series(name: str, ticker: str, refresh: bool) -> pd.Series:
    """Eine Kursreihe aus dem lokalen Cache (`data/<name>.csv`), bei Bedarf von Yahoo geladen."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.csv"
    if refresh or not path.exists():
        _download(ticker).to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]


def load_closes(refresh: bool = False) -> pd.DataFrame:
    """Schlusskurse aller Quellen, auf SPY-Handelstage ausgerichtet."""
    cols = {name: _load_series(name, ticker, refresh) for name, ticker in TICKERS.items()}
    spy_days = cols["spy"].index
    # VIX/TNX an SPY-Tage hängen; Lücken (Feiertage einer Quelle) mit dem
    # letzten bekannten Wert füllen — nie mit einem späteren.
    frame = pd.DataFrame({k: v.reindex(spy_days).ffill() for k, v in cols.items()})
    return frame.dropna()


def load_sector_closes(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Neun Sektor-ETFs, je ein closes-Frame wie bei `load_closes()`, nur dass
    Spalte "spy" der jeweilige ETF ist. Alle neun (plus VIX/TNX) werden auf
    die Schnittmenge ihrer Handelstage ausgerichtet — dieselbe gemeinsame
    Kalenderbasis, die die Zeitdisziplin über alle Märkte hinweg garantiert.
    """
    etf_series = {name: _load_series(name, ticker, refresh) for name, ticker in SECTOR_TICKERS.items()}
    vix = _load_series("vix", TICKERS["vix"], refresh)
    tnx = _load_series("tnx", TICKERS["tnx"], refresh)

    shared = None
    for s in etf_series.values():
        shared = s.index if shared is None else shared.intersection(s.index)
    shared = shared.sort_values()

    out = {}
    for name, s in etf_series.items():
        out[name] = pd.DataFrame({
            "spy": s.reindex(shared),
            "vix": vix.reindex(shared).ffill(),
            "tnx": tnx.reindex(shared).ffill(),
        }).dropna()
    return out


@dataclass(frozen=True)
class Market:
    closes: pd.DataFrame  # Spalten spy, vix, tnx; Index = Handelstage

    @property
    def days(self) -> pd.DatetimeIndex:
        return self.closes.index

    def until(self, cutoff: str | pd.Timestamp) -> "Market":
        """Alles VOR `cutoff` — der Rest wird nicht versteckt, sondern entfernt."""
        return Market(self.closes.loc[self.closes.index < pd.Timestamp(cutoff)])

    def next_day_returns(self) -> pd.Series:
        """Rendite von Schluss t bis Schluss t+1; am letzten Tag NaN."""
        spy = self.closes["spy"]
        return (spy.shift(-1) / spy - 1.0).rename("ret1")
