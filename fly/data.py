"""
Marktdaten: was die Fliege riechen kann.

SPY ab 1993 (dividendenbereinigt, also Total Return), dazu VIX und die Rendite
10-jähriger US-Staatsanleihen. Alle drei reichen bis 1993 zurück.

Für die v3-Sinne kommen vier ETFs dazu, die es 1993 noch nicht gab: HYG/LQD
(Kreditaufschlag), RSP (Marktbreite gegen SPY), GLD und TLT (Fluchtwerte).
Ältester davon ist HYG (Start April 2007) — `load_closes(senses="v3")` lädt
sie mit, wodurch `dropna()` die Historie automatisch auf ~2007 kürzt.
`load_closes(senses="v2")` (Standard) lädt nur die alten drei und behält die
volle Historie ab 1993.

`Market.until(cutoff)` schneidet die Daten physisch ab. Die Evolution bekommt
nur dieses gekürzte Objekt; der Friedhofs-Zeitraum existiert für sie nicht.
`Market.since(start)` schneidet von unten ab — für faire Vergleiche zwischen
v2- und v3-Sinnen auf demselben Zeitraum.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START = "1993-01-01"
TICKERS = {"spy": "SPY", "vix": "^VIX", "tnx": "^TNX"}
# v3: Kreditaufschlag (HYG/LQD), Marktbreite (RSP/SPY), Fluchtwerte (GLD, TLT).
# HYG (Start 2007-04) ist der jüngste — bestimmt, wo die v3-Historie beginnt.
TICKERS_V3 = {"hyg": "HYG", "lqd": "LQD", "rsp": "RSP", "gld": "GLD", "tlt": "TLT"}


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


def load_closes(refresh: bool = False, senses: str = "v2") -> pd.DataFrame:
    """
    Schlusskurse aller Quellen, auf SPY-Handelstage ausgerichtet.
    `senses="v3"` lädt zusätzlich die vier neuen ETFs dazu; `dropna()` kürzt
    die Historie dadurch automatisch auf deren gemeinsamen Startpunkt.
    """
    tickers = {**TICKERS, **(TICKERS_V3 if senses == "v3" else {})}
    DATA_DIR.mkdir(exist_ok=True)
    cols = {}
    for name, ticker in tickers.items():
        path = DATA_DIR / f"{name}.csv"
        if refresh or not path.exists():
            _download(ticker).to_csv(path)
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        cols[name] = s
    spy_days = cols["spy"].index
    # VIX/TNX an SPY-Tage hängen; Lücken (Feiertage einer Quelle) mit dem
    # letzten bekannten Wert füllen — nie mit einem späteren.
    frame = pd.DataFrame({k: v.reindex(spy_days).ffill() for k, v in cols.items()})
    return frame.dropna()


@dataclass(frozen=True)
class Market:
    closes: pd.DataFrame  # Spalten spy, vix, tnx; Index = Handelstage

    @property
    def days(self) -> pd.DatetimeIndex:
        return self.closes.index

    def until(self, cutoff: str | pd.Timestamp) -> "Market":
        """Alles VOR `cutoff` — der Rest wird nicht versteckt, sondern entfernt."""
        return Market(self.closes.loc[self.closes.index < pd.Timestamp(cutoff)])

    def since(self, start: str | pd.Timestamp) -> "Market":
        """Alles AB `start` — für einen fairen Vergleich verschiedener Sinne auf gleichem Zeitraum."""
        return Market(self.closes.loc[self.closes.index >= pd.Timestamp(start)])

    def next_day_returns(self) -> pd.Series:
        """Rendite von Schluss t bis Schluss t+1; am letzten Tag NaN."""
        spy = self.closes["spy"]
        return (spy.shift(-1) / spy - 1.0).rename("ret1")
