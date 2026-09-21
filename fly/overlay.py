"""
Die Positionsschicht: deterministisches Python über der Fliege.

Befund: Die Fliege ist ein Krisen-Detektor. Sie schlägt SPY in jeder Krise und
verliert in jedem ruhigen Bullenmarkt. Also bekommt sie nur dort eine Stimme,
wo sie etwas kann.

Regel (bedingtes Volatility-Targeting nach Bongaerts, Kang & van Dijk,
Financial Analysts Journal 2020), VOR dem ersten Test festgelegt:

    Grundposition  = Long mit Risikoziel: TARGET_VOL / aktuelle Vola, gedeckelt bei `cap`
    Fliege greift nur ein, wenn die Vola EXTREM ist (über dem 80. Perzentil
    ihrer eigenen Geschichte bis gestern). Dann gilt das Schwarm-Signal:
    Long -> Grundposition, NO TRADE -> 0, Short -> -Grundposition.

Alle Größen sind zum Schluss des Tages bekannt; die Position gilt für den
nächsten Tag — genau wie das Schwarm-Signal selbst.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_VOL = 0.15      # Jahresvola, auf die die Grundposition zielt
EXTREME_Q = 0.80       # ab diesem Perzentil gilt die Vola als extrem
VOL_DAYS = 20
MIN_HISTORY = 252      # vorher gibt es kein "extrem" — dann nur Grundposition


def exposure(spy_close: pd.Series, swarm: pd.Series, cap: float = 1.0) -> pd.DataFrame:
    """Zielgewicht in SPY je Tag (Anteil am Kapital, negativ = short)."""
    r = np.log(spy_close).diff()
    vol = r.rolling(VOL_DAYS).std() * np.sqrt(252)
    # Perzentil nur aus der Vergangenheit (bis gestern), damit heute nichts vorausweiß.
    threshold = vol.shift(1).expanding(MIN_HISTORY).quantile(EXTREME_Q)
    base = (TARGET_VOL / vol).clip(upper=cap)
    extreme = vol > threshold
    sig = swarm.reindex(spy_close.index).fillna(0)
    w = base.where(~extreme, base * sig)
    return pd.DataFrame({"gewicht": w.fillna(0.0), "basis": base, "vola": vol,
                         "extrem": extreme.fillna(False), "schwarm": sig})


def returns(weights: pd.Series, spy_close: pd.Series, cost: float = 0.0005,
            borrow: float = 0.01 / 252, financing: float = 0.05 / 252) -> pd.Series:
    """
    Tagesergebnis eines Gewichtspfads. Gewicht von Tag t gilt für t -> t+1.
    Kosten je umgeschichtetem Anteil, Leihgebühr für Short, und für Hebel über 1
    Finanzierungskosten (5 % p.a. — bewusst pessimistisch).
    """
    nxt = spy_close.shift(-1) / spy_close - 1.0
    turnover = weights.diff().abs().fillna(weights.abs())
    lev = (weights.abs() - 1.0).clip(lower=0)
    out = weights * nxt - cost * turnover - borrow * (weights < 0) * weights.abs() - financing * lev
    return out.fillna(0.0)
