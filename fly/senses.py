"""
Die Sinne der Fliege — ihre Projektionsneuronen.

Bei der echten Fliege melden ~50 Glomeruli, welcher Duft gerade in der Luft
liegt. Hier melden 15 Marktmerkmale, wie der Markt heute "riecht". Jedes
Merkmal wird in zwei Kanäle zerlegt — AN (Wert über normal) und AUS (Wert unter
normal) —, wie die ON/OFF-Zellen im Sehsystem der Fliege. So kann eine
Kenyon-Zelle auf "Volatilität ungewöhnlich HOCH" hören, ohne dass ihr ein
negatives Vorzeichen dazwischenfunkt.

Kein Merkmal nutzt Daten nach dem Schluss des jeweiligen Tages. Der Test
`test_senses_do_not_look_ahead` prüft das, indem er die Zukunft abschneidet
und verlangt, dass sich die Vergangenheit nicht ändert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_WINDOW = 252   # "normal" heißt: gemessen am letzten Börsenjahr
Z_CLIP = 4.0


def raw_features(closes: pd.DataFrame) -> pd.DataFrame:
    spy, vix, tnx = closes["spy"], closes["vix"], closes["tnx"]
    logp = np.log(spy)
    r = logp.diff()

    f = pd.DataFrame(index=closes.index)
    for n in (1, 5, 20, 60, 120, 250):
        f[f"ret_{n}"] = logp.diff(n)                      # Momentum auf 6 Zeitskalen
    f["vol_20"] = r.rolling(20).std()
    f["vol_ratio"] = r.rolling(20).std() / r.rolling(120).std()  # wird es unruhiger?
    f["dist_sma50"] = logp - logp.rolling(50).mean()
    f["dist_sma200"] = logp - logp.rolling(200).mean()
    f["drawdown_250"] = logp - logp.rolling(250).max()    # wie weit unter dem Jahreshoch
    f["vix_level"] = np.log(vix)
    f["vix_change_5"] = np.log(vix).diff(5)
    f["tnx_change_20"] = tnx.diff(20)
    f["tnx_change_60"] = tnx.diff(60)
    return f


# Merkmale, deren NIVEAU nur relativ Sinn ergibt: gemessen am Mittel des letzten
# Jahres ("ist die Vola heute höher als normal?"). Alle anderen sind RICHTUNGEN,
# deren Vorzeichen selbst die Information ist ("über oder unter dem 200er-Schnitt?").
# Würde man die auch am Jahresmittel messen, hieße nach einem Jahr Aufwärtstrend
# "knapp über dem Schnitt" plötzlich "unter normal" — die Fliege verlöre genau
# den Trend, den sie riechen soll.
LEVELS = ("vol_20", "vol_ratio", "vix_level")


def channels(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Merkmale als rollierende z-Werte, aufgeteilt in AN- und AUS-Kanal.
    Niveaus: (x - Jahresmittel) / Jahresstreuung. Richtungen: x / typische
    Größe (RMS) des letzten Jahres — Vorzeichen bleibt erhalten.
    Tage, an denen noch nicht genug Geschichte für ein "normal" vorliegt,
    werden entfernt statt geraten.
    """
    raw = raw_features(closes)
    roll = raw.rolling(Z_WINDOW, min_periods=Z_WINDOW)
    level = (raw - roll.mean()) / roll.std()
    direction = raw / np.sqrt((raw ** 2).rolling(Z_WINDOW, min_periods=Z_WINDOW).mean())
    z = pd.concat([level[c] if c in LEVELS else direction[c] for c in raw.columns],
                  axis=1).clip(-Z_CLIP, Z_CLIP)

    on = z.clip(lower=0).add_suffix("+")
    off = (-z).clip(lower=0).add_suffix("-")
    return pd.concat([on, off], axis=1).dropna()
