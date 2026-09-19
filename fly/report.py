"""Kennzahlen für Tages-PnL-Reihen — Fliegenschwarm und Buy & Hold gleich gemessen."""
from __future__ import annotations

import numpy as np
import pandas as pd


def stats(daily: pd.Series, positions: pd.Series | None = None) -> dict:
    daily = daily.fillna(0.0)
    equity = (1.0 + daily).cumprod()
    years = len(daily) / 252
    sd = daily.std()
    out = {
        "Tage": len(daily),
        "Gesamt": equity.iloc[-1] - 1.0,
        "p.a.": equity.iloc[-1] ** (1 / years) - 1.0 if years > 0 else np.nan,
        "Vola": sd * np.sqrt(252),
        "Sharpe": daily.mean() / sd * np.sqrt(252) if sd > 0 else 0.0,
        "MaxDD": (1.0 - equity / equity.cummax()).max(),
    }
    if positions is not None:
        out["investiert"] = float((positions != 0).mean())
        out["short"] = float((positions == -1).mean())
        out["Wechsel"] = int((positions.diff().fillna(positions.iloc[0]) != 0).sum())
    return out


def table(rows: dict[str, dict]) -> str:
    """Markdown-Tabelle: eine Zeile je Strategie."""
    pct = {"Gesamt", "p.a.", "Vola", "MaxDD", "investiert", "short"}
    cols = list(dict.fromkeys(k for r in rows.values() for k in r))
    lines = ["| | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, r in rows.items():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if v == "":
                cells.append("")
            elif c in pct:
                cells.append(f"{v:+.1%}" if c in {"Gesamt", "p.a."} else f"{v:.1%}")
            elif isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
