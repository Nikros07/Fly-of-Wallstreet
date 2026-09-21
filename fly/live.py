"""
Live-Betrieb der Kolonie: täglich weiterleben, wöchentlich auslesen, abstimmen.

    python -m fly.live init     Kolonie gründen und von 1994 bis heute leben lassen
    python -m fly.live step     täglich: neue Kurse holen, weiterleben, Signal schreiben

Zustand (state/colony.npz) und Signale (signals.csv) liegen im Repo-Verzeichnis;
der GitHub-Actions-Lauf (.github/workflows/daily.yml) bewahrt den Zustand auf
dem Zweig `fly-state` auf und die Signale auf `main`.

Buchhaltung: Die Rendite des jüngsten Tages steht erst am nächsten Handelstag
fest. Der gespeicherte Zustand lebt deshalb nur bis zum VORLETZTEN Tag. Das
Signal für heute stammt aus einer Wegwerf-Kopie, die den letzten Tag erlebt —
so bleibt der echte Zustand bitgenau auf dem Pfad des Backtests.

Ist ein Alpaca-PAPIER-Konto hinterlegt (Umgebungsvariablen ALPACA_KEY_ID und
ALPACA_SECRET_KEY), wird die Papier-Position an das Signal angeglichen. Mit
echtem Geld handelt dieser Code nicht (siehe fly/broker.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd

from fly.__main__ import build
from fly.colony import Colony, ColonyConfig
from fly.population import World

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state" / "colony.npz"
META = ROOT / "state" / "meta.json"
SIGNALS = ROOT / "signals.csv"
NAMES = {1: "LONG", 0: "NO TRADE", -1: "SHORT"}

# Die Einstellung, die im Backtest gewählt wurde (README, Abschnitt Wochenzucht).
LIVE_CONFIG = dict(window=504, exams=2)


def _world():
    market, skull = build(refresh=True)
    return market, skull, World.build(market, skull)


def _publish(col: Colony, world: World, market, T: int) -> int:
    """Signal für den letzten bekannten Tag aus einer Kopie — der Zustand bleibt unberührt."""
    ghost = copy.deepcopy(col)
    swarm, pos = ghost.advance(world, T - 1, T)
    signal = int(swarm[-1])
    day = world.days[T - 1]
    voters = ghost.swarm_idx
    longs = int((pos[voters, -1] == 1).sum()) if len(voters) else 0
    shorts = int((pos[voters, -1] == -1).sum()) if len(voters) else 0
    row = pd.DataFrame([{
        "tag": day.date(), "signal": NAMES[signal], "stimmen_long": longs,
        "stimmen_short": shorts, "schwarm": len(voters), "quorum": col.cfg.quorum,
        "spy_schluss": round(float(market.closes["spy"].iloc[-1]), 2),
    }])
    # Gleicher Tag nochmal (zweiter Lauf, jetzt mit endgültigem Schlusskurs):
    # Zeile ERSETZEN, nicht überspringen — der spätere Lauf weiß es besser.
    old = pd.read_csv(SIGNALS, dtype=str) if SIGNALS.exists() else pd.DataFrame()
    if len(old):
        old = old[old["tag"] != str(day.date())]
    pd.concat([old, row.astype(str)], ignore_index=True).to_csv(SIGNALS, index=False)
    print(f"{day:%Y-%m-%d}: {NAMES[signal]}  ({longs}× Long, {shorts}× Short von "
          f"{len(voters)} Schwarm-Fliegen, Quorum {col.cfg.quorum})")

    if os.environ.get("ALPACA_KEY_ID") and os.environ.get("ALPACA_SECRET_KEY"):
        # Ein Broker-Fehler darf Signal und Zustand nicht mitreißen: Die werden
        # danach noch ins Repo geschrieben. Fehler laut melden, Lauf nicht abbrechen.
        from fly.broker import sync_paper_position
        try:
            sync_paper_position(signal, price=float(market.closes["spy"].iloc[-1]))
        except Exception as e:                                   # noqa: BLE001
            print(f"::warning::Papierkonto nicht angeglichen: {type(e).__name__}: {e}")
    return signal


def _save(col: Colony, world: World, T: int) -> None:
    STATE.parent.mkdir(exist_ok=True)
    col.save(STATE)
    META.write_text(json.dumps({"last_day": str(world.days[T - 2].date()),
                                "config": {k: v for k, v in vars(col.cfg).items()}},
                               indent=2, default=str), encoding="utf-8")


def init(seed: int) -> None:
    market, skull, world = _world()
    T = len(world.days)
    col = Colony.found(ColonyConfig(seed=seed, **LIVE_CONFIG), skull.n_kc)
    print(f"Kolonie gegründet (Seed {seed}), lebt {world.days[0]:%Y-%m-%d} bis "
          f"{world.days[T - 2]:%Y-%m-%d} …", flush=True)
    col.advance(world, 0, T - 1)
    _save(col, world, T)
    _publish(col, world, market, T)


def step() -> None:
    if not STATE.exists():
        sys.exit("Kein Zustand — erst 'python -m fly.live init' laufen lassen.")
    col = Colony.load(STATE)
    market, skull, world = _world()
    T = len(world.days)
    last = json.loads(META.read_text(encoding="utf-8"))["last_day"]
    t0 = int(world.days.searchsorted(pd.Timestamp(last))) + 1
    if t0 > T - 1:
        # Yahoo liefert weniger Tage als beim letzten Lauf (veraltete Antwort).
        # Dann weder Signal noch Order — sonst erlebte die Kopie einen Tag doppelt.
        print(f"Kursdaten enden {world.days[-1]:%Y-%m-%d}, Zustand ist schon weiter — nichts zu tun.")
        return
    if t0 < T - 1:
        col.advance(world, t0, T - 1)
        _save(col, world, T)
        print(f"{T - 1 - t0} Handelstag(e) weitergelebt.")
    _publish(col, world, market, T)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="fly.live", description="Kolonie im Live-Betrieb")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--seed", type=int, default=1)
    sub.add_parser("step")
    args = p.parse_args()
    init(args.seed) if args.cmd == "init" else step()


if __name__ == "__main__":
    main()
