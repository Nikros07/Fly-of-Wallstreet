"""
Der tägliche Lauf: Was sagt der Schwarm für morgen?

    python -m fly.daily --init results/<lauf>/swarm.npz   einmalig: Schwarm einsetzen
    python -m fly.daily                                   täglich: Kurse holen, lernen, abstimmen

Gedacht für einen kostenlosen GitHub-Actions-Cron (.github/workflows/daily.yml).
Der Lauf schreibt seinen Zustand zurück ins Repo:

    swarm/current.npz   Gedächtnis und Gene der 10 Schwarm-Fliegen
    swarm/state.json    bis zu welchem Tag sie gelebt haben
    swarm/signals.csv   jede Entscheidung mit Datum und Stimmenzahl

Damit ist jedes Signal vor dem nächsten Handelstag im Git-Verlauf festgeschrieben
und nachträglich nicht mehr hübschbar.

WICHTIG: Hier wird NICHTS gehandelt und kein Konto angefasst. Der Lauf schreibt
auf, was der Schwarm täte. Ob daraus je echtes Geld wird, entscheidet Nick —
und erst, nachdem eine Version im Walk-forward und auf dem Friedhof bestanden hat.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fly.__main__ import build
from fly.population import LONG, SHORT, Population, World, live

SWARM_DIR = Path(__file__).resolve().parent.parent / "swarm"
MEMORY = SWARM_DIR / "current.npz"
STATE = SWARM_DIR / "state.json"
SIGNALS = SWARM_DIR / "signals.csv"
NAMES = {1: "LONG", 0: "NO TRADE", -1: "SHORT"}


def init(source: Path, quorum: int) -> None:
    SWARM_DIR.mkdir(exist_ok=True)
    shutil.copy(source, MEMORY)
    STATE.write_text(json.dumps({"last_day": None, "quorum": quorum, "source": str(source)},
                                indent=2), encoding="utf-8")
    print(f"Schwarm eingesetzt aus {source}. Nächster Lauf lässt ihn bis heute weiterleben.")


def run(max_catchup: int = 400) -> int:
    if not MEMORY.exists():
        sys.exit("Kein Schwarm vorhanden — erst 'python -m fly.daily --init <swarm.npz>' laufen lassen.")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    swarm = Population.load(MEMORY)

    market, skull = build(refresh=True)
    world = World.build(market, skull)

    # Ab wo weiterleben? Beim ersten Lauf nur ein begrenztes Stück Vergangenheit
    # nachholen, damit ein vergessener Cron nicht Jahre nachrechnet.
    last = state.get("last_day")
    start = (int(np.searchsorted(world.days, pd.Timestamp(last))) + 1 if last
             else max(0, len(world.days) - max_catchup))
    end = len(world.days)
    if start >= end:
        print(f"Nichts Neues seit {last} — kein Handelstag dazugekommen.")
        return 0

    positions = live(swarm, world, start, end, learn=True)
    longs = (positions[:, -1] == 1).sum()
    shorts = (positions[:, -1] == -1).sum()
    quorum = state["quorum"]
    signal = 1 if longs >= quorum else -1 if shorts >= quorum else 0
    day = world.days[end - 1]

    values = swarm.values(world.kc[end - 1])
    row = pd.DataFrame([{
        "tag": day.date(), "signal": NAMES[signal], "stimmen_long": int(longs),
        "stimmen_short": int(shorts), "quorum": quorum,
        "wert_long": round(float(values[:, LONG].mean()), 4),
        "wert_short": round(float(values[:, SHORT].mean()), 4),
        "spy": round(float(market.closes["spy"].iloc[-1]), 2),
    }])
    row.to_csv(SIGNALS, mode="a", header=not SIGNALS.exists(), index=False)

    sv = swarm
    np.savez_compressed(MEMORY, go=sv.go, nogo=sv.nogo, ids=sv.ids,
                        **{f"gene_{k}": v for k, v in sv.genes.items()})
    state["last_day"] = str(day.date())
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"{day:%Y-%m-%d}: {NAMES[signal]}  ({longs}× Long, {shorts}× Short, "
          f"{len(sv.ids) - longs - shorts}× still; Quorum {quorum})")
    print(f"{end - start} Handelstag(e) nachgelebt und gelernt.")
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="fly.daily", description="Tagessignal des Schwarms")
    p.add_argument("--init", type=Path, help="swarm.npz eines Zuchtlaufs als neuen Schwarm einsetzen")
    p.add_argument("--quorum", type=int, default=6, help="Stimmen von 10 für ein Signal")
    args = p.parse_args()
    if args.init:
        init(args.init, args.quorum)
    else:
        sys.exit(run())


if __name__ == "__main__":
    main()
