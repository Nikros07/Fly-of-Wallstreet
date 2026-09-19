"""
Kommandozeile.

    python -m fly evolve                    eine Zucht, Walk-Forward-Ergebnis
    python -m fly lab --seeds 1 2 3         Zucht + beide Kontrollversuche, mehrere Seeds
    python -m fly evolve --friedhof         zusätzlich der Friedhofs-Test (sparsam benutzen!)

Der Friedhof (ab 2025) ist die einzige Prüfung, die die Evolution nie gesehen
hat. Jedes Mal, wenn man ihn anschaut und danach am Design dreht, wird er ein
Stück weniger unberührt. Deshalb ist er ein Flag und nicht der Standard.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fly.data import Market, load_closes
from fly.evolution import Config, Result, evolve
from fly.population import World, live, pnl, vote
from fly.report import stats, table
from fly.skull import Skull
from fly.senses import channels

CUTOFF = "2025-01-01"
SKULL_SEED = 0
RESULTS = Path(__file__).resolve().parent.parent / "results"


def build(refresh: bool):
    market = Market(load_closes(refresh=refresh))
    n_channels = channels(market.closes.iloc[:400]).shape[1]
    skull = Skull.grow(n_channels, seed=SKULL_SEED)
    return market, skull


def buy_and_hold(world: World, days: pd.DatetimeIndex) -> pd.Series:
    ret = pd.Series(world.ret1, index=world.days).reindex(days).fillna(0.0)
    return ret.rename("SPY")


def run_one(market, skull, cfg: Config, generations: int | None, quiet: bool) -> tuple[Result, World]:
    world = World.build(market.until(CUTOFF), skull)   # der Friedhof existiert hier nicht
    started = time.time()

    def progress(row):
        if not quiet and row["generation"] % 10 == 0:
            print(f"  Gen {row['generation']:3d} {row['quarter']}  "
                  f"beste {row['best_fitness']:+.2f}  Überlebende {row['survivor_fitness']:+.2f}  "
                  f"investiert {row['exposure']:.0%}  Horizont {row['gene_horizon']:.0f}")

    res = evolve(world, cfg, skull.n_kc, generations, progress)
    if not quiet:
        print(f"  {len(res.generations)} Generationen in {time.time() - started:.0f}s")
    return res, world


def friedhof(market, skull, res: Result) -> pd.DataFrame:
    """Die Überlebenden leben weiter in einer Zeit, die die Evolution nie sah."""
    full = World.build(market, skull)
    t0 = int(np.searchsorted(full.days, pd.Timestamp(CUTOFF)))
    t1 = len(full.days) - 1                   # der letzte Tag hat noch kein Ergebnis
    flies = res.survivors.take(np.arange(res.survivors.size))
    positions = live(flies, full, t0, t1, learn=True)
    swarm = vote(positions, res.config.quorum)
    days = full.days[t0:t1]
    return pd.DataFrame({
        "swarm": pnl(swarm[None, :], full, t0, t1)[0],
        "position": swarm,
        "SPY": buy_and_hold(full, days).to_numpy(),
    }, index=days)


def save(res: Result, world: World, extra: str = "") -> Path:
    cfg = res.config
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.control}_s{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    res.generations.to_csv(out / "generations.csv", index=False)
    pd.DataFrame({"swarm": res.walk_forward, "position": res.walk_forward_positions}) \
        .to_csv(out / "walk_forward.csv")
    rows = {
        "Schwarm (walk-forward)": stats(res.walk_forward, res.walk_forward_positions),
        "SPY Buy & Hold": stats(buy_and_hold(world, res.walk_forward.index)),
    }
    (out / "report.md").write_text(
        f"# Zucht {cfg.control}, Seed {cfg.seed}\n\n{table(rows)}\n{extra}", encoding="utf-8")
    return out


def cmd_evolve(args, market, skull):
    cfg = Config(population=args.population, survivors=args.survivors,
                 quorum=args.quorum, control=args.control, seed=args.seed)
    print(f"Zucht: {cfg.population} Fliegen, {cfg.survivors} überleben, "
          f"Kontrolle={cfg.control}, Seed={cfg.seed}")
    res, world = run_one(market, skull, cfg, args.generations, quiet=False)
    wf = res.walk_forward
    rows = {"Schwarm (walk-forward)": stats(wf, res.walk_forward_positions),
            "SPY Buy & Hold": stats(buy_and_hold(world, wf.index))}
    print(f"\nWalk-forward {wf.index[0]:%Y-%m} bis {wf.index[-1]:%Y-%m} "
          "(jedes Quartal vorhergesagt, bevor es ausgelesen wurde):\n")
    print(table(rows))

    extra = ""
    if args.friedhof:
        fh = friedhof(market, skull, res)
        rows_fh = {"Schwarm (Friedhof)": stats(fh["swarm"], fh["position"]),
                   "SPY Buy & Hold": stats(fh["SPY"])}
        extra = f"\n## Friedhof ab {CUTOFF}\n\n{table(rows_fh)}\n"
        print(f"\nFRIEDHOF ab {CUTOFF} — nie von der Evolution gesehen:\n")
        print(table(rows_fh))
    print(f"\nGespeichert: {save(res, world, extra)}")


def cmd_lab(args, market, skull):
    """Echte Zucht gegen beide Kontrollversuche. Nur wenn sie klar vorn liegt, lernt sie etwas."""
    rows = {}
    bench = None
    for seed in args.seeds:
        for control in ("none", "random-selection", "shuffled-dopamine"):
            cfg = Config(population=args.population, survivors=args.survivors,
                         quorum=args.quorum, control=control, seed=seed)
            print(f"… {control}, Seed {seed}")
            res, world = run_one(market, skull, cfg, args.generations, quiet=True)
            save(res, world)
            rows[f"{control} s{seed}"] = stats(res.walk_forward, res.walk_forward_positions)
            bench = stats(buy_and_hold(world, res.walk_forward.index))
    rows["SPY Buy & Hold"] = bench
    print("\nWalk-forward, alle Läufe:\n")
    print(table(rows))


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows-Konsole: Umlaute
    p = argparse.ArgumentParser(prog="fly", description="Fly of Wallstreet — Fliegenzucht für SPY")
    p.add_argument("--refresh", action="store_true", help="Kursdaten neu laden")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("evolve", "lab"):
        s = sub.add_parser(name)
        s.add_argument("--population", type=int, default=50)
        s.add_argument("--survivors", type=int, default=10)
        s.add_argument("--quorum", type=int, default=6)
        s.add_argument("--generations", type=int, default=None, help="nur die ersten N Quartale")
        if name == "evolve":
            s.add_argument("--seed", type=int, default=1)
            s.add_argument("--control", default="none",
                           choices=["none", "random-selection", "shuffled-dopamine"])
            s.add_argument("--friedhof", action="store_true")
        else:
            s.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()
    market, skull = build(args.refresh)
    {"evolve": cmd_evolve, "lab": cmd_lab}[args.cmd](args, market, skull)


if __name__ == "__main__":
    main()
