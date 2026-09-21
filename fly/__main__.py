"""
Kommandozeile.

    python -m fly evolve                    eine Zucht, Walk-Forward-Ergebnis
    python -m fly diagnose                  riecht eine einzelne Fliege überhaupt etwas?
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

from fly.data import Market, load_closes, load_sector_closes
from fly.evolution import Config, Result, evolve, evolve_multi, hatch
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


def build_multi(refresh: bool):
    """Neun Sektor-ETFs statt SPY — derselbe Schädel, dieselben Sinneskanäle je Markt."""
    sector_closes = load_sector_closes(refresh=refresh)
    markets = {name: Market(df) for name, df in sector_closes.items()}
    any_df = next(iter(sector_closes.values()))
    n_channels = channels(any_df.iloc[:400]).shape[1]
    skull = Skull.grow(n_channels, seed=SKULL_SEED)
    return markets, skull


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


def run_one_multi(markets: dict, skull, cfg: Config, generations: int | None, quiet: bool):
    """Wie run_one(), aber die Fliege riecht jeden Tag alle neun Sektor-ETFs."""
    names = list(markets.keys())
    worlds = [World.build(markets[name].until(CUTOFF), skull) for name in names]
    started = time.time()

    def progress(row):
        if not quiet and row["generation"] % 10 == 0:
            print(f"  Gen {row['generation']:3d} {row['quarter']}  "
                  f"beste {row['best_fitness']:+.2f}  Überlebende {row['survivor_fitness']:+.2f}  "
                  f"investiert {row['exposure']:.0%}  Horizont {row['gene_horizon']:.0f}")

    res = evolve_multi(worlds, cfg, skull.n_kc, generations, progress, names=names)
    if not quiet:
        print(f"  {len(res.generations)} Generationen in {time.time() - started:.0f}s")
    return res, worlds, names


def buy_and_hold_multi(worlds: list[World], days: pd.DatetimeIndex) -> pd.Series:
    """Gleichgewichtetes Buy & Hold über alle neun Sektor-ETFs — derselbe Vergleichsmaßstab
    wie die Fliegen-Portfolio-PnL (Mittelwert der Einzelmärkte)."""
    rets = np.mean([pd.Series(w.ret1, index=w.days).reindex(days).fillna(0.0).to_numpy()
                    for w in worlds], axis=0)
    return pd.Series(rets, index=days, name="Sektor-ETFs")


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
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.fitness}_{cfg.control}_s{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    res.generations.to_csv(out / "generations.csv", index=False)
    sv = res.survivors
    pd.DataFrame({"id": sv.ids, "eltern": [p if p else "Urfliege" for p in sv.parents],
                  **{k: np.round(v, 5) for k, v in sv.genes.items()}}).to_csv(
        out / "survivors.csv", index=False)
    np.savez_compressed(out / "swarm.npz", go=sv.go, nogo=sv.nogo, ids=sv.ids,
                        **{f"gene_{k}": v for k, v in sv.genes.items()})
    pd.DataFrame({"swarm": res.walk_forward, "position": res.walk_forward_positions}) \
        .to_csv(out / "walk_forward.csv")
    rows = {
        "Schwarm (walk-forward)": stats(res.walk_forward, res.walk_forward_positions),
        "SPY Buy & Hold": stats(buy_and_hold(world, res.walk_forward.index)),
    }
    (out / "report.md").write_text(
        f"# Zucht {cfg.fitness} / {cfg.control}, Seed {cfg.seed}\n\n{table(rows)}\n{extra}", encoding="utf-8")
    return out


def save_multi(res: Result, worlds: list[World], names: list[str], extra: str = "") -> Path:
    """Wie save(), aber für die Mehrmarkt-Zucht — der Benchmark ist das gleichgewichtete
    Sektor-ETF-Portfolio, nicht mehr SPY allein."""
    cfg = res.config
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}_multi_{cfg.fitness}_{cfg.control}_s{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    res.generations.to_csv(out / "generations.csv", index=False)
    sv = res.survivors
    pd.DataFrame({"id": sv.ids, "eltern": [p if p else "Urfliege" for p in sv.parents],
                  **{k: np.round(v, 5) for k, v in sv.genes.items()}}).to_csv(
        out / "survivors.csv", index=False)
    np.savez_compressed(out / "swarm.npz", go=sv.go, nogo=sv.nogo, ids=sv.ids,
                        **{f"gene_{k}": v for k, v in sv.genes.items()})
    res.walk_forward.to_frame("swarm").join(res.walk_forward_positions).to_csv(out / "walk_forward.csv")
    rows = {
        "Schwarm (walk-forward, Portfolio)": stats(res.walk_forward),
        "Sektor-ETFs Buy & Hold (gleichgewichtet)": stats(buy_and_hold_multi(worlds, res.walk_forward.index)),
    }
    (out / "report.md").write_text(
        f"# Mehrmarkt-Zucht {cfg.fitness} / {cfg.control}, Seed {cfg.seed}\n"
        f"Märkte: {', '.join(n.upper() for n in names)}\n\n{table(rows)}\n{extra}", encoding="utf-8")
    return out


def make_config(args, **over) -> Config:
    base = dict(population=args.population, survivors=args.survivors, quorum=args.quorum,
                exams=args.exams)
    base.update(over)
    return Config(**base)


def cmd_evolve(args, market, skull):
    cfg = make_config(args, control=args.control, fitness=args.fitness, seed=args.seed)
    print(f"Zucht: {cfg.population} Fliegen, {cfg.survivors} überleben, Fitness={cfg.fitness}, "
          f"Kontrolle={cfg.control}, Seed={cfg.seed}")
    res, world = run_one(market, skull, cfg, args.generations, quiet=False)
    wf = res.walk_forward
    rows = {"Schwarm (walk-forward)": stats(wf, res.walk_forward_positions),
            "SPY Buy & Hold": stats(buy_and_hold(world, wf.index))}
    print()
    print(f"Walk-forward {wf.index[0]:%Y-%m} bis {wf.index[-1]:%Y-%m} "
          "(jedes Quartal vorhergesagt, bevor es ausgelesen wurde):")
    print()
    print(table(rows))

    extra = ""
    if args.friedhof:
        fh = friedhof(market, skull, res)
        rows_fh = {"Schwarm (Friedhof)": stats(fh["swarm"], fh["position"]),
                   "SPY Buy & Hold": stats(fh["SPY"])}
        extra = f"\n## Friedhof ab {CUTOFF}\n\n{table(rows_fh)}\n"
        print()
        print(f"FRIEDHOF ab {CUTOFF} — nie von der Evolution gesehen:")
        print()
        print(table(rows_fh))
    print()
    print(f"Gespeichert: {save(res, world, extra)}")


def cmd_lab(args, market, skull):
    """
    Echte Zucht gegen die Kontrollversuche, je Fitness-Modus und Seed.
    Eine Variante lernt nur dann etwas, wenn sie ihre eigene Kontrolle über
    ALLE Seeds hinweg schlägt — ein einzelner guter Seed ist Anekdote.
    """
    runs, bench = [], None
    for fitness in args.fitness:
        for control in args.controls:
            for seed in args.seeds:
                cfg = make_config(args, control=control, fitness=fitness, seed=seed)
                print(f"… Fitness={fitness}, Kontrolle={control}, Seed {seed}", flush=True)
                res, world = run_one(market, skull, cfg, args.generations, quiet=True)
                save(res, world)
                runs.append({"fitness": fitness, "control": control, "seed": seed,
                             **stats(res.walk_forward, res.walk_forward_positions)})
                bench = stats(buy_and_hold(world, res.walk_forward.index))

    df = pd.DataFrame(runs)
    metrics = [c for c in df.columns if c not in ("fitness", "control", "seed")]
    rows = {f"{r['fitness']} / {r['control']} / s{r['seed']}": {k: r[k] for k in metrics}
            for r in df.to_dict("records")}
    rows["SPY Buy & Hold"] = bench
    print()
    print("Walk-forward, alle Läufe:")
    print()
    print(table(rows))

    mean = df.groupby(["fitness", "control"], sort=False)[["p.a.", "Sharpe", "MaxDD", "investiert"]].mean()
    summary = {f"{f} / {c}": r.to_dict() for (f, c), r in mean.iterrows()}
    summary["SPY Buy & Hold"] = {k: bench[k] for k in ("p.a.", "Sharpe", "MaxDD")}
    print()
    print("Mittel über Seeds:")
    print()
    print(table(summary))
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"lab_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)


def cmd_evolve_multi(args, markets, skull):
    """Wie cmd_evolve(), aber die Fliege riecht jeden Tag alle neun Sektor-ETFs
    statt nur SPY (ein Gehirn, ein Gedächtnis, 9 Lernschritte pro Tag)."""
    cfg = make_config(args, control=args.control, fitness=args.fitness, seed=args.seed)
    print(f"Zucht (Mehrmarkt, {len(markets)} Sektor-ETFs): {cfg.population} Fliegen, "
          f"{cfg.survivors} überleben, Fitness={cfg.fitness}, Kontrolle={cfg.control}, Seed={cfg.seed}")
    res, worlds, names = run_one_multi(markets, skull, cfg, args.generations, quiet=False)
    wf = res.walk_forward
    rows = {"Schwarm (walk-forward, Portfolio)": stats(wf),
            "Sektor-ETFs Buy & Hold (gleichgewichtet)": stats(buy_and_hold_multi(worlds, wf.index))}
    print()
    print(f"Walk-forward {wf.index[0]:%Y-%m} bis {wf.index[-1]:%Y-%m} "
          "(jedes Quartal vorhergesagt, bevor es ausgelesen wurde):")
    print()
    print(table(rows))
    print()
    print(f"Gespeichert: {save_multi(res, worlds, names)}")


def cmd_lab_multi(args, markets, skull):
    """Wie cmd_lab(), aber in der Mehrmarkt-Variante (9 Sektor-ETFs, ein Gehirn)."""
    runs, bench = [], None
    for fitness in args.fitness:
        for control in args.controls:
            for seed in args.seeds:
                cfg = make_config(args, control=control, fitness=fitness, seed=seed)
                print(f"… Mehrmarkt Fitness={fitness}, Kontrolle={control}, Seed {seed}", flush=True)
                res, worlds, names = run_one_multi(markets, skull, cfg, args.generations, quiet=True)
                save_multi(res, worlds, names)
                runs.append({"fitness": fitness, "control": control, "seed": seed,
                             **stats(res.walk_forward)})
                bench = stats(buy_and_hold_multi(worlds, res.walk_forward.index))

    df = pd.DataFrame(runs)
    metrics = [c for c in df.columns if c not in ("fitness", "control", "seed")]
    rows = {f"{r['fitness']} / {r['control']} / s{r['seed']}": {k: r[k] for k in metrics}
            for r in df.to_dict("records")}
    rows["Sektor-ETFs Buy & Hold"] = bench
    print()
    print("Walk-forward, alle Läufe (Mehrmarkt):")
    print()
    print(table(rows))

    mean = df.groupby(["fitness", "control"], sort=False)[["p.a.", "Sharpe", "MaxDD"]].mean()
    summary = {f"{f} / {c}": r.to_dict() for (f, c), r in mean.iterrows()}
    summary["Sektor-ETFs Buy & Hold"] = {k: bench[k] for k in ("p.a.", "Sharpe", "MaxDD")}
    print()
    print("Mittel über Seeds (Mehrmarkt):")
    print()
    print(table(summary))
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"lab_multi_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)


def cmd_diagnose(args, market, skull):
    """
    Riecht eine einzelne Fliege überhaupt etwas? Ohne Evolution, über die
    ganze Zeit vor dem Friedhof: Wie stark hängt ihr Long-Gefühl mit der
    späteren Rendite zusammen (IC = Rangkorrelation), echt gegen gemischt?
    Liegt "echt" nicht klar über "gemischt", kann keine Zucht der Welt helfen.
    """
    world = World.build(market.until(CUTOFF), skull)
    T = len(world.days)
    grid = [(lr, fg, h) for lr in (0.02, 0.1) for fg in (1e-4, 1e-3) for h in (1, 5, 10)]
    warmup = 250
    print(f"{'':10}{'lr':>6}{'forget':>8}{'h':>4}{'IC':>8}")
    for name, w in (("echt", world),
                    ("gemischt", world.with_shuffled_learning(np.random.default_rng(0)))):
        pop = hatch(len(grid), skull.n_kc, np.random.default_rng(1))
        pop.genes["lr"][:] = [g[0] for g in grid]
        pop.genes["forget"][:] = [g[1] for g in grid]
        pop.genes["horizon"][:] = [g[2] for g in grid]
        pop.genes["miss_weight"][:] = 1.0
        vals = np.zeros((pop.size, T, 2))
        live(pop, w, 0, T, values_out=vals)
        for i, (lr, fg, h) in enumerate(grid):
            y = world.fwd[:, h]
            ok = np.isfinite(y) & (np.arange(T) >= warmup)
            ic = pd.Series(vals[i, ok, 0]).rank().corr(pd.Series(y[ok]).rank())
            print(f"{name:10}{lr:>6}{fg:>8.0e}{h:>4}{ic:>+8.3f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows-Konsole: Umlaute
    p = argparse.ArgumentParser(prog="fly", description="Fly of Wallstreet — Fliegenzucht für SPY")
    p.add_argument("--refresh", action="store_true", help="Kursdaten neu laden")
    sub = p.add_subparsers(dest="cmd", required=True)
    fitness_modes = ["worst", "pooled", "excess"]
    controls = ["none", "random-selection", "shuffled-dopamine"]
    for name in ("evolve", "lab"):
        s = sub.add_parser(name)
        s.add_argument("--population", type=int, default=50)
        s.add_argument("--survivors", type=int, default=10)
        s.add_argument("--quorum", type=int, default=6)
        s.add_argument("--exams", type=int, default=2, help="frühere Quartale als Prüfung")
        s.add_argument("--generations", type=int, default=None, help="nur die ersten N Quartale")
        s.add_argument("--multi", action="store_true",
                       help="9 Sektor-ETFs (XLK…XLB) statt SPY — ein Gehirn, 9 Lernschritte/Tag")
        if name == "evolve":
            s.add_argument("--seed", type=int, default=1)
            s.add_argument("--control", default="none", choices=controls)
            s.add_argument("--fitness", default="excess", choices=fitness_modes)
            s.add_argument("--friedhof", action="store_true")
        else:
            s.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
            s.add_argument("--fitness", nargs="+", default=["excess"], choices=fitness_modes)
            s.add_argument("--controls", nargs="+", default=controls, choices=controls)
    sub.add_parser("diagnose")
    args = p.parse_args()

    if getattr(args, "multi", False):
        if getattr(args, "friedhof", False):
            sys.exit("--multi und --friedhof zusammen sind nicht implementiert — "
                     "der Friedhof bleibt für die Mehrmarkt-Variante unberührt.")
        markets, skull = build_multi(args.refresh)
        {"evolve": cmd_evolve_multi, "lab": cmd_lab_multi}[args.cmd](args, markets, skull)
    else:
        market, skull = build(args.refresh)
        {"evolve": cmd_evolve, "lab": cmd_lab, "diagnose": cmd_diagnose}[args.cmd](args, market, skull)


if __name__ == "__main__":
    main()
