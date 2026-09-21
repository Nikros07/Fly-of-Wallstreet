"""
Kommandozeile.

    python -m fly evolve                    eine Zucht, Walk-Forward-Ergebnis
    python -m fly diagnose                  riecht eine einzelne Fliege überhaupt etwas?
    python -m fly lab --seeds 1 2 3         Zucht + beide Kontrollversuche, mehrere Seeds
    python -m fly evolve --friedhof         zusätzlich der Friedhofs-Test (sparsam benutzen!)
    python -m fly evolve --senses v3        neue Sinne (Kredit/Breite/Flucht) statt v2
    python -m fly evolve --since 2008-01-01 nur Tage ab diesem Datum (fairer Vergleich)

Der Friedhof (ab 2025) ist die einzige Prüfung, die die Evolution nie gesehen
hat. Jedes Mal, wenn man ihn anschaut und danach am Design dreht, wird er ein
Stück weniger unberührt. Deshalb ist er ein Flag und nicht der Standard.

`--senses` (oder Umgebungsvariable FLY_SENSES) wählt zwischen v2 (SPY/VIX/Zins,
ab 1993) und v3 (+Kredit/Breite/Flucht, Historie automatisch ab ~2007/2008).
Weil v3 eine kürzere Historie hat, sind v2- und v3-Ergebnisse nur mit `--since`
auf demselben Zeitraum fair vergleichbar.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fly.data import Market, load_closes
from fly.evolution import Config, Result, evolve, hatch
from fly.population import World, live, pnl, vote
from fly.report import stats, table
from fly.skull import Skull
from fly.senses import channels

CUTOFF = "2025-01-01"
SKULL_SEED = 0
RESULTS = Path(__file__).resolve().parent.parent / "results"


def build(refresh: bool, senses: str = "v2", since: str | None = None):
    market = Market(load_closes(refresh=refresh, senses=senses))
    if since:
        market = market.since(since)
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


def save(res: Result, world: World, extra: str = "", senses: str = "v2") -> Path:
    cfg = res.config
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}_{senses}_{cfg.fitness}_{cfg.control}_s{cfg.seed}"
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


def make_config(args, **over) -> Config:
    base = dict(population=args.population, survivors=args.survivors, quorum=args.quorum,
                exams=args.exams)
    base.update(over)
    return Config(**base)


def cmd_evolve(args, market, skull):
    cfg = make_config(args, control=args.control, fitness=args.fitness, seed=args.seed)
    print(f"Zucht: {cfg.population} Fliegen, {cfg.survivors} überleben, Fitness={cfg.fitness}, "
          f"Kontrolle={cfg.control}, Seed={cfg.seed}, Sinne={args.senses}")
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
    print(f"Gespeichert: {save(res, world, extra, senses=args.senses)}")


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
                save(res, world, senses=args.senses)
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


REGIMES = {
    "2000-02 Dotcom-Crash": ("2000", "2002"), "2003-07 Bullen": ("2003", "2007"),
    "2008-09 Finanzkrise": ("2008", "2009"), "2010-19 Bullen": ("2010", "2019"),
    "2020 Corona": ("2020", "2020"), "2021 Bullen": ("2021", "2021"),
    "2022 Zinsschock": ("2022", "2022"), "2023-24 Bullen": ("2023", "2024"),
    "2025- Friedhof": ("2025", "2099"),
}


def regimes(fly: pd.Series, spy: pd.Series) -> dict:
    """p.a.-Rendite von Fliege und SPY je Marktphase — wo verdient sie, wo verliert sie?"""
    rows = {}
    for name, (a, b) in REGIMES.items():
        f, m = fly.loc[a:b], spy.loc[a:b]
        if len(f) < 20:
            continue
        pa = lambda x: (1 + x).prod() ** (252 / len(x)) - 1
        rows[name] = {"Fliege": pa(f), "SPY": pa(m), "Diff": pa(f) - pa(m)}
    return rows


def cmd_colony(args, market, skull):
    """
    Wochenzucht (fly/colony.py): 50 Fliegen leben ohne Pause, jede Woche sterben
    die schlechtesten 10 % und die besten 10 % werden geklont. Die Kolonie lebt
    ab 1994; bewertet wird der Schwarm ab `--report-from` (Standard 2000).
    """
    from fly.colony import Colony, ColonyConfig
    from fly.population import MAX_HORIZON, realized
    world = World.build(market.until(CUTOFF), skull)
    T = len(world.days)
    # Vergleichsmaßstab mit derselben Ausführung: Eröffnung zu Eröffnung
    bench = pd.Series(np.nan_to_num(world.open_returns()), index=world.days)
    runs, flies = [], []

    def swarm_pnl(pos, w, t0, t1, past=None):
        pad = np.zeros((1, MAX_HORIZON), np.int8) if past is None else past[None, -MAX_HORIZON:]
        return realized(np.concatenate([pad, pos[None]], axis=1), t0 - MAX_HORIZON, t0, t1, w)[0]

    for control in args.controls:
        for seed in args.seeds:
            cfg = ColonyConfig(control=control, seed=seed, turnover=args.turnover,
                               window=args.window, quorum=args.quorum, exams=args.exams)
            col = Colony.found(cfg, skull.n_kc)
            learn = (world.with_shuffled_learning(np.random.default_rng(seed))
                     if control == "shuffled-dopamine" else None)
            started = time.time()
            swarm, _ = col.advance(world, 0, T, learn)
            daily = pd.Series(swarm_pnl(swarm, world, 0, T), index=world.days)
            pos = pd.Series(swarm, index=world.days)
            rep = slice(args.report_from, pd.Timestamp(CUTOFF) - pd.Timedelta(days=1))
            runs.append({"control": control, "seed": seed,
                         **stats(daily.loc[rep], pos.loc[rep])})
            print(f"… {control:18} Seed {seed}: Sharpe {runs[-1]['Sharpe']:+.2f}  "
                  f"p.a. {runs[-1]['p.a.']:+.1%}  ({time.time() - started:.0f}s)", flush=True)

            out = RESULTS / f"colony_w{args.window}_e{args.exams}_{control}_s{seed}"
            out.mkdir(parents=True, exist_ok=True)
            col.save(out / "colony.npz")
            pd.DataFrame(col.log).to_csv(out / "weeks.csv", index=False)
            pd.DataFrame({"swarm": daily, "position": pos}).to_csv(out / "walk_forward.csv")

            if control == "none":
                if args.friedhof:
                    full = World.build(market, skull)
                    t0, t1 = int(np.searchsorted(full.days, pd.Timestamp(CUTOFF))), len(full.days)
                    fs, _ = col.advance(full, t0, t1)
                    fdaily = pd.Series(swarm_pnl(fs, full, t0, t1, past=swarm), index=full.days[t0:t1])
                    daily = pd.concat([daily, fdaily])
                    bench = pd.Series(np.nan_to_num(full.open_returns()), index=full.days)
                    col.save(out / "colony_heute.npz")
                    runs[-1]["Friedhof Sharpe"] = stats(fdaily)["Sharpe"]
                    runs[-1]["Friedhof p.a."] = stats(fdaily)["p.a."]
                flies.append(daily)

    df = pd.DataFrame(runs)
    mean = df.groupby("control", sort=False).mean(numeric_only=True)
    summary = {c: {k: r[k] for k in ("p.a.", "Sharpe", "MaxDD", "investiert", "Wechsel")
                   if k in r} for c, r in mean.iterrows()}
    spy_rep = bench.loc[args.report_from:pd.Timestamp(CUTOFF) - pd.Timedelta(days=1)]
    summary["SPY Buy & Hold"] = {k: v for k, v in stats(spy_rep).items()
                                 if k in ("p.a.", "Sharpe", "MaxDD")}
    print(f"\nWochenzucht, Schwarm walk-forward ab {args.report_from} bis 2024, "
          f"Mittel über {len(args.seeds)} Seeds:\n")
    print(table(summary))
    if flies:
        avg = pd.concat(flies, axis=1).mean(axis=1)
        print("\nJe Marktphase (echte Zucht, Mittel über Seeds):\n")
        print(table(regimes(avg, bench.reindex(avg.index).fillna(0.0))))
    if args.friedhof:
        fh = df.dropna(subset=["Friedhof Sharpe"])
        spy_fh = stats(bench.loc[CUTOFF:])
        print(f"\nFRIEDHOF ab {CUTOFF}: Schwarm Sharpe {fh['Friedhof Sharpe'].mean():+.2f} "
              f"(p.a. {fh['Friedhof p.a.'].mean():+.1%}) — SPY Sharpe {spy_fh['Sharpe']:+.2f} "
              f"(p.a. {spy_fh['p.a.']:+.1%})")
    df.to_csv(RESULTS / f"colony_lab_w{args.window}_e{args.exams}.csv", index=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows-Konsole: Umlaute
    p = argparse.ArgumentParser(prog="fly", description="Fly of Wallstreet — Fliegenzucht für SPY")
    p.add_argument("--refresh", action="store_true", help="Kursdaten neu laden")
    p.add_argument("--senses", default=os.environ.get("FLY_SENSES", "v2"), choices=["v2", "v3"],
                    help="Sinnesumfang: v2 (SPY/VIX/Zins, ab 1993) oder v3 (+Kredit/Breite/Flucht, "
                         "Historie automatisch kürzer). Auch per Umgebungsvariable FLY_SENSES.")
    p.add_argument("--since", default=None,
                    help="nur Tage ab diesem Datum (z.B. 2008-01-01) — für einen fairen Vergleich "
                         "von v2- und v3-Sinnen auf demselben Zeitraum")
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
    c = sub.add_parser("colony", help="Wochenzucht: jede Woche 10 %% Tod, 10 %% Klone")
    c.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    c.add_argument("--controls", nargs="+", default=controls, choices=controls)
    c.add_argument("--turnover", type=float, default=0.10)
    c.add_argument("--window", type=int, default=126, help="Handelstage Bilanz")
    c.add_argument("--quorum", type=int, default=6)
    c.add_argument("--exams", type=int, default=0, help="Prüfungen auf älteren Zeiträumen je Woche")
    c.add_argument("--report-from", default="2000-01-01")
    c.add_argument("--friedhof", action="store_true")
    args = p.parse_args()
    print(f"Sinne={args.senses}" + (f", ab {args.since}" if args.since else "") + f", Friedhof-Cutoff={CUTOFF}")
    market, skull = build(args.refresh, args.senses, args.since)
    {"evolve": cmd_evolve, "lab": cmd_lab, "diagnose": cmd_diagnose, "colony": cmd_colony}[args.cmd](args, market, skull)


if __name__ == "__main__":
    main()
