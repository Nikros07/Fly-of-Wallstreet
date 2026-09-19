"""
Die Zucht: Auslese, Kreuzung, Mutation.

Ablauf pro Generation (= ein Kalenderquartal):
  1. LEBEN     Alle Fliegen erleben das neue Quartal, handeln und lernen.
               Das ist echte Vorhersage: entschieden wird vor dem Lernen.
  2. PRÜFEN    Dazu zwei zufällige FRÜHERE Quartale, ohne Lernen: Hält das
               Gelernte auch dort? Es zählt das SCHLECHTESTE der drei
               Ergebnisse — wer irgendwo durchfällt, stirbt.
  3. AUSLESE   Die besten `survivors` überleben mit ihrem Gedächtnis, der
               Rest wird gelöscht.
  4. NACHWUCHS Je zwei Überlebende zeugen ein Kind: Jedes Gen und jedes
               gelernte Muster (je Kenyon-Zelle) stammt zufällig von Mutter
               oder Vater. Danach mutieren die Gene.

Die Mutationsstärke ist selbst ein Gen: Zuchtlinien, bei denen große Sprünge
sich lohnen, behalten sie; andere werden von selbst ruhiger.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fly.population import Population, World, live, pnl, sharpe, vote

# Name: (untere Grenze, obere Grenze, logarithmisch?, Startbereich)
GENES: dict[str, tuple[float, float, bool, tuple[float, float]]] = {
    "lr":            (0.005, 0.5,  True,  (0.01, 0.2)),    # wie stark ein Dopamin-Stoß wirkt
    "forget":        (1e-4,  0.05, True,  (5e-4, 1e-2)),   # wie schnell Erinnerungen verblassen
    "loss_aversion": (1.0,   4.0,  False, (1.0, 3.0)),     # wie viel mehr Verlust schmerzt
    "miss_weight":   (0.0,   1.0,  False, (0.0, 0.4)),     # wie laut Verpasstes zählt ("leise")
    "thr_long":      (0.0,   0.5,  False, (0.0, 0.2)),     # Mut für Long
    "thr_short":     (0.0,   0.7,  False, (0.1, 0.5)),     # Mut für Short — gegen den Wind, also höher
    "horizon":       (1,     10,   False, (1, 10)),        # nach wie vielen Tagen abgerechnet wird
    "mut_scale":     (0.01,  0.5,  True,  (0.05, 0.2)),    # wie wild die Kinder mutieren
}
INTEGER_GENES = {"horizon"}


@dataclass
class Config:
    population: int = 50
    survivors: int = 10
    exams: int = 2
    quorum: int = 6                 # von 10 Schwarm-Fliegen
    control: str = "none"           # none | random-selection | shuffled-dopamine
    seed: int = 1


def hatch(n: int, n_kc: int, rng: np.random.Generator, first_id: int = 0) -> Population:
    """Frisch geschlüpfte Fliegen: zufällige Gene, leeres Gedächtnis."""
    genes = {}
    for name, (lo, hi, log, (a, b)) in GENES.items():
        if name in INTEGER_GENES:
            genes[name] = rng.integers(int(a), int(b) + 1, n).astype(float)
        elif log:
            genes[name] = np.exp(rng.uniform(np.log(a), np.log(b), n))
        else:
            genes[name] = rng.uniform(a, b, n)
    ones = np.ones((n, 2, n_kc), dtype=np.float32)
    return Population(genes, ones, ones.copy(), np.arange(first_id, first_id + n),
                      [None] * n)


def breed(parents: Population, n: int, rng: np.random.Generator, first_id: int) -> Population:
    ma = rng.integers(0, parents.size, n)
    pa = (ma + rng.integers(1, parents.size, n)) % parents.size   # nie mit sich selbst

    genes = {}
    for name in GENES:
        pick = rng.random(n) < 0.5
        genes[name] = np.where(pick, parents.genes[name][ma], parents.genes[name][pa])

    # Gedächtnis: je Kenyon-Zelle erbt das Kind das Muster komplett von einem
    # Elternteil — GO und NOGO beider Aktionen zusammen, damit eine
    # Erinnerung nicht halb von der Mutter und halb vom Vater stammt.
    n_kc = parents.go.shape[2]
    mask = (rng.random((n, 1, n_kc)) < 0.5)
    go = np.where(mask, parents.go[ma], parents.go[pa])
    nogo = np.where(mask, parents.nogo[ma], parents.nogo[pa])

    kids = Population(genes, go, nogo, np.arange(first_id, first_id + n),
                      [(int(parents.ids[m]), int(parents.ids[p])) for m, p in zip(ma, pa)])
    mutate(kids, rng)
    return kids


def mutate(pop: Population, rng: np.random.Generator) -> None:
    n = pop.size
    tau = 1.0 / np.sqrt(len(GENES))
    pop.genes["mut_scale"] *= np.exp(tau * rng.standard_normal(n))
    sigma = pop.genes["mut_scale"]
    for name, (lo, hi, log, _) in GENES.items():
        if name == "mut_scale":
            pass
        elif name in INTEGER_GENES:
            step = np.rint(rng.standard_normal(n) * sigma * (hi - lo))
            pop.genes[name] = pop.genes[name] + step
        elif log:
            pop.genes[name] = pop.genes[name] * np.exp(sigma * rng.standard_normal(n))
        else:
            pop.genes[name] = pop.genes[name] + sigma * (hi - lo) * rng.standard_normal(n)
        pop.genes[name] = np.clip(pop.genes[name], lo, hi)


@dataclass
class Result:
    survivors: Population
    generations: pd.DataFrame         # eine Zeile je Generation
    walk_forward: pd.Series           # Tages-PnL des Schwarms, Quartal für Quartal vorhergesagt
    walk_forward_positions: pd.Series
    config: Config


def evolve(world: World, cfg: Config, n_kc: int, max_generations: int | None = None,
           progress=None) -> Result:
    rng = np.random.default_rng(cfg.seed)
    learn_world = world.with_shuffled_learning(rng) if cfg.control == "shuffled-dopamine" else world

    pop = hatch(cfg.population, n_kc, rng)
    next_id = cfg.population
    quarters = world.quarters()
    if max_generations:
        quarters = quarters[:max_generations]

    rows, wf_pnl, wf_pos, wf_days = [], [], [], []
    for g, (t0, t1) in enumerate(quarters):
        positions = live(pop, learn_world, t0, t1, learn=True)
        live_score = sharpe(pnl(positions, world, t0, t1))

        # Walk-forward: Die Überlebenden der VORIGEN Generation stehen vorn in
        # der Population. Ihr Abstimmungsergebnis in diesem Quartal ist eine
        # echte Vorhersage — noch bevor dieses Quartal irgendwen ausliest.
        swarm_pos = vote(positions[:cfg.survivors], cfg.quorum) if g > 0 else \
            np.zeros(t1 - t0, dtype=np.int8)
        wf_pos.append(swarm_pos)
        wf_pnl.append(pnl(swarm_pos[None, :], world, t0, t1)[0])
        wf_days.append(world.days[t0:t1])

        scores = [live_score]
        if g > 0:
            for e in rng.choice(g, size=min(cfg.exams, g), replace=False):
                e0, e1 = quarters[e]
                exam_pos = live(pop, world, e0, e1, learn=False)
                scores.append(sharpe(pnl(exam_pos, world, e0, e1)))
        fitness = np.min(np.stack(scores), axis=0)

        if cfg.control == "random-selection":
            order = rng.permutation(pop.size)
        else:
            order = np.argsort(-fitness, kind="stable")
        keep = order[:cfg.survivors]

        rows.append({
            "generation": g,
            "quarter": str(world.days[t0].to_period("Q")),
            "best_fitness": float(fitness.max()),
            "median_fitness": float(np.median(fitness)),
            "survivor_fitness": float(fitness[keep].mean()),
            "exposure": float((positions[keep] != 0).mean()),
            "short_share": float((positions[keep] == -1).mean()),
            "swarm_exposure": float((swarm_pos != 0).mean()),
            **{f"gene_{k}": float(np.median(v[keep])) for k, v in pop.genes.items()},
        })
        if progress:
            progress(rows[-1])

        survivors = pop.take(keep)
        kids = breed(survivors, cfg.population - cfg.survivors, rng, next_id)
        next_id += kids.size
        pop = Population.concat(survivors, kids)

    days = pd.DatetimeIndex(np.concatenate(wf_days))
    return Result(
        survivors=pop.take(np.arange(cfg.survivors)),
        generations=pd.DataFrame(rows),
        walk_forward=pd.Series(np.concatenate(wf_pnl), index=days, name="swarm"),
        walk_forward_positions=pd.Series(np.concatenate(wf_pos), index=days, name="position"),
        config=cfg,
    )
