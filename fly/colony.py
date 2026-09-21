"""
Die Kolonie: eine Fliegenzucht, die ohne Pause weiterlebt.

Statt Generationen im Quartalstakt lebt eine feste Population von 50 Fliegen
Tag für Tag. Jede Woche, bevor der erste Handelstag der neuen Woche beginnt:

    1. BILANZ    Jede Fliege wird an ihren eigenen Vorhersagen der letzten
                 26 Wochen gemessen (Überschuss gegen Buy & Hold, risikobereinigt).
                 Das sind echte Vorhersagen: entschieden wurde immer vor dem Lernen.
    2. TOD       Die schlechtesten 10 % sterben.
    3. KLON      Die besten 10 % werden geklont — mit Gedächtnis UND Bilanz.
                 Nur die Gene mutieren. Die geerbte Bilanz wird Tag für Tag durch
                 die eigene ersetzt; so braucht kein Neugeborenes Welpenschutz,
                 und niemand wird nach einer einzigen Glückswoche befördert.
    4. SCHWARM   Die 10 Besten (vor dem Klonen, also ohne Doppelgänger) stimmen
                 die ganze nächste Woche über ab.

Backtest und Live-Betrieb benutzen exakt dieselbe `advance()`-Funktion. Ob man
die Kolonie ein Jahr am Stück oder täglich einen Tag weiterleben lässt, ergibt
bitgenau dasselbe (Test: test_colony_same_result_in_one_go_or_day_by_day).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fly.evolution import NOISE_FLOOR, hatch, mutate
from fly.population import MAX_HORIZON, Population, World, live, realized, vote


@dataclass
class ColonyConfig:
    population: int = 50
    turnover: float = 0.10       # Anteil, der jede Woche stirbt bzw. geklont wird
    window: int = 126            # Handelstage Bilanz (≈ 26 Wochen)
    min_days: int = 20           # kürzere Bilanz: weder Tod noch Klon noch Schwarm
    swarm: int = 10
    quorum: int = 6
    control: str = "none"        # none | random-selection | shuffled-dopamine
    exams: int = 0               # Prüfungen auf zufälligen ÄLTEREN Zeiträumen je Woche
    exam_len: int = 63           # Handelstage je Prüfung (≈ ein Quartal)
    seed: int = 1


@dataclass
class Colony:
    cfg: ColonyConfig
    pop: Population
    record: np.ndarray           # (P, window) tägliche Überschussrendite, neueste rechts
    count: np.ndarray            # (P,) wie viele Tage davon echt sind
    past: np.ndarray             # (P, MAX_HORIZON) letzte Positionen, für Lernen und Kosten
    swarm_idx: np.ndarray        # wer diese Woche abstimmt (leer = NO TRADE)
    week: pd.Period | None
    next_id: int
    rng: np.random.Generator
    log: list = field(default_factory=list)

    @classmethod
    def found(cls, cfg: ColonyConfig, n_kc: int) -> "Colony":
        rng = np.random.default_rng(cfg.seed)
        P = cfg.population
        return cls(cfg, hatch(P, n_kc, rng), np.zeros((P, cfg.window)), np.zeros(P, int),
                   np.zeros((P, MAX_HORIZON), np.int8), np.array([], int), None, P, rng)

    # ── Leben ───────────────────────────────────────────────────────────────

    def advance(self, world: World, t0: int, t1: int, learn_world: World | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Lebt die Tage [t0, t1). Rückgabe: Schwarm-Position je Tag (n,) und die
        Positionen aller Fliegen (P, n). Wochenwechsel lösen vorher die Auslese aus.
        `learn_world` ersetzt nur das Lernsignal (Kontrollversuch), gehandelt und
        bilanziert wird immer in `world`.
        """
        learn_world = learn_world or world
        weeks = world.days[t0:t1].to_period("W-FRI")
        swarm_pos = np.zeros(t1 - t0, np.int8)
        all_pos = np.zeros((self.pop.size, t1 - t0), np.int8)

        start = t0
        while start < t1:
            wk = weeks[start - t0]
            if self.week is not None and wk != self.week:
                self._select(world, start)
            self.week = wk
            end = start
            while end < t1 and weeks[end - t0] == wk:
                end += 1
            pos = live(self.pop, learn_world, start, end, past=self.past)
            full = np.concatenate([self.past, pos], axis=1)
            daily = realized(full, start - MAX_HORIZON, start, end, world)
            excess = daily - np.nan_to_num(world.open_returns()[start:end])[None, :]
            self._book(excess)
            self.past = np.concatenate([self.past, pos], axis=1)[:, -MAX_HORIZON:]
            if len(self.swarm_idx) >= self.cfg.quorum:
                swarm_pos[start - t0:end - t0] = vote(pos[self.swarm_idx], self.cfg.quorum)
            all_pos[:, start - t0:end - t0] = pos
            start = end
        return swarm_pos, all_pos

    def _book(self, excess: np.ndarray) -> None:
        n = excess.shape[1]
        self.record = np.concatenate([self.record, excess], axis=1)[:, -self.cfg.window:]
        self.count = np.minimum(self.count + n, self.cfg.window)

    # ── Auslese ─────────────────────────────────────────────────────────────

    def fitness(self, exam_excess: np.ndarray | None = None) -> np.ndarray:
        """
        Risikobereinigter Überschuss über die echte Bilanz, auf Wunsch zusammen
        mit Prüfungstagen (P, E) aus älteren Zeiträumen. Zu kurze Bilanz = NaN.
        """
        W = self.cfg.window
        mask = np.arange(W)[None, :] >= (W - self.count)[:, None]
        rec = self.record
        if exam_excess is not None and exam_excess.size:
            rec = np.concatenate([rec, exam_excess], axis=1)
            mask = np.concatenate([mask, np.ones_like(exam_excess, bool)], axis=1)
        n = np.maximum(mask.sum(axis=1), 1)
        mean = np.where(mask, rec, 0.0).sum(axis=1) / n
        sd = np.sqrt(np.where(mask, (rec - mean[:, None]) ** 2, 0.0).sum(axis=1) / n)
        f = mean / (sd + NOISE_FLOOR) * np.sqrt(252)
        return np.where(self.count >= self.cfg.min_days, f, np.nan)

    def _exams(self, world: World, now: int) -> np.ndarray | None:
        """
        Die Fliegen handeln, ohne zu lernen, in zufälligen früheren Zeiträumen
        (nur Tage vor `now`).

        Ehrlich benannt: Das ist KEINE Prüfung außerhalb der Stichprobe. Das
        Gedächtnis hat die Ergebnisse dieser Tage längst gelernt; die Prüfung
        misst, wie viel altes Wissen eine Fliege noch trägt — sie begünstigt
        langsames Vergessen. Die Walk-forward-Entscheidungen bleiben davon
        unberührt echte Vorhersagen (Test: test_colony_decisions_do_not_depend…).
        """
        L = self.cfg.exam_len
        if self.cfg.exams == 0 or now - L - MAX_HORIZON <= 0:
            return None
        out = []
        for e0 in self.rng.integers(0, now - L, self.cfg.exams):
            e0 = int(e0)
            pos = live(self.pop, world, e0, e0 + L, learn=False)
            full = np.concatenate([np.zeros((pos.shape[0], MAX_HORIZON), np.int8), pos], axis=1)
            out.append(realized(full, e0 - MAX_HORIZON, e0, e0 + L, world)
                       - np.nan_to_num(world.open_returns()[e0:e0 + L])[None, :])
        return np.concatenate(out, axis=1)

    def _select(self, world: World, now: int) -> None:
        cfg, rng = self.cfg, self.rng
        fit = self.fitness(self._exams(world, now))
        adults = np.flatnonzero(np.isfinite(fit))
        if cfg.control == "random-selection":
            ranked = rng.permutation(adults)
        else:
            ranked = adults[np.argsort(-fit[adults], kind="stable")]

        self.swarm_idx = ranked[:cfg.swarm] if len(ranked) >= cfg.swarm else np.array([], int)
        k = int(round(cfg.turnover * cfg.population))
        if len(ranked) < 2 * k or k == 0:
            self.log.append({"week": str(self.week), "adults": len(adults), "died": 0})
            return
        best, worst = ranked[:k], ranked[-k:]

        clones = self.pop.take(best)
        mutate(clones, rng)
        clones.ids = np.arange(self.next_id, self.next_id + k)
        clones.parents = [(int(i),) for i in self.pop.ids[best]]
        self.next_id += k
        for j, (dead, parent) in enumerate(zip(worst, best)):
            for name in self.pop.genes:
                self.pop.genes[name][dead] = clones.genes[name][j]
            self.pop.go[dead] = clones.go[j]
            self.pop.nogo[dead] = clones.nogo[j]
            self.pop.ids[dead] = clones.ids[j]
            self.pop.parents[dead] = clones.parents[j]
            self.record[dead] = self.record[parent]
            self.count[dead] = self.count[parent]
            self.past[dead] = self.past[parent]

        self.log.append({
            "week": str(self.week), "adults": len(adults), "died": k,
            "best": float(fit[best[0]]), "median": float(np.median(fit[adults])),
            **{f"gene_{g}": float(np.median(v)) for g, v in self.pop.genes.items()},
        })

    # ── Speichern (für den Live-Betrieb) ────────────────────────────────────

    def save(self, path) -> None:
        pop = self.pop
        np.savez_compressed(
            path, go=pop.go, nogo=pop.nogo,
            ids=pop.ids, record=self.record, count=self.count,
            past=self.past, swarm_idx=self.swarm_idx, next_id=self.next_id,
            week=str(self.week) if self.week is not None else "",
            rng=json.dumps(self.rng.bit_generator.state),
            **{f"gene_{k}": v for k, v in pop.genes.items()},
            **{f"cfg_{k}": v for k, v in vars(self.cfg).items()})

    @classmethod
    def load(cls, path) -> "Colony":
        z = np.load(path, allow_pickle=False)
        cfg = ColonyConfig(**{k[4:]: z[k].item() for k in z.files if k.startswith("cfg_")})
        genes = {k[5:]: z[k] for k in z.files if k.startswith("gene_")}
        pop = Population(genes, z["go"].astype(np.float32), z["nogo"].astype(np.float32),
                         z["ids"], [None] * len(z["ids"]))
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(z["rng"]))
        week = str(z["week"])
        return cls(cfg, pop, z["record"].astype(float), z["count"], z["past"], z["swarm_idx"],
                   pd.Period(week, "W-FRI") if week else None, int(z["next_id"]), rng)
