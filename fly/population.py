"""
Die lernende Hälfte des Pilzkörpers, für eine ganze Population auf einmal.

Jede Fliege hat pro Aktion (0 = Long, 1 = Short) zwei Ausgangsneuronen:
  GO     "dieses Muster führt zu Gutem"
  NOGO   "dieses Muster führt zu Schlechtem"
Der Wert einer Aktion ist GO minus NOGO, gemittelt über die feuernden
Kenyon-Zellen — also zwischen -1 und +1.

Lernen wie in der echten Fliege, nur durch ABSCHWÄCHEN:
  * Belohnung schwächt die NOGO-Synapsen der gerade aktiven Kenyon-Zellen
  * Bestrafung schwächt ihre GO-Synapsen
  * alle Synapsen erholen sich langsam Richtung 1 (Vergessen)
Gewichte bleiben dadurch immer in [0, 1]; nichts kann explodieren.

NO TRADE ist der Ruhezustand: gehandelt wird nur, wenn ein Wert die
Handelsschwelle (ein Gen) übersteigt. Eine frisch geschlüpfte Fliege hat
überall GO = NOGO = 1, also Wert 0, und handelt nicht.

Zeitdisziplin: Am Tag t entscheidet die Fliege ZUERST und lernt DANACH — und
zwar nur aus einem Ergebnis, das zum Schluss von Tag t vollständig bekannt ist
(Muster von Tag t-h, Rendite von t-h bis t). Jede Handelsentscheidung ist
damit eine echte Vorhersage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fly.data import Market
from fly.dopamine import dopamine
from fly.senses import channels, raw_features
from fly.skull import Skull

LONG, SHORT = 0, 1
MAX_HORIZON = 10
COST = 0.0005            # 5 Basispunkte je Einheit Positionswechsel (Gebühr + Spread)
BORROW = 0.01 / 252      # Leihgebühr pro Short-Tag (1 % p.a.)


@dataclass(frozen=True)
class World:
    """Alles, was die Fliegen erleben können — vorberechnet, für alle gleich."""
    days: pd.DatetimeIndex
    kc: np.ndarray        # (T, active) feuernde Kenyon-Zellen je Tag
    ret1: np.ndarray      # (T,) Rendite Schluss t -> Schluss t+1 (Handelsergebnis)
    fwd: np.ndarray       # (T, MAX_HORIZON+1) log-Rendite t -> t+h (Lernsignal)
    vol: np.ndarray       # (T,) tägliche Schwankung, bekannt zum Schluss von t

    @classmethod
    def build(cls, market: Market, skull: Skull) -> "World":
        ch = channels(market.closes)
        days = ch.index
        logp = np.log(market.closes["spy"])
        fwd = np.stack([(logp.shift(-h) - logp).reindex(days).to_numpy()
                        for h in range(MAX_HORIZON + 1)], axis=1)
        vol = raw_features(market.closes)["vol_20"].reindex(days).to_numpy()
        return cls(days=days,
                   kc=skull.smell(ch.to_numpy()),
                   ret1=market.next_day_returns().reindex(days).to_numpy(),
                   fwd=fwd,
                   vol=vol)

    def with_shuffled_learning(self, rng: np.random.Generator) -> "World":
        """
        Kontrollversuch: Die Fliegen handeln in der echten Welt, bekommen aber
        Dopamin für die Ergebnisse zufälliger ANDERER Tage. Wer so genauso gut
        abschneidet, hat nie etwas gelernt.
        """
        perm = rng.permutation(len(self.days))
        return World(self.days, self.kc, self.ret1, self.fwd[perm], self.vol[perm])

    def quarters(self) -> list[tuple[int, int]]:
        """Tagesindex-Spannen [t0, t1) je Kalenderquartal."""
        q = self.days.to_period("Q")
        edges = np.flatnonzero(np.r_[True, q[1:] != q[:-1], True])
        return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


@dataclass
class Population:
    genes: dict[str, np.ndarray]     # Gen-Name -> Wert je Fliege, Form (P,)
    go: np.ndarray                   # (P, 2, n_kc) float32
    nogo: np.ndarray                 # (P, 2, n_kc) float32
    ids: np.ndarray                  # fortlaufende Fliegen-Nummer, für den Stammbaum
    parents: list = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.ids)

    def take(self, idx: np.ndarray) -> "Population":
        return Population({k: v[idx].copy() for k, v in self.genes.items()},
                          self.go[idx].copy(), self.nogo[idx].copy(),
                          self.ids[idx].copy(), [self.parents[i] for i in idx])

    @staticmethod
    def concat(a: "Population", b: "Population") -> "Population":
        return Population({k: np.concatenate([a.genes[k], b.genes[k]]) for k in a.genes},
                          np.concatenate([a.go, b.go]), np.concatenate([a.nogo, b.nogo]),
                          np.concatenate([a.ids, b.ids]), a.parents + b.parents)

    def values(self, active: np.ndarray) -> np.ndarray:
        """Wert von Long und Short für ein Feuermuster, Form (P, 2)."""
        return (self.go[:, :, active] - self.nogo[:, :, active]).mean(axis=2)

    def decide(self, active: np.ndarray) -> np.ndarray:
        v = self.values(active)
        vl, vs = v[:, LONG], v[:, SHORT]
        go_long = (vl > self.genes["thr_long"]) & (vl >= vs)
        go_short = (vs > self.genes["thr_short"]) & (vs > vl)
        return go_long.astype(np.int8) - go_short.astype(np.int8)


def live(pop: Population, world: World, t0: int, t1: int, learn: bool = True) -> np.ndarray:
    """
    Lässt alle Fliegen die Tage [t0, t1) erleben. Rückgabe: Positionen (P, n)
    mit +1 Long, 0 NO TRADE, -1 Short. Mit learn=False ist es eine Prüfung:
    Die Fliege handelt nur mit dem, was sie schon weiß.
    """
    P, n = pop.size, t1 - t0
    positions = np.zeros((P, n), dtype=np.int8)
    for t in range(t0, t1):
        positions[:, t - t0] = pop.decide(world.kc[t])
        if learn:
            _learn(pop, world, t, t0, positions)
    return positions


def _learn(pop: Population, world: World, t: int, t0: int, positions: np.ndarray) -> None:
    g = pop.genes
    h = g["horizon"].astype(int)
    s = t - h                                       # der Tag, dessen Ergebnis heute feststeht
    ok = s >= 0
    s = np.where(ok, s, 0)

    move = world.fwd[s, h]                          # log-Rendite s -> t
    scale = world.vol[s] * np.sqrt(h)
    outcome = np.stack([move - 2 * COST,            # was Long gebracht hätte
                        -move - 2 * COST - BORROW * h], axis=1)
    z = outcome / scale[:, None]

    # Was hat die Fliege am Tag s tatsächlich getan? Vor diesem Lebensabschnitt: nichts.
    in_window = s >= t0
    taken = np.where(in_window, positions[np.arange(pop.size), np.clip(s - t0, 0, None)], 0)
    miss = g["miss_weight"][:, None]
    weight = np.stack([np.where(taken == 1, 1.0, miss[:, 0]),
                       np.where(taken == -1, 1.0, miss[:, 0])], axis=1)

    d = dopamine(z, g["loss_aversion"][:, None]) * weight
    d = np.where(ok[:, None] & np.isfinite(d), d, 0.0)

    rows = np.arange(pop.size)[:, None, None]
    acts = np.arange(2)[None, :, None]
    cells = world.kc[s][:, None, :]                  # (P, 1, active)
    lr = g["lr"][:, None, None]
    reward = np.clip(d, 0, None)[:, :, None]
    punish = np.clip(-d, 0, None)[:, :, None]
    pop.nogo[rows, acts, cells] *= (1.0 - lr * reward)
    pop.go[rows, acts, cells] *= (1.0 - lr * punish)

    f = g["forget"][:, None, None].astype(np.float32)
    pop.go += f * (1.0 - pop.go)
    pop.nogo += f * (1.0 - pop.nogo)


def pnl(positions: np.ndarray, world: World, t0: int, t1: int) -> np.ndarray:
    """Tagesergebnis je Fliege nach Kosten, Form (P, n). Start aus NO TRADE."""
    ret = np.nan_to_num(world.ret1[t0:t1])
    prev = np.concatenate([np.zeros((positions.shape[0], 1), np.int8), positions[:, :-1]], axis=1)
    turnover = np.abs(positions.astype(float) - prev)
    return positions * ret - COST * turnover - BORROW * (positions == -1)


def sharpe(daily: np.ndarray) -> np.ndarray:
    """Annualisierter Sharpe je Zeile; wer nie handelt, bekommt 0 — NO TRADE ist kein Fehler."""
    sd = daily.std(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = daily.mean(axis=-1) / sd * np.sqrt(252)
    return np.where(sd > 1e-12, s, 0.0)


def vote(positions: np.ndarray, quorum: int) -> np.ndarray:
    """Schwarm: Long/Short nur, wenn mindestens `quorum` Fliegen zustimmen."""
    longs = (positions == 1).sum(axis=0)
    shorts = (positions == -1).sum(axis=0)
    out = np.zeros(positions.shape[1], dtype=np.int8)
    out[longs >= quorum] = 1
    out[shorts >= quorum] = -1
    out[(longs >= quorum) & (shorts >= quorum)] = 0   # Schwarm uneins -> NO TRADE
    return out
