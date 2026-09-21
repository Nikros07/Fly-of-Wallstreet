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

NO TRADE ist der Ruhezustand: gehandelt wird nur, wenn das Muster vertraut
ist (Neuheitsdetektor) UND ein Wert die Handelsschwelle (ein Gen) übersteigt.
Eine frisch geschlüpfte Fliege hat überall GO = NOGO = 1 — nichts ist ihr
vertraut, sie handelt nicht.

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
NOVELTY = 0.05           # unter so viel Erfahrung gilt ein Muster als unbekannt


@dataclass(frozen=True)
class World:
    """Alles, was die Fliegen erleben können — vorberechnet, für alle gleich."""
    days: pd.DatetimeIndex
    kc: np.ndarray        # (T, active) feuernde Kenyon-Zellen je Tag
    ret1: np.ndarray      # (T,) Rendite Schluss t -> Schluss t+1 (Handelsergebnis)
    fwd: np.ndarray       # (T, MAX_HORIZON+1) log-Rendite t -> t+h (Lernsignal)
    vol: np.ndarray       # (T,) tägliche Schwankung, bekannt zum Schluss von t
    ropen: np.ndarray | None = None   # (T,) Eröffnung t-1 -> Eröffnung t (Ausführung)

    def open_returns(self) -> np.ndarray:
        """Eröffnung-zu-Eröffnung; ohne Eröffnungskurse (Testwelten) Schluss t-1 -> t."""
        if self.ropen is not None:
            return self.ropen
        return np.r_[np.nan, self.ret1[:-1]]

    @classmethod
    def build(cls, market: Market, skull: Skull) -> "World":
        ch = channels(market.closes)
        days = ch.index
        logp = np.log(market.closes["spy"])
        fwd = np.stack([(logp.shift(-h) - logp).reindex(days).to_numpy()
                        for h in range(MAX_HORIZON + 1)], axis=1)
        vol = raw_features(market.closes)["vol_20"].reindex(days).to_numpy()
        ropen = None
        if "spy_open" in market.closes:
            o = market.closes["spy_open"]
            ropen = (o / o.shift(1) - 1.0).reindex(days).to_numpy()
        return cls(days=days,
                   kc=skull.smell(ch.to_numpy()),
                   ret1=market.next_day_returns().reindex(days).to_numpy(),
                   fwd=fwd,
                   vol=vol,
                   ropen=ropen)

    def with_shuffled_learning(self, rng: np.random.Generator) -> "World":
        """
        Kontrollversuch: Die Fliegen handeln in der echten Welt, bekommen aber
        Dopamin für die Ergebnisse zufälliger ANDERER Tage. Wer so genauso gut
        abschneidet, hat nie etwas gelernt.

        Nur aus der VERGANGENHEIT gezogen (Tag s bekommt das Ergebnis eines
        Tages <= s): Eine Mischung über alle Tage verriet der Kontrolle die
        Durchschnittsrendite der ganzen Stichprobe — sie wusste vorab, dass es
        sich lohnt, long zu sein, und war eine Kontrolle mit Zukunftswissen.
        """
        T = len(self.days)
        idx = (rng.random(T) * (np.arange(T) + 1)).astype(int)
        return World(self.days, self.kc, self.ret1, self.fwd[idx], self.vol[idx], self.ropen)

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

    @classmethod
    def load(cls, path) -> "Population":
        """Einen gespeicherten Schwarm (swarm.npz aus results/) wieder zum Leben erwecken."""
        z = np.load(path)
        genes = {k[5:]: z[k] for k in z.files if k.startswith("gene_")}
        return cls(genes, z["go"], z["nogo"], z["ids"], [None] * len(z["ids"]))

    def values(self, active: np.ndarray) -> np.ndarray:
        """Wert von Long und Short für ein Feuermuster, Form (P, 2)."""
        return (self.go[:, :, active] - self.nogo[:, :, active]).mean(axis=2)

    def experience(self, active: np.ndarray) -> np.ndarray:
        """
        Wie vertraut ist dieses Muster? 0 = nie Dopamin dafür bekommen
        (alle Synapsen noch auf 1), wächst mit jedem Lernschritt. Form (P,).
        """
        return (2.0 - self.go[:, :, active] - self.nogo[:, :, active]).mean(axis=(1, 2))

    def decide(self, active: np.ndarray) -> np.ndarray:
        v = self.values(active)
        vl, vs = v[:, LONG], v[:, SHORT]
        # Neuheitsdetektor: Unbekanntes wird nicht gehandelt, egal wie mutig
        # die Schwelle ist. Das hält NO TRADE als Ruhezustand, obwohl
        # Schwellen negativ sein dürfen.
        familiar = self.experience(active) > NOVELTY
        go_long = familiar & (vl > self.genes["thr_long"]) & (vl >= vs)
        go_short = familiar & (vs > self.genes["thr_short"]) & (vs > vl)
        return go_long.astype(np.int8) - go_short.astype(np.int8)


def live(pop: Population, world: World, t0: int, t1: int, learn: bool = True,
         values_out: np.ndarray | None = None, past: np.ndarray | None = None) -> np.ndarray:
    """
    Lässt alle Fliegen die Tage [t0, t1) erleben. Rückgabe: Positionen (P, n)
    mit +1 Long, 0 NO TRADE, -1 Short. Mit learn=False ist es eine Prüfung:
    Die Fliege handelt nur mit dem, was sie schon weiß.

    `past` (P, MAX_HORIZON): die Positionen der Tage t0-MAX_HORIZON .. t0-1.
    Ohne sie gilt die Zeit davor als NO TRADE. Wer die Tage in Stücken erleben
    lässt (Woche für Woche, Tag für Tag im Live-Betrieb), muss sie weiterreichen,
    sonst weiß die Fliege an jeder Stückgrenze nicht mehr, was sie getan hat —
    und lernt anders als in einem Zug.

    `values_out` (P, n, 2) nimmt auf Wunsch die Werte auf, auf denen jede
    Entscheidung beruhte — für die Diagnose, ob die Fliege etwas "riecht".
    """
    P, n = pop.size, t1 - t0
    buf = np.zeros((P, MAX_HORIZON + n), dtype=np.int8)   # [Vergangenheit | dieses Stück]
    if past is not None:
        buf[:, :MAX_HORIZON] = past
    for t in range(t0, t1):
        if values_out is not None:
            values_out[:, t - t0] = pop.values(world.kc[t])
        buf[:, MAX_HORIZON + t - t0] = pop.decide(world.kc[t])
        if learn:
            _learn(pop, world, t, t0 - MAX_HORIZON, buf)
    return buf[:, MAX_HORIZON:]


def _learn(pop: Population, world: World, t: int, origin: int, buf: np.ndarray) -> None:
    """`buf[:, i]` ist die Position am Tag origin + i."""
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

    # Was hat die Fliege am Tag s tatsächlich getan? Unbekannt vor dem Puffer: nichts.
    col = s - origin
    known = col >= 0
    taken = np.where(known, buf[np.arange(pop.size), np.clip(col, 0, None)], 0)
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


def pnl(positions: np.ndarray, world: World, t0: int, t1: int,
        prev: np.ndarray | None = None) -> np.ndarray:
    """
    Tagesergebnis je Fliege nach Kosten, Form (P, n). `prev` (P,) ist die
    Position am Vortag; ohne sie startet das Stück aus NO TRADE und zahlt den
    Einstieg — richtig für einen echten Start, falsch mitten im Leben.
    """
    ret = np.nan_to_num(world.ret1[t0:t1])
    first = np.zeros((positions.shape[0], 1), np.int8) if prev is None else         np.asarray(prev, np.int8).reshape(-1, 1)
    before = np.concatenate([first, positions[:, :-1]], axis=1)
    turnover = np.abs(positions.astype(float) - before)
    return positions * ret - COST * turnover - BORROW * (positions == -1)


EXEC_LAG = 2   # Entscheidung am Schluss t -> gekauft zur Eröffnung t+1 -> Ergebnis ab Eröffnung t+2


def realized(full: np.ndarray, origin: int, t0: int, t1: int, world: World) -> np.ndarray:
    """
    Tatsächlich erzieltes Ergebnis der Tage [t0, t1), gebucht an dem Tag, an dem
    es feststeht. `full[:, i]` ist die Position vom Schluss des Tages origin + i.

    Die Position vom Schluss t-2 wurde zur Eröffnung t-1 ausgeführt und bringt
    Eröffnung t-1 -> Eröffnung t; das steht zur Eröffnung von t fest. Die Kosten
    fallen beim Umschichten zur Eröffnung t-1 an. Keine Rendite, die man real
    nicht bekommen hätte, und keine, die erst später bekannt wäre.
    """
    r = np.nan_to_num(world.open_returns()[t0:t1])

    def col(lag):
        i = np.arange(t0, t1) - lag - origin
        out = np.zeros((full.shape[0], t1 - t0), np.int8)
        ok = i >= 0
        out[:, ok] = full[:, i[ok]]
        return out

    held, before = col(EXEC_LAG), col(EXEC_LAG + 1)
    turnover = np.abs(held.astype(float) - before)
    return held * r - COST * turnover - BORROW * (held == -1)


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
