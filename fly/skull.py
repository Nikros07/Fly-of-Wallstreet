"""
Der Schädel: die fest verdrahtete Hälfte des Pilzkörpers.

Alle Fliegen einer Zucht teilen denselben Schädel. Nur deshalb können Kinder
das Gelernte ihrer Eltern erben und kreuzen: Kenyon-Zelle 417 steht bei jeder
Fliege für dasselbe Marktmuster, also bedeutet ihr Gewicht bei jeder Fliege
dasselbe.

Aufbau wie im Konnektom:
  * jede Kenyon-Zelle hört auf wenige, ZUFÄLLIG gewählte Sinneskanäle
    (Fliege: ~7 von ~50 Glomeruli; die Verdrahtung ist nachweislich zufällig)
  * das APL-Neuron hemmt alle Kenyon-Zellen, sodass nur die stärksten ~5 %
    feuern — aus jeder Marktlage wird ein scharfes, sparsames Muster

Weil der Schädel nie lernt, lässt sich das Feuermuster für jeden Tag einmal
vorab berechnen. Gelernt wird ausschließlich dahinter (siehe population.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Skull:
    wiring: np.ndarray       # (n_kc, fan_in) Indizes der Sinneskanäle je Kenyon-Zelle
    n_channels: int
    active: int              # wie viele Kenyon-Zellen das APL-Neuron durchlässt

    @property
    def n_kc(self) -> int:
        return self.wiring.shape[0]

    @classmethod
    def grow(cls, n_channels: int, n_kc: int = 2000, fan_in: int = 7,
             sparsity: float = 0.05, seed: int = 0) -> "Skull":
        rng = np.random.default_rng(seed)
        wiring = np.stack([rng.choice(n_channels, size=fan_in, replace=False)
                           for _ in range(n_kc)])
        return cls(wiring=wiring, n_channels=n_channels,
                   active=max(1, int(round(sparsity * n_kc))))

    def smell(self, channels: np.ndarray) -> np.ndarray:
        """
        Sinneskanäle (T, n_channels) -> Indizes der feuernden Kenyon-Zellen (T, active).

        Kenyon-Antwort = Summe der Eingänge. APL = die `active` stärksten
        gewinnen, alle anderen schweigen. Gleichstände bricht argpartition
        deterministisch — das Muster eines Tages ist reproduzierbar.
        """
        drive = channels[:, self.wiring].sum(axis=2)            # (T, n_kc)
        winners = np.argpartition(-drive, self.active - 1, axis=1)[:, :self.active]
        return np.sort(winners, axis=1)
