"""
Das Belohnungssystem: aus einem Marktergebnis wird ein Dopamin-Stoß.

Positiv  = Belohnung   -> die Fliege mag dieses Muster für diese Aktion mehr
Negativ  = Bestrafung  -> die Fliege meidet dieses Muster für diese Aktion

Das Ergebnis kommt schon risikobereinigt herein (`z`): Rendite geteilt durch
die Schwankung, die in dieser Zeit zu erwarten war. Ein Plus von 1 % ist in
ruhigen Zeiten eine Überraschung, in einem Crash nur Rauschen — die Fliege soll
Überraschungen lernen, nicht Lautstärke.

Hier ist der Ort, an dem der "Charakter" der Zucht entsteht. Die Gene, die hier
wirken (Verlustangst), vererben sich und werden von der Evolution ausgelesen.
"""
from __future__ import annotations

import numpy as np

SATURATION = 2.0   # ab 2 Standardabweichungen ist ein Ergebnis "maximal" — Ausreißer übertönen nichts


def dopamine(z: np.ndarray, loss_aversion: np.ndarray) -> np.ndarray:
    """
    z              risikobereinigtes Ergebnis je Fliege und Aktion, Form (P, 2)
    loss_aversion  Gen je Fliege, Form (P, 1): wie viel stärker ein Verlust schmerzt

    Rückgabe in [-1, 1]. tanh sättigt sanft: kleine Ergebnisse wirken fast
    linear, große stoßen an eine Grenze, statt ein Gedächtnis auf einen Schlag
    zu überschreiben.
    """
    d = np.tanh(z / SATURATION)
    pain = np.minimum(d * loss_aversion, 0.0).clip(-1.0, 0.0)
    return np.where(d < 0, pain, d)
