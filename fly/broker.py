"""
Angleichen eines Alpaca-PAPIERKONTOS an das Schwarm-Signal.

Absichtlich nur Papiergeld: Die Adresse ist fest verdrahtet und nicht per
Umgebungsvariable umstellbar. Wer eines Tages echtes Geld einsetzen will, muss
diese Datei bewusst ändern — nachdem eine Version Walk-forward UND Friedhof
bestanden und Monate im Papierkonto überzeugt hat.

Schlüssel kommen ausschließlich aus Umgebungsvariablen (GitHub-Secrets
ALPACA_KEY_ID / ALPACA_SECRET_KEY) und stehen nie im Code oder im Repo.

Ablauf je Lauf (nach Börsenschluss, Orders werden zur nächsten Eröffnung ausgeführt):
  1. offene SPY-Orders vom Vortag stornieren
  2. Ziel = Signal × ganzes Kapital in ganzen Stück SPY (Short nur in ganzen Stück)
  3. Wechsel Long <-> Short in zwei Schritten: heute schließen, morgen neu eröffnen
     (eine einzige Order über null hinweg lehnt Alpaca ab)
"""
from __future__ import annotations

import math
import os
import time

import requests

PAPER = "https://paper-api.alpaca.markets"   # fest: Papiergeld
SYMBOL = "SPY"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"],
                      "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]})
    return s


def _current_qty(s: requests.Session) -> int:
    r = s.get(f"{PAPER}/v2/positions/{SYMBOL}", timeout=20)
    if r.status_code == 404:
        return 0
    r.raise_for_status()
    return int(float(r.json()["qty"]))


def _get(s: requests.Session, path: str, **params):
    r = s.get(f"{PAPER}{path}", params=params or None, timeout=20)
    r.raise_for_status()                         # 401 & Co. laut, nicht als falsches JSON
    return r.json()


def _cancel_and_wait(s: requests.Session, orders: list, wait: float = 15.0) -> None:
    """Stornieren und abwarten — sonst blockiert eine halb stornierte Order die neue."""
    for o in orders:
        s.delete(f"{PAPER}/v2/orders/{o['id']}", timeout=20)
    deadline = time.time() + wait
    for o in orders:
        while time.time() < deadline:
            if _get(s, f"/v2/orders/{o['id']}").get("status") in ("canceled", "filled", "expired", "rejected"):
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Order {o['id']} ließ sich nicht rechtzeitig stornieren")


def sync_paper_position(signal: int, price: float, fraction: float = 1.0) -> dict:
    s = _session()
    equity = float(_get(s, "/v2/account")["equity"])
    current = _current_qty(s)
    target = signal * math.floor(equity * fraction / price)
    if current and target and (current > 0) != (target > 0):
        target = 0                               # erst schließen, Gegenrichtung beim nächsten Lauf
    diff = target - current
    side = "buy" if diff > 0 else "sell"

    open_orders = _get(s, "/v2/orders", status="open", symbols=SYMBOL)
    # Idempotent: Liegt genau die nötige Order schon da (Doppellauf), nichts tun.
    if diff and len(open_orders) == 1 and open_orders[0].get("side") == side \
            and int(float(open_orders[0].get("qty", 0))) == abs(diff):
        print(f"Papierkonto: passende Order liegt schon ({side} {abs(diff)} {SYMBOL}).")
        return {"current": current, "target": target, "order": open_orders[0]["id"]}
    if open_orders:
        _cancel_and_wait(s, open_orders)
    if diff == 0:
        print(f"Papierkonto: {current} {SYMBOL} — passt zum Signal.")
        return {"current": current, "target": target, "order": None}

    order = {"symbol": SYMBOL, "qty": str(abs(diff)), "side": side,
             "type": "market", "time_in_force": "day"}
    r = s.post(f"{PAPER}/v2/orders", json=order, timeout=20)
    r.raise_for_status()
    print(f"Papierkonto: {current} -> {target} {SYMBOL} ({order['side']} {abs(diff)}), "
          f"wird zur nächsten Eröffnung ausgeführt.")
    return {"current": current, "target": target, "order": r.json().get("id")}
