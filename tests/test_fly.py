"""
Die Garantien, auf denen jedes Ergebnis steht.

Kein Test prüft, ob die Fliege Geld verdient — das ist eine Frage an die
Daten, nicht an den Code. Getestet wird, dass sie es nicht durch einen
Blick in die Zukunft tun KANN und dass Lernen und Vererbung das tun, was die
Biologie vorgibt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fly.data import Market
from fly.dopamine import dopamine
from fly.evolution import GENES, breed, hatch, mutate
from fly.population import LONG, SHORT, MAX_HORIZON, World, live, sharpe, vote
from fly.senses import channels
from fly.skull import Skull


# ─── Hilfsmittel ─────────────────────────────────────────────────────────────

def fake_closes(n=900, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({
        "spy": 100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, n))),
        "vix": 20 * np.exp(np.cumsum(rng.normal(0, 0.03, n))),
        "tnx": 4 + np.cumsum(rng.normal(0, 0.03, n)),
    }, index=idx)


def fake_world(T=200, n_kc=300, active=15, seed=0) -> World:
    rng = np.random.default_rng(seed)
    kc = np.sort(np.stack([rng.choice(n_kc, active, replace=False) for _ in range(T)]), axis=1)
    ret1 = rng.normal(0, 0.01, T)
    logp = np.r_[0, np.cumsum(np.log1p(ret1))]
    # fwd[t, h] = log(p[t+h] / p[t]); am Ende NaN, wo t+h hinter den Daten liegt
    fwd = np.full((T, MAX_HORIZON + 1), np.nan)
    price = logp[:T]
    for h in range(MAX_HORIZON + 1):
        fwd[:T - h, h] = price[h:] - price[:T - h]
    return World(pd.bdate_range("2000-01-03", periods=T), kc, ret1, fwd, np.full(T, 0.01))


# ─── Keine Zukunft ───────────────────────────────────────────────────────────

def test_senses_do_not_look_ahead():
    closes = fake_closes()
    full = channels(closes)
    cut = channels(closes.iloc[:700])
    pd.testing.assert_frame_equal(full.loc[cut.index], cut)


def test_market_until_removes_the_future():
    m = Market(fake_closes())
    cut = m.days[500]
    assert m.until(cut).days.max() < cut


def test_decisions_do_not_depend_on_later_outcomes():
    """Fälscht man alles, was nach Tag k passiert, bleiben die Entscheidungen bis k gleich."""
    world = fake_world()
    rng = np.random.default_rng(3)
    a = hatch(20, 300, np.random.default_rng(1))
    b = hatch(20, 300, np.random.default_rng(1))
    a.genes["lr"][:] = b.genes["lr"][:] = 0.3          # kräftig lernen, damit es etwas zu verändern gibt
    k = 120

    fwd = world.fwd.copy()
    T = len(world.days)
    for h in range(MAX_HORIZON + 1):
        late = np.arange(T) + h > k                      # Ergebnisse, die erst nach Tag k feststehen
        fwd[late, h] = rng.normal(0, 0.05, late.sum())
    ret1 = world.ret1.copy()
    ret1[k:] = rng.normal(0, 0.05, T - k)
    forged = World(world.days, world.kc, ret1, fwd, world.vol)

    pa = live(a, world, 0, T)
    pb = live(b, forged, 0, T)
    np.testing.assert_array_equal(pa[:, :k + 1], pb[:, :k + 1])
    assert (pa != 0).any(), "Test ist nur aussagekräftig, wenn überhaupt gehandelt wird"


# ─── Schädel ─────────────────────────────────────────────────────────────────

def test_apl_lets_exactly_the_sparse_few_fire():
    skull = Skull.grow(n_channels=30, n_kc=500, sparsity=0.05, seed=7)
    x = np.random.default_rng(0).random((40, 30))
    kc = skull.smell(x)
    assert kc.shape == (40, 25)
    assert all(len(np.unique(row)) == 25 for row in kc)


def test_shared_skull_is_reproducible():
    a, b = Skull.grow(30, seed=0), Skull.grow(30, seed=0)
    np.testing.assert_array_equal(a.wiring, b.wiring)


# ─── Lernen ──────────────────────────────────────────────────────────────────

def test_fresh_fly_does_not_trade():
    pop = hatch(10, 300, np.random.default_rng(0))
    assert (pop.decide(np.arange(15)) == 0).all()


def test_unfamiliar_pattern_is_not_traded_even_by_a_bold_fly():
    """Negative Schwelle = sehr mutig. Trotzdem: Unbekanntes wird nicht gehandelt."""
    world = fake_world()
    pop = hatch(1, 300, np.random.default_rng(0))
    pop.genes.update(thr_long=np.array([-0.4]), thr_short=np.array([-0.4]),
                     horizon=np.array([1.0]), lr=np.array([0.3]), miss_weight=np.array([1.0]))
    assert pop.decide(np.arange(15))[0] == 0
    live(pop, world, 0, 60)                                  # lernt Muster aus 60 Tagen
    seen = world.kc[30]
    never_seen = np.setdiff1d(np.arange(300), world.kc[:60].ravel())
    assert pop.decide(seen)[0] != 0
    if len(never_seen) >= 5:
        assert pop.decide(never_seen[:15])[0] == 0


def test_reward_raises_and_punishment_lowers_the_value():
    world = fake_world()
    for sign in (+1, -1):
        pop = hatch(1, 300, np.random.default_rng(0))
        pop.genes.update(horizon=np.array([1.0]), lr=np.array([0.3]),
                         forget=np.array([1e-4]), miss_weight=np.array([1.0]))
        w = World(world.days, world.kc, world.ret1,
                  np.full_like(world.fwd, sign * 0.03), world.vol)  # Markt steigt bzw. fällt immer
        live(pop, w, 0, 50)
        v = pop.values(w.kc[10])[0]
        assert np.sign(v[LONG]) == sign and np.sign(v[SHORT]) == -sign
        assert 0 <= pop.go.min() and pop.go.max() <= 1 and 0 <= pop.nogo.min() <= 1


def test_losses_hurt_more_with_loss_aversion():
    z = np.array([[1.0, -1.0]])
    calm = dopamine(z, np.array([[1.0]]))
    scared = dopamine(z, np.array([[3.0]]))
    assert calm[0, 0] == scared[0, 0]
    assert scared[0, 1] < calm[0, 1] <= 0
    assert np.abs(dopamine(np.array([[50.0, -50.0]]), np.array([[4.0]]))).max() <= 1.0


# ─── Evolution ───────────────────────────────────────────────────────────────

def test_child_inherits_each_memory_whole_from_one_parent():
    rng = np.random.default_rng(0)
    parents = hatch(2, 300, rng)
    parents.go[0], parents.nogo[0] = 0.2, 0.3
    parents.go[1], parents.nogo[1] = 0.7, 0.9
    kids = breed(parents, 5, rng, first_id=100)
    for i in range(kids.size):
        from_mother = (kids.go[i] == 0.2).all(axis=0)
        from_father = (kids.go[i] == 0.7).all(axis=0)
        assert (from_mother ^ from_father).all()
        np.testing.assert_array_equal(kids.nogo[i][:, from_mother], np.float32(0.3))
        np.testing.assert_array_equal(kids.nogo[i][:, from_father], np.float32(0.9))


def test_saved_swarm_comes_back_identical(tmp_path):
    from fly.population import Population
    rng = np.random.default_rng(0)
    pop = hatch(3, 50, rng)
    pop.go[:] = rng.random(pop.go.shape)
    np.savez_compressed(tmp_path / "swarm.npz", go=pop.go, nogo=pop.nogo, ids=pop.ids,
                        **{f"gene_{k}": v for k, v in pop.genes.items()})
    back = Population.load(tmp_path / "swarm.npz")
    np.testing.assert_array_equal(back.go, pop.go)
    assert back.genes.keys() == pop.genes.keys()
    np.testing.assert_array_equal(back.decide(np.arange(10)), pop.decide(np.arange(10)))


def test_mutation_keeps_genes_in_bounds():
    rng = np.random.default_rng(0)
    pop = hatch(200, 10, rng)
    pop.genes["mut_scale"][:] = 0.5
    for _ in range(20):
        mutate(pop, rng)
    for name, (lo, hi, *_rest) in GENES.items():
        assert pop.genes[name].min() >= lo and pop.genes[name].max() <= hi, name
    assert np.all(pop.genes["horizon"] == np.rint(pop.genes["horizon"]))


# ─── Schwarm & Bewertung ─────────────────────────────────────────────────────

def test_swarm_trades_only_with_a_clear_majority():
    pos = np.array([[1, 1, -1, 0]] * 6 + [[0, -1, -1, 0]] * 4)
    np.testing.assert_array_equal(vote(pos, quorum=6), [1, 1, -1, 0])
    np.testing.assert_array_equal(vote(pos[:5], quorum=6), [0, 0, 0, 0])


def test_no_trade_scores_zero_not_an_error():
    assert sharpe(np.zeros((1, 60)))[0] == 0.0


# ─── Kolonie (Wochenzucht) ───────────────────────────────────────────────────

def small_colony(seed=1, control="none"):
    from fly.colony import Colony, ColonyConfig
    cfg = ColonyConfig(population=20, window=30, min_days=10, swarm=5, quorum=3,
                       control=control, seed=seed)
    col = Colony.found(cfg, 300)
    col.pop.genes["lr"][:] = 0.3                      # kräftig lernen, damit gehandelt wird
    return col


def test_colony_same_result_in_one_go_or_day_by_day(tmp_path):
    """Backtest am Stück == Live-Betrieb Tag für Tag, inklusive Speichern und Laden."""
    from fly.colony import Colony
    world = fake_world()
    T = len(world.days)
    a = small_colony()
    swarm_a, pos_a = a.advance(world, 0, T)

    b = small_colony()
    swarm_b, pos_b = [], []
    for t in range(T):
        s, p = b.advance(world, t, t + 1)
        swarm_b.append(s); pos_b.append(p)
        if t % 37 == 0:                               # zwischendurch speichern und neu laden
            b.save(tmp_path / "c.npz")
            b = Colony.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(swarm_a, np.concatenate(swarm_b))
    np.testing.assert_array_equal(pos_a, np.concatenate(pos_b, axis=1))
    assert (pos_a != 0).any() and len(a.log) > 0


def test_colony_decisions_do_not_depend_on_later_outcomes():
    world = fake_world()
    T, k = len(world.days), 120
    rng = np.random.default_rng(5)
    fwd = world.fwd.copy()
    for h in range(MAX_HORIZON + 1):
        late = np.arange(T) + h > k
        fwd[late, h] = rng.normal(0, 0.05, late.sum())
    ret1 = world.ret1.copy()
    ret1[k:] = rng.normal(0, 0.05, T - k)
    forged = World(world.days, world.kc, ret1, fwd, world.vol)

    sa, pa = small_colony().advance(world, 0, T)
    sb, pb = small_colony().advance(forged, 0, T)
    # Auslese am Wochenende nach Tag k darf schon anders sein — bis k selbst nicht.
    np.testing.assert_array_equal(pa[:, :k + 1], pb[:, :k + 1])
    np.testing.assert_array_equal(sa[:k + 1], sb[:k + 1])


def test_colony_kills_worst_and_clones_best():
    world = fake_world()
    col = small_colony()
    col.advance(world, 0, 60)
    fit = col.fitness()
    adults = np.flatnonzero(np.isfinite(fit))
    order = adults[np.argsort(-fit[adults], kind="stable")]
    best, worst = order[:2], order[-2:]
    best_go = col.pop.go[best].copy()
    ids_before = col.pop.ids.copy()
    col._select(world, 60)
    for dead, parent_go in zip(worst, best_go):
        np.testing.assert_array_equal(col.pop.go[dead], parent_go)   # Gedächtnis geklont
    assert set(col.pop.ids[worst]).isdisjoint(ids_before)            # neue Fliegen
    np.testing.assert_array_equal(col.swarm_idx, order[:5])          # die Besten stimmen ab


# ─── Papier-Broker (ohne Netz) ───────────────────────────────────────────────

class FakeAlpaca:
    def __init__(self, qty, equity=10_000.0, pending=None):
        self.qty, self.equity, self.orders, self.deleted = qty, equity, [], []
        self.pending = pending if pending is not None else [{"id": "alt", "side": "buy", "qty": "1"}]

    def get(self, url, params=None, timeout=None):
        class R:
            def __init__(s, code, data): s.status_code, s._d = code, data
            def json(s): return s._d
            def raise_for_status(s): pass
        if url.endswith("/v2/orders"):
            return R(200, self.pending)
        if "/v2/orders/" in url:
            return R(200, {"status": "canceled"})
        if url.endswith("/v2/account"):
            return R(200, {"equity": str(self.equity)})
        return R(404, {}) if self.qty == 0 else R(200, {"qty": str(self.qty)})

    def post(self, url, json=None, timeout=None):
        self.orders.append(json)
        r = self.get("x/v2/account")
        r._d = {"id": "neu"}
        return r

    def delete(self, url, timeout=None):
        self.deleted.append(url)


@pytest.mark.parametrize("qty,signal,expected", [
    (0, 1, ("buy", "20")),        # flat -> long: 10.000 / 500 = 20 Stück
    (20, 1, None),                # schon richtig -> keine Order
    (20, 0, ("sell", "20")),      # long -> NO TRADE
    (20, -1, ("sell", "20")),     # long -> short: heute nur schließen
    (-20, 1, ("buy", "20")),      # short -> long: heute nur schließen
])
def test_paper_broker_follows_signal(monkeypatch, qty, signal, expected):
    import fly.broker as broker
    fake = FakeAlpaca(qty)
    monkeypatch.setattr(broker, "_session", lambda: fake)
    broker.sync_paper_position(signal, price=500.0)
    assert fake.deleted, "fremde offene Orders werden storniert"
    if expected is None:
        assert fake.orders == []
    else:
        assert (fake.orders[0]["side"], fake.orders[0]["qty"]) == expected
    assert broker.PAPER.startswith("https://paper-api.")


# ─── Positionsschicht ────────────────────────────────────────────────────────

def test_overlay_does_not_look_ahead_and_fly_only_speaks_in_extremes():
    from fly.overlay import exposure
    closes = fake_closes(n=900)["spy"]
    swarm = pd.Series(np.random.default_rng(1).choice([-1, 0, 1], len(closes)), index=closes.index)
    full = exposure(closes, swarm)
    cut = exposure(closes.iloc[:600], swarm.iloc[:600])
    pd.testing.assert_series_equal(full["gewicht"].iloc[:600], cut["gewicht"])
    calm = ~full["extrem"] & full["basis"].notna()
    np.testing.assert_allclose(full.loc[calm, "gewicht"], full.loc[calm, "basis"])
    assert full["gewicht"].abs().max() <= 1.0 + 1e-12          # ohne Hebel (cap=1)


def test_paper_broker_is_idempotent_on_double_run(monkeypatch):
    """Zweiter Lauf am selben Abend: die richtige Order liegt schon — nichts stornieren, nichts neu."""
    import fly.broker as broker
    fake = FakeAlpaca(0, pending=[{"id": "gestern", "side": "buy", "qty": "20"}])
    monkeypatch.setattr(broker, "_session", lambda: fake)
    broker.sync_paper_position(1, price=500.0)
    assert fake.deleted == [] and fake.orders == []


def test_shuffled_control_does_not_learn_from_the_future():
    """Die Kontrolle mischt nur Vergangenes — gefälschte Zukunft ändert nichts bis Tag k."""
    world = fake_world()
    T, k = len(world.days), 120
    rng = np.random.default_rng(9)
    fwd = world.fwd.copy()
    for h in range(MAX_HORIZON + 1):
        late = np.arange(T) + h > k
        fwd[late, h] = rng.normal(0, 0.05, late.sum())
    forged = World(world.days, world.kc, world.ret1, fwd, world.vol)
    a = small_colony(control="shuffled-dopamine")
    b = small_colony(control="shuffled-dopamine")
    _, pa = a.advance(world, 0, T, world.with_shuffled_learning(np.random.default_rng(1)))
    _, pb = b.advance(forged, 0, T, forged.with_shuffled_learning(np.random.default_rng(1)))
    np.testing.assert_array_equal(pa[:, :k + 1], pb[:, :k + 1])


def test_position_earns_only_from_next_open():
    """Entscheidung am Schluss t -> Ausführung Eröffnung t+1 -> Ergebnis gebucht am Tag t+2."""
    from fly.population import realized
    T = 12
    ropen = np.arange(T, dtype=float) / 100          # Tag t: Eröffnung t-1 -> t bringt t %
    w = World(pd.bdate_range("2020-01-01", periods=T), np.zeros((T, 1), int), np.zeros(T),
              np.zeros((T, MAX_HORIZON + 1)), np.ones(T), ropen)
    pos = np.zeros((1, T), np.int8)
    pos[0, 5] = 1                                    # nur am Schluss von Tag 5 long
    out = realized(pos, 0, 0, T, w)[0]
    assert out[7] == pytest.approx(0.07 - 0.0005)    # Tag 7: Eröffnung 6 -> 7, Einstiegskosten
    assert out[6] == 0 and out[5] == 0               # vorher nichts
    assert out[8] == pytest.approx(-0.0005)          # Ausstieg zur Eröffnung 7 kostet
