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
from fly.population import (LONG, SHORT, MAX_HORIZON, World, live, live_multi, pnl,
                            pnl_multi, shared_days, sharpe, vote)
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


# ─── Mehrmarkt ───────────────────────────────────────────────────────────────

def test_multi_market_decisions_do_not_depend_on_later_outcomes_in_any_market():
    """
    Wie test_decisions_do_not_depend_on_later_outcomes, aber über mehrere Märkte
    gleichzeitig: Fälscht man ALLES nach Tag k in JEDEM der Märkte, bleiben die
    Entscheidungen bis k in ALLEN Märkten gleich. Das sichert genau die
    Zeitdisziplin, um die es bei mehreren Märkten geht: Tag t muss für Markt 9
    abgeschlossen sein, bevor Markt 1 den nächsten Tag beginnt — sonst könnte
    ein Muster aus Markt 1 am selben Kalendertag schon "wissen", was in Markt 9
    erst später passiert.
    """
    worlds = [fake_world(seed=s) for s in range(3)]
    T = len(worlds[0].days)
    rng = np.random.default_rng(3)
    a = hatch(20, 300, np.random.default_rng(1))
    b = hatch(20, 300, np.random.default_rng(1))
    a.genes["lr"][:] = b.genes["lr"][:] = 0.3
    k = 120

    def forge(world):
        fwd = world.fwd.copy()
        for h in range(MAX_HORIZON + 1):
            late = np.arange(T) + h > k
            fwd[late, h] = rng.normal(0, 0.05, late.sum())
        ret1 = world.ret1.copy()
        ret1[k:] = rng.normal(0, 0.05, T - k)
        return World(world.days, world.kc, ret1, fwd, world.vol)

    forged = [forge(w) for w in worlds]

    pa = live_multi(a, worlds, 0, T)
    pb = live_multi(b, forged, 0, T)
    np.testing.assert_array_equal(pa[:, :, :k + 1], pb[:, :, :k + 1])
    assert (pa != 0).any(), "Test ist nur aussagekräftig, wenn überhaupt gehandelt wird"


def test_market_order_within_a_day_does_not_matter():
    """
    Alle neun Börsen schließen gleichzeitig: Entscheidung und Lernergebnis für
    einen Markt dürfen nicht davon abhängen, an welcher Stelle er in der Liste
    steht — sonst hätte ein Markt, der zuerst drankommt, am selben Tag schon
    einen Wissensvorsprung vor einem, der später drankommt (er wüsste, was die
    Fliege durch den ersten Markt heute bereits gelernt hat).
    """
    worlds = [fake_world(seed=s) for s in range(3)]
    T = len(worlds[0].days)
    pop_a = hatch(10, 300, np.random.default_rng(2))
    pop_b = hatch(10, 300, np.random.default_rng(2))
    pop_a.genes["lr"][:] = pop_b.genes["lr"][:] = 0.3

    pos_forward = live_multi(pop_a, worlds, 0, T)
    pos_reversed = live_multi(pop_b, worlds[::-1], 0, T)

    np.testing.assert_array_equal(pos_forward[0], pos_reversed[-1])
    np.testing.assert_array_equal(pos_forward[-1], pos_reversed[0])
    assert (pos_forward != 0).any(), "Test ist nur aussagekräftig, wenn überhaupt gehandelt wird"


def test_portfolio_pnl_is_the_mean_of_the_single_market_results():
    """Das Portfolio ist der Mittelwert der Einzelmarkt-Ergebnisse — jeder Markt
    trägt also höchstens 1/W des Kapitals."""
    worlds = [fake_world(seed=s) for s in range(3)]
    pop = hatch(5, 300, np.random.default_rng(0))
    positions = live_multi(pop, worlds, 0, 50, learn=False)
    result = pnl_multi(positions, worlds, 0, 50)
    manual = np.mean([pnl(positions[i], w, 0, 50) for i, w in enumerate(worlds)], axis=0)
    np.testing.assert_array_equal(result, manual)


def test_shared_days_rejects_mismatched_calendars():
    a = fake_world(T=200, seed=0)
    b = fake_world(T=150, seed=0)
    with pytest.raises(ValueError):
        shared_days([a, b])


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
