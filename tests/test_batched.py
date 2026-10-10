"""Phase 3: the batched engine makes exactly the reference engine's transitions."""

import random
from dataclasses import replace

import numpy as np
import pytest

from pokearena import batched, engine
from pokearena.data import load_tables
from pokearena.engine import DONE, FORCED_SWITCH, NORMAL, TURN_LIMIT, new_battle
from pokearena.rng import hash32, rand32

T = load_tables()
PRE = batched.precompute(T)


def random_battle(rng: random.Random) -> engine.Battle:
    sizes = rng.choice([(3, 3), (3, 3), (3, 3), (1, 1), (2, 3), (1, 3)])
    b = new_battle([rng.sample(range(30), sizes[0]), rng.sample(range(30), sizes[1])], rng.randrange(2**32))
    if rng.random() < 0.1:                       # start some near the turn limit so draws happen
        b = replace(b, turn=TURN_LIMIT - rng.randrange(1, 6))
    return b


def choose(mask: np.ndarray, rng: random.Random, switchy: bool) -> int:
    legal = np.flatnonzero(mask)
    if not len(legal):
        return rng.randrange(7)                  # ignored; any value must be accepted
    switches = [a for a in legal if a >= 4]
    if switchy and switches and rng.random() < 0.8:
        return int(rng.choice(switches))
    return int(rng.choice(legal))


# Tables and RNG

def test_damage_table_equals_the_reference_and_the_arithmetic_path():
    n = len(T.species_ids)
    a, d, s = np.meshgrid(np.arange(n), np.arange(n), np.arange(4), indexing="ij")
    assert (batched.damage_arithmetic(T, a, d, s) == PRE.damage).all()
    for attacker in range(n):
        for defender in range(n):
            for slot in range(4):
                assert PRE.damage[attacker, defender, slot] == engine.damage(
                    T, attacker, defender, int(T.species_moves[attacker, slot]))


def test_vectorized_rng_matches_scalar():
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 2**32, 10_000, dtype=np.int64)
    assert (batched.hash32(xs) == np.array([hash32(int(x)) for x in xs])).all()
    seeds = rng.integers(0, 2**32, 1000, dtype=np.int64)
    turns = rng.integers(1, 101, 1000).astype(np.int32)
    expected = np.array([rand32(int(s), int(t), 0) for s, t in zip(seeds, turns)])
    assert (batched.rand32(seeds, turns, 0) == expected).all()


# Conversion

def test_round_trip_through_batch_state():
    rng = random.Random(0)
    battles = [random_battle(rng) for _ in range(200)]
    assert batched.to_battles(batched.from_battles(battles)) == battles


def test_masks_equal_the_reference():
    rng = random.Random(1)
    battles = [random_battle(rng) for _ in range(300)]
    s = batched.from_battles(battles)
    for _ in range(30):
        masks = batched.action_masks(s)
        for i, b in enumerate(batched.to_battles(s)):
            for side in (0, 1):
                assert (masks[i, side] == engine.action_mask(b, side)).all()
        s = batched.step(s, batched.random_actions(s, np.random.default_rng(0)), PRE)


# Differential stepping

@pytest.mark.parametrize("chunk", range(4))
def test_batched_steps_equal_reference_steps(chunk):
    rng = random.Random(100 + chunk)
    battles = [random_battle(rng) for _ in range(500)]
    switchy = [rng.random() < 0.2 for _ in battles]       # some battles mostly switch, to reach draws
    s = batched.from_battles(battles)
    seen = {NORMAL: 0, FORCED_SWITCH: 0}
    outcomes = set()
    while any(b.phase != DONE for b in battles):
        actions = np.array([[choose(engine.action_mask(b, side), rng, switchy[i]) for side in (0, 1)]
                            for i, b in enumerate(battles)])
        for b in battles:
            if b.phase != DONE:
                seen[b.phase] += 1
        battles = [engine.step(b, tuple(int(a) for a in actions[i]))[0] if b.phase != DONE else b
                   for i, b in enumerate(battles)]
        arithmetic = batched.step(s, actions, PRE, table_damage=False)
        s = batched.step(s, actions, PRE)
        assert batched.to_battles(s) == battles
        assert all((getattr(arithmetic, f) == getattr(s, f)).all() for f in ("hp", "active", "phase", "outcome"))
    outcomes = {b.outcome for b in battles}
    assert seen[NORMAL] > 3000 and seen[FORCED_SWITCH] > 300
    assert outcomes == {0, 1, engine.DRAW}


def test_finished_battles_do_not_change():
    b = replace(new_battle([[0, 1, 2], [3, 4, 5]], 9), phase=DONE, outcome=1)
    s = batched.from_battles([b])
    assert batched.to_battles(batched.step(s, np.array([[0, 0]]), PRE)) == [b]


def test_illegal_actions_raise():
    s = batched.from_battles([new_battle([[0, 1, 2], [3, 4, 5]], 0)])
    with pytest.raises(ValueError):
        batched.step(s, np.array([[4, 0]]), PRE)              # switch to the active slot
    with pytest.raises(ValueError):
        batched.step(s, np.array([[7, 0]]), PRE)


def test_padded_one_v_one_matches_the_reference():
    b = new_battle([[engine_species("weavile")], [engine_species("garchomp")]], 0)
    s = batched.from_battles([b])
    assert batched.action_masks(s)[0, 0].tolist() == [True] * 4 + [False] * 3
    nb, _ = engine.step(b, (2, 0))
    assert batched.to_battles(batched.step(s, np.array([[2, 0]]), PRE)) == [nb]


def engine_species(name: str) -> int:
    return T.species_ids.index(name)


# Auto-reset and random play

def test_reset_done_replaces_only_finished_battles():
    rng = np.random.default_rng(0)
    n = 64
    teams = np.stack([np.stack([rng.permutation(30)[:3] for _ in range(2)]) for _ in range(n)])
    s = batched.new_batch(teams, rng.integers(0, 2**32, n), PRE)
    for _ in range(16):
        s = batched.step(s, batched.random_actions(s, rng), PRE)
    done = s.phase == DONE
    assert 0 < done.sum() < n
    fresh_teams = teams[::-1].copy()
    r = batched.reset_done(s, fresh_teams, np.arange(n), PRE)
    assert (r.species[done] == fresh_teams[done]).all() and (r.turn[done] == 0).all()
    assert (r.phase[done] == NORMAL).all()
    for field in ("species", "hp", "active", "phase", "turn", "outcome", "seed"):
        assert (getattr(r, field)[~done] == getattr(s, field)[~done]).all()


def test_random_actions_are_legal():
    rng = np.random.default_rng(1)
    battles = [random_battle(random.Random(i)) for i in range(256)]
    s = batched.from_battles(battles)
    for _ in range(40):
        s = batched.step(s, batched.random_actions(s, rng), PRE, check=True)
    assert (s.phase == DONE).mean() > 0.9


def test_state_size():
    s = batched.from_battles([new_battle([[0, 1, 2], [3, 4, 5]], 0)])
    assert s.nbytes_per_battle() == 4 * (6 + 6 + 2 + 3) + 8
