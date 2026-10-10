"""Phase 1: the CPU reference engine in 1v1.

Damage is checked exhaustively against an exact-fraction oracle written from
RULES.md, and whole battles are checked against an independent turn oracle
over thousands of random matchups, seeds and move choices.
"""

import math
import random
from dataclasses import replace
from fractions import Fraction

import pytest

from pokearena.data import SPE, load_tables
from pokearena.engine import (
    DONE,
    DRAW,
    N_ACTIONS,
    NORMAL,
    ONGOING,
    TURN_LIMIT,
    Battle,
    action_mask,
    damage,
    effectiveness4,
    new_battle,
    step,
)
from pokearena.rng import MASK32, hash32, rand32

T = load_tables()
S = {name: i for i, name in enumerate(T.species_ids)}


def slot(species: str, move: str) -> int:
    return [T.move_ids[m] for m in T.species_moves[S[species]]].index(move)


def duel(a: str, b: str, seed: int = 0) -> Battle:
    return new_battle([[S[a]], [S[b]]], seed)


# Oracles, written from RULES.md without the engine's integer tricks

def oracle_effectiveness(move_type: int, defender: int) -> Fraction:
    eff = Fraction(1)
    for d in T.species_types[defender]:
        if d >= 0:
            eff *= Fraction(int(T.type_chart[move_type, d]), 4)
    return eff


def oracle_damage(attacker: int, defender: int, move: int) -> int:
    move_type = int(T.move_type[move])
    attack = int(T.species_stats[attacker, T.move_attack_stat[move]])
    defense = int(T.species_stats[defender, T.move_defense_stat[move]])
    level = 50
    base = math.floor(
        Fraction((Fraction(2 * level, 5) + 2) * int(T.move_power[move]) * attack, defense) / 50 + 2
    )
    stab = Fraction(3, 2) if move_type in T.species_types[attacker] else Fraction(1)
    eff = oracle_effectiveness(move_type, defender)
    if eff == 0:
        return 0
    return max(1, math.floor(base * stab * eff))


def oracle_turn(b: Battle, actions: tuple[int, int]) -> tuple[tuple[int, int], int | None]:
    """Expected (hp0, hp1) after a 1v1 turn, and the winner if any."""
    species = (b.teams[0][0], b.teams[1][0])
    moves = [int(T.species_moves[species[s], actions[s]]) for s in (0, 1)]
    turn = b.turn + 1
    key = [(int(T.move_priority[moves[s]]), int(T.species_stats[species[s], SPE])) for s in (0, 1)]
    if key[0] != key[1]:
        first = 0 if key[0] > key[1] else 1
    else:
        first = rand32(b.seed, turn, 0) & 1
    hp = [b.hp[0][0], b.hp[1][0]]
    for side in (first, 1 - first):
        if hp[side] == 0:
            break
        foe = 1 - side
        hp[foe] = max(0, hp[foe] - oracle_damage(species[side], species[foe], moves[side]))
        if hp[foe] == 0:
            return (hp[0], hp[1]), side
    return (hp[0], hp[1]), None


# RNG

def test_rng_pinned_values():
    # The spec in RULES.md section 10, pinned so every other engine can be checked
    # against the same numbers.
    assert hash32(0) == 0
    assert [hash32(x) for x in (1, 2, 12345, MASK32)] == [0x688990C0, 0xD1132181, 0x912EFCF7, 0x6768824A]
    assert [rand32(0, 1, 0), rand32(12345, 7, 0), rand32(MASK32, 100, 3)] == [0xD69D7985, 0xFB357EDF, 0xEB0BE27C]
    assert all(0 <= hash32(x) <= MASK32 for x in range(1000))


def test_rand32_is_deterministic_and_balanced():
    assert rand32(7, 3, 0) == rand32(7, 3, 0)
    assert rand32(7, 3, 0) != rand32(7, 3, 1) != rand32(7, 4, 0)
    bits = [rand32(seed, turn, 0) & 1 for seed in range(200) for turn in range(1, 51)]
    assert 0.47 < sum(bits) / len(bits) < 0.53


# Damage

def test_damage_matches_oracle_for_every_matchup_and_move():
    checked = 0
    for attacker in range(len(T.species_ids)):
        for defender in range(len(T.species_ids)):
            for move in T.species_moves[attacker]:
                assert damage(T, attacker, defender, int(move)) == oracle_damage(attacker, defender, int(move)), (
                    T.species_ids[attacker], T.species_ids[defender], T.move_ids[move])
                checked += 1
    assert checked == 30 * 30 * 4


def test_effectiveness_matches_oracle():
    for move_type in range(18):
        for defender in range(len(T.species_ids)):
            assert Fraction(effectiveness4(T, move_type, defender), 4) == oracle_effectiveness(move_type, defender)


def test_damage_by_hand():
    eq = T.move_ids.index("earthquake")
    icebeam = T.move_ids.index("icebeam")
    # Garchomp Earthquake into Snorlax: base (22*100*150//85)//50 + 2 = 79, STAB -> 118
    assert damage(T, S["garchomp"], S["snorlax"], eq) == 118
    # Lapras Ice Beam into Garchomp: base 41, STAB and 4x -> 246
    assert damage(T, S["lapras"], S["garchomp"], icebeam) == 246


def test_immunities_deal_zero():
    assert damage(T, S["jolteon"], S["gengar"], T.move_ids.index("hypervoice")) == 0
    assert damage(T, S["garchomp"], S["corviknight"], T.move_ids.index("earthquake")) == 0
    assert damage(T, S["gardevoir"], S["tyranitar"], T.move_ids.index("psychic")) == 0


def test_damage_is_at_least_one_unless_immune():
    for attacker in range(30):
        for defender in range(30):
            for move in T.species_moves[attacker]:
                d = damage(T, attacker, defender, int(move))
                assert (d == 0) == (effectiveness4(T, int(T.move_type[move]), defender) == 0)


# Move order

def test_priority_beats_speed():
    b = duel("scizor", "garchomp")
    assert T.species_stats[S["scizor"], SPE] < T.species_stats[S["garchomp"], SPE]
    _, events = step(b, (slot("scizor", "bulletpunch"), slot("garchomp", "dragonclaw")))
    assert events[0] == {"type": "order", "first": 0, "reason": "priority"}


def test_faster_acts_first_at_equal_priority():
    b = duel("scizor", "garchomp")
    _, events = step(b, (slot("scizor", "xscissor"), slot("garchomp", "dragonclaw")))
    assert events[0] == {"type": "order", "first": 1, "reason": "speed"}


def test_speed_ties_use_the_seeded_tie_breaker():
    firsts = set()
    for seed in range(64):
        b = duel("garchomp", "garchomp", seed)
        nb, events = step(b, (slot("garchomp", "dragonclaw"), slot("garchomp", "dragonclaw")))
        assert events[0]["reason"] == "tie"
        assert events[0]["first"] == rand32(seed, 1, 0) & 1
        assert step(b, (1, 1))[0] == step(b, (1, 1))[0]
        firsts.add(events[0]["first"])
    assert firsts == {0, 1}


# Fainting, victory, turn limit

def test_fainted_pokemon_does_not_move():
    # Weavile outspeeds Garchomp and Icicle Crash (4x) knocks it out first.
    b = duel("weavile", "garchomp")
    nb, events = step(b, (slot("weavile", "iciclecrash"), slot("garchomp", "earthquake")))
    assert [e["type"] for e in events] == ["order", "move", "faint", "end"]
    assert nb.hp == ((b.hp[0][0],), (0,))
    assert nb.outcome == 0 and nb.phase == DONE


def test_turn_limit_is_a_draw():
    b = replace(duel("snorlax", "corviknight"), turn=TURN_LIMIT - 1)
    nb, events = step(b, (slot("snorlax", "bodyslam"), slot("corviknight", "rocksmash")))
    assert min(nb.hp[0][0], nb.hp[1][0]) > 0
    assert nb.outcome == DRAW and nb.phase == DONE and nb.turn == TURN_LIMIT
    assert events[-1] == {"type": "end", "outcome": DRAW, "reason": "turn_limit"}


def test_no_steps_after_the_end():
    b = replace(duel("snorlax", "lapras"), phase=DONE, outcome=0)
    with pytest.raises(ValueError):
        step(b, (0, 0))


# Action masks

def test_1v1_mask_is_the_four_moves():
    b = duel("gengar", "lapras")
    for side in (0, 1):
        assert action_mask(b, side).tolist() == [True] * 4 + [False] * 3
    with pytest.raises(ValueError):
        step(b, (4, 0))


def test_3v3_mask_allows_switching_to_living_teammates():
    b = new_battle([[S["gengar"], S["lapras"], S["scizor"]], [S["garchomp"], S["snorlax"], S["jolteon"]]], 0)
    assert action_mask(b, 0).tolist() == [True] * 4 + [False, True, True]
    b = replace(b, hp=((100, 0, 50), b.hp[1]))
    assert action_mask(b, 0).tolist() == [True] * 4 + [False, False, True]
    assert action_mask(b, 0).shape == (N_ACTIONS,)


def test_teams_must_be_distinct_species():
    with pytest.raises(ValueError):
        new_battle([[S["gengar"], S["gengar"]], [S["lapras"]]], 0)


# Randomized battles against the independent oracle

def random_battles(n: int, seed: int):
    rng = random.Random(seed)
    for _ in range(n):
        a, b = rng.randrange(30), rng.randrange(30)
        yield new_battle([[a], [b]], rng.randrange(2**32)), rng


@pytest.mark.parametrize("chunk", range(4))
def test_random_battles_match_the_oracle(chunk):
    turns = 0
    for b, rng in random_battles(1000, seed=chunk):
        history = []
        while b.phase != DONE:
            actions = (rng.randrange(4), rng.randrange(4))
            expected_hp, winner = oracle_turn(b, actions)
            before = b
            b, events = step(b, actions)
            history.append((actions, b, events))
            turns += 1

            assert (b.hp[0][0], b.hp[1][0]) == expected_hp
            assert b.turn == before.turn + 1 <= TURN_LIMIT
            assert all(0 <= h <= before.hp[s][0] for s, h in enumerate((b.hp[0][0], b.hp[1][0])))
            if winner is not None:
                assert b.outcome == winner and b.hp[1 - winner][0] == 0 and b.hp[winner][0] > 0
            elif b.turn == TURN_LIMIT:
                assert b.outcome == DRAW
            else:
                assert b.outcome == ONGOING and b.phase == NORMAL
            # The first mover winning means its one attack ended the battle; otherwise both act.
            moves = [e for e in events if e["type"] == "move"]
            assert len(moves) == (1 if winner is not None and winner == events[0]["first"] else 2)

        # Replaying the same actions from the same start reproduces everything.
        replay = new_battle(history[0][1].teams, history[0][1].seed)
        for actions, state, events in history:
            replay, replay_events = step(replay, actions)
            assert replay == state and replay_events == events
    assert turns > 1000


def test_every_random_battle_ends():
    outcomes = {0: 0, 1: 0, DRAW: 0}
    for b, rng in random_battles(500, seed=99):
        while b.phase != DONE:
            b, _ = step(b, (rng.randrange(4), rng.randrange(4)))
        outcomes[b.outcome] += 1
    assert outcomes[0] > 0 and outcomes[1] > 0
