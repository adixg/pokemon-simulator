"""Phase 0: the static tables match RULES.md and known Gen 9 facts."""

import numpy as np
import pytest

from pokearena.data import ATK, DEF, HP, NO_TYPE, SPA, SPD, SPE, STAT_KEYS, load_raw, load_tables

T = load_tables()
RAW = load_raw()


def eff(attacker: str, defender: str) -> int:
    return int(T.type_chart[T.types.index(attacker), T.types.index(defender)])


# Type chart

def test_type_chart_shape_and_values():
    assert len(T.types) == 18
    assert T.type_chart.shape == (18, 18)
    assert set(np.unique(T.type_chart)) <= {0, 2, 4, 8}


def test_exactly_the_eight_gen9_immunities():
    immune = {(T.types[a], T.types[d]) for a, d in zip(*np.nonzero(T.type_chart == 0))}
    assert immune == {
        ("Normal", "Ghost"), ("Ghost", "Normal"), ("Fighting", "Ghost"), ("Poison", "Steel"),
        ("Ground", "Flying"), ("Electric", "Ground"), ("Psychic", "Dark"), ("Dragon", "Fairy"),
    }


@pytest.mark.parametrize("attacker,defender,expected", [
    ("Fire", "Grass", 8), ("Water", "Fire", 8), ("Grass", "Water", 8), ("Ice", "Dragon", 8),
    ("Fairy", "Dragon", 8), ("Steel", "Fairy", 8), ("Ghost", "Ghost", 8), ("Dark", "Psychic", 8),
    ("Fighting", "Normal", 8), ("Rock", "Flying", 8), ("Bug", "Psychic", 8), ("Poison", "Fairy", 8),
    ("Fire", "Water", 2), ("Steel", "Steel", 2), ("Fighting", "Fairy", 2), ("Bug", "Fire", 2),
    ("Dragon", "Steel", 2), ("Grass", "Grass", 2), ("Electric", "Dragon", 2),
    ("Normal", "Normal", 4), ("Ghost", "Fighting", 4), ("Dark", "Steel", 4),
])
def test_known_matchups(attacker, defender, expected):
    assert eff(attacker, defender) == expected


def test_dual_type_products_reach_quarter_and_quadruple():
    # Ice against Garchomp (Dragon/Ground) is 4x; Fire against Volcarona (Bug/Fire)
    # is neutral; Ground against Corviknight (Flying/Steel) is immune.
    assert eff("Ice", "Dragon") * eff("Ice", "Ground") == 64      # 4x in quarters squared
    assert eff("Fire", "Bug") * eff("Fire", "Fire") == 16         # 1x
    assert eff("Ground", "Flying") * eff("Ground", "Steel") == 0
    assert eff("Grass", "Fire") * eff("Grass", "Dragon") == 4     # 0.25x


# Stats

def battle_stat(base: int, hp: bool) -> int:
    core = (2 * base + 31) * 50 // 100
    return core + 60 if hp else core + 5


def test_stats_follow_the_level_50_formula():
    for s in RAW["species"]:
        for k in STAT_KEYS:
            assert s["stats"][k] == battle_stat(s["base_stats"][k], k == "hp"), (s["id"], k)


def test_known_stats_by_hand():
    garchomp = T.species_stats[T.species_ids.index("garchomp")]
    assert list(garchomp) == [183, 150, 115, 100, 105, 122]
    snorlax = T.species_stats[T.species_ids.index("snorlax")]
    assert snorlax[HP] == 235 and snorlax[SPE] == 50


# Moves (RULES.md section 3)

def test_moves_are_plain_damaging_moves():
    for m in RAW["moves"]:
        assert m["power"] > 0, m["id"]
        assert m["category"] in ("Physical", "Special"), m["id"]
        assert m["type"] in T.types, m["id"]
        assert -7 <= m["priority"] <= 5, m["id"]
        assert m["accuracy"] >= RAW["rules"]["min_accuracy"], m["id"]


def test_priority_moves_exist():
    priorities = {m["id"]: m["priority"] for m in RAW["moves"]}
    assert priorities["extremespeed"] == 2
    assert priorities["quickattack"] == priorities["bulletpunch"] == priorities["iceshard"] == 1
    assert priorities["earthquake"] == 0


def test_every_move_is_used_and_ids_are_unique():
    used = {m for s in RAW["species"] for m in s["moves"]}
    ids = [m["id"] for m in RAW["moves"]]
    assert len(ids) == len(set(ids))
    assert used == set(ids)


# Species (RULES.md section 8)

def test_roster():
    ids = [s["id"] for s in RAW["species"]]
    assert len(ids) == 30 and len(set(ids)) == 30
    for s in RAW["species"]:
        assert 1 <= len(s["types"]) <= 2, s["id"]
        assert len(s["moves"]) == 4 and len(set(s["moves"])) == 4, s["id"]


def test_roster_covers_every_type():
    species_types = {t for s in RAW["species"] for t in s["types"]}
    move_types = {m["type"] for m in RAW["moves"]}
    assert species_types == set(T.types)
    assert move_types == set(T.types)


def test_every_species_has_a_stab_move():
    stab = {m["id"]: m["type"] for m in RAW["moves"]}
    for s in RAW["species"]:
        assert any(stab[m] in s["types"] for m in s["moves"]), s["id"]


# Arrays

def test_arrays_match_records_and_are_read_only():
    arrays = [T.type_chart, T.move_type, T.move_power, T.move_category, T.move_priority,
              T.species_types, T.species_stats, T.species_moves]
    for a in arrays:
        assert a.dtype == np.int32
        assert not a.flags.writeable
    n_moves, n_species = len(RAW["moves"]), len(RAW["species"])
    assert T.move_power.shape == (n_moves,)
    assert T.species_stats.shape == (n_species, 6)
    assert T.species_moves.shape == (n_species, 4)
    for i, s in enumerate(RAW["species"]):
        assert [T.move_ids[j] for j in T.species_moves[i]] == s["moves"]
        assert T.types[T.species_types[i, 0]] == s["types"][0]
        if len(s["types"]) == 1:
            assert T.species_types[i, 1] == NO_TYPE
    lapras = T.species_ids.index("lapras")
    assert [T.types[t] for t in T.species_types[lapras]] == ["Water", "Ice"]
    assert T.species_stats[:, [ATK, DEF, SPA, SPD]].min() > 0


def test_damage_intermediates_fit_int32():
    # RULES.md section 4: 22 * power * attack is the largest intermediate.
    largest = 22 * int(T.move_power.max()) * int(T.species_stats[:, [ATK, SPA]].max())
    base = largest // int(T.species_stats[:, [DEF, SPD]].min()) // 50 + 2
    assert largest < 2**31 and base * 3 * 16 < 2**31


def test_attack_and_defense_stats_follow_category_in_v1():
    physical = T.move_category == 0
    assert (T.move_attack_stat[physical] == ATK).all() and (T.move_defense_stat[physical] == DEF).all()
    assert (T.move_attack_stat[~physical] == SPA).all() and (T.move_defense_stat[~physical] == SPD).all()


def test_v1_moves_carry_the_step_5b_fields_unused():
    assert RAW["rules"]["mechanics"] == []
    for m in RAW["moves"]:
        assert m["recoil"] is None and m["drain"] is None and m["multihit"] is None, m["id"]
        assert m["effect"] is None, m["id"]
        assert isinstance(m["never_misses"], bool)
