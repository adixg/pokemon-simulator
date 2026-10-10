"""The move-eligibility rules in tools/build_data.py, with and without the step 5b mechanics.

Uses hand-written move records shaped like Showdown's moves.json, so it needs
no downloaded data.
"""

import pytest

from build_data import MECHANICS, move_problems, move_record


def move(**overrides) -> dict:
    record = {"accuracy": 100, "basePower": 80, "category": "Physical", "name": "Test",
              "priority": 0, "flags": {}, "target": "normal", "type": "Normal"}
    return record | overrides


HYDRO_PUMP = move(name="Hydro Pump", accuracy=80, basePower=110, category="Special", type="Water")
BRAVE_BIRD = move(name="Brave Bird", basePower=120, type="Flying", recoil=[33, 100])
GIGA_DRAIN = move(name="Giga Drain", basePower=75, category="Special", type="Grass", drain=[1, 2])
BULLET_SEED = move(name="Bullet Seed", basePower=25, type="Grass", multihit=[2, 5])
BODY_PRESS = move(name="Body Press", type="Fighting", overrideOffensiveStat="def")
PSYSHOCK = move(name="Psyshock", category="Special", type="Psychic", overrideDefensiveStat="def")
SUCKER_PUNCH = move(name="Sucker Punch", basePower=70, priority=1, type="Dark")

STEP_5B = [
    ("hydropump", HYDRO_PUMP, "accuracy"),
    ("bravebird", BRAVE_BIRD, "recoil"),
    ("gigadrain", GIGA_DRAIN, "drain"),
    ("bulletseed", BULLET_SEED, "multihit"),
    ("bodypress", BODY_PRESS, "stat_override"),
    ("psyshock", PSYSHOCK, "stat_override"),
    ("suckerpunch", SUCKER_PUNCH, "suckerpunch"),
]


def test_plain_move_is_allowed():
    assert move_problems("tackle", move()) == []


@pytest.mark.parametrize("move_id,record,mechanic", STEP_5B)
def test_excluded_in_v1(move_id, record, mechanic):
    assert move_problems(move_id, record) != []


@pytest.mark.parametrize("move_id,record,mechanic", STEP_5B)
def test_allowed_once_its_mechanic_is_enabled(move_id, record, mechanic):
    assert move_problems(move_id, record, frozenset({mechanic})) == []


@pytest.mark.parametrize("move_id,record,mechanic", STEP_5B)
def test_other_mechanics_do_not_unlock_it(move_id, record, mechanic):
    others = frozenset(MECHANICS) - {mechanic}
    assert move_problems(move_id, record, others) != []


def test_step_5b_never_unlocks_state_dependent_mechanics():
    everything = frozenset(MECHANICS)
    assert move_problems("flamecharge", move(self={"boosts": {"spe": 1}}), everything)
    assert move_problems("knockoff", move(basePower=65, type="Dark"), everything)
    assert move_problems("hyperbeam", move(flags={"recharge": 1}), everything)
    assert move_problems("willowisp", move(category="Status", basePower=0, status="brn"), everything)


def test_records_carry_5b_fields():
    assert move_record("bravebird", BRAVE_BIRD)["recoil"] == [33, 100]
    assert move_record("gigadrain", GIGA_DRAIN)["drain"] == [1, 2]
    assert move_record("bulletseed", BULLET_SEED)["multihit"] == [2, 5]
    assert move_record("doublekick", move(multihit=2))["multihit"] == [2, 2]
    assert move_record("suckerpunch", SUCKER_PUNCH)["effect"] == "suckerpunch"
    body_press = move_record("bodypress", BODY_PRESS)
    assert (body_press["attack_stat"], body_press["defense_stat"]) == ("def", "def")
    psyshock = move_record("psyshock", PSYSHOCK)
    assert (psyshock["attack_stat"], psyshock["defense_stat"]) == ("spa", "def")
    aerial_ace = move_record("aerialace", move(accuracy=True))
    assert aerial_ace["never_misses"] and aerial_ace["accuracy"] == 100
