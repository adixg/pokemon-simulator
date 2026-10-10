"""Phase 2: switching, forced switches, full 3v3 battles, agents, replays and rendering."""

import io
import random
from dataclasses import replace

import pytest
from rich.console import Console
from test_engine import oracle_damage

from pokearena import replay as replays
from pokearena.agents import Agent, GreedyAgent, RandomAgent
from pokearena.cli import main as cli_main
from pokearena.data import SPE, load_tables
from pokearena.engine import (
    DONE,
    DRAW,
    FORCED_SWITCH,
    NORMAL,
    ONGOING,
    SWITCH_BASE,
    TURN_LIMIT,
    Battle,
    action_mask,
    new_battle,
    step,
)
from pokearena.render import render
from pokearena.rng import rand32

T = load_tables()
S = {name: i for i, name in enumerate(T.species_ids)}


def team(*names: str) -> list[int]:
    return [S[n] for n in names]


def slot(species: str, move: str) -> int:
    return [T.move_ids[m] for m in T.species_moves[S[species]]].index(move)


# Switching

def test_switch_resolves_before_the_attack():
    # P0 switches Garchomp out to Corviknight; P1's Garchomp Earthquake hits Corviknight, which is immune.
    b = new_battle([team("garchomp", "corviknight"), team("garchomp")], seed=0)
    nb, events = step(b, (SWITCH_BASE + 1, slot("garchomp", "earthquake")))
    assert events[0] == {"type": "switch", "side": 0, "from": 0, "to": 1, "forced": False}
    assert events[1]["type"] == "move" and events[1]["target"] == "corviknight" and events[1]["damage"] == 0
    assert nb.active == (1, 0) and nb.hp == b.hp and nb.turn == 1 and nb.phase == NORMAL


def test_both_switch():
    b = new_battle([team("gengar", "lapras"), team("scizor", "snorlax")], seed=0)
    nb, events = step(b, (SWITCH_BASE + 1, SWITCH_BASE + 1))
    assert [e["type"] for e in events] == ["switch", "switch"]
    assert nb.active == (1, 1) and nb.hp == b.hp and nb.turn == 1


def test_switched_in_pokemon_takes_the_full_hit():
    b = new_battle([team("lapras", "garchomp"), team("weavile")], seed=0)
    nb, events = step(b, (SWITCH_BASE + 1, slot("weavile", "iciclecrash")))
    expected = oracle_damage(S["weavile"], S["garchomp"], T.move_ids.index("iciclecrash"))
    assert events[1]["damage"] == min(expected, b.hp[0][1])
    assert nb.hp[0] == (b.hp[0][0], max(0, b.hp[0][1] - expected))


# Forced switches

def knocked_out_lead():
    # Weavile outspeeds and knocks out Garchomp; P1 still has Snorlax.
    b = new_battle([team("weavile"), team("garchomp", "snorlax")], seed=0)
    return step(b, (slot("weavile", "iciclecrash"), slot("garchomp", "earthquake")))


def test_faint_with_teammates_left_forces_a_switch():
    b, events = knocked_out_lead()
    assert events[-1] == {"type": "forced_switch", "sides": [1]}
    assert b.phase == FORCED_SWITCH and b.outcome == ONGOING and b.turn == 1
    assert action_mask(b, 0).tolist() == [False] * 7
    assert action_mask(b, 1).tolist() == [False] * 4 + [False, True, False]


def test_forced_switch_is_not_a_turn():
    b, _ = knocked_out_lead()
    nb, events = step(b, (0, SWITCH_BASE + 1))                # P0's action is ignored
    assert events == [{"type": "switch", "side": 1, "from": 0, "to": 1, "forced": True}]
    assert nb.phase == NORMAL and nb.turn == 1 and nb.active == (0, 1) and nb.hp == b.hp


def test_attacking_during_a_forced_switch_is_illegal():
    b, _ = knocked_out_lead()
    with pytest.raises(ValueError):
        step(b, (0, 0))


def test_cannot_switch_to_fainted_or_active():
    b = new_battle([team("gengar", "lapras", "scizor"), team("garchomp")], seed=0)
    b = replace(b, hp=((b.hp[0][0], 0, b.hp[0][2]), b.hp[1]))
    for illegal in (SWITCH_BASE, SWITCH_BASE + 1):
        with pytest.raises(ValueError):
            step(b, (illegal, 0))


def test_last_faint_ends_the_battle_without_a_forced_switch():
    b = new_battle([team("weavile"), team("garchomp", "snorlax")], seed=0)
    b = replace(b, hp=(b.hp[0], (b.hp[1][0], 0)))
    nb, events = step(b, (slot("weavile", "iciclecrash"), slot("garchomp", "earthquake")))
    assert nb.phase == DONE and nb.outcome == 0 and events[-1]["type"] == "end"


# Turn limit

class AlwaysSwitch(Agent):
    name = "always-switch"

    def act(self, b, side):
        mask = action_mask(b, side)
        return next(a for a in range(SWITCH_BASE, 7) if mask[a]), None


def test_endless_switching_is_a_draw():
    replay = replays.play([team("gengar", "lapras"), team("scizor", "snorlax")], 0, (AlwaysSwitch(), AlwaysSwitch()))
    assert replay["result"] == {"outcome": DRAW, "turns": TURN_LIMIT}
    assert replay["turns"][-1]["events"][-1] == {"type": "end", "outcome": DRAW, "reason": "turn_limit"}


# Randomized 3v3 battles against an independent oracle

def oracle(b: Battle, actions: tuple[int, int]):
    """Expected (hp, active, phase, outcome) after one step, written from RULES.md section 6."""
    hp = [list(b.hp[0]), list(b.hp[1])]
    active = list(b.active)
    if b.phase == FORCED_SWITCH:
        for side in (0, 1):
            if hp[side][active[side]] == 0:
                active[side] = actions[side] - SWITCH_BASE
        return hp, active, NORMAL, ONGOING
    turn = b.turn + 1
    for side in (0, 1):
        if actions[side] >= SWITCH_BASE:
            active[side] = actions[side] - SWITCH_BASE
    attacking = [s for s in (0, 1) if actions[s] < SWITCH_BASE]
    moves = {s: int(T.species_moves[b.teams[s][b.active[s]], actions[s]]) for s in attacking}
    if len(attacking) == 2:
        key = [(int(T.move_priority[moves[s]]), int(T.species_stats[b.teams[s][b.active[s]], SPE])) for s in (0, 1)]
        first = (0 if key[0] > key[1] else 1) if key[0] != key[1] else rand32(b.seed, turn, 0) & 1
        attacking = [first, 1 - first]
    for side in attacking:
        foe = 1 - side
        if hp[side][active[side]] == 0:
            continue
        dealt = oracle_damage(b.teams[side][active[side]], b.teams[foe][active[foe]], moves[side])
        hp[foe][active[foe]] = max(0, hp[foe][active[foe]] - dealt)
        if not any(hp[foe]):
            return hp, active, DONE, side
    if turn >= TURN_LIMIT:
        return hp, active, DONE, DRAW
    if any(hp[s][active[s]] == 0 for s in (0, 1)):
        return hp, active, FORCED_SWITCH, ONGOING
    return hp, active, NORMAL, ONGOING


def random_3v3(rng: random.Random) -> Battle:
    return new_battle([rng.sample(range(30), 3), rng.sample(range(30), 3)], rng.randrange(2**32))


@pytest.mark.parametrize("chunk", range(4))
def test_random_3v3_battles_match_the_oracle(chunk):
    rng = random.Random(1000 + chunk)
    steps = forced = 0
    for _ in range(500):
        b = random_3v3(rng)
        while b.phase != DONE:
            masks = [action_mask(b, s) for s in (0, 1)]
            assert masks[0].any() or masks[1].any()
            if b.phase == NORMAL:
                assert all(b.hp[s][b.active[s]] > 0 for s in (0, 1))
            else:
                forced += 1
                # In v1 only one active Pokémon can faint per turn.
                assert sum(m.any() for m in masks) == 1
            actions = tuple(int(rng.choice([a for a in range(7) if m[a]])) if m.any() else 0 for m in masks)
            hp, active, phase, outcome = oracle(b, actions)
            b, _ = step(b, actions)
            steps += 1
            assert [list(h) for h in b.hp] == hp
            assert list(b.active) == active and b.phase == phase and b.outcome == outcome
            assert b.turn <= TURN_LIMIT
    assert steps > 5000 and forced > 1000


# Agents

def test_greedy_picks_the_most_damaging_move():
    b = new_battle([team("lapras"), team("garchomp")], seed=0)
    action, _ = GreedyAgent().act(b, 0)
    assert T.move_ids[T.species_moves[S["lapras"], action]] == "icebeam"


def test_greedy_sends_in_the_hardest_hitter():
    b, _ = knocked_out_lead()                      # P1 replaces Garchomp facing Weavile
    b = replace(b, teams=(b.teams[0], (S["garchomp"], S["lucario"])))
    assert GreedyAgent().act(b, 1)[0] == SWITCH_BASE + 1


def test_random_agent_only_plays_legal_actions_and_reports_its_policy():
    agent = RandomAgent(seed=0)
    b, _ = knocked_out_lead()
    for _ in range(50):
        action, policy = agent.act(b, 1)
        assert action == SWITCH_BASE + 1 and policy[SWITCH_BASE + 1] == 1.0


def test_greedy_beats_random():
    rng = random.Random(7)
    wins = 0
    for i in range(300):
        teams = [rng.sample(range(30), 3), rng.sample(range(30), 3)]
        side = i % 2                                  # greedy plays both sides equally
        agents = (GreedyAgent(), RandomAgent(i)) if side == 0 else (RandomAgent(i), GreedyAgent())
        wins += replays.play(teams, i, agents)["result"]["outcome"] == side
    assert wins / 300 > 0.8


# Replays

def sample_replay(seed: int = 3) -> dict:
    rng = random.Random(seed)
    teams = [rng.sample(range(30), 3), rng.sample(range(30), 3)]
    return replays.play(teams, seed, (GreedyAgent(), RandomAgent(seed)))


def test_replay_round_trips_and_verifies(tmp_path):
    replay = sample_replay()
    path = tmp_path / "battle.json"
    replays.save(replay, path)
    loaded = replays.load(path)
    assert loaded == replay
    final = replays.verify(loaded)
    assert final.outcome == replay["result"]["outcome"]
    assert replay["format"] == "pokearena-replay" and replay["players"] == ["greedy", "random"]
    assert {r["phase"] for r in replay["turns"]} <= {"normal", "forced_switch"}


def test_verify_catches_a_tampered_replay():
    replay = sample_replay()
    move = next(e for r in replay["turns"] for e in r["events"] if e["type"] == "move")
    move["damage"] += 1
    with pytest.raises(AssertionError):
        replays.verify(replay)


def test_many_replays_verify():
    for seed in range(200):
        replays.verify(sample_replay(seed))


# Rendering

def render_text(replay: dict) -> str:
    console = Console(file=io.StringIO(), width=100, color_system=None)
    render(replay, console)
    return console.file.getvalue()


def test_render_shows_turns_moves_faints_and_result():
    replay = sample_replay()
    text = render_text(replay)
    assert "Turn 1" in text and " used " in text and "fainted!" in text
    assert "wins!" in text or "Draw" in text
    assert text.count("replacement") == sum(r["phase"] == "forced_switch" for r in replay["turns"])


def test_render_tracks_hp_from_events():
    b = new_battle([team("garchomp", "corviknight"), team("garchomp")], seed=0)
    replay = replays.play([team("garchomp", "corviknight"), team("garchomp")], 0, (GreedyAgent(), GreedyAgent()))
    text = render_text(replay)
    assert f"{b.hp[0][0]}/{b.hp[0][0]}" in text            # full HP shown on turn 1
    assert "/" in text.splitlines()[-1]


def test_render_every_kind_of_replay():
    for seed in range(50):
        render_text(sample_replay(seed))
    render_text(replays.play([team("gengar", "lapras"), team("scizor", "snorlax")], 0,
                             (AlwaysSwitch(), AlwaysSwitch())))


def test_cli_play_and_show(tmp_path, capsys):
    path = tmp_path / "b.json"
    assert cli_main(["play", "greedy", "random", "--seed", "5", "--save", str(path), "--quiet"]) == 0
    assert cli_main(["show", str(path), "--verify"]) == 0
    out = capsys.readouterr().out
    assert "verified" in out and "Turn 1" in out
    assert cli_main(["play", "greedy", "greedy", "--teams", "garchomp,gengar/lapras,scizor", "--quiet"]) == 0
