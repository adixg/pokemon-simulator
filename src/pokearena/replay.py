"""Replay files: (teams, seed, actions) plus the events of every turn.

A battle is fully determined by its teams, seed and actions (RULES.md section
10), so a replay can always be re-run; the events are stored as well so that
viewers never need the engine. `verify` re-runs a replay and checks that it
reproduces every event, which makes any replay a determinism test.

The GPU engine records actions only. The CPU engine regenerates the events.
"""

import json
from pathlib import Path

from pokearena.agents import Agent
from pokearena.data import Tables, load_raw, load_tables
from pokearena.engine import DONE, FORCED_SWITCH, Battle, action_mask, new_battle, step

FORMAT = "pokearena-replay"
VERSION = 1
PHASE_NAMES = {0: "normal", FORCED_SWITCH: "forced_switch"}


def play(teams, seed: int, agents: tuple[Agent, Agent], tables: Tables | None = None) -> dict:
    """Play a battle between two agents and return its replay."""
    t = tables or load_tables()
    b = new_battle(teams, seed, t)
    turns = []
    while b.phase != DONE:
        actions, policies = [], []
        for side in (0, 1):
            if action_mask(b, side).any():
                action, policy = agents[side].act(b, side)
            else:
                action, policy = None, None
            actions.append(action)
            policies.append(policy)
        phase = b.phase
        b, events = step(b, tuple(a if a is not None else 0 for a in actions), t)
        record = {"turn": b.turn, "phase": PHASE_NAMES[phase], "actions": actions, "events": events}
        if any(p is not None for p in policies):
            record["policy"] = [None if p is None else [round(x, 4) for x in p] for p in policies]
        turns.append(record)
    return {
        "format": FORMAT,
        "version": VERSION,
        "data_version": load_raw()["version"],
        "seed": seed,
        "players": [agents[0].name, agents[1].name],
        "teams": [[t.species_ids[s] for s in team] for team in b.teams],
        "turns": turns,
        "result": {"outcome": b.outcome, "turns": b.turn},
    }


def initial_battle(replay: dict, tables: Tables | None = None) -> Battle:
    t = tables or load_tables()
    index = {s: i for i, s in enumerate(t.species_ids)}
    return new_battle([[index[s] for s in team] for team in replay["teams"]], replay["seed"], t)


def verify(replay: dict, tables: Tables | None = None) -> Battle:
    """Re-run the replay; raise AssertionError at the first turn that differs. Returns the final battle."""
    t = tables or load_tables()
    if replay.get("format") != FORMAT or replay.get("version") != VERSION:
        raise ValueError("not a version 1 pokearena replay")
    b = initial_battle(replay, t)
    for i, record in enumerate(replay["turns"]):
        actions = tuple(a if a is not None else 0 for a in record["actions"])
        b, events = step(b, actions, t)
        assert events == record["events"], f"turn record {i} (turn {record['turn']}): events differ"
    assert b.phase == DONE and b.outcome == replay["result"]["outcome"], "final outcome differs"
    return b


def save(replay: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(replay, separators=(",", ":")) + "\n")


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
