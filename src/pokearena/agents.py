"""Baseline agents for completing battles. Phase 7 adds the real classical agents.

An agent maps (battle, side) to an action, and may also report its policy:
the probability it gave each of the 7 actions, which replays record.
"""

import random

import numpy as np

from pokearena.data import Tables, load_tables
from pokearena.engine import N_MOVES, SWITCH_BASE, Battle, action_mask, damage


class Agent:
    name = "agent"

    def act(self, b: Battle, side: int) -> tuple[int, list[float] | None]:
        """The chosen action, and the policy behind it if the agent has one."""
        raise NotImplementedError


class RandomAgent(Agent):
    """Uniform over legal actions."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, b, side):
        mask = action_mask(b, side)
        legal = np.flatnonzero(mask)
        policy = (mask / mask.sum()).tolist()
        return int(self.rng.choice(legal)), policy


class GreedyAgent(Agent):
    """Picks the move that deals the most damage this turn and never switches
    voluntarily. Forced to switch, it sends in the Pokémon whose best move hits
    the opposing active Pokémon hardest. Ties go to the lowest action index.
    """

    name = "greedy"

    def __init__(self, tables: Tables | None = None):
        self.t = tables or load_tables()

    def best_damage(self, attacker: int, defender: int, cap: int) -> tuple[int, int]:
        """(damage capped at the defender's HP, move slot) of the attacker's best move."""
        options = [(min(damage(self.t, attacker, defender, int(m)), cap), -slot)
                   for slot, m in enumerate(self.t.species_moves[attacker])]
        dealt, neg_slot = max(options)
        return dealt, -neg_slot

    def act(self, b, side):
        mask = action_mask(b, side)
        foe = 1 - side
        defender = b.teams[foe][b.active[foe]]
        cap = b.hp[foe][b.active[foe]]
        if mask[:N_MOVES].any():
            return self.best_damage(b.teams[side][b.active[side]], defender, cap)[1], None
        choices = [(self.best_damage(b.teams[side][slot], defender, cap)[0], -slot)
                   for slot in range(len(b.teams[side])) if mask[SWITCH_BASE + slot]]
        return SWITCH_BASE - max(choices)[1], None


AGENTS = {"random": RandomAgent, "greedy": GreedyAgent}


def make_agent(name: str, seed: int = 0) -> Agent:
    if name == "random":
        return RandomAgent(seed)
    if name in AGENTS:
        return AGENTS[name]()
    raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
