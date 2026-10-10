"""CPU reference engine (RULES.md). Clarity over speed: every faster engine is
tested against this one.

A battle is an immutable `Battle`; `step` returns a new one plus the events
that happened, so search can branch from any state without copying.

Phase 1 covers moves, damage, move order, fainting, victory and the turn
limit. Switching (actions 4-6) and forced switches arrive in Phase 2.
"""

from dataclasses import dataclass, replace

import numpy as np

from pokearena.data import HP, NO_TYPE, SPE, Tables, load_tables
from pokearena.rng import SPEED_TIE_DRAW, rand32

N_ACTIONS = 7
N_MOVES = 4
SWITCH_BASE = 4          # action 4 + i switches to team slot i
TURN_LIMIT = 100

NORMAL, FORCED_SWITCH, DONE = 0, 1, 2
ONGOING, DRAW = -1, 2    # Battle.outcome is ONGOING, DRAW, or the winning side (0 or 1)


@dataclass(frozen=True)
class Battle:
    teams: tuple[tuple[int, ...], tuple[int, ...]]   # species indices; fixed for the battle
    seed: int
    hp: tuple[tuple[int, ...], tuple[int, ...]]
    active: tuple[int, int]
    phase: int = NORMAL
    turn: int = 0            # normal turns resolved so far; the next one is turn + 1
    outcome: int = ONGOING


def new_battle(teams, seed: int, tables: Tables | None = None) -> Battle:
    t = tables or load_tables()
    teams = tuple(tuple(int(s) for s in team) for team in teams)
    for team in teams:
        if not 1 <= len(team) <= 3 or len(set(team)) != len(team):
            raise ValueError(f"a team is 1-3 distinct species, got {team}")
    hp = tuple(tuple(int(t.species_stats[s, HP]) for s in team) for team in teams)
    return Battle(teams=teams, seed=seed, hp=hp, active=(0, 0))


def effectiveness4(t: Tables, move_type: int, defender: int) -> int:
    """Type effectiveness against a species, in quarters: 0, 1, 2, 4, 8 or 16."""
    first, second = t.species_types[defender]
    eff = int(t.type_chart[move_type, first])
    if second != NO_TYPE:
        eff = eff * int(t.type_chart[move_type, second]) // 4
    return eff


def damage(t: Tables, attacker: int, defender: int, move: int) -> int:
    """RULES.md section 4, in integer arithmetic."""
    move_type = int(t.move_type[move])
    attack = int(t.species_stats[attacker, t.move_attack_stat[move]])
    defense = int(t.species_stats[defender, t.move_defense_stat[move]])
    base = (22 * int(t.move_power[move]) * attack // defense) // 50 + 2
    stab2 = 3 if move_type in t.species_types[attacker] else 2
    eff4 = effectiveness4(t, move_type, defender)
    if eff4 == 0:
        return 0
    return max(1, base * stab2 * eff4 // 8)


def active_species(b: Battle, side: int) -> int:
    return b.teams[side][b.active[side]]


def action_mask(b: Battle, side: int) -> np.ndarray:
    """Which of the 7 actions the side may take (RULES.md section 5)."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if b.phase == NORMAL:
        mask[:N_MOVES] = True
        for slot, hp in enumerate(b.hp[side]):
            mask[SWITCH_BASE + slot] = hp > 0 and slot != b.active[side]
    return mask


def _order(t: Tables, b: Battle, moves: tuple[int, int], turn: int) -> tuple[int, str]:
    """The side that acts first, and why (RULES.md section 6)."""
    p0, p1 = (int(t.move_priority[m]) for m in moves)
    if p0 != p1:
        return (0 if p0 > p1 else 1), "priority"
    s0, s1 = (int(t.species_stats[active_species(b, side), SPE]) for side in (0, 1))
    if s0 != s1:
        return (0 if s0 > s1 else 1), "speed"
    return rand32(b.seed, turn, SPEED_TIE_DRAW) & 1, "tie"


def step(b: Battle, actions: tuple[int, int], tables: Tables | None = None) -> tuple[Battle, list[dict]]:
    """Resolve one turn. Returns the new battle and its events."""
    t = tables or load_tables()
    if b.phase == DONE:
        raise ValueError("the battle is over")
    for side, action in enumerate(actions):
        if not action_mask(b, side)[action]:
            raise ValueError(f"side {side}: action {action} is not legal")
        if action >= SWITCH_BASE:
            raise NotImplementedError("switching arrives in Phase 2")

    turn = b.turn + 1
    moves = tuple(int(t.species_moves[active_species(b, side), actions[side]]) for side in (0, 1))
    first, reason = _order(t, b, moves, turn)
    events: list[dict] = [{"type": "order", "first": first, "reason": reason}]
    hp = [list(b.hp[0]), list(b.hp[1])]
    outcome = ONGOING

    for side in (first, 1 - first):
        foe = 1 - side
        attacker, defender = active_species(b, side), active_species(b, foe)
        if hp[side][b.active[side]] == 0:
            break                                   # fainted before acting: move cancelled
        dealt = min(damage(t, attacker, defender, moves[side]), hp[foe][b.active[foe]])
        hp[foe][b.active[foe]] -= dealt
        events.append({
            "type": "move", "side": side, "move": t.move_ids[moves[side]],
            "damage": dealt, "eff4": effectiveness4(t, int(t.move_type[moves[side]]), defender),
            "hp_after": hp[foe][b.active[foe]],
        })
        if hp[foe][b.active[foe]] == 0:
            events.append({"type": "faint", "side": foe, "slot": b.active[foe]})
            if any(hp[foe]):
                raise NotImplementedError("forced switches arrive in Phase 2")
            outcome = side

    if outcome == ONGOING and turn >= TURN_LIMIT:
        outcome = DRAW
    if outcome != ONGOING:
        events.append({"type": "end", "outcome": outcome, "reason": "turn_limit" if outcome == DRAW else "all_fainted"})
    nxt = replace(
        b, hp=(tuple(hp[0]), tuple(hp[1])), turn=turn, outcome=outcome,
        phase=DONE if outcome != ONGOING else NORMAL,
    )
    return nxt, events
