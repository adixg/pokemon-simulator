"""CPU reference engine (RULES.md). Clarity over speed: every faster engine is
tested against this one.

A battle is an immutable `Battle`; `step` returns a new one plus the events
that happened, so search can branch from any state without copying.

A normal turn resolves voluntary switches, then attacks in priority/speed
order, then faints and the victory check. A faint with teammates left puts the
battle in FORCED_SWITCH, where only the side(s) whose active Pokémon fainted
choose a replacement; that phase is not a turn.
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


def must_replace(b: Battle, side: int) -> bool:
    """True when the side's active Pokémon has fainted and it must send in another."""
    return b.phase == FORCED_SWITCH and b.hp[side][b.active[side]] == 0


def action_mask(b: Battle, side: int) -> np.ndarray:
    """Which of the 7 actions the side may take (RULES.md section 5).

    All false when the side has no decision: the battle is over, or the other
    side is replacing a fainted Pokémon.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if b.phase == NORMAL:
        mask[:N_MOVES] = True
    if b.phase == NORMAL or must_replace(b, side):
        for slot, hp in enumerate(b.hp[side]):
            mask[SWITCH_BASE + slot] = hp > 0 and slot != b.active[side]
    return mask


def _order(t: Tables, b: Battle, moves: tuple[int, int], turn: int) -> tuple[int, str]:
    """The side that acts first when both attack, and why (RULES.md section 6)."""
    p0, p1 = (int(t.move_priority[m]) for m in moves)
    if p0 != p1:
        return (0 if p0 > p1 else 1), "priority"
    s0, s1 = (int(t.species_stats[active_species(b, side), SPE]) for side in (0, 1))
    if s0 != s1:
        return (0 if s0 > s1 else 1), "speed"
    return rand32(b.seed, turn, SPEED_TIE_DRAW) & 1, "tie"


def step(b: Battle, actions: tuple[int, int], tables: Tables | None = None) -> tuple[Battle, list[dict]]:
    """Resolve one turn, or one forced-switch phase. Returns the new battle and its events.

    The action of a side with no decision (an all-false mask) is ignored.
    """
    t = tables or load_tables()
    if b.phase == DONE:
        raise ValueError("the battle is over")
    for side, action in enumerate(actions):
        mask = action_mask(b, side)
        if mask.any() and not (0 <= action < N_ACTIONS and mask[action]):
            raise ValueError(f"side {side}: action {action} is not legal")
    if b.phase == FORCED_SWITCH:
        return _replace_fainted(b, actions)

    turn = b.turn + 1
    active = list(b.active)
    hp = [list(b.hp[0]), list(b.hp[1])]
    events: list[dict] = []

    # 1. Voluntary switches, before any attack.
    for side in (0, 1):
        if actions[side] >= SWITCH_BASE:
            to = actions[side] - SWITCH_BASE
            events.append({"type": "switch", "side": side, "from": active[side], "to": to, "forced": False})
            active[side] = to

    # 2-4. Attacks: priority, then speed, then the seeded tie breaker.
    attackers = [side for side in (0, 1) if actions[side] < SWITCH_BASE]
    moves = {side: int(t.species_moves[active_species(b, side), actions[side]]) for side in attackers}
    if len(attackers) == 2:
        first, reason = _order(t, b, (moves[0], moves[1]), turn)
        events.append({"type": "order", "first": first, "reason": reason})
        attackers = [first, 1 - first]

    # 5-6. Faints cancel the fainted Pokémon's move; an empty team loses.
    outcome = ONGOING
    for side in attackers:
        foe = 1 - side
        if hp[side][active[side]] == 0:
            continue
        defender = b.teams[foe][active[foe]]
        dealt = min(damage(t, b.teams[side][active[side]], defender, moves[side]), hp[foe][active[foe]])
        hp[foe][active[foe]] -= dealt
        events.append({
            "type": "move", "side": side, "move": t.move_ids[moves[side]], "target": t.species_ids[defender],
            "damage": dealt, "eff4": effectiveness4(t, int(t.move_type[moves[side]]), defender),
            "hp_after": hp[foe][active[foe]],
        })
        if hp[foe][active[foe]] == 0:
            events.append({"type": "faint", "side": foe, "slot": active[foe]})
            if not any(hp[foe]):
                outcome = side
                break

    if outcome == ONGOING and turn >= TURN_LIMIT:
        outcome = DRAW
    replacing = [side for side in (0, 1) if hp[side][active[side]] == 0]
    if outcome != ONGOING:
        phase = DONE
        events.append({"type": "end", "outcome": outcome,
                       "reason": "turn_limit" if outcome == DRAW else "all_fainted"})
    elif replacing:
        phase = FORCED_SWITCH
        events.append({"type": "forced_switch", "sides": replacing})
    else:
        phase = NORMAL
    nxt = replace(b, hp=(tuple(hp[0]), tuple(hp[1])), active=(active[0], active[1]),
                  phase=phase, turn=turn, outcome=outcome)
    return nxt, events


def _replace_fainted(b: Battle, actions: tuple[int, int]) -> tuple[Battle, list[dict]]:
    """The forced-switch phase: each side whose active Pokémon fainted sends in its choice."""
    active = list(b.active)
    events = []
    for side in (0, 1):
        if must_replace(b, side):
            to = actions[side] - SWITCH_BASE
            events.append({"type": "switch", "side": side, "from": active[side], "to": to, "forced": True})
            active[side] = to
    return replace(b, active=(active[0], active[1]), phase=NORMAL), events
