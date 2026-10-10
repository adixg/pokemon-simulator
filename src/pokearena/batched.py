"""Batched engine: N battles stepped at once with NumPy array operations.

Structure of arrays: each field of the battle state is one array with the
battle as its leading axis, and a step is a fixed sequence of whole-array
operations with no per-battle Python. Battles in different phases (normal
turn, forced switch, finished) share a batch; each operation is masked to the
battles it applies to. This file is the template for the PyTorch engine in
Phase 4, which repeats the same operations on GPU tensors.

Teams always have TEAM_SIZE slots. A smaller team is padded with species
PAD and 0 HP, which the rules already treat as fainted: it can never be
switched to and never counts as a remaining Pokémon. So a 1v1 battle is a
3v3 battle with four padded slots, and behaves identically.

Damage depends only on (attacker species, defender species, move slot) in
v1, so it is precomputed into a [S, S, 4] table once (`damage_table`); the
step is then gathers and comparisons. `damage_arithmetic` computes the same
values from the stat tables on the fly, for comparison.

Every state transition must equal the reference engine's; tests/test_batched.py
checks this battle by battle.
"""

from dataclasses import dataclass, replace

import numpy as np

from pokearena import engine
from pokearena.data import HP, NO_TYPE, SPE, Tables, load_tables
from pokearena.engine import (
    DONE,
    DRAW,
    FORCED_SWITCH,
    N_ACTIONS,
    N_MOVES,
    NORMAL,
    ONGOING,
    SWITCH_BASE,
    TURN_LIMIT,
    Battle,
)
from pokearena.rng import MASK32

TEAM_SIZE = 3
PAD = -1


@dataclass(frozen=True)
class BatchState:
    species: np.ndarray   # [N, 2, 3] int32, PAD for empty slots; fixed for each battle
    hp: np.ndarray        # [N, 2, 3] int32
    active: np.ndarray    # [N, 2]    int32
    phase: np.ndarray     # [N]       int32
    turn: np.ndarray      # [N]       int32
    outcome: np.ndarray   # [N]       int32
    seed: np.ndarray      # [N]       int64, a 32-bit value

    def __len__(self) -> int:
        return len(self.phase)

    def nbytes_per_battle(self) -> int:
        return sum(getattr(self, f).nbytes for f in self.__dataclass_fields__) // len(self)


@dataclass(frozen=True)
class Precomputed:
    """Per-species and per-move lookups the step gathers from."""
    damage: np.ndarray     # [S, S, 4] damage of attacker's move slot against defender
    move_of: np.ndarray    # [S, 4] move index of each species' move slot
    priority: np.ndarray   # [M]
    speed: np.ndarray      # [S]
    max_hp: np.ndarray     # [S]
    tables: Tables


def damage_table(t: Tables) -> np.ndarray:
    n = len(t.species_ids)
    table = np.zeros((n, n, N_MOVES), dtype=np.int32)
    for a in range(n):
        for d in range(n):
            for s in range(N_MOVES):
                table[a, d, s] = engine.damage(t, a, d, int(t.species_moves[a, s]))
    return table


def precompute(t: Tables | None = None) -> Precomputed:
    t = t or load_tables()
    return Precomputed(
        damage=damage_table(t),
        move_of=np.asarray(t.species_moves),
        priority=np.asarray(t.move_priority),
        speed=np.asarray(t.species_stats[:, SPE]),
        max_hp=np.asarray(t.species_stats[:, HP]),
        tables=t,
    )


def damage_arithmetic(t: Tables, attacker: np.ndarray, defender: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """RULES.md section 4 over arrays, without the precomputed table."""
    move = t.species_moves[attacker, slot]
    move_type = t.move_type[move]
    attack = t.species_stats[attacker, t.move_attack_stat[move]]
    defense = t.species_stats[defender, t.move_defense_stat[move]]
    base = (22 * t.move_power[move] * attack // defense) // 50 + 2
    stab2 = np.where((t.species_types[attacker] == move_type[..., None]).any(-1), 3, 2)
    first, second = t.species_types[defender, 0], t.species_types[defender, 1]
    eff4 = t.type_chart[move_type, first]
    eff4 = np.where(second != NO_TYPE, eff4 * t.type_chart[move_type, np.maximum(second, 0)] // 4, eff4)
    dealt = np.maximum(1, base * stab2 * eff4 // 8)
    return np.where(eff4 == 0, 0, dealt).astype(np.int32)


def hash32(x: np.ndarray) -> np.ndarray:
    """rng.hash32 over int64 arrays, masking to 32 bits after every multiply."""
    x = x & MASK32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & MASK32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & MASK32
    return x ^ (x >> 16)


def rand32(seed: np.ndarray, turn: np.ndarray, draw: int) -> np.ndarray:
    return hash32(seed ^ hash32((turn.astype(np.int64) << 8) | draw))


# Building and converting

def new_batch(teams: np.ndarray, seeds: np.ndarray, pre: Precomputed) -> BatchState:
    """teams: [N, 2, 3] species indices (PAD for empty slots); seeds: [N]."""
    species = np.asarray(teams, dtype=np.int32)
    n = len(species)
    hp = np.where(species == PAD, 0, pre.max_hp[np.maximum(species, 0)]).astype(np.int32)
    return BatchState(
        species=species, hp=hp,
        active=np.zeros((n, 2), dtype=np.int32),
        phase=np.full(n, NORMAL, dtype=np.int32),
        turn=np.zeros(n, dtype=np.int32),
        outcome=np.full(n, ONGOING, dtype=np.int32),
        seed=np.asarray(seeds, dtype=np.int64) & MASK32,
    )


def from_battles(battles: list[Battle]) -> BatchState:
    species = np.full((len(battles), 2, TEAM_SIZE), PAD, dtype=np.int32)
    hp = np.zeros((len(battles), 2, TEAM_SIZE), dtype=np.int32)
    for i, b in enumerate(battles):
        for side in (0, 1):
            species[i, side, :len(b.teams[side])] = b.teams[side]
            hp[i, side, :len(b.hp[side])] = b.hp[side]
    return BatchState(
        species=species, hp=hp,
        active=np.array([b.active for b in battles], dtype=np.int32),
        phase=np.array([b.phase for b in battles], dtype=np.int32),
        turn=np.array([b.turn for b in battles], dtype=np.int32),
        outcome=np.array([b.outcome for b in battles], dtype=np.int32),
        seed=np.array([b.seed & MASK32 for b in battles], dtype=np.int64),
    )


def to_battles(s: BatchState) -> list[Battle]:
    battles = []
    for i in range(len(s)):
        size = [int((s.species[i, side] != PAD).sum()) for side in (0, 1)]
        battles.append(Battle(
            teams=tuple(tuple(int(x) for x in s.species[i, side, :size[side]]) for side in (0, 1)),
            seed=int(s.seed[i]),
            hp=tuple(tuple(int(x) for x in s.hp[i, side, :size[side]]) for side in (0, 1)),
            active=(int(s.active[i, 0]), int(s.active[i, 1])),
            phase=int(s.phase[i]), turn=int(s.turn[i]), outcome=int(s.outcome[i]),
        ))
    return battles


# Stepping

def _take(a: np.ndarray, index: np.ndarray) -> np.ndarray:
    """a[n, side, index[n, side]] for [N, 2, K] a and [N, 2] index."""
    return np.take_along_axis(a, index[..., None], axis=2)[..., 0]


def action_masks(s: BatchState) -> np.ndarray:
    """[N, 2, 7] legal actions, as engine.action_mask for every battle and side."""
    normal = (s.phase == NORMAL)[:, None]                              # [N, 1]
    active_fainted = _take(s.hp, s.active) == 0                         # [N, 2]
    replacing = (s.phase == FORCED_SWITCH)[:, None] & active_fainted    # [N, 2]
    slots = np.arange(TEAM_SIZE)
    can_switch_to = (s.hp > 0) & (slots != s.active[..., None])         # [N, 2, 3]
    masks = np.zeros((len(s), 2, N_ACTIONS), dtype=bool)
    masks[..., :N_MOVES] = normal[..., None]
    masks[..., SWITCH_BASE:] = can_switch_to & (normal | replacing)[..., None]
    return masks


def step(s: BatchState, actions: np.ndarray, pre: Precomputed, check: bool = True,
         table_damage: bool = True) -> BatchState:
    """Advance every unfinished battle by one step. actions: [N, 2] int.

    Finished battles are left unchanged, and the action of a side with no
    decision is ignored, as in engine.step. With `check`, illegal actions raise.
    With `table_damage` off, damage is computed arithmetically instead of looked up.
    """
    actions = np.asarray(actions)
    if check:
        masks = action_masks(s)
        has_choice = masks.any(-1)
        legal = np.take_along_axis(masks, np.clip(actions, 0, N_ACTIONS - 1)[..., None], axis=2)[..., 0]
        bad = has_choice & ~(legal & (actions >= 0) & (actions < N_ACTIONS))
        if bad.any():
            n, side = np.argwhere(bad)[0]
            raise ValueError(f"battle {n} side {side}: action {actions[n, side]} is not legal")

    normal = s.phase == NORMAL                                           # [N]
    forced = s.phase == FORCED_SWITCH
    hp = s.hp.copy()
    rows = np.arange(len(s))

    # Switches: voluntary ones in normal turns, replacements in forced-switch phases.
    switching = actions >= SWITCH_BASE                                   # [N, 2]
    replacing = forced[:, None] & (_take(s.hp, s.active) == 0)
    switches = (normal[:, None] & switching) | replacing
    active = np.where(switches, actions - SWITCH_BASE, s.active).astype(np.int32)

    # Attacks, in normal turns only. The attacker did not switch, so its species
    # is the same before and after; the defender is whoever is in now.
    attacking = normal[:, None] & ~switching                             # [N, 2]
    me = _take(s.species, active)                                        # [N, 2] active species
    foe = me[:, ::-1]
    slot = np.clip(actions, 0, N_MOVES - 1)
    if table_damage:
        dealt = pre.damage[me, foe, slot]                                # [N, 2]
    else:
        dealt = damage_arithmetic(pre.tables, me, foe, slot)

    turn = np.where(normal, s.turn + 1, s.turn)
    priority = pre.priority[pre.move_of[me, slot]]
    speed = pre.speed[me]
    tie = (rand32(s.seed, turn, 0) & 1).astype(np.int32)
    first = np.where(priority[:, 0] != priority[:, 1], (priority[:, 1] > priority[:, 0]).astype(np.int32),
                     np.where(speed[:, 0] != speed[:, 1], (speed[:, 1] > speed[:, 0]).astype(np.int32), tie))
    # With one attacker, it acts "first"; with none, nothing happens below.
    first = np.where(attacking[:, 0] & attacking[:, 1], first, np.where(attacking[:, 0], 0, 1))

    for order in (first, 1 - first):
        target = 1 - order
        target_slot = active[rows, target]
        acts = attacking[rows, order] & (hp[rows, order, active[rows, order]] > 0)
        remaining = hp[rows, target, target_slot]
        hp[rows, target, target_slot] = np.where(acts, remaining - np.minimum(dealt[rows, order], remaining), remaining)

    # Victory, turn limit, and the next phase.
    out = np.full(len(s), ONGOING, dtype=np.int32)
    out = np.where(normal & ~hp[:, 1].any(-1), 0, out)
    out = np.where(normal & ~hp[:, 0].any(-1), 1, out)
    out = np.where(normal & (out == ONGOING) & (turn >= TURN_LIMIT), DRAW, out)
    active_fainted = (_take(hp, active) == 0).any(-1)
    phase = np.where(out != ONGOING, DONE, np.where(normal & active_fainted, FORCED_SWITCH, NORMAL))
    phase = np.where(normal | forced, phase, s.phase).astype(np.int32)
    outcome = np.where(normal, out, s.outcome).astype(np.int32)

    return replace(s, hp=hp, active=active, phase=phase, turn=turn.astype(np.int32), outcome=outcome)


def random_actions(s: BatchState, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random legal action for every side ([N, 2]; 0 where there is no choice)."""
    masks = action_masks(s)
    scores = rng.random(masks.shape) * masks
    return scores.argmax(-1)


def reset_done(s: BatchState, teams: np.ndarray, seeds: np.ndarray, pre: Precomputed) -> BatchState:
    """Replace every finished battle with a new one from teams[i] / seeds[i] ([N, 2, 3] / [N])."""
    done = s.phase == DONE
    fresh = new_batch(teams, seeds, pre)
    pick = lambda new, old: np.where(done.reshape(-1, *([1] * (old.ndim - 1))), new, old)  # noqa: E731
    return BatchState(**{f: pick(getattr(fresh, f), getattr(s, f)) for f in s.__dataclass_fields__})
