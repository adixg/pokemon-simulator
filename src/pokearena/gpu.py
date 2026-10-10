"""PyTorch engine: the batched engine's step as tensor operations, on GPU or CPU.

The same structure-of-arrays state and the same sequence of masked whole-batch
operations as batched.py, so the two can be compared line by line. Selecting
a team slot is written as a one-hot product instead of a gather or scatter,
which keeps every operation elementwise over [N, 2, 3] and lets torch.compile
fuse the step into a few kernels.

`step` is plain eager PyTorch. `compiled_step(mode)` wraps the same function
with torch.compile; mode "reduce-overhead" also captures it in a CUDA graph.
Every transition must equal the reference engine's (tests/test_gpu.py).
"""

from dataclasses import dataclass, fields

import numpy as np
import torch

from pokearena import batched
from pokearena.engine import DONE, DRAW, FORCED_SWITCH, N_ACTIONS, N_MOVES, NORMAL, ONGOING, SWITCH_BASE, TURN_LIMIT
from pokearena.rng import MASK32

TEAM_SIZE = batched.TEAM_SIZE
PAD = batched.PAD


@dataclass(frozen=True)
class TorchState:
    species: torch.Tensor   # [N, 2, 3] int32
    hp: torch.Tensor        # [N, 2, 3] int32
    active: torch.Tensor    # [N, 2]    int32
    phase: torch.Tensor     # [N]       int32
    turn: torch.Tensor      # [N]       int32
    outcome: torch.Tensor   # [N]       int32
    seed: torch.Tensor      # [N]       int64, a 32-bit value

    def __len__(self) -> int:
        return self.phase.shape[0]

    def astuple(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=True)
class TorchTables:
    damage: torch.Tensor     # [S, S, 4] int32
    priority: torch.Tensor   # [S, 4] int32, priority of each species' move slot
    speed: torch.Tensor      # [S] int32
    max_hp: torch.Tensor     # [S] int32


def tables_on(device: str | torch.device, pre: batched.Precomputed | None = None) -> TorchTables:
    pre = pre or batched.precompute()
    as_tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.int32, device=device)  # noqa: E731
    return TorchTables(
        damage=as_tensor(pre.damage),
        priority=as_tensor(pre.priority[pre.move_of]),
        speed=as_tensor(pre.speed),
        max_hp=as_tensor(pre.max_hp),
    )


# Conversion

def from_batch(s: batched.BatchState, device: str | torch.device) -> TorchState:
    return TorchState(**{f.name: torch.as_tensor(getattr(s, f.name), device=device) for f in fields(TorchState)})


def to_batch(s: TorchState) -> batched.BatchState:
    return batched.BatchState(**{f.name: getattr(s, f.name).cpu().numpy() for f in fields(TorchState)})


def new_batch(teams: torch.Tensor, seeds: torch.Tensor, t: TorchTables) -> TorchState:
    """teams: [N, 2, 3] species (PAD for empty slots); seeds: [N]. Both on the target device."""
    species = teams.to(torch.int32)
    n, device = species.shape[0], species.device
    hp = torch.where(species == PAD, 0, t.max_hp[species.clamp(min=0)])
    return TorchState(
        species=species, hp=hp.to(torch.int32),
        active=torch.zeros((n, 2), dtype=torch.int32, device=device),
        phase=torch.full((n,), NORMAL, dtype=torch.int32, device=device),
        turn=torch.zeros(n, dtype=torch.int32, device=device),
        outcome=torch.full((n,), ONGOING, dtype=torch.int32, device=device),
        seed=seeds.to(torch.int64) & MASK32,
    )


# Random numbers (RULES.md section 10), in int64 masked to 32 bits

def hash32(x: torch.Tensor) -> torch.Tensor:
    x = x & MASK32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & MASK32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & MASK32
    return x ^ (x >> 16)


def rand32(seed: torch.Tensor, turn: torch.Tensor, draw: int) -> torch.Tensor:
    return hash32(seed ^ hash32((turn.to(torch.int64) << 8) | draw))


# Stepping

def _slot_onehot(index: torch.Tensor) -> torch.Tensor:
    """[N, 2] slot indices -> [N, 2, 3] bool."""
    return index.unsqueeze(-1) == torch.arange(TEAM_SIZE, device=index.device, dtype=index.dtype)


def action_masks(s: TorchState) -> torch.Tensor:
    """[N, 2, 7] legal actions, as engine.action_mask for every battle and side."""
    active_hot = _slot_onehot(s.active)
    normal = (s.phase == NORMAL).unsqueeze(-1)                                   # [N, 1]
    active_fainted = ((s.hp * active_hot).sum(-1) == 0)                          # [N, 2]
    replacing = (s.phase == FORCED_SWITCH).unsqueeze(-1) & active_fainted
    can_switch = (s.hp > 0) & ~active_hot & (normal | replacing).unsqueeze(-1)   # [N, 2, 3]
    moves = normal.unsqueeze(-1).expand(-1, 2, N_MOVES)
    return torch.cat([moves, can_switch], dim=-1)


def step_tensors(species, hp, active, phase, turn, outcome, seed, actions, damage, priority, speed):
    """One step over raw tensors (the form torch.compile sees). Returns the new
    (hp, active, phase, turn, outcome); species and seed never change.
    """
    normal = phase == NORMAL                                                     # [N]
    forced = phase == FORCED_SWITCH
    active_hot = _slot_onehot(active)
    active_hp = (hp * active_hot).sum(-1)                                        # [N, 2]

    # Switches: voluntary ones in normal turns, replacements in forced-switch phases.
    switching = actions >= SWITCH_BASE
    switches = (normal.unsqueeze(-1) & switching) | (forced.unsqueeze(-1) & (active_hp == 0))
    active = torch.where(switches, actions - SWITCH_BASE, active).to(torch.int32)
    active_hot = _slot_onehot(active)

    # Attacks in normal turns. Attackers did not switch; defenders are whoever is in now.
    attacking = normal.unsqueeze(-1) & ~switching                                # [N, 2]
    me = (species * active_hot).sum(-1)                                          # [N, 2] active species
    foe = me.flip(-1)
    slot = actions.clamp(0, N_MOVES - 1)
    dealt = damage[me, foe, slot]                                                # [N, 2]

    turn = torch.where(normal, turn + 1, turn)
    prio = priority[me, slot]
    spd = speed[me]
    tie = (rand32(seed, turn, 0) & 1).to(torch.int32)
    by_speed = torch.where(spd[:, 0] != spd[:, 1], (spd[:, 1] > spd[:, 0]).to(torch.int32), tie)
    first = torch.where(prio[:, 0] != prio[:, 1], (prio[:, 1] > prio[:, 0]).to(torch.int32), by_speed)
    both = attacking[:, 0] & attacking[:, 1]
    first = torch.where(both, first, (~attacking[:, 0]).to(torch.int32))         # lone attacker acts first

    side_ids = torch.arange(2, device=hp.device)
    for k in range(2):
        attacker = first if k == 0 else 1 - first                                # [N]
        attacker_hot = side_ids == attacker.unsqueeze(-1)                        # [N, 2]
        alive = ((hp * active_hot).sum(-1) > 0)                                  # [N, 2]
        acts = (attacking & attacker_hot & alive).any(-1)                        # [N]
        hit = (attacker_hot * dealt).sum(-1)                                     # [N] damage the attacker deals
        target_hot = active_hot & ~attacker_hot.unsqueeze(-1)                    # [N, 2, 3] the defender's slot
        remaining = (hp * target_hot).sum((-1, -2))                              # [N]
        loss = torch.where(acts, torch.minimum(hit, remaining), 0)
        hp = hp - target_hot * loss.view(-1, 1, 1)

    # Victory, turn limit, next phase.
    out = torch.full_like(outcome, ONGOING)
    out = torch.where(normal & ~(hp[:, 1] > 0).any(-1), 0, out)
    out = torch.where(normal & ~(hp[:, 0] > 0).any(-1), 1, out)
    out = torch.where(normal & (out == ONGOING) & (turn >= TURN_LIMIT), DRAW, out)
    active_fainted = ((hp * active_hot).sum(-1) == 0).any(-1)
    next_phase = torch.where(out != ONGOING, DONE, torch.where(normal & active_fainted, FORCED_SWITCH, NORMAL))
    phase = torch.where(normal | forced, next_phase, phase).to(torch.int32)
    outcome = torch.where(normal, out, outcome).to(torch.int32)
    return hp.to(torch.int32), active, phase, turn.to(torch.int32), outcome


def check_actions(s: TorchState, actions: torch.Tensor) -> None:
    masks = action_masks(s)
    in_range = (actions >= 0) & (actions < N_ACTIONS)
    legal = torch.gather(masks, 2, actions.clamp(0, N_ACTIONS - 1).long().unsqueeze(-1)).squeeze(-1)
    bad = masks.any(-1) & ~(legal & in_range)
    if bad.any():
        n, side = torch.nonzero(bad)[0].tolist()
        raise ValueError(f"battle {n} side {side}: action {int(actions[n, side])} is not legal")


def step(s: TorchState, actions: torch.Tensor, t: TorchTables, check: bool = True, fn=step_tensors) -> TorchState:
    """Advance every unfinished battle by one step. actions: [N, 2] int32 on the state's device."""
    if check:
        check_actions(s, actions)
    hp, active, phase, turn, outcome = fn(*s.astuple(), actions.to(torch.int32), t.damage, t.priority, t.speed)
    return TorchState(species=s.species, hp=hp, active=active, phase=phase, turn=turn, outcome=outcome, seed=s.seed)


def compiled_step(mode: str | None = None):
    """step_tensors under torch.compile ("default", "max-autotune", or "reduce-overhead" for CUDA graphs)."""
    return torch.compile(step_tensors, mode=mode, dynamic=False)


def random_actions(s: TorchState, generator: torch.Generator | None = None) -> torch.Tensor:
    """A uniformly random legal action per side ([N, 2] int32; 0 where there is no choice)."""
    masks = action_masks(s)
    scores = torch.rand(masks.shape, device=masks.device, generator=generator) * masks
    return scores.argmax(-1).to(torch.int32)


def reset_done(s: TorchState, teams: torch.Tensor, seeds: torch.Tensor, t: TorchTables) -> TorchState:
    """Replace every finished battle with a new one from teams[i] / seeds[i]."""
    done = s.phase == DONE
    fresh = new_batch(teams, seeds, t)
    pick = lambda new, old: torch.where(done.view(-1, *([1] * (old.dim() - 1))), new, old)  # noqa: E731
    return TorchState(**{f.name: pick(getattr(fresh, f.name), getattr(s, f.name)) for f in fields(TorchState)})
