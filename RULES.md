# PokéArena Rules

PokéArena is a deliberately simplified Pokémon-style battle environment designed for high-throughput simulation, search, and reinforcement-learning research.

It is **not** intended to reproduce the complete Pokémon Gen 9 battle engine. The environment preserves the strategic core of Pokémon—type matchups, move selection, switching, speed, and simultaneous opponent decisions—while removing mechanics that would make GPU simulation irregular or unnecessarily complex.

## 1. Battle Format

- Battles are **3v3 singles**.
- Each player has three Pokémon and one active Pokémon at a time.
- Each Pokémon has exactly four moves.
- Pokémon are level 50.
- A player wins when all three opposing Pokémon have fainted.
- Initial training uses full information: both teams, moves, HP, and stats are visible to both agents.

## 2. Pokémon Stats

Each Pokémon has:

- HP
- Attack
- Defense
- Special Attack
- Special Defense
- Speed
- one or two types

Battle stats are generated from canonical base stats using:

- Level: 50
- IVs: 31
- EVs: 0
- Neutral nature

This gives every species one deterministic stat configuration.

## 3. Moves

Every Pokémon has a fixed four-move loadout.

Version 1 permits only direct damaging moves.

Each move contains:

- type
- base power
- Physical or Special category
- priority

Version 1 excludes:

- status moves
- stat changes
- recoil
- healing
- multi-hit moves
- conditional-power moves
- variable-priority effects
- secondary effects

Moves are treated as 100% accurate in v1.

Two clarifications define which real moves are eligible:

- **Accuracy at least 90%.** Moves below that (Hydro Pump, Fire Blast, Stone Edge) are not used, since treating them as perfectly accurate would make them strictly better than their reliable counterparts. The real accuracy is kept in the data for when accuracy is introduced.
- **Chance-based secondary effects are ignored, not excluded.** Flamethrower is a 90-power Fire move; its 10% burn chance does not exist in v1. Excluding every such move would leave Fire, Ice and Electric with almost no usable attacks. Moves whose extra effect always happens (recoil, draining, stat changes on the user, multi-hit) or whose power depends on battle state remain excluded. The data marks affected moves with `ignored_secondary`.

The move data is built by `tools/build_data.py` from Pokémon Showdown's published data files, which enforces these rules and checks every loadout is learnable by its species.

## 4. Damage

Physical moves use Attack and Defense.

Special moves use Special Attack and Special Defense.

Damage follows the simplified deterministic Pokémon damage structure:

\[
BaseDamage =
\left\lfloor
\left(
\frac{
(\frac{2L}{5}+2) \times Power \times Attack
}{
Defense
}
\right)
\frac{1}{50}
+2
\right\rfloor
\]

The final damage is:

\[
Damage =
\max\left(
1,
\left\lfloor
BaseDamage \times STAB \times TypeEffectiveness
\right\rfloor
\right)
\]

except when TypeEffectiveness = 0, in which case damage is 0.

STAB is:

\[
STAB =
\begin{cases}
1.5 & \text{if the move matches either attacker type}\\
1.0 & \text{otherwise}
\end{cases}
\]

Type effectiveness follows the Gen 9 type chart.

For dual-type defenders, effectiveness against both types is multiplied.

Examples include:

- \(2\times\)
- \(4\times\)
- \(1\times\)
- \(0.5\times\)
- \(0.25\times\)
- \(0\times\)

Version 1 does not include the usual random damage multiplier or critical hits.

### Integer Implementation

Both formulas above are computed in integer arithmetic only, so that CPU and GPU implementations agree bit for bit. With level 50, \(\frac{2L}{5}+2 = 22\):

```
base   = (22 * power * attack // defense) // 50 + 2
damage = base * stab2 * eff4 // 8
if eff4 > 0: damage = max(1, damage)
```

where `//` is floor division on non-negative integers, `stab2` is STAB in halves (2 or 3) and `eff4` is type effectiveness in quarters (0, 1, 2, 4, 8 or 16). This is exactly equal to the formulas above: `floor(floor(x) / 50) = floor(x / 50)`, and `stab2 * eff4 / 8` is an exact rational, so there is a single rounding step. All intermediates fit in a signed 32-bit integer. Floating-point arithmetic is never used in damage.

## 5. Action Space

Each player has seven fixed action IDs:

| Action | Meaning |
|---|---|
| 0 | Move slot 0 |
| 1 | Move slot 1 |
| 2 | Move slot 2 |
| 3 | Move slot 3 |
| 4 | Switch to team slot 0 |
| 5 | Switch to team slot 1 |
| 6 | Switch to team slot 2 |

Illegal actions are represented through an action mask.

Examples of illegal actions include:

- switching to the currently active Pokémon
- switching to a fainted Pokémon
- attacking while a forced switch is required

Agents must never intentionally execute masked actions.

## 6. Turn Resolution

Both players select actions simultaneously.

A normal turn resolves in this order:

1. Voluntary switches
2. Move priority
3. Speed
4. Move execution
5. Faint handling
6. Victory check

### Switching

Voluntary switches resolve before attacks.

If Player 1 switches and Player 2 attacks, Player 2's attack targets the newly switched-in Pokémon.

If both players switch, both switches occur before the turn ends.

### Attack Ordering

If both players attack:

1. Higher-priority move acts first.
2. If priorities match, the faster active Pokémon acts first.
3. If Speed also matches, the winner is chosen by the seeded tie breaker in section 10.

Given the same state, actions, and RNG seed, simulation must produce exactly the same result.

### Fainting

If a Pokémon faints before executing its selected move, its move is cancelled.

If an active Pokémon faints and at least one teammate remains, that player enters a `FORCED_SWITCH` phase.

During a forced switch, only legal switch actions are permitted.

The player who is not replacing a Pokémon makes no decision: its action mask is all false and its action is ignored.

If the fainted Pokémon was the player's last, there is no forced switch; the victory check ends the battle.

In v1 at most one active Pokémon can faint per turn: moves resolve one at a time, and a Pokémon that faints before acting cannot attack. If a later mechanic (recoil, for example) allows both to faint in the same turn, both players select their replacements simultaneously.

The next normal turn begins after all required replacements have been selected.

### Turn Limit

A battle lasts at most **100 normal turns**. Forced-switch phases do not count as turns.

If no player has won when turn 100 has resolved, the battle ends in a **draw**.

Without a limit, a battle can last forever: both players can switch every turn, or two active Pokémon can be unable to damage each other (Normal-type moves against a Ghost type). Every episode must terminate for rollouts, search, and batched GPU simulation.

## 7. Mechanics Excluded From v1

The following mechanics are intentionally excluded:

- abilities
- held items
- Terastallization
- weather
- terrain
- status conditions
- entry hazards
- stat stages
- critical hits
- random damage rolls
- move accuracy
- move PP
- secondary move effects
- Dynamax
- Z-Moves

These may be introduced individually in later versions only if they produce useful strategic behavior or a worthwhile systems problem.

Planned for v1b (roadmap step 5b), because they add no state that persists across turns: move accuracy, recoil, drain, moves that use a different attacking or defending stat (Body Press, Psyshock), Sucker Punch, and optionally multi-hit moves. Everything else in this list and in section 3's exclusions waits for a later version.

Completeness with Pokémon Showdown is explicitly **not** a project objective.

## 8. Teams

The initial environment uses a curated pool of approximately 30 Pokémon.

Each Pokémon receives one predefined four-move loadout.

Initial experiments use predefined teams for reproducibility.

Later experiments may uniformly or procedurally generate 3-Pokémon teams from the curated pool.

Duplicate Pokémon within one team are not permitted.

## 9. Environment State

The dynamic battle state consists only of information that can change during a battle:

- Pokémon HP
- active Pokémon index
- fainted/alive state
- current phase
- turn number
- battle seed

The RNG is counter-based (section 10), so the battle seed never changes and there is no separate mutable RNG state.

Species data, types, stats, and move definitions are immutable lookup tables and are not duplicated in every environment state.

## 10. Determinism

The simulator must support deterministic replay.

A battle is fully determined by:

\[
(initial\ state,\ action\ sequence,\ RNG\ seed)
\]

CPU and GPU implementations must produce identical state transitions for identical inputs.

### Random Numbers

Library random number generators (NumPy, PyTorch, CUDA) produce different streams and are never used inside the simulator. Randomness comes from a counter-based hash computed in plain 32-bit integer arithmetic:

```
def hash32(x):                      # all operations mod 2**32
    x ^= x >> 16
    x  = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x  = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x

def rand32(seed, turn, draw):
    return hash32(seed ^ hash32((turn << 8) | draw))
```

`seed` is the 32-bit battle seed, `turn` the normal-turn number, and `draw` the index of the draw within that turn (0–255).

The v1 speed tie on turn `t` uses `rand32(seed, t, 0) & 1`: 0 means player 0 acts first, 1 means player 1 does. Later mechanics (accuracy, damage rolls) take the next draw indices.

PyTorch has limited unsigned 32-bit support, so the GPU implementation computes in `int64` and masks with `0xFFFFFFFF` after every multiply.

## 11. Design Principle

When deciding whether to add a mechanic, use this test:

> Does this mechanic create meaningful additional strategy worth its simulator complexity?

If not, it does not belong in the environment.

The project's objective is a compact benchmark for high-throughput multi-agent search and learning, not Pokémon mechanical completeness.
