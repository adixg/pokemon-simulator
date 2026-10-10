"""Counter-based random numbers (RULES.md section 10).

Pure functions of (seed, turn, draw) in 32-bit integer arithmetic, so every
engine computes the same values without carrying generator state.
"""

MASK32 = 0xFFFFFFFF


def hash32(x: int) -> int:
    x &= MASK32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK32
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK32
    x ^= x >> 16
    return x


def rand32(seed: int, turn: int, draw: int) -> int:
    """The `draw`-th random number of normal turn `turn` in the battle with `seed`."""
    return hash32((seed & MASK32) ^ hash32((turn << 8) | draw))


SPEED_TIE_DRAW = 0
