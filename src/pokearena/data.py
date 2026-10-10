"""Static game tables (RULES.md sections 2-4, 8) as JSON records and integer arrays.

The arrays are the immutable lookup tables shared by every battle (RULES.md
section 9). Everything is int32 so the CPU engine, the batched engine and the
GPU engine index the same values.
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "pokearena_v1.json"

STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
HP, ATK, DEF, SPA, SPD, SPE = range(6)
PHYSICAL, SPECIAL = 0, 1
NO_TYPE = -1


@dataclass(frozen=True)
class Tables:
    types: tuple[str, ...]
    move_ids: tuple[str, ...]
    species_ids: tuple[str, ...]
    type_chart: np.ndarray       # [T, T] effectiveness in quarters, [attacking, defending]
    move_type: np.ndarray        # [M]
    move_power: np.ndarray       # [M]
    move_category: np.ndarray    # [M] PHYSICAL or SPECIAL
    move_priority: np.ndarray    # [M]
    species_types: np.ndarray    # [S, 2] second is NO_TYPE for single-typed species
    species_stats: np.ndarray    # [S, 6] in STAT_KEYS order
    species_moves: np.ndarray    # [S, 4] move indices


def load_raw(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text())


def _frozen(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    array.setflags(write=False)
    return array


@cache
def load_tables(path: Path = DATA_PATH) -> Tables:
    raw = load_raw(path)
    types = tuple(raw["types"])
    type_index = {t: i for i, t in enumerate(types)}
    moves = raw["moves"]
    move_index = {m["id"]: i for i, m in enumerate(moves)}
    species = raw["species"]
    return Tables(
        types=types,
        move_ids=tuple(m["id"] for m in moves),
        species_ids=tuple(s["id"] for s in species),
        type_chart=_frozen(raw["type_chart_eff4"]),
        move_type=_frozen([type_index[m["type"]] for m in moves]),
        move_power=_frozen([m["power"] for m in moves]),
        move_category=_frozen([PHYSICAL if m["category"] == "Physical" else SPECIAL for m in moves]),
        move_priority=_frozen([m["priority"] for m in moves]),
        species_types=_frozen([
            [type_index[s["types"][0]], type_index[s["types"][1]] if len(s["types"]) > 1 else NO_TYPE]
            for s in species
        ]),
        species_stats=_frozen([[s["stats"][k] for k in STAT_KEYS] for s in species]),
        species_moves=_frozen([[move_index[m] for m in s["moves"]] for s in species]),
    )
