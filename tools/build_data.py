"""Build data/pokearena_v1.json from Pokémon Showdown's published data files.

Downloads pokedex.json, moves.json, learnsets.json and typechart.js from
play.pokemonshowdown.com into data/raw/ (gitignored) unless already present,
then applies the curated species and move lists below and the v1 move rules
from RULES.md section 3. Every check that fails is reported, and nothing is
written unless all pass.

The raw files are parsed as data only: JSON with the json module, and
typechart.js with a regex. Nothing downloaded is ever executed.

Usage: uv run python tools/build_data.py [--refresh]
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "pokearena_v1.json"
SOURCE = "https://play.pokemonshowdown.com/data/"
FILES = ["pokedex.json", "moves.json", "learnsets.json", "typechart.js"]

TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground",
    "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]

# Species -> fixed four-move loadout (RULES.md section 8). Showdown IDs.
ROSTER = {
    "charizard": ["flamethrower", "airslash", "dragonpulse", "earthquake"],
    "blastoise": ["surf", "icebeam", "darkpulse", "aurasphere"],
    "venusaur": ["energyball", "sludgebomb", "earthpower", "earthquake"],
    "jolteon": ["thunderbolt", "shadowball", "hypervoice", "quickattack"],
    "gengar": ["shadowball", "sludgebomb", "thunderbolt", "dazzlinggleam"],
    "alakazam": ["psychic", "shadowball", "dazzlinggleam", "energyball"],
    "machamp": ["brickbreak", "poisonjab", "bulletpunch", "firepunch"],
    "dragonite": ["dragonclaw", "extremespeed", "earthquake", "firepunch"],
    "gyarados": ["waterfall", "crunch", "earthquake", "bodyslam"],
    "snorlax": ["bodyslam", "earthquake", "crunch", "firepunch"],
    "lapras": ["surf", "icebeam", "thunderbolt", "psychic"],
    "arcanine": ["flamethrower", "extremespeed", "crunch", "psychicfangs"],
    "scizor": ["bulletpunch", "xscissor", "aerialace", "brickbreak"],
    "tyranitar": ["crunch", "rockslide", "earthquake", "icepunch"],
    "garchomp": ["earthquake", "dragonclaw", "poisonjab", "ironhead"],
    "lucario": ["aurasphere", "flashcannon", "darkpulse", "vacuumwave"],
    "gardevoir": ["psychic", "moonblast", "thunderbolt", "shadowball"],
    "metagross": ["ironhead", "zenheadbutt", "bulletpunch", "earthquake"],
    "weavile": ["iceshard", "iciclecrash", "bite", "brickbreak"],
    "magnezone": ["thunderbolt", "flashcannon", "triattack", "discharge"],
    "togekiss": ["dazzlinggleam", "airslash", "aurasphere", "flamethrower"],
    "volcarona": ["flamethrower", "bugbuzz", "psychic", "airslash"],
    "excadrill": ["earthquake", "ironhead", "rockslide", "brickbreak"],
    "ferrothorn": ["seedbomb", "ironhead", "poisonjab", "bulldoze"],
    "sylveon": ["moonblast", "hypervoice", "shadowball", "psychic"],
    "hydreigon": ["darkpulse", "dragonpulse", "flamethrower", "earthpower"],
    "corviknight": ["drillpeck", "ironhead", "bodyslam", "rocksmash"],
    "dragapult": ["dragonpulse", "shadowball", "thunderbolt", "quickattack"],
    "mamoswine": ["earthquake", "iceshard", "iciclecrash", "rockslide"],
    "starmie": ["surf", "psychic", "icebeam", "thunderbolt"],
}

# RULES.md section 3. Keys in Showdown's move data that mean the move does
# something beyond fixed-power damage.
EXCLUDED_KEYS = {
    "basePowerCallback", "boosts", "breaksProtect", "callsMove", "chloroblastRecoil",
    "condition", "critRatio", "damage", "drain", "forceSwitch", "hasCrashDamage", "heal",
    "ignoreDefensive", "ignoreImmunity", "isMax", "isNonstandard", "isZ", "mindBlownRecoil",
    "multiaccuracy", "multihit", "ohko", "overrideDefensiveStat", "overrideOffensivePokemon",
    "overrideOffensiveStat", "pseudoWeather", "recoil", "secondaries", "self", "selfBoost",
    "selfSwitch", "selfdestruct", "sideCondition", "slotCondition", "smartTarget",
    "stallingMove", "status", "stealsBoosts", "struggleRecoil", "terrain", "volatileStatus",
    "weather", "willCrit",
}
EXCLUDED_FLAGS = {"charge", "recharge", "futuremove"}
# Power or success that depends on battle state, implemented in Showdown's
# server code and therefore invisible in the client's moves.json.
CONDITIONAL = {
    "acrobatics", "belch", "brine", "bugbite", "collisioncourse", "covet", "electrodrift",
    "expandingforce", "facade", "fakeout", "falseswipe", "fellstinger", "ficklebeam",
    "firstimpression", "flyingpress", "freezedry", "grassyglide", "hex", "jawlock",
    "knockoff", "lashout", "lastresort", "pluck", "poltergeist", "ragingbull", "retaliate",
    "snore", "steelroller", "suckerpunch", "thief", "upperhand", "venoshock", "weatherball",
}
MIN_ACCURACY = 90

# Mechanics that roadmap step 5b brings back, and what each one unlocks.
# A data version lists the mechanics it enables; v1 enables none.
MECHANICS = {
    "accuracy": "moves below MIN_ACCURACY",
    "recoil": "the recoil key",
    "drain": "the drain key",
    "multihit": "the multihit key",
    "stat_override": "the overrideOffensiveStat and overrideDefensiveStat keys (Body Press, Psyshock)",
    "suckerpunch": "Sucker Punch, which fails unless the target chose a damaging move",
}
MECHANIC_KEYS = {
    "recoil": {"recoil"},
    "drain": {"drain"},
    "multihit": {"multihit", "smartTarget"},     # smartTarget only matters in doubles
    "stat_override": {"overrideOffensiveStat", "overrideDefensiveStat"},
}
ENABLED = {"v1": frozenset()}
VERSION = "v1"
SINGLE_TARGETS = {"normal", "any", "allAdjacent", "allAdjacentFoes"}

# damageTaken codes in typechart.js -> effectiveness in quarters (RULES.md 4)
EFF4 = {0: 4, 1: 8, 2: 2, 3: 0}


def fetch(refresh: bool) -> dict[str, str]:
    RAW.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in FILES:
        path = RAW / name
        if refresh or not path.exists():
            with urllib.request.urlopen(SOURCE + name, timeout=60) as response:
                path.write_bytes(response.read())
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def parse_typechart(text: str) -> dict[str, dict[str, int]]:
    """{defending type: {attacking type: damageTaken code}} from typechart.js."""
    chart = {}
    for defender, body in re.findall(r"(\w+):\{damageTaken:\{([^}]*)\}", text):
        codes = dict(re.findall(r"(\w+):(\d)", body))
        chart[defender.capitalize()] = {k: int(v) for k, v in codes.items() if k[0].isupper()}
    return chart


def stat(base: int, hp: bool) -> int:
    """Level 50, 31 IVs, 0 EVs, neutral nature (RULES.md section 2)."""
    core = (2 * base + 31) * 50 // 100
    return core + 60 if hp else core + 5


def move_problems(move_id: str, move: dict, mechanics: frozenset[str] = frozenset()) -> list[str]:
    """Reasons the move is not allowed with the given mechanics enabled."""
    excluded = EXCLUDED_KEYS - {k for m in mechanics for k in MECHANIC_KEYS.get(m, ())}
    problems = []
    if move["category"] == "Status" or not move["basePower"]:
        problems.append("not a damaging move")
    if keys := sorted(excluded & move.keys()):
        problems.append(f"excluded mechanics {keys}")
    if flags := sorted(EXCLUDED_FLAGS & move["flags"].keys()):
        problems.append(f"excluded flags {flags}")
    if move_id in CONDITIONAL and not (move_id == "suckerpunch" and "suckerpunch" in mechanics):
        problems.append("conditional power or success")
    if (move["accuracy"] is not True and move["accuracy"] < MIN_ACCURACY
            and "accuracy" not in mechanics):
        problems.append(f"accuracy {move['accuracy']} < {MIN_ACCURACY}")
    if move["target"] not in SINGLE_TARGETS:
        problems.append(f"target {move['target']}")
    if move["type"] not in TYPES:
        problems.append(f"type {move['type']}")
    return problems


def learnable(species_id: str, move_id: str, pokedex: dict, learnsets: dict) -> bool:
    """Learnable by the species or any pre-evolution, in any generation."""
    current = species_id
    while current:
        if move_id in learnsets.get(current, {}).get("learnset", {}):
            return True
        prevo = pokedex[current].get("prevo")
        current = re.sub(r"[^a-z0-9]", "", prevo.lower()) if prevo else None
    return False


def move_record(move_id: str, move: dict) -> dict:
    """The move as stored, including fields for mechanics v1 does not simulate.

    `attack_stat` and `defense_stat` are explicit so Body Press (Defense as its
    attacking stat) and Psyshock (targets Defense) need no special case.
    """
    physical = move["category"] == "Physical"
    multihit = move.get("multihit")
    return {
        "id": move_id,
        "name": move["name"],
        "type": move["type"],
        "category": move["category"],
        "power": move["basePower"],
        "priority": move["priority"],
        "attack_stat": move.get("overrideOffensiveStat", "atk" if physical else "spa"),
        "defense_stat": move.get("overrideDefensiveStat", "def" if physical else "spd"),
        "accuracy": 100 if move["accuracy"] is True else move["accuracy"],
        "never_misses": move["accuracy"] is True,
        "recoil": move.get("recoil"),            # [numerator, denominator] of damage dealt
        "drain": move.get("drain"),              # [numerator, denominator] of damage dealt
        "multihit": [multihit, multihit] if isinstance(multihit, int) else multihit,  # [min, max]
        "effect": move_id if move_id == "suckerpunch" else None,
        "ignored_secondary": "secondary" in move,
    }


def build(refresh: bool) -> int:
    mechanics = ENABLED[VERSION]
    assert mechanics <= MECHANICS.keys(), mechanics - MECHANICS.keys()
    digests = fetch(refresh)
    pokedex = json.loads((RAW / "pokedex.json").read_text())
    moves = json.loads((RAW / "moves.json").read_text())
    learnsets = json.loads((RAW / "learnsets.json").read_text())
    chart = parse_typechart((RAW / "typechart.js").read_text())
    errors = []

    # Type chart, eff4[attacking][defending]
    type_chart = []
    for attacker in TYPES:
        row = []
        for defender in TYPES:
            code = chart.get(defender, {}).get(attacker)
            if code not in EFF4:
                errors.append(f"type chart: no entry for {attacker} -> {defender}")
                code = 0
            row.append(EFF4[code])
        type_chart.append(row)

    # Moves used by the roster, in first-use order
    move_ids = list(dict.fromkeys(m for loadout in ROSTER.values() for m in loadout))
    move_rows = []
    for move_id in move_ids:
        move = moves.get(move_id)
        if move is None:
            errors.append(f"move {move_id}: not in moves.json")
            continue
        errors += [f"move {move_id}: {p}" for p in move_problems(move_id, move, mechanics)]
        move_rows.append(move_record(move_id, move))

    species_rows = []
    for species_id, loadout in ROSTER.items():
        entry = pokedex.get(species_id)
        if entry is None:
            errors.append(f"species {species_id}: not in pokedex.json")
            continue
        if len(set(loadout)) != 4:
            errors.append(f"species {species_id}: loadout must be four distinct moves")
        for move_id in loadout:
            if not learnable(species_id, move_id, pokedex, learnsets):
                errors.append(f"species {species_id}: cannot learn {move_id}")
        base = entry["baseStats"]
        species_rows.append({
            "id": species_id,
            "name": entry["name"],
            "num": entry["num"],
            "types": entry["types"],
            "base_stats": base,
            "stats": {k: stat(v, k == "hp") for k, v in base.items()},
            "moves": loadout,
        })

    if errors:
        print(f"{len(errors)} problem(s); nothing written:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    OUT.write_text(json.dumps({
        "version": 1,
        "source": {
            "url": SOURCE,
            "files": digests,
            "built": date.today().isoformat(),
            "license": "Pokémon Showdown data, MIT License",
        },
        "rules": {
            "level": 50, "iv": 31, "ev": 0, "min_accuracy": MIN_ACCURACY,
            "mechanics": sorted(mechanics),
        },
        "types": TYPES,
        "type_chart_eff4": type_chart,
        "moves": move_rows,
        "species": species_rows,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(species_rows)} species, {len(move_rows)} moves")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download the raw files")
    sys.exit(build(parser.parse_args().refresh))
