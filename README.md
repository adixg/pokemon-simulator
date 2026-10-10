# PokéArena

A deliberately simplified Pokémon battle environment for high-throughput simulation, search and reinforcement learning. The rules are in `RULES.md` and the plan in `ROADMAP.md`.

## Usage

```bash
uv sync
uv run python tools/build_data.py   # rebuild data/pokearena_v1.json from Showdown's data
uv run pytest

uv run pokearena play greedy random --seed 7 --save battle.json   # run and render a battle
uv run pokearena play greedy random --teams garchomp,gengar,scizor/dragapult,lapras,tyranitar
uv run pokearena show battle.json --step --verify                 # replay turn by turn, re-checking every event
uv run python tools/bench_cpu.py                                   # CPU steps/sec baseline -> results/
```

Agents: `random` (uniform over legal actions) and `greedy` (most damage this turn, never switches voluntarily).

`data/pokearena_v1.json` is committed, so building it is only needed after changing the roster or the move rules. The raw Showdown files are downloaded to `data/raw/` (gitignored); `--refresh` re-downloads them.

Game data comes from [Pokémon Showdown](https://github.com/smogon/pokemon-showdown), MIT License.
