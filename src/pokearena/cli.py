"""Command line: `pokearena play` runs a battle, `pokearena show` renders a replay.

    uv run pokearena play greedy random --seed 7 --save battle.json
    uv run pokearena play greedy random --teams garchomp,gengar,scizor/dragapult,lapras,tyranitar
    uv run pokearena show battle.json --step
"""

import argparse
import random
import sys

from pokearena import replay as replays
from pokearena.agents import AGENTS, make_agent
from pokearena.data import load_tables
from pokearena.render import render


def parse_teams(text: str | None, seed: int) -> list[list[int]]:
    t = load_tables()
    if text is None:                         # two random teams of three, decided by the seed
        rng = random.Random(seed)
        return [rng.sample(range(len(t.species_ids)), 3) for _ in range(2)]
    index = {s: i for i, s in enumerate(t.species_ids)}
    teams = []
    for team in text.split("/"):
        unknown = [s for s in team.split(",") if s not in index]
        if unknown:
            raise SystemExit(f"unknown species {unknown}; choose from {', '.join(t.species_ids)}")
        teams.append([index[s] for s in team.split(",")])
    if len(teams) != 2:
        raise SystemExit("--teams needs two teams separated by '/'")
    return teams


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokearena")
    commands = parser.add_subparsers(dest="command", required=True)

    play = commands.add_parser("play", help="run a battle between two agents and render it")
    play.add_argument("agents", nargs=2, choices=sorted(AGENTS), metavar="AGENT",
                      help=f"one of {', '.join(sorted(AGENTS))}")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--teams", help="species ids, e.g. garchomp,gengar,scizor/dragapult,lapras,tyranitar")
    play.add_argument("--save", help="write the replay JSON here")
    play.add_argument("--quiet", action="store_true", help="do not render")
    play.add_argument("--step", action="store_true", help="wait for Enter after each turn")

    show = commands.add_parser("show", help="render a replay file")
    show.add_argument("path")
    show.add_argument("--step", action="store_true", help="wait for Enter after each turn")
    show.add_argument("--verify", action="store_true", help="re-run the replay and check every event")

    args = parser.parse_args(argv)
    if args.command == "play":
        agents = (make_agent(args.agents[0], args.seed), make_agent(args.agents[1], args.seed + 1))
        replay = replays.play(parse_teams(args.teams, args.seed), args.seed, agents)
        if args.save:
            replays.save(replay, args.save)
        if not args.quiet:
            render(replay, step=args.step)
    else:
        replay = replays.load(args.path)
        if args.verify:
            replays.verify(replay)
            print("verified: every event reproduced")
        render(replay, step=args.step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
