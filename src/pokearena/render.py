"""Terminal rendering of replays with rich.

Reads only the replay and the static tables (for names, max HP and speed);
the battle state shown is rebuilt from the events, never by re-running the
engine.
"""

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from pokearena.data import HP, SPE, Tables, load_raw, load_tables
from pokearena.engine import DRAW, N_MOVES, SWITCH_BASE

BAR_WIDTH = 20
EFFECT_TEXT = {0: "no effect", 1: "¼× not very effective", 2: "not very effective",
               8: "super effective", 16: "4× super effective"}


class _View:
    """The battle as the events describe it."""

    def __init__(self, replay: dict, t: Tables):
        self.t = t
        self.index = {s: i for i, s in enumerate(t.species_ids)}
        self.names = {m["id"]: m["name"] for m in load_raw()["moves"]} | {
            s["id"]: s["name"] for s in load_raw()["species"]}
        self.teams = [[self.index[s] for s in team] for team in replay["teams"]]
        self.max_hp = [[int(t.species_stats[s, HP]) for s in team] for team in self.teams]
        self.hp = [list(team) for team in self.max_hp]
        self.active = [0, 0]
        self.players = replay["players"]

    def species(self, side: int, slot: int | None = None) -> int:
        return self.teams[side][self.active[side] if slot is None else slot]

    def name(self, side: int, slot: int | None = None) -> str:
        return self.names[self.t.species_ids[self.species(side, slot)]]

    def action_label(self, side: int, action: int) -> str:
        if action < N_MOVES:
            return self.names[self.t.move_ids[self.t.species_moves[self.species(side), action]]]
        return "→ " + self.name(side, action - SWITCH_BASE)

    def hp_bar(self, side: int) -> Text:
        slot = self.active[side]
        hp, max_hp = self.hp[side][slot], self.max_hp[side][slot]
        filled = round(BAR_WIDTH * hp / max_hp)
        color = "green" if hp * 2 > max_hp else "yellow" if hp * 5 > max_hp else "red"
        line = Text(f" P{side}  {self.name(side):<12}")
        line.append("█" * filled, style=color).append("░" * (BAR_WIDTH - filled), style="dim")
        line.append(f" {hp:>3}/{max_hp:<3}   ")
        for slot_i in range(len(self.teams[side])):
            if slot_i != slot:
                alive = self.hp[side][slot_i] > 0
                line.append(f"{self.name(side, slot_i)} ", style="" if alive else "dim strike")
                line.append("● " if alive else "✕ ", style="green" if alive else "red")
        return line

    def describe(self, event: dict) -> Text | None:
        kind = event["type"]
        if kind == "order":
            side = event["first"]
            if event["reason"] == "speed":
                speeds = [int(self.t.species_stats[self.species(s), SPE]) for s in (side, 1 - side)]
                why = f"speed {speeds[0]} vs {speeds[1]}"
            else:
                why = {"priority": "priority move", "tie": "speed tie, seeded"}[event["reason"]]
            return Text(f" {self.name(side)} moves first ({why})", style="dim")
        if kind == "switch":
            side = event["side"]
            before = self.name(side, event["from"])
            self.active[side] = event["to"]
            if event["forced"]:
                return Text(f" P{side} sent out {self.name(side)}")
            return Text(f" P{side} withdrew {before} and sent out {self.name(side)}", style="cyan")
        if kind == "move":
            side, foe = event["side"], 1 - event["side"]
            self.hp[foe][self.active[foe]] = event["hp_after"]
            line = Text(f" {self.name(side)} used {self.names[event['move']]} → {self.name(foe)}  ")
            line.append(str(event["damage"]), style="bold")
            if event["eff4"] != 4:
                style = "bold green" if event["eff4"] > 4 else "yellow" if event["eff4"] else "red"
                line.append(f"  {EFFECT_TEXT[event['eff4']]}", style=style)
            return line
        if kind == "faint":
            return Text(f" {self.name(event['side'], event['slot'])} fainted!", style="bold red")
        if kind == "end":
            if event["outcome"] == DRAW:
                return Text(" Draw: turn limit reached", style="bold yellow")
            winner = event["outcome"]
            return Text(f" P{winner} ({self.players[winner]}) wins!", style="bold green")
        return None                                  # forced_switch: shown by the next record

    def policy_line(self, side: int, policy: list[float]) -> Text:
        top = sorted(((p, a) for a, p in enumerate(policy) if p > 0), reverse=True)[:3]
        parts = " · ".join(f"{self.action_label(side, a)} {p:.0%}" for p, a in top)
        return Text(f" P{side} policy: {parts}", style="magenta")


def render(replay: dict, console: Console | None = None, step: bool = False, tables: Tables | None = None) -> None:
    """Print a replay turn by turn. With `step`, wait for Enter after each turn."""
    console = console or Console()
    view = _View(replay, tables or load_tables())
    console.print(Text(f" P0 {view.players[0]} vs P1 {view.players[1]}   seed {replay['seed']}", style="bold"))
    for record in replay["turns"]:
        forced = record["phase"] == "forced_switch"
        title = f"Turn {record['turn']}" + (" · replacement" if forced else "")
        console.print(Rule(title, style="dim" if forced else "blue"))
        if not forced:
            for side in (0, 1):
                console.print(view.hp_bar(side))
            console.print()
        for side, policy in enumerate(record.get("policy") or [None, None]):
            if policy is not None:
                console.print(view.policy_line(side, policy))
        for event in record["events"]:
            line = view.describe(event)
            if line is not None:
                console.print(line)
        if step:
            console.input("[dim]Enter for the next turn[/dim]")
    console.print(Rule(style="blue"))
    for side in (0, 1):
        console.print(view.hp_bar(side))
