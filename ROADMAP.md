# PokéArena Roadmap

Rules of the environment are in `RULES.md`. The scope is phases 0–9. Phases 10–12 are deferred and not started until 0–9 are done and written up.

| Phase | Build | Success criterion |
|---|---|---|
| **0. Data** | Export ~30 species, type chart, ~100 simple moves from Showdown into compact JSON/arrays | Static tables validated |
| **1. CPU 1v1** | Python reference engine: damage, types, move ordering, fainting | Thousands of randomized unit tests pass |
| **2. CPU 3v3** | Switching, forced switching, action masks, turn limit, full battle loop, replay JSON, terminal renderer | Greedy/random agents can complete battles; any replay renders turn by turn |
| **3. Batched CPU** | Structure-of-arrays representation; remove object-heavy hot path | Establish CPU steps/sec baseline |
| **4. PyTorch CUDA** | 1k → 16k → 65k+ simultaneous environments | Exact agreement with reference simulator |
| **5. Systems benchmark** | RTX 4060 scaling curves, profiling, state-size analysis | Demonstrate when GPU overtakes CPU |
| **6. Triton** | Fuse turn-resolution/damage kernels and reduce intermediate tensors | Meaningful speedup over PyTorch or explain why not |
| **7. Classical agents** | Random, greedy damage, heuristic switcher, simultaneous-move search | Establish meaningful agent baselines |
| **8. PPO self-play** | GPU-resident observations → policy → actions → simulator → rollout buffer | Agent decisively beats random/greedy |
| **9. Compute study** | Compare search, PPO, PPO+search under fixed wall-clock/GPU budgets | Main project result: strength/GPU-hour |

## Notes per phase

- **1–4. Exact agreement** depends on the integer damage formula and the counter-based RNG in `RULES.md`. Build a differential test early: random states and actions run through both engines, every field compared.
- **2. Replays and rendering.** A replay is `(teams, seed, actions)` plus a per-turn event log (move order, damage, effectiveness, faints, switches), so viewers never re-run the engine. It is a few KB of JSON per battle. The terminal renderer (`rich`: HP bars, moves, damage, effectiveness) reads replays and is the debugging tool from here on. Every later visual reads the same format.
- **4. Before Triton**, try `torch.compile` and CUDA graphs on the step function. At 65k environments the step does little arithmetic per battle, so eager PyTorch is bound by kernel launches and memory traffic. Phase 6 compares against the best non-Triton version, not eager. Finished battles reset in place, with masking rather than per-environment branching.
- **7. Search must handle simultaneous moves.** Alternating-move minimax or MCTS assumes one player sees the other's choice. Use decoupled UCT (each player runs its own bandit at a node) or solve the small action-by-action matrix game at each node (LP or regret matching).
- **9. Strength metric.** Win rate against random or greedy agents saturates quickly. Use a round-robin Elo among all agents, plus exploitability where it can be computed: exact solutions for 1v1, and for specific 3v3 team pairs if the reachable state set turns out small enough (measure this), otherwise an approximate best response trained against the frozen policy.

## Visuals

- **Phase 2:** replay JSON and terminal renderer (above).
- **Phases 5 and 9:** benchmark and compute-study plots as matplotlib PNGs in the write-up.
- **Phase 12:** a static HTML/JS replay viewer on `adixg.github.io` that loads replay JSON: Showdown sprites, HP bars, a step-through turn log, play/pause, and the agent's action probabilities each turn. Ships with curated battles (PPO vs greedy, PPO vs search). Sprites are loaded from Showdown's sprite server (`play.pokemonshowdown.com/sprites/`), not copied into the repo.

## Minimum shippable version

Phases 0–5 + 7 + 8: "a Pokémon battle simulator at X M steps/s on an RTX 4060, Y× faster than CPU, with a PPO agent trained in Z minutes." Write it up and put it on the resume before continuing to 6 and 9.

## Deferred (10–12)

| Phase | Build | Success criterion |
|---|---|---|
| **10. Mechanics expansion** | Add mechanics one at a time: accuracy → status → Tera, only when justified | Quantify effect on strategy and throughput |
| **11. Multi-GPU/PACE** | Run same simulator on RTX 4060 vs H100/H200; larger batches and kernel retuning | Hardware scaling analysis |
| **12. Portfolio polish** | Interactive battle viewer, benchmark plots, replay viewer, technical write-up | Someone can understand project in 60 seconds |
