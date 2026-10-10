"""Phase 4: the PyTorch engine makes exactly the reference engine's transitions, on CPU and CUDA."""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_batched import choose, random_battle  # noqa: E402

from pokearena import batched, engine, gpu  # noqa: E402
from pokearena.engine import DONE, FORCED_SWITCH, NORMAL  # noqa: E402
from pokearena.rng import hash32, rand32  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
PRE = batched.precompute()


def to_reference(s: gpu.TorchState) -> list[engine.Battle]:
    return batched.to_battles(gpu.to_batch(s))


def from_reference(battles, device) -> gpu.TorchState:
    return gpu.from_batch(batched.from_battles(battles), device)


@pytest.mark.parametrize("device", DEVICES)
def test_rng_matches_scalar(device):
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 2**32, 10_000, dtype=np.int64)
    got = gpu.hash32(torch.as_tensor(xs, device=device)).cpu().numpy()
    assert (got == np.array([hash32(int(x)) for x in xs])).all()
    seeds, turns = xs[:1000], rng.integers(1, 101, 1000).astype(np.int32)
    got = gpu.rand32(torch.as_tensor(seeds, device=device), torch.as_tensor(turns, device=device), 0).cpu().numpy()
    assert (got == np.array([rand32(int(s), int(t), 0) for s, t in zip(seeds, turns)])).all()


@pytest.mark.parametrize("device", DEVICES)
def test_round_trip_and_masks(device):
    rng = random.Random(0)
    battles = [random_battle(rng) for _ in range(300)]
    s = from_reference(battles, device)
    assert to_reference(s) == battles
    t = gpu.tables_on(device, PRE)
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(30):
        masks = gpu.action_masks(s).cpu().numpy()
        for i, b in enumerate(to_reference(s)):
            for side in (0, 1):
                assert (masks[i, side] == engine.action_mask(b, side)).all()
        s = gpu.step(s, gpu.random_actions(s, gen), t)


def run_differential(device: str, seed: int, n: int, step_fn=gpu.step_tensors) -> dict:
    rng = random.Random(seed)
    battles = [random_battle(rng) for _ in range(n)]
    switchy = [rng.random() < 0.2 for _ in battles]
    t = gpu.tables_on(device, PRE)
    s = from_reference(battles, device)
    seen = {NORMAL: 0, FORCED_SWITCH: 0}
    while any(b.phase != DONE for b in battles):
        actions = np.array([[choose(engine.action_mask(b, side), rng, switchy[i]) for side in (0, 1)]
                            for i, b in enumerate(battles)], dtype=np.int32)
        for b in battles:
            if b.phase != DONE:
                seen[b.phase] += 1
        battles = [engine.step(b, tuple(int(a) for a in actions[i]))[0] if b.phase != DONE else b
                   for i, b in enumerate(battles)]
        s = gpu.step(s, torch.as_tensor(actions, device=device), t, fn=step_fn)
        assert to_reference(s) == battles
    return {"seen": seen, "outcomes": {b.outcome for b in battles}}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk", range(2))
def test_steps_equal_reference_steps(device, chunk):
    result = run_differential(device, 200 + chunk, 1000)
    assert result["seen"][NORMAL] > 6000 and result["seen"][FORCED_SWITCH] > 600
    assert result["outcomes"] == {0, 1, engine.DRAW}


@pytest.mark.skipif("cuda" not in DEVICES, reason="needs CUDA")
def test_compiled_step_equals_reference_steps():
    result = run_differential("cuda", 300, 500, step_fn=gpu.compiled_step())
    assert result["outcomes"] == {0, 1, engine.DRAW}


@pytest.mark.parametrize("device", DEVICES)
def test_matches_batched_numpy_at_scale(device):
    """65k random battles, stepped in lockstep with the NumPy engine to the end."""
    n = 65536
    rng = np.random.default_rng(1)
    teams = np.stack([np.stack([rng.permutation(30)[:3] for _ in range(2)]) for _ in range(n)]).astype(np.int32)
    seeds = rng.integers(0, 2**32, n)
    s_np = batched.new_batch(teams, seeds, PRE)
    t = gpu.tables_on(device, PRE)
    s_t = gpu.new_batch(torch.as_tensor(teams, device=device), torch.as_tensor(seeds, device=device), t)
    steps = 0
    while not (s_np.phase == DONE).all():
        actions = batched.random_actions(s_np, rng)
        s_np = batched.step(s_np, actions, PRE)
        s_t = gpu.step(s_t, torch.as_tensor(actions, dtype=torch.int32, device=device), t, check=False)
        steps += 1
    b = gpu.to_batch(s_t)
    for field in ("hp", "active", "phase", "turn", "outcome"):
        assert (getattr(b, field) == getattr(s_np, field)).all(), field
    assert steps > 20


@pytest.mark.parametrize("device", DEVICES)
def test_illegal_actions_raise(device):
    s = from_reference([engine.new_battle([[0, 1, 2], [3, 4, 5]], 0)], device)
    t = gpu.tables_on(device, PRE)
    for bad in ([4, 0], [7, 0], [-1, 0]):
        with pytest.raises(ValueError):
            gpu.step(s, torch.tensor([bad], dtype=torch.int32, device=device), t)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_done_and_random_play(device):
    n = 512
    rng = np.random.default_rng(0)
    teams = torch.as_tensor(np.stack([np.stack([rng.permutation(30)[:3] for _ in range(2)]) for _ in range(n)]),
                            device=device)
    t = gpu.tables_on(device, PRE)
    s = gpu.new_batch(teams, torch.arange(n, device=device), t)
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(16):
        s = gpu.step(s, gpu.random_actions(s, gen), t, check=True)
    done = (s.phase == DONE)
    assert 0 < int(done.sum()) < n
    r = gpu.reset_done(s, teams.flip(0), torch.arange(n, device=device) + n, t)
    assert bool((r.phase[done] == NORMAL).all()) and bool((r.turn[done] == 0).all())
    assert torch.equal(r.hp[~done], s.hp[~done]) and torch.equal(r.species[done], teams.flip(0)[done])
