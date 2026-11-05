"""Check Sokoban terminal-reward semantics.

Run from the repo root inside the sokoban env:

    python scripts/check_sokoban_reward.py [max_episodes]

Part 1 (raw SokobanEnv): on the step where info['won'] becomes True, is the
upstream `done` flag also True?  `SokobanWorker.step` only gives reward 10 when
`done and info['won']`, so a winning step with done=False would never be paid.

Part 2 (SokobanWorker): replay the same kind of winning episode through the
worker and check it returns reward 10.0 on the winning step and 0.0 elsewhere.
"""
import sys

import numpy as np

from agent_system.environments.env_package.sokoban.envs import SokobanWorker
from agent_system.environments.env_package.sokoban.sokoban import SokobanEnv

ENV_KWARGS = dict(dim_room=(6, 6), num_boxes=1, max_steps=15, search_depth=30)
MAX_STEPS = ENV_KWARGS["max_steps"]


def find_winning_episode(max_episodes, rng):
    """Random-walk until an episode wins; return (seed, action list up to the win)."""
    env = SokobanEnv("rgb_array", **ENV_KWARGS)
    for seed in range(max_episodes):
        env.reset(seed=seed)
        actions = []
        for _ in range(MAX_STEPS):
            action = int(rng.integers(1, 5))
            actions.append(action)
            _, reward, done, info = env.step(action)
            if info["won"]:
                return seed, actions, reward, done
            if done:
                break
    return None


def main():
    max_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = np.random.default_rng(0)

    found = find_winning_episode(max_episodes, rng)
    if found is None:
        print(f"no winning episode in {max_episodes} random episodes; "
              f"rerun with a larger count, e.g. {max_episodes * 5}")
        return
    seed, actions, raw_reward, raw_done = found

    print("== Part 1: raw SokobanEnv ==")
    print(f"seed={seed} winning_step={len(actions) - 1} raw_reward={raw_reward} done={raw_done} won=True")
    if raw_done:
        print("OK: done is True on the winning step.")
    else:
        print("PROBLEM: done is False on the winning step -> `done and won` never pays 10.")

    print("== Part 2: SokobanWorker.step ==")
    worker = SokobanWorker("rgb_array", dict(ENV_KWARGS))
    worker.reset(seed)
    for t, action in enumerate(actions):
        _, reward, done, info = worker.step(action)
        print(f"step={t} action={action} reward={reward} done={done} won={info['won']}")
    expected = 10.0 if raw_done else None
    print(f"final worker reward={reward} (expected {expected})")


if __name__ == "__main__":
    main()
