"""
Entry point for running agents against the DCS RL environment.

Usage
-----
    python main.py [fsp_path] [mode] [episodes]

Arguments
---------
fsp_path : path to the .fsp instance file
           (default: fsp/NonBlocking/Benchmark/AT/AT-2-2.fsp)
mode     : random | dqn   (default: dqn)
episodes : number of episodes for random mode  (default: 5)

Examples
--------
    python main.py                                          # DQN on AT-2-2
    python main.py path/to/AT-2-3.fsp dqn                  # DQN on custom instance
    python main.py path/to/AT-1-1.fsp random 20            # random agent, 20 episodes
"""

import sys
from pathlib import Path

# environment.py starts the JVM at import time; all other Python imports follow.
from environment import DCSEnvironment
from agents.random_agent import run_episode
from agents.dqn_agent import DQNAgent, train

FSP_DEFAULT = str(
    Path(__file__).parent.parent / "fsp" / "NonBlocking" / "Benchmark" / "AT" / "AT-2-2.fsp"
)
RESULTS_DIR = str(Path(__file__).parent / "results")


def run_random(fsp_path: str, num_episodes: int) -> None:
    env = DCSEnvironment()
    print(f"Random agent | {Path(fsp_path).name} | {num_episodes} episodes")
    print("-" * 60)
    for ep in range(1, num_episodes + 1):
        info = run_episode(env, fsp_path)
        print(
            f"Episode {ep:2d} | steps={info['steps']:4d} | "
            f"reward={info['total_reward']:5d}"
        )


def run_dqn(fsp_path: str) -> None:
    env   = DCSEnvironment(feature_type="BASIC")
    agent = DQNAgent()

    print(f"DQN agent | {Path(fsp_path).name}")
    print(f"Stop: 1000 episodes | 100 000 steps | patience 100")
    print("-" * 60)

    train(
        env, agent, fsp_path,
        max_episodes=1000,
        max_steps=100_000,
        patience=100,
        results_dir=RESULTS_DIR,
        verbose=True,
    )


def main() -> None:
    fsp_path     = sys.argv[1] if len(sys.argv) > 1 else FSP_DEFAULT
    mode         = sys.argv[2] if len(sys.argv) > 2 else "dqn"
    num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    if mode == "random":
        run_random(fsp_path, num_episodes)
    elif mode == "dqn":
        run_dqn(fsp_path)
    else:
        print(f"Unknown mode '{mode}'. Choose: random | dqn")
        sys.exit(1)


if __name__ == "__main__":
    main()
