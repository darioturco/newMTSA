"""
Entry point for running agents against the DCS RL environment.

Usage
-----
    python main.py [fsp_path] [mode] [episodes]

Arguments
---------
fsp_path : path to the .fsp instance file
           (default: fsp/NonBlocking/Benchmark/AT/AT-2-2.fsp)
mode     : random | dqn | dqn-lstm | ppo | ppo-lstm | sac | sac-lstm
           (default: dqn)
episodes : number of episodes for random mode  (default: 5)

Examples
--------
    python main.py                                          # DQN on AT-2-2
    python main.py path/to/AT-2-3.fsp dqn                  # DQN flat
    python main.py path/to/AT-2-3.fsp dqn-lstm             # DQN with LSTM
    python main.py path/to/AT-2-3.fsp ppo                  # PPO flat
    python main.py path/to/AT-2-3.fsp ppo-lstm             # PPO with LSTM
    python main.py path/to/AT-2-3.fsp sac                  # SAC flat
    python main.py path/to/AT-2-3.fsp sac-lstm             # SAC with LSTM
    python main.py path/to/AT-1-1.fsp random 20            # random agent, 20 episodes
"""

import sys
from pathlib import Path

# environment.py starts the JVM at import time; all other Python imports follow.
from environment import DCSEnvironment
from agents.random_agent import run_episode
from agents.dqn_agent import DQNAgent, train
from agents.ppo_agent import PPOAgent, train as ppo_train
from agents.sac_agent import SACAgent, train as sac_train

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*model version conversion.*")

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


def run_dqn(fsp_path: str, use_lstm: bool = False) -> None:
    env   = DCSEnvironment(feature_type="BASIC")
    agent = DQNAgent(use_lstm=use_lstm)

    tag = "DQN-LSTM" if use_lstm else "DQN"
    print(f"{tag} agent | {Path(fsp_path).name}")
    print(f"Stop: 10000 episodes | 100 000 steps | patience 100")
    print("-" * 60)

    train(
        env, agent, fsp_path,
        max_episodes=10000,
        max_steps=100_000,
        patience=100,
        results_dir=RESULTS_DIR,
        verbose=True,
    )


def run_ppo(fsp_path: str, use_lstm: bool = False) -> None:
    env   = DCSEnvironment(feature_type="BASIC")
    agent = PPOAgent(use_lstm=use_lstm)

    tag = "PPO-LSTM" if use_lstm else "PPO"
    print(f"{tag} agent | {Path(fsp_path).name}")
    print(f"Stop: 10000 episodes | 100 000 steps | patience 100")
    print("-" * 60)

    ppo_train(
        env, agent, fsp_path,
        max_episodes=10000,
        max_steps=100_000,
        patience=100,
        results_dir=RESULTS_DIR,
        verbose=True,
    )


def run_sac(fsp_path: str, use_lstm: bool = False) -> None:
    env   = DCSEnvironment(feature_type="BASIC")
    agent = SACAgent(use_lstm=use_lstm)

    tag = "SAC-LSTM" if use_lstm else "SAC"
    print(f"{tag} agent | {Path(fsp_path).name}")
    print(f"Stop: 10000 episodes | 100 000 steps | patience 100")
    print("-" * 60)

    sac_train(
        env, agent, fsp_path,
        max_episodes=10000,
        max_steps=100_000,
        patience=100,
        results_dir=RESULTS_DIR,
        verbose=True,
    )


# TODO: make a config file with all the parameters for each agent, instead of hardcoding them here in main.py.
#   Each agent will have his own dictionary of parametrs and that dict is going to be passed at the train function.
def main() -> None:
    fsp_path     = sys.argv[1] if len(sys.argv) > 1 else None
    mode         = sys.argv[2] if len(sys.argv) > 2 else "dqn"
    if fsp_path is None:
        instance = "AT"
        n = 2
        k = 2
        fsp_path = f"{Path(__file__).parent.parent}\\fsp\\NonBlocking\\Benchmark\\{instance}\\{instance}-{n}-{k}.fsp"

    if mode == "random":
        run_random(fsp_path, 1)
    elif mode == "dqn":
        run_dqn(fsp_path, use_lstm=False)
    elif mode == "dqn-lstm":
        run_dqn(fsp_path, use_lstm=True)
    elif mode == "ppo":
        run_ppo(fsp_path, use_lstm=False)
    elif mode == "ppo-lstm":
        run_ppo(fsp_path, use_lstm=True)
    elif mode == "sac":
        run_sac(fsp_path, use_lstm=False)
    elif mode == "sac-lstm":
        run_sac(fsp_path, use_lstm=True)
    else:
        print(f"Unknown mode '{mode}'. Choose: random | dqn | dqn-lstm | ppo | ppo-lstm | sac | sac-lstm")
        sys.exit(1)


if __name__ == "__main__":
    main()
