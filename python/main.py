"""
Entry point for running agents against the DCS RL environment.

Usage
-----
    python main.py [fsp_path] [mode] [network] [episodes]
    python main.py --graph [output.png]

Arguments
---------
fsp_path : path to the .fsp instance file
           (default: fsp/NonBlocking/Benchmark/AT/AT-2-2.fsp)
mode     : random | dqn | ppo | sac | superdfs_dqn | superdfs_ppo | superdfs_sac  (default: dqn)
network  : flat | lstm | transformer  (default: flat)
episodes : number of episodes for random mode  (default: 1)

Examples
--------
    python main.py                                            # DQN flat on AT-2-2
    python main.py path/to/AT-2-3.fsp dqn                    # DQN flat
    python main.py path/to/AT-2-3.fsp dqn lstm               # DQN with LSTM
    python main.py path/to/AT-2-3.fsp dqn transformer        # DQN with Transformer
    python main.py path/to/AT-2-3.fsp ppo flat               # PPO flat
    python main.py path/to/AT-2-3.fsp ppo lstm               # PPO with LSTM
    python main.py path/to/AT-2-3.fsp sac transformer        # SAC with Transformer
    python main.py path/to/AT-1-1.fsp random flat 20         # random agent, 20 episodes
    python main.py path/to/AT-2-2.fsp superdfs_dqn           # SuperDFS + DQN picker (SUPER features)
    python main.py path/to/AT-2-2.fsp superdfs_ppo           # SuperDFS + PPO picker (SUPER features)
    python main.py path/to/AT-2-2.fsp superdfs_sac           # SuperDFS + SAC picker (SUPER features)
    python main.py path/to/AT-2-2.fsp dqn flat SUPER         # same as superdfs_dqn (SUPER features → flat only)
    python main.py --graph                                    # bar plot of benchmark results
    python main.py --graph out.png                            # save graph to specific path
    python main.py --graph out.png 2500 blocking              # with budget and problem type in title
"""

import sys
import os
from pathlib import Path

# Handle --graph before JVM-starting imports so it works without Java/ONNX.
if len(sys.argv) > 1 and sys.argv[1] == "--graph":
    from graph_results import graph_results
    output       = sys.argv[2] if len(sys.argv) > 2 else None
    budget       = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
    problem_type = sys.argv[4] if len(sys.argv) > 4 else None
    graph_results(output, budget, problem_type)
    sys.exit(0)

# environment.py starts the JVM at import time; all other Python imports follow.
from environment import DCSEnvironment
from agents.random_agent import run_episode
from agents.dqn_agent import DQNAgent, train
from agents.ppo_agent import PPOAgent, train as ppo_train
from agents.sac_agent import SACAgent, train as sac_train
from config import DQN as DQN_CONFIG, PPO as PPO_CONFIG, SAC as SAC_CONFIG, TRAIN as TRAIN_CONFIG

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*model version conversion.*")

# ── hardcodeable defaults ──────────────────────────────────────────────────────
DEFAULT_FSP_PATH = "..\\fsp\\Blocking\\Benchmark\\DP\\DP-2-2.fsp"          # None → built-in AT-2-2.fsp
DEFAULT_MODE     = "dqn"         # random | dqn | ppo | sac
DEFAULT_NETWORK  = "flat"        # flat | lstm | transformer  ← change here to switch globally
DEFAULT_EPISODES = 1             # only used for random mode

# ── feature set selection ───────────────────────────────────────────────────────
# BASIC : action one-hot, state label, controllable, phase, deadlock, neighborhood, …
# ROL   : all BASIC features + role-based component encoding + action one-hot + has_index
# SUPER : per-transition scoring features (abstract action, normalised indices,
#         dest state, role fractions, changed-role encoding, phase) — all in [-1,1].
#         Designed for cross-instance generalisation. Use with heuristic=SUPER_RL
#         (superdfs_dqn mode) or with standard dqn/ppo/sac (full frontier scored).
FEATURE_TYPE     = "ROL"       # BASIC | ROL | SUPER  ← change here to switch globally
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_BASE = Path(__file__).parent / "results"


def _results_dir(fsp_path: str, agent: str, network: str, feature_type: str) -> str:
    """Full save path: results/<blocking>/<Family>/<feature>/<agent>_<network>"""
    p = Path(fsp_path)
    family   = p.parent.name                                      # e.g. "DP"
    parts_lc = [part.lower() for part in p.parts]
    if "nonblocking" in parts_lc:
        problem_type = "nonblocking"
    elif "blocking" in parts_lc:
        problem_type = "blocking"
    else:
        problem_type = "unknown"
    feature  = feature_type.lower()                               # e.g. "rol"
    return str(RESULTS_BASE / problem_type / family / feature / f"{agent}_{network}")


def _network_flags(network: str) -> dict:
    return {
        "use_lstm":        network == "lstm",
        "use_transformer": network == "transformer",
    }


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


def run_dqn(fsp_path: str, network: str, feature_type: str) -> None:
    env   = DCSEnvironment(feature_type=feature_type)
    agent = DQNAgent(**{**DQN_CONFIG, **_network_flags(network)})

    tag_map = {"lstm": "DQN-LSTM", "transformer": "DQN-Transformer"}
    print(f"{tag_map.get(network, 'DQN')} agent | {Path(fsp_path).name}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | "
          f"{TRAIN_CONFIG['max_steps']:,} steps | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    train(env, agent, fsp_path, **TRAIN_CONFIG,
          results_dir=_results_dir(fsp_path, "dqn", network, feature_type), verbose=True)


def run_ppo(fsp_path: str, network: str, feature_type: str) -> None:
    env   = DCSEnvironment(feature_type=feature_type)
    agent = PPOAgent(**{**PPO_CONFIG, **_network_flags(network)})

    tag_map = {"lstm": "PPO-LSTM", "transformer": "PPO-Transformer"}
    print(f"{tag_map.get(network, 'PPO')} agent | {Path(fsp_path).name}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | "
          f"{TRAIN_CONFIG['max_steps']:,} steps | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    ppo_train(env, agent, fsp_path, **TRAIN_CONFIG,
              results_dir=_results_dir(fsp_path, "ppo", network, feature_type), verbose=True)


def run_superdfs_dqn(fsp_path: str) -> None:
    """SuperDFS heuristic with a DQN flat picker for the controllable choice."""
    env   = DCSEnvironment(heuristic="SUPER_RL", feature_type="SUPER")
    agent = DQNAgent(**{**DQN_CONFIG, "use_lstm": False, "use_transformer": False})

    print(f"SuperDFS-DQN | {Path(fsp_path).name}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | "
          f"{TRAIN_CONFIG['max_steps']:,} steps | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    train(env, agent, fsp_path, **TRAIN_CONFIG,
          results_dir=_results_dir(fsp_path, "superdfs_dqn", "flat", "super"), verbose=True)


def run_superdfs_ppo(fsp_path: str) -> None:
    """SuperDFS heuristic with a PPO flat picker for the controllable choice."""
    env   = DCSEnvironment(heuristic="SUPER_RL", feature_type="SUPER")
    agent = PPOAgent(**{**PPO_CONFIG, "use_lstm": False, "use_transformer": False})

    print(f"SuperDFS-PPO | {Path(fsp_path).name}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | "
          f"{TRAIN_CONFIG['max_steps']:,} steps | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    ppo_train(env, agent, fsp_path, **TRAIN_CONFIG,
              results_dir=_results_dir(fsp_path, "superdfs_ppo", "flat", "super"), verbose=True)


def run_superdfs_sac(fsp_path: str) -> None:
    """SuperDFS heuristic with a SAC flat picker for the controllable choice."""
    env   = DCSEnvironment(heuristic="SUPER_RL", feature_type="SUPER")
    agent = SACAgent(**{**SAC_CONFIG, "use_lstm": False, "use_transformer": False})

    print(f"SuperDFS-SAC | {Path(fsp_path).name}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | "
          f"{TRAIN_CONFIG['max_steps']:,} steps | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    sac_train(env, agent, fsp_path, **TRAIN_CONFIG,
              results_dir=_results_dir(fsp_path, "superdfs_sac", "flat", "super"), verbose=True)


def run_sac(fsp_path: str, network: str, feature_type: str) -> None:
    env   = DCSEnvironment(feature_type=feature_type)
    agent = SACAgent(**{**SAC_CONFIG, **_network_flags(network)})

    tag_map = {"lstm": "SAC-LSTM", "transformer": "SAC-Transformer"}
    print(f"{tag_map.get(network, 'SAC')} agent | {Path(fsp_path).name}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | "
          f"{TRAIN_CONFIG['max_steps']:,} steps | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    sac_train(env, agent, fsp_path, **TRAIN_CONFIG,
              results_dir=_results_dir(fsp_path, "sac", network, feature_type), verbose=True)


def train_agent(fsp_path: str, mode: str, network: str = "flat", feature_type: str = FEATURE_TYPE, episodes: int = 1) -> None:
    if feature_type == "SUPER":
        if network != "flat":
            print(f"SUPER features only support flat network (got '{network}').")
            sys.exit(1)
        if mode == "dqn" or mode == "superdfs_dqn":
            run_superdfs_dqn(fsp_path)
        elif mode == "ppo" or mode == "superdfs_ppo":
            run_superdfs_ppo(fsp_path)
        elif mode == "sac" or mode == "superdfs_sac":
            run_superdfs_sac(fsp_path)
        else:
            print(f"Unknown mode '{mode}' for SUPER features. Choose: dqn | ppo | sac")
            sys.exit(1)
    elif mode == "random":
        run_random(fsp_path, episodes)
    elif mode == "dqn":
        run_dqn(fsp_path, network, feature_type)
    elif mode == "ppo":
        run_ppo(fsp_path, network, feature_type)
    elif mode == "sac":
        run_sac(fsp_path, network, feature_type)
    elif mode == "superdfs_dqn":
        run_superdfs_dqn(fsp_path)
    elif mode == "superdfs_ppo":
        run_superdfs_ppo(fsp_path)
    elif mode == "superdfs_sac":
        run_superdfs_sac(fsp_path)
    else:
        print(
            f"Unknown mode '{mode}'. Choose: random | dqn | ppo | sac | superdfs_dqn | superdfs_ppo | superdfs_sac"
        )
        sys.exit(1)


def main() -> None:
    # --graph is intercepted before this point
    fsp_path     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FSP_PATH
    mode         = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODE
    network      = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_NETWORK
    episodes     = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_EPISODES
    feature_type = sys.argv[5] if len(sys.argv) > 5 else FEATURE_TYPE

    #train_agent(fsp_path, mode, network, feature_type, episodes)
    train_agent("..\\fsp\\Blocking\\Benchmark\\AT\\AT-2-2.fsp", "dqn", "flat", "SUPER")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\TA\\TA-2-2.fsp", "dqn", "flat", "ROL")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\TA\\TA-2-2.fsp", "dqn", "flat", "BASIC")
    


    #os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0") # Computer sleep after training to save energy

if __name__ == "__main__":
    main()
