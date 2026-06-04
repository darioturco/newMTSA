"""
Entry point for running agents against the DCS RL environment.

Usage
-----
    python main.py [fsp_path] [mode] [network] [episodes] [feature_type]
    python main.py --graph [output.png]

Arguments
---------
fsp_path     : path to the .fsp instance file
               (default: fsp/Blocking/Benchmark/DP/DP-2-2.fsp)
mode         : random | dqn | ppo | sac | path  (default: dqn)
network      : flat | lstm | transformer  (default: flat)
episodes     : number of episodes for random mode  (default: 1)
feature_type : BASIC | ROL | SUPER | SUPER_CUSTOM | SUPER_GENERAL  (default: ROL)
               SUPER / SUPER_CUSTOM / SUPER_GENERAL automatically use heuristic=SUPER_RL.

Examples
--------
    python main.py                                            # DQN flat on DP-2-2
    python main.py path/to/AT-2-3.fsp dqn                    # DQN flat
    python main.py path/to/AT-2-3.fsp dqn lstm               # DQN with LSTM
    python main.py path/to/AT-2-3.fsp dqn transformer        # DQN with Transformer
    python main.py path/to/AT-2-3.fsp ppo flat               # PPO flat
    python main.py path/to/AT-2-3.fsp ppo lstm               # PPO with LSTM
    python main.py path/to/AT-2-3.fsp sac transformer        # SAC with Transformer
    python main.py path/to/AT-1-1.fsp random flat 20         # random agent, 20 episodes
    python main.py path/to/AT-2-2.fsp dqn flat SUPER         # DQN + SUPER_RL heuristic
    python main.py path/to/AT-2-2.fsp ppo flat SUPER_CUSTOM  # PPO + SUPER_RL heuristic
    python main.py path/to/AT-2-2.fsp path flat SUPER        # PathLearning + SUPER_RL
    python main.py --graph                                    # bar plot of benchmark results
    python main.py --graph out.png                            # save graph to specific path
    python main.py --graph out.png 2500 blocking              # with budget and problem type in title
"""

import sys
import os
import time
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
from agents.path_learning_agent import PathLearningAgent, train as path_train
from config import DQN as DQN_CONFIG, PPO as PPO_CONFIG, SAC as SAC_CONFIG, TRAIN as TRAIN_CONFIG, PATH_LEARNING as PATH_LEARNING_CONFIG

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*model version conversion.*")

# ── hardcodeable defaults ──────────────────────────────────────────────────────
DEFAULT_FSP_PATH = "..\\fsp\\Blocking\\Benchmark\\DP\\DP-2-2.fsp"
DEFAULT_MODE     = "dqn"         # random | dqn | ppo | sac | path
DEFAULT_NETWORK  = "flat"        # flat | lstm | transformer
DEFAULT_EPISODES = 1             # only used for random mode

# ── feature set selection ───────────────────────────────────────────────────────
# BASIC        : action one-hot, state label, controllable, phase, deadlock, neighborhood, …
# ROL          : all BASIC features + role-based component encoding + action one-hot + has_index
# SUPER        : per-transition scoring features — all in [-1,1]. Requires heuristic=SUPER_RL
#                (auto-selected when feature_type is SUPER / SUPER_CUSTOM / SUPER_GENERAL).
# SUPER_CUSTOM : family-specific features (AT:7, DP:7, BW:8, CM:4, TL:2, TA:7). Flat only.
# SUPER_GENERAL: SUPER + cross-candidate index ranking; family-agnostic, no hardcoded substates. Flat only.
FEATURE_TYPE     = "ROL"       # BASIC | ROL | SUPER | SUPER_CUSTOM | SUPER_GENERAL  ← change here to switch globally
VERBOSE_HEURISTIC = True      # Print every SuperRL expansion + decision candidates (very noisy)
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_BASE = Path(__file__).parent / "results"

_SUPER_FEATURE_TYPES = ("SUPER", "SUPER_CUSTOM", "SUPER_GENERAL")


def _make_env(feature_type: str, verbose: bool = False) -> DCSEnvironment:
    """Build DCSEnvironment, auto-selecting SUPER_RL heuristic for SUPER features."""
    heuristic = "SUPER_RL" if feature_type in _SUPER_FEATURE_TYPES else "FIRST"
    return DCSEnvironment(heuristic=heuristic, feature_type=feature_type, verbose=verbose)


def _results_dir(fsp_path: str, agent: str, network: str, feature_type: str) -> str:
    """Full save path: results/<blocking>/<Family>/<feature>/<agent>_<network>"""
    p = Path(fsp_path)
    family   = p.parent.name
    parts_lc = [part.lower() for part in p.parts]
    if "nonblocking" in parts_lc:
        problem_type = "nonblocking"
    elif "blocking" in parts_lc:
        problem_type = "blocking"
    else:
        problem_type = "unknown"
    return str(RESULTS_BASE / problem_type / family / feature_type.lower() / f"{agent}_{network}")


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
    env   = _make_env(feature_type, VERBOSE_HEURISTIC)
    agent = DQNAgent(**{**DQN_CONFIG, **_network_flags(network)})

    tag_map = {"lstm": "DQN-LSTM", "transformer": "DQN-Transformer"}
    print(f"{tag_map.get(network, 'DQN')} agent | {Path(fsp_path).name} | features={feature_type}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | "
          f"{TRAIN_CONFIG['max_steps']:,} steps | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    train(env, agent, fsp_path, **TRAIN_CONFIG,
          results_dir=_results_dir(fsp_path, "dqn", network, feature_type), verbose=True)


def run_ppo(fsp_path: str, network: str, feature_type: str) -> None:
    env   = _make_env(feature_type, VERBOSE_HEURISTIC)
    agent = PPOAgent(**{**PPO_CONFIG, **_network_flags(network)})

    tag_map = {"lstm": "PPO-LSTM", "transformer": "PPO-Transformer"}
    print(f"{tag_map.get(network, 'PPO')} agent | {Path(fsp_path).name} | features={feature_type}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | "
          f"{TRAIN_CONFIG['max_steps']:,} steps | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    ppo_train(env, agent, fsp_path, **TRAIN_CONFIG,
              results_dir=_results_dir(fsp_path, "ppo", network, feature_type), verbose=True)


def run_sac(fsp_path: str, network: str, feature_type: str) -> None:
    env   = _make_env(feature_type, VERBOSE_HEURISTIC)
    agent = SACAgent(**{**SAC_CONFIG, **_network_flags(network)})

    tag_map = {"lstm": "SAC-LSTM", "transformer": "SAC-Transformer"}
    print(f"{tag_map.get(network, 'SAC')} agent | {Path(fsp_path).name} | features={feature_type}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | "
          f"{TRAIN_CONFIG['max_steps']:,} steps | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    sac_train(env, agent, fsp_path, **TRAIN_CONFIG,
              results_dir=_results_dir(fsp_path, "sac", network, feature_type), verbose=True)


def run_path(fsp_path: str, feature_type: str) -> None:
    env   = _make_env(feature_type, VERBOSE_HEURISTIC)
    agent = PathLearningAgent(**PATH_LEARNING_CONFIG)

    print(f"Path agent | {Path(fsp_path).name} | features={feature_type}")
    print(f"Stop: {TRAIN_CONFIG['max_episodes']} episodes | patience {TRAIN_CONFIG['patience']}")
    print("-" * 60)

    path_train(env, agent, fsp_path, **TRAIN_CONFIG,
               results_dir=_results_dir(fsp_path, "path", "flat", feature_type),
               verbose=True)


def train_agent(fsp_path: str, mode: str, network: str = "flat", feature_type: str = FEATURE_TYPE, episodes: int = 1) -> None:
    if feature_type in _SUPER_FEATURE_TYPES and network != "flat":
        print(f"{feature_type} features only support flat network (got '{network}').")
        sys.exit(1)

    if mode == "random":
        run_random(fsp_path, episodes)
    elif mode == "dqn":
        run_dqn(fsp_path, network, feature_type)
    elif mode == "ppo":
        run_ppo(fsp_path, network, feature_type)
    elif mode == "sac":
        run_sac(fsp_path, network, feature_type)
    elif mode == "path":
        run_path(fsp_path, feature_type)
    else:
        print(f"Unknown mode '{mode}'. Choose: random | dqn | ppo | sac | path")
        sys.exit(1)


def main() -> None:
    # --graph is intercepted before this point
    fsp_path     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FSP_PATH
    mode         = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODE
    network      = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_NETWORK
    episodes     = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_EPISODES
    feature_type = sys.argv[5] if len(sys.argv) > 5 else FEATURE_TYPE

    #train_agent(fsp_path, mode, network, feature_type, episodes)
    
    #train_agent("..\\fsp\\Blocking\\Benchmark\\TL\\TL-2-2.fsp", "path", "flat", "SUPER_CUSTOM")
    train_agent("..\\fsp\\Blocking\\Benchmark\\AT\\AT-2-2.fsp", "path", "flat", "SUPER_CUSTOM")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\DP\\DP-2-2.fsp", "path", "flat", "SUPER_CUSTOM")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\BW\\BW-2-2.fsp", "path", "flat", "SUPER_CUSTOM")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\TA\\TA-2-2.fsp", "path", "flat", "SUPER_CUSTOM")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\CM\\CM-2-2.fsp", "path", "flat", "SUPER_CUSTOM")



    #train_agent("..\\fsp\\Blocking\\Benchmark\\TL\\TL-2-2.fsp", "path", "flat", "SUPER_GENERAL")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\AT\\AT-2-2.fsp", "path", "flat", "SUPER_GENERAL")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\DP\\DP-2-2.fsp", "path", "flat", "SUPER_GENERAL")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\BW\\BW-2-2.fsp", "path", "flat", "SUPER_GENERAL")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\TA\\TA-2-2.fsp", "path", "flat", "SUPER_GENERAL")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\CM\\CM-2-2.fsp", "path", "flat", "SUPER_GENERAL")



    #train_agent("..\\fsp\\Blocking\\Benchmark\\TA\\TA-2-2.fsp", "dqn", "flat", "ROL")
    #train_agent("..\\fsp\\Blocking\\Benchmark\\TA\\TA-2-2.fsp", "dqn", "flat", "BASIC")

    #os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0") # Computer sleep after training to save energy

if __name__ == "__main__":
    try:
        t0 = time.time()
        main()
        elapsed = time.time() - t0
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"\nTraining time: {h:02d}:{m:02d}:{s:02d} ({elapsed:.1f}s total)")
    except KeyboardInterrupt:
        print("\n[Interrupted]", flush=True)
        os._exit(0)
