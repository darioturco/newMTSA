"""
PathLearning agent for the DCS RL environment.

Algorithm
---------
Round 0  — Run N random episodes. Record (frontier_feats, chosen_index) at every
           decision point and the episode's cumulative reward.
Round k  — Run N episodes using the current network (argmax of network score).
           Same recording.

After each round:
  1. Pick the episode with highest cumulative reward.
  2. Build a binary classification dataset from it:
       chosen transition  → y = 1
       all other transitions at the same step → y = 0
  3. Reinitialise the network weights and train to overfit (BCE, many epochs).

Designed for heuristic=SUPER_RL with SUPER or SUPER_CUSTOM features.
"""

import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import MLPScorer


class PathLearningAgent:
    """
    Parameters
    ----------
    random_episodes  : exploration episodes in round 0 (random policy)
    guided_episodes  : episodes per round k>0 (network argmax policy)
    random_explore   : random episodes run after replaying a known best_path
    max_rounds      : outer loop iterations
    train_epochs    : gradient steps per round used to overfit the dataset
    hidden_size     : MLP hidden units
    lr              : Adam learning rate
    save_frequency  : save ONNX checkpoint every N rounds
    best_path       : dict mapping family name → action list, or None
    """

    def __init__(
        self,
        random_episodes: int = 10,
        guided_episodes: int = 10,
        max_rounds: int = 500,
        train_epochs: int = 5000,
        hidden_size: int = 128,
        lr: float = 1e-3,
        save_frequency: int = 10,
        epsilon_decay: float = 0.995,
        epsilon_end: float = 0.0,
        explore_all: bool = False,
        best_path: Optional[dict] = None,
        random_explore: int = 100,
    ):
        self.random_episodes = random_episodes
        self.guided_episodes = guided_episodes
        self.max_rounds      = max_rounds
        self.train_epochs    = train_epochs
        self.hidden_size     = hidden_size
        self.lr              = lr
        self.save_frequency  = save_frequency
        self.epsilon_decay   = epsilon_decay
        self.epsilon_end     = epsilon_end
        self.explore_all     = explore_all
        self.best_path       = best_path
        self.random_explore  = random_explore
        self.epsilon: float  = 1.0

        self.net: Optional[nn.Module]    = None
        self.optimizer                   = None
        self.feature_size: Optional[int] = None
        self._initialized: bool          = False
        self.trained: bool               = False

    # ── network init ──────────────────────────────────────────────────────────

    def _init_network(self, feature_size: int) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.feature_size = feature_size
        self.net       = MLPScorer(feature_size, self.hidden_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def _reset_network(self) -> None:
        """Reinitialise weights so each round trains from scratch on the new dataset."""
        if not self._initialized:
            return
        self.net       = MLPScorer(self.feature_size, self.hidden_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    # ── episode running ───────────────────────────────────────────────────────

    def _run_episode_random(self, env, fsp_path: str) -> tuple:
        """Random episode. Returns (episode_data, total_reward, realizable, director_transitions, transitions_explored)."""
        frontier = env.reset(fsp_path)
        if not self._initialized and frontier:
            self._init_network(len(frontier[0]))
        episode_data    = []
        total_reward    = 0
        ep_info         = {}
        cum_expansions  = 0
        while not env.is_finished:
            if not frontier:
                break
            action = random.randrange(len(frontier))
            episode_data.append((list(frontier), action, cum_expansions))
            frontier, _, done, ep_info = env.step(action)
            cum_expansions += ep_info.get("expansions", 0)
            total_reward   -= ep_info.get("expansions", 0)
        return (episode_data, total_reward, ep_info.get("realizable"),
                ep_info.get("director_transitions"), ep_info.get("transitions_explored"))

    def _run_episode_fixed(self, env, fsp_path: str, actions: list) -> tuple:
        """Replay a fixed action sequence. Returns same 5-tuple as other episode runners."""
        frontier = env.reset(fsp_path)
        if not self._initialized and frontier:
            self._init_network(len(frontier[0]))
        episode_data   = []
        total_reward   = 0
        ep_info        = {}
        cum_expansions = 0
        for step, action in enumerate(actions):
            if env.is_finished or not frontier:
                break
            if action >= len(frontier):
                path_so_far = [a for _, a, *_ in episode_data]
                raise ValueError(
                    f"best_path[{step}] = {action} out of range: "
                    f"frontier has {len(frontier)} element(s).\n"
                    f"Path replayed so far: {path_so_far}\n"
                    f"Verify best_path was obtained from this exact instance."
                )
            episode_data.append((list(frontier), action, cum_expansions))
            frontier, _, done, ep_info = env.step(action)
            cum_expansions += ep_info.get("expansions", 0)
            total_reward   -= ep_info.get("expansions", 0)
        return (episode_data, total_reward, ep_info.get("realizable"),
                ep_info.get("director_transitions"), ep_info.get("transitions_explored"))

    def _run_episode_guided(self, env, fsp_path: str) -> tuple:
        """Epsilon-greedy guided episode. Returns (episode_data, total_reward, realizable, director_transitions, transitions_explored)."""
        frontier = env.reset(fsp_path)
        episode_data   = []
        total_reward   = 0
        ep_info        = {}
        cum_expansions = 0
        while not env.is_finished:
            if not frontier:
                break
            if random.random() < self.epsilon:
                action = random.randrange(len(frontier))
            else:
                action = self._select_action(frontier)
            episode_data.append((list(frontier), action, cum_expansions))
            frontier, _, done, ep_info = env.step(action)
            cum_expansions += ep_info.get("expansions", 0)
            total_reward   -= ep_info.get("expansions", 0)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return (episode_data, total_reward, ep_info.get("realizable"),
                ep_info.get("director_transitions"), ep_info.get("transitions_explored"))

    def _explore_all_episodes(self, env, fsp_path: str, verbose: bool = False) -> list:
        """DFS over all paths via prefix replay, keeping only min-step (fewest decisions) episodes."""
        all_episodes = []
        ep_count     = [0]
        min_steps    = [float("inf")]

        def dfs(action_prefix: list):
            depth = len(action_prefix)
            if depth > min_steps[0]:
                return

            frontier = env.reset(fsp_path)
            if not self._initialized and frontier:
                self._init_network(len(frontier[0]))

            episode_data     = []
            total_reward     = 0
            cum_expansions   = 0
            ep_info          = {}

            for action in action_prefix:
                if env.is_finished or not frontier:
                    break
                episode_data.append((list(frontier), action, cum_expansions))
                frontier, _, done, ep_info = env.step(action)
                cum_expansions += ep_info.get("expansions", 0)
                total_reward   -= ep_info.get("expansions", 0)

            if env.is_finished or not frontier:
                realizable = ep_info.get("realizable")
                if not realizable:
                    return
                steps = len(episode_data)
                if steps < min_steps[0]:
                    min_steps[0] = steps
                    all_episodes.clear()
                if steps == min_steps[0]:
                    ep_count[0] += 1
                    tup = (episode_data, total_reward, realizable,
                           ep_info.get("director_transitions"), ep_info.get("transitions_explored"))
                    all_episodes.append(tup)
                    if verbose:
                        path_str = " ".join(str(a) for _, a, *_ in episode_data)
                        dt  = ep_info.get("director_transitions")
                        tr  = ep_info.get("transitions_explored")
                        dt_str = f" | director={dt:5d}" if dt is not None else ""
                        tr_str = f" | expanded={tr:6d}" if tr is not None else ""
                        print(f"[Explore {ep_count[0]:5d}] steps={steps:3d}"
                              f"{dt_str}{tr_str} | reward={total_reward:7d}"
                              f" | path=[{path_str}]", flush=True)
                return

            for action in range(len(frontier)):
                dfs(action_prefix + [action])

        dfs([])
        if verbose:
            if all_episodes:
                best_ep  = max(all_episodes, key=lambda x: x[1])
                path_str = " ".join(str(a) for _, a, *_ in best_ep[0])
                dt = best_ep[3]
                te = best_ep[4]
                print(f"[Explore done] Path found (steps={min_steps[0]}, director_transitions={dt}, transitions={te}): [{path_str}]", flush=True)
            else:
                print(f"[Explore done] No realizable path found", flush=True)
        return all_episodes

    def _select_action(self, frontier_feats: list) -> int:
        with torch.no_grad():
            t = torch.tensor(np.array(frontier_feats, dtype=np.float32))
            return int(self.net(t).argmax())

    def _run_episode_greedy(self, env, fsp_path: str) -> tuple:
        """Greedy episode using trained network (epsilon=0). Same 5-tuple as other runners."""
        frontier = env.reset(fsp_path)
        episode_data   = []
        total_reward   = 0
        ep_info        = {}
        cum_expansions = 0
        while not env.is_finished:
            if not frontier:
                break
            action = self._select_action(frontier)
            episode_data.append((list(frontier), action, cum_expansions))
            frontier, _, done, ep_info = env.step(action)
            cum_expansions += ep_info.get("expansions", 0)
            total_reward   -= ep_info.get("expansions", 0)
        return (episode_data, total_reward, ep_info.get("realizable"),
                ep_info.get("director_transitions"), ep_info.get("transitions_explored"))

    # ── dataset construction ──────────────────────────────────────────────────

    def _build_dataset(self, episode_data: list) -> tuple:
        """Build (X, y) from one episode. Chosen → y=1, rest → y=0."""
        X, y = [], []
        for frontier_feats, chosen, *_ in episode_data:
            for i, feat in enumerate(frontier_feats):
                X.append(feat)
                y.append(1.0 if i == chosen else 0.0)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _build_ranked_dataset(self, step_map: dict,
                               extra_negatives: list = None) -> tuple:
        """Build (X, y) with ranked integer targets derived from step_map.

        step_map maps tuple(feat) -> (gano, perdio) where:
          gano  = steps where this FV was chosen in best_data
          perdio = steps where this FV was in the frontier but NOT chosen

        Ranking logic
        -------------
        "A dominates B" if gano_A ∩ perdio_B != ∅:
          A won at some step where B was present and lost, so A > B.

        y for non-winners (gano=[]) : -1
        y for winners               : len(gano) - n_dominated_by
          where n_dominated_by = number of distinct winner FVs that dominate this one.

        Example with gano_sizes all = 3:
          FV1 dominated by 0 others → y = 3
          FV3 dominated by FV1      → y = 2
          FV4 dominated by FV1, FV3 → y = 1
          FV2 no wins               → y = -1

        Trained with MSELoss (not BCE) because targets can be negative or > 1.
        During inference argmax is used, so absolute scale does not matter —
        only the ordering needs to be learned.

        Inconsistency: if A dominates B AND B dominates A the partial order has
        a cycle and the network cannot satisfy both constraints. Printed as warning.
        """
        # Step-sets won by each winner FV, used for dominance checks
        winner_sets = {k: set(gano) for k, (gano, _) in step_map.items() if gano}

        # Detect cycles of length 2: A > B and B > A simultaneously
        winner_list = list(winner_sets.items())
        for i, (k_a, gano_a) in enumerate(winner_list):
            perdio_a = set(step_map[k_a][1])
            for k_b, gano_b in winner_list[i + 1:]:
                perdio_b = set(step_map[k_b][1])
                # A dominates B: A won at a step where B also appeared
                # B dominates A: B won at a step where A also appeared
                if (gano_a & perdio_b) and (gano_b & perdio_a):
                    print(
                        f"  [INCONSISTENCY] FV {list(k_a)} beats FV {list(k_b)} "
                        f"AND FV {list(k_b)} beats FV {list(k_a)}",
                        flush=True,
                    )

        X, y = [], []
        for key, (gano, perdio) in step_map.items():
            if not gano:
                # Never chosen → lowest score
                y_val = -1.0
            else:
                perdio_set = set(perdio)
                # Count how many other winners ever beat this FV
                n_beaten_by = sum(
                    1 for k2, gs in winner_sets.items()
                    if k2 != key and gs & perdio_set
                )
                # More wins + fewer dominators = higher rank
                y_val = float(len(gano) - n_beaten_by)
            X.append(list(key))
            y.append(y_val)

        # FVs from random episodes not seen in best_data → forced negative
        if extra_negatives:
            existing = {tuple(x) for x in X}
            for feat in extra_negatives:
                key = tuple(feat)
                if key not in existing and key not in step_map:
                    X.append(list(feat))
                    y.append(-1.0)
                    existing.add(key)

        # Collapse gaps: replace positive y values with dense ranks (1, 2, 3, ...)
        # so [-1, 1, 4, 60, -1] → [-1, 1, 2, 3, -1]. Ties share the same rank.
        unique_pos = sorted(set(v for v in y if v > 0))
        rank_map = {v: float(r + 1) for r, v in enumerate(unique_pos)}
        y = [rank_map[v] if v > 0 else v for v in y]

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── training ──────────────────────────────────────────────────────────────

    def _train_until_converged(self, X: np.ndarray, y: np.ndarray,
                               tol: float = 1e-5, max_epochs: int = 200_000,
                               verbose: bool = False,
                               step_map: dict = None,
                               use_mse: bool = False) -> float:
        """Train from scratch until loss < tol or max_epochs. Returns final loss."""
        if not self._initialized or len(X) == 0:
            return float("inf")
        self._reset_network()
        X_t     = torch.tensor(X)
        y_t     = torch.tensor(y)
        loss_fn = nn.MSELoss() if use_mse else nn.BCEWithLogitsLoss()
        last_loss = float("inf")
        for epoch in range(max_epochs):
            self.optimizer.zero_grad()
            loss = loss_fn(self.net(X_t).squeeze(-1), y_t)
            loss.backward()
            self.optimizer.step()
            last_loss = loss.item()
            if last_loss < tol:
                print(f"  Converged at epoch {epoch + 1} | loss={last_loss:.2e}", flush=True)
                break
        if verbose:
            with torch.no_grad():
                preds = self.net(X_t).squeeze(-1).tolist()
            n_pos = int((y > 0).sum())
            n_neg = len(y) - n_pos
            print(f"  [Training set] {len(X)} samples ({n_pos} positive, {n_neg} negative):", flush=True)
            for feat, label, pred in zip(X, y, preds):
                sign = "+" if label > 0 else "-"
                feat_str = " ".join(f"{v:6.3f}" for v in feat)
                steps_str = ""
                if step_map is not None:
                    entry = step_map.get(tuple(feat))
                    if entry:
                        gano, perdio = entry
                        parts = []
                        if gano:
                            parts.append(f"gano={gano}")
                        if perdio:
                            parts.append(f"perdio={perdio}")
                        if parts:
                            steps_str = " | " + " | ".join(parts)
                print(f"    [{sign}] y={label:+.1f} pred={pred:7.4f} | [{feat_str}]{steps_str}", flush=True)
        return last_loss

    def _print_best_path(self, best_data: list) -> None:
        path_str = " ".join(str(a) for _, a, *_ in best_data)
        print(f"  [Best path] decisions={len(best_data)} | path=[{path_str}]", flush=True)
        with torch.no_grad():
            for frontier_feats, chosen, *rest in best_data:
                cum_exp = rest[0] if rest else "?"
                t = torch.tensor(np.array(frontier_feats, dtype=np.float32))
                scores = self.net(t).tolist()
                print(f"  [Step {cum_exp+1}] {len(frontier_feats)} candidate(s):", flush=True)
                for i, (feat, score) in enumerate(zip(frontier_feats, scores)):
                    feat_str = " ".join(f"{v:6.3f}" for v in feat)
                    marker = "  <- CHOSEN" if i == chosen else ""
                    print(f"    [{i}] score={score:7.4f} | [{feat_str}]{marker}", flush=True)

    # ── persistence ───────────────────────────────────────────────────────────

    def save_onnx(self, path: str) -> None:
        if not self._initialized:
            return
        self.net.eval()
        torch.onnx.export(
            self.net, torch.zeros(1, self.feature_size), path,
            input_names=["features"], output_names=["score"],
            dynamic_axes={"features": {0: "N"}, "score": {0: "N"}},
            opset_version=17, export_params=True, dynamo=False,
        )

    def save_model(self, path: str) -> None:
        if not self._initialized:
            return
        torch.save({
            "feature_size": self.feature_size,
            "hidden_size":  self.hidden_size,
            "net":          self.net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
        }, path)

    @classmethod
    def load_model(cls, path: str, **kwargs) -> "PathLearningAgent":
        payload = torch.load(path, map_location="cpu")
        agent = cls(hidden_size=payload["hidden_size"], **kwargs)
        agent._init_network(payload["feature_size"])
        agent.net.load_state_dict(payload["net"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        return agent


# ── shared supervised training helper ─────────────────────────────────────────

def _supervised_train_on_path(
    agent: "PathLearningAgent",
    env,
    fsp_path: str,
    best_data: list,
    best_director,
    best_transitions,
    random_ep_data: list,
    results_path: Path,
    label: str,
    verbose: bool,
) -> None:
    """Build dataset from best_data (positives) + random episodes (negatives) and train to convergence."""
    if verbose:
        dt_str   = f" | director={best_director:5d}" if best_director is not None else ""
        tr_str   = f" | expanded={best_transitions:6d}" if best_transitions is not None else ""
        path_str = " ".join(str(a) for _, a, *_ in best_data)
        print(f"[{label}] Solved{dt_str}{tr_str} | steps={len(best_data)} | path=[{path_str}]", flush=True)

    step_map = {}
    for step_idx, (frontier_feats, chosen, *_) in enumerate(best_data):
        chosen_key = tuple(frontier_feats[chosen])
        seen_keys = set()
        for feat in frontier_feats:
            key = tuple(feat)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if key not in step_map:
                step_map[key] = ([], [])
            if key == chosen_key:
                step_map[key][0].append(step_idx)
            else:
                step_map[key][1].append(step_idx)

    extra_neg = [feat for ep_data in random_ep_data
                 for frontier, _, *_ in ep_data for feat in frontier]
    X, y = agent._build_ranked_dataset(step_map, extra_negatives=extra_neg)
    n_pos = int((y > 0).sum())

    if verbose:
        print(f"[{label}] Dataset: {len(X)} unique vectors"
              f" ({n_pos} positive, {len(X) - n_pos} negative) | Training to convergence ...", flush=True)

    loss = agent._train_until_converged(X, y, verbose=verbose, step_map=step_map, use_mse=True)

    if verbose:
        print(f"[{label}] Final loss={loss:.2e}", flush=True)
        print(f"[{label}] Network greedy replay:", flush=True)
    net_data, _, _, r_director, r_transitions = agent._run_episode_greedy(env, fsp_path)
    net_path = [a for _, a, *_ in net_data]
    best_path_actions = [a for _, a, *_ in best_data]
    match = net_path == best_path_actions
    if verbose:
        agent._print_best_path(net_data)
        if not match:
            print(f"  [Replay] path=[{' '.join(str(a) for a in net_path)}] (best=[{' '.join(str(a) for a in best_path_actions)}])", flush=True)
    else:
        best_path_str = " ".join(str(a) for a in best_path_actions)
        net_path_str  = " ".join(str(a) for a in net_path)
        print(f"  [Replay] best=[{best_path_str}]", flush=True)
        print(f"           got= [{net_path_str}] | match={match}", flush=True)

    agent.save_onnx(str(results_path / "path_best.onnx"))
    agent.save_model(str(results_path / "path_best.pt"))


# ── training loop ──────────────────────────────────────────────────────────────

def train(
    env,
    agent: PathLearningAgent,
    fsp_path: str,
    max_episodes: int = 5000,
    max_steps: int = 1_000_000,   # unused; kept for API compatibility
    patience: int = 500,
    results_dir: str = "results",
    verbose: bool = True,
) -> PathLearningAgent:

    env.reset(fsp_path)
    print(f"Instance type : {'non-blocking' if env.is_nonblocking else 'blocking'}")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "training.csv"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["round", "phase", "best_reward", "train_loss", "best_steps",
             "director_transitions", "transitions_explored", "epsilon", "realizable"]
        )

    # Resolve effective best_path: dict lookup by family, or None
    family = Path(fsp_path).stem.split('-')[0]
    if isinstance(agent.best_path, dict):
        effective_best_path = agent.best_path.get(family)
    else:
        effective_best_path = None

    total_episodes = 0
    best_ever_reward = float("-inf")
    no_improve       = 0
    stop_reason      = f"max rounds ({agent.max_rounds})"

    # ── known best path: run 100 random episodes, pick overall best, train ───
    if effective_best_path is not None:
        env.set_verbose(False)
        fixed_tup = agent._run_episode_fixed(env, fsp_path, effective_best_path)
        fixed_data, fixed_reward, fixed_real, fixed_director, fixed_transitions = fixed_tup

        print(f"[best_path] Replaying known path (reward={fixed_reward}) | "
              f"Running {agent.random_explore} random episodes ...", flush=True)
        random_episodes = []
        for _ in range(agent.random_explore):
            ep_tup = agent._run_episode_random(env, fsp_path)
            random_episodes.append(ep_tup)

        best_random_tup = max(random_episodes, key=lambda x: x[1])
        if best_random_tup[1] > fixed_reward:
            best_data, best_reward, best_real, best_director, best_transitions = best_random_tup
            print(f"[best_path] Random found better path (reward={best_reward})", flush=True)
        else:
            best_data, best_reward, best_real, best_director, best_transitions = (
                fixed_data, fixed_reward, fixed_real, fixed_director, fixed_transitions
            )
            print(f"[best_path] Known path remains best (reward={best_reward})", flush=True)

        all_ep_data = [fixed_data] + [ep[0] for ep in random_episodes]
        _supervised_train_on_path(
            agent, env, fsp_path,
            best_data, best_director, best_transitions,
            all_ep_data, results_path, "best_path", verbose,
        )
        agent.trained = True
        return agent

    # ── explore_all: DFS finds best path → same one-shot training then stop ──
    if agent.explore_all:
        if verbose:
            print(f"[explore_all] Exhaustive DFS over all paths ...", flush=True)
        episodes = agent._explore_all_episodes(env, fsp_path, verbose=verbose)
        total_episodes += len(episodes)

        global_best = max(episodes, key=lambda x: x[1])
        best_data, _, best_real, best_director, best_transitions = global_best

        step_map = {}
        for step_idx, (frontier_feats, chosen, *_) in enumerate(best_data):
            chosen_key = tuple(frontier_feats[chosen])
            seen_keys = set()
            for feat in frontier_feats:
                key = tuple(feat)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if key not in step_map:
                    step_map[key] = ([], [])
                if key == chosen_key:
                    step_map[key][0].append(step_idx)
                else:
                    step_map[key][1].append(step_idx)
        X, y = agent._build_ranked_dataset(step_map)
        if verbose:
            dt_str   = f" | director={best_director:5d}" if best_director is not None else ""
            tr_str   = f" | expanded={best_transitions:6d}" if best_transitions is not None else ""
            path_str = " ".join(str(a) for _, a, *_ in best_data)
            print(f"[explore_all] Solved{dt_str}{tr_str} | steps={len(best_data)} | path=[{path_str}]", flush=True)
            print(f"[explore_all] Dataset: {len(X)} samples | Training to convergence ...", flush=True)

        loss = agent._train_until_converged(X, y, verbose=verbose, step_map=step_map, use_mse=True)

        if verbose:
            print(f"[explore_all] Final loss={loss:.2e}", flush=True)
            print(f"[explore_all] Network greedy replay:", flush=True)
        net_data, _, _, r_director, r_transitions = agent._run_episode_greedy(env, fsp_path)
        net_path = [a for _, a, *_ in net_data]
        best_path_actions = [a for _, a, *_ in best_data]
        match = net_path == best_path_actions
        if verbose:
            agent._print_best_path(net_data)
            if not match:
                print(f"  [Replay] path=[{' '.join(str(a) for a in net_path)}] (best=[{' '.join(str(a) for a in best_path_actions)}])", flush=True)
        else:
            dt_str = f"{r_director}" if r_director is not None else "?"
            tr_str = f"{r_transitions}" if r_transitions is not None else "?"
            print(f"  [Replay] expansions={tr_str} | decisions={len(net_data)} | director={dt_str} | match_best={match}", flush=True)

        agent.save_onnx(str(results_path / "path_best.onnx"))
        agent.save_model(str(results_path / "path_best.pt"))
        agent.trained = True
        return agent

    # ── round 0: random exploration ─────────────────────────────────────────
    if verbose:
        print(f"[Round   0] {agent.random_episodes} random episodes ...", flush=True)
    env.set_verbose(False)
    episodes = []
    for _ in range(agent.random_episodes):
        ep_data, ep_reward, realizable, director, transitions = agent._run_episode_random(env, fsp_path)
        episodes.append((ep_data, ep_reward, realizable, director, transitions))
        total_episodes += 1

    global_best = max(episodes, key=lambda x: x[1])
    best_data, best_reward, best_real, best_director, best_transitions = global_best

    if verbose:
        dt_str   = f" | director={best_director:5d}" if best_director is not None else ""
        tr_str   = f" | expanded={best_transitions:6d}" if best_transitions is not None else ""
        path_str = " ".join(str(a) for _, a, *_ in best_data)
        print(f"[Round   0] best_reward={best_reward:7.0f} | steps={len(best_data):4d}{dt_str}{tr_str} | path=[{path_str}]", flush=True)
        print(f"[Round   0] Building dataset ({sum(len(f) for f, _, *_ in best_data)} samples) "
              f"and training {agent.train_epochs} epochs ...", flush=True)

    step_map = {}
    for step_idx, (frontier_feats, chosen, *_) in enumerate(best_data):
        chosen_key = tuple(frontier_feats[chosen])
        seen_keys = set()
        for feat in frontier_feats:
            key = tuple(feat)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if key not in step_map:
                step_map[key] = ([], [])
            if key == chosen_key:
                step_map[key][0].append(step_idx)
            else:
                step_map[key][1].append(step_idx)
    X, y  = agent._build_ranked_dataset(step_map)
    loss  = agent._train_until_converged(X, y, verbose=verbose, step_map=step_map, use_mse=True)

    if verbose:
        print(f"[Round   0] Feature size={agent.feature_size} | train loss={loss:.6f}", flush=True)

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([0, "random", best_reward, f"{loss:.6f}", len(best_data),
                                 best_director, best_transitions, agent.epsilon, best_real])

    best_ever_reward = best_reward

    # ── rounds 1 .. max_rounds: guided ────────────────────────────────────────
    for round_idx in range(1, agent.max_rounds + 1):
        if total_episodes >= max_episodes:
            stop_reason = f"max episodes ({max_episodes})"
            break

        n_ep = min(agent.guided_episodes, max_episodes - total_episodes)
        episodes = []
        env.set_verbose(False)
        for _ in range(n_ep):
            ep_data, ep_reward, realizable, director, transitions = agent._run_episode_guided(env, fsp_path)
            episodes.append((ep_data, ep_reward, realizable, director, transitions))
            total_episodes += 1

        round_best = max(episodes, key=lambda x: x[1])
        if round_best[1] > global_best[1]:
            global_best = round_best
        best_data, best_reward, best_real, best_director, best_transitions = global_best

        step_map = {}
        for step_idx, (frontier_feats, chosen, *_) in enumerate(best_data):
            chosen_key = tuple(frontier_feats[chosen])
            seen_keys = set()
            for feat in frontier_feats:
                key = tuple(feat)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if key not in step_map:
                    step_map[key] = ([], [])
                if key == chosen_key:
                    step_map[key][0].append(step_idx)
                else:
                    step_map[key][1].append(step_idx)
        X, y  = agent._build_ranked_dataset(step_map)
        loss  = agent._train_until_converged(X, y, verbose=verbose, step_map=step_map, use_mse=True)

        if verbose:
            imp      = "↑" if round_best[1] > best_ever_reward else " "
            dt_str   = f" | director={best_director:5d}" if best_director is not None else ""
            tr_str   = f" | expanded={best_transitions:6d}" if best_transitions is not None else ""
            path_str = " ".join(str(a) for _, a, *_ in best_data)
            print(
                f"[Round {round_idx:3d}] best_reward={best_reward:7.0f}{imp} | "
                f"steps={len(best_data):4d}{dt_str}{tr_str} | "
                f"ε={agent.epsilon:.4f} | loss={loss:.6f} | "
                f"ep={total_episodes} | patience={no_improve}/{patience} | "
                f"path=[{path_str}]",
                flush=True,
            )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [round_idx, "guided", best_reward, f"{loss:.6f}", len(best_data),
                 best_director, best_transitions, f"{agent.epsilon:.4f}", best_real]
            )

        if round_idx % agent.save_frequency == 0:
            agent.save_onnx(str(results_path / f"path_r{round_idx:04d}.onnx"))

        if round_best[1] > best_ever_reward:
            best_ever_reward = round_best[1]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            stop_reason = f"patience ({patience} rounds without improvement)"
            break

    if verbose:
        print(f"\n[Final replay] Network greedy:", flush=True)
    net_data, _, _, r_director, r_transitions = agent._run_episode_greedy(env, fsp_path)
    net_path = [a for _, a, *_ in net_data]
    best_path_actions = [a for _, a, *_ in best_data]
    match = net_path == best_path_actions
    if verbose:
        agent._print_best_path(net_data)
        if not match:
            print(f"  [Final replay] path=[{' '.join(str(a) for a in net_path)}] (best=[{' '.join(str(a) for a in best_path_actions)}])", flush=True)
    else:
        dt_str = f"{r_director}" if r_director is not None else "?"
        tr_str = f"{r_transitions}" if r_transitions is not None else "?"
        print(f"  [Final replay] expansions={tr_str} | decisions={len(net_data)} | director={dt_str} | match_best={match}", flush=True)

    print(
        f"\n[STOP] PathLearning | {stop_reason}"
        f"  —  total episodes: {total_episodes}"
        f"  —  best reward: {best_ever_reward}"
        f"  —  end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if agent._initialized:
        agent.save_onnx(str(results_path / "path_final.onnx"))

    agent.trained = True
    return agent
