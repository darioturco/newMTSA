"""
DQN agent for the DCS RL environment.

Architecture note
-----------------
The action space is variable (frontier size changes at every step). To handle
this, the Q-network maps a single transition's feature vector to a scalar Q-value.
Action selection = argmax Q over all frontier elements.  This makes the network
architecture independent of frontier size.

Q(s, a) ≈ network(feature_vector_of_a)

Experience stored in the replay buffer:
  (feat_chosen, reward, next_feats_list, done)
"""

import csv
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── network ────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, feature_size)  →  output: (batch,)
        return self.net(x).squeeze(-1)


# ── replay buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, feat_chosen: list, reward: float,
             next_feats: list, done: bool) -> None:
        self._buf.append((
            np.asarray(feat_chosen, dtype=np.float32),
            float(reward),
            [np.asarray(f, dtype=np.float32) for f in next_feats],
            bool(done),
        ))

    def sample(self, batch_size: int) -> list:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ── agent ──────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Parameters
    ----------
    hidden_size        : neurons per hidden layer
    lr                 : Adam learning rate
    gamma              : discount factor
    buffer_size        : replay buffer capacity
    batch_size         : minibatch size for each gradient step
    epsilon_start      : initial exploration probability
    epsilon_end        : minimum exploration probability
    epsilon_decay      : multiplicative decay applied after each episode
    target_update_freq : copy q_net → target_net every N environment steps
    min_replay_size    : start training only after this many transitions stored
    """

    def __init__(
        self,
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        target_update_freq: int = 200,
        min_replay_size: int = 500,
    ):
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.q_net: Optional[QNetwork] = None
        self.target_net: Optional[QNetwork] = None
        self.optimizer = None
        self.feature_size: Optional[int] = None
        self.total_steps: int = 0

    # ── lazy network init ────────────────────────────────────────────────────

    def _init_networks(self, feature_size: int) -> None:
        if self.q_net is not None:
            return
        self.feature_size = feature_size
        self.q_net = QNetwork(feature_size, self.hidden_size)
        self.target_net = QNetwork(feature_size, self.hidden_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    # ── action selection ─────────────────────────────────────────────────────

    def select_action(self, frontier_feats: list) -> int:
        """ε-greedy: random with prob ε, else argmax Q over frontier elements."""
        if random.random() < self.epsilon:
            return random.randrange(len(frontier_feats))
        with torch.no_grad():
            t = torch.tensor(np.array(frontier_feats, dtype=np.float32))
            return int(self.q_net(t).argmax())

    # ── one environment step ─────────────────────────────────────────────────

    def observe(self, feat_chosen: list, reward: float,
                next_feats: list, done: bool) -> Optional[float]:
        """Store transition and trigger one gradient step. Returns loss or None."""
        self.replay_buffer.push(feat_chosen, reward, next_feats, done)
        self.total_steps += 1

        loss = self._train_step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss

    # ── training ─────────────────────────────────────────────────────────────

    def _train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        feats, rewards, next_feats_list, dones = zip(*batch)

        feats_t   = torch.tensor(np.stack(feats))                        # (B, F)
        rewards_t = torch.tensor(np.array(rewards, dtype=np.float32))    # (B,)
        dones_t   = torch.tensor(np.array(dones,   dtype=np.float32))    # (B,)

        current_q = self.q_net(feats_t)   # (B,)

        with torch.no_grad():
            next_q = torch.zeros(self.batch_size)
            for i, (nf, d) in enumerate(zip(next_feats_list, dones_t)):
                if not d and nf:
                    nf_t = torch.tensor(np.stack(nf))
                    next_q[i] = self.target_net(nf_t).max()

        targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── persistence ──────────────────────────────────────────────────────────

    def save_onnx(self, path: str) -> None:
        """Export the Q-network to ONNX. Input: (batch, feature_size) → (batch,)."""
        dummy = torch.zeros(1, self.feature_size)
        torch.onnx.export(
            self.q_net,
            dummy,
            path,
            input_names=["features"],
            output_names=["q_value"],
            dynamic_axes={"features": {0: "batch_size"}},
            opset_version=11,
        )


# ── training loop ──────────────────────────────────────────────────────────────

def train(
    env,
    agent: DQNAgent,
    fsp_path: str,
    max_episodes: int = 1000,
    max_steps: int = 100_000,
    patience: int = 100,
    results_dir: str = "results",
    verbose: bool = True,
) -> DQNAgent:
    """
    Train *agent* on *env* using the given FSP instance.

    Stop criteria (first to trigger wins):
      • max_episodes episodes completed
      • max_steps total environment steps reached
      • patience episodes without improvement in the 10-episode average reward

    Every 10 episodes the Q-network is exported to an ONNX file in *results_dir*.
    Episode rewards are appended to ``<results_dir>/dqn_training.csv``.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "dqn_training.csv"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_reward", "steps", "realizable"])

    best_avg      = float("-inf")
    no_improve    = 0
    recent        = deque(maxlen=10)
    global_steps  = 0
    stop_reason   = f"max episodes ({max_episodes})"

    for ep in range(1, max_episodes + 1):
        frontier = env.reset(fsp_path)

        # Lazy network init on first episode that has a non-empty frontier
        if frontier and agent.q_net is None:
            agent._init_networks(len(frontier[0]))

        ep_reward = 0
        ep_steps  = 0
        ep_info   = {}

        while not env._done:
            if not frontier:
                break

            action      = agent.select_action(frontier)
            feat_chosen = frontier[action]

            frontier, reward, done, ep_info = env.step(action)
            next_feats = frontier if (not done and frontier) else []

            if agent.q_net is not None:
                agent.observe(feat_chosen, reward, next_feats, done)

            ep_reward    += reward
            ep_steps     += 1
            global_steps += 1

            if global_steps >= max_steps:
                stop_reason = f"max steps ({max_steps:,})"
                break

        agent.decay_epsilon()

        # ── logging ──────────────────────────────────────────────────────────
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, ep_reward, ep_steps, ep_info.get("realizable")]
            )

        # ── ONNX checkpoint ───────────────────────────────────────────────────
        if ep % 10 == 0 and agent.q_net is not None:
            onnx_path = results_path / f"dqn_ep{ep:04d}.onnx"
            agent.save_onnx(str(onnx_path))

        # ── patience ──────────────────────────────────────────────────────────
        recent.append(ep_reward)
        avg = sum(recent) / len(recent)
        if avg > best_avg:
            best_avg   = avg
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            print(
                f"Ep {ep:4d} | reward={ep_reward:6d} | steps={ep_steps:4d} | "
                f"ε={agent.epsilon:.3f} | avg10={avg:7.1f} | "
                f"patience={no_improve}/{patience}"
            )

        if no_improve >= patience:
            stop_reason = f"patience ({patience} episodes without improvement)"
            break

        if global_steps >= max_steps:
            break

    print(f"\n[STOP] {stop_reason}  —  total steps: {global_steps:,}")
    return agent
