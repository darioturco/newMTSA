"""
DQN agent for the DCS RL environment.

Architecture note
-----------------
Two network modes are available (selected by use_lstm):

Flat mode (use_lstm=False):
  Q(s, a) ≈ MLP(feature_vector_of_a)
  Action selection = argmax Q over all frontier elements.

LSTM mode (use_lstm=True):
  At each step the LSTM consumes the previously chosen transition's
  feature vector, producing a hidden state h_t that summarises history.
  Q(s, a) ≈ MLP(concat(h_t, feature_vector_of_a))
  Buffer stores full episodes; training uses BPTT over sampled windows.

Replay buffers
--------------
Flat  : ReplayBuffer      — stores (feat_chosen, reward, next_feats, done)
LSTM  : EpisodeReplayBuffer — stores full episodes; samples seq_len windows
"""

import csv
import random
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── flat network ───────────────────────────────────────────────────────────────

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
        # x: (batch, feature_size) → (batch,)
        return self.net(x).squeeze(-1)


# ── LSTM network ───────────────────────────────────────────────────────────────

class LSTMQNetwork(nn.Module):
    """
    LSTM history encoder + MLP scoring head.

    forward_sequence — training: process a full window, return all hidden states.
    forward_step     — inference: process one step, update running hidden state.
    score            — shared head: given h_t and candidates → Q-values.
    """

    def __init__(self, feature_size: int, lstm_hidden: int, mlp_hidden: int):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(feature_size, lstm_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + feature_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward_sequence(self, prev_feats: torch.Tensor, hidden=None):
        """prev_feats: (1, T, F) → all_h: (T, lstm_hidden), new hidden"""
        all_h, state = self.lstm(prev_feats, hidden)
        return all_h.squeeze(0), state

    def forward_step(self, prev_feat: torch.Tensor, hidden=None):
        """prev_feat: (F,) → h: (lstm_hidden,), new hidden"""
        x = prev_feat.unsqueeze(0).unsqueeze(0)  # (1, 1, F)
        out, state = self.lstm(x, hidden)
        return out.squeeze(), state

    def score(self, h: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """h: (lstm_hidden,), candidates: (N, F) → (N,)"""
        h_exp = h.unsqueeze(0).expand(candidates.size(0), -1)
        return self.head(torch.cat([h_exp, candidates], dim=-1)).squeeze(-1)


# ── replay buffers ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, feat_chosen: np.ndarray, reward: float,
             next_feats: list, done: bool) -> None:
        self._buf.append((
            feat_chosen,
            float(reward),
            [np.asarray(f, dtype=np.float32) for f in next_feats],
            bool(done),
        ))

    def sample(self, batch_size: int) -> list:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


class EpisodeReplayBuffer:
    """
    Stores complete episodes; samples contiguous windows of length seq_len.

    Each step: (prev_chosen_feat, frontier_feats, action, reward, done)
      prev_chosen_feat — feature of the transition expanded at the previous step
                         (zero vector at episode start)
      frontier_feats   — full set of candidate features at this step
    """

    def __init__(self, capacity: int, seq_len: int):
        self._episodes: deque = deque(maxlen=capacity)
        self._current: list = []
        self.seq_len = seq_len

    def push_step(self, prev_chosen: np.ndarray, frontier: list,
                  action: int, reward: float, done: bool) -> None:
        self._current.append((
            prev_chosen.copy(),
            [np.asarray(f, dtype=np.float32) for f in frontier],
            int(action),
            float(reward),
            bool(done),
        ))
        if done:
            if self._current:
                self._episodes.append(self._current)
            self._current = []

    def sample(self, n_seqs: int) -> List[list]:
        """Return n_seqs windows of length seq_len (or full episode if shorter)."""
        if not self._episodes:
            return []
        chosen = random.choices(self._episodes, k=n_seqs)
        windows = []
        for ep in chosen:
            T = len(ep)
            if T <= self.seq_len:
                windows.append(ep)
            else:
                start = random.randrange(0, T - self.seq_len + 1)
                windows.append(ep[start : start + self.seq_len])
        return windows

    def __len__(self) -> int:
        return len(self._episodes)


# ── agent ──────────────────────────────────────────────────────────────────────

# TODO: Make a base Agent class and move common code (e.g. epsilon decay, ONNX export) there, since PPO and SAC agents also have those features. DQNAgent should then inherit from that base class and only implement the DQN-specific parts.
#   Also separate the ReplayBuffer in a separate file, since it's also used by the PPO and SAC agents (although they use a different variant of it, but still some code could be shared). The same applies to the LSTMQNetwork, since it's also used by the SAC and PPO agent.
#   All the agent files must be in a separate folder called "agents".

class DQNAgent:
    """
    Parameters
    ----------
    use_lstm           : use LSTMQNetwork + EpisodeReplayBuffer
    lstm_hidden        : LSTM hidden size (LSTM mode only)
    seq_len            : BPTT window length for episode buffer
    min_episodes       : warmup — train only after this many episodes stored
    episode_capacity   : max episodes in EpisodeReplayBuffer
    hidden_size        : MLP hidden size (both modes)
    lr                 : Adam learning rate
    gamma              : discount factor
    buffer_size        : ReplayBuffer capacity (flat mode only)
    batch_size         : transitions per gradient step (flat) or sequences (LSTM)
    epsilon_start/end/decay : ε-greedy schedule
    target_update_freq : hard-copy q_net → target_net every N steps
    min_replay_size    : warmup transitions (flat mode only)
    """

    def __init__(
        self,
        use_lstm: bool = False,
        lstm_hidden: int = 64,
        seq_len: int = 16,
        min_episodes: int = 50,
        episode_capacity: int = 1000,
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
        self.use_lstm = use_lstm
        self.lstm_hidden = lstm_hidden
        self.seq_len = seq_len
        self.min_episodes = min_episodes
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
        self.episode_buffer = EpisodeReplayBuffer(episode_capacity, seq_len) if use_lstm else None

        self.q_net: Optional[nn.Module] = None
        self.target_net: Optional[nn.Module] = None
        self.optimizer = None
        self.feature_size: Optional[int] = None
        self._initialized: bool = False
        self.total_steps: int = 0

        # LSTM per-episode state
        self._prev_chosen: Optional[np.ndarray] = None
        self._hidden = None

    # ── lazy network init ────────────────────────────────────────────────────

    def _init_networks(self, feature_size: int) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.feature_size = feature_size
        if self.use_lstm:
            self.q_net     = LSTMQNetwork(feature_size, self.lstm_hidden, self.hidden_size)
            self.target_net = LSTMQNetwork(feature_size, self.lstm_hidden, self.hidden_size)
        else:
            self.q_net     = QNetwork(feature_size, self.hidden_size)
            self.target_net = QNetwork(feature_size, self.hidden_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def reset_episode(self) -> None:
        """Reset per-episode LSTM state. Must be called at the start of every episode."""
        if self.feature_size is not None:
            self._prev_chosen = np.zeros(self.feature_size, dtype=np.float32)
        self._hidden = None

    # ── action selection ─────────────────────────────────────────────────────

    def select_action(self, frontier_feats: list) -> int:
        """ε-greedy over frontier elements."""
        if random.random() < self.epsilon:
            return random.randrange(len(frontier_feats))
        with torch.no_grad():
            if self.use_lstm:
                prev_t = torch.tensor(self._prev_chosen)
                h, self._hidden = self.q_net.forward_step(prev_t, self._hidden)
                cands = torch.tensor(np.array(frontier_feats, dtype=np.float32))
                return int(self.q_net.score(h, cands).argmax())
            else:
                t = torch.tensor(np.array(frontier_feats, dtype=np.float32))
                return int(self.q_net(t).argmax())

    # ── observe ──────────────────────────────────────────────────────────────

    def observe(
        self,
        frontier_feats: list,
        action: int,
        reward: float,
        next_feats: list,
        done: bool,
    ) -> Optional[float]:
        """
        Store transition and trigger one gradient step.

        frontier_feats : full frontier before the step (list of feature vectors)
        action         : chosen index into frontier_feats
        next_feats     : frontier after the step (used by flat mode only)
        """
        self.total_steps += 1
        feat_chosen = np.asarray(frontier_feats[action], dtype=np.float32)

        if self.use_lstm:
            self.episode_buffer.push_step(
                self._prev_chosen, frontier_feats, action, reward, done
            )
            self._prev_chosen = feat_chosen
            loss = self._train_step_lstm()
        else:
            self.replay_buffer.push(feat_chosen, reward, next_feats, done)
            loss = self._train_step_flat()

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss

    # ── training: flat ───────────────────────────────────────────────────────

    def _train_step_flat(self) -> Optional[float]:
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        feats, rewards, next_feats_list, dones = zip(*batch)

        feats_t   = torch.tensor(np.stack(feats))
        rewards_t = torch.tensor(np.array(rewards, dtype=np.float32))
        dones_t   = torch.tensor(np.array(dones,   dtype=np.float32))

        current_q = self.q_net(feats_t)

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

    # ── training: LSTM ───────────────────────────────────────────────────────

    def _train_step_lstm(self) -> Optional[float]:
        if len(self.episode_buffer) < self.min_episodes:
            return None

        windows = self.episode_buffer.sample(self.batch_size)
        if not windows:
            return None

        all_current_q: list = []
        all_target_q:  list = []

        for window in windows:
            T = len(window)
            prev_feats_t = torch.tensor(
                np.stack([step[0] for step in window]), dtype=torch.float32
            ).unsqueeze(0)  # (1, T, F)

            all_h, _          = self.q_net.forward_sequence(prev_feats_t)
            with torch.no_grad():
                all_h_tgt, _  = self.target_net.forward_sequence(prev_feats_t)

            for t, (_, frontier, action, reward, done) in enumerate(window):
                ft = torch.tensor(np.stack(frontier), dtype=torch.float32)
                all_current_q.append(self.q_net.score(all_h[t], ft)[action])

                with torch.no_grad():
                    if done or t == T - 1:
                        tgt = float(reward)
                    else:
                        nf = window[t + 1][1]
                        if nf:
                            nft = torch.tensor(np.stack(nf), dtype=torch.float32)
                            tgt = reward + self.gamma * self.target_net.score(
                                all_h_tgt[t + 1], nft
                            ).max().item()
                        else:
                            tgt = float(reward)
                    all_target_q.append(tgt)

        if not all_current_q:
            return None

        current_q_t = torch.stack(all_current_q)
        target_q_t  = torch.tensor(all_target_q, dtype=torch.float32)
        loss = F.mse_loss(current_q_t, target_q_t)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── persistence ──────────────────────────────────────────────────────────

    def save_onnx(self, path: str) -> None:
        if self.use_lstm:
            # TODO: implment the LTSM save, that is important to test the agent in other instances size.
            print("[WARN] ONNX export not supported for LSTM mode.")
            return
        
        self.q_net.eval()
        dummy = torch.zeros(1, self.feature_size)
        torch.onnx.export(
            self.q_net, dummy, path,
            input_names=["features"], output_names=["q_value"],
            dynamic_axes={"features": {0: "batch_size"}},
            opset_version=17,
            export_params=True,
            dynamo=False
        )


# ── training loop ──────────────────────────────────────────────────────────────
# TODO: make this function part of the Agent class. 
#   Also implment a trained agent flag, that initialy is false and before the end of training is set to true.
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
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "dqn_training.csv"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_reward", "steps", "realizable"])

    best_ep_reward = float("-inf")
    best_avg     = float("-inf")
    no_improve   = 0
    recent: deque = deque(maxlen=10)
    global_steps = 0
    stop_reason  = f"max episodes ({max_episodes})"

    for ep in range(1, max_episodes + 1):
        frontier = env.reset(fsp_path)

        if frontier and not agent._initialized:
            agent._init_networks(len(frontier[0]))

        agent.reset_episode()

        ep_reward = 0
        ep_steps  = 0
        ep_info   = {}

        while not env._done: # TODO: cambia el nombre _done, pones is_finish, que no sea un atributo privado.
            if not frontier:
                break

            prev_frontier = frontier
            action        = agent.select_action(frontier)

            frontier, reward, done, ep_info = env.step(action)
            next_feats = frontier if (not done and frontier) else []

            if agent._initialized:
                agent.observe(prev_frontier, action, reward, next_feats, done)

            ep_reward    += reward
            ep_steps     += 1
            global_steps += 1

            if global_steps >= max_steps:
                stop_reason = f"max steps ({max_steps:,})"
                break

        agent.decay_epsilon()

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, ep_reward, ep_steps, ep_info.get("realizable")]
            )

        if ep % 10 == 0 and agent._initialized: # TODO: make save frequency configurable, not a hardcored 10 (in all agents)
            onnx_path = results_path / f"dqn_ep{ep:04d}.onnx"
            agent.save_onnx(str(onnx_path))

        recent.append(ep_reward)
        avg = sum(recent) / len(recent)
        if (avg > best_avg) or (ep_reward < best_ep_reward):
            best_ep_reward = ep_reward
            best_avg   = avg
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            print(
                f"Ep {ep:4d} | reward={ep_reward:6d} | steps={ep_steps:4d} | "
                f"ε={agent.epsilon:.5f} | avg10={avg:7.1f} | "
                f"patience={no_improve}/{patience}" # TODO: after the pacience add the loss value of the network in that episode, that is important to understand if the agent is still learning or not.
            )

        if no_improve >= patience:
            stop_reason = f"patience ({patience} episodes without improvement)"
            break
        if global_steps >= max_steps:
            break

    print(f"\n[STOP] {stop_reason}  —  total steps: {global_steps:,}") # TODO: all the total of episodes
    return agent
