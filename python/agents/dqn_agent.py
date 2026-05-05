"""
DQN agent for the DCS RL environment.

Architecture note
-----------------
Three network modes are available (selected by use_lstm / use_transformer):

Flat mode (use_lstm=False, use_transformer=False):
  Q(s, a) ≈ MLP(feature_vector_of_a)
  Action selection = argmax Q over all frontier elements.

LSTM mode (use_lstm=True):
  At each step the LSTM consumes the previously chosen transition's
  feature vector, producing a hidden state h_t that summarises history.
  Q(s, a) ≈ MLP(concat(h_t, feature_vector_of_a))
  Buffer stores full episodes; training uses BPTT over sampled windows.

Transformer mode (use_transformer=True):
  Same as LSTM mode but uses a Transformer encoder instead.
  The 'hidden state' is a growing list of previous feature vectors (KV cache).
  Q(s, a) ≈ MLP(concat(h_t, feature_vector_of_a))

Replay buffers
--------------
Flat       : ReplayBuffer        — stores (feat_chosen, reward, next_feats, done)
Sequential : EpisodeReplayBuffer — stores full episodes; samples seq_len windows
"""

import csv
import re
import random
from collections import deque
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent
from .networks import (
    MLPScorer, LSTMNet, TransformerNet,
    OnnxLSTMWrapper, OnnxTransformerWrapper,
    ReplayBuffer, EpisodeReplayBuffer,
)


class DQNAgent(BaseAgent):
    """
    Parameters
    ----------
    use_lstm              : use LSTMNet + EpisodeReplayBuffer
    use_transformer       : use TransformerNet + EpisodeReplayBuffer (overrides use_lstm)
    lstm_hidden           : LSTM hidden size
    seq_len               : BPTT window length
    min_episodes          : warmup episodes before sequential training starts
    episode_capacity      : max episodes in EpisodeReplayBuffer
    hidden_size           : MLP head hidden size
    lr                    : Adam learning rate
    gamma                 : discount factor
    buffer_size           : ReplayBuffer capacity (flat mode only)
    batch_size            : transitions (flat) or sequences (sequential) per step
    epsilon_start/end/decay : ε-greedy schedule
    target_update_freq    : update target net every N steps
    tau                   : Polyak averaging coefficient (1.0 = hard copy)
    min_replay_size       : warmup transitions (flat mode only)
    weight_decay          : L2 regularisation in Adam optimiser
    double_dqn            : use Double DQN for target computation
    d_model               : Transformer embedding dimension
    nhead                 : number of Transformer attention heads
    num_transformer_layers: number of Transformer encoder layers
    save_frequency        : save checkpoint every N episodes
    """

    def __init__(
        self,
        use_lstm: bool = False,
        use_transformer: bool = False,
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
        epsilon_decay: Union[float, str] = "auto",
        target_update_freq: int = 200,
        tau: float = 1.0,
        min_replay_size: int = 500,
        weight_decay: float = 0.0,
        double_dqn: bool = False,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        save_frequency: int = 10,
        epsilon_decay_episodes: int = 250,
    ):
        super().__init__(epsilon_start, epsilon_end, epsilon_decay, save_frequency)
        self._auto_decay: bool = isinstance(epsilon_decay, str) and epsilon_decay.lower() in ("auto", "automatic")
        self._epsilon_decay_config = epsilon_decay
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.use_lstm              = use_lstm
        self.use_transformer       = use_transformer
        self.lstm_hidden           = lstm_hidden
        self.seq_len               = seq_len
        self.min_episodes          = min_episodes
        self.hidden_size           = hidden_size
        self.lr                    = lr
        self.gamma                 = gamma
        self.batch_size            = batch_size
        self.target_update_freq    = target_update_freq
        self.tau                   = tau
        self.min_replay_size       = min_replay_size
        self.weight_decay          = weight_decay
        self.double_dqn            = double_dqn
        self.d_model               = d_model
        self.nhead                 = nhead
        self.num_transformer_layers = num_transformer_layers

        self._use_sequential: bool = use_transformer or use_lstm

        self.replay_buffer  = ReplayBuffer(buffer_size)
        self.episode_buffer = (
            EpisodeReplayBuffer(episode_capacity, seq_len)
            if self._use_sequential else None
        )

        self.q_net:      Optional[nn.Module] = None
        self.target_net: Optional[nn.Module] = None
        self.optimizer   = None
        self.feature_size: Optional[int] = None
        self._initialized: bool = False
        self.total_steps:  int  = 0
        self.last_loss:    Optional[float] = None

        # per-episode sequential state (LSTM tuple or Transformer cache list)
        self._prev_chosen: Optional[np.ndarray] = None
        self._hidden = None

    # ── lazy network init ────────────────────────────────────────────────────

    def _init_networks(self, feature_size: int) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.feature_size = feature_size
        if self.use_transformer:
            self.q_net = TransformerNet(
                feature_size, self.d_model, self.nhead,
                self.num_transformer_layers, self.hidden_size,
            )
            self.target_net = TransformerNet(
                feature_size, self.d_model, self.nhead,
                self.num_transformer_layers, self.hidden_size,
            )
        elif self.use_lstm:
            self.q_net      = LSTMNet(feature_size, self.lstm_hidden, self.hidden_size)
            self.target_net = LSTMNet(feature_size, self.lstm_hidden, self.hidden_size)
        else:
            self.q_net      = MLPScorer(feature_size, self.hidden_size)
            self.target_net = MLPScorer(feature_size, self.hidden_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def reset_episode(self) -> None:
        """Reset per-episode sequential state. Call at the start of every episode."""
        if self.feature_size is not None:
            self._prev_chosen = np.zeros(self.feature_size, dtype=np.float32)
        self._hidden = None

    # ── target network update ────────────────────────────────────────────────

    def _update_target(self) -> None:
        """Polyak average: tau=1.0 is a hard copy, tau<1.0 is a soft update."""
        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ── action selection ─────────────────────────────────────────────────────

    def select_action(self, frontier_feats: list) -> int:
        """ε-greedy over frontier elements."""
        if random.random() < self.epsilon:
            return random.randrange(len(frontier_feats))
        with torch.no_grad():
            if self._use_sequential:
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
        """Store transition and trigger one gradient step."""
        self.total_steps += 1
        feat_chosen = np.asarray(frontier_feats[action], dtype=np.float32)

        if self._use_sequential:
            self.episode_buffer.push_step(
                self._prev_chosen, frontier_feats, action, reward, done
            )
            self._prev_chosen = feat_chosen
            loss = self._train_step_sequential()
        else:
            self.replay_buffer.push(feat_chosen, reward, next_feats, done)
            loss = self._train_step_flat()

        if self.total_steps % self.target_update_freq == 0:
            self._update_target()

        if loss is not None:
            self.last_loss = loss
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
                    if self.double_dqn:
                        best_a = self.q_net(nf_t).argmax()
                        next_q[i] = self.target_net(nf_t)[best_a]
                    else:
                        next_q[i] = self.target_net(nf_t).max()

        targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)
        loss = F.mse_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    # ── training: sequential (LSTM & Transformer share this path) ─────────────

    def _train_step_sequential(self) -> Optional[float]:
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
                            if self.double_dqn:
                                best_a = self.q_net.score(all_h[t + 1], nft).argmax()
                                tgt = reward + self.gamma * self.target_net.score(
                                    all_h_tgt[t + 1], nft
                                )[best_a].item()
                            else:
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

    # ── episode buffer warmup ─────────────────────────────────────────────────

    def _fill_episode_buffer(self, env, fsp_path: str) -> None:
        """Pre-fill episode buffer with random episodes before training starts."""
        n = self.min_episodes
        log_every = max(1, n // 5)
        print(f"[Warmup] Collecting {n} random episodes...", flush=True)
        for i in range(n):
            frontier = env.reset(fsp_path)
            if frontier and not self._initialized:
                self._init_networks(len(frontier[0]))
            self.reset_episode()
            while not env.is_finished:
                if not frontier:
                    break
                action        = random.randrange(len(frontier))
                prev_frontier = frontier
                frontier, reward, done, _ = env.step(action)
                self.episode_buffer.push_step(
                    self._prev_chosen, prev_frontier, action, reward, done
                )
                self._prev_chosen = np.asarray(prev_frontier[action], dtype=np.float32)
            if (i + 1) % log_every == 0 or i + 1 == n:
                print(f"[Warmup] {i + 1}/{n} episodes", flush=True)
        print(f"[Warmup] Done — buffer has {len(self.episode_buffer)} episodes.", flush=True)

    def _estimate_avg_episode_length(self, env, fsp_path: str, n: int = 20) -> float:
        """Run n random episodes and return average step count."""
        total = 0
        for _ in range(n):
            frontier = env.reset(fsp_path)
            steps = 0
            while not env.is_finished:
                if not frontier:
                    break
                frontier, _, _, _ = env.step(random.randrange(len(frontier)))
                steps += 1
            total += steps
        return total / n if n > 0 else 1.0

    # ── persistence ──────────────────────────────────────────────────────────

    def save_onnx(self, path: str) -> None:
        F = self.feature_size
        self.q_net.eval()
        if self.use_transformer:
            wrapper = OnnxTransformerWrapper(self.q_net)
            torch.onnx.export(
                wrapper,
                (torch.zeros(1, 1, F), torch.zeros(3, F)),
                path,
                input_names=["sequence", "candidates"],
                output_names=["scores"],
                dynamic_axes={"sequence": {1: "T"}, "candidates": {0: "N"}, "scores": {0: "N"}},
                opset_version=17, export_params=True, dynamo=False,
            )
        elif self.use_lstm:
            H = self.lstm_hidden
            wrapper = OnnxLSTMWrapper(self.q_net)
            torch.onnx.export(
                wrapper,
                (torch.zeros(1, 1, F), torch.zeros(1, 1, H), torch.zeros(1, 1, H), torch.zeros(3, F)),
                path,
                input_names=["prev_feat", "h_in", "c_in", "candidates"],
                output_names=["scores", "h_out", "c_out"],
                dynamic_axes={"candidates": {0: "N"}, "scores": {0: "N"}},
                opset_version=17, export_params=True, dynamo=False,
            )
        else:
            torch.onnx.export(
                self.q_net, torch.zeros(1, F), path,
                input_names=["features"], output_names=["q_value"],
                dynamic_axes={"features": {0: "N"}, "q_value": {0: "N"}},
                opset_version=17, export_params=True, dynamo=False,
            )

    def save_model(self, path: str) -> None:
        if not self._initialized:
            print("[WARN] Agent not initialized; nothing to save.")
            return
        torch.save({
            "feature_size": self.feature_size,
            "config": self._config_dict(),
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    @classmethod
    def load_model(cls, path: str) -> "DQNAgent":
        payload = torch.load(path, map_location="cpu")
        agent = cls(**payload["config"])
        agent._init_networks(payload["feature_size"])
        agent.q_net.load_state_dict(payload["q_net"])
        agent.target_net.load_state_dict(payload["target_net"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        return agent

    def _config_dict(self) -> dict:
        return {
            "use_lstm": self.use_lstm,
            "use_transformer": self.use_transformer,
            "lstm_hidden": self.lstm_hidden,
            "seq_len": self.seq_len,
            "min_episodes": self.min_episodes,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "epsilon_start": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self._epsilon_decay_config,
            "target_update_freq": self.target_update_freq,
            "tau": self.tau,
            "weight_decay": self.weight_decay,
            "double_dqn": self.double_dqn,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_transformer_layers": self.num_transformer_layers,
            "save_frequency": self.save_frequency,
            "epsilon_decay_episodes": self.epsilon_decay_episodes,
        }


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
    env.reset(fsp_path)
    print(f"Instance type: {'non-blocking' if env.is_nonblocking else 'blocking'}")
    if env.uses_features() and env.frontier:
        print(f"Feature size  : {len(env.frontier[0])}")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "dqn_training.csv"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_reward", "steps", "realizable"])

    best_ep_reward = float("-inf")
    best_avg     = float("-inf")
    best_loss    = float("inf")
    no_improve   = 0
    recent: deque = deque(maxlen=50)
    global_steps = 0
    stop_reason  = f"max episodes ({max_episodes})"

    if agent._use_sequential:
        agent._fill_episode_buffer(env, fsp_path)

    if agent._auto_decay:
        if agent._use_sequential:
            avg_ep_len = agent.episode_buffer.avg_episode_length()
        else:
            avg_ep_len = agent._estimate_avg_episode_length(env, fsp_path)
        target_steps = agent.epsilon_decay_episodes * avg_ep_len
        agent.epsilon_decay = (agent.epsilon_end / agent.epsilon) ** (1.0 / target_steps)
        print(f"[Auto decay] avg_ep_len={avg_ep_len:.1f}  decay={agent.epsilon_decay:.7f}  "
              f"(ε {agent.epsilon:.2f}→{agent.epsilon_end} in ~{agent.epsilon_decay_episodes} episodes)", flush=True)

    for ep in range(1, max_episodes + 1):
        frontier = env.reset(fsp_path)

        if frontier and not agent._initialized:
            agent._init_networks(len(frontier[0]))

        agent.reset_episode()

        ep_reward = 0
        ep_steps  = 0
        ep_info   = {}

        while not env.is_finished:
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
            agent.decay_epsilon()

            if global_steps >= max_steps:
                stop_reason = f"max steps ({max_steps:,})"
                break

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, ep_reward, ep_steps, ep_info.get("realizable")]
            )

        if ep % agent.save_frequency == 0 and agent._initialized:
            agent.save_onnx(str(results_path / f"dqn_ep{ep:04d}.onnx"))
            if agent._use_sequential:
                agent.save_model(str(results_path / f"dqn_ep{ep:04d}.pt"))

        recent.append(ep_reward)
        avg = sum(recent) / len(recent)
        new_loss_min = agent.last_loss is not None and agent.last_loss < best_loss
        if new_loss_min:
            best_loss = agent.last_loss
        if (avg > best_avg) or (ep_reward > best_ep_reward) or new_loss_min:
            best_ep_reward = ep_reward
            best_avg   = avg
            no_improve = 0
        elif agent.epsilon <= agent.epsilon_end:
            no_improve += 1

        if verbose:
            loss_str = f"{agent.last_loss:.5f}" if agent.last_loss is not None else "N/A   "
            print(
                f"Ep {ep:4d} | reward={ep_reward:6d} | steps={ep_steps:4d} | "
                f"ε={agent.epsilon:.5f} | avg50={avg:7.1f} | "
                f"loss={loss_str} | patience={no_improve}/{patience}"
            )

        if no_improve >= patience:
            stop_reason = f"patience ({patience} episodes without improvement)"
            break
        if global_steps >= max_steps:
            break

    print(
        f"\n[STOP] DQN | {stop_reason}  —  total steps: {global_steps:,}  —  total episodes: {ep}"
        f"  —  best reward: {best_ep_reward}"
    )
    agent.trained = True
    return agent
