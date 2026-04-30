"""
SAC (Soft Actor-Critic) agent for the DCS RL environment.

Architecture note
-----------------
Two network modes (selected by use_lstm):

Flat mode (use_lstm=False):
  Actor and Q-networks are MLPs operating on individual feature vectors.
  π(a|s) = softmax_a [ actor(feature_a) ]
  Q(s,a) = q_net(feature_a)

LSTM mode (use_lstm=True):
  Each network (actor, q1, q2 and their targets) has its own LSTM that
  consumes the previously chosen transition's feature vector, producing
  h_t that encodes history.
  score(a) = head(concat(h_t, feature_a))
  During inference the actor LSTM hidden state is maintained across steps.
  During training all networks replay episode windows from scratch (h_0=0).

Replay buffers
--------------
Flat : ReplayBuffer       — (curr_feats, action, reward, next_feats, done)
LSTM : EpisodeReplayBuffer — full episodes; sampled as seq_len windows
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


# ── flat networks ──────────────────────────────────────────────────────────────

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
        return self.net(x).squeeze(-1)


class ActorNetwork(nn.Module):
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
        return self.net(x).squeeze(-1)


# ── LSTM network (shared structure for actor and Q-nets) ──────────────────────

class LSTMNet(nn.Module):
    """
    LSTM history encoder + MLP scoring head.
    Used for actor and both Q-networks.

    forward_sequence — training: full window in one call → all hidden states.
    forward_step     — inference: single step → updated hidden state.
    score            — given h_t and candidates → scalar scores / Q-values.
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

    def push(self, curr_feats: list, action: int, reward: float,
             next_feats: list, done: bool) -> None:
        self._buf.append((
            [np.asarray(f, dtype=np.float32) for f in curr_feats],
            int(action),
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

class SACAgent:
    """
    Parameters
    ----------
    use_lstm          : use LSTMNet + EpisodeReplayBuffer
    lstm_hidden       : LSTM hidden size (LSTM mode only)
    seq_len           : BPTT window length
    min_episodes      : warmup episodes before LSTM training starts
    episode_capacity  : max episodes in EpisodeReplayBuffer
    hidden_size       : MLP hidden size (both modes)
    lr_q / lr_actor / lr_alpha : learning rates
    gamma             : discount factor
    tau               : Polyak averaging for target networks
    buffer_size       : ReplayBuffer capacity (flat mode)
    batch_size        : transitions (flat) or sequences (LSTM) per step
    target_entropy    : desired entropy (nats)
    auto_alpha        : auto-tune temperature
    init_alpha        : initial temperature
    min_replay_size   : warmup transitions (flat mode)
    target_update_freq: soft-update targets every N steps
    """

    def __init__(
        self,
        use_lstm: bool = False,
        lstm_hidden: int = 64,
        seq_len: int = 16,
        min_episodes: int = 50,
        episode_capacity: int = 1000,
        hidden_size: int = 128,
        lr_q: float = 3e-4,
        lr_actor: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_entropy: float = 1.0,
        auto_alpha: bool = True,
        init_alpha: float = 0.2,
        min_replay_size: int = 500,
        target_update_freq: int = 1,
    ):
        self.use_lstm    = use_lstm
        self.lstm_hidden = lstm_hidden
        self.seq_len     = seq_len
        self.min_episodes = min_episodes
        self.hidden_size = hidden_size
        self.lr_q        = lr_q
        self.lr_actor    = lr_actor
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.target_entropy   = target_entropy
        self.auto_alpha       = auto_alpha
        self.min_replay_size  = min_replay_size
        self.target_update_freq = target_update_freq

        self.replay_buffer  = ReplayBuffer(buffer_size)
        self.episode_buffer = EpisodeReplayBuffer(episode_capacity, seq_len) if use_lstm else None

        self.feature_size: Optional[int] = None
        self._initialized: bool = False
        self.total_steps: int = 0

        # networks (lazy init)
        self.actor:    Optional[nn.Module] = None
        self.q1:       Optional[nn.Module] = None
        self.q2:       Optional[nn.Module] = None
        self.target_q1: Optional[nn.Module] = None
        self.target_q2: Optional[nn.Module] = None
        self.actor_optimizer = None
        self.q_optimizer     = None

        # temperature
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32,
                                      requires_grad=True)
        self.alpha: float = init_alpha
        self.alpha_optimizer = (
            optim.Adam([self.log_alpha], lr=lr_alpha) if auto_alpha else None
        )

        # LSTM per-episode inference state (actor only)
        self._prev_chosen: Optional[np.ndarray] = None
        self._actor_hidden = None

    # ── lazy network init ────────────────────────────────────────────────────

    def _init_networks(self, feature_size: int) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.feature_size = feature_size
        if self.use_lstm:
            self.actor    = LSTMNet(feature_size, self.lstm_hidden, self.hidden_size)
            self.q1       = LSTMNet(feature_size, self.lstm_hidden, self.hidden_size)
            self.q2       = LSTMNet(feature_size, self.lstm_hidden, self.hidden_size)
            self.target_q1 = LSTMNet(feature_size, self.lstm_hidden, self.hidden_size)
            self.target_q2 = LSTMNet(feature_size, self.lstm_hidden, self.hidden_size)
        else:
            self.actor    = ActorNetwork(feature_size, self.hidden_size)
            self.q1       = QNetwork(feature_size, self.hidden_size)
            self.q2       = QNetwork(feature_size, self.hidden_size)
            self.target_q1 = QNetwork(feature_size, self.hidden_size)
            self.target_q2 = QNetwork(feature_size, self.hidden_size)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.eval()
        self.target_q2.eval()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr_q
        )

    def reset_episode(self) -> None:
        """Reset per-episode LSTM state. Must be called at the start of every episode."""
        if self.feature_size is not None:
            self._prev_chosen = np.zeros(self.feature_size, dtype=np.float32)
        self._actor_hidden = None

    # ── action selection ─────────────────────────────────────────────────────

    def select_action(self, frontier_feats: list) -> int:
        feats_t = torch.tensor(np.array(frontier_feats, dtype=np.float32))
        with torch.no_grad():
            if self.use_lstm:
                prev_t = torch.tensor(self._prev_chosen)
                h, self._actor_hidden = self.actor.forward_step(prev_t, self._actor_hidden)
                scores = self.actor.score(h, feats_t)
            else:
                scores = self.actor(feats_t)
            action = torch.distributions.Categorical(logits=scores).sample()
        return int(action)

    # ── observe ──────────────────────────────────────────────────────────────

    def observe(
        self,
        curr_feats: list,
        action: int,
        reward: float,
        next_feats: list,
        done: bool,
    ) -> Optional[float]:
        self.total_steps += 1

        if self.use_lstm:
            self.episode_buffer.push_step(
                self._prev_chosen, curr_feats, action, reward, done
            )
            self._prev_chosen = np.asarray(curr_feats[action], dtype=np.float32)
            loss = self._train_step_lstm()
        else:
            self.replay_buffer.push(curr_feats, action, reward, next_feats, done)
            loss = self._train_step_flat()

        return loss

    # ── training: flat ───────────────────────────────────────────────────────

    def _train_step_flat(self) -> Optional[float]:
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        curr_feats_list, actions, rewards, next_feats_list, dones = zip(*batch)

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        dones_t   = torch.tensor(dones,   dtype=torch.float32)

        # critic targets: V(s') = Σ_a π(a|s')[min_Q(s',a') - α log π]
        with torch.no_grad():
            next_v = torch.zeros(self.batch_size)
            for i, (nf, d) in enumerate(zip(next_feats_list, dones_t)):
                if not d and nf:
                    nf_t      = torch.tensor(np.stack(nf))
                    log_probs = F.log_softmax(self.actor(nf_t), dim=0)
                    probs     = log_probs.exp()
                    min_q     = torch.min(self.target_q1(nf_t), self.target_q2(nf_t))
                    next_v[i] = (probs * (min_q - self.alpha * log_probs)).sum()

        targets = rewards_t + self.gamma * next_v * (1.0 - dones_t)

        q1_vals, q2_vals = [], []
        for cf, a in zip(curr_feats_list, actions):
            cf_t     = torch.tensor(np.stack(cf))
            feat_a   = cf_t[a : a + 1]
            q1_vals.append(self.q1(feat_a))
            q2_vals.append(self.q2(feat_a))

        curr_q1 = torch.cat(q1_vals)
        curr_q2 = torch.cat(q2_vals)
        critic_loss = F.mse_loss(curr_q1, targets) + F.mse_loss(curr_q2, targets)
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.q_optimizer.step()

        actor_losses: List[torch.Tensor] = []
        log_pi_means: List[torch.Tensor] = []
        for cf in curr_feats_list:
            cf_t      = torch.tensor(np.stack(cf))
            log_probs = F.log_softmax(self.actor(cf_t), dim=0)
            probs     = log_probs.exp()
            with torch.no_grad():
                min_q = torch.min(self.q1(cf_t), self.q2(cf_t))
            actor_losses.append(-(probs * (min_q - self.alpha * log_probs)).sum())
            log_pi_means.append((probs * log_probs).sum())

        actor_loss = torch.stack(actor_losses).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        if self.auto_alpha:
            log_pi_mean = torch.stack(log_pi_means).mean().detach()
            alpha_loss  = -(self.log_alpha.exp() * (log_pi_mean + self.target_entropy))
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().detach())

        if self.total_steps % self.target_update_freq == 0:
            self._soft_update(self.q1, self.target_q1)
            self._soft_update(self.q2, self.target_q2)

        return critic_loss.item()

    # ── training: LSTM ───────────────────────────────────────────────────────

    def _train_step_lstm(self) -> Optional[float]:
        if len(self.episode_buffer) < self.min_episodes:
            return None

        windows = self.episode_buffer.sample(self.batch_size)
        if not windows:
            return None

        q1_preds, q2_preds, q_targets = [], [], []
        actor_losses, log_pi_all      = [], []

        for window in windows:
            T = len(window)
            prev_feats_t = torch.tensor(
                np.stack([s[0] for s in window]), dtype=torch.float32
            ).unsqueeze(0)  # (1, T, F)

            # Run all networks over window (h_0 = zeros)
            all_h_actor, _ = self.actor.forward_sequence(prev_feats_t)
            all_h_q1,    _ = self.q1.forward_sequence(prev_feats_t)
            all_h_q2,    _ = self.q2.forward_sequence(prev_feats_t)
            with torch.no_grad():
                all_h_tq1, _ = self.target_q1.forward_sequence(prev_feats_t)
                all_h_tq2, _ = self.target_q2.forward_sequence(prev_feats_t)

            for t, (_, frontier, action, reward, done) in enumerate(window):
                ft = torch.tensor(np.stack(frontier), dtype=torch.float32)  # (N, F)

                # ── critic predictions ───────────────────────────────────────
                q1_preds.append(self.q1.score(all_h_q1[t], ft)[action])
                q2_preds.append(self.q2.score(all_h_q2[t], ft)[action])

                # ── critic targets ───────────────────────────────────────────
                with torch.no_grad():
                    if done or t == T - 1:
                        tgt = float(reward)
                    else:
                        nf = window[t + 1][1]
                        if nf:
                            nft = torch.tensor(np.stack(nf), dtype=torch.float32)
                            # V(s') uses actor hidden at t+1, target Q at t+1
                            next_log_probs = F.log_softmax(
                                self.actor.score(all_h_actor[t + 1], nft), dim=0
                            )
                            next_probs = next_log_probs.exp()
                            min_q_next = torch.min(
                                self.target_q1.score(all_h_tq1[t + 1], nft),
                                self.target_q2.score(all_h_tq2[t + 1], nft),
                            )
                            next_v = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum()
                            tgt = reward + self.gamma * next_v.item()
                        else:
                            tgt = float(reward)
                q_targets.append(tgt)

                # ── actor loss ───────────────────────────────────────────────
                log_probs = F.log_softmax(self.actor.score(all_h_actor[t], ft), dim=0)
                probs     = log_probs.exp()
                with torch.no_grad():
                    min_q = torch.min(
                        self.q1.score(all_h_q1[t].detach(), ft),
                        self.q2.score(all_h_q2[t].detach(), ft),
                    )
                actor_losses.append(-(probs * (min_q - self.alpha * log_probs)).sum())
                log_pi_all.append((probs * log_probs).sum())

        # ── critic update ────────────────────────────────────────────────────
        q1_t   = torch.stack(q1_preds)
        q2_t   = torch.stack(q2_preds)
        tgt_t  = torch.tensor(q_targets, dtype=torch.float32)
        critic_loss = F.mse_loss(q1_t, tgt_t) + F.mse_loss(q2_t, tgt_t)
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.q_optimizer.step()

        # ── actor update ─────────────────────────────────────────────────────
        actor_loss = torch.stack(actor_losses).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ── temperature update ────────────────────────────────────────────────
        if self.auto_alpha:
            log_pi_mean = torch.stack(log_pi_all).mean().detach()
            alpha_loss  = -(self.log_alpha.exp() * (log_pi_mean + self.target_entropy))
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().detach())

        # ── soft target update ────────────────────────────────────────────────
        if self.total_steps % self.target_update_freq == 0:
            self._soft_update(self.q1, self.target_q1)
            self._soft_update(self.q2, self.target_q2)

        return critic_loss.item()

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for s_p, t_p in zip(source.parameters(), target.parameters()):
            t_p.data.copy_(self.tau * s_p.data + (1.0 - self.tau) * t_p.data)

    # ── persistence ──────────────────────────────────────────────────────────

    def save_onnx(self, path: str) -> None:
        if self.use_lstm:
            print("[WARN] ONNX export not supported for LSTM mode.")
            return

        self.actor.eval()
        dummy = torch.zeros(1, self.feature_size)
        torch.onnx.export(
            self.actor, dummy, path,
            input_names=["features"], output_names=["score"],
            dynamic_axes={"features": {0: "batch_size"}},
            opset_version=17,
            export_params=True,
            dynamo=False
        )


# ── training loop ──────────────────────────────────────────────────────────────

def train(
    env,
    agent: SACAgent,
    fsp_path: str,
    max_episodes: int = 1000,
    max_steps: int = 100_000,
    patience: int = 100,
    results_dir: str = "results",
    verbose: bool = True,
) -> SACAgent:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "sac_training.csv"

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

        while not env._done:
            if not frontier:
                break

            curr_frontier = frontier
            action        = agent.select_action(curr_frontier)

            frontier, reward, done, ep_info = env.step(action)
            next_feats = frontier if (not done and frontier) else []

            if agent._initialized:
                agent.observe(curr_frontier, action, reward, next_feats, done)

            ep_reward    += reward
            ep_steps     += 1
            global_steps += 1

            if global_steps >= max_steps:
                stop_reason = f"max steps ({max_steps:,})"
                break

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, ep_reward, ep_steps, ep_info.get("realizable")]
            )

        if ep % 10 == 0 and agent._initialized:
            onnx_path = results_path / f"sac_ep{ep:04d}.onnx"
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
                f"α={agent.alpha:.4f} | avg10={avg:7.1f} | "
                f"patience={no_improve}/{patience}"
            )

        if no_improve >= patience:
            stop_reason = f"patience ({patience} episodes without improvement)"
            break
        if global_steps >= max_steps:
            break

    print(f"\n[STOP] {stop_reason}  —  total steps: {global_steps:,}")
    return agent
