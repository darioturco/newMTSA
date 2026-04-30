"""
PPO agent for the DCS RL environment.

Architecture note
-----------------
Two network modes (selected by use_lstm):

Flat mode (use_lstm=False):
  π(a|s) = softmax_a [ actor(feature_a) ]
  V(s)   = critic( mean_pool(frontier_feats) )

LSTM mode (use_lstm=True):
  Shared LSTM consumes the previously chosen transition's feature vector,
  producing h_t that encodes exploration history.
  π(a|s) = softmax_a [ actor_head(concat(h_t, feature_a)) ]
  V(s)   = critic_head(h_t)
  Training re-runs the LSTM over the full rollout sequence for each PPO epoch.
"""

import csv
import random
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── flat networks ──────────────────────────────────────────────────────────────

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
        # x: (N, F) → scores: (N,)
        return self.net(x).squeeze(-1)


class CriticNetwork(nn.Module):
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
        # x: (1, F) mean-pooled frontier → scalar
        return self.net(x).squeeze(-1)


# ── LSTM network ───────────────────────────────────────────────────────────────

class LSTMActorCritic(nn.Module):
    """
    Shared LSTM encoder with separate actor and critic heads.

    Actor head : concat(h_t, feat_a) → scalar score per candidate
    Critic head: h_t → scalar value

    forward_sequence — training: run LSTM over full episode, return all h_t.
    forward_step     — inference: run one step, update running hidden state.
    actor_scores     — score all candidates given h_t.
    critic_value     — state value given h_t.
    """

    def __init__(self, feature_size: int, lstm_hidden: int, mlp_hidden: int):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(feature_size, lstm_hidden, batch_first=True)
        self.actor_head = nn.Sequential(
            nn.Linear(lstm_hidden + feature_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden),
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

    def actor_scores(self, h: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """h: (lstm_hidden,), candidates: (N, F) → (N,)"""
        h_exp = h.unsqueeze(0).expand(candidates.size(0), -1)
        return self.actor_head(torch.cat([h_exp, candidates], dim=-1)).squeeze(-1)

    def critic_value(self, h: torch.Tensor) -> torch.Tensor:
        """h: (lstm_hidden,) → scalar"""
        return self.critic_head(h.unsqueeze(0)).squeeze()


# ── rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one episode of transitions for the PPO update.
    prev_chosen_feats stores the feature of the previously expanded transition
    (zero vector at episode start). Used only in LSTM mode during update.
    """

    def __init__(self):
        self.prev_chosen_feats: List[np.ndarray] = []
        self.frontier_feats:    List[List[np.ndarray]] = []
        self.actions:    List[int]   = []
        self.log_probs:  List[float] = []
        self.rewards:    List[float] = []
        self.values:     List[float] = []
        self.dones:      List[bool]  = []

    def push(
        self,
        prev_chosen_feat: np.ndarray,
        frontier_feats: list,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.prev_chosen_feats.append(prev_chosen_feat.copy())
        self.frontier_feats.append([np.asarray(f, dtype=np.float32) for f in frontier_feats])
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.actions)


# ── agent ──────────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Parameters
    ----------
    use_lstm     : use LSTMActorCritic (LSTM mode) vs separate actor/critic (flat)
    lstm_hidden  : LSTM hidden size (LSTM mode only)
    hidden_size  : MLP hidden size (both modes)
    lr           : Adam learning rate
    gamma        : discount factor
    gae_lambda   : GAE lambda
    clip_eps     : PPO clipping epsilon
    value_coef   : weight of value loss
    entropy_coef : entropy bonus weight
    ppo_epochs   : gradient passes per rollout
    mini_batch   : mini-batch size; 0 = full rollout (forced in LSTM mode)
    """

    def __init__(
        self,
        use_lstm: bool = False,
        lstm_hidden: int = 64,
        hidden_size: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        mini_batch: int = 64,
    ):
        self.use_lstm    = use_lstm
        self.lstm_hidden = lstm_hidden
        self.hidden_size = hidden_size
        self.lr          = lr
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.clip_eps    = clip_eps
        self.value_coef  = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs  = ppo_epochs
        self.mini_batch  = mini_batch

        # networks (lazy init)
        self.actor:  Optional[ActorNetwork]    = None
        self.critic: Optional[CriticNetwork]   = None
        self.net:    Optional[LSTMActorCritic] = None
        self.optimizer = None
        self.feature_size: Optional[int] = None
        self._initialized: bool = False

        self.rollout = RolloutBuffer()
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
            self.net = LSTMActorCritic(feature_size, self.lstm_hidden, self.hidden_size)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.actor  = ActorNetwork(feature_size, self.hidden_size)
            self.critic = CriticNetwork(feature_size, self.hidden_size)
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=self.lr,
            )

    def reset_episode(self) -> None:
        """Reset per-episode LSTM state. Must be called at the start of every episode."""
        if self.feature_size is not None:
            self._prev_chosen = np.zeros(self.feature_size, dtype=np.float32)
        self._hidden = None

    # ── action selection ─────────────────────────────────────────────────────

    def select_action(self, frontier_feats: list) -> Tuple[int, float, float]:
        """Returns (action_index, log_prob, value_estimate)."""
        feats_t = torch.tensor(np.array(frontier_feats, dtype=np.float32))
        with torch.no_grad():
            if self.use_lstm:
                prev_t  = torch.tensor(self._prev_chosen)
                h, self._hidden = self.net.forward_step(prev_t, self._hidden)
                scores  = self.net.actor_scores(h, feats_t)
                value   = self.net.critic_value(h).item()
            else:
                scores  = self.actor(feats_t)
                value   = self.critic(feats_t.mean(dim=0, keepdim=True)).item()

            dist     = torch.distributions.Categorical(logits=scores)
            action   = dist.sample()
            log_prob = dist.log_prob(action).item()

        return int(action), log_prob, value

    # ── store transition ─────────────────────────────────────────────────────

    def observe(
        self,
        frontier_feats: list,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        prev = (
            self._prev_chosen
            if self._prev_chosen is not None
            else np.zeros(len(frontier_feats[0]), dtype=np.float32)
        )
        self.rollout.push(prev, frontier_feats, action, log_prob, reward, value, done)
        if self.use_lstm:
            self._prev_chosen = np.asarray(frontier_feats[action], dtype=np.float32)
        self.total_steps += 1

    # ── PPO update ───────────────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """Run PPO update over collected rollout. Clears buffer. Returns mean loss."""
        T = len(self.rollout)
        if T == 0:
            return None
        return self._update_lstm() if self.use_lstm else self._update_flat()

    def _gae(self):
        """Compute GAE advantages and discounted returns. Common to both modes."""
        T = len(self.rollout)
        returns    = np.zeros(T, dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            r    = self.rollout.rewards[t]
            v    = self.rollout.values[t]
            done = float(self.rollout.dones[t])
            next_v = 0.0 if t == T - 1 else self.rollout.values[t + 1] * (1.0 - done)
            delta  = r + self.gamma * next_v - v
            gae    = delta + self.gamma * self.gae_lambda * (1.0 - done) * gae
            advantages[t] = gae
            returns[t]    = gae + v
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    # ── flat update ──────────────────────────────────────────────────────────

    def _update_flat(self) -> Optional[float]:
        T = len(self.rollout)
        advantages, returns = self._gae()

        indices  = list(range(T))
        mb_size  = self.mini_batch if self.mini_batch > 0 else T
        total_loss, n_updates = 0.0, 0

        for _ in range(self.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, T, mb_size):
                batch_idx = indices[start : start + mb_size]

                mb_new_log_probs, mb_entropies, mb_new_values = [], [], []
                for i in batch_idx:
                    feats_t = torch.tensor(
                        np.array(self.rollout.frontier_feats[i], dtype=np.float32)
                    )
                    scores = self.actor(feats_t)
                    dist   = torch.distributions.Categorical(logits=scores)
                    act_t  = torch.tensor(self.rollout.actions[i])
                    mb_new_log_probs.append(dist.log_prob(act_t))
                    mb_entropies.append(dist.entropy())
                    mean_feat = feats_t.mean(dim=0, keepdim=True)
                    mb_new_values.append(self.critic(mean_feat).squeeze())

                new_log_probs = torch.stack(mb_new_log_probs)
                entropy       = torch.stack(mb_entropies).mean()
                new_values    = torch.stack(mb_new_values)
                old_log_probs = torch.tensor(
                    [self.rollout.log_probs[i] for i in batch_idx], dtype=torch.float32
                )
                adv_batch = torch.tensor(
                    [advantages[i] for i in batch_idx], dtype=torch.float32
                )
                ret_batch = torch.tensor(
                    [returns[i] for i in batch_idx], dtype=torch.float32
                )

                ratio  = torch.exp(new_log_probs - old_log_probs)
                surr1  = ratio * adv_batch
                surr2  = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_batch
                loss   = (
                    -torch.min(surr1, surr2).mean()
                    + self.value_coef  * F.mse_loss(new_values, ret_batch)
                    - self.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 1.0
                )
                self.optimizer.step()
                total_loss += loss.item()
                n_updates  += 1

        self.rollout.clear()
        return total_loss / n_updates if n_updates else None

    # ── LSTM update ──────────────────────────────────────────────────────────

    def _update_lstm(self) -> Optional[float]:
        """
        Re-run the LSTM over the full rollout for each PPO epoch.
        Mini-batching is not used (would break temporal ordering).
        """
        T = len(self.rollout)
        advantages, returns = self._gae()

        # (1, T, F) sequence of prev_chosen_feats for LSTM input
        prev_feats_seq = torch.tensor(
            np.stack(self.rollout.prev_chosen_feats), dtype=torch.float32
        ).unsqueeze(0)

        old_log_probs_t = torch.tensor(self.rollout.log_probs, dtype=torch.float32)
        advantages_t    = torch.tensor(advantages, dtype=torch.float32)
        returns_t       = torch.tensor(returns,    dtype=torch.float32)

        total_loss, n_updates = 0.0, 0

        for _ in range(self.ppo_epochs):
            # Re-run LSTM from h=0 over the full episode
            all_h, _ = self.net.forward_sequence(prev_feats_seq)  # (T, lstm_hidden)

            new_log_probs_list, entropies_list, new_values_list = [], [], []
            for t in range(T):
                h_t     = all_h[t]
                feats_t = torch.tensor(
                    np.array(self.rollout.frontier_feats[t], dtype=np.float32)
                )
                scores  = self.net.actor_scores(h_t, feats_t)
                dist    = torch.distributions.Categorical(logits=scores)
                act_t   = torch.tensor(self.rollout.actions[t])
                new_log_probs_list.append(dist.log_prob(act_t))
                entropies_list.append(dist.entropy())
                new_values_list.append(self.net.critic_value(h_t))

            new_log_probs = torch.stack(new_log_probs_list)
            entropy       = torch.stack(entropies_list).mean()
            new_values    = torch.stack(new_values_list)

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t
            loss  = (
                -torch.min(surr1, surr2).mean()
                + self.value_coef   * F.mse_loss(new_values, returns_t)
                - self.entropy_coef * entropy
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_updates  += 1

        self.rollout.clear()
        return total_loss / n_updates if n_updates else None

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
    agent: PPOAgent,
    fsp_path: str,
    max_episodes: int = 1000,
    max_steps: int = 100_000,
    patience: int = 100,
    results_dir: str = "results",
    verbose: bool = True,
) -> PPOAgent:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "ppo_training.csv"

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

            curr_frontier             = frontier
            action, log_prob, value   = agent.select_action(curr_frontier)

            frontier, reward, done, ep_info = env.step(action)

            if agent._initialized:
                agent.observe(curr_frontier, action, log_prob, reward, value, done)

            ep_reward    += reward
            ep_steps     += 1
            global_steps += 1

            if global_steps >= max_steps:
                stop_reason = f"max steps ({max_steps:,})"
                break

        if agent._initialized:
            agent.update()

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, ep_reward, ep_steps, ep_info.get("realizable")]
            )

        if ep % 10 == 0 and agent._initialized:
            onnx_path = results_path / f"ppo_ep{ep:04d}.onnx"
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
                f"avg10={avg:7.1f} | patience={no_improve}/{patience}"
            )

        if no_improve >= patience:
            stop_reason = f"patience ({patience} episodes without improvement)"
            break
        if global_steps >= max_steps:
            break

    print(f"\n[STOP] {stop_reason}  —  total steps: {global_steps:,}")
    return agent
