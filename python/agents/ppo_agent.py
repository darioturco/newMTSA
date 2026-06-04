"""
PPO agent for the DCS RL environment.

Architecture note
-----------------
Three network modes (selected by use_lstm / use_transformer):

Flat mode (use_lstm=False, use_transformer=False):
  π(a|s) = softmax_a [ actor(feature_a) ]
  V(s)   = critic( mean_pool(frontier_feats) )

LSTM mode (use_lstm=True):
  Shared LSTM consumes the previously chosen transition's feature vector,
  producing h_t that encodes exploration history.
  π(a|s) = softmax_a [ actor_head(concat(h_t, feature_a)) ]
  V(s)   = critic_head(h_t)
  Training re-runs the LSTM over the full rollout sequence for each PPO epoch.

Transformer mode (use_transformer=True):
  Same as LSTM mode but uses a Transformer encoder instead.
  The 'hidden state' is a growing list of previous feature vectors (KV cache).
"""

import csv
import re
import random
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent
from .networks import (
    MLPScorer, LSTMActorCritic, TransformerActorCritic,
    OnnxLSTMActorWrapper, OnnxTransformerActorWrapper,
    RolloutBuffer,
)


class PPOAgent(BaseAgent):
    """
    Parameters
    ----------
    use_lstm              : use LSTMActorCritic (LSTM mode)
    use_transformer       : use TransformerActorCritic (overrides use_lstm)
    lstm_hidden           : LSTM hidden size
    hidden_size           : MLP hidden size
    lr                    : Adam learning rate
    gamma                 : discount factor
    gae_lambda            : GAE lambda
    clip_eps              : PPO clipping epsilon
    value_coef            : weight of value loss
    entropy_coef          : entropy bonus weight
    ppo_epochs            : gradient passes per rollout
    mini_batch            : mini-batch size; 0 = full rollout (forced in sequential mode)
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
        hidden_size: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        mini_batch: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        save_frequency: int = 10,
    ):
        super().__init__(save_frequency=save_frequency)
        self.use_lstm              = use_lstm
        self.use_transformer       = use_transformer
        self.lstm_hidden           = lstm_hidden
        self.hidden_size           = hidden_size
        self.lr                    = lr
        self.gamma                 = gamma
        self.gae_lambda            = gae_lambda
        self.clip_eps              = clip_eps
        self.value_coef            = value_coef
        self.entropy_coef          = entropy_coef
        self.ppo_epochs            = ppo_epochs
        self.mini_batch            = mini_batch
        self.d_model               = d_model
        self.nhead                 = nhead
        self.num_transformer_layers = num_transformer_layers

        self._use_sequential: bool = use_transformer or use_lstm

        # networks (lazy init)
        self.net:    Optional[nn.Module] = None   # sequential mode
        self.actor:  Optional[nn.Module] = None   # flat mode
        self.critic: Optional[nn.Module] = None   # flat mode
        self.optimizer = None
        self.feature_size: Optional[int] = None
        self._initialized: bool = False

        self.rollout = RolloutBuffer()
        self.total_steps: int = 0
        self.last_loss:   Optional[float] = None

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
            self.net = TransformerActorCritic(
                feature_size, self.d_model, self.nhead,
                self.num_transformer_layers, self.hidden_size,
            )
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.use_lstm:
            self.net = LSTMActorCritic(feature_size, self.lstm_hidden, self.hidden_size)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.actor  = MLPScorer(feature_size, self.hidden_size)
            self.critic = MLPScorer(feature_size, self.hidden_size)
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=self.lr,
            )

    def reset_episode(self) -> None:
        """Reset per-episode sequential state. Call at the start of every episode."""
        if self.feature_size is not None:
            self._prev_chosen = np.zeros(self.feature_size, dtype=np.float32)
        self._hidden = None

    # ── action selection ─────────────────────────────────────────────────────

    def select_action(self, frontier_feats: list) -> Tuple[int, float, float]:
        """Returns (action_index, log_prob, value_estimate)."""
        feats_t = torch.tensor(np.array(frontier_feats, dtype=np.float32))
        with torch.no_grad():
            if self._use_sequential:
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
        if self._use_sequential:
            self._prev_chosen = np.asarray(frontier_feats[action], dtype=np.float32)
        self.total_steps += 1

    # ── PPO update ───────────────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """Run PPO update over collected rollout. Clears buffer. Returns mean loss."""
        if len(self.rollout) == 0:
            return None
        loss = self._update_sequential() if self._use_sequential else self._update_flat()
        if loss is not None:
            self.last_loss = loss
        return loss

    def _gae(self):
        """Compute GAE advantages and discounted returns."""
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

    # ── sequential update (LSTM & Transformer share this path) ────────────────

    def _update_sequential(self) -> Optional[float]:
        """Re-run the encoder over the full rollout for each PPO epoch."""
        T = len(self.rollout)
        advantages, returns = self._gae()

        # (1, T, F) sequence of prev_chosen_feats for encoder input
        prev_feats_seq = torch.tensor(
            np.stack(self.rollout.prev_chosen_feats), dtype=torch.float32
        ).unsqueeze(0)

        old_log_probs_t = torch.tensor(self.rollout.log_probs, dtype=torch.float32)
        advantages_t    = torch.tensor(advantages, dtype=torch.float32)
        returns_t       = torch.tensor(returns,    dtype=torch.float32)

        total_loss, n_updates = 0.0, 0

        for _ in range(self.ppo_epochs):
            # Re-run encoder from h=0 over the full episode
            all_h, _ = self.net.forward_sequence(prev_feats_seq)  # (T, hidden)

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
        F = self.feature_size
        if self.use_transformer:
            self.net.eval()
            wrapper = OnnxTransformerActorWrapper(self.net)
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
            self.net.eval()
            wrapper = OnnxLSTMActorWrapper(self.net)
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
            self.actor.eval()
            torch.onnx.export(
                self.actor, torch.zeros(1, F), path,
                input_names=["features"], output_names=["score"],
                dynamic_axes={"features": {0: "N"}, "score": {0: "N"}},
                opset_version=17, export_params=True, dynamo=False,
            )

    def save_model(self, path: str) -> None:
        if not self._initialized:
            print("[WARN] Agent not initialized; nothing to save.")
            return
        payload = {
            "feature_size": self.feature_size,
            "config": self._config_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self._use_sequential:
            payload["net"] = self.net.state_dict()
        else:
            payload["actor"]  = self.actor.state_dict()
            payload["critic"] = self.critic.state_dict()
        torch.save(payload, path)

    @classmethod
    def load_model(cls, path: str) -> "PPOAgent":
        payload = torch.load(path, map_location="cpu")
        agent = cls(**payload["config"])
        agent._init_networks(payload["feature_size"])
        if agent._use_sequential:
            agent.net.load_state_dict(payload["net"])
        else:
            agent.actor.load_state_dict(payload["actor"])
            agent.critic.load_state_dict(payload["critic"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        return agent

    def _config_dict(self) -> dict:
        return {
            "use_lstm": self.use_lstm,
            "use_transformer": self.use_transformer,
            "lstm_hidden": self.lstm_hidden,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_eps,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "ppo_epochs": self.ppo_epochs,
            "mini_batch": self.mini_batch,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_transformer_layers": self.num_transformer_layers,
            "save_frequency": self.save_frequency,
        }


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
    env.reset(fsp_path)
    print(f"Instance type: {'non-blocking' if env.is_nonblocking else 'GR(1)'}")
    if env.uses_features() and env.frontier:
        print(f"Feature size  : {len(env.frontier[0])}")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "training.csv"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_reward", "steps", "expansions", "loss", "realizable", "director_transitions", "ctrl_actions"])

    best_ep_reward = float("-inf")
    best_avg     = float("-inf")
    best_loss    = float("inf")
    no_improve   = 0
    recent: deque = deque(maxlen=50)
    global_steps = 0
    stop_reason  = f"max episodes ({max_episodes})"

    feat_counter: Counter = Counter()
    feature_names: list = None
    is_super_rl = getattr(env, "_heuristic_name", "") == "SUPER_RL"

    for ep in range(1, max_episodes + 1):
        frontier = env.reset(fsp_path)

        if feature_names is None and frontier:
            try:
                raw = env._dcs.getFeatureNames()
                feature_names = [str(n) for n in raw]
            except Exception:
                pass

        if frontier and not agent._initialized:
            agent._init_networks(len(frontier[0]))

        agent.reset_episode()

        ep_reward     = 0
        ep_steps      = 0
        ep_expansions = 0
        ep_info       = {}
        ep_trace: list = [] if is_super_rl else None

        while not env.is_finished:
            if not frontier:
                break

            for fv in frontier:
                feat_counter[tuple(fv)] += 1

            curr_frontier             = frontier
            action, log_prob, value   = agent.select_action(curr_frontier)
            if ep_trace is not None:
                ep_trace.append(action)

            frontier, reward, done, ep_info = env.step(action)

            if agent._initialized:
                agent.observe(curr_frontier, action, log_prob, reward, value, done)

            ep_reward     += reward
            ep_steps      += 1
            ep_expansions += ep_info.get("expansions", 0)
            global_steps  += 1

        if agent._initialized:
            agent.update()

        realizable = ep_info.get("realizable")
        if realizable is None and env.is_finished:
            realizable = bool(env._dcs.isRealizable())
        loss_val   = f"{agent.last_loss:.6f}" if agent.last_loss is not None else ""
        ctrl_acts = " ".join(map(str, ep_trace)) if ep_trace is not None else ""
        director_trans = ep_info.get("director_transitions")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, ep_reward, ep_steps, ep_expansions, loss_val,
                 realizable, director_trans, ctrl_acts]
            )

        if ep % agent.save_frequency == 0 and agent._initialized:
            agent.save_onnx(str(results_path / f"ppo_ep{ep:04d}.onnx"))
            if agent._use_sequential:
                agent.save_model(str(results_path / f"ppo_ep{ep:04d}.pt"))

        recent.append(ep_reward)
        avg = sum(recent) / len(recent)
        new_loss_min    = agent.last_loss is not None and agent.last_loss < best_loss
        new_best_reward = ep_reward > best_ep_reward
        new_best_avg    = avg > best_avg
        if new_loss_min:
            best_loss = agent.last_loss
        if new_best_reward:
            best_ep_reward = ep_reward
        if new_best_avg:
            best_avg = avg
        if new_best_reward or new_best_avg or new_loss_min:
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            loss_str = f"{agent.last_loss:.5f}" if agent.last_loss is not None else "N/A   "
            if is_super_rl:
                tb  = ep_info.get("terminal_bonus") or 0
                dt  = ep_info.get("director_transitions") or 0
                exp_str = f" | expansions={ep_expansions:6d} | director={dt:6d} | bonus={tb:6d}"
            else:
                exp_str = ""
            print(
                f"Ep {ep:4d} | reward={ep_reward:6d} | steps={ep_steps:4d}{exp_str} | "
                f"avg50={avg:7.1f} | loss={loss_str} | patience={no_improve}/{patience}"
            )

        if no_improve >= patience:
            stop_reason = f"patience ({patience} episodes without improvement)"
            break
        if global_steps >= max_steps:
            stop_reason = f"max steps ({max_steps:,})"
            break

    print(
        f"\n[STOP] PPO | {stop_reason}  —  total steps: {global_steps:,}  —  total episodes: {ep}"
        f"  —  best reward: {best_ep_reward}"
        f"  —  end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


    if feat_counter:
        dim = len(next(iter(feat_counter)))
        if feature_names and len(feature_names) == dim:
            header = feature_names + ["count"]
        else:
            header = [f"f{i}" for i in range(dim)] + ["count"]
        feat_path = results_path / "feature_vectors.csv"
        with open(feat_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for vec, cnt in sorted(feat_counter.items(), key=lambda x: -x[1]):
                w.writerow([*vec, cnt])
        if verbose:
            print(f"[Features] {len(feat_counter):,} unique vectors → {feat_path}")

    agent.trained = True
    return agent
