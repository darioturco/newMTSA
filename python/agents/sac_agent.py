"""
SAC (Soft Actor-Critic) agent for the DCS RL environment.

Architecture note
-----------------
Three network modes (selected by use_lstm / use_transformer):

Flat mode (use_lstm=False, use_transformer=False):
  Actor and Q-networks are MLPs operating on individual feature vectors.
  π(a|s) = softmax_a [ actor(feature_a) ]
  Q(s,a) = q_net(feature_a)

LSTM mode (use_lstm=True):
  Each network (actor, q1, q2 and their targets) has its own LSTMNet that
  consumes the previously chosen transition's feature vector, producing
  h_t that encodes history.

Transformer mode (use_transformer=True):
  Same as LSTM mode but uses TransformerNet for all networks.
  The 'hidden state' for inference is a growing list of previous feature
  vectors (KV cache).

Replay buffers
--------------
Flat       : SACReplayBuffer     — (curr_feats, action, reward, next_feats, done)
Sequential : EpisodeReplayBuffer — full episodes; sampled as seq_len windows
"""

import csv
import re
import random
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent
from .networks import (
    MLPScorer, LSTMNet, TransformerNet,
    OnnxLSTMWrapper, OnnxTransformerWrapper,
    SACReplayBuffer, EpisodeReplayBuffer,
)


class SACAgent(BaseAgent):
    """
    Parameters
    ----------
    use_lstm              : use LSTMNet + EpisodeReplayBuffer
    use_transformer       : use TransformerNet + EpisodeReplayBuffer (overrides use_lstm)
    lstm_hidden           : LSTM hidden size
    seq_len               : BPTT window length
    min_episodes          : warmup episodes before sequential training starts
    episode_capacity      : max episodes in EpisodeReplayBuffer
    hidden_size           : MLP hidden size
    lr_q / lr_actor / lr_alpha : learning rates
    gamma                 : discount factor
    tau                   : Polyak averaging for target networks
    buffer_size           : SACReplayBuffer capacity (flat mode)
    batch_size            : transitions (flat) or sequences (sequential) per step
    target_entropy        : desired entropy (nats)
    auto_alpha            : auto-tune temperature
    init_alpha            : initial temperature
    min_replay_size       : warmup transitions (flat mode)
    target_update_freq    : soft-update targets every N steps
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
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        save_frequency: int = 10,
    ):
        super().__init__(save_frequency=save_frequency)
        self.use_lstm              = use_lstm
        self.use_transformer       = use_transformer
        self.lstm_hidden           = lstm_hidden
        self.seq_len               = seq_len
        self.min_episodes          = min_episodes
        self.hidden_size           = hidden_size
        self.lr_q                  = lr_q
        self.lr_actor              = lr_actor
        self.lr_alpha              = lr_alpha
        self.gamma                 = gamma
        self.tau                   = tau
        self.batch_size            = batch_size
        self.target_entropy        = target_entropy
        self.auto_alpha            = auto_alpha
        self.min_replay_size       = min_replay_size
        self.target_update_freq    = target_update_freq
        self.d_model               = d_model
        self.nhead                 = nhead
        self.num_transformer_layers = num_transformer_layers

        self._use_sequential: bool = use_transformer or use_lstm

        self.replay_buffer  = SACReplayBuffer(buffer_size)
        self.episode_buffer = (
            EpisodeReplayBuffer(episode_capacity, seq_len)
            if self._use_sequential else None
        )

        self.feature_size: Optional[int] = None
        self._initialized: bool = False
        self.total_steps:  int  = 0
        self.last_loss:    Optional[float] = None

        # networks (lazy init)
        self.actor:     Optional[nn.Module] = None
        self.q1:        Optional[nn.Module] = None
        self.q2:        Optional[nn.Module] = None
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

        # per-episode sequential state (actor only)
        self._prev_chosen:    Optional[np.ndarray] = None
        self._actor_hidden = None

    # ── lazy network init ────────────────────────────────────────────────────

    def _init_networks(self, feature_size: int) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.feature_size = feature_size

        def _make_net():
            if self.use_transformer:
                return TransformerNet(
                    feature_size, self.d_model, self.nhead,
                    self.num_transformer_layers, self.hidden_size,
                )
            elif self.use_lstm:
                return LSTMNet(feature_size, self.lstm_hidden, self.hidden_size)
            else:
                return MLPScorer(feature_size, self.hidden_size)

        self.actor     = _make_net()
        self.q1        = _make_net()
        self.q2        = _make_net()
        self.target_q1 = _make_net()
        self.target_q2 = _make_net()

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.eval()
        self.target_q2.eval()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr_q
        )

    def reset_episode(self) -> None:
        """Reset per-episode sequential state. Call at the start of every episode."""
        if self.feature_size is not None:
            self._prev_chosen = np.zeros(self.feature_size, dtype=np.float32)
        self._actor_hidden = None

    # ── action selection ─────────────────────────────────────────────────────

    def select_action(self, frontier_feats: list) -> int:
        feats_t = torch.tensor(np.array(frontier_feats, dtype=np.float32))
        with torch.no_grad():
            if self._use_sequential:
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

        if self._use_sequential:
            self.episode_buffer.push_step(
                self._prev_chosen, curr_feats, action, reward, done
            )
            self._prev_chosen = np.asarray(curr_feats[action], dtype=np.float32)
            loss = self._train_step_sequential()
        else:
            self.replay_buffer.push(curr_feats, action, reward, next_feats, done)
            loss = self._train_step_flat()

        if loss is not None:
            self.last_loss = loss
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
            cf_t   = torch.tensor(np.stack(cf))
            feat_a = cf_t[a : a + 1]
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

    # ── training: sequential (LSTM & Transformer share this path) ─────────────

    def _train_step_sequential(self) -> Optional[float]:
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

            all_h_actor, _ = self.actor.forward_sequence(prev_feats_t)
            all_h_q1,    _ = self.q1.forward_sequence(prev_feats_t)
            all_h_q2,    _ = self.q2.forward_sequence(prev_feats_t)
            with torch.no_grad():
                all_h_tq1, _ = self.target_q1.forward_sequence(prev_feats_t)
                all_h_tq2, _ = self.target_q2.forward_sequence(prev_feats_t)

            for t, (_, frontier, action, reward, done) in enumerate(window):
                ft = torch.tensor(np.stack(frontier), dtype=torch.float32)

                q1_preds.append(self.q1.score(all_h_q1[t], ft)[action])
                q2_preds.append(self.q2.score(all_h_q2[t], ft)[action])

                with torch.no_grad():
                    if done or t == T - 1:
                        tgt = float(reward)
                    else:
                        nf = window[t + 1][1]
                        if nf:
                            nft = torch.tensor(np.stack(nf), dtype=torch.float32)
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

                log_probs = F.log_softmax(self.actor.score(all_h_actor[t], ft), dim=0)
                probs     = log_probs.exp()
                with torch.no_grad():
                    min_q = torch.min(
                        self.q1.score(all_h_q1[t].detach(), ft),
                        self.q2.score(all_h_q2[t].detach(), ft),
                    )
                actor_losses.append(-(probs * (min_q - self.alpha * log_probs)).sum())
                log_pi_all.append((probs * log_probs).sum())

        q1_t  = torch.stack(q1_preds)
        q2_t  = torch.stack(q2_preds)
        tgt_t = torch.tensor(q_targets, dtype=torch.float32)
        critic_loss = F.mse_loss(q1_t, tgt_t) + F.mse_loss(q2_t, tgt_t)
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.q_optimizer.step()

        actor_loss = torch.stack(actor_losses).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        if self.auto_alpha:
            log_pi_mean = torch.stack(log_pi_all).mean().detach()
            alpha_loss  = -(self.log_alpha.exp() * (log_pi_mean + self.target_entropy))
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().detach())

        if self.total_steps % self.target_update_freq == 0:
            self._soft_update(self.q1, self.target_q1)
            self._soft_update(self.q2, self.target_q2)

        return critic_loss.item()

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for s_p, t_p in zip(source.parameters(), target.parameters()):
            t_p.data.copy_(self.tau * s_p.data + (1.0 - self.tau) * t_p.data)

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

    # ── persistence ──────────────────────────────────────────────────────────

    def save_onnx(self, path: str) -> None:
        F = self.feature_size
        self.actor.eval()
        if self.use_transformer:
            wrapper = OnnxTransformerWrapper(self.actor)
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
            wrapper = OnnxLSTMWrapper(self.actor)
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
                self.actor, torch.zeros(1, F), path,
                input_names=["features"], output_names=["score"],
                dynamic_axes={"features": {0: "N"}, "score": {0: "N"}},
                opset_version=17, export_params=True, dynamo=False,
            )

    def save_model(self, path: str) -> None:
        if not self._initialized:
            print("[WARN] Agent not initialized; nothing to save.")
            return
        torch.save({
            "feature_size": self.feature_size,
            "config": self._config_dict(),
            "actor":     self.actor.state_dict(),
            "q1":        self.q1.state_dict(),
            "q2":        self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer":     self.q_optimizer.state_dict(),
        }, path)

    @classmethod
    def load_model(cls, path: str) -> "SACAgent":
        payload = torch.load(path, map_location="cpu")
        agent = cls(**payload["config"])
        agent._init_networks(payload["feature_size"])
        agent.actor.load_state_dict(payload["actor"])
        agent.q1.load_state_dict(payload["q1"])
        agent.q2.load_state_dict(payload["q2"])
        agent.target_q1.load_state_dict(payload["target_q1"])
        agent.target_q2.load_state_dict(payload["target_q2"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.q_optimizer.load_state_dict(payload["q_optimizer"])
        return agent

    def _config_dict(self) -> dict:
        return {
            "use_lstm": self.use_lstm,
            "use_transformer": self.use_transformer,
            "lstm_hidden": self.lstm_hidden,
            "seq_len": self.seq_len,
            "min_episodes": self.min_episodes,
            "hidden_size": self.hidden_size,
            "lr_q": self.lr_q,
            "lr_actor": self.lr_actor,
            "lr_alpha": self.lr_alpha,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "target_entropy": self.target_entropy,
            "auto_alpha": self.auto_alpha,
            "init_alpha": self.alpha,
            "min_replay_size": self.min_replay_size,
            "target_update_freq": self.target_update_freq,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_transformer_layers": self.num_transformer_layers,
            "save_frequency": self.save_frequency,
        }


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

    if agent._use_sequential:
        agent._fill_episode_buffer(env, fsp_path)

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

            curr_frontier = frontier
            action        = agent.select_action(curr_frontier)
            if ep_trace is not None:
                ep_trace.append(action)

            frontier, reward, done, ep_info = env.step(action)
            next_feats = frontier if (not done and frontier) else []

            if agent._initialized:
                agent.observe(curr_frontier, action, reward, next_feats, done)

            ep_reward     += reward
            ep_steps      += 1
            ep_expansions += ep_info.get("expansions", 0)
            global_steps  += 1

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
            agent.save_onnx(str(results_path / f"sac_ep{ep:04d}.onnx"))
            if agent._use_sequential:
                agent.save_model(str(results_path / f"sac_ep{ep:04d}.pt"))

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
                f"α={agent.alpha:.4f} | avg50={avg:7.1f} | "
                f"loss={loss_str} | patience={no_improve}/{patience}"
            )

        if no_improve >= patience:
            stop_reason = f"patience ({patience} episodes without improvement)"
            break
        if global_steps >= max_steps:
            stop_reason = f"max steps ({max_steps:,})"
            break

    print(
        f"\n[STOP] SAC | {stop_reason}  —  total steps: {global_steps:,}  —  total episodes: {ep}"
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
