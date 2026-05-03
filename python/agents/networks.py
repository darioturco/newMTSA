"""
Shared network architectures and replay buffers for DQN, PPO, and SAC agents.
"""

import random
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# ── MLP scorer (flat mode) ─────────────────────────────────────────────────────

class MLPScorer(nn.Module):
    """2-layer MLP: feature_size → 1. Used as Q-net, actor, and critic in flat mode."""

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


# ── LSTM networks ──────────────────────────────────────────────────────────────

class LSTMNet(nn.Module):
    """
    LSTM history encoder + MLP scoring head (single head).
    Used for DQN Q-network and SAC actor/Q-networks.

    API:
      forward_sequence(prev_feats) → (all_h: (T, lstm_hidden), state)
      forward_step(prev_feat, hidden) → (h: (lstm_hidden,), new_hidden)
      score(h, candidates) → (N,)
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


class LSTMActorCritic(nn.Module):
    """
    Shared LSTM encoder with separate actor and critic heads. Used by PPO.

    API:
      forward_sequence(prev_feats) → (all_h: (T, lstm_hidden), state)
      forward_step(prev_feat, hidden) → (h: (lstm_hidden,), new_hidden)
      actor_scores(h, candidates) → (N,)
      critic_value(h) → scalar
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


# ── Transformer networks ───────────────────────────────────────────────────────

class TransformerNet(nn.Module):
    """
    Transformer encoder + MLP scoring head (single head).
    Drop-in replacement for LSTMNet with the same forward_sequence/forward_step/score API.

    The 'hidden state' for inference is a growing list of detached previous feature
    tensors (acts as a KV cache). Pass None to reset at episode start.

    API:
      forward_sequence(prev_feats) → (all_h: (T, d_model), None)
      forward_step(prev_feat, cache) → (h: (d_model,), new_cache)
      score(h, candidates) → (N,)
    """

    def __init__(
        self,
        feature_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        mlp_hidden: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(feature_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + feature_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward_sequence(self, prev_feats: torch.Tensor, hidden=None):
        """prev_feats: (1, T, F) → all_h: (T, d_model), None"""
        x = self.input_proj(prev_feats)   # (1, T, d_model)
        h = self.transformer(x)           # (1, T, d_model)
        return h.squeeze(0), None         # (T, d_model), no state

    def forward_step(self, prev_feat: torch.Tensor, cache=None):
        """
        prev_feat: (F,), cache: list[Tensor(F,)] | None
        Returns: h (d_model,), updated cache list
        """
        if cache is None:
            cache = []
        new_cache = cache + [prev_feat.detach()]
        seq = torch.stack(new_cache).unsqueeze(0)  # (1, T, F)
        x = self.input_proj(seq)                   # (1, T, d_model)
        h_all = self.transformer(x)                # (1, T, d_model)
        return h_all.squeeze(0)[-1], new_cache     # last position, cache

    def score(self, h: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """h: (d_model,), candidates: (N, F) → (N,)"""
        h_exp = h.unsqueeze(0).expand(candidates.size(0), -1)
        return self.head(torch.cat([h_exp, candidates], dim=-1)).squeeze(-1)


class TransformerActorCritic(nn.Module):
    """
    Transformer encoder with separate actor and critic heads. Used by PPO.
    Drop-in replacement for LSTMActorCritic with the same API.

    API:
      forward_sequence(prev_feats) → (all_h: (T, d_model), None)
      forward_step(prev_feat, cache) → (h: (d_model,), new_cache)
      actor_scores(h, candidates) → (N,)
      critic_value(h) → scalar
    """

    def __init__(
        self,
        feature_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        mlp_hidden: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(feature_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.actor_head = nn.Sequential(
            nn.Linear(d_model + feature_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward_sequence(self, prev_feats: torch.Tensor, hidden=None):
        """prev_feats: (1, T, F) → all_h: (T, d_model), None"""
        x = self.input_proj(prev_feats)
        h = self.transformer(x)
        return h.squeeze(0), None

    def forward_step(self, prev_feat: torch.Tensor, cache=None):
        if cache is None:
            cache = []
        new_cache = cache + [prev_feat.detach()]
        seq = torch.stack(new_cache).unsqueeze(0)
        x = self.input_proj(seq)
        h_all = self.transformer(x)
        return h_all.squeeze(0)[-1], new_cache

    def actor_scores(self, h: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """h: (d_model,), candidates: (N, F) → (N,)"""
        h_exp = h.unsqueeze(0).expand(candidates.size(0), -1)
        return self.actor_head(torch.cat([h_exp, candidates], dim=-1)).squeeze(-1)

    def critic_value(self, h: torch.Tensor) -> torch.Tensor:
        """h: (d_model,) → scalar"""
        return self.critic_head(h.unsqueeze(0)).squeeze()


# ── ONNX export wrappers ───────────────────────────────────────────────────────

class OnnxLSTMWrapper(nn.Module):
    """
    Single-head LSTM step wrapper for ONNX export (DQN / SAC actor).
    Inputs:  prev_feat (1,1,F), h_in (1,1,H), c_in (1,1,H), candidates (N,F)
    Outputs: scores (N,), h_out (1,1,H), c_out (1,1,H)
    """
    def __init__(self, net: "LSTMNet"):
        super().__init__()
        self.lstm = net.lstm
        self.head = net.head

    def forward(self, prev_feat, h_in, c_in, candidates):
        out, (h_out, c_out) = self.lstm(prev_feat, (h_in, c_in))
        h = out.squeeze(0).squeeze(0)                               # (H,)
        h_exp = h.unsqueeze(0).expand(candidates.size(0), -1)
        scores = self.head(torch.cat([h_exp, candidates], dim=-1)).squeeze(-1)
        return scores, h_out, c_out


class OnnxTransformerWrapper(nn.Module):
    """
    Single-head Transformer full-sequence wrapper for ONNX export (DQN / SAC actor).
    Inputs:  sequence (1,T,F), candidates (N,F)
    Outputs: scores (N,)
    """
    def __init__(self, net: "TransformerNet"):
        super().__init__()
        self.input_proj = net.input_proj
        self.transformer = net.transformer
        self.head = net.head

    def forward(self, sequence, candidates):
        x     = self.input_proj(sequence)                          # (1,T,d_model)
        h_all = self.transformer(x)                                # (1,T,d_model)
        h     = h_all.squeeze(0)[-1]                               # (d_model,)
        h_exp = h.unsqueeze(0).expand(candidates.size(0), -1)
        scores = self.head(torch.cat([h_exp, candidates], dim=-1)).squeeze(-1)
        return scores


class OnnxLSTMActorWrapper(nn.Module):
    """
    Actor-only LSTM step wrapper for ONNX export (PPO).
    Inputs:  prev_feat (1,1,F), h_in (1,1,H), c_in (1,1,H), candidates (N,F)
    Outputs: scores (N,), h_out (1,1,H), c_out (1,1,H)
    """
    def __init__(self, net: "LSTMActorCritic"):
        super().__init__()
        self.lstm       = net.lstm
        self.actor_head = net.actor_head

    def forward(self, prev_feat, h_in, c_in, candidates):
        out, (h_out, c_out) = self.lstm(prev_feat, (h_in, c_in))
        h = out.squeeze(0).squeeze(0)                               # (H,)
        h_exp = h.unsqueeze(0).expand(candidates.size(0), -1)
        scores = self.actor_head(torch.cat([h_exp, candidates], dim=-1)).squeeze(-1)
        return scores, h_out, c_out


class OnnxTransformerActorWrapper(nn.Module):
    """
    Actor-only Transformer full-sequence wrapper for ONNX export (PPO).
    Inputs:  sequence (1,T,F), candidates (N,F)
    Outputs: scores (N,)
    """
    def __init__(self, net: "TransformerActorCritic"):
        super().__init__()
        self.input_proj  = net.input_proj
        self.transformer = net.transformer
        self.actor_head  = net.actor_head

    def forward(self, sequence, candidates):
        x     = self.input_proj(sequence)
        h_all = self.transformer(x)
        h     = h_all.squeeze(0)[-1]                               # (d_model,)
        h_exp = h.unsqueeze(0).expand(candidates.size(0), -1)
        scores = self.actor_head(torch.cat([h_exp, candidates], dim=-1)).squeeze(-1)
        return scores


# ── replay buffers ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Flat replay buffer for DQN: stores (feat_chosen, reward, next_feats, done)."""

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


class SACReplayBuffer:
    """Flat replay buffer for SAC: stores (curr_feats, action, reward, next_feats, done)."""

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


class RolloutBuffer:
    """
    Stores one episode of transitions for the PPO update.
    prev_chosen_feats stores the feature of the previously expanded transition
    (zero vector at episode start). Used in LSTM/Transformer mode during update.
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
