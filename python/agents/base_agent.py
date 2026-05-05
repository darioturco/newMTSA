"""
Base agent with attributes shared by DQN, PPO, and SAC.
"""
from typing import Optional, Union


class BaseAgent:
    """
    Provides epsilon-greedy decay, training state flag, and save-frequency config.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: Union[float, str] = 0.997,
        save_frequency: int = 10,
    ):
        self.epsilon:        float          = epsilon_start
        self.epsilon_end:    float          = epsilon_end
        self.epsilon_decay:  Optional[float] = (
            None if isinstance(epsilon_decay, str) else float(epsilon_decay)
        )
        self.save_frequency: int            = save_frequency
        self.trained:        bool           = False

    def decay_epsilon(self) -> None:
        if self.epsilon_decay is None:
            return
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
