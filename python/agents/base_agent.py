"""
Base agent with attributes shared by DQN, PPO, and SAC.
"""


class BaseAgent:
    """
    Provides epsilon-greedy decay, training state flag, and save-frequency config.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        save_frequency: int = 10,
    ):
        self.epsilon:        float = epsilon_start
        self.epsilon_end:    float = epsilon_end
        self.epsilon_decay:  float = epsilon_decay
        self.save_frequency: int   = save_frequency
        self.trained:        bool  = False

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
