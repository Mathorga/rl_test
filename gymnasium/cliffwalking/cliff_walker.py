import numpy as np
import gymnasium as gym

class CliffWalker:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.001,
        starting_epsilon: float = 1.0,
        min_epsilon: float = 0.0,
        epsilon_decay: float = 0.001,
        discount_factor: float = 0.95
    ) -> None:
        """
        Inizialize a q-learning epsilon-greedy agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update q-values (0.0-1.0)
            starting_epsilon: Starting exploration rate, usually 1.0 (all exploration)
            min_epsilon: Minimum exploration rate, usually 0.1 (10% exploration, 90% exploitation)
            epsilon_decay: How much to reduce epsilon each episode
            discount_factor: How much to value future rewards (0.0-1.0)
        """

        self.env: gym.Env = env

        self.q_table: dict
