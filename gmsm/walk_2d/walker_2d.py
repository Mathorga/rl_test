import pickle
import numpy as np
import gymnasium as gym

class Walker2D:
    def __init__(
        self,
        env: gym.Env,
        policy_file_path: str | None = None,
        learning_rate: float = 0.001,
        starting_epsilon: float = 1.0,
        min_epsilon: float = 0.0,
        epsilon_decay: float = 0.001,
        discount_factor: float = 0.95
    ) -> None:
        """
        Inizialize a q-learning epsilon-greedy agent for solving cliffwalker (discrete action space).

        Args:
            env: The training environment
            learning_rate: How quickly to update q-values (0.0-1.0)
            starting_epsilon: Starting exploration rate, usually 1.0 (all exploration)
            min_epsilon: Minimum exploration rate, usually 0.1 (10% exploration, 90% exploitation)
            epsilon_decay: How much to reduce epsilon each episode
            discount_factor: How much to value future rewards (0.0-1.0)
        """

        self.env: gym.Env = env

        # Store observation-action relations in a dictionary.
        # Discrete spaces only.
        self.q_table: np.ndarray
        if policy_file_path is not None:
            f = open(policy_file_path, "rb")
            self.q_table = pickle.load(f)
            f.close()
        else:
            self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        # Store learning rate to use during q-table update.
        self.learning_rate: float = learning_rate

        # Ratio between choice of future rewards vs immediate rewards.
        self.discount_factor: float = discount_factor

        # Epsilon-greedy algorithm parameters.
        self.epsilon: float = starting_epsilon
        self.min_epsilon: float = min_epsilon
        self.epsilon_decay: float = epsilon_decay

        # Keep track of learning progress.
        self.training_error: list[float] = []

    def get_action(
        self,
        obs: list[int]
    ) -> int:
        """
        Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """

        if np.random.random() < self.epsilon:
            # Explore.
            return self.env.action_space.sample()

        else:
            # Exploit.
            return int(np.argmax(self.q_table[obs]))

    def decay_epsilon(self) -> None:
        """
        Reduce exploration rate after each episode.
        """

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon - self.epsilon_decay
        )

    def update(
        self,
        obs: list[int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: list[int]
    ) -> None:
        """
        Update the q_table based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """

        # What's the best we could do from the next observation?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value: int = (not terminated) * np.max(self.q_table[next_obs])

        # What should the q-value be? (Bellman equation).
        target: float = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference: float = target - self.q_table[obs][action]

        # Update our estimate in the direction of the error.
        # Learning rate controls how big steps we take.
        self.q_table[obs][action] += self.learning_rate * temporal_difference

        # Track learning progress.
        self.training_error.append(temporal_difference)