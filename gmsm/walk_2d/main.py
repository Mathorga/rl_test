import sys
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from walker_2d import Walker2D

policy_file_path: str =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pkl")

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def plot_training(
    env: gym.Env,
    agent: Walker2D
) -> None:
    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()

def train(
    env: gym.Env,
    episodes_count: int = 1000
):
    global policy_file_path

    # Create the environment.
    env = gym.wrappers.RecordEpisodeStatistics(
        env,
        buffer_length = episodes_count
    )

    starting_epsilon: float = 1.0

    # Create the agent living in the environment.
    agent: Walker2D = Walker2D(
        env = env,
        learning_rate = 0.001,
        starting_epsilon = starting_epsilon,
        min_epsilon = 0.1,
        epsilon_decay = starting_epsilon / (episodes_count / 2)
    )

    for episode in tqdm(range(episodes_count)):
        # Start a new run.
        (obs, info) = env.reset()
        done: bool = False

        while not done:
            # Let the agent pick an action.
            action: int = agent.get_action(obs = obs)

            # Apply the action to the environment and retrieve the next observations from it.
            (next_obs, reward, terminated, truncated, info) = env.step(action)

            # Learn from the obtained experience.
            agent.update(
                obs = obs,
                action = action,
                reward = reward,
                terminated = terminated,
                next_obs = next_obs
            )

            # Prepare for the next step.
            done = terminated or truncated
            obs = next_obs

        # Reduce the exploration rate (agent becomes less random over time).
        agent.decay_epsilon()

    # Store trained q-table.
    f = open(policy_file_path,"wb")
    pickle.dump(agent.q_table, f)
    f.close()

    print(agent.q_table)

    plot_training(
        env = env,
        agent = agent
    )

def run(env: gym.Env):
    global policy_file_path

    # Create the agent living in the environment.
    agent: Walker2D = Walker2D(
        env = env,
        policy_file_path = policy_file_path,
        # Make sure no exploration is done.
        starting_epsilon = 0.0,
        min_epsilon = 0.0
    )

    # Reset the environment before starting training the agent.
    (obs, info) = env.reset()
    done: bool = False

    while not done:
        action = agent.get_action(obs = obs)
        (obs, reward, terminated, truncated, info) = env.step(action)
        done = terminated or truncated

if __name__ == "__main__":
    env_id: str = "CliffWalking-v1"
    # env_id: str = "FrozenLake-v1"
    run_env: gym.Env = gym.make(
        env_id,
        render_mode = "human",
        is_slippery = False
    )
    train_env: gym.Env = gym.make(
        env_id,
        is_slippery = False
    )
    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "train":
                if len(sys.argv) >= 3:
                    train(env = train_env, episodes_count= int(sys.argv[2]))
                else:
                    train(env = train_env, episodes_count = 1000)
            case "run":
                run(env = run_env)
            case _:
                run(env = run_env)
    else:
        run(env = run_env)