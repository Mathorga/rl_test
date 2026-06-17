import gymnasium as gym

def run():
    env: gym.Env = gym.make(
        "CliffWalking-v1",
        render_mode = "human"
    )

    obs, info = env.reset()
    done: bool = False

    while not done:
        action = env.action_space.sample()
        (obs, reward, terminated, truncated, info) = env.step(action)
        done = terminated or truncated

if __name__ == "__main__":
    run()