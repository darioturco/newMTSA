import random


def run_episode(env, fsp_path: str) -> dict:
    """Run one episode with a uniformly-random action policy. Returns info dict."""
    frontier = env.reset(fsp_path)
    total_reward = 0
    steps = 0
    info = {}

    while not env._done:
        if not frontier:
            break
        action = random.randrange(len(frontier))
        frontier, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    info["steps"] = steps
    info["total_reward"] = total_reward
    return info
