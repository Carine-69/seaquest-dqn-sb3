import argparse
import os

import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ──────────────────────────────────────────────────────────────
# Loads a trained DQN model and runs it on ALE/Seaquest-v5
# with deterministic=True (greedy policy — no exploration).
# Renders the game in a GUI window so you can watch the agent.
#
# HOW THE AGENT BEHAVES (based on training at 50k steps):
#   - Agent mostly fires randomly and moves without clear strategy
#   - Dies quickly from oxygen running out (hasn't learned to surface)
#   - Expected reward per episode: ~2–17 (very low, agent is still learning)
#
# EXPECTED vs ACTUAL:
#   - Expected (fully trained): 500–2000 reward, surfaces for oxygen,
#     shoots enemies, rescues divers
#   - Actual (50k steps): 2–17 reward, dies within seconds from oxygen
#
# AGENT DIES WHEN:
#   - Oxygen bar runs out (must surface to refill) — most common at 50k steps
#   - Hit by an enemy torpedo
#   - Direct collision with enemy submarine or shark
# ──────────────────────────────────────────────────────────────

ENV_ID  = "ALE/Seaquest-v5"
N_STACK = 4


def make_env(render_mode="human"):
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env = AtariWrapper(env)
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)
    return vec_env


def play(model_path: str, n_episodes: int = 5):
    """
    Load the trained model, run n_episodes with greedy actions,
    print per-episode reward and a summary at the end.
    """
    print(f"\nLoading model from: {model_path}")
    model = DQN.load(model_path)

    env = make_env(render_mode="human")

    episode_rewards = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # deterministic=True -> greedy policy (no epsilon exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

        episode_rewards.append(total_reward)
        print(f"  Episode {ep}/{n_episodes}  |  Reward: {total_reward:.0f}")

    env.close()

    # ── Summary ──
    rewards = np.array(episode_rewards)
    print(f"\n{'='*45}")
    print(f"  Episodes played : {n_episodes}")
    print(f"  Mean reward     : {rewards.mean():.1f}")
    print(f"  Min  reward     : {rewards.min():.1f}")
    print(f"  Max  reward     : {rewards.max():.1f}")
    print(f"  Std  reward     : {rewards.std():.1f}")
    print(f"{'='*45}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play ALE/Seaquest-v5 with a trained DQN model — Member 2"
    )
    parser.add_argument(
        "--model", type=str, default="dqn_model.zip",
        help="Path to trained model .zip file (default: dqn_model.zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes (default: 5)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("  Train a model first with train.py, or pass --model <path>")
        return

    play(args.model, n_episodes=args.episodes)


if __name__ == "__main__":
    main()
