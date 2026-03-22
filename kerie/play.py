import argparse
import os

import numpy as np
import gymnasium as gym
import ale_py
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

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


def play(model_path: str, n_episodes: int = 5, save_gif: bool = False, gif_path: str = "gameplay.gif"):
    print(f"\nLoading model from: {model_path}")
    model = DQN.load(model_path)

    # Use rgb_array to capture frames for GIF, human to show GUI
    render_mode = "rgb_array" if save_gif else "human"
    env = make_env(render_mode=render_mode)

    episode_rewards = []
    frames = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # deterministic=True -> greedy policy (no epsilon exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

            if save_gif:
                frame = env.render()
                # Take every 3rd frame to keep GIF size reasonable
                if len(frames) % 3 == 0:
                    frames.append(Image.fromarray(frame))

        episode_rewards.append(total_reward)
        print(f"  Episode {ep}/{n_episodes}  |  Reward: {total_reward:.0f}")

    env.close()

    # Save GIF
    if save_gif and frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # 50ms per frame = 20fps
            loop=0,
        )
        print(f"\n  GIF saved -> {gif_path} ({len(frames)} frames)")

    # Summary
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
    parser.add_argument(
        "--save-gif", action="store_true",
        help="Save gameplay as GIF instead of showing GUI window"
    )
    parser.add_argument(
        "--gif-path", type=str, default="gameplay.gif",
        help="Output path for GIF (default: gameplay.gif)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("  Train a model first with train.py, or pass --model <path>")
        return

    play(args.model, n_episodes=args.episodes,
         save_gif=args.save_gif, gif_path=args.gif_path)


if __name__ == "__main__":
    main()
