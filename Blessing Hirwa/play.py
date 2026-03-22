#!/usr/bin/env python
# play.py - Load trained DQN and run greedy evaluation on ALE/Seaquest-v5.
# Usage:
#     python play.py                              # 5 episodes with GUI
#     python play.py --model dqn_model.zip --episodes 10
#     python play.py --no_render                  # headless / server

import argparse
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

gym.register_envs(ale_py)


def make_eval_env(render_mode='human'):
    def _init():
        env = gym.make('ALE/Seaquest-v5', render_mode=render_mode)
        env = AtariWrapper(env)
        return env
    vec_env = DummyVecEnv([_init])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate DQN on ALE/Seaquest-v5')
    p.add_argument('--model',     type=str, default='dqn_model.zip', help='Path to saved model')
    p.add_argument('--episodes',  type=int, default=5,               help='Number of episodes')
    p.add_argument('--no_render', action='store_true',               help='Disable GUI rendering')
    return p.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.no_render else 'human'

    print(f'Loading model: {args.model}')
    model = DQN.load(args.model)

    env = make_eval_env(render_mode=render_mode)
    episode_rewards = []

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # deterministic=True -> greedy (argmax Q-value)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            done = done[0]

        episode_rewards.append(total_reward)
        print(f'  Episode {ep:>2} | Reward: {total_reward:.2f}')

    env.close()
    print(f'\n--- Summary ({args.episodes} episodes) ---')
    print(f'  Mean: {np.mean(episode_rewards):.2f}')
    print(f'  Max:  {np.max(episode_rewards):.2f}')
    print(f'  Min:  {np.min(episode_rewards):.2f}')


if __name__ == '__main__':
    main()
