#!/usr/bin/env python
# train.py - Member 3: DQN MlpPolicy on ALE/Seaquest-v5
# Usage:
#     python train.py                    # defaults (Exp 10 best-of-high)
#     python train.py --exp_id 5         # run a preset experiment config
#     python train.py --lr 0.001 --gamma 0.98 --batch_size 128 --timesteps 200000

import os
import argparse
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

gym.register_envs(ale_py)

# ---- Member 3 High-Range Preset Configs ------------------------------------
PRESETS = {
    1:  dict(lr=0.001, gamma=0.97, batch_size=128, eps_start=1.0, eps_end=0.05, eps_fraction=0.10),
    2:  dict(lr=0.002, gamma=0.97, batch_size=128, eps_start=1.0, eps_end=0.05, eps_fraction=0.10),
    3:  dict(lr=0.001, gamma=0.98, batch_size=128, eps_start=1.0, eps_end=0.05, eps_fraction=0.10),
    4:  dict(lr=0.003, gamma=0.97, batch_size=256, eps_start=1.0, eps_end=0.05, eps_fraction=0.10),
    5:  dict(lr=0.001, gamma=0.99, batch_size=128, eps_start=1.0, eps_end=0.05, eps_fraction=0.10),
    6:  dict(lr=0.002, gamma=0.98, batch_size=128, eps_start=0.9, eps_end=0.05, eps_fraction=0.10),
    7:  dict(lr=0.001, gamma=0.97, batch_size=256, eps_start=1.0, eps_end=0.01, eps_fraction=0.10),
    8:  dict(lr=0.003, gamma=0.98, batch_size=128, eps_start=1.0, eps_end=0.05, eps_fraction=0.05),
    9:  dict(lr=0.005, gamma=0.97, batch_size=256, eps_start=1.0, eps_end=0.05, eps_fraction=0.25),
    10: dict(lr=0.005, gamma=0.99, batch_size=256, eps_start=0.9, eps_end=0.01, eps_fraction=0.20),
}


def make_env():
    def _init():
        env = gym.make('ALE/Seaquest-v5')
        env = AtariWrapper(env)
        return env
    vec_env = DummyVecEnv([_init])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.reward_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0 and len(self.model.ep_info_buffer) > 0:
            mean_r = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            self.reward_history.append(float(mean_r))
            print(f'  [{self.n_calls:>7,} steps] Mean reward: {mean_r:.2f}')
        return True


def parse_args():
    p = argparse.ArgumentParser(description='Train DQN MlpPolicy on ALE/Seaquest-v5')
    p.add_argument('--exp_id',       type=int,   default=None,    help='Preset experiment 1-10')
    p.add_argument('--lr',           type=float, default=0.005,   help='Learning rate')
    p.add_argument('--gamma',        type=float, default=0.99,    help='Discount factor')
    p.add_argument('--batch_size',   type=int,   default=256,     help='Batch size')
    p.add_argument('--eps_start',    type=float, default=0.9,     help='Initial epsilon')
    p.add_argument('--eps_end',      type=float, default=0.01,    help='Final epsilon')
    p.add_argument('--eps_fraction', type=float, default=0.20,    help='Fraction of steps for eps decay')
    p.add_argument('--timesteps',    type=int,   default=200_000, help='Total training timesteps')
    p.add_argument('--save_path',    type=str,   default='dqn_model', help='Model save path (no .zip)')
    p.add_argument('--tb_log',       type=str,   default='./tb_logs', help='TensorBoard log dir')
    return p.parse_args()


def main():
    args = parse_args()

    if args.exp_id is not None:
        cfg = PRESETS[args.exp_id]
        args.lr           = cfg['lr']
        args.gamma        = cfg['gamma']
        args.batch_size   = cfg['batch_size']
        args.eps_start    = cfg['eps_start']
        args.eps_end      = cfg['eps_end']
        args.eps_fraction = cfg['eps_fraction']
        print(f'Loaded preset Experiment {args.exp_id}')

    print(f'\nHyperparameters:')
    print(f'  Policy: MlpPolicy')
    print(f'  lr={args.lr}  gamma={args.gamma}  batch={args.batch_size}')
    print(f'  eps: {args.eps_start} -> {args.eps_end} over {args.eps_fraction*100:.0f}% of {args.timesteps:,} steps\n')

    env = make_env()
    callback = RewardLoggerCallback(log_freq=10_000, verbose=1)

    model = DQN(
        policy='MlpPolicy',
        env=env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_initial_eps=args.eps_start,
        exploration_final_eps=args.eps_end,
        exploration_fraction=args.eps_fraction,
        buffer_size=100_000,
        learning_starts=10_000,
        target_update_interval=1_000,
        train_freq=4,
        tensorboard_log=args.tb_log,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    model.save(args.save_path)
    env.close()
    print(f'\nTraining complete. Model saved -> {args.save_path}.zip')


if __name__ == '__main__':
    main()
