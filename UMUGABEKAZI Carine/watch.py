
#to watch seaquest in play
# import argparse
import os
import numpy as np
import imageio
import argparse
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

DEFAULT_MODEL = "runs/exp10_best_of_low_extended/eval/best_model.zip"
ENV_ID        = "ALE/Seaquest-v5"
N_STACK       = 4


def make_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = AtariWrapper(env)
        return env
    vec = DummyVecEnv([_init])
    vec = VecFrameStack(vec, n_stack=N_STACK)
    return vec


def record(model_path, n_episodes=5):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    video_path = model_path.replace(".zip", "_gameplay.mp4")
    print(f"\n  Model   : {model_path}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Output  : {video_path}\n")

    env    = make_env()
    model  = DQN.load(model_path, env=env)
    scores = []
    frames = []

    for ep in range(n_episodes):
        obs   = env.reset()
        done  = False
        score = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            score += float(reward[0])
            steps += 1

            frame = env.render()
            if frame is not None:
                if isinstance(frame, (list, tuple)):
                    frame = frame[0]
                frames.append(frame.copy())

            if done[0]:
                break

        scores.append(score)
        print(f"  Episode {ep+1:02d}: score={score:.1f}  steps={steps}")

    env.close()

    print(f"\n  Writing {len(frames)} frames to video...")
    imageio.mimwrite(video_path, frames, fps=15, quality=8)
    print(f"  Saved -> {video_path}")
    print(f"\n  Open with:")
    print(f"    vlc {video_path}")
    print(f"    xdg-open {video_path}")
    print(f"    mpv {video_path}")
    print(f"\n  Results:")
    print(f"  Mean  : {np.mean(scores):.2f}")
    print(f"  Best  : {np.max(scores):.2f}")
    print(f"  Worst : {np.min(scores):.2f}")


def evaluate(model_path, n_episodes=20):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"\n  Evaluating : {model_path}")
    print(f"  Episodes   : {n_episodes}\n")

    env    = make_env()
    model  = DQN.load(model_path, env=env)
    scores = []

    for ep in range(n_episodes):
        obs   = env.reset()
        done  = False
        score = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            score += float(reward[0])
            if done[0]:
                break
        scores.append(score)
        print(f"  Episode {ep+1:02d}: {score:.1f}")

    env.close()
    print(f"\n  Mean   : {np.mean(scores):.2f}")
    print(f"  Std    : {np.std(scores):.2f}")
    print(f"  Best   : {np.max(scores):.2f}")
    print(f"  Worst  : {np.min(scores):.2f}")
    print(f"  Median : {np.median(scores):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Watch DQN agent play Seaquest")
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval",     action="store_true",
                        help="Just print scores, no video")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args.model, n_episodes=args.episodes)
    else:
        record(args.model, n_episodes=args.episodes)