import argparse
import os
import torch
from datetime import datetime

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback

# GPU check
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*60}")
print(f"  train_extended.py — Extended GPU Training")
print(f"  Device : {device.upper()}")
if device == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  WARNING: CUDA not found — training on CPU (slower)")
    print("  See GPU SETUP section at bottom of this file")
print(f"{'='*60}\n")

EXPERIMENTS = {
    1: dict(
        name="exp1_baseline_low",
        learning_rate=0.00001, gamma=0.90, batch_size=16,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.10,
        default_timesteps=1_000_000,
        notes="Baseline low — very slow learning, stable",
    ),
    2: dict(
        name="exp2_lr_bump",
        learning_rate=0.00002, gamma=0.91, batch_size=16,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.10,
        default_timesteps=1_000_000,
        notes="Slight lr bump — marginally faster convergence",
    ),
    3: dict(
        name="exp3_bigger_batch",
        learning_rate=0.00003, gamma=0.90, batch_size=32,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.10,
        default_timesteps=1_000_000,
        notes="Bigger batch — more stable gradient updates",
    ),
    4: dict(
        name="exp4_lower_eps_start",
        learning_rate=0.00001, gamma=0.92, batch_size=16,
        exploration_initial_eps=0.8, exploration_final_eps=0.05,
        exploration_fraction=0.10,
        default_timesteps=1_000_000,
        notes="Lower eps_start — less early exploration",
    ),
    5: dict(
        name="exp5_lower_eps_end",
        learning_rate=0.00005, gamma=0.91, batch_size=32,
        exploration_initial_eps=1.0, exploration_final_eps=0.01,
        exploration_fraction=0.10,
        default_timesteps=10_000_000,
        notes="Lower eps_end — more exploitation late game [EXTENDED: 10M]",
    ),
    6: dict(
        name="exp6_faster_decay",
        learning_rate=0.00001, gamma=0.93, batch_size=16,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.05,
        default_timesteps=10_000_000,
        notes="Faster decay — exploitation kicks in early [EXTENDED: 10M]",
    ),
    7: dict(
        name="exp7_slower_decay",
        learning_rate=0.00002, gamma=0.92, batch_size=32,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.20,
        default_timesteps=10_000_000,
        notes="Slower decay — longer exploration phase [EXTENDED: 10M]",
    ),
    8: dict(
        name="exp8_mixed_low",
        learning_rate=0.00003, gamma=0.93, batch_size=16,
        exploration_initial_eps=0.9, exploration_final_eps=0.01,
        exploration_fraction=0.10,
        default_timesteps=10_000_000,
        notes="Mixed low — observe combined effect [EXTENDED: 10M]",
    ),
    9: dict(
        name="exp9_moderate_decay",
        learning_rate=0.00004, gamma=0.90, batch_size=32,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.15,
        default_timesteps=10_000_000,
        notes="Moderate decay — balanced explore/exploit [EXTENDED: 10M]",
    ),
    10: dict(
        name="exp10_best_of_low",
        learning_rate=0.00005, gamma=0.93, batch_size=32,
        exploration_initial_eps=0.8, exploration_final_eps=0.01,
        exploration_fraction=0.10,
        default_timesteps=10_000_000,
        notes="Best-of-low — final low-range candidate [EXTENDED: 10M]",
    ),
}

ENV_ID          = "ALE/Seaquest-v5"
BUFFER_SIZE     = 50_000
LEARNING_STARTS = 10_000
TARGET_UPDATE   = 1_000
N_STACK         = 4

# Wrwite summary
RESULTS_FILE = "results_extended.txt"


def make_env(render_mode=None):
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env = AtariWrapper(env)
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)
    return vec_env


def log_result(line):
    """Append a line to the results summary file and print it."""
    print(line)
    with open(RESULTS_FILE, "a") as f:
        f.write(line + "\n")


def run_experiment(exp_id: int, total_timesteps: int = None):
    cfg   = EXPERIMENTS[exp_id]
    steps = total_timesteps if total_timesteps is not None else cfg["default_timesteps"]
    start = datetime.now()

    print(f"\n{'='*60}")
    print(f"  Experiment {exp_id:02d}: {cfg['name']}")
    print(f"  Notes      : {cfg['notes']}")
    print(f"  Device     : {device.upper()}")
    print(f"  Steps      : {steps:,}")
    print(f"  lr={cfg['learning_rate']}  gamma={cfg['gamma']}  "
          f"batch={cfg['batch_size']}  "
          f"eps_start={cfg['exploration_initial_eps']}  "
          f"eps_end={cfg['exploration_final_eps']}  "
          f"eps_frac={cfg['exploration_fraction']}")
    print(f"{'='*60}")
# extended train and save

    run_dir  = os.path.join("runs", cfg["name"] + "_extended")
    log_dir  = os.path.join(run_dir, "tensorboard")
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(run_dir,  exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    train_env = make_env()
    eval_env  = make_env()

    model = DQN(
        policy                  = "CnnPolicy",
        env                     = train_env,
        learning_rate           = cfg["learning_rate"],
        gamma                   = cfg["gamma"],
        batch_size              = cfg["batch_size"],
        exploration_initial_eps = cfg["exploration_initial_eps"],
        exploration_final_eps   = cfg["exploration_final_eps"],
        exploration_fraction    = cfg["exploration_fraction"],
        buffer_size             = BUFFER_SIZE,
        learning_starts         = LEARNING_STARTS,
        target_update_interval  = TARGET_UPDATE,
        tensorboard_log         = log_dir,
        verbose                 = 1,
        device = device,
    )

    eval_freq = 200_000 if steps >= 10_000_000 else 50_000

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = eval_dir,
        log_path             = eval_dir,
        eval_freq            = eval_freq,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
    )

    model.learn(
        total_timesteps     = steps,
        callback            = eval_callback,
        tb_log_name         = cfg["name"] + "_extended",
        reset_num_timesteps = True,
    )

    # Save final model
    model_path = os.path.join(run_dir, "dqn_model")
    model.save(model_path)

    best_reward = _read_best_reward(eval_dir)
    elapsed     = datetime.now() - start

    train_env.close()
    eval_env.close()

    log_result(
        f"Exp {exp_id:02d} | {cfg['name']:<30} | "
        f"steps={steps:>9,} | "
        f"best_reward={best_reward:>8} | "
        f"device={device} | "
        f"time={str(elapsed).split('.')[0]}"
    )

    return model_path + ".zip", best_reward


def _read_best_reward(eval_dir):
    """
    Try to read the best mean reward recorded by EvalCallback.
    EvalCallback saves evaluations.npz in the eval directory.
    """
    try:
        import numpy as np
        npz_path = os.path.join(eval_dir, "evaluations.npz")
        if os.path.exists(npz_path):
            data    = np.load(npz_path)
            results = data["results"]
            means   = results.mean(axis=1)
            return round(float(means.max()), 2)
    except Exception:
        pass
    return "N/A"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extended GPU training — experiments 5-10 at 5M steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_extended.py --high                        retrain exp 5-10 (10M steps)
  python train_extended.py --exp 10                      single experiment
  python train_extended.py --exp 10 --timesteps 10000000 custom steps
  python train_extended.py --all                         all 10 experiments
  tensorboard --logdir runs/                             monitor training
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--exp", type=int, choices=list(EXPERIMENTS.keys()),
        metavar="N", help="Run a single experiment (1-10)"
    )
    group.add_argument(
        "--high", action="store_true",
        help="Run experiments 5-10 only (10M steps each)"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Run all 10 experiments (1-4 at 1M, 5-10 at 5M steps)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override timesteps (default: per-experiment setting)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    header = (
        f"\n{'='*80}\n"
        f"  train_extended.py — Results\n"
        f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Device  : {device.upper()}"
        + (f" — {torch.cuda.get_device_name(0)}" if device == "cuda" else "") + "\n"
        f"  Game    : ALE/Seaquest-v5\n"
        f"{'='*80}\n"
        f"{'Exp':<6} | {'Name':<30} | {'Steps':>12} | {'Best Reward':>12} | "
        f"{'Device':<6} | {'Time'}\n"
        f"{'-'*80}"
    )
    print(header)
    with open(RESULTS_FILE, "a") as f:
        f.write(header + "\n")

    results = {}

    if args.high:
        print("Running experiments 5-10 with 10M steps each...\n")
        for exp_id in range(5, 11):
            path, reward = run_experiment(exp_id, total_timesteps=args.timesteps)
            results[exp_id] = {"path": path, "reward": reward}

    elif args.all:
        for exp_id in EXPERIMENTS:
            path, reward = run_experiment(exp_id, total_timesteps=args.timesteps)
            results[exp_id] = {"path": path, "reward": reward}

    else:
        path, reward = run_experiment(args.exp, total_timesteps=args.timesteps)
        results[args.exp] = {"path": path, "reward": reward}

    # Final summary
    footer = f"\n{'='*80}\n  SUMMARY\n{'='*80}"
    log_result(footer)
    for exp_id, r in results.items():
        log_result(f"  Exp {exp_id:02d}: best_reward={r['reward']}  model={r['path']}")

    log_result(
        f"\n  Results saved to: {RESULTS_FILE}\n"
        f"  TensorBoard    : tensorboard --logdir runs/\n"
        f"{'='*80}\n"
    )


if __name__ == "__main__":
    main()
