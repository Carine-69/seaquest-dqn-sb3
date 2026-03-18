import argparse
import os

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# ──────────────────────────────────────────────────────────────
Mid-Range Hyperparameter Experiments (CnnPolicy)
# ──────────────────────────────────────────────────────────────
#
# Why CnnPolicy?
#   Seaquest observations are raw RGB pixels (210x160x3).
#   A CNN automatically extracts spatial features (submarine position,
#   divers, enemies) from the pixel grid — something an MLP treating
#   each pixel as an independent feature cannot do efficiently.
#
# Parameter ranges (mid):
#   learning_rate          : 0.0001 – 0.0005
#   gamma                  : 0.94 – 0.96
#   batch_size             : 64 (fixed)
#   exploration_initial_eps: 0.9 – 1.0
#   exploration_final_eps  : 0.02 – 0.05
#   exploration_fraction   : 0.05 – 0.25
# ──────────────────────────────────────────────────────────────

EXPERIMENTS = {
    1: dict(
        name="exp1_baseline_mid",
        learning_rate=0.0001, gamma=0.94, batch_size=64,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.10,
        notes="Baseline mid — good balance",
    ),
    2: dict(
        name="exp2_lr_bump",
        learning_rate=0.0002, gamma=0.94, batch_size=64,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.10,
        notes="Slight lr bump — faster policy updates",
    ),
    3: dict(
        name="exp3_higher_gamma",
        learning_rate=0.0001, gamma=0.95, batch_size=64,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.10,
        notes="Higher gamma — values future rewards more",
    ),
    4: dict(
        name="exp4_lower_eps_start",
        learning_rate=0.0003, gamma=0.94, batch_size=64,
        exploration_initial_eps=0.9, exploration_final_eps=0.05,
        exploration_fraction=0.10,
        notes="Lower eps_start — less initial randomness",
    ),
    5: dict(
        name="exp5_lower_eps_end",
        learning_rate=0.0001, gamma=0.96, batch_size=64,
        exploration_initial_eps=1.0, exploration_final_eps=0.02,
        exploration_fraction=0.10,
        notes="Lower eps_end — more greedy late stage",
    ),
    6: dict(
        name="exp6_faster_decay",
        learning_rate=0.0002, gamma=0.95, batch_size=64,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.05,
        notes="Faster decay — earlier shift to exploit",
    ),
    7: dict(
        name="exp7_slower_decay",
        learning_rate=0.0001, gamma=0.94, batch_size=64,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.25,
        notes="Slower decay — prolonged exploration",
    ),
    8: dict(
        name="exp8_mixed_mid",
        learning_rate=0.0004, gamma=0.95, batch_size=64,
        exploration_initial_eps=0.9, exploration_final_eps=0.02,
        exploration_fraction=0.10,
        notes="Mixed mid — combined moderate changes",
    ),
    9: dict(
        name="exp9_moderate_decay_high_gamma",
        learning_rate=0.0003, gamma=0.96, batch_size=64,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.15,
        notes="Moderate decay with higher gamma",
    ),
    10: dict(
        name="exp10_best_of_mid",
        learning_rate=0.0005, gamma=0.95, batch_size=64,
        exploration_initial_eps=0.9, exploration_final_eps=0.02,
        exploration_fraction=0.20,
        notes="Best-of-mid — final mid-range candidate",
    ),
}

ENV_ID          = "ALE/Seaquest-v5"
TOTAL_TIMESTEPS = 1_000_000
BUFFER_SIZE     = 100_000
LEARNING_STARTS = 10_000
TARGET_UPDATE   = 1_000
N_STACK         = 4


def make_env(render_mode=None):
    """
    Build a single Seaquest environment with standard Atari pre-processing:
      - AtariWrapper  : grayscale, resize to 84x84, frame skip (4), clip rewards
      - VecFrameStack : stack 4 consecutive frames so the agent perceives motion
    """
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env = AtariWrapper(env)
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)
    return vec_env


def run_experiment(exp_id: int, total_timesteps: int = TOTAL_TIMESTEPS):
    cfg = EXPERIMENTS[exp_id]
    print(f"\n{'='*60}")
    print(f"  Experiment {exp_id:02d}: {cfg['name']}")
    print(f"  Notes      : {cfg['notes']}")
    print(f"  lr={cfg['learning_rate']}  gamma={cfg['gamma']}  "
          f"batch={cfg['batch_size']}  eps_start={cfg['exploration_initial_eps']}  "
          f"eps_end={cfg['exploration_final_eps']}  "
          f"eps_frac={cfg['exploration_fraction']}")
    print(f"{'='*60}")

    run_dir  = os.path.join("runs", cfg["name"])
    log_dir  = os.path.join(run_dir, "tensorboard")
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(run_dir,  exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    train_env = make_env()
    eval_env  = make_env()

    model = DQN(
        policy                 = "CnnPolicy",
        env                    = train_env,
        learning_rate          = cfg["learning_rate"],
        gamma                  = cfg["gamma"],
        batch_size             = cfg["batch_size"],
        exploration_initial_eps= cfg["exploration_initial_eps"],
        exploration_final_eps  = cfg["exploration_final_eps"],
        exploration_fraction   = cfg["exploration_fraction"],
        buffer_size            = BUFFER_SIZE,
        learning_starts        = LEARNING_STARTS,
        target_update_interval = TARGET_UPDATE,
        tensorboard_log        = log_dir,
        verbose                = 1,
        optimize_memory_usage  = False,
    )

    # Stop early if eval reward doesn't improve for 3 consecutive evaluations
    early_stop = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=3,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = eval_dir,
        log_path             = eval_dir,
        eval_freq            = 50_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
        callback_after_eval  = early_stop,
    )

    model.learn(
        total_timesteps    = total_timesteps,
        callback           = eval_callback,
        tb_log_name        = cfg["name"],
        reset_num_timesteps= True,
    )

    model_path = os.path.join(run_dir, "dqn_model")
    model.save(model_path)
    print(f"\n  Model saved  ->  {model_path}.zip")

    train_env.close()
    eval_env.close()

    return model_path + ".zip"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DQN (CnnPolicy) on ALE/Seaquest-v5 — Member 2 Mid-Range"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--exp", type=int, choices=list(EXPERIMENTS.keys()),
        metavar="N", help="Run a single experiment (1-10)"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Run all 10 experiments sequentially"
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        results = {}
        for exp_id in EXPERIMENTS:
            path = run_experiment(exp_id, total_timesteps=args.timesteps)
            results[exp_id] = path

        print("\n\n" + "=" * 60)
        print("  All experiments complete. Saved models:")
        for exp_id, path in results.items():
            print(f"    Exp {exp_id:02d}: {path}")
        print("=" * 60)
    else:
        run_experiment(args.exp, total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
