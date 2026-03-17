import os
import csv
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENTS = {
    1:  dict(name="exp1_baseline_low",    lr=0.00001, gamma=0.90, batch=16, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    2:  dict(name="exp2_lr_bump",         lr=0.00002, gamma=0.91, batch=16, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    3:  dict(name="exp3_bigger_batch",    lr=0.00003, gamma=0.90, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    4:  dict(name="exp4_lower_eps_start", lr=0.00001, gamma=0.92, batch=16, eps_start=0.8, eps_end=0.05, eps_frac=0.10),
    5:  dict(name="exp5_lower_eps_end",   lr=0.00005, gamma=0.91, batch=32, eps_start=1.0, eps_end=0.01, eps_frac=0.10),
    6:  dict(name="exp6_faster_decay",    lr=0.00001, gamma=0.93, batch=16, eps_start=1.0, eps_end=0.05, eps_frac=0.05),
    7:  dict(name="exp7_slower_decay",    lr=0.00002, gamma=0.92, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.20),
    8:  dict(name="exp8_mixed_low",       lr=0.00003, gamma=0.93, batch=16, eps_start=0.9, eps_end=0.01, eps_frac=0.10),
    9:  dict(name="exp9_moderate_decay",  lr=0.00004, gamma=0.90, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.15),
    10: dict(name="exp10_best_of_low",    lr=0.00005, gamma=0.93, batch=32, eps_start=0.8, eps_end=0.01, eps_frac=0.10),
}

RUNS_DIR    = "runs"
RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

rows = []
all_eval_means = {}   #
all_timesteps  = {}   

for exp_id, cfg in EXPERIMENTS.items():
    npz_path = os.path.join(RUNS_DIR, cfg["name"], "eval", "evaluations.npz")

    if not os.path.exists(npz_path):
        print(f"  ⚠  Exp {exp_id:02d} — evaluations.npz not found, skipping.")
        continue

    data      = np.load(npz_path)
    timesteps = data["timesteps"]     
    results   = data["results"]         

    eval_means = results.mean(axis=1)

    mean_reward  = round(float(eval_means.mean()), 2)
    best_reward  = round(float(results.max()), 2)
    final_reward = round(float(eval_means[-1]), 2)
    n_evals      = len(timesteps)
    last_step    = int(timesteps[-1])

    rows.append({
        "exp_id":        exp_id,
        "name":          cfg["name"],
        "lr":            cfg["lr"],
        "gamma":         cfg["gamma"],
        "batch":         cfg["batch"],
        "eps_start":     cfg["eps_start"],
        "eps_end":       cfg["eps_end"],
        "eps_frac":      cfg["eps_frac"],
        "mean_reward":   mean_reward,
        "best_reward":   best_reward,
        "final_reward":  final_reward,
        "n_evals":       n_evals,
        "last_timestep": last_step,
    })

    all_eval_means[exp_id] = eval_means
    all_timesteps[exp_id]  = timesteps

    print(f"  ✓  Exp {exp_id:02d} ({cfg['name']})")
    print(f"       mean={mean_reward}  best={best_reward}  final={final_reward}  evals={n_evals}")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(timesteps, eval_means, color="steelblue", linewidth=2, label="Mean Reward")
    ax.fill_between(
        timesteps,
        results.min(axis=1),
        results.max(axis=1),
        alpha=0.2, color="steelblue", label="Min/Max Range"
    )
    ax.axhline(best_reward, color="red", linestyle="--", linewidth=1, label=f"Best={best_reward}")

    ax.set_title(
        f"Exp {exp_id:02d} — {cfg['name']}\n"
        f"lr={cfg['lr']}  gamma={cfg['gamma']}  batch={cfg['batch']}  "
        f"eps_start={cfg['eps_start']}  eps_end={cfg['eps_end']}  eps_frac={cfg['eps_frac']}",
        fontsize=10
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward (5 episodes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(PLOTS_DIR, f"exp{exp_id:02d}_{cfg['name']}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"       plot → {plot_path}")

if rows:
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  ✓  Saved → {OUTPUT_CSV}")

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10.colors
    for exp_id, eval_means in all_eval_means.items():
        cfg   = EXPERIMENTS[exp_id]
        steps = all_timesteps[exp_id]
        ax.plot(
            steps, eval_means,
            color=colors[exp_id - 1],
            linewidth=1.8,
            label=f"Exp {exp_id:02d} (lr={cfg['lr']}, γ={cfg['gamma']}, b={cfg['batch']})"
        )

    ax.set_title("All 10 Experiments — Mean Reward over Timesteps (CNNPolicy)", fontsize=13)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward (5 episodes)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    combined_path = os.path.join(PLOTS_DIR, "all_experiments_combined.png")
    plt.tight_layout()
    plt.savefig(combined_path, dpi=150)
    plt.close()
    print(f"  ✓  Combined plot → {combined_path}")

    fig, ax = plt.subplots(figsize=(12, 6))

    exp_labels    = [f"Exp {r['exp_id']:02d}" for r in rows]
    final_rewards = [r["final_reward"] for r in rows]
    best_rewards  = [r["best_reward"]  for r in rows]

    x     = np.arange(len(rows))
    width = 0.35

    bars1 = ax.bar(x - width/2, final_rewards, width, label="Final Reward", color="steelblue")
    bars2 = ax.bar(x + width/2, best_rewards,  width, label="Best Reward",  color="darkorange")

    ax.set_title("Final vs Best Reward per Experiment (CNNPolicy — Low Range)", fontsize=13)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    bar_path = os.path.join(PLOTS_DIR, "final_vs_best_reward.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"  ✓  Bar chart → {bar_path}")

    # Best experiment summary
    best = max(rows, key=lambda r: r["best_reward"])
    print(f"\n  🏆  Best experiment: Exp {best['exp_id']:02d} ({best['name']})")
    print(f"       best_reward={best['best_reward']}  mean_reward={best['mean_reward']}")

else:
    print("\n  ✗  No results extracted. Check your runs/ folder.")
