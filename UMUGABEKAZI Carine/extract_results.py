import os
import csv
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENTS = {
    1:  dict(name="exp1_baseline_low",             lr=0.00001, gamma=0.90, batch=16, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    2:  dict(name="exp2_lr_bump",                  lr=0.00002, gamma=0.91, batch=16, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    3:  dict(name="exp3_bigger_batch",             lr=0.00003, gamma=0.90, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    4:  dict(name="exp4_lower_eps_start",          lr=0.00001, gamma=0.92, batch=16, eps_start=0.8, eps_end=0.05, eps_frac=0.10),
    5:  dict(name="exp5_lower_eps_end",            lr=0.00005, gamma=0.91, batch=32, eps_start=1.0, eps_end=0.01, eps_frac=0.10),
    6:  dict(name="exp6_faster_decay",             lr=0.00001, gamma=0.93, batch=16, eps_start=1.0, eps_end=0.05, eps_frac=0.05),
    7:  dict(name="exp7_slower_decay",             lr=0.00002, gamma=0.92, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.20),
    8:  dict(name="exp8_mixed_low",                lr=0.00003, gamma=0.93, batch=16, eps_start=0.9, eps_end=0.01, eps_frac=0.10),
    9:  dict(name="exp9_moderate_decay",           lr=0.00004, gamma=0.90, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.15),
    10: dict(name="exp10_best_of_low",             lr=0.00005, gamma=0.93, batch=32, eps_start=0.8, eps_end=0.01, eps_frac=0.10),
    # ── Extended 10M runs ─────────────────────────────────────────────────────
    11: dict(name="exp1_baseline_low_extended",    lr=0.00001, gamma=0.90, batch=16, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    12: dict(name="exp2_lr_bump_extended",         lr=0.00002, gamma=0.91, batch=16, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    13: dict(name="exp3_bigger_batch_extended",    lr=0.00003, gamma=0.90, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    14: dict(name="exp4_lower_eps_start_extended", lr=0.00001, gamma=0.92, batch=16, eps_start=0.8, eps_end=0.05, eps_frac=0.10),
    15: dict(name="exp5_lower_eps_end_extended",   lr=0.00005, gamma=0.91, batch=32, eps_start=1.0, eps_end=0.01, eps_frac=0.10),
    16: dict(name="exp6_faster_decay_extended",    lr=0.00001, gamma=0.93, batch=16, eps_start=1.0, eps_end=0.05, eps_frac=0.05),
    17: dict(name="exp7_slower_decay_extended",    lr=0.00002, gamma=0.92, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.20),
    18: dict(name="exp8_mixed_low_extended",       lr=0.00003, gamma=0.93, batch=16, eps_start=0.9, eps_end=0.01, eps_frac=0.10),
    19: dict(name="exp9_moderate_decay_extended",  lr=0.00004, gamma=0.90, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.15),
    20: dict(name="exp10_best_of_low_extended",    lr=0.00005, gamma=0.93, batch=32, eps_start=0.8, eps_end=0.01, eps_frac=0.10),
}

RUNS_DIR    = "runs"
RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

rows           = []
all_eval_means = {}
all_timesteps  = {}

for exp_id, cfg in EXPERIMENTS.items():

    npz_path = os.path.join(RUNS_DIR, cfg["name"], "eval", "evaluations.npz")

    if not os.path.exists(npz_path):
        print(f"  --  Exp {exp_id:02d} ({cfg['name']}) — not found, skipping.")
        continue

    data         = np.load(npz_path)
    timesteps    = data["timesteps"]
    results      = data["results"]
    eval_means   = results.mean(axis=1)
    mean_reward  = round(float(eval_means.mean()), 2)
    best_reward  = round(float(results.max()), 2)
    final_reward = round(float(eval_means[-1]), 2)
    last_step    = int(timesteps[-1])
    is_extended  = "_extended" in cfg["name"]
    tag          = "extended" if is_extended else "original"

    rows.append({
        "exp_id":        exp_id,
        "run_type":      tag,
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
        "n_evals":       len(timesteps),
        "last_timestep": last_step,
    })

    all_eval_means[exp_id] = eval_means
    all_timesteps[exp_id]  = timesteps

    print(f"  ok  Exp {exp_id:02d} {tag:10s} ({cfg['name']}) — "
          f"mean={mean_reward:7.2f}  best={best_reward:7.2f}  "
          f"final={final_reward:7.2f}  steps={last_step:>10,}")

    # Individual plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, eval_means, color="steelblue", linewidth=2, label="Mean Reward")
    ax.fill_between(timesteps, results.min(axis=1), results.max(axis=1),
                    alpha=0.2, color="steelblue", label="Min/Max Range")
    ax.axhline(best_reward, color="red", linestyle="--",
               linewidth=1, label=f"Best={best_reward}")
    ax.set_title(
        f"Exp {exp_id:02d} ({tag}) — {cfg['name']}\n"
        f"lr={cfg['lr']}  gamma={cfg['gamma']}  batch={cfg['batch']}  "
        f"eps=[{cfg['eps_start']}->{cfg['eps_end']}]  frac={cfg['eps_frac']}  steps={last_step:,}",
        fontsize=9
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward (5 episodes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"exp{exp_id:02d}_{cfg['name']}.png"), dpi=150)
    plt.close()


if rows:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved CSV -> {OUTPUT_CSV}")


if all_eval_means:
    fig, ax = plt.subplots(figsize=(16, 7))
    colors  = plt.cm.tab10.colors   

    for exp_id, cfg in EXPERIMENTS.items():
        if exp_id not in all_eval_means:
            continue
        is_extended = "_extended" in cfg["name"]
        color_idx   = (exp_id - 1) % 10          
        ax.plot(
            all_timesteps[exp_id],
            all_eval_means[exp_id],
            color     = colors[color_idx],
            linewidth = 2.2 if is_extended else 1.5,
            linestyle = "-"  if is_extended else "--",
            label     = f"Exp {exp_id:02d} {'ext' if is_extended else 'orig'} "
                        f"(lr={cfg['lr']}, γ={cfg['gamma']})"
        )

    ax.set_title(
        "All Experiments — Original (dashed) vs Extended (solid)\nCNNPolicy on ALE/Seaquest-v5",
        fontsize=13)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward (5 episodes)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "all_experiments_combined.png"), dpi=150)
    plt.close()
    print(f"  Combined plot -> {PLOTS_DIR}/all_experiments_combined.png")


if rows:
    exp_nums   = list(range(1, 11))
    exp_labels = [f"Exp {i:02d}" for i in exp_nums]

    def get_reward(exp_num, run_type, key):
        target_id = exp_num if run_type == "original" else exp_num + 10
        for r in rows:
            if r["exp_id"] == target_id:
                return r[key]
        return 0

    x     = np.arange(len(exp_nums))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))

    b1 = ax.bar(x - 1.5*width, [get_reward(i,"original","final_reward") for i in exp_nums], width, label="Original final",  color="steelblue",      alpha=0.85)
    b2 = ax.bar(x - 0.5*width, [get_reward(i,"original","best_reward")  for i in exp_nums], width, label="Original best",   color="cornflowerblue", alpha=0.85)
    b3 = ax.bar(x + 0.5*width, [get_reward(i,"extended","final_reward") for i in exp_nums], width, label="Extended final",  color="darkorange",     alpha=0.85)
    b4 = ax.bar(x + 1.5*width, [get_reward(i,"extended","best_reward")  for i in exp_nums], width, label="Extended best",   color="gold",           alpha=0.85)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_title(
        "Original (1M) vs Extended (10M) — Final and Best Reward\nCNNPolicy on ALE/Seaquest-v5",
        fontsize=12)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "original_vs_extended_rewards.png"), dpi=150)
    plt.close()
    print(f"  Bar chart -> {PLOTS_DIR}/original_vs_extended_rewards.png")


if rows:
    best_overall = max(rows, key=lambda r: r["best_reward"])
    orig_rows    = [r for r in rows if r["run_type"] == "original"]
    ext_rows     = [r for r in rows if r["run_type"] == "extended"]
    best_orig    = max(orig_rows, key=lambda r: r["best_reward"]) if orig_rows else None
    best_ext     = max(ext_rows,  key=lambda r: r["best_reward"]) if ext_rows  else None

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    if best_orig:
        print(f"  Best original : {best_orig['name']:<38} best={best_orig['best_reward']:>8}  mean={best_orig['mean_reward']}")
    if best_ext:
        print(f"  Best extended : {best_ext['name']:<38} best={best_ext['best_reward']:>8}  mean={best_ext['mean_reward']}")
    print(f"  Best overall  : {best_overall['name']:<38} best={best_overall['best_reward']:>8}  mean={best_overall['mean_reward']}")
    print(f"{'='*70}")
    print(f"  CSV   -> {OUTPUT_CSV}")
    print(f"  Plots -> {PLOTS_DIR}/")
    print(f"{'='*70}")
else:
    print("\n  No results found. Check your runs/ folder.")