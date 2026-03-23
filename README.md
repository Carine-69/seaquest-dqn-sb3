# DQN Seaquest — Stable Baselines 3

Deep Q-Network agent trained on **ALE/Seaquest-v5** using [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://gymnasium.farama.org/).

## Agent Gameplay

<video src="Carine Umugabekazi/best_model_gameplay.mp4" controls width="480"></video>

Watch gameplay video

https://github.com/user-attachments/assets/07fb1fbf-4f4d-40e6-acae-a2de0df1ea00


> Trained DQN agent playing


 ALE/Seaquest-v5 using greedy Q-policy (deterministic=True).

## Environment

| Property | Value |
|---|---|
| Environment | `ALE/Seaquest-v5` |
| Action Space | `Discrete(18)` — NOOP, FIRE, UP, RIGHT, LEFT, DOWN, diagonals + fire combos |
| Observation Space | `Box(0, 255, (210, 160, 3), uint8)` — RGB pixel frames |
| Reward | Enemy subs (+20), rescued divers (+50–1000), oxygen bonus on surface |

## Project Structure

```
seaquest-dqn-sb3/
├── train.py                            # Training script (CnnPolicy, argparse CLI)
├── play.py                             # Evaluation script — greedy playback
├── dqn_model.zip                       # Best trained model (CnnPolicy)
├── requirements.txt                    # Python dependencies
├── README.md
│
├── Blessing Hirwa/                     # MlpPolicy + high-range experiments
│   ├── dqn_mlp_seaquest_experiments.ipynb  # Main notebook (10 experiments, 200k steps each)
│   ├── train.py
│   ├── play.py
│   └── dqn_model.zip
│
├── Carine Umugabekazi/                 # CnnPolicy + low-range experiments
│   ├── train.py
│   ├── play.py                         # Record gameplay video / evaluate model
│   ├── cnn_experiments.md
│   ├── best_model_gameplay.mp4
│   └── results/
│
└── Kerie Izere/                        # CnnPolicy + mid-range experiments
    ├── train.py
    ├── play.py
    └── kaggle_notebook.ipynb
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m AutoROM --accept-license

# Train a single experiment (1-10)
python train.py --exp 5

# Train all 10 experiments
python train.py --all

# Train with custom timesteps
python train.py --exp 10 --timesteps 500000

# Evaluate the best model (GUI window)
python play.py

# Evaluate with more episodes
python play.py --model dqn_model.zip --episodes 10

# Save gameplay as GIF
python play.py --save-gif --gif-path gameplay.gif
```

## Group Task Division

| Task | Carine | Kerie | Blessing |
|---|---|---|---|
| train.py (CnnPolicy) | Owner | Reviews | Reviews |
| train.py (MlpPolicy) | Reviews | Reviews | Owner |
| play.py | Reviews | Owner | Reviews |
| argparse / config | — | — | Owner |
| GitHub Repo & README | — | Records clip | Owner |
| Hyperparameter experiments | Low range (10) | Mid range (10) | High range (10) |

## Full Hyperparameter Results (30 Experiments)

### Carine Umugabekazi — Low Range (CnnPolicy)

**Baseline (1M steps):**

| # | lr | gamma | batch | eps_start | eps_end | eps_frac | Best Reward | Final Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00001 | 0.90 | 16 | 1.0 | 0.05 | 0.10 | 19.0 | 9.4 | Very slow learning, stable but minimal progress |
| 2 | 0.00002 | 0.91 | 16 | 1.0 | 0.05 | 0.10 | 24.0 | 8.2 | Slight lr bump, marginally faster convergence |
| 3 | 0.00003 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | 29.0 | 23.6 | Bigger batch gave more stable gradient updates |
| 4 | 0.00001 | 0.92 | 16 | 0.8 | 0.05 | 0.10 | 21.0 | 10.6 | Less early exploration hurt initial learning |
| 5 | 0.00005 | 0.91 | 32 | 1.0 | 0.01 | 0.10 | 30.0 | 26.6 | Low eps_end + high lr — best at 1M steps |
| 6 | 0.00001 | 0.93 | 16 | 1.0 | 0.05 | 0.05 | 20.0 | 8.8 | Fast decay shifted to exploitation too early |
| 7 | 0.00002 | 0.92 | 32 | 1.0 | 0.05 | 0.20 | 21.0 | 17.2 | Longer exploration phase, moderate result |
| 8 | 0.00003 | 0.93 | 16 | 0.9 | 0.01 | 0.10 | 22.0 | 19.8 | Combined effects — decent performance |
| 9 | 0.00004 | 0.90 | 32 | 1.0 | 0.05 | 0.15 | 33.0 | 16.0 | Highest single reward at 1M, balanced explore/exploit |
| 10 | 0.00005 | 0.93 | 32 | 0.8 | 0.01 | 0.10 | 22.0 | 20.6 | Best-of-low candidate |

**Extended training (10M steps, Exps 5–10):**

| # | lr | gamma | batch | eps_end | Best Reward | Final Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|
| **5** | **0.00005** | **0.91** | **32** | **0.01** | **239.0** | **53.8** | Agent learned to shoot enemies and rescue divers consistently |
| 6 | 0.00001 | 0.93 | 16 | 0.05 | 80.0 | 24.2 | Low lr limited learning even with 10M steps |
| 7 | 0.00002 | 0.92 | 32 | 0.05 | 201.0 | 50.8 | Batch=32 + slow decay gave strong long-term learning |
| 8 | 0.00003 | 0.93 | 16 | 0.01 | 186.0 | 54.8 | Small batch partially compensated by aggressive exploitation |
| 9 | 0.00004 | 0.90 | 32 | 0.05 | 192.0 | 57.0 | Moderate decay, reward still climbing at 10M |
| **10** | **0.00005** | **0.93** | **32** | **0.01** | **272.0** | **61.4** | **Best overall — high lr + high gamma + low eps_end** |

> Best model: **Exp 10 extended** — reward **272.0**, mean **53.24**. This is the `dqn_model.zip` at root.

### Kerie Izere — Mid Range (CnnPolicy)

| # | lr | gamma | batch | eps_start | eps_end | eps_frac | Steps | Best Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.94 | 64 | 1.0 | 0.05 | 0.10 | 200k | 18.8 | Baseline mid — stable, agent shoots enemies reliably |
| 2 | 0.0002 | 0.94 | 64 | 1.0 | 0.05 | 0.10 | 425k | 21.0 | Faster learning, early stopped |
| **3** | **0.0001** | **0.95** | **64** | **1.0** | **0.05** | **0.10** | **1M** | **44.4** | **Best — gamma=0.95 made agent value survival (oxygen surfacing)** |
| 4 | 0.0003 | 0.94 | 64 | 0.9 | 0.05 | 0.10 | 1M | 22.8 | Higher lr helped, less initial exploration was okay |
| 5 | 0.0001 | 0.96 | 64 | 1.0 | 0.02 | 0.10 | 1M | 20.2 | Low eps_end helped exploitation, gamma too high added noise |
| 6 | 0.0002 | 0.95 | 64 | 1.0 | 0.05 | 0.05 | 200k | 16.8 | Fast decay committed to exploitation too early |
| 7 | 0.0001 | 0.94 | 64 | 1.0 | 0.05 | 0.25 | 200k | 13.8 | Slow decay — too much random exploration wasted steps |
| 8 | 0.0004 | 0.95 | 64 | 0.9 | 0.02 | 0.10 | 1M | 21.0 | Mixed mid — moderate performance |
| 9 | 0.0003 | 0.96 | 64 | 1.0 | 0.05 | 0.15 | 400k | 20.0 | Early stopped, reward was still climbing |
| 10 | 0.0005 | 0.95 | 64 | 0.9 | 0.02 | 0.20 | 325k | 5.8 | lr=0.0005 too aggressive — Q-values oscillated and collapsed |

> Best model: **Exp 3** — reward **44.4** at 1M steps. Gamma=0.95 was the most impactful parameter.

### Blessing Hirwa — High Range (MlpPolicy, 200k timesteps)

| # | lr | gamma | batch | eps_start | eps_end | eps_frac | Noted Behavior |
|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 0.97 | 128 | 1.0 | 0.05 | 0.10 | Baseline high — fast learning but prone to overfitting |
| 2 | 0.002 | 0.97 | 128 | 1.0 | 0.05 | 0.10 | Larger lr caused instability, reward oscillated |
| 3 | 0.001 | 0.98 | 128 | 1.0 | 0.05 | 0.10 | Higher gamma improved long-horizon planning slightly |
| 4 | 0.003 | 0.97 | 256 | 1.0 | 0.05 | 0.10 | Big batch + high lr — gradient steps too aggressive, policy collapsed |
| 5 | 0.001 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | Max gamma — most stable among high-range, valued long-term rewards |
| 6 | 0.002 | 0.98 | 128 | 0.9 | 0.05 | 0.10 | Lower eps_start reduced early exploration, slightly faster convergence |
| 7 | 0.001 | 0.97 | 256 | 1.0 | 0.01 | 0.10 | Big batch + low eps_end — more exploitation late, moderate improvement |
| 8 | 0.003 | 0.98 | 128 | 1.0 | 0.05 | 0.05 | Fast eps decay + high lr — exploited too early before policy was good |
| 9 | 0.005 | 0.97 | 256 | 1.0 | 0.05 | 0.25 | Very high lr caused Q-value divergence, reward plateau |
| 10 | 0.005 | 0.99 | 256 | 0.9 | 0.01 | 0.20 | Best-of-high — slow decay + high gamma balanced exploration, best MLP result |

> Greedy evaluation of the best MLP model: **mean reward 2.33** over 3 episodes. MlpPolicy cannot extract spatial features from raw pixels, confirming that CNN is essential for pixel-based Atari environments.

## Best Model Comparison

| Name | Policy | Best Config | Steps | Best Reward |
|---|---|---|---|---|
| **Carine Umugabekazi** | **CnnPolicy** | **lr=0.00005, gamma=0.93, batch=32** | **10M** | **272.0** |
| Kerie Izere | CnnPolicy | lr=0.0001, gamma=0.95, batch=64 | 1M | 44.4 |
| Blessing Hirwa | MlpPolicy | lr=0.001, gamma=0.97, batch=128 | 200k | ~2.3 |

The final `dqn_model.zip` is Carine's Exp 10 extended (CnnPolicy, best reward 272.0).

## MLP vs CNN Policy Comparison

| Feature | CnnPolicy | MlpPolicy |
|---|---|---|
| Input handling | Conv layers extract spatial features | Flattens all pixels — no spatial bias |
| Best reward achieved | **272.0** (Carine Exp 10, 10M steps) | **~2.3** (Blessing, 200k steps) |
| Atari suitability | Standard for pixel-based environments | Designed for low-dimensional / RAM observations |

**Conclusion:** CnnPolicy vastly outperforms MlpPolicy on pixel-based Atari. Even comparing at the same step count (200k), CNN achieved 18.8+ while MLP scored ~2.3 — a 8x gap. MLP flattens the 84x84x4 image into 28,224 raw inputs and loses all spatial structure (edges, enemy positions, oxygen bar), while CNN extracts these features via learned convolutional filters. This confirms why convolutional architectures are the standard for image-based reinforcement learning, as established by Mnih et al. (2015).

## Technologies

- Python 3.10+
- Stable Baselines 3
- Gymnasium (ALE/Seaquest-v5)
- PyTorch
- TensorBoard
