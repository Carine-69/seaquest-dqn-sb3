# DQN Seaquest — Stable Baselines 3

Deep Q-Network agent trained on **ALE/Seaquest-v5** using [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://gymnasium.farama.org/).

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
├── README.md
├── .gitignore
├── DQN Seaquest Group Plan.pdf
│
├── Blessing Hirwa/                     # Member 3 — MlpPolicy + high-range experiments
│   ├── member3_dqn_seaquest.ipynb      # Main notebook (10 experiments, 200k steps each)
│   ├── train.py                        # MlpPolicy training script (argparse CLI)
│   ├── play.py                         # Evaluation script — greedy playback
│   ├── dqn_model.zip                   # Best model from high-range experiments
│   └── requirements.txt
│
├── Carine Umugabekazi/                 # Member 1 — CnnPolicy + low-range experiments
│   ├── train.py
│   ├── train_extended.py
│   ├── cnn_experiments.md
│   ├── best_model_gameplay.mp4
│   └── results/
│
└── Kerie Izere/                        # Member 2 — play.py + mid-range experiments
    ├── train.py
    ├── play.py
    ├── README.md
    ├── gameplay.gif
    └── kaggle_notebook.ipynb
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m AutoROM --accept-license

# Train with default best config (Exp 10)
python train.py

# Train a specific preset experiment
python train.py --exp_id 5

# Train with custom hyperparameters
python train.py --lr 0.001 --gamma 0.98 --batch_size 128 --timesteps 200000

# Evaluate the best model (GUI)
python play.py

# Evaluate headless
python play.py --model dqn_model.zip --episodes 10 --no_render
```

## Group Task Division

| Task | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| train.py (CnnPolicy) | Owner | Reviews | Reviews |
| train.py (MlpPolicy) | Reviews | Reviews | Owner |
| play.py | Reviews | Owner | Reviews |
| argparse / config | — | — | Owner |
| GitHub Repo & README | — | Records clip | Owner |
| Hyperparameter experiments | Low range (10) | Mid range (10) | High range (10) |

## Full Hyperparameter Results (30 Experiments)

### Carine Umugabekazi — Low Range (CnnPolicy, 1M timesteps)

| # | lr | gamma | batch | eps_start | eps_end | eps_frac | Final Reward | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.00001 | 0.90 | 16 | 1.0 | 0.05 | 0.10 | 9.4 | Baseline low |
| 2 | 0.00002 | 0.91 | 16 | 1.0 | 0.05 | 0.10 | 8.2 | Slight lr bump |
| 3 | 0.00003 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | 23.6 | Bigger batch |
| 4 | 0.00001 | 0.92 | 16 | 0.8 | 0.05 | 0.10 | 10.6 | Lower eps_start |
| 5 | 0.00005 | 0.91 | 32 | 1.0 | 0.01 | 0.10 | 26.6 | Lower eps_end |
| 6 | 0.00001 | 0.93 | 16 | 1.0 | 0.05 | 0.05 | 8.8 | Faster decay |
| 7 | 0.00002 | 0.92 | 32 | 1.0 | 0.05 | 0.20 | 17.2 | Slower decay |
| 8 | 0.00003 | 0.93 | 16 | 0.9 | 0.01 | 0.10 | 19.8 | Mixed low |
| 9 | 0.00004 | 0.90 | 32 | 1.0 | 0.05 | 0.15 | 16.0 | Moderate decay |
| 10 | 0.00005 | 0.93 | 32 | 0.8 | 0.01 | 0.10 | 20.6 | Best-of-low |

### Kerie Izere — Mid Range (CnnPolicy, 200k timesteps)

| # | lr | gamma | batch | eps_start | eps_end | eps_frac | Best Reward | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.94 | 64 | 1.0 | 0.05 | 0.10 | 18.8 | Baseline mid |
| 2 | 0.0002 | 0.94 | 64 | 1.0 | 0.05 | 0.10 | 15.4 | Slight lr bump |
| 3 | 0.0001 | 0.95 | 64 | 1.0 | 0.05 | 0.10 | 18.6 | Higher gamma |
| 4 | 0.0003 | 0.94 | 64 | 0.9 | 0.05 | 0.10 | 8.0 | Lower eps_start |
| 5 | 0.0001 | 0.96 | 64 | 1.0 | 0.02 | 0.10 | 13.0 | Lower eps_end |
| 6 | 0.0002 | 0.95 | 64 | 1.0 | 0.05 | 0.05 | 16.8 | Faster decay |
| 7 | 0.0001 | 0.94 | 64 | 1.0 | 0.05 | 0.25 | 13.8 | Slower decay |
| 8 | 0.0004 | 0.95 | 64 | 0.9 | 0.02 | 0.10 | 9.2 | Mixed mid |
| 9 | 0.0003 | 0.96 | 64 | 1.0 | 0.05 | 0.15 | — | Incomplete |
| 10 | 0.0005 | 0.95 | 64 | 0.9 | 0.02 | 0.20 | — | Incomplete |

### Blessing Hirwa — High Range (MlpPolicy, 200k timesteps)

| # | lr | gamma | batch | eps_start | eps_end | eps_frac | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 0.97 | 128 | 1.0 | 0.05 | 0.10 | Baseline high |
| 2 | 0.002 | 0.97 | 128 | 1.0 | 0.05 | 0.10 | Larger lr |
| 3 | 0.001 | 0.98 | 128 | 1.0 | 0.05 | 0.10 | Higher gamma |
| 4 | 0.003 | 0.97 | 256 | 1.0 | 0.05 | 0.10 | Big batch + high lr |
| 5 | 0.001 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | Max gamma |
| 6 | 0.002 | 0.98 | 128 | 0.9 | 0.05 | 0.10 | Lower eps_start + high lr |
| 7 | 0.001 | 0.97 | 256 | 1.0 | 0.01 | 0.10 | Big batch + low eps_end |
| 8 | 0.003 | 0.98 | 128 | 1.0 | 0.05 | 0.05 | Faster decay + high lr |
| 9 | 0.005 | 0.97 | 256 | 1.0 | 0.05 | 0.25 | Slow decay + very high lr |
| 10 | 0.005 | 0.99 | 256 | 0.9 | 0.01 | 0.20 | Best-of-high candidate |

## MLP vs CNN Policy Comparison

| Feature | CnnPolicy | MlpPolicy |
|---|---|---|
| Input handling | Conv layers extract spatial features | Flattens pixels — no spatial bias |
| Parameter efficiency | Moderate (shared conv base) | Very high (84x84x4 = 28,224 inputs) |
| Atari performance | **Significantly better** | Lower bound / ablation baseline |
| Best suited for | Raw pixel observations | RAM obs or low-dim state |

**Conclusion:** CnnPolicy substantially outperforms MlpPolicy on pixel-based Atari. MLP flattens the image and loses all spatial structure, while CNN extracts edges, shapes, and motion via convolution. The high-range MLP experiments confirm this gap — even with aggressive hyperparameters, MLP cannot compensate for the lack of spatial feature extraction.

## Technologies

- Python 3.10+
- Stable Baselines 3
- Gymnasium (ALE/Seaquest-v5)
- PyTorch
- TensorBoard
