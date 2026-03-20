# Member 2 — Mid-Range Hyperparameter Experiments

**Environment:** ALE/Seaquest-v5 | **Policy:** CnnPolicy

## How to Run

```bash
# Single experiment
python kerie/train.py --exp 4

# All 10
python kerie/train.py --all --timesteps 50000

# Play
python kerie/play.py --model kerie/runs_kaggle/exp1_baseline_mid/dqn_model.zip
```

## Experiments

| #   | lr     | gamma | eps_start | eps_end | eps_frac | Notes                       |
| --- | ------ | ----- | --------- | ------- | -------- | --------------------------- |
| 1   | 0.0001 | 0.94  | 1.0       | 0.05    | 0.10     | Baseline mid                |
| 2   | 0.0002 | 0.94  | 1.0       | 0.05    | 0.10     | Slight lr bump              |
| 3   | 0.0001 | 0.95  | 1.0       | 0.05    | 0.10     | Higher gamma                |
| 4   | 0.0003 | 0.94  | 0.9       | 0.05    | 0.10     | Lower eps_start             |
| 5   | 0.0001 | 0.96  | 1.0       | 0.02    | 0.10     | Lower eps_end               |
| 6   | 0.0002 | 0.95  | 1.0       | 0.05    | 0.05     | Faster decay                |
| 7   | 0.0001 | 0.94  | 1.0       | 0.05    | 0.25     | Slower decay                |
| 8   | 0.0004 | 0.95  | 0.9       | 0.02    | 0.10     | Mixed mid                   |
| 9   | 0.0003 | 0.96  | 1.0       | 0.05    | 0.15     | Moderate decay + high gamma |
| 10  | 0.0005 | 0.95  | 0.9       | 0.02    | 0.20     | Best-of-mid candidate       |

## Results

### Session 1 — Local CPU (50k steps)

> Agent mostly dies from oxygen running out, random movement, barely learned anything.

| #     | Best Reward |
| ----- | ----------- |
| 1     | 7.40        |
| 2     | 5.20        |
| 3     | 4.20        |
| **4** | **9.60**    |
| 5     | 4.20        |
| 6     | 2.40        |
| 7     | 3.80        |
| 8     | 4.20        |
| 9     | 6.40        |
| 10    | 5.00        |

**Best:** Exp 4 — `eps_start=0.9` reduced early randomness, agent exploited sooner.

### Session 2 — Kaggle GPU (200k steps)

> Agent starts hitting enemies consistently. More stable behavior, rewards doubled vs session 1.

| #     | Best Reward | Final Reward |
| ----- | ----------- | ------------ |
| **1** | **18.8**    | 15.6         |
| 2     | 15.4        | 15.4         |
| 3     | 18.6        | 18.6         |
| 4     | 8.0         | 8.0          |
| 5     | 13.0        | 13.0         |
| 6     | 16.8        | 16.8         |
| 7     | 13.8        | 11.8         |
| 8     | 9.2         | 9.2          |
| 9     | incomplete  | —            |
| 10    | incomplete  | —            |

**Best:** Exp 1 (baseline) — reward 18.8, std 1.0 (very stable). Agent learned to shoot enemies reliably.

**Improvement from session 1 → 2:** Mean reward ~5 → ~19 (4x better with 4x more steps).

**Still needs:** 1M+ steps to learn oxygen surfacing and diver rescues for higher rewards.
