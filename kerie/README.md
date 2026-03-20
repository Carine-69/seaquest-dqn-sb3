# Member 2 — Mid-Range Hyperparameter Experiments

**Environment:** ALE/Seaquest-v5 | **Policy:** CnnPolicy | **Steps:** 50,000

## How to Run

```bash
# Single experiment
python kerie/train.py --exp 4

# All 10
python kerie/train.py --all --timesteps 50000

# Play
python kerie/play.py --model kerie/runs/exp4_lower_eps_start/dqn_model.zip
```

## Experiments

| #     | lr         | gamma    | eps_start | eps_end  | eps_frac | Best Reward |
| ----- | ---------- | -------- | --------- | -------- | -------- | ----------- |
| 1     | 0.0001     | 0.94     | 1.0       | 0.05     | 0.10     | 7.40        |
| 2     | 0.0002     | 0.94     | 1.0       | 0.05     | 0.10     | 5.20        |
| 3     | 0.0001     | 0.95     | 1.0       | 0.05     | 0.10     | 4.20        |
| **4** | **0.0003** | **0.94** | **0.9**   | **0.05** | **0.10** | **9.60**    |
| 5     | 0.0001     | 0.96     | 1.0       | 0.02     | 0.10     | 4.20        |
| 6     | 0.0002     | 0.95     | 1.0       | 0.05     | 0.05     | 2.40        |
| 7     | 0.0001     | 0.94     | 1.0       | 0.05     | 0.25     | 3.80        |
| 8     | 0.0004     | 0.95     | 0.9       | 0.02     | 0.10     | 4.20        |
| 9     | 0.0003     | 0.96     | 1.0       | 0.05     | 0.15     | 6.40        |
| 10    | 0.0005     | 0.95     | 0.9       | 0.02     | 0.20     | 5.00        |

**Best:** Exp 4 — lower `eps_start=0.9` reduced early randomness, letting the agent exploit sooner.
